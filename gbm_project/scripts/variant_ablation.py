#!/usr/bin/env python3
"""
模型变体对照（定位噪声来源）
=============================

测试三种模型变体在**同一个训练分布 prompt** 下的输出:
    A. base only                          (纯基座)
    B. base + Llamole merged              (不含 GBM adapter)
    C. base + Llamole merged + GBM LoRA   (完整模型)

目的: 判断 SMILES 尾部的噪声到底来自
    (a) GBM LoRA 权重本身
    (b) prompt 格式与训练分布的偏离
    (c) 采样参数

用法:
    python scripts/variant_ablation.py
"""

import sys
import json
import re
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(GBM_ROOT))
sys.path.append(str(GBM_ROOT / "src"))

from gbm_project.improved_smiles_extraction import validate_smiles
from property_pipeline.smiles_repair import repair_smiles

BASE = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
LLAMOLE = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")
GBM_FIXED = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed")

# 与训练数据完全一致的 instruction（不含任何额外 Target 行）
INSTR = (
    "Generate a GBM drug candidate SMILES for Mesenchymal-like GBM cells.\n"
    "Target metrics: BBB: good permeability (logBB > -0.5); GBM Activity: good; "
    "Toxicity: moderate acceptable; Selectivity: high (GBM/normal ratio)\n"
    "Output format: SMILES: <canonical_SMILES>"
)


def build_base():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
    return m, tok


def gen(model, tok, prompt, max_new_tokens=48, temperature=0.9, top_p=0.95, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    full = f"Instruction: {prompt}\nInput: \nResponse:"
    inputs = tok(full, return_tensors="pt").to(model.device)
    ilen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.01,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][ilen:], skip_special_tokens=True)


def evaluate(model, tok, label, n=8, temperature=0.9):
    raw_ok, repaired_ok, samples = 0, 0, []
    for i in range(n):
        resp = gen(model, tok, INSTR, temperature=temperature, seed=1000 + i)
        m = re.search(r"SMILES\s*:\s*([^\n]+)", resp)
        cand = m.group(1).strip() if m else resp.strip()
        if validate_smiles(cand):
            raw_ok += 1
        fixed, _ = repair_smiles(cand)
        if fixed:
            repaired_ok += 1
        if len(samples) < 3:
            samples.append((resp.strip()[:130], fixed))
    print(f"\n{'=' * 70}")
    print(f"{label}  T={temperature}   原始有效 {raw_ok}/{n}   修复后有效 {repaired_ok}/{n}")
    for r, f in samples:
        print(f"   raw : {r}")
        print(f"   fix : {f}")
    return {"raw_ok": raw_ok, "repaired_ok": repaired_ok, "samples": samples}


def main():
    results = {}

    # ---- A. base only ----
    from peft import PeftModel

    m, tok = build_base()
    m = m.to("cuda:0")
    results["A_base_only"] = evaluate(m, tok, "A. base only")
    del m
    torch.cuda.empty_cache()

    # ---- B. base + Llamole merged ----
    m, tok = build_base()
    m = PeftModel.from_pretrained(m, LLAMOLE).merge_and_unload()
    m = m.to("cuda:0")
    results["B_base_llamole"] = evaluate(m, tok, "B. base + Llamole")
    del m
    torch.cuda.empty_cache()

    # ---- C. base + Llamole + GBM LoRA ----
    m, tok = build_base()
    m = PeftModel.from_pretrained(m, LLAMOLE).merge_and_unload()
    m = PeftModel.from_pretrained(m, GBM_FIXED)
    m = m.to("cuda:0")
    results["C_full"] = evaluate(m, tok, "C. base + Llamole + GBM LoRA")
    del m
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("汇总（原始有效 / 修复后有效，共 8 次采样）:")
    for k, v in results.items():
        print(f"  {k}: {v['raw_ok']}/8  ->  {v['repaired_ok']}/8")

    out = GBM_ROOT / "experiments" / "variant_ablation.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
