#!/usr/bin/env python3
"""
采样参数消融（定稿前最后一轮）
==============================

项目自身配置 (configs/gbm_generation.yaml):
    temperature: 0.7
    top_p: 0.9
    max_new_tokens: 256

之前测试用的是 0.9-1.2，温度过高是尾部噪声的主因。
这里按项目配置 + 邻域做对照。

同时对比两种截断策略:
    A. 第一个 "SMILES:" 之后取到行尾，再做状态感知截断修复
    B. 同上但不修复（原始）
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
from property_pipeline.model_loader import load_merged_llamole_gbm

INSTR = (
    "Generate a GBM drug candidate SMILES for Mesenchymal-like GBM cells.\n"
    "Target metrics: BBB: good permeability (logBB > -0.5); GBM Activity: good; "
    "Toxicity: moderate acceptable; Selectivity: high (GBM/normal ratio)\n"
    "Output format: SMILES: <canonical_SMILES>"
)


def gen(model, tok, prompt, max_new_tokens, temperature, top_p, seed):
    torch.manual_seed(seed)
    full = f"Instruction: {prompt}\nInput:\nResponse:"
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


def first_smiles(resp):
    m = re.search(r"SMILES\s*:\s*([^\n]*)", resp)
    return m.group(1).strip() if m else ""


def main():
    model, tok = load_merged_llamole_gbm(
        base_model_path=str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        llamole_adapter=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        gbm_adapter=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed"),
        device="cuda:0",
    )

    configs = [
        ("项目默认 T=0.7 p=0.9 mnt=256", 256, 0.7, 0.9),
        ("T=0.7 p=0.9 mnt=64", 64, 0.7, 0.9),
        ("T=0.6 p=0.9 mnt=256", 256, 0.6, 0.9),
        ("T=0.8 p=0.92 mnt=256", 256, 0.8, 0.92),
        ("T=1.0 p=0.95 mnt=256", 256, 1.0, 0.95),
    ]

    n = 6
    summary = {}
    for label, mnt, temp, top_p in configs:
        raw_ok, fix_ok, mols = 0, 0, []
        for i in range(n):
            resp = gen(model, tok, INSTR, mnt, temp, top_p, 2000 + i)
            cand = first_smiles(resp)
            if validate_smiles(cand):
                raw_ok += 1
            fixed, _ = repair_smiles(cand)
            if fixed:
                fix_ok += 1
                mols.append(fixed)
        summary[label] = {"raw_ok": raw_ok, "fix_ok": fix_ok, "mols": mols}
        print(f"\n{'=' * 70}")
        print(f"{label}:  原始 {raw_ok}/{n}   修复后 {fix_ok}/{n}")
        for m in mols[:4]:
            print(f"    {m}")

    print("\n" + "=" * 70)
    print("汇总:")
    for k, v in summary.items():
        print(f"  {k}: raw {v['raw_ok']}/{n}  fix {v['fix_ok']}/{n}")

    out = GBM_ROOT / "experiments" / "sampling_ablation.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
