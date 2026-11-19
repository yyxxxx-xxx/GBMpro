#!/usr/bin/env python3
"""
Prompt 格式消融（快速诊断）
===========================

目的: 找出让 GBM LoRA 输出有效 SMILES 的推理配置。

候选维度:
    1. prompt 是否严格复刻训练数据（不加额外 "Target:" 行）
    2. temperature (0.7 / 0.9 / 1.0 / 1.2)
    3. 是否用 chat template

输出: 每种配置的原始响应 + 有效 SMILES 数量。
"""

import sys
import json
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
from property_pipeline.model_loader import load_merged_llamole_gbm

# 严格复刻训练数据的 instruction
TRAIN_INSTRUCTION = (
    "Generate a GBM drug candidate SMILES for Mesenchymal-like GBM cells.\n"
    "Target metrics: BBB: good permeability (logBB > -0.5); GBM Activity: good; "
    "Toxicity: moderate acceptable; Selectivity: high (GBM/normal ratio)\n"
    "Output format: SMILES: <canonical_SMILES>"
)

# 训练数据里的真实 response 前缀
TRAIN_PREFIX = "SMILES: "


def gen(model, tokenizer, prompt, suffix="", max_new_tokens=48, temperature=1.0, top_p=0.95):
    full = f"Instruction: {prompt}\nInput:\nResponse:{suffix}"
    inputs = tokenizer(full, return_tensors="pt").to(model.device)
    ilen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.01,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][ilen:], skip_special_tokens=True)


def count_valid(resp):
    import re
    m = re.search(r"SMILES\s*:\s*([^\n]+)", resp)
    cands = []
    if m:
        cands.append(m.group(1).strip())
    else:
        cands = re.findall(r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.]{8,})\b", resp)
    return sum(1 for c in cands if validate_smiles(c)), cands


def main():
    model, tokenizer = load_merged_llamole_gbm(
        base_model_path=str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        llamole_adapter=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        gbm_adapter=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed"),
        device="cuda:0",
    )

    configs = [
        ("strict-train, suffix='SMILES: ', T=0.7", TRAIN_INSTRUCTION, "SMILES: ", 0.7),
        ("strict-train, suffix='SMILES: ', T=1.0", TRAIN_INSTRUCTION, "SMILES: ", 1.0),
        ("strict-train, suffix='SMILES: ', T=1.2", TRAIN_INSTRUCTION, "SMILES: ", 1.2),
        ("strict-train, no suffix, T=0.7", TRAIN_INSTRUCTION, "", 0.7),
        ("strict-train, no suffix, T=1.0", TRAIN_INSTRUCTION, "", 1.0),
        ("strict-train, no suffix, T=1.2", TRAIN_INSTRUCTION, "", 1.2),
    ]

    results = {}
    for label, prompt, suffix, temp in configs:
        total_valid, total, samples = 0, 0, []
        for _ in range(5):
            resp = gen(model, tokenizer, prompt, suffix, temperature=temp)
            n_valid, cands = count_valid(resp)
            total_valid += n_valid > 0
            total += 1
            if len(samples) < 2:
                samples.append((resp.strip()[:120], cands[:1]))
        results[label] = {"valid_rate": f"{total_valid}/{total}", "samples": samples}
        print(f"\n{'=' * 70}")
        print(f"{label}   -> 有效率 {total_valid}/{total}")
        for r, c in samples:
            print(f"   raw: {r}")
            print(f"   cand: {c}")

    print("\n" + "=" * 70)
    print("汇总:")
    for k, v in results.items():
        print(f"  {k}: {v['valid_rate']}")

    out = GBM_ROOT / "experiments" / "prompt_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
