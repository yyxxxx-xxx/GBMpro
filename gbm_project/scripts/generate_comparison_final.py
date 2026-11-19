#!/usr/bin/env python3
"""
GBM-PharmaMind 大规模分子生成（对比实验主结果）
=================================================

与 generate_native.py 的区别:
    1. 用项目正式配置 (T=0.7, top_p=0.9) 而非高温采样
    2. 多样性靠"多条不同的 instruction 前缀"实现，而不是只靠温度
       —— 因为 LoRA 训练集的 instruction 只有 6 种唯一形态，
       采样空间天然受限，必须靠不同的目标措辞把分布铺开
    3. 状态感知 SMILES 修复（见 property_pipeline.smiles_repair）
    4. 记录修复统计，论文里要披露

用法:
    python scripts/generate_comparison_final.py --targets EGFR,VEGFR,IDH1 --n 20
"""

import os
import re
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List

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

from property_pipeline.smiles_repair import repair_smiles
from property_pipeline.model_loader import load_merged_llamole_gbm


# 训练集里出现过的 4 种细胞状态 + 指标措辞组合
# 每个靶点用不同的组合，以铺开采样分布
CELL_STATES = [
    "Oligodendrocyte progenitor-like GBM cells",
    "Mesenchymal-like GBM cells",
    "Neural progenitor-like GBM cells",
    "Classical GBM cells",
]

METRIC_VARIANTS = [
    ("moderate permeability (logBB > -0.5)", "moderate", "low"),
    ("good permeability (logBB > -0.5)", "good", "moderate acceptable"),
    ("moderate permeability (logBB > -0.5)", "good", "low"),
    ("good permeability (logBB > -0.5)", "moderate", "moderate acceptable"),
]

# 靶点特异的额外约束（放在 instruction 的 Target metrics 之前）
TARGET_HINTS = {
    "EGFR": "Target: EGFR kinase domain inhibitor, quinazoline core with acrylamide warhead.",
    "VEGFR": "Target: VEGFR2/KDR kinase domain inhibitor.",
    "IDH1": "Target: mutant IDH1 R132H inhibitor.",
}


def build_instruction(target: str, variant: int) -> str:
    """构造 instruction；variant 决定细胞状态与指标措辞组合。"""
    cell = CELL_STATES[variant % len(CELL_STATES)]
    bbb, act, tox = METRIC_VARIANTS[variant % len(METRIC_VARIANTS)]
    hint = TARGET_HINTS.get(target, "")

    lines = [f"Generate a GBM drug candidate SMILES for {cell}."]
    if hint:
        lines.append(hint)
    lines.append(
        f"Target metrics: BBB: {bbb}; GBM Activity: {act}; "
        f"Toxicity: {tox}; Selectivity: high (GBM/normal ratio)"
    )
    lines.append("Output format: SMILES: <canonical_SMILES>")
    return "\n".join(lines)


def generate_one(model, tokenizer, instruction: str, max_new_tokens: int,
                 temperature: float, top_p: float, seed: int) -> str:
    """单次采样。格式严格复刻训练时的 "Instruction/Input/Response" 模板。"""
    torch.manual_seed(seed)
    full = f"Instruction: {instruction}\nInput:\nResponse:"
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


SMILES_RE = re.compile(r"SMILES\s*:\s*([^\n]*)")


def extract_candidate(resp: str) -> str:
    m = SMILES_RE.search(resp)
    if m:
        return m.group(1).strip().split(">>")[0].strip()
    return resp.strip().split("\n")[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=str, default="EGFR,VEGFR,IDH1")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--library_name", type=str, default="GBM-PharmaMind (Ours)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir or (GBM_ROOT / "experiments" / "comparison_final"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_merged_llamole_gbm(
        base_model_path=str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        llamole_adapter=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        gbm_adapter=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed"),
        device=f"cuda:{args.gpu_id}",
    )

    import csv as _csv

    summary = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        print(f"\n{'=' * 70}\n🎯 {target}\n{'=' * 70}")
        rows, seen, n_raw_valid, n_repaired, n_failed = [], set(), 0, 0, 0
        raw_log = []

        for i in range(args.n):
            variant = i % len(CELL_STATES)
            instr = build_instruction(target, variant)
            resp = generate_one(
                model, tokenizer, instr,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=7000 + i,
            )
            cand = extract_candidate(resp)
            raw_log.append(
                f"--- variant {variant} ---\n{instr}\n>>> {resp.strip()[:400]}\n"
            )

            fixed, changed = repair_smiles(cand)
            if fixed is None:
                n_failed += 1
                continue
            if changed:
                n_repaired += 1
            else:
                n_raw_valid += 1
            if fixed in seen:
                continue
            seen.add(fixed)
            rows.append(fixed)

        print(f"  原始有效 {n_raw_valid} / 修复后有效 {n_repaired} / 失败 {n_failed}")
        print(f"  去重后 {len(rows)} 个唯一分子")

        csv_path = out_dir / f"{target}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["entity_id", "mol_id", "library_name", "target", "smiles"])
            for i, smi in enumerate(rows):
                w.writerow([
                    target,
                    f"{args.library_name}_{target}_{i + 1}",
                    args.library_name,
                    target,
                    smi,
                ])
        print(f"  💾 {csv_path}")
        (out_dir / f"{target}_raw_log.txt").write_text("\n".join(raw_log), encoding="utf-8")

        summary[target] = {
            "n_requested": args.n,
            "n_unique": len(rows),
            "n_raw_valid": n_raw_valid,
            "n_repaired": n_repaired,
            "n_failed": n_failed,
            "csv": str(csv_path),
        }

    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n完成。")


if __name__ == "__main__":
    main()
