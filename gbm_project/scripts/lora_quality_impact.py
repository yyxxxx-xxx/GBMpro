#!/usr/bin/env python3
"""
LoRA 质量影响评估（消融的关键一环）
==================================

问题: GBM LoRA 是在 Llamole 适配器之上继续训练的。训练集只有 32 个唯一分子
      （全是简单苯甲酰胺/苯胺类，MW 150-250）。这种数据继续微调，
      极可能让模型**遗忘** Llamole 原本的药物生成能力，退化成只会输出那几个模板。

对照实验:
    A. base + Llamole              (Llamole 原生能力)
    B. base + Llamole + GBM LoRA   (完整模型)

每个变体采样 N 个分子，用统一的本地管线算:
    - 有效 SMILES 率
    - 药物相似性: MW / TPSA / LogP / QED / SA score
    - 骨架多样性: 唯一 Murcko 骨架数、归一化骨架熵
    - 训练集重叠率（直接出现在训练集里的比例 = 背答案的程度）

输出: JSON + 控制台对比表
"""

import os
import re
import sys
import json
import math
import argparse
import warnings
from collections import Counter
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
from property_pipeline.descriptors import compute_descriptors
from property_pipeline.scaffold import compute_scaffold_metrics
from property_pipeline.model_loader import load_merged_llamole_gbm

BASE = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
LLAMOLE = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")
GBM_FIXED = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed")
TRAIN_JSON = GBM_ROOT / "data/lora_datasets/gbm_evaluation_expanded_lora_train.json"

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


def build_instruction(target: str, variant: int) -> str:
    cell = CELL_STATES[variant % len(CELL_STATES)]
    bbb, act, tox = METRIC_VARIANTS[variant % len(METRIC_VARIANTS)]
    hint = "Target: EGFR kinase domain inhibitor, quinazoline core with acrylamide warhead."
    return "\n".join([
        f"Generate a GBM drug candidate SMILES for {cell}.",
        hint,
        f"Target metrics: BBB: {bbb}; GBM Activity: {act}; "
        f"Toxicity: {tox}; Selectivity: high (GBM/normal ratio)",
        "Output format: SMILES: <canonical_SMILES>",
    ])


def load_train_smiles() -> set:
    data = json.load(open(TRAIN_JSON))
    out = set()
    for r in data:
        s = r["response"].strip()
        if s.upper().startswith("SMILES:"):
            s = s[7:].strip()
        fixed, _ = repair_smiles(s)
        if fixed:
            out.add(fixed)
    return out


def sample_molecules(model, tok, n, temperature=0.9, top_p=0.95, seed_base=5000):
    smis, raw_ok, repaired, failed = [], 0, 0, 0
    for i in range(n):
        instr = build_instruction("EGFR", i % len(CELL_STATES))
        torch.manual_seed(seed_base + i)
        full = f"Instruction: {instr}\nInput:\nResponse:"
        inputs = tok(full, return_tensors="pt").to(model.device)
        ilen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128, temperature=temperature, top_p=top_p,
                do_sample=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        resp = tok.decode(out[0][ilen:], skip_special_tokens=True)
        m = re.search(r"SMILES\s*:\s*([^\n]*)", resp)
        cand = (m.group(1) if m else resp).strip().split(">>")[0].strip()
        fixed, changed = repair_smiles(cand)
        if fixed is None:
            failed += 1
            continue
        if changed:
            repaired += 1
        else:
            raw_ok += 1
        if fixed not in smis:
            smis.append(fixed)
    return smis, {"raw_ok": raw_ok, "repaired": repaired, "failed": failed}


def quality_metrics(smiles_list: List[str], train_smiles: set) -> Dict:
    if not smiles_list:
        return {}
    descs = [compute_descriptors(s) for s in smiles_list]

    def mean(key):
        vals = [d.get(key) for d in descs if d.get(key) is not None and d[key] == d[key]]
        return round(sum(vals) / len(vals), 2) if vals else None

    scaf = compute_scaffold_metrics(smiles_list)
    scaffolds = [r.get("Scaffold") for r in scaf if r.get("Scaffold")]
    n_unique_scaf = len(set(scaffolds))

    # 归一化骨架熵
    cnt = Counter(scaffolds)
    n = sum(cnt.values())
    entropy = None
    if n > 1 and len(cnt) > 1:
        h = -sum((c / n) * math.log(c / n) for c in cnt.values())
        entropy = round(h / math.log(n), 3)

    overlap = len(set(smiles_list) & train_smiles)

    return {
        "n_unique": len(smiles_list),
        "MW": mean("MW"),
        "TPSA": mean("TPSA"),
        "LogP": mean("LogP"),
        "QED_mean": mean("QED") if "QED" in descs[0] else None,
        "SA": mean("Synthetic Accessibility"),
        "PAINS_mean": mean("PAINS #alerts"),
        "n_unique_scaffolds": n_unique_scaf,
        "scaffold_entropy": entropy,
        "overlap_with_train": overlap,
        "overlap_pct": round(overlap / len(smiles_list) * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output", type=str,
                        default=str(GBM_ROOT / "experiments" / "lora_quality_impact.json"))
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train_smiles = load_train_smiles()
    print(f"训练集唯一分子: {len(train_smiles)}")

    results = {}

    # ---- A. base + Llamole ----
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
    m = PeftModel.from_pretrained(m, LLAMOLE).merge_and_unload()
    m = m.to("cuda:0")
    smis_a, stat_a = sample_molecules(m, tok, args.n)
    results["A_base_llamole"] = {"stats": stat_a, **quality_metrics(smis_a, train_smiles)}
    print(f"\nA. base+Llamole: {results['A_base_llamole']}")
    del m
    torch.cuda.empty_cache()

    # ---- B. base + Llamole + GBM LoRA ----
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
    m = PeftModel.from_pretrained(m, LLAMOLE).merge_and_unload()
    m = PeftModel.from_pretrained(m, GBM_FIXED)
    m = m.to("cuda:0")
    smis_b, stat_b = sample_molecules(m, tok, args.n)
    results["B_full_model"] = {"stats": stat_b, **quality_metrics(smis_b, train_smiles)}
    print(f"\nB. 完整模型: {results['B_full_model']}")
    del m
    torch.cuda.empty_cache()

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 对比表 ----
    print("\n" + "=" * 84)
    hdr = f"{'指标':<26}{'A. base+Llamole':>26}{'B. +GBM LoRA (完整)':>26}"
    print(hdr)
    print("-" * 84)
    a, b = results["A_base_llamole"], results["B_full_model"]
    for key, label in [
        ("n_unique", "有效唯一分子"),
        ("MW", "MW (平均)"),
        ("TPSA", "TPSA (平均)"),
        ("LogP", "LogP (平均)"),
        ("SA", "SA score (平均)"),
        ("n_unique_scaffolds", "唯一 Murcko 骨架"),
        ("scaffold_entropy", "归一化骨架熵"),
        ("overlap_with_train", "与训练集重叠数"),
        ("overlap_pct", "与训练集重叠 %"),
    ]:
        va, vb = a.get(key), b.get(key)
        print(f"{label:<26}{str(va):>26}{str(vb):>26}")
    print("=" * 84)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
