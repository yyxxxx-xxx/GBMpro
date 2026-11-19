#!/usr/bin/env python3
"""
用新训练的 LoRA 生成 3 靶点分子（走与对比报告一致的管线）
=========================================================

与 generate_comparison_final.py 的差别：
  1. adapter 路径可配置（默认指向新训练的 -gbm-real）
  2. instruction 使用**推理时同一套 prompt 模板**（base_gbm_prompt +
     target_specific_prompts + CoT + 生成指令），与新训练集形态一致
  3. 同时记录 raw 输出与修复统计

用法
----
    python scripts/generate_with_new_adapter.py \
        --adapter /root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-real \
        --out experiments/ours_real
"""
import os
import re
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from rdkit import Chem, RDLogger
from transformers import StoppingCriteria, StoppingCriteriaList

RDLogger.DisableLog("rdApp")

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

# 推理时用的目标专有描述（来自 configs/english_gbm_prompts.yaml）
TARGET_KEYS = {
    "EGFR": "EGFR",
    "VEGFR": "VEGF_VEGFR",
    "IDH1": "IDH1_IDH2",
}


def build_prompt(prompts: dict, target: str, cell_desc: str, metrics: str,
                 use_cot: bool) -> str:
    """复用推理 prompt 模板"""
    base = prompts["base_gbm_prompt"].strip()
    req = (
        f"- Target cell phenotype: {cell_desc}\n"
        f"- Target metrics: {metrics}\n"
        f"- Molecular features: MW 300-500 Da, logP 2-4, TPSA 60-120, "
        f"HBD 1-4, HBA 3-8"
    )
    domain = base.replace("{specific_requirements}", req)
    tp = prompts.get("target_specific_prompts", {}).get(TARGET_KEYS.get(target, target), "")
    if tp:
        domain += "\n\n" + tp.strip()
    if use_cot:
        cot = prompts.get("cot_reasoning_templates", {}).get(
            "step_by_step_design", "").strip()
        if cot:
            domain += "\n\n" + cot
    domain += ("\n\nBased on the above analysis, generate a novel GBM candidate "
               "molecule, including SMILES string and design rationale.")
    return domain


CELL_STATES = [
    "Mesenchymal-like GBM cells",
    "Neural progenitor-like GBM cells",
    "Astrocyte-like GBM cells",
    "Oligodendrocyte precursor-like GBM cells",
]

METRIC_VARIANTS = [
    ("good permeability (logBB > -0.5)", "high", "low"),
    ("moderate permeability (logBB > -0.5)", "moderate", "moderate acceptable"),
    ("good permeability (logBB > -0.5)", "moderate", "low"),
    ("moderate permeability (logBB > -0.5)", "high", "moderate acceptable"),
]

SMILES_RE = re.compile(r"SMILES\s*:\s*([^\n]*)")
# 多片段分隔（盐/混合物）以及截断后常见噪声
FRAG_SPLIT = re.compile(r"[.\s]")


def _best_fragment(raw: str) -> str:
    """从模型输出里挑出最有信息量的单个 SMILES 片段。

    模型常常输出多个片段（. 分隔的盐、或截断残留），也常常把 SMILES
    拼上解释文字。这里按"能过 RDKit 且重原子数最多"的规则挑主片段，
    避免把整串噪声当成一个分子。
    """
    raw = raw.strip()
    # 去掉尾部常见噪声 token
    raw = raw.split(">>")[0]
    raw = re.sub(r"[,;].*$", "", raw)
    raw = raw.rstrip(".")

    best, best_score = None, -1
    # 先按 '.' 拆成候选片段（含空）
    frags = [f for f in raw.split(".") if f.strip()]
    for f in frags:
        f = f.strip().rstrip(".").strip()
        if not f:
            continue
        try:
            mol = Chem.MolFromSmiles(f)
        except Exception:
            mol = None
        if mol is None:
            continue
        n = mol.GetNumHeavyAtoms()
        if n < 5:          # 太短的片段（单个甲基/卤素）无意义
            continue
        if n > best_score:
            best, best_score = f, n
    return best if best is not None else raw


def _heavy_atoms(smiles: str) -> int:
    try:
        m = Chem.MolFromSmiles(smiles)
        return m.GetNumHeavyAtoms() if m is not None else 0
    except Exception:
        return 0


class _NewlineStopper(StoppingCriteria):
    """训练集的 response 形如 "SMILES: <one smiles>"（单行）。

    模型常常在 SMILES 之后继续输出混合物 / 解释文字，
    这里在检测到换行 token 时立即停止，既保证分子完整，
    又能把每个样本的生成时间砍掉一半以上。
    """

    def __init__(self, newline_token_id: int, prompt_len: int):
        self.newline_token_id = newline_token_id
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # 只看本步新生成的 token
        gen = input_ids[0, self.prompt_len:]
        return bool(len(gen) > 0 and int(gen[-1]) == self.newline_token_id)


def extract_candidate(resp: str) -> str:
    m = SMILES_RE.search(resp)
    if m:
        return _best_fragment(m.group(1))
    # 没有 "SMILES:" 标记时，退而求其次：取第一段像 SMILES 的文本
    for line in resp.strip().split("\n"):
        line = line.strip()
        if re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\%/]+$", line) and len(line) > 8:
            return _best_fragment(line)
    return resp.strip().split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=str, default="EGFR,VEGFR,IDH1")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--min_heavy_atoms", type=int, default=12)
    ap.add_argument("--adapter", type=str,
                    default=str(PROJECT_ROOT / "saves" /
                                "Llamole-Qwen2-7B-Instruct-Adapter-gbm-real"))
    ap.add_argument("--prompts", type=str,
                    default=str(GBM_ROOT / "configs/english_gbm_prompts.yaml"))
    ap.add_argument("--out", type=str,
                    default=str(GBM_ROOT / "experiments/ours_real"))
    ap.add_argument("--library_name", type=str, default="GBM-PharmaMind (Ours)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = yaml.safe_load(open(args.prompts))

    print("=" * 70)
    print("用新 LoRA 生成 3 靶点分子")
    print("=" * 70)
    print(f"adapter: {args.adapter}")

    model, tokenizer = load_merged_llamole_gbm(
        base_model_path=str(PROJECT_ROOT / "models/Qwen2-7B-Instruct"),
        llamole_adapter=str(PROJECT_ROOT / "saves/Llamole-Qwen2-7B-Instruct-Adapter"),
        gbm_adapter=args.adapter,
        device=f"cuda:{args.gpu_id}",
    )

    import csv as _csv

    # 换行 token id（Qwen2 用 "Ċ" 编码）
    _newline_token_id = tokenizer.encode("\n", add_special_tokens=False)
    if not _newline_token_id:
        _newline_token_id = [tokenizer.eos_token_id]
    _newline_token_id = int(_newline_token_id[-1])
    print(f"newline token id = {_newline_token_id}")

    summary = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        print(f"\n{'='*70}\n🎯 {target}\n{'='*70}")
        rows, seen = [], set()
        n_raw_valid = n_repaired = n_failed = 0
        raw_log = []
        t0 = time.time()

        for i in range(args.n):
            variant = i % len(CELL_STATES)
            cell = CELL_STATES[variant]
            bbb, act, tox = METRIC_VARIANTS[variant % len(METRIC_VARIANTS)]
            metrics = (f"BBB: {bbb}; GBM Activity: {act}; Toxicity: {tox}; "
                       f"Selectivity: high (GBM/normal ratio)")
            use_cot = (i % 2 == 0)
            instr = build_prompt(prompts, target, cell, metrics, use_cot)

            torch.manual_seed(9000 + i)
            full = f"Instruction: {instr}\nInput:\nResponse:"
            inputs = tokenizer(full, return_tensors="pt").to(model.device)
            ilen = inputs["input_ids"].shape[1]
            _newline_stopper = _NewlineStopper(_newline_token_id, ilen)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=args.temperature > 0.01,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([_newline_stopper]),
                )
            resp = tokenizer.decode(out[0][ilen:], skip_special_tokens=True)
            raw_log.append(f"--- variant {variant} (cot={use_cot}) ---\n"
                           f">>> {resp.strip()[:600]}\n")

            cand = extract_candidate(resp)
            fixed, changed = repair_smiles(cand)
            if fixed is None:
                n_failed += 1
                continue
            # 类药性下限：重原子太少（< 12）的片段是生成噪声而非分子，
            # 与 rescore_parallel 的类药预筛口径保持一致
            if _heavy_atoms(fixed) < args.min_heavy_atoms:
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
        print(f"  去重后 {len(rows)} 个唯一分子  ({time.time()-t0:.0f}s)")

        csv_path = out_dir / f"{target}.csv"
        with open(csv_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["entity_id", "mol_id", "library_name", "target", "smiles"])
            for i, s in enumerate(rows):
                w.writerow([target, f"{args.library_name}_{target}_{i+1}",
                            args.library_name, target, s])
        (out_dir / f"{target}_raw_log.txt").write_text("\n".join(raw_log))
        summary[target] = {
            "n_requested": args.n, "n_unique": len(rows),
            "n_raw_valid": n_raw_valid, "n_repaired": n_repaired,
            "n_failed": n_failed, "csv": str(csv_path),
        }

    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"\n完成 -> {out_dir}")


if __name__ == "__main__":
    main()
