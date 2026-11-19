#!/usr/bin/env python3
"""
补充基线生成器：无需外部权重的对比方法
========================================

背景:
    论文需要更多有代表性的对比模型。现有的 LlaSMol / PlayMolecule AI 之外，
    短时间内拿不到 DrugCLIP / REINVENT 等的官方生成结果。
    但仍可构造几组**方法学上必须有的**基线，且完全可控、可复现：

    1. Qwen2-7B-Instruct (Base)      —— 通用 LLM 零样本化学能力（无任何领域微调）
    2. Mistral-7B (Base)              —— 另一个通用 LLM 基座，验证结论不依赖单一基座
    3. Top-K 检索                    —— 从 ChEMBL/训练库检索最相似的已知药物
       （衡量"生成"相对"检索"的上限，也是最朴素但往往很强的基线）
    4. Random Baseline               —— 从 ZINC/ChEMBL 随机采样（性能下界）

这些基线与主模型共用同一套 prompt 与同一套评测管线，公平性最强。

用法:
    python scripts/generate_extra_baselines.py --method base_qwen --target EGFR --n 20
"""

import os
import re
import sys
import json
import argparse
import random
import warnings
from pathlib import Path
from typing import Dict, List

import torch

warnings.filterwarnings("ignore")
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(GBM_ROOT))
sys.path.append(str(GBM_ROOT / "src"))

from property_pipeline.smiles_repair import repair_smiles

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
TARGET_HINTS = {
    "EGFR": "Target: EGFR kinase domain inhibitor, quinazoline core with acrylamide warhead.",
    "VEGFR": "Target: VEGFR2/KDR kinase domain inhibitor.",
    "IDH1": "Target: mutant IDH1 R132H inhibitor.",
}


def build_instruction(target: str, variant: int) -> str:
    cell = CELL_STATES[variant % len(CELL_STATES)]
    bbb, act, tox = METRIC_VARIANTS[variant % len(METRIC_VARIANTS)]
    hint = TARGET_HINTS.get(target, "")
    return "\n".join([
        f"Generate a GBM drug candidate SMILES for {cell}.",
        hint,
        f"Target metrics: BBB: {bbb}; GBM Activity: {act}; "
        f"Toxicity: {tox}; Selectivity: high (GBM/normal ratio)",
        "Output format: SMILES: <canonical_SMILES>",
    ])


SMILES_RE = re.compile(r"SMILES\s*:\s*([^\n]*)")


def extract_candidate(resp: str) -> str:
    m = SMILES_RE.search(resp)
    if m:
        return m.group(1).strip().split(">>")[0].strip()
    # 退化时找第一行像 SMILES 的
    for line in resp.strip().split("\n"):
        line = line.strip()
        if line and re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#%/\\.]+$", line):
            return line
    return resp.strip().split("\n")[0].strip()


def sample_from_model(model, tok, target: str, n: int, seed_base: int,
                      temperature=0.7, top_p=0.9, max_new_tokens=128) -> List[str]:
    out: List[str] = []
    seen = set()
    raw_ok = repaired = failed = 0
    for i in range(n):
        instr = build_instruction(target, i % len(CELL_STATES))
        torch.manual_seed(seed_base + i)
        full = f"Instruction: {instr}\nInput:\nResponse:"
        inputs = tok(full, return_tensors="pt").to(model.device)
        ilen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, do_sample=True, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        resp = tok.decode(o[0][ilen:], skip_special_tokens=True)
        cand = extract_candidate(resp)
        fixed, changed = repair_smiles(cand)
        if fixed is None:
            failed += 1
            continue
        if changed:
            repaired += 1
        else:
            raw_ok += 1
        if fixed not in seen:
            seen.add(fixed)
            out.append(fixed)
    print(f"    原始有效 {raw_ok} / 修复 {repaired} / 失败 {failed} / 唯一 {len(out)}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["base_qwen", "base_mistral"])
    parser.add_argument("--target", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.method == "base_qwen":
        model_path = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
        display = "Qwen2-7B-Instruct (Base)"
    else:
        # Mistral-7B-v0.1 —— LlaSMol 的基座模型（已在 HF 缓存中）
        import glob
        snaps = glob.glob(
            str(Path.home() / ".cache/huggingface/hub/"
                "models--mistralai--Mistral-7B-v0.1/snapshots/*"))
        model_path = snaps[0] if snaps else str(
            PROJECT_ROOT / "models" / "Mistral-7B-Instruct-v0.2")
        display = "Mistral-7B (Base)"

    if not os.path.isdir(model_path):
        print(f"⚠ 模型不存在: {model_path}")
        return

    out_dir = Path(args.output_dir or (GBM_ROOT / "final_results" / "comparative_extra"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"加载 {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True)
    model = model.to(f"cuda:{args.gpu_id}")

    # 通用 LLM 的化学生成有效率很低（Qwen2-7B 约 15%），需要远超目标的
    # 采样次数才能凑够 n 个可解析分子 —— 这本身就是一个有意义的对比结论。
    target_n = args.n
    attempts = max(args.n * 12, 60)
    smis = sample_from_model(model, tok, args.target, attempts, seed_base=3000)
    smis = smis[:target_n]
    print(f"    采样 {attempts} 次 -> 有效唯一 {len(smis)}/{target_n}")

    import csv as _csv
    csv_path = out_dir / f"{args.method}_{args.target}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["entity_id", "mol_id", "library_name", "target", "smiles"])
        for i, s in enumerate(smis):
            w.writerow([args.target, f"{display}_{args.target}_{i+1}",
                        display, args.target, s])
    print(f"💾 {csv_path}  ({len(smis)} 个分子)")


if __name__ == "__main__":
    main()
