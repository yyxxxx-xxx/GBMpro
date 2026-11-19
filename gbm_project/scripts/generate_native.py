#!/usr/bin/env python3
"""
GBM LoRA 原生提示生成（对齐训练分布）
========================================

关键发现:
    GBM LoRA 的训练数据格式是 (instruction, response) 配对，例如

        instruction: "Generate a GBM drug candidate SMILES for Mesenchymal-like GBM cells.
                       Target metrics: BBB: good permeability (logBB > -0.5); GBM Activity: good;
                       Toxicity: moderate acceptable; Selectivity: high (GBM/normal ratio)
                       Output format: SMILES: <canonical_SMILES>"
        response:    "SMILES: O=C(Nc1ccccc1)c1ccccc1"

    所以推理时必须:
        1. 用同样的 "Output format: SMILES: <canonical_SMILES>" 结尾
        2. 让模型续写 "SMILES: " 之后的内容 —— 这就是它学到的模式
        3. 每次只生成一个分子，多样性靠采样温度/不同前缀实现

    用 chat template 或 few-shot 提示都会让模型脱离训练分布，
    输出退化成逆合成叙述或重复循环。

用法:
    python scripts/generate_native.py --target EGFR --n 20 --temperature 1.0
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

from gbm_project.improved_smiles_extraction import validate_smiles
from property_pipeline.model_loader import load_merged_llamole_gbm


# 靶点 -> 训练数据里的细胞状态描述 + 目标指标措辞
TARGET_PROFILES = {
    "EGFR": {
        "cells": "Oligodendrocyte progenitor-like GBM cells",
        "bbb": "good permeability (logBB > -0.5)",
        "activity": "good",
        "toxicity": "low",
        "selectivity": "high (GBM/normal ratio)",
        "extra": "Target: EGFR kinase domain inhibitor with quinazoline core and acrylamide warhead.",
    },
    "VEGF": {
        "cells": "Oligodendrocyte progenitor-like GBM cells",
        "bbb": "good permeability (logBB > -0.5)",
        "activity": "good",
        "toxicity": "low",
        "selectivity": "high (GBM/normal ratio)",
        "extra": "Target: VEGFR2/KDR kinase domain inhibitor.",
    },
    "IDH1": {
        "cells": "Mesenchymal-like GBM cells",
        "bbb": "moderate permeability (logBB > -0.5)",
        "activity": "moderate",
        "toxicity": "moderate acceptable",
        "selectivity": "high (GBM/normal ratio)",
        "extra": "Target: mutant IDH1 R132H inhibitor.",
    },
}


def build_instruction(target: str, variant: int = 0) -> str:
    """
    构造与训练数据同分布的 instruction。

    variant 用于制造多样性：不同的措辞变体让采样分布铺开，
    而不是 20 次完全相同的条件（那只会得到几乎相同的分子）。
    """
    p = TARGET_PROFILES.get(target, TARGET_PROFILES["EGFR"])

    bbb = p["bbb"]
    activity = p["activity"]
    tox = p["toxicity"]
    sel = p["selectivity"]

    if variant % 3 == 1:
        bbb = "moderate permeability (logBB > -0.5)"
        activity = "moderate"
    elif variant % 3 == 2:
        bbb = "good permeability (logBB > -0.5)"
        activity = "good"
        tox = "moderate acceptable"

    return (
        f"Generate a GBM drug candidate SMILES for {p['cells']}.\n"
        f"{p['extra']}\n"
        f"Target metrics: BBB: {bbb}; GBM Activity: {activity}; "
        f"Toxicity: {tox}; Selectivity: {sel}\n"
        f"Output format: SMILES: <canonical_SMILES>"
    )


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 1.0, top_p: float = 0.95) -> str:
    """
    单次续写生成。

    训练时的确切文本格式（见 gbm_lora_finetuner.tokenize_function）:
        "Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
    其中 input_text 是空字符串。所以推理必须构造:

        Instruction: <instruction>
        Input:
        Response:

    缺了 "Response:" 前缀，模型会续写出 "Input: \nResponse: SMILES: ..." 这种
    嵌套结构，导致提取到的 SMILES 前面粘上多余内容。

    不使用 chat template —— 训练时是纯文本续写。
    """
    full_prompt = f"Instruction: {prompt}\nInput:\nResponse:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
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
            repetition_penalty=1.0,
        )
    return tokenizer.decode(out[0][ilen:], skip_special_tokens=True)


SMILES_RE = re.compile(r"SMILES\s*:\s*([^\n]+)")


def extract_smiles_from_response(resp: str) -> List[str]:
    """
    从模型响应中提取 SMILES。

    模型有时会在一行里输出多个以 "." 分隔的片段（训练数据里的 response
    偶尔就是这种形式），有时只有一个。需要都尝试。
    """
    out: List[str] = []
    m = SMILES_RE.search(resp)
    if m:
        tail = m.group(1).strip()
        # 去掉句尾标点
        tail = tail.split(">>")[0].strip()
        out.append(tail)
    # 兜底: 直接找看起来像 SMILES 的 token
    if not out:
        for m2 in re.finditer(r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.]{8,})\b", resp):
            out.append(m2.group(1).strip())
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate molecules using the native GBM LoRA prompt format")
    parser.add_argument("--targets", type=str, default="EGFR,VEGF,IDH1")
    parser.add_argument("--n", type=int, default=20, help="每个靶点生成数量")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--library_name", type=str, default="GBM-PharmaMind")
    args = parser.parse_args()

    out_dir = Path(args.output_dir or (GBM_ROOT / "experiments" / "native_gen"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_merged_llamole_gbm(
        base_model_path=str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        llamole_adapter=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        gbm_adapter=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed"),
        device=f"cuda:{args.gpu_id}",
    )

    summary = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        print(f"\n{'=' * 70}\n🎯 {target}\n{'=' * 70}")
        smiles_list: List[str] = []
        raw_log = []

        for i in range(args.n):
            variant = i % 3
            prompt = build_instruction(target, variant)
            resp = generate_one(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            raw_log.append(f"--- prompt variant {variant} ---\n{prompt}\n>>> {resp}\n")

            cands = extract_smiles_from_response(resp)
            picked = None
            for c in cands:
                if validate_smiles(c):
                    picked = c
                    break
            if picked and picked not in smiles_list:
                smiles_list.append(picked)
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{args.n} 采样, 有效去重 {len(smiles_list)}")

        print(f"  ✓ 有效分子 {len(smiles_list)}/{args.n}")
        (out_dir / f"{target}_raw_log.txt").write_text("\n".join(raw_log), encoding="utf-8")

        import csv as _csv

        csv_path = out_dir / f"{target}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["entity_id", "mol_id", "library_name", "target", "smiles"])
            for i, smi in enumerate(smiles_list):
                w.writerow([
                    target,
                    f"{args.library_name}_{target}_{i + 1}",
                    args.library_name,
                    target,
                    smi,
                ])
        print(f"  💾 {csv_path}")
        summary[target] = {"n": len(smiles_list), "csv": str(csv_path)}

    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n完成。")


if __name__ == "__main__":
    main()
