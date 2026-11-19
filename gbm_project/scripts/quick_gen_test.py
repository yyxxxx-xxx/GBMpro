#!/usr/bin/env python3
"""
快速生成测试（用于验证 prompt 格式是否正确）
=============================================

单靶点、单轮、少量分子，专用于快速迭代调试 prompt 格式。

用法:
    python scripts/quick_gen_test.py --target EGFR --n 5
"""

import os
import re
import sys
import json
import yaml
import argparse
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

from gbm_project.src.gbm_prompt_generator import GBMPromptGenerator
from gbm_project.src.gbm_knowledge_base import GBMKnowledgeBase
from gbm_project.improved_smiles_extraction import extract_multiple_smiles, validate_smiles
from property_pipeline.model_loader import load_merged_llamole_gbm

PROMPTS = {
    "EGFR": (
        "Generate valid SMILES for {target} drug molecules targeting brain tumours.\n\n"
        "Example 1:\n"
        "SMILES 1: Cc1cc(nc(-c2ccc(Cl)cc2)n1)C(=O)Nc3ccc(F)cc3\n"
        "Example 2:\n"
        "SMILES 1: COc1ccc(cc1)C(=O)Nc2cc(C)nc(-c3ccccn3)c2\n"
        "Example 3:\n"
        "SMILES 1: CC(=O)Nc1ccc(-n2ccnc2)cc1C(=O)Nc3ccccc3\n\n"
        "Now generate {n} new valid SMILES:\n"
        "SMILES 1:"
    ),
    "VEGF": (
        "Generate valid SMILES for VEGFR2 drug molecules targeting brain tumours.\n\n"
        "Example 1:\n"
        "SMILES 1: Cc1cc(nc(-c2ccc(Cl)cc2)n1)C(=O)Nc3ccc(F)cc3\n"
        "Example 2:\n"
        "SMILES 1: COc1ccc(cc1)C(=O)Nc2cc(C)nc(-c3ccccn3)c2\n"
        "Example 3:\n"
        "SMILES 1: CC(=O)Nc1ccc(-n2ccnc2)cc1C(=O)Nc3ccccc3\n\n"
        "Now generate {n} new valid SMILES:\n"
        "SMILES 1:"
    ),
    "IDH1": (
        "Generate valid SMILES for mutant IDH1 drug molecules targeting brain tumours.\n\n"
        "Example 1:\n"
        "SMILES 1: Cc1cc(nc(-c2ccc(Cl)cc2)n1)C(=O)Nc3ccc(F)cc3\n"
        "Example 2:\n"
        "SMILES 1: COc1ccc(cc1)C(=O)Nc2cc(C)nc(-c3ccccn3)c2\n"
        "Example 3:\n"
        "SMILES 1: CC(=O)Nc1ccc(-n2ccnc2)cc1C(=O)Nc3ccccc3\n\n"
        "Now generate {n} new valid SMILES:\n"
        "SMILES 1:"
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="EGFR")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_domain_prompt", action="store_true")
    parser.add_argument("--no_gbm", action="store_true",
                        help="只加载 Llamole 适配器（不加载 GBM LoRA），用于对照实验")
    parser.add_argument("--native_load", action="store_true",
                        help="用 ToT 脚本的原生加载路径（原始双前缀适配器）")
    args = parser.parse_args()

    if args.native_load:
        # 复现 generate_tot_molecules.py 的加载路径：
        #   base -> merge Llamole -> PeftModel.from_pretrained(GBM 原始适配器)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        bp = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(bp, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(bp, torch_dtype=torch.float16)
        base = PeftModel.from_pretrained(
            base, str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")
        ).merge_and_unload()
        if args.no_gbm:
            model = base
        else:
            model = PeftModel.from_pretrained(
                base,
                str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed"),
            )
        model = model.to(f"cuda:{args.gpu_id}")
        model.eval()
        print("[native load] ok")
    else:
        gbm = (
            str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed")
            if args.no_gbm
            else str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed")
        )
        model, tokenizer = load_merged_llamole_gbm(
            base_model_path=str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
            llamole_adapter=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
            gbm_adapter=gbm,
            device=f"cuda:{args.gpu_id}",
        )

    prompt = PROMPTS.get(args.target, PROMPTS["EGFR"]).format(n=args.n, target=args.target)

    if args.use_domain_prompt:
        kb = GBMKnowledgeBase(
            str(GBM_ROOT / "data/gbm_targets/gbm_targets.json"),
            str(GBM_ROOT / "data/gbm_clinical/gbm_clinical_data.json"),
            str(GBM_ROOT / "data/gbm_molecules/gbm_molecules.json"),
        )
        pg = GBMPromptGenerator(kb, str(GBM_ROOT / "configs/gbm_prompts.yaml"))
        dp = pg.generate_domain_prompt(args.target, constraints={"molecular_weight_max": 500})
        prompt = dp + "\n\n" + prompt

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"--- prompt (前 400 字符) ---\n{formatted[:400]}\n---")

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    ilen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][ilen:], skip_special_tokens=True)
    print("=" * 60)
    print("RAW OUTPUT:")
    print(raw[:2000])
    print("=" * 60)

    smis = []
    for s in extract_multiple_smiles(raw, max_count=60):
        s = s.strip()
        if s and s not in smis and validate_smiles(s):
            smis.append(s)
    print(f"\n有效 SMILES: {len(smis)}")
    for i, s in enumerate(smis[:args.n], 1):
        print(f"  {i}. {s}")


if __name__ == "__main__":
    main()
