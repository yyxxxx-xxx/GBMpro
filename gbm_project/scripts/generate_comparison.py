#!/usr/bin/env python3
"""
多靶点分子生成（对比实验用）
=============================

使用训练好的 Llamole + GBM LoRA 适配器做纯推理生成（不训练）。
逐靶点生成指定数量的分子，输出统一 schema 的 CSV，供 compute_properties.py 打分。

与 batch_generate_all_targets.py 的区别:
    batch_generate_all_targets.py 直接用 PeftModel.from_pretrained 加载，会因
    新增 special token 导致 embed_tokens 尺寸不匹配而失败。
    本脚本走项目自带的 GBMLoRAFinetuner.load_finetuned_model()（原始 Llamole 加载路径），
    能正确处理 new_special_tokens 的 vocab resize。

用法:
    python scripts/generate_comparison.py --targets EGFR,VEGFR,IDH1 --num_molecules 20
"""

import os
import re
import sys
import json
import yaml
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"
# 注意: 必须让根目录优先, 否则 gbm_project/src 会遮蔽根目录的 src 包
# (Llamole 的 src.hparams / src.model 需要从根目录解析)
sys.path.insert(0, str(PROJECT_ROOT))
if str(GBM_ROOT) not in sys.path:
    sys.path.append(str(GBM_ROOT))
if str(GBM_ROOT / "src") not in sys.path:
    sys.path.append(str(GBM_ROOT / "src"))

from gbm_project.src.gbm_prompt_generator import GBMPromptGenerator
from gbm_project.src.gbm_knowledge_base import GBMKnowledgeBase
from gbm_project.improved_smiles_extraction import extract_multiple_smiles, validate_smiles
from property_pipeline.model_loader import load_merged_llamole_gbm


TARGET_ALIASES = {
    "EGFR": "EGFR",
    "IDH1": "IDH1_IDH2",
    "IDH1_IDH2": "IDH1_IDH2",
    "VEGF": "VEGF_VEGFR",
    "VEGFR": "VEGF_VEGFR",
    "VEGF_VEGFR": "VEGF_VEGFR",
    "MGMT": "MGMT",
    "PD1_PDL1": "PD1_PDL1",
    "PI3K_AKT_mTOR": "PI3K_AKT_mTOR",
    "MDM2": "MDM2",
}

TARGET_CONSTRAINTS = {
    "EGFR": {
        "molecular_weight_max": 500,
        "logp_min": 2.0,
        "logp_max": 4.0,
        "target_bbb": "high",
        "target_selectivity": "EGFRvIII > WT-EGFR >10-fold",
    },
    "VEGF_VEGFR": {
        "molecular_weight_max": 500,
        "logp_min": 1.5,
        "logp_max": 4.0,
        "target_bbb": "moderate",
        "target_selectivity": "VEGFR2 >> VEGFR1",
    },
    "IDH1_IDH2": {
        "molecular_weight_max": 450,
        "logp_min": 1.0,
        "logp_max": 3.5,
        "target_bbb": "high",
        "target_selectivity": "IDH1 R132H > wild-type",
    },
}


def build_prompt(prompt_gen, cot_template, target_key):
    """构造 CoT + 领域知识 + 编号输出格式的完整提示。"""
    domain_prompt = prompt_gen.generate_domain_prompt(
        target_key, constraints=TARGET_CONSTRAINTS.get(target_key, {})
    )
    instruction = (
        "Generate 20 GBM novel drug candidates for this target. "
        "Output each candidate as 'SMILES: <smiles>' on a separate line."
    )
    output_format = """
Output format: Number each candidate as:
1. SMILES_string
2. SMILES_string
...
20. SMILES_string

Requirements:
- Valid SMILES only, no explanation text
- Prefer kinase-inhibitor-like scaffolds (quinazoline / pyrimidine / indazole / benzimidazole)
- Keep molecular weight within 300-500 Da
- Include hydrogen-bond donors and acceptors suitable for target engagement
- Only output the numbered SMILES list, no explanations."""

    return cot_template + "\n\n" + domain_prompt + "\n\n" + instruction + output_format


def generate(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 2048):
    """
    调用模型生成。

    关键: 必须使用 tokenizer 的 chat template。
    Llamole 是基于 Qwen2-Instruct 的 chat 模型，训练时用的是
    "<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n" 格式。
    如果用裸的 "Instruction:/Input:/Response:" 拼接，模型会脱离训练分布，
    输出退化成重复循环（表现为大量 "1,1,1,1,..."）。
    """
    import torch

    messages = [{"role": "user", "content": input_text or instruction}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


def extract_smiles(raw_response, limit=60):
    """
    从模型输出提取 RDKit 验证通过的 SMILES（去重保序）。

    不用简单的 "数字. SMILES" 正则 —— 模型常在一行里输出多个候选，
    正则会把后续内容混进来。改用项目自带的 improved_smiles_extraction，
    它对各种格式（编号列表 / SMILES: xxx / 纯行）都有兜底。
    """
    from gbm_project.improved_smiles_extraction import extract_multiple_smiles

    smiles_list = []
    for s in extract_multiple_smiles(raw_response, max_count=limit * 3):
        s = s.strip()
        if s and s not in smiles_list and validate_smiles(s):
            smiles_list.append(s)
        if len(smiles_list) >= limit:
            break
    return smiles_list[:limit]


def main():
    parser = argparse.ArgumentParser(description="Generate molecules for comparison experiments")
    parser.add_argument("--targets", type=str, default="EGFR,VEGFR,IDH1")
    parser.add_argument("--num_molecules", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--library_name", type=str, default="gbm_pharmamind")
    parser.add_argument("--rounds", type=int, default=1, help="生成轮数（每轮采样一次）")
    args = parser.parse_args()

    base_dir = GBM_ROOT
    out_dir = Path(
        args.output_dir
        or (base_dir / "experiments" / f"comparison_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GBM 多靶点分子生成（对比实验，纯推理）")
    print("=" * 70)

    # 注意: 不能用 gbm_lora_finetuner.load_finetuned_model() —— 它会因
    # "Only one adapter is supported at a time." 抛错，且 PeftModel 直接套第二层
    # 适配器会因 key 前缀多一层 base_model. 而报 embed_tokens size mismatch。
    # 这里改用 property_pipeline.model_loader 的多适配器合并加载器。
    llamole_adapter = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")
    # 注意: 必须用 fix_adapter_prefix.py 生成的规范化副本。
    # 原始适配器的 key 是双层前缀 "base_model.model.base_model.model...",
    # PeftModel.from_pretrained() 找不到 key 会**静默跳过**，
    # 导致 LoRA 权重根本没加载 -> 模型退化成基座 -> 输出变成逆合成文本。
    gbm_adapter = str(
        PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed"
    )

    print("🔄 加载模型（训练好的权重，仅推理）...")
    model, tokenizer = load_merged_llamole_gbm(
        base_model_path=str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        llamole_adapter=llamole_adapter,
        gbm_adapter=gbm_adapter,
        device=f"cuda:{args.gpu_id}",
    )
    print("✅ 模型加载完成")

    kb = GBMKnowledgeBase(
        str(base_dir / "data/gbm_targets/gbm_targets.json"),
        str(base_dir / "data/gbm_clinical/gbm_clinical_data.json"),
        str(base_dir / "data/gbm_molecules/gbm_molecules.json"),
    )
    prompt_gen = GBMPromptGenerator(kb, str(base_dir / "configs/gbm_prompts.yaml"))
    with open(base_dir / "configs/gbm_prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    cot_template = prompts.get("cot_reasoning_templates", {}).get("step_by_step_design", "")

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    summary = {}

    for raw_target in targets:
        target_key = TARGET_ALIASES.get(raw_target, raw_target)
        print("\n" + "=" * 70)
        print(f"🎯 靶点: {raw_target}  (config key: {target_key})")
        print("=" * 70)

        final_prompt = build_prompt(prompt_gen, cot_template, target_key)
        all_smiles = []

        for rnd in range(args.rounds):
            print(f"  [轮次 {rnd + 1}/{args.rounds}] 生成中...")
            raw = generate(model, tokenizer, f"Generate {args.num_molecules} candidates.", final_prompt)
            got = extract_smiles(raw)
            print(f"    提取到 {len(got)} 个有效 SMILES")
            all_smiles.extend(got)

            # 保存原始输出（便于追溯 CoT 推理）
            (out_dir / f"{raw_target}_round{rnd + 1}_raw.txt").write_text(raw, encoding="utf-8")

        # 去重
        unique = list(dict.fromkeys(all_smiles))
        print(f"  合计 {len(all_smiles)} 个，去重后 {len(unique)} 个")

        # 输出 CSV
        csv_path = out_dir / f"{raw_target}.csv"
        import csv as _csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["entity_id", "mol_id", "library_name", "target", "smiles"])
            for i, smi in enumerate(unique):
                w.writerow([target_key, f"{args.library_name}_{raw_target}_{i + 1}", args.library_name, raw_target, smi])
        print(f"  💾 已保存: {csv_path}")

        summary[raw_target] = {
            "target_key": target_key,
            "n_generated": len(all_smiles),
            "n_unique": len(unique),
            "csv": str(csv_path),
            "smiles": unique,
        }

    manifest = out_dir / "generation_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📋 清单: {manifest}")
    print("\n下一步: python scripts/compute_properties.py --input_dir "
          f"{out_dir} --output_dir {out_dir}/scored")


if __name__ == "__main__":
    main()
