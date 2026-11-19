#!/usr/bin/env python3
"""
适配器 key 前缀规范化
=====================

问题:
    GBM 适配器是在"已经套了一层 PeftModel 的模型"上再套一层保存的,
    所以它的 safetensors key 多了两层 base_model.model.:

        checkpoint: base_model.model.base_model.model.model.layers.0....
        peft 期望:  base_model.model.model.layers.0....

    后果:
        PeftModel.from_pretrained() 找不到对应 key, 会打印
        "Found missing adapter keys while loading the checkpoint" 警告,
        然后**静默跳过** —— LoRA 权重根本没加载, 模型退化成未微调的基座。
        表现为: 生成输出退化成逆合成文本 / 重复循环, 而不是分子 SMILES。

修复:
    生成一份前缀已规范化的适配器副本, 使所有现有脚本
    (generate_tot_molecules.py / batch_generate_all_targets.py 等)
    都能用原生 PeftModel.from_pretrained() 正确加载。

用法:
    python scripts/fix_adapter_prefix.py
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def normalize_prefix(state: dict) -> dict:
    """
    把 "base_model.model.base_model.model.model.X" 规范化为 "base_model.model.model.X"。

    同时处理 modules_to_save 的 embed_tokens / lm_head（它们不是 LoRA，
    key 里没有 lora_A/lora_B）。
    """
    out = {}
    DOUBLE = "base_model.model.base_model.model."
    SINGLE = "base_model.model."

    for k, v in state.items():
        new_k = k
        if k.startswith(DOUBLE):
            new_k = SINGLE + k[len(DOUBLE):]
        elif k.startswith(SINGLE):
            # 已经是单层前缀，保持
            pass
        out[new_k] = v
    return out


def fix_adapter(src: str, dst: str, overwrite: bool = False) -> dict:
    src_p, dst_p = Path(src), Path(dst)

    if dst_p.exists() and not overwrite:
        print(f"  [skip] {dst_p} 已存在")
        return {"status": "exists"}

    st_path = src_p / "adapter_model.safetensors"
    if not st_path.exists():
        print(f"  [FAIL] 找不到 {st_path}")
        return {"status": "missing"}

    state = load_file(str(st_path))
    n_before = len(state)
    normalized = normalize_prefix(state)
    n_changed = sum(1 for k in state if normalized.get(k) is not state[k])

    dst_p.mkdir(parents=True, exist_ok=True)
    for f in src_p.iterdir():
        if f.is_file() and f.name != "adapter_model.safetensors":
            shutil.copy2(f, dst_p / f.name)

    save_file(normalized, str(dst_p / "adapter_model.safetensors"))

    print(f"  ✓ {dst_p.name}: {n_before} 个 key, 规范化 {n_changed} 个")
    return {"status": "ok", "n_keys": n_before, "n_changed": n_changed}


def main():
    parser = argparse.ArgumentParser(description="Normalize LoRA adapter key prefixes")
    parser.add_argument(
        "--src",
        type=str,
        default=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models"),
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("适配器 key 前缀规范化")
    print("=" * 70)

    src = Path(args.src)
    state = load_file(str(src / "adapter_model.safetensors"))
    n_double = sum(1 for k in state if k.startswith("base_model.model.base_model.model."))
    print(f"  源适配器: {src}")
    print(f"  总 key 数: {len(state)}, 其中双层前缀: {n_double}")

    res = fix_adapter(args.src, args.dst, overwrite=args.overwrite)
    print(f"\n结果: {res}")

    # 校验
    dst_state = load_file(str(Path(args.dst) / "adapter_model.safetensors"))
    samples = [k for k in list(dst_state)[:3]]
    print("\n规范化后的 key 样例:")
    for k in samples:
        print(f"  {k}")

    # 确认 modules_to_save 设置仍然一致
    cfg = json.loads((Path(args.dst) / "adapter_config.json").read_text())
    print(f"\nadapter_config modules_to_save: {cfg.get('modules_to_save')}")
    print(f"adapter_config target_modules: {cfg.get('target_modules')}")
    print(
        "\n提示: 若 adapter_config 的 target_modules 缺少 mlp 模块"
        "(down_proj/gate_proj/up_proj)，而 checkpoint 里实际包含它们，"
        "PeftModel 依然能加载（PEFT 按 key 匹配），但建议补全以避免歧义。"
    )


if __name__ == "__main__":
    main()
