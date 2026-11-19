#!/usr/bin/env python3
"""
P0 诊断：验证 GBM adapter 的 LoRA 权重在推理加载路径上是否真的生效。

背景：
  GBM adapter 是在"已经套了一层 PeftModel 的 Llamole 模型"上再套一层训练并保存的，
  safetensors key 多了两层 base_model.model. 前缀：
      base_model.model.base_model.model.model.layers.0...
  而 PEFT 期望：
      base_model.model.model.layers.0...
  key 对不上 -> PeftModel.from_pretrained 打印 missing keys 警告后**静默跳过**，
  lora_B 初始化为 0 -> 数学上等价于恒等映射 -> 模型退化成未微调基座。

本脚本对每个候选 adapter 逐个检查：
  1. safetensors key 集合 vs PEFT 期望 key 集合（交集/缺失/多余）
  2. adapter_config 声明的 target_modules/r 与实际 tensor 形状是否一致
  3. 实际加载后 lora_B 是否非零（真正的判据）
  4. embed_tokens / lm_head 是否还在（modules_to_save 是否被丢掉）
"""
import os
import sys
import json
import glob
import warnings

import torch
from safetensors.torch import load_file

BASE = "/root/Llamole-main/models/Qwen2-7B-Instruct"
SAVES = "/root/Llamole-main/saves"
NUM_LAYERS = 28
SELF_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP = {"gate_proj", "up_proj", "down_proj"}


def expected_keys(target_modules):
    """按 Qwen2-7B 结构生成 PEFT 期望的 key 集合"""
    exp = set()
    for layer in range(NUM_LAYERS):
        for m in target_modules:
            parents = []
            if m in SELF_ATTN:
                parents.append("self_attn")
            if m in MLP:
                parents.append("mlp")
            for p in parents:
                for ab in ("lora_A", "lora_B"):
                    exp.add(f"base_model.model.model.layers.{layer}.{p}.{m}.{ab}.weight")
    return exp


def diagnose(adapter_dir, verbose=True):
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not (os.path.exists(cfg_path) and os.path.exists(st_path)):
        return {"dir": adapter_dir, "status": "missing_files"}

    cfg = json.load(open(cfg_path))
    state = load_file(st_path)
    got = set(state.keys())
    tm = set(cfg.get("target_modules") or [])
    r = cfg.get("r")

    exp = expected_keys(tm)
    inter = exp & got
    missing = exp - got
    extra = got - exp

    # 形状一致性检查：PEFT 要求 lora_A 形状 (r, in_features), lora_B 形状 (out_features, r)
    shape_bad = []
    for k, v in state.items():
        if "lora_A" in k and v.shape[0] != r:
            shape_bad.append((k, tuple(v.shape), r))

    has_embed = any("embed_tokens" in k for k in got)
    has_lm_head = any("lm_head" in k for k in got)

    res = {
        "dir": os.path.basename(adapter_dir),
        "n_tensors": len(got),
        "r": r,
        "target_modules": sorted(tm),
        "modules_to_save": cfg.get("modules_to_save"),
        "expected_n": len(exp),
        "matched": len(inter),
        "missing": len(missing),
        "extra": len(extra),
        "shape_mismatch": len(shape_bad),
        "has_embed_tokens": has_embed,
        "has_lm_head": has_lm_head,
    }
    if verbose:
        print(f"\n--- {res['dir']} ---")
        print(f"  r={r}  target_modules={res['target_modules']}")
        print(f"  tensors: got={len(got)}  expected={len(exp)}  "
              f"matched={len(inter)}  missing={len(missing)}  extra={len(extra)}")
        print(f"  shape mismatch (lora_A dim != r): {len(shape_bad)}")
        if shape_bad:
            print(f"    e.g. {shape_bad[0][0]} {shape_bad[0][1]} vs r={shape_bad[0][2]}")
        print(f"  embed_tokens saved: {has_embed}   lm_head saved: {has_lm_head}")
        if missing:
            print(f"  missing sample: {sorted(missing)[:2]}")
    return res


def main():
    print("=" * 74)
    print("P0 诊断：GBM adapter LoRA 权重加载有效性")
    print("=" * 74)

    adapters = sorted(glob.glob(os.path.join(SAVES, "*Adapter*")))
    results = []
    for a in adapters:
        results.append(diagnose(a))

    # 实际加载测试（用一个 adapter 做真实 PEFT 加载，看 lora_B 是否非零）
    print("\n" + "=" * 74)
    print("真实 PEFT 加载测试（Qwen2-7B-Instruct）")
    print("=" * 74)

    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map={"": 0},
        trust_remote_code=True, local_files_only=True,
    )

    for a in adapters:
        name = os.path.basename(a)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                pm = PeftModel.from_pretrained(model, a)
                bvals = [float(p.abs().mean()) for n, p in pm.named_parameters()
                         if "lora_B" in n]
                nz = sum(1 for v in bvals if v > 1e-8)
                warns = [str(x.message)[:110] for x in w if "missing" in str(x.message).lower()]
                print(f"\n{name}:")
                print(f"  loaded OK | lora_B tensors={len(bvals)}  nonzero={nz}")
                print(f"  mean|lora_B| = {sum(bvals)/max(len(bvals),1):.6f}")
                if warns:
                    print(f"  WARNING: {warns[0]}")
                if nz == 0:
                    print("  >>> 结论: LoRA 权重全为零 -> 等价于未微调基座 (BUG)")
                pm = pm.unload()
            except Exception as e:
                print(f"\n{name}: LOAD FAILED -> {type(e).__name__}: {str(e)[:200]}")
                try:
                    model = model.unload()
                except Exception:
                    pass

    print("\n" + "=" * 74)
    print("诊断结论汇总")
    print("=" * 74)
    for r in results:
        if "status" in r:
            continue
        ok = (r["matched"] == r["expected_n"] and r["shape_mismatch"] == 0)
        print(f"  {r['dir']:<58} key完整={ok}  matched={r['matched']}/{r['expected_n']}")


if __name__ == "__main__":
    main()
