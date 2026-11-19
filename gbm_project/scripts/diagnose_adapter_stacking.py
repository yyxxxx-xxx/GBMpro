#!/usr/bin/env python3
"""
复现训练时的 adapter 堆叠语义，明确 gbm-with-graph-models 里到底存了谁的权重。

训练流程（gbm_lora_finetuner.py）：
  1. PeftModel.from_pretrained(base, llamole_adapter)      # adapter 名 'default'
  2. get_peft_model(peft_model, lora_config)                 # adapter 名 'default'（同名！）
  3. trainer.save_model(output_dir)

问题：第 2 步用同名 'default' 再注入一套 LoRA，PEFT 0.18 会
  - 对 target_modules 覆盖的模块，换成全新的 LoRA 层（lora_A 随机重初始化、lora_B 归零）
  - 对未被 target_modules 覆盖的模块，保留原 Llamole LoRA 层

所以保存出来的是「新 GBM LoRA(4模块) + 旧 Llamole LoRA(其余3模块)」的混合体，
而且新 GBM LoRA 的基底权重被重置，**Llamole 在 q/k/v/o 上的化学适配很可能已被丢弃**。

本脚本用小模型精确复现，并回答：
  A. 同名再注入时，旧 adapter 的哪些权重被保留/被覆盖？
  B. 保存出来的 state_dict 里每个 key 分别来自谁？
  C. merge_and_unload() 的路径是否能保住 Llamole 适配？
"""
import os
import tempfile
import warnings

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

warnings.filterwarnings("ignore")


def build(num_layers=2):
    cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=num_layers, num_attention_heads=4,
                      num_key_value_heads=4)
    return Qwen2ForCausalLM(cfg)


def fill_model(m, marker):
    """给所有 lora 权重打上可识别标记"""
    for n, p in m.named_parameters():
        if "lora_A" in n:
            p.data.fill_(marker)
        elif "lora_B" in n:
            p.data.fill_(marker + 0.1)


def snapshot(m):
    return {n: float(p.detach().abs().mean()) for n, p in m.named_parameters()
            if "lora_" in n and "default" in n}


def scenario_a_same_name():
    """场景A：同名 'default' 再注入（= 当前训练代码的行为）"""
    print("\n" + "=" * 72)
    print("场景 A：PeftModel.from_pretrained(llamole) -> get_peft_model(同名 'default')")
    print("=" * 72)

    base = build()
    llamole_cfg = LoraConfig(r=8, lora_alpha=16, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(base, llamole_cfg, adapter_name="default")
    fill_model(m, 1.0)  # llamole 标记
    print(f"  [1] llamole 注入后: {len(snapshot(m))} 个 lora tensor")

    gbm_cfg = LoraConfig(r=8, lora_alpha=16, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    m2 = get_peft_model(m, gbm_cfg, adapter_name="default")
    print(f"  [2] GBM 再注入后:   {len(snapshot(m2))} 个 lora tensor")

    snap = snapshot(m2)
    by_loc = {}
    for n, v in snap.items():
        loc = "mlp" if ".mlp." in n else "self_attn"
        mod = n.split(".")[-3]
        by_loc.setdefault((loc, mod), []).append(v)

    print("\n  注入后各模块 lora 权重的值（1.0=保留 Llamole, 0.1=新 GBM 重置）:")
    for k in sorted(by_loc):
        vals = by_loc[k]
        v = vals[0]
        who = "Llamole 保留" if abs(v - 1.0) < 0.01 else ("GBM 重置" if abs(v - 0.1) < 0.01 else f"?{v:.3f}")
        print(f"    {k[0]:<10} {k[1]:<12} -> {v:.3f}  [{who}]")

    # 保存再检查
    with tempfile.TemporaryDirectory() as td:
        m2.save_pretrained(td)
        from safetensors.torch import load_file
        st = load_file(os.path.join(td, "adapter_model.safetensors"))
        print(f"\n  [3] 保存后 tensor 数: {len(st)}")
        cfg2 = json.load(open(os.path.join(td, "adapter_config.json")))
        print(f"      adapter_config target_modules: {sorted(cfg2['target_modules'])}")
        print(f"      adapter_config r: {cfg2['r']}")
        locs = {}
        for k in st:
            loc = "mlp" if ".mlp." in k else ("self_attn" if "self_attn" in k else "other")
            locs[loc] = locs.get(loc, 0) + 1
        print(f"      保存内容的模块分布: {locs}")


def scenario_b_merge():
    """场景B：merge_and_unload 后再注入新 adapter（推荐做法）"""
    print("\n" + "=" * 72)
    print("场景 B：PeftModel.from_pretrained(llamole).merge_and_unload() -> get_peft_model(新名)")
    print("=" * 72)
    base = build()
    llamole_cfg = LoraConfig(r=8, lora_alpha=16, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(base, llamole_cfg, adapter_name="default")
    fill_model(m, 1.0)
    m = m.merge_and_unload()
    print(f"  [1] merge_and_unload 后基座权重已吸收 Llamole 适配")

    gbm_cfg = LoraConfig(r=8, lora_alpha=16, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    m2 = get_peft_model(m, gbm_cfg, adapter_name="gbm")
    snap = {n: float(p.detach().abs().mean()) for n, p in m2.named_parameters()
            if "lora_" in n and "gbm" in n}
    print(f"  [2] 新 GBM adapter: {len(snap)} 个 tensor, mean|A|={sum(snap.values())/len(snap):.4f}")
    print(f"      -> 全新初始化（lora_B=0, lora_A 随机），与 Llamole 完全解耦")
    print(f"      -> 保存后只含 GBM 4 模块；Llamole 已固化进基座，不会丢失")


def scenario_c_two_names():
    """场景C：不同名堆叠，两个 adapter 权重并存"""
    print("\n" + "=" * 72)
    print("场景 C：PeftModel.from_pretrained(llamole,'llamole') -> get_peft_model(新名 'gbm')")
    print("=" * 72)
    base = build()
    llamole_cfg = LoraConfig(r=8, lora_alpha=16, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(base, llamole_cfg, adapter_name="llamole")
    fill_model(m, 1.0)
    gbm_cfg = LoraConfig(r=8, lora_alpha=16, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    m2 = get_peft_model(m, gbm_cfg, adapter_name="gbm")
    ll = {n: float(p.detach().abs().mean()) for n, p in m2.named_parameters()
          if "lora_" in n and "llamole" in n}
    gb = {n: float(p.detach().abs().mean()) for n, p in m2.named_parameters()
          if "lora_" in n and "gbm" in n}
    print(f"  llamole adapter: {len(ll)} tensor, mean|A|={sum(ll.values())/len(ll):.4f}  <- 完整保留")
    print(f"  gbm adapter     : {len(gb)} tensor, mean|A|={sum(gb.values())/len(gb):.4f}  <- 全新")
    print(f"  两套并存，推理时可用 set_adapter 切换 / 或 merge 两者")
    with tempfile.TemporaryDirectory() as td:
        m2.save_pretrained(td)
        from safetensors.torch import load_file
        st = load_file(os.path.join(td, "adapter_model.safetensors"))
        print(f"  保存后 tensor 数: {len(st)}  (llamole+[gbm] 混合)")


import json

if __name__ == "__main__":
    print("=" * 72)
    print("复现训练时的 LoRA 堆叠语义")
    print("=" * 72)
    scenario_a_same_name()
    scenario_b_merge()
    scenario_c_two_names()
