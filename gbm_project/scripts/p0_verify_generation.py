#!/usr/bin/env python3
"""
P0 端到端验证：在真实推理路径上，对比三个 adapter 的生成行为。

对照：
  A) gbm-with-graph-models  (当前 configs/gbm_generation.yaml 指向的，key 前缀坏了)
  B) gbm-fixed              (前缀规范化后的同一份权重)
  C) llamole base           (未做 GBM 微调的化学基座，作为参照)

关键判据：LoRA 是否真的生效。
  - 若 A 与 C 输出几乎一致 -> A 的 LoRA 未加载（退化成基座）
  - 若 A 与 B 明显不同 -> 前缀确实是根因
"""
import os
import sys
import json
import time
import warnings

import torch

sys.path.insert(0, "/root/Llamole-main")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

warnings.filterwarnings("ignore")

BASE = "/root/Llamole-main/models/Qwen2-7B-Instruct"
SAVES = "/root/Llamole-main/saves"

PROMPT = """You are an expert in GBM drug development, specializing in novel drug candidates for Glioblastoma Multiforme (GBM).

Please design novel drug candidates with the following characteristics:
Design GBM candidates targeting the EGFR pathway:
- Prioritize targeting EGFRvIII mutants
- Molecular weight: 400-550 Da
- logP: 2-4
- Tyrosine kinase inhibitor structure

Based on the above analysis, generate a novel GBM candidate molecule, including SMILES string and design rationale."""


def build_model(adapter_paths):
    """按 Llamole 的 merge 语义加载：先 merge 第一个，再加载第二个"""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map={"": 0},
        trust_remote_code=True, local_files_only=True,
    )
    merged = []
    for i, p in enumerate(adapter_paths):
        print(f"  loading adapter[{i}]: {os.path.basename(p)}", flush=True)
        model = PeftModel.from_pretrained(model, p)
        model = model.merge_and_unload()
        merged.append(p)
    return model


def gen(model, tok, prompt, n=3, max_new_tokens=160):
    from transformers import GenerationConfig
    cfg = GenerationConfig(
        max_length=len(tok(prompt)["input_ids"]) + max_new_tokens,
        max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9,
        do_sample=True, num_return_sequences=n,
        pad_token_id=tok.pad_token_id,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=cfg)
    outs = []
    for o in out:
        txt = tok.decode(o, skip_special_tokens=True)
        if txt.startswith(prompt):
            txt = txt[len(prompt):]
        outs.append(txt.strip())
    return outs


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True,
                                        padding_side="left", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    llamole = os.path.join(SAVES, "Llamole-Qwen2-7B-Instruct-Adapter")

    cases = [
        ("A: with-graph-models (BROKEN prefix)", [llamole, os.path.join(SAVES, "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models")]),
        ("B: gbm-fixed (prefix normalized)",      [llamole, os.path.join(SAVES, "Llamole-Qwen2-7B-Instruct-Adapter-gbm-fixed")]),
        ("C: llamole base only",                  [llamole]),
    ]

    results = {}
    for label, paths in cases:
        print("\n" + "=" * 74)
        print(f"CASE {label}")
        print("=" * 74, flush=True)
        t0 = time.time()
        model = build_model(paths)
        # 报告实际生效的 lora 状态
        lora_b = [float(p.abs().mean()) for n, p in model.named_parameters() if "lora_B" in n]
        print(f"  lora_B tensors in final model: {len(lora_b)}"
              f"{' (merged into base)' if not lora_b else ''}", flush=True)
        outs = gen(model, tok, PROMPT)
        results[label] = outs
        for i, o in enumerate(outs):
            print(f"\n  --- sample {i} ---")
            print("  " + o[:600].replace("\n", "\n  "))
        print(f"\n  ({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache()

    # 保存结果供比对
    with open("/root/Llamole-main/gbm_project/experiments/p0_adapter_gen_compare.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nsaved -> experiments/p0_adapter_gen_compare.json")


if __name__ == "__main__":
    main()
