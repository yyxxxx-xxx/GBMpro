#!/usr/bin/env python3
"""
RL checkpoint 独立评估
======================

⚠️ 为什么必须有独立评估:
   训练日志里的 reward 是**on-policy**的 —— 模型在训练时看到的分子,
   采样温度 0.8, 且组内相对比较。奖励 hacking (reward hacking) 恰恰
   表现为"训练 reward 涨了但独立采样没变好"。
   所以必须: 用**固定的独立 prompt 集**采样, 统一重打分, 与 SFT 基线对比。

用法:
    # 评估 RL checkpoint
    python3 scripts/eval_rl_checkpoint.py \
        --rl_adapter ../experiments/rl_m2_grpo/rl_lora_final.safetensors \
        --out_dir ../experiments/rl_m2_grpo/eval

    # 只跑 SFT 基线 (对照组)
    python3 scripts/eval_rl_checkpoint.py \
        --baseline_only --out_dir ../experiments/rl_m2_grpo/eval_baseline
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

GBM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GBM / "src"))
sys.path.insert(0, str(GBM / "scripts"))

import numpy as np
import torch


def build_model(rl_adapter=None, sft_adapter=None, base_model=None):
    """构建 (agent, prior) —— 有 rl_adapter 就加载 RL LoRA, 否则只有 SFT。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cuda:0")
    model.config.use_cache = True

    llamole = "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter"
    if Path(llamole).exists():
        model = PeftModel.from_pretrained(model, llamole)
        model = model.merge_and_unload()

    sft = sft_adapter or \
        "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-active"
    model = PeftModel.from_pretrained(model, sft, is_trainable=True)
    model = model.merge_and_unload()

    if rl_adapter:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"],
                         bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg, adapter_name="rl")
        from safetensors.torch import load_file
        sd = load_file(str(rl_adapter))
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, sd, adapter_name="rl")
        print(f"[eval] 已加载 RL adapter: {rl_adapter}", flush=True)

    return model, tokenizer


def sample_molecules(model, tokenizer, prompts, n_per_prompt,
                     temperature=0.8, top_p=0.92, max_new_tokens=512, seed=0,
                     chunk=8):
    """在固定 prompt 集上采样。

    ⚠️ 必须分批 (chunk): 一次生成 25 条序列的 logits 会 OOM
       (实测: 25 x 650 x 152k x 4B = 8.38 GB 分配失败)。
       每批 8 条是实测安全值。
    """
    from rl.rollout import RolloutEngine, RolloutConfig

    engine = RolloutEngine(model, tokenizer, RolloutConfig(
        temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens))
    out = []
    for pi, p in enumerate(prompts):
        done = 0
        k = 0
        while done < n_per_prompt:
            n = min(chunk, n_per_prompt - done)
            samples = engine.generate([p], n_groups=1, group_size=n,
                                      seed=seed + pi * 1000 + k)
            for s in samples:
                if s.valid:
                    out.append({"prompt_id": pi, "smiles": s.smiles})
            done += n
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_adapter", default=None,
                    help="RL LoRA safetensors (不给就是只跑 SFT 基线)")
    ap.add_argument("--baseline_only", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=8)
    ap.add_argument("--n_per_prompt", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1234,
                    help="⚠️ SFT 与 RL 必须用**同一个 seed**, "
                         "否则采到的分子不同, 对比不公平")
    ap.add_argument("--n_workers", type=int, default=15)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", default="IDH1",
                    help="靶点: IDH1 / EGFR / VEGFR "
                         "(受体已就绪, 见 configs/receptors.yaml)")
    ap.add_argument("--base_model",
                    default="/root/Llamole-main/models/Qwen2-7B-Instruct")
    ap.add_argument("--sft_adapter",
                    default="/root/Llamole-main/saves/"
                            "Llamole-Qwen2-7B-Instruct-Adapter-gbm-active")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 固定的独立 prompt 集 (与训练用的 bank 相同, 但采样是新的) ---
    from rl.prompt_bank import PromptBank, format_full_prompt
    bank = PromptBank.build(args.target, n_prompts=args.n_prompts)
    prompts = [format_full_prompt(p) for p in bank.prompts]
    print(f"[eval] 独立评估 prompt 集: {len(prompts)} 条", flush=True)

    model, tok = build_model(
        rl_adapter=None if args.baseline_only else args.rl_adapter,
        sft_adapter=args.sft_adapter,
        base_model=args.base_model)

    print(f"[eval] 采样 {args.n_per_prompt} 个/提示 x {len(prompts)} 提示 "
          f"(seed={args.seed}) ...", flush=True)
    mols = sample_molecules(model, tok, prompts, args.n_per_prompt,
                            temperature=args.temperature, seed=args.seed)
    print(f"[eval] 采样到 {len(mols)} 个有效分子", flush=True)

    tag = "baseline_sft" if args.baseline_only else "rl"
    smi_path = out_dir / f"{tag}_smiles.json"
    json.dump(mols, open(smi_path, "w"), indent=2, ensure_ascii=False)

    # --- 统一重打分 (与训练同一个 reward 服务) ---
    from rl.reward_service import RewardService, RewardConfig
    svc = RewardService(RewardConfig(
        target=args.target, n_workers=args.n_workers,
        cache_path=out_dir / f"{tag}_reward_cache.jsonl"))
    try:
        recs = svc.score_batch([m["smiles"] for m in mols])
    finally:
        svc.close()

    rew = np.array([r.reward for r in recs])
    vina = np.array([r.vina for r in recs if r.vina is not None]) \
        if any(r.vina is not None for r in recs) else np.array([])

    stats = {
        "tag": tag,
        "target": args.target,
        "n_sampled": len(mols),
        "n_scored": len(recs),
        "reward_mean": float(rew.mean()) if len(rew) else 0.0,
        "reward_std": float(rew.std()) if len(rew) else 0.0,
        "reward_median": float(np.median(rew)) if len(rew) else 0.0,
        "reward_max": float(rew.max()) if len(rew) else 0.0,
        "reward_p90": float(np.percentile(rew, 90)) if len(rew) else 0.0,
        "vina_mean": float(vina.mean()) if len(vina) else None,
        "vina_best": float(vina.min()) if len(vina) else None,  # vina 越负越好
        "n_unique": len(set(m["smiles"] for m in mols)),
        "seed": args.seed,
    }
    (out_dir / f"{tag}_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"[{tag}] n={stats['n_scored']} 唯一 {stats['n_unique']}")
    print(f"  reward: mean {stats['reward_mean']:.4f}  "
          f"median {stats['reward_median']:.4f}  max {stats['reward_max']:.4f}")
    if stats["vina_mean"] is not None:
        print(f"  vina:   mean {stats['vina_mean']:.3f}  best {stats['vina_best']:.3f}")
    print(f"{'='*60}\n")
    print(f"结果: {out_dir}")


if __name__ == "__main__":
    main()
