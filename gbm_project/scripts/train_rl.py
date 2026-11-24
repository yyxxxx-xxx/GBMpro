#!/usr/bin/env python3
"""
RL 训练 CLI
===========

用法:
    python3 scripts/train_rl.py \
        --target IDH1 \
        --algorithm grpo \
        --steps 60 \
        --n_prompts 4 --group_size 8 \
        --lr 1e-6 --beta_kl 0.02 \
        --out_dir final_results/comparative/Ours_GRPO_IDH1 \
        --resume

⚠️ 环境: 必须用 **base env** (`/root/anaconda3`)。
   实测 base env 是唯一同时具备 CUDA(4卡) + vina + meeko + admet_ai
   的环境, 且与反馈闭环的 reward 同源 (差 3.4e-9)。
   查看 docs/RL_INTEGRATION_DESIGN.md §5.3 / 附录 B。

⚠️ 环境变量 (防止 admet-ai 抢 GPU 显存):
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GBM_ROOT = PROJECT_ROOT / "gbm_project"

if str(GBM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(GBM_ROOT / "src"))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GBM RL 训练 (GRPO / DAP-Lite)")
    ap.add_argument("--target", default="IDH1")
    ap.add_argument("--algorithm", default="grpo", choices=["grpo", "dap"])
    ap.add_argument("--steps", type=int, default=60)

    # 模型
    ap.add_argument("--base_model", default="/root/Llamole-main/models/Qwen2-7B-Instruct")
    ap.add_argument("--llamole_adapter",
                    default="/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter")
    ap.add_argument("--sft_adapter",
                    default="/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-active")

    # RL 超参
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta_kl", type=float, default=0.02)
    ap.add_argument("--sigma0", type=float, default=8.0)
    ap.add_argument("--train_batch", type=int, default=8,
                    help="⚠️ 实测 16 会 OOM (峰值 >23.7GB)")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--kl_estimator", default="k3", choices=["k1", "k2", "k3"])

    # Rollout (默认基于实测有效率, 见 rl/rollout.py 的表格)
    ap.add_argument("--n_prompts", type=int, default=4,
                    help="每步从 bank 里抽几个 prompt")
    ap.add_argument("--prompt_bank_size", type=int, default=8,
                    help="prompt bank 总条数 (0=用 n_prompts)。"
                         "唯一上限 8 (4 cell states × 2 种 metric 字符串)。"
                         "bank 大于每步采样数才能跨 prompt 泛化")
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--min_heavy_atoms", type=int, default=12)
    ap.add_argument("--save_every", type=int, default=5,
                    help="每 N 步存一次 latest checkpoint (0=只在结束时存)")
    ap.add_argument("--ma_window", type=int, default=5,
                    help="reward 滑动平均窗口 (判断收敛用, 单步噪声太大)")

    # 多样性 (与 REINVENT 对齐)
    ap.add_argument("--bucket_size", type=int, default=10)
    ap.add_argument("--minscore", type=float, default=0.4)
    ap.add_argument("--no_diversity_filter", action="store_true")

    # RL LoRA
    ap.add_argument("--rl_lora_r", type=int, default=16)
    ap.add_argument("--rl_lora_alpha", type=int, default=32)
    ap.add_argument("--rl_lora_dropout", type=float, default=0.05)

    # 其它
    ap.add_argument("--n_workers", type=int, default=15,
                    help="Vina 进程池大小 (60 核 / 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--device", default="cuda:0",
                    help="训练用 GPU (如 cuda:1)。4 卡可并行跑消融。")
    return ap


def main():
    args = build_argparser().parse_args()

    # ★ 显存碎片缓解 (实测 B=16 OOM 的对策)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ★ 多卡并行: 把目标物理卡映射成进程内唯一的 cuda:0。
    #   为什么必须这么做: PEFT 的 `load_peft_weights` 内部**硬编码**用
    #   "cuda:0" 加载 adapter 权重 —— 即使模型建在 cuda:2, 加载权重时
    #   仍会往 GPU 0 分配 1GB, 导致多卡并行时 OOM (实测 4 卡并行崩 2 个)。
    #   用 CUDA_VISIBLE_DEVICES 隔离后, 进程内 "cuda:0" 就是目标物理卡。
    if args.device.startswith("cuda:") and ":" in args.device:
        _idx = args.device.split(":")[1]
        if _idx.isdigit() and _idx != "0":
            os.environ["CUDA_VISIBLE_DEVICES"] = _idx
            print(f"[device] CUDA_VISIBLE_DEVICES={_idx} "
                  f"(进程内 {args.device} == 物理卡 {_idx})", flush=True)
            args.device = "cuda:0"
    elif args.device == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    out_dir = Path(args.out_dir or
                   f"final_results/comparative/Ours_{args.algorithm.upper()}_{args.target}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from rl.trainer import TrainerConfig, RLTrainer

    cfg = TrainerConfig(
        base_model=args.base_model,
        llamole_adapter=args.llamole_adapter,
        sft_adapter=args.sft_adapter,
        target=args.target,
        algorithm=args.algorithm,
        lr=args.lr,
        beta_kl=args.beta_kl,
        sigma0=args.sigma0,
        max_grad_norm=args.max_grad_norm,
        train_batch=args.train_batch,
        grad_accum=args.grad_accum,
        kl_estimator=args.kl_estimator,
        n_prompts=args.n_prompts,
        prompt_bank_size=args.prompt_bank_size,
        group_size=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        min_heavy_atoms=args.min_heavy_atoms,
        save_every=args.save_every,
        ma_window=args.ma_window,
        diversity_bucket_size=args.bucket_size,
        diversity_minscore=args.minscore,
        enable_diversity_filter=not args.no_diversity_filter,
        rl_lora_r=args.rl_lora_r,
        rl_lora_alpha=args.rl_lora_alpha,
        rl_lora_dropout=args.rl_lora_dropout,
        seed=args.seed,
        n_workers=args.n_workers,
    )

    # 把配置存下来, 便于复现
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    trainer = RLTrainer(cfg, out_dir, device=args.device)
    if args.resume:
        trainer.load_checkpoint("latest")
    trainer.train(n_steps=args.steps, log_every=args.log_every)


if __name__ == "__main__":
    main()
