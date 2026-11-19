#!/usr/bin/env python3
"""
GBM LoRA 微调 —— 真实分子版
============================

与 train_gbm_lora_correct.py / train_gbm_lora_with_graph_models.py 的区别：

1. **数据集**：用 PubChem 解析的真实分子（旧集 982 药名 -> 只有 32 个唯一 SMILES）
2. **prompt 形态对齐**：训练 instruction 复用推理时的领域 prompt + 目标知识 + CoT
3. **response-only loss**：label 只落在 "Response: SMILES: xxx" 上
4. **Llamole 适配不被丢弃**：先 merge_and_unload 再注入新 LoRA
5. **modules_to_save**：保留 embed_tokens / lm_head（化学 token 表示）
6. **保存后校验**：规范化 key 前缀 + 校验 lora_B 非零

用法：
    python scripts/train_gbm_lora_real.py \
        --dataset data/lora_datasets/real_qa_train.json \
        --output_dir /root/Llamole-main/saves/...-gbm-real
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gbm_lora_finetuner import run_gbm_lora_finetuning


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str,
                    default="/root/Llamole-main/models/Qwen2-7B-Instruct")
    ap.add_argument("--llamole_adapter", type=str,
                    default="/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter")
    ap.add_argument("--dataset", type=str,
                    default="/root/Llamole-main/gbm_project/data/lora_datasets/real_qa_train.json")
    ap.add_argument("--output_dir", type=str,
                    default="/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-real")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    args = ap.parse_args()

    print("=" * 66)
    print("GBM LoRA 训练（真实分子数据 + response-only loss）")
    print("=" * 66)
    for name, p in [("Base model", args.base_model),
                    ("Llamole adapter", args.llamole_adapter),
                    ("Dataset", args.dataset)]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found: {p}")
        print(f"✅ {name}: {p}")
    print(f"📁 Output: {args.output_dir}")

    lora_config = {
        "r": args.r,
        "lora_alpha": args.r * 2,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        # 覆盖注意力 + MLP，与 Llamole 的 target_modules 一致
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        # modules_to_save 故意不开。
        # 开了会让 embed_tokens(152064x3584) + lm_head(152064x3584) = 1.09B 参数可训练
        # （占全模型 12.6%），在 ~1700 条样本上极易过拟合。
        # 实测（scripts/verify_special_token_preservation.py）确认
        # merge_and_unload 已把 Llamole 的 embed/lm_head 完整吸收进基座
        # （特殊 token 行 max|diff| = 0.0），所以不开也不会丢化学 token 表示。
        "modules_to_save": None,
        # 先 merge Llamole 再注入，避免化学适配被丢弃
        "merge_llamole_before_lora": True,
        # 梯度只落在 response
        "response_only_loss": True,
        "max_length": 768,
        # 显存保护：7.6B fp16 + 764token prompt 不开梯度检查点必 OOM
        "gradient_checkpointing": True,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "gradient_accumulation_steps": args.grad_accum,
        "save_steps": 100,
        "logging_steps": 20,
        "fp16": True,
    }

    print(f"\nLoRA config: r={args.r} alpha={args.r*2} lr={args.lr} "
          f"epochs={args.epochs}")
    print(f"  target_modules: {lora_config['target_modules']}")
    print(f"  modules_to_save: {lora_config['modules_to_save']} "
          f"(merge 已吸收 embed/lm_head，避免 1.09B 参数过拟合)")
    print(f"  response_only_loss: {lora_config['response_only_loss']}")

    os.environ.setdefault("SKIP_KBIT_PREP", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    run_gbm_lora_finetuning(
        base_model_path=args.base_model,
        llamole_adapter_path=args.llamole_adapter,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        lora_config=lora_config,
    )

    print("\n" + "=" * 66)
    print(f"训练完成 -> {args.output_dir}")
    print("=" * 66)


if __name__ == "__main__":
    main()
