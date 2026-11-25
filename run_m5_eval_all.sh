#!/bin/bash
# M5 四组评估编排 (通用版)
# ==========================
# 对每个 (靶点, advantage_mode) 组合执行:
#   1) SFT 基线评估 (独立 prompt 集 + 固定 seed=1234 + 统一重打分)
#   2) RL checkpoint 评估
#
# ⚠️ SFT 与 RL 必须用**同一个 seed** —— 否则采到的分子不同, 对比不公平。
#    eval_rl_checkpoint.py 的 --seed 保证这一点。
#
# ⚠️ 基线只需跑一次 (SFT 与靶点/advantage_mode 无关), 后续复用。
#
# 用法:
#   ./run_m5_eval_all.sh EGFR:zscore EGFR:pairwise VEGFR:zscore VEGFR:pairwise
#   ./run_m5_eval_all.sh all          # 跑全部四组
set -u
cd /root/Llamole-main/gbm_project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_PROMPTS=8
N_PER_PROMPT=50
SEED=1234
GPU="${EVAL_GPU:-0}"

if [ "$#" -eq 0 ]; then
    echo "用法: $0 <TARGET:MODE> ... | all"
    exit 1
fi

if [ "$1" = "all" ]; then
    set -- EGFR:zscore EGFR:pairwise VEGFR:zscore VEGFR:pairwise
fi

for SPEC in "$@"; do
    TARGET="${SPEC%%:*}"
    MODE="${SPEC##*:}"
    TGT_LC=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')
    TRAIN_DIR="../experiments/m5_${TGT_LC}_${MODE}"
    EVAL_DIR="../experiments/eval_m5_${TGT_LC}_${MODE}"

    echo ""
    echo "================================================================"
    echo "[M5-EVAL] target=$TARGET  advantage_mode=$MODE"
    echo "================================================================"

    if [ ! -f "${TRAIN_DIR}/rl_lora_final.safetensors" ]; then
        echo "[M5-EVAL] ❌ 找不到 ${TRAIN_DIR}/rl_lora_final.safetensors, 跳过"
        continue
    fi

    # --- 1. SFT 基线 (只跑一次, 后续复用) ---
    if [ ! -f "${EVAL_DIR}/baseline_sft_stats.json" ]; then
        echo "[M5-EVAL] ▶ 采样 SFT 基线 (${TGT_LC}) ..."
        python3 scripts/eval_rl_checkpoint.py \
            --target "$TARGET" --baseline_only \
            --n_prompts $N_PROMPTS --n_per_prompt $N_PER_PROMPT \
            --seed $SEED --n_workers 12 \
            --device "cuda:${GPU}" \
            --out_dir "$EVAL_DIR" 2>&1 | tail -20
    else
        echo "[M5-EVAL] ✓ 复用已有基线 ${EVAL_DIR}/baseline_sft_stats.json"
    fi

    # --- 2. RL checkpoint ---
    echo "[M5-EVAL] ▶ 采样 RL checkpoint (${TGT_LC}, ${MODE}) ..."
    python3 scripts/eval_rl_checkpoint.py \
        --target "$TARGET" \
        --rl_adapter "${TRAIN_DIR}/rl_lora_final.safetensors" \
        --n_prompts $N_PROMPTS --n_per_prompt $N_PER_PROMPT \
        --seed $SEED --n_workers 12 \
        --device "cuda:${GPU}" \
        --out_dir "$EVAL_DIR" 2>&1 | tail -20

    echo "[M5-EVAL] ✅ 完成 $EVAL_DIR"
done

echo ""
echo "[M5-EVAL] 全部完成"
