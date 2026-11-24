#!/bin/bash
# M5 启动: 在指定 GPU 上跑一个实验组
#
# 用法: ./run_m5_one.sh <TARGET> <GPU> <STEPS> <ADVANTAGE_MODE> <TAG>
#   例: ./run_m5_one.sh EGFR 0 30 zscore        m5_egfr_zscore
#       ./run_m5_one.sh EGFR 0 30 pairwise_scaled m5_egfr_pairwise
set -u
cd /root/Llamole-main

TARGET="$1"; GPU="$2"; STEPS="${3:-30}"; AMODE="${4:-zscore}"; TAG="${5:-m5_${TARGET,,}_${AMODE}}"

OUT="experiments/${TAG}"
LOG="${OUT}.log"

echo "[M5] 启动 ${TAG}"
echo "[M5]   target=${TARGET} gpu=${GPU} steps=${STEPS} advantage_mode=${AMODE}"

cd /root/Llamole-main/gbm_project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup python3 scripts/train_rl.py \
    --target "$TARGET" --algorithm grpo --steps "$STEPS" \
    --advantage_mode "$AMODE" \
    --n_prompts 4 --group_size 16 --prompt_bank_size 8 \
    --train_batch 4 --grad_accum 1 --lr 1e-5 --beta_kl 0.08 \
    --n_workers 12 --save_every 10 --ma_window 5 \
    --device "cuda:${GPU}" \
    --out_dir "../${OUT}" \
    > "../${LOG}" 2>&1 < /dev/null &

PID=$!
echo "[M5] PID=${PID}  日志=../${LOG}  产物=../${OUT}"
disown
exit 0
