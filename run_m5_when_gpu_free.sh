#!/bin/bash
# M5 启动守卫: 轮询等待显存充足的 GPU, 一旦满足条件立即启动训练
# 用法: ./run_m5_when_gpu_free.sh EGFR 0 30
set -u
cd /root/Llamole-main

TARGET="$1"; GPU="$2"; STEPS="${3:-30}"
NEED_MB=20000          # 训练峰值 ~19.5GB, 留 500MB 余量
POLL=60                # 轮询间隔(秒)
TIMEOUT=14400          # 最长等待 4 小时

TGT_LC=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')
OUT="experiments/m5_${TGT_LC}_kl008"
LOG="${OUT}.log"

echo "[M5-GUARD] 等待 GPU ${GPU} 释放 >= ${NEED_MB} MiB ... (每 ${POLL}s 轮询, 上限 ${TIMEOUT}s)"
START=$(date +%s)

while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1)
    FREE=${FREE:-0}
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if [ "$FREE" -ge "$NEED_MB" ]; then
        echo "[M5-GUARD] ✅ GPU ${GPU} 空闲 ${FREE} MiB (等待 $ELAPSED s), 启动训练"
        cd /root/Llamole-main/gbm_project
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        setsid nohup python3 scripts/train_rl.py \
            --target "$TARGET" --algorithm grpo --steps "$STEPS" \
            --n_prompts 4 --group_size 16 --prompt_bank_size 8 \
            --train_batch 4 --grad_accum 1 --lr 1e-5 --beta_kl 0.08 \
            --n_workers 12 --save_every 10 --ma_window 5 \
            --device "cuda:${GPU}" \
            --out_dir "../${OUT}" \
            > "../${LOG}" 2>&1 < /dev/null &
        echo "[M5-GUARD] PID=$!  日志=../${LOG}"
        disown
        exit 0
    fi

    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "[M5-GUARD] ❌ 等待超时 ${TIMEOUT}s, GPU ${GPU} 仍只有 ${FREE} MiB 空闲, 放弃启动"
        exit 1
    fi

    echo "[M5-GUARD] GPU ${GPU} 空闲 ${FREE} MiB < ${NEED_MB} MiB, 继续等待 ($ELAPSED s)"
    sleep "$POLL"
done
