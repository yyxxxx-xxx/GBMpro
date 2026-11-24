#!/bin/bash
# M5 启动守卫 v2: 轮询等待**任意**一张 GPU 释放到阈值, 立即在第一张满足的卡上启动
#
# 用法: ./run_m5_when_gpu_free.sh <TARGET> <STEPS> <ADVANTAGE_MODE> <TAG>
#   例: ./run_m5_when_gpu_free.sh EGFR 30 zscore m5_egfr_zscore
#
# 与 v1 的区别:
#   v1  绑定固定 GPU 编号, 该卡不空就一直等
#   v2  扫描所有卡, 哪张先空就用哪张 —— GPU 0/1 提前释放时也能立刻开跑
set -u
cd /root/Llamole-main

TARGET="$1"; STEPS="${2:-30}"; AMODE="${3:-zscore}"; TAG="${4:-m5_${TARGET,,}_${AMODE}}"

NEED_MB=20000          # 训练峰值 ~19.5GB, 留 500MB 余量
POLL=60                # 轮询间隔(秒)
TIMEOUT=14400          # 最长等待 4 小时
EXCLUDE="${EXCLUDE_GPU:-}"   # 排除的 GPU (逗号分隔), 用于避免多组抢同一张卡

OUT="experiments/${TAG}"
LOG="${OUT}.log"

echo "[M5-GUARD] 等待任意 GPU 空闲 >= ${NEED_MB} MiB (排除: ${EXCLUDE:-无})"
echo "[M5-GUARD]   target=${TARGET} steps=${STEPS} advantage_mode=${AMODE}"
START=$(date +%s)

while true; do
    # 枚举所有 GPU, 找第一张空闲够的
    SELECTED=""
    while IFS= read -r line; do
        IDX=$(echo "$line" | cut -d, -f1)
        FREE=$(echo "$line" | cut -d, -f2)
        # 排除列表检查
        if [ -n "$EXCLUDE" ] && echo ",${EXCLUDE}," | grep -q ",${IDX},"; then
            continue
        fi
        if [ "$FREE" -ge "$NEED_MB" ]; then
            SELECTED="$IDX"
            SELECTED_FREE="$FREE"
            break
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if [ -n "$SELECTED" ]; then
        echo "[M5-GUARD] ✅ GPU ${SELECTED} 空闲 ${SELECTED_FREE} MiB (等待 $ELAPSED s), 启动 ${TAG}"
        cd /root/Llamole-main/gbm_project
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        setsid nohup python3 scripts/train_rl.py \
            --target "$TARGET" --algorithm grpo --steps "$STEPS" \
            --advantage_mode "$AMODE" \
            --n_prompts 4 --group_size 16 --prompt_bank_size 8 \
            --train_batch 4 --grad_accum 1 --lr 1e-5 --beta_kl 0.08 \
            --n_workers 12 --save_every 10 --ma_window 5 \
            --device "cuda:${SELECTED}" \
            --out_dir "../${OUT}" \
            > "../${LOG}" 2>&1 < /dev/null &
        echo "[M5-GUARD] PID=$!  日志=../${LOG}"
        disown
        exit 0
    fi

    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "[M5-GUARD] ❌ 等待超时 ${TIMEOUT}s, 无 GPU 达到 ${NEED_MB} MiB, 放弃启动"
        exit 1
    fi

    echo "[M5-GUARD] 无可用 GPU (需 ${NEED_MB} MiB), 继续等待 ($ELAPSED s)"
    sleep "$POLL"
done
