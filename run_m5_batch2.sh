#!/bin/bash
# M5 第二批: 等当前 zscore 组跑完 + 对应 GPU 释放后, 启动 pairwise_scaled 组
#
# 逻辑: 监视 m5_egfr_zscore / m5_vegfr_zscore 的训练进程,
#       一旦某个退出且对应 GPU 空闲足够, 就在该 GPU 上启动对应的 pairwise 组
set -u
cd /root/Llamole-main

NEED_MB=20000
POLL=60
TIMEOUT=14400

# 靶点 -> GPU 映射 (zscore 组用的卡)
declare -A GPU_OF=( [EGFR]=0 [VEGFR]=1 )

echo "[M5-BATCH2] 等待第一批评测完成 (EGFR@GPU0, VEGFR@GPU1) ..."

while true; do
    ALL_DONE=1
    for T in EGFR VEGFR; do
        G=${GPU_OF[$T]}
        PIDFILE=$(ls experiments/m5_${T,,}_zscore/train_log.jsonl 2>/dev/null)
        # 判断是否训练结束: 进程不在 + 日志存在 + 行数达到目标
        RUNNING=$(ps -ef | grep "[t]rain_rl.py" | grep -- "--target $T" | grep -c "zscore")
        if [ "$RUNNING" -gt 0 ]; then
            ALL_DONE=0
            continue
        fi
        # 进程已结束 -> 尝试启动 pairwise
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G" | head -1)
        if [ "$FREE" -ge "$NEED_MB" ]; then
            echo "[M5-BATCH2] ✅ $T 的 zscore 组已完成, GPU${G} 空闲 ${FREE} MiB, 启动 pairwise_scaled"
            ./run_m5_one.sh "$T" "$G" 30 pairwise_scaled "m5_${T,,}_pairwise"
            sleep 120   # 避免两个组同时抢卡
        else
            ALL_DONE=0
        fi
    done

    if [ "$ALL_DONE" -eq 1 ]; then
        echo "[M5-BATCH2] ✅ 全部四组已启动完毕"
        exit 0
    fi
    sleep "$POLL"
done
