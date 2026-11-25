#!/bin/bash
# M5 第二批: 等当前 zscore 组跑完 + 对应 GPU 释放后, 启动 pairwise_scaled 组
#
# ⚠️ **bug 修复**: v1 用 "进程不在" 判断 zscore 组是否完成, 但 zscore 组
#    退出后每一轮循环都会重复触发启动, 导致 pairwise 组被启动多次
#    (实测跑了 319/399 步而非 30 步, 因为 --resume 会累加步数)。
#    修复: 用**标记文件**记录"该组已启动过", 保证每组只启动一次。
set -u
cd /root/Llamole-main

NEED_MB=20000
POLL=60
TIMEOUT=14400
MARK_DIR=".m5_launched"

# 靶点 -> GPU 映射 (zscore 组用的卡)
declare -A GPU_OF=( [EGFR]=0 [VEGFR]=1 )

mkdir -p "$MARK_DIR"

echo "[M5-BATCH2] 等待第一批评测完成 (EGFR@GPU0, VEGFR@GPU1) ..."

while true; do
    ALL_DONE=1
    for T in EGFR VEGFR; do
        G=${GPU_OF[$T]}
        MARK="$MARK_DIR/$T"
        # 已启动过 -> 跳过 (关键修复: 保证只启动一次)
        if [ -f "$MARK" ]; then
            continue
        fi

        RUNNING=$(ps -ef | grep "[t]rain_rl.py" | grep -- "--target $T" | grep -c "zscore")
        if [ "$RUNNING" -gt 0 ]; then
            ALL_DONE=0
            continue
        fi

        # zscore 进程已结束 -> 等 GPU 释放后启动 pairwise
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G" | head -1)
        if [ "$FREE" -ge "$NEED_MB" ]; then
            echo "[M5-BATCH2] ✅ $T 的 zscore 组已完成, GPU${G} 空闲 ${FREE} MiB, 启动 pairwise_scaled"
            touch "$MARK"          # ← 先打标记, 再启动, 防止竞态重复启动
            ./run_m5_one.sh "$T" "$G" 30 pairwise_scaled "m5_${T,,}_pairwise"
            sleep 120
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
