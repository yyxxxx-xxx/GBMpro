#!/bin/bash
# M5 评估流水线: 等 GPU 上的训练完全结束后, 启动评估
#
# 用法: ./run_m5_eval_pipeline.sh
#
# 逻辑:
#   1) 轮询等待 train_rl.py 全部退出
#   2) GPU0 跑 EGFR 两组 (zscore -> pairwise), GPU1 跑 VEGFR 两组
#   3) 每组结束后做显著性检验
set -u
cd /root/Llamole-main

echo "[PIPE] 等待训练进程全部退出 ..."
while ps -ef | grep -q "[t]rain_rl.py"; do
    sleep 30
done
echo "[PIPE] ✅ 训练全部结束, 开始评估"

# GPU0: EGFR 两组
(
    EVAL_GPU=0 ./run_m5_eval_all.sh EGFR:zscore EGFR:pairwise \
        > experiments/m5_eval_egfr.log 2>&1
    echo "[PIPE] ✅ EGFR 评估完成" >> experiments/m5_eval_status.log
) &

# GPU1: VEGFR 两组
(
    EVAL_GPU=1 ./run_m5_eval_all.sh VEGFR:zscore VEGFR:pairwise \
        > experiments/m5_eval_vegfr.log 2>&1
    echo "[PIPE] ✅ VEGFR 评估完成" >> experiments/m5_eval_status.log
) &

wait
echo "[PIPE] ✅ 全部评估完成"
