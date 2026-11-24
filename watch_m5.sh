#!/bin/bash
# M5 监控: 每 2 分钟汇报一次四组进度
cd /root/Llamole-main
while true; do
    echo "===== $(date '+%H:%M:%S') ====="
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | \
        awk -F', ' '{printf "GPU%s used=%s free=%s\n",$1,$2,$3}'
    for d in experiments/m5_*/; do
        T=$(basename $d)
        LOG="$d/train_log.jsonl"
        if [ -f "$LOG" ]; then
            N=$(wc -l < "$LOG")
            LAST=$(tail -1 "$LOG" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(f\"step={d.get('step')} R={d.get('reward_mean',0):.4f} KL={d.get('kl',0):.4f} skip={d.get('n_skipped_groups',0)}\")
except: print('?')
")
            printf "  %-24s steps=%-4s %s\n" "$T" "$N" "$LAST"
        else
            printf "  %-24s (未开始)\n" "$T"
        fi
    done
    echo ""
    sleep 120
done
