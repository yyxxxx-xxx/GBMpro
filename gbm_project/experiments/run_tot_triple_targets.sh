#!/bin/bash
# EGFR / VEGF_VEGFR / IDH1_IDH2 各生成 200 个分子
# EGFR       -> GPU 1
# VEGF_VEGFR -> GPU 2
# IDH1_IDH2  -> GPU 3
#
# 用法:
#   cd /root/Llamole-main
#   bash gbm_project/experiments/run_tot_triple_targets.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT = gbm_project/experiments 的 parent
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "PROJECT_ROOT=$PROJECT_ROOT"

# 输出目录（时间戳统一，三个靶点共享）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/gbm_project/experiments/tot_batch_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"
echo "[run_tot_triple_targets] Output dir: $OUTPUT_DIR"

COMMON_ARGS="--num_molecules 200 --k 4 --b 3 --prompts_config_path gbm_project/configs/english_gbm_prompts.yaml --max_attempts 300 --max_no_new_streak 0"

run_target() {
    local target=$1
    local gpu_id=$2
    local log_file="$OUTPUT_DIR/${target}.log"
    echo "[run_tot_triple_targets] Starting $target on GPU $gpu_id -> $log_file"
    nohup python3 "$PROJECT_ROOT/gbm_project/experiments/generate_tot_batch_candidates.py" \
        --targets "$target" \
        $COMMON_ARGS \
        --gpu_id "$gpu_id" \
        > "$log_file" 2>&1 &
    echo "[run_tot_triple_targets] $target launched (PID=$!, GPU=$gpu_id)"
}

# 启动三个任务（后台并行）
run_target "EGFR"        1
run_target "VEGF_VEGFR"  2
run_target "IDH1_IDH2"   3

echo ""
echo "[run_tot_triple_targets] All 3 jobs launched in background."
echo "[run_tot_triple_targets] Output dir: $OUTPUT_DIR"
echo "[run_tot_triple_targets] Logs:"
echo "  EGFR:        $OUTPUT_DIR/EGFR.log"
echo "  VEGF_VEGFR:  $OUTPUT_DIR/VEGF_VEGFR.log"
echo "  IDH1_IDH2:   $OUTPUT_DIR/IDH1_IDH2.log"
echo ""
echo "Monitor:  tail -f $OUTPUT_DIR/<target>.log"
echo "Check GPU: nvidia-smi"
echo "Check PIDs: ps aux | grep generate_tot_batch_candidates"
