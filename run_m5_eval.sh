#!/bin/bash
# M5 多靶点评估编排
# ==================
# 对每个靶点依次执行:
#   1) SFT 基线评估 (独立 prompt 集 + 固定 seed=1234 + 统一重打分)
#   2) RL checkpoint 评估
#   3) 显著性检验 (Mann-Whitney / Cliff's d / 配对检验 / Cohen's d)
#
# ⚠️ SFT 与 RL 必须用**同一个 seed** —— 否则采到的分子不同, 对比不公平。
#    这一点在 eval_rl_checkpoint.py 里由 --seed 保证。
#
# 用法:
#   ./run_m5_eval.sh EGFR VEGFR
set -u
cd /root/Llamole-main/gbm_project
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for TARGET in "$@"; do
    TGT_LC=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')
    TRAIN_DIR="../experiments/m5_${TGT_LC}_kl008"
    EVAL_DIR="../experiments/eval_m5_${TGT_LC}_kl008"

    echo ""
    echo "================================================================"
    echo "[M5-EVAL] target = $TARGET"
    echo "================================================================"

    if [ ! -f "${TRAIN_DIR}/rl_lora_final.safetensors" ]; then
        echo "[M5-EVAL] ❌ 找不到 ${TRAIN_DIR}/rl_lora_final.safetensors, 跳过"
        continue
    fi

    mkdir -p "${EVAL_DIR}"

    # --- 1) RL checkpoint ---
    echo "[M5-EVAL] 评估 RL checkpoint ..."
    python3 scripts/eval_rl_checkpoint.py \
        --target "$TARGET" \
        --rl_adapter "${TRAIN_DIR}/rl_lora_final.safetensors" \
        --n_prompts 8 --n_per_prompt 50 --temperature 0.8 --seed 1234 \
        --n_workers 12 \
        --out_dir "${EVAL_DIR}"

    # --- 2) SFT 基线 ---
    echo "[M5-EVAL] 评估 SFT 基线 ..."
    python3 scripts/eval_rl_checkpoint.py \
        --target "$TARGET" \
        --baseline_only \
        --n_prompts 8 --n_per_prompt 50 --temperature 0.8 --seed 1234 \
        --n_workers 12 \
        --out_dir "${EVAL_DIR}"

    # --- 3) 显著性检验 ---
    echo "[M5-EVAL] 显著性检验 ..."
    python3 scripts/compare_eval.py "${EVAL_DIR}" || echo "[M5-EVAL] ⚠️ 检验失败"

    # --- 4) 导出 CSV ---
    echo "[M5-EVAL] 导出 CSV ..."
    python3 scripts/export_molecules_csv.py "${EVAL_DIR}" --include_baseline \
        || echo "[M5-EVAL] ⚠️ CSV 导出失败"

    echo "[M5-EVAL] ✅ $TARGET 完成 -> ${EVAL_DIR}"
done

echo ""
echo "[M5-EVAL] 全部完成"
