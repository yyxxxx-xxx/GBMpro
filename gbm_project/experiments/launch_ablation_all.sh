#!/bin/bash
# =============================================================================
# GBM Ablation Experiment Launcher
# GPU 1 → Ablation 1b (CoT)
# GPU 2 → Ablation 2 (No Feedback) then Ablation 3 (No Physical Eval)
# GPU 3 → Ablation 4 (No Domain Knowledge) then Ablation 5 (No GBM LoRA)
# =============================================================================

set -e

EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGETS="EGFR IDH1_IDH2 VEGF_VEGFR"
N_MOL="50"
N_ATTEMPT="300"

echo "=== Experiment dir: $EXPDIR ==="
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# ── GPU 1: Ablation 1b (CoT) ───────────────────────────────────────────────
(
    SCRIPT="$EXPDIR/generate_ablation_1_cot.py"
    echo ">>> GPU 1 | ablation_1b_chain_of_thought | mode=cot"
    echo ">>> CMD: python $SCRIPT --gpu_id 1 --targets $TARGETS --num_molecules $N_MOL --max_attempts $N_ATTEMPT --mode cot"
    cd "$EXPDIR" && python "$SCRIPT" \
        --gpu_id 1 \
        --targets $TARGETS \
        --num_molecules $N_MOL \
        --max_attempts $N_ATTEMPT \
        --mode cot
    echo ">>> GPU 1 DONE: ablation_1b_chain_of_thought"
) &
PID1=$!

# ── GPU 2: Ablation 2 then 3 ───────────────────────────────────────────────
(
    echo ">>> GPU 2 | ablation_2_no_feedback"
    SCRIPT="$EXPDIR/generate_ablation_2_no_feedback.py"
    cd "$EXPDIR" && python "$SCRIPT" \
        --gpu_id 2 \
        --targets $TARGETS \
        --num_molecules $N_MOL \
        --max_attempts $N_ATTEMPT
    echo ">>> GPU 2 DONE: ablation_2_no_feedback"

    echo ">>> GPU 2 | ablation_3_no_physical_eval"
    SCRIPT="$EXPDIR/generate_ablation_3_no_physical_eval.py"
    cd "$EXPDIR" && python "$SCRIPT" \
        --gpu_id 2 \
        --targets $TARGETS \
        --num_molecules $N_MOL \
        --max_attempts $N_ATTEMPT
    echo ">>> GPU 2 DONE: ablation_3_no_physical_eval"
) &
PID2=$!

# ── GPU 3: Ablation 4 then 5 ───────────────────────────────────────────────
(
    echo ">>> GPU 3 | ablation_4_no_domain_knowledge"
    SCRIPT="$EXPDIR/generate_ablation_4_no_domain_knowledge.py"
    cd "$EXPDIR" && python "$SCRIPT" \
        --gpu_id 3 \
        --targets $TARGETS \
        --num_molecules $N_MOL \
        --max_attempts $N_ATTEMPT
    echo ">>> GPU 3 DONE: ablation_4_no_domain_knowledge"

    echo ">>> GPU 3 | ablation_5_no_gbm_lora"
    SCRIPT="$EXPDIR/generate_ablation_5_no_gbm_lora.py"
    cd "$EXPDIR" && python "$SCRIPT" \
        --gpu_id 3 \
        --targets $TARGETS \
        --num_molecules $N_MOL \
        --max_attempts $N_ATTEMPT
    echo ">>> GPU 3 DONE: ablation_5_no_gbm_lora"
) &
PID3=$!

echo ""
echo "=== All GPU workers launched ==="
echo "GPU 1 (Ablation 1b CoT):     PID $PID1"
echo "GPU 2 (Ablation 2+3):        PID $PID2"
echo "GPU 3 (Ablation 4+5):        PID $PID3"
echo ""

fail=0
wait $PID1 || { echo "GPU 1 FAILED"; fail=1; }
wait $PID2 || { echo "GPU 2 FAILED"; fail=1; }
wait $PID3 || { echo "GPU 3 FAILED"; fail=1; }

echo ""
echo "============================================================"
if [[ $fail -eq 0 ]]; then
    echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
else
    echo "SOME EXPERIMENTS FAILED"
fi
echo "============================================================"
exit $fail
