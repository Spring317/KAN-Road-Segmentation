#!/bin/bash
# 8GB VRAM Constraint Automated KAN Deploy Script (SEQUENTIAL - 1 GPU)
# Iterates through KAN architectures automatically ONE AT A TIME.
# Uses Batch Size 4 and Gradient Accumulation 4 to match original Batch Size 16.
# Automatic Mixed Precision (AMP) is natively enabled via PyTorch AMP within train.py.
#
# MODES:
#   RESUME=True  → continue each experiment from its latest checkpoint_last.pth
#                  logs are APPENDED (>>) so the full history is preserved
#                  set EPOCHS to total desired epochs (e.g. 200 = 100 already done + 100 more)
#   RESUME=False → fresh start (overwrites existing checkpoints and logs)
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE=4
GRAD_ACCUM=4
EPOCHS=200           # Total epochs across full training run (already done + new)

# ← Set to True to continue from the last checkpoint of each experiment,
#      or False to start fresh (existing outputs will be overwritten).
RESUME=True

echo "========================================================"
echo " KAN Permutation Deploy  |  RESUME=${RESUME}  |  EPOCHS=${EPOCHS}"
echo "========================================================"
mkdir -p outputs

for KAN_TYPE in FasterKAN ReLU HardSwish PWLO TeLU; do
    echo "------------------------------------------"
    echo " Deploying: ${KAN_TYPE}-KAN  (RESUME=${RESUME})"
    echo "------------------------------------------"

    LOG_FILE="outputs/terminal_${KAN_TYPE}.log"

    COMMON_ARGS=(
        --name         "bdd100k_${KAN_TYPE}"
        --model_name   UKAN
        --kan_type     "$KAN_TYPE"
        --batch_size   "$BATCH_SIZE"
        --grad_accum_steps "$GRAD_ACCUM"
        --use_amp      True
        --epochs       "$EPOCHS"
        --resume       "$RESUME"
        --num_workers  4
        --compile_model True
    )

    if [ "$RESUME" = "True" ]; then
        # Append to existing log so terminal history is continuous
        CUDA_VISIBLE_DEVICES=0 python train.py "${COMMON_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
    else
        # Fresh start – overwrite log
        CUDA_VISIBLE_DEVICES=0 python train.py "${COMMON_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    fi

    echo " → Launched in background  (log: ${LOG_FILE})"
done

echo ""
echo "All ${#} jobs launched. Waiting for completion…"
wait
echo "All experiments finished!"
