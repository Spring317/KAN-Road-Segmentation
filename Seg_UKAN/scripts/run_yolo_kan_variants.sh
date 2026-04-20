#!/bin/bash
# 8GB VRAM Constraint Automated YOLO-KAN Deploy Script
# Iterates through KAN architectures and both Frozen/Unfrozen backbone configs.
# 
# MODES:
#   RESUME=True  → continue each experiment from its latest checkpoint_last.pth
#                  logs are APPENDED (>>) so the full history is preserved
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE=4
GRAD_ACCUM=4
EPOCHS=200           # Total epochs across full training run (already done + new)

RESUME=True

echo "========================================================"
echo " YOLO-KAN Permutation Deploy | RESUME=${RESUME} | EPOCHS=${EPOCHS}"
echo "========================================================"
mkdir -p outputs

for FREEZE_BACKBONE in True False; do
    for KAN_TYPE in FasterKAN ReLU HardSwish PWLO TeLU; do
        if [ "$FREEZE_BACKBONE" = "True" ]; then
            CONFIG_NAME="bdd100k_yolo_kan_${KAN_TYPE}_frozen"
        else
            CONFIG_NAME="bdd100k_yolo_kan_${KAN_TYPE}_unfrozen"
        fi

        echo "------------------------------------------"
        echo " Deploying: ${KAN_TYPE}-KAN (Freeze=${FREEZE_BACKBONE})"
        echo "------------------------------------------"

        LOG_FILE="outputs/terminal_${CONFIG_NAME}.log"

        COMMON_ARGS=(
            --name         "${CONFIG_NAME}"
            --model_name   "yolo_kan"
            --arch         "YOLOKANSeg"
            --kan_type     "${KAN_TYPE}"
            --yolo_freeze_backbone "${FREEZE_BACKBONE}"
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
done

# Wait for all background instances to finish
wait

echo ""
echo "All YOLO-KAN experiments finished!"
