#!/bin/bash
# Evaluate all trained KAN variants sequentially.
# Batch size is forced to 1 for consistent FPS measurement.
#
# Usage:
#   bash eval_all_variants.sh                               # GPU, auto-discover checkpoints in outputs/
#   bash eval_all_variants.sh /path/to/outputs              # GPU, custom output_dir
#   bash eval_all_variants.sh /path/to/outputs --cpu        # CPU mode, custom output_dir
#   USE_CPU=1 bash eval_all_variants.sh                     # CPU mode via env var
#   MODEL_PATH=/path/to/best.pth bash eval_all_variants.sh  # explicit checkpoint for every variant

BATCH_SIZE=1
OUTPUT_DIR="${1:-outputs}"   # First positional arg overrides output_dir; defaults to "outputs"

# CPU mode: set USE_CPU=1 env var OR pass --cpu as second positional arg
USE_CPU=${USE_CPU:-0}
if [ "${2}" = "--cpu" ]; then
    USE_CPU=1
fi

if [ "${USE_CPU}" = "1" ]; then
    DEVICE_LABEL="CPU"
else
    DEVICE_LABEL="GPU (CUDA_VISIBLE_DEVICES=0)"
fi

echo "========================================================"
echo " KAN Variants Evaluation  |  BATCH_SIZE=${BATCH_SIZE}  |  DEVICE=${DEVICE_LABEL}"
echo " OUTPUT_DIR=${OUTPUT_DIR}"
echo "========================================================"
mkdir -p "${OUTPUT_DIR}"

for KAN_TYPE in FasterKAN HardSwish PWLO ReLU TeLU; do
    NAME="bdd100k_${KAN_TYPE}"
    LOG_FILE="${OUTPUT_DIR}/eval_${KAN_TYPE}.log"

    echo "------------------------------------------"
    echo " Evaluating: ${KAN_TYPE}-KAN  (name=${NAME})"
    echo "------------------------------------------"

    # Build optional extra arguments
    EXTRA_ARGS=()
    if [ -n "${MODEL_PATH}" ]; then
        EXTRA_ARGS+=(--model_path "${MODEL_PATH}")
    fi
    if [ "${USE_CPU}" = "1" ]; then
        EXTRA_ARGS+=(--cpu)
    fi

    if [ "${USE_CPU}" = "1" ]; then
        python val.py \
            --name        "${NAME}"       \
            --output_dir  "${OUTPUT_DIR}" \
            --batch_size  "${BATCH_SIZE}" \
            "${EXTRA_ARGS[@]}"            \
            > "${LOG_FILE}" 2>&1
    else
        CUDA_VISIBLE_DEVICES=0 python val.py \
            --name        "${NAME}"       \
            --output_dir  "${OUTPUT_DIR}" \
            --batch_size  "${BATCH_SIZE}" \
            "${EXTRA_ARGS[@]}"            \
            > "${LOG_FILE}" 2>&1
    fi

    echo " → Done  (log: ${LOG_FILE})"
done

echo ""
echo "All evaluations finished!"
