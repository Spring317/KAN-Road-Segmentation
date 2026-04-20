#!/bin/bash
# Evaluate all trained KAN variants sequentially.
# Batch size is forced to 1 for consistent FPS measurement.
#
# Usage:
#   bash eval_all_variants.sh                                        # GPU, all defaults
#   bash eval_all_variants.sh <OUTPUT_DIR>                           # custom outputs dir
#   bash eval_all_variants.sh <OUTPUT_DIR> <DATA_PATH>               # custom outputs + dataset path
#   bash eval_all_variants.sh <OUTPUT_DIR> <DATA_PATH> --cpu         # CPU mode
#   USE_CPU=1 bash eval_all_variants.sh                              # CPU mode via env var
#   MODEL_PATH=/path/to/best.pth bash eval_all_variants.sh           # explicit checkpoint

BATCH_SIZE=1
OUTPUT_DIR="${1:-outputs}"
DATA_PATH="${2:-/mnt/ssd-0/M2_internship/bdd100k_seg/bdd100k/seg}"

# CPU mode: set USE_CPU=1 env var OR pass --cpu as third positional arg
USE_CPU=${USE_CPU:-0}
if [ "${3}" = "--cpu" ] || [ "${2}" = "--cpu" ]; then
    USE_CPU=1
fi

if [ "${USE_CPU}" = "1" ]; then
    DEVICE_LABEL="CPU"
else
    DEVICE_LABEL="GPU (CUDA_VISIBLE_DEVICES=0)"
fi

echo "========================================================"
echo " KAN Variants Evaluation"
echo "   BATCH_SIZE : ${BATCH_SIZE}"
echo "   OUTPUT_DIR : ${OUTPUT_DIR}"
echo "   DATA_PATH  : ${DATA_PATH}"
echo "   DEVICE     : ${DEVICE_LABEL}"
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
            --data_path   "${DATA_PATH}"  \
            --batch_size  "${BATCH_SIZE}" \
            "${EXTRA_ARGS[@]}"            \
            > "${LOG_FILE}" 2>&1
    else
        CUDA_VISIBLE_DEVICES=0 python val.py \
            --name        "${NAME}"       \
            --output_dir  "${OUTPUT_DIR}" \
            --data_path   "${DATA_PATH}"  \
            --batch_size  "${BATCH_SIZE}" \
            "${EXTRA_ARGS[@]}"            \
            > "${LOG_FILE}" 2>&1
    fi

    echo " → Done  (log: ${LOG_FILE})"
done

echo ""
echo "All evaluations finished!"
