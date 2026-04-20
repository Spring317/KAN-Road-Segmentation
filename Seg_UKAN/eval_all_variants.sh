#!/bin/bash
# Evaluate all trained KAN variants sequentially on 1 GPU
# Batch size is forced to 1 for consistent FPS measurement.
#
# Usage:
#   bash eval_all_variants.sh                          # auto-discover checkpoints
#   bash eval_all_variants.sh /path/to/outputs         # custom output_dir (checkpoints at <OUTPUT_DIR>/bdd100k_<KAN>/...)
#   MODEL_PATH=/abs/path/to/best.pth bash eval_all_variants.sh  # same checkpoint for every variant (testing only)

BATCH_SIZE=1
OUTPUT_DIR="${1:-outputs}"   # First positional arg overrides output_dir; defaults to "outputs"

echo "========================================================"
echo " KAN Variants Evaluation  |  BATCH_SIZE=${BATCH_SIZE}  |  OUTPUT_DIR=${OUTPUT_DIR}"
echo "========================================================"
mkdir -p "${OUTPUT_DIR}"

for KAN_TYPE in FasterKAN HardSwish PWLO ReLU TeLU; do
    NAME="bdd100k_${KAN_TYPE}"
    LOG_FILE="${OUTPUT_DIR}/eval_${KAN_TYPE}.log"

    echo "------------------------------------------"
    echo " Evaluating: ${KAN_TYPE}-KAN  (name=${NAME})"
    echo "------------------------------------------"

    # Build optional --model_path argument
    EXTRA_ARGS=()
    if [ -n "${MODEL_PATH}" ]; then
        EXTRA_ARGS+=(--model_path "${MODEL_PATH}")
    fi

    CUDA_VISIBLE_DEVICES=0 python val.py \
        --name        "${NAME}"       \
        --output_dir  "${OUTPUT_DIR}" \
        --batch_size  "${BATCH_SIZE}" \
        "${EXTRA_ARGS[@]}"            \
        > "${LOG_FILE}" 2>&1

    echo " → Done  (log: ${LOG_FILE})"
done

echo ""
echo "All evaluations finished!"
