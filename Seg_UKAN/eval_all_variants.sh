#!/bin/bash
# Evaluate all trained KAN variants sequentially on 1 GPU
# Batch size is forced to 1 for consistent FPS measurement.

BATCH_SIZE=1

echo "========================================================"
echo " KAN Variants Evaluation  |  BATCH_SIZE=${BATCH_SIZE}"
echo "========================================================"
mkdir -p outputs

for KAN_TYPE in FasterKAN HardSwish PWLO ReLU TeLU; do
    NAME="bdd100k_${KAN_TYPE}"
    LOG_FILE="outputs/eval_${KAN_TYPE}.log"

    echo "------------------------------------------"
    echo " Evaluating: ${KAN_TYPE}-KAN  (name=${NAME})"
    echo "------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 python val.py \
        --name        "${NAME}" \
        --batch_size  "${BATCH_SIZE}" \
        > "${LOG_FILE}" 2>&1

    echo " → Done  (log: ${LOG_FILE})"
done

echo ""
echo "All evaluations finished!"
