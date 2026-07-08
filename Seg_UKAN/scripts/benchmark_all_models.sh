#!/bin/bash
# Benchmark all INT8 models on the BDD100K val set

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $(basename "${SCRIPT_DIR}") == "scripts" ]]; then
    PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
else
    PROJECT_DIR="${SCRIPT_DIR}"
fi
OUTPUTS_DIR="${PROJECT_DIR}/outputs"

MODEL_NAME="$1"
DATA_PATH="/home/pi/kan_segmentation_deploy/bdd100k/seg"
LOG_FILE="${PROJECT_DIR}/benchmark_results.txt"

: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "  Data path: ${DATA_PATH}"
echo "  Log file:  ${LOG_FILE}"
if [ -n "$MODEL_NAME" ]; then
    echo "  Benchmarking Specific Model: ${MODEL_NAME}"
    if [ -d "${OUTPUTS_DIR}/${MODEL_NAME}" ]; then
        dirs=("${OUTPUTS_DIR}/${MODEL_NAME}")
    else
        echo "  Error: Model directory ${OUTPUTS_DIR}/${MODEL_NAME} not found!"
        exit 1
    fi
else
    echo "  Benchmarking All INT8 Models"
    dirs=("${OUTPUTS_DIR}"/*)
fi
echo "======================================================================"

for exp_dir in "${dirs[@]}"; do
    if [ -d "${exp_dir}" ]; then
        exp_name=$(basename "${exp_dir}")
        echo "----------------------------------------------------------------------"
        echo "  Experiment: ${exp_name}"
        echo "----------------------------------------------------------------------"
        
        if [ -f "${exp_dir}/model_int8.tflite" ]; then
            python "${PROJECT_DIR}/inference_rpi.py" \
                --model "${exp_dir}/model_int8.tflite" \
                --validate_dir "${DATA_PATH}" \
                --runtime tflite
        else
            echo "  [SKIPPED] No INT8 TFLite model found for ${exp_name}"
        fi

        if [ -f "${exp_dir}/model_int8.onnx" ]; then
            python "${PROJECT_DIR}/inference_rpi.py" \
                --model "${exp_dir}/model_int8.onnx" \
                --validate_dir "${DATA_PATH}" \
                --runtime onnx
        else
            echo "  [SKIPPED] No INT8 ONNX model found for ${exp_name}"
        fi
    fi
done

echo ""
echo "All benchmarks finished!"
