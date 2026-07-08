#!/bin/bash
# Evaluate all trained YOLO-KAN variants sequentially.
# Batch size is forced to 1 by default for consistent FPS measurement.
#
# Usage:
#   bash scripts/eval_yolo_kan_variants.sh                                          # GPU, all defaults (both frozen & unfrozen)
#   bash scripts/eval_yolo_kan_variants.sh --freeze                                 # GPU, frozen backbone only
#   bash scripts/eval_yolo_kan_variants.sh --unfreeze                               # GPU, unfrozen backbone only
#   bash scripts/eval_yolo_kan_variants.sh --freeze <OUTPUT_DIR>                    # custom outputs dir
#   bash scripts/eval_yolo_kan_variants.sh --freeze <OUTPUT_DIR> <DATA_PATH>        # custom outputs + dataset path
#   bash scripts/eval_yolo_kan_variants.sh --freeze <OUTPUT_DIR> <DATA_PATH> --cpu  # CPU mode
#   USE_CPU=1 bash scripts/eval_yolo_kan_variants.sh                                # CPU mode via env var
#   MODEL_PATH=/path/to/best.pth bash scripts/eval_yolo_kan_variants.sh             # explicit checkpoint
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE=${BATCH_SIZE:-1}
NUM_THREADS=${NUM_THREADS:-0}
NUM_WORKERS=${NUM_WORKERS:--1}

# ── Parse backbone mode ──────────────────────────────────────────────────────
BACKBONE_MODES=()
case "$1" in
    --freeze)   BACKBONE_MODES=("frozen");            shift ;;
    --unfreeze) BACKBONE_MODES=("unfrozen");          shift ;;
    *)          BACKBONE_MODES=("frozen" "unfrozen")         ;;
esac

OUTPUT_DIR="${1:-outputs}"
DATA_PATH="${2:-/mnt/ssd-0/M2_internship/bdd100k_seg/bdd100k/seg}"

# ── CPU mode: set USE_CPU=1 env var OR pass --cpu as positional arg ──────────
USE_CPU=${USE_CPU:-0}
for arg in "$@"; do
    if [ "$arg" = "--cpu" ]; then
        USE_CPU=1
    fi
done

if [ "${USE_CPU}" = "1" ]; then
    DEVICE_LABEL="CPU"
else
    DEVICE_LABEL="GPU (CUDA_VISIBLE_DEVICES=0)"
fi

echo "========================================================"
echo " YOLO-KAN Variants Evaluation"
echo "   BATCH_SIZE      : ${BATCH_SIZE}"
echo "   OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "   DATA_PATH       : ${DATA_PATH}"
echo "   DEVICE          : ${DEVICE_LABEL}"
echo "   BACKBONE MODES  : ${BACKBONE_MODES[*]}"
echo "========================================================"
mkdir -p "${OUTPUT_DIR}"

TOTAL=0
PASS=0
FAIL=0

for MODE in "${BACKBONE_MODES[@]}"; do
    for KAN_TYPE in FasterKAN ReLU HardSwish PWLO TeLU; do
        NAME="bdd100k_yolo_kan_${KAN_TYPE}_${MODE}"
        CONFIG_FILE="${OUTPUT_DIR}/${NAME}/config.yml"
        LOG_FILE="${OUTPUT_DIR}/eval_yolo_kan_${KAN_TYPE}_${MODE}.log"

        echo "------------------------------------------"
        echo " Evaluating: ${KAN_TYPE}-KAN  (backbone=${MODE})"
        echo "   name     : ${NAME}"
        echo "   config   : ${CONFIG_FILE}"
        echo "------------------------------------------"

        # Skip if config doesn't exist (variant was never trained)
        if [ ! -f "${CONFIG_FILE}" ]; then
            echo " ⚠  Config not found — skipping ${NAME}"
            echo ""
            continue
        fi

        TOTAL=$((TOTAL + 1))

        # Build optional extra arguments
        EXTRA_ARGS=()
        if [ -n "${MODEL_PATH}" ]; then
            EXTRA_ARGS+=(--model_path "${MODEL_PATH}")
        fi
        if [ "${USE_CPU}" = "1" ]; then
            EXTRA_ARGS+=(--cpu --num_threads "${NUM_THREADS}" --num_workers "${NUM_WORKERS}")
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

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -eq 0 ]; then
            PASS=$((PASS + 1))
            echo " ✓ Done  (log: ${LOG_FILE})"
        else
            FAIL=$((FAIL + 1))
            echo " ✗ FAILED (exit ${EXIT_CODE}) — check ${LOG_FILE}"
        fi
        echo ""
    done
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo " YOLO-KAN Evaluation Summary"
echo "   Total : ${TOTAL}"
echo "   Pass  : ${PASS}"
echo "   Fail  : ${FAIL}"
echo "========================================================"

# ── Aggregate metrics into a single table ────────────────────────────────────
SUMMARY_FILE="${OUTPUT_DIR}/yolo_kan_eval_summary.txt"
{
    printf "%-45s  %8s  %8s  %8s  %8s\n" "Model" "IoU" "Dice" "mAP" "FPS"
    printf "%-45s  %8s  %8s  %8s  %8s\n" "-----" "---" "----" "---" "---"
    for MODE in "${BACKBONE_MODES[@]}"; do
        for KAN_TYPE in FasterKAN ReLU HardSwish PWLO TeLU; do
            NAME="bdd100k_yolo_kan_${KAN_TYPE}_${MODE}"
            METRICS="${OUTPUT_DIR}/${NAME}/val_metrics.txt"
            if [ -f "${METRICS}" ]; then
                IOU=$(grep "Overall IoU"  "${METRICS}" | awk '{print $NF}')
                DICE=$(grep "Overall Dice" "${METRICS}" | awk '{print $NF}')
                MAP=$(grep "Overall mAP"  "${METRICS}" | awk '{print $NF}')
                FPS=$(grep "Average FPS"   "${METRICS}" | awk '{print $NF}')
                printf "%-45s  %8s  %8s  %8s  %8s\n" "${NAME}" "${IOU}" "${DICE}" "${MAP}" "${FPS}"
            fi
        done
    done
} > "${SUMMARY_FILE}"

echo ""
echo "Aggregated results:"
echo ""
cat "${SUMMARY_FILE}"
echo ""
echo "Summary saved to ${SUMMARY_FILE}"
echo ""
echo "All YOLO-KAN evaluations finished!"
