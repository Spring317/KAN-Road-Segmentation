#!/usr/bin/env bash
# ============================================================================
# Batch export & quantize all trained models to ONNX INT8 + TFLite INT8
# for Raspberry Pi 4B deployment.
#
# Each model is exported STRICTLY ONE AT A TIME (sequential, no background
# jobs).  Thread counts for PyTorch / ORT / TensorFlow are capped to avoid
# saturating all CPU cores and exhausting RAM.
#
# Usage:
#   bash scripts/export_all_quantized.sh
#   bash scripts/export_all_quantized.sh --onnx-only
#   bash scripts/export_all_quantized.sh --tflite-only
#   bash scripts/export_all_quantized.sh --num-calib=200
#   bash scripts/export_all_quantized.sh --resume        # skip already-done models
#   bash scripts/export_all_quantized.sh --threads=4     # CPU threads per job (default 4)
#   bash scripts/export_all_quantized.sh --onnx-only --resume
#
# NOTE: Uses "conda run -n ukan python ..." so the ukan environment does NOT
#       need to be manually activated before running this script.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# ── Configuration ──────────────────────────────────────────────────────────
NUM_CALIB="${NUM_CALIB:-100}"
FORMATS="onnx tflite"
LOG_DIR="${PROJECT_DIR}/outputs"
RESUME=0  # --resume: skip models that already have all output files
THREADS=4 # CPU threads per export job — keep low to avoid OOM

# Pause (seconds) between models to let the OS reclaim memory before the
# next job starts.  Increase on machines with limited RAM.
INTER_MODEL_PAUSE=10

# Parse flags
for arg in "$@"; do
  case "$arg" in
  --onnx-only) FORMATS="onnx" ;;
  --tflite-only) FORMATS="tflite" ;;
  --num-calib=*) NUM_CALIB="${arg#*=}" ;;
  --threads=*) THREADS="${arg#*=}" ;;
  --resume) RESUME=1 ;;
  esac
done

# ── UKAN models (6) ───────────────────────────────────────────────────────
UKAN_EXPERIMENTS=(
  # "bdd100k_UNet_baseline"
  # "bdd100k_FasterKAN"
  # "bdd100k_ReLU"
  # "bdd100k_HardSwish"
  # "bdd100k_PWLO"
  "bdd100k_TeLU"
)

# ── YOLO-KAN models (6) ──────────────────────────────────────────────────
YOLOKAN_EXPERIMENTS=(
  " "
  # "bdd100k_yolo_kan_FasterKAN_frozen"
  # "bdd100k_yolo_kan_ReLU_frozen"
  # "bdd100k_yolo_kan_HardSwish_frozen"
  # "bdd100k_yolo_kan_PWLO_frozen"
  # "bdd100k_yolo_kan_TeLU_frozen"
  # "bdd100k_yolo_kan_FasterKAN_unfrozen"
)

ALL_EXPERIMENTS=("${UKAN_EXPERIMENTS[@]}" "${YOLOKAN_EXPERIMENTS[@]}")

echo "════════════════════════════════════════════════════════════════"
echo "  Batch Export & Quantization Pipeline  (calibration fix applied)"
echo "  Models:      ${#ALL_EXPERIMENTS[@]}"
echo "  Formats:     ${FORMATS}"
echo "  Calibration: ${NUM_CALIB} images"
echo "  CPU threads: ${THREADS} per job"
echo "  Inter-pause: ${INTER_MODEL_PAUSE}s"
echo "  Resume mode: $([ ${RESUME} -eq 1 ] && echo 'ON (skip completed)' || echo 'OFF (re-export all)')"
echo "════════════════════════════════════════════════════════════════"

SUCCEEDED=0
FAILED=0
SKIPPED=0
FAILED_NAMES=()

# ── Helper: check if all requested output files already exist ─────────────
outputs_exist() {
  local exp="$1"
  local all_exist=1
  if [[ "${FORMATS}" == *"onnx"* ]]; then
    [ -f "${LOG_DIR}/${exp}/model_int8.onnx" ] || all_exist=0
  fi
  if [[ "${FORMATS}" == *"tflite"* ]]; then
    [ -f "${LOG_DIR}/${exp}/model_int8.tflite" ] || all_exist=0
  fi
  return $((1 - all_exist))
}

# ── Helper: delete stale output files before re-exporting ─────────────────
remove_stale_outputs() {
  local exp="$1"
  if [[ "${FORMATS}" == *"onnx"* ]]; then
    rm -f "${LOG_DIR}/${exp}/model_int8.onnx"
    rm -f "${LOG_DIR}/${exp}/model_int8_preprocessed.onnx"
  fi
  if [[ "${FORMATS}" == *"tflite"* ]]; then
    rm -f "${LOG_DIR}/${exp}/model_int8.tflite"
  fi
}

# ── Main loop — strictly sequential, one model at a time ──────────────────
for exp in "${ALL_EXPERIMENTS[@]}"; do
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Processing: ${exp}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # --resume: skip if all requested outputs are already present
  if [ ${RESUME} -eq 1 ] && outputs_exist "${exp}"; then
    echo "  ↷ Skipping (outputs already exist, --resume is ON)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Remove stale INT8 artefacts so we never accidentally ship old ones
  remove_stale_outputs "${exp}"

  LOG_FILE="${LOG_DIR}/export_${exp}.log"

  # ── Resource limits passed as env vars into the child process ──────────
  # OMP / MKL / OpenBLAS  → cap CPU threads used by PyTorch & ORT
  # TF_NUM_*              → cap TensorFlow threads (TFLite conversion)
  # CUDA_VISIBLE_DEVICES="" → export is CPU-only; prevent TF from grabbing GPU VRAM
  # MALLOC_TRIM_THRESHOLD_=0 → ask glibc to return free pages to OS promptly
  #
  # --skip_accuracy removes the second model-load pass (halves peak RAM)
  # -----------------------------------------------------------------------
  if conda run -n ukan \
    env \
    OMP_NUM_THREADS="${THREADS}" \
    MKL_NUM_THREADS="${THREADS}" \
    OPENBLAS_NUM_THREADS="${THREADS}" \
    NUMEXPR_NUM_THREADS="${THREADS}" \
    TF_NUM_INTEROP_THREADS="${THREADS}" \
    TF_NUM_INTRAOP_THREADS="${THREADS}" \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES="" \
    MALLOC_TRIM_THRESHOLD_=0 \
    python "${PROJECT_DIR}/export_quantize.py" \
    --name "${exp}" \
    --num_calib "${NUM_CALIB}" \
    --formats ${FORMATS} \
    --skip_accuracy \
    2>&1 | tee "${LOG_FILE}"; then
    SUCCEEDED=$((SUCCEEDED + 1))
    echo "  ✓ ${exp} exported successfully"
  else
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("${exp}")
    echo "  ✗ ${exp} FAILED — see ${LOG_FILE}"
    # Continue to the next model instead of aborting the whole batch
  fi

  # Pause between models — lets the OS reclaim RAM before the next job
  if [ ${INTER_MODEL_PAUSE} -gt 0 ]; then
    echo "  … waiting ${INTER_MODEL_PAUSE}s for memory reclaim …"
    sleep ${INTER_MODEL_PAUSE}
  fi
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  BATCH EXPORT COMPLETE"
echo "  Succeeded: ${SUCCEEDED}/${#ALL_EXPERIMENTS[@]}"
echo "  Skipped:   ${SKIPPED}/${#ALL_EXPERIMENTS[@]}"
echo "  Failed:    ${FAILED}/${#ALL_EXPERIMENTS[@]}"
if [ ${FAILED} -gt 0 ]; then
  echo "  Failed experiments:"
  for name in "${FAILED_NAMES[@]}"; do
    echo "    - ${name}"
  done
fi
echo "════════════════════════════════════════════════════════════════"

# ── Size report ────────────────────────────────────────────────────────────
echo ""
echo "  Model Size Report:"
echo "  ──────────────────────────────────────────────────────────────"
printf "  %-45s %12s %8s %8s\n" "Experiment" "PTH" "ONNX8" "TFL8"
echo "  ──────────────────────────────────────────────────────────────"

for exp in "${ALL_EXPERIMENTS[@]}"; do
  PTH_SIZE="—"
  ONNX_SIZE="—"
  TFL_SIZE="—"

  # checkpoint_best.pth (training saves to this name), fall back to model_best.pth
  pth_ckpt="${LOG_DIR}/${exp}/checkpoint_best.pth"
  pth_model="${LOG_DIR}/${exp}/model_best.pth"
  onnx="${LOG_DIR}/${exp}/model_int8.onnx"
  tfl="${LOG_DIR}/${exp}/model_int8.tflite"

  if [ -f "${pth_ckpt}" ]; then
    PTH_SIZE="$(du -h "${pth_ckpt}" | cut -f1)"
  elif [ -f "${pth_model}" ]; then
    PTH_SIZE="$(du -h "${pth_model}" | cut -f1)"
  fi
  if [ -f "${onnx}" ]; then
    ONNX_SIZE="$(du -h "${onnx}" | cut -f1)"
  fi
  if [ -f "${tfl}" ]; then
    TFL_SIZE="$(du -h "${tfl}" | cut -f1)"
  fi

  printf "  %-45s %12s %8s %8s\n" "${exp}" "${PTH_SIZE}" "${ONNX_SIZE}" "${TFL_SIZE}"
done
echo "  ──────────────────────────────────────────────────────────────"

exit ${FAILED}
