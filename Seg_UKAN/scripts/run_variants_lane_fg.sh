#!/bin/bash
# Binary Road Segmentation (lane_fg) — KAN Variant Mass Training (1 GPU)
# Trains all KAN architectures with --label_grouping lane_fg (2-class: road vs background).
# Uses Batch Size 4 and Gradient Accumulation 4 to match effective Batch Size 16.
#
# MODES:
#   RESUME=True  → continue each experiment from its latest checkpoint_last.pth
#                  logs are APPENDED (>>) so the full history is preserved
#                  set EPOCHS to total desired epochs (e.g. 200 = 100 already done + 100 more)
#   RESUME=False → fresh start (overwrites existing checkpoints and logs)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BATCH_SIZE=16
GRAD_ACCUM=1
EPOCHS=200

# ← Set to True to continue from the last checkpoint of each experiment,
#      or False to start fresh (existing outputs will be overwritten).
RESUME=False

echo "========================================================"
echo " KAN lane_fg (Binary Road) Deploy  |  RESUME=${RESUME}  |  EPOCHS=${EPOCHS}"
echo " Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
mkdir -p outputs

run_job() {
  local KAN_TYPE=$1
  local GPU_ID=$2
  local LOG_FILE="outputs/terminal_lane_fg_${KAN_TYPE}.log"

  local COMMON_ARGS=(
    --name "bdd100k_lane_fg_${KAN_TYPE}"
    --model_name UKAN
    --kan_type "$KAN_TYPE"
    --label_grouping lane_fg
    --input_w 640
    --input_h 384
    --batch_size "$BATCH_SIZE"
    --grad_accum_steps "$GRAD_ACCUM"
    --use_amp True
    --epochs "$EPOCHS"
    --resume "$RESUME"
    --num_workers 4
    --compile_model True
  )

  echo " [$(date '+%H:%M:%S')] Launched ${KAN_TYPE}-KAN (lane_fg) on GPU ${GPU_ID}  |  log: ${LOG_FILE}"

  if [ "$RESUME" = "True" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py "${COMMON_ARGS[@]}" >>"$LOG_FILE" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py "${COMMON_ARGS[@]}" >"$LOG_FILE" 2>&1
  fi
  
  local EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo " [$(date '+%H:%M:%S')] ✓ ${KAN_TYPE}-KAN (lane_fg) finished successfully on GPU ${GPU_ID}"
  else
    echo " [$(date '+%H:%M:%S')] ✗ ${KAN_TYPE}-KAN (lane_fg) FAILED on GPU ${GPU_ID} with exit code ${EXIT_CODE}"
    echo "   └─ Check log: ${LOG_FILE}"
  fi
}

echo " Monitor tips:"
echo "   tail -f outputs/terminal_lane_fg_FasterKAN.log   # follow one job"
echo "   tail -n5 outputs/terminal_lane_fg_*.log          # check all latest"
echo ""

# Queue for GPU 0
(
  run_job "FasterKAN" 0
  run_job "HardSwish" 0
  run_job "TeLU" 0
) &
PID0=$!

# Queue for GPU 1
(
  run_job "ReLU" 1
  run_job "PWLO" 1
) &
PID1=$!

# Wait for both GPU queues to finish
wait $PID0
wait $PID1

echo ""
echo "========================================================"
echo " All lane_fg experiments finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
