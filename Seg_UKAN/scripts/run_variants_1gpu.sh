#!/bin/bash
# 8GB VRAM Constraint Automated KAN Deploy Script (PARALLEL - 1 GPU)
# Iterates through KAN architectures automatically, all launched in parallel.
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
EPOCHS=200 # Total epochs across full training run (already done + new)

# ← Set to True to continue from the last checkpoint of each experiment,
#      or False to start fresh (existing outputs will be overwritten).
RESUME=False

echo "========================================================"
echo " KAN Permutation Deploy  |  RESUME=${RESUME}  |  EPOCHS=${EPOCHS}"
echo " Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
mkdir -p outputs

declare -a PIDS=()
declare -a NAMES=()
declare -a LOGS=()
JOB_COUNT=0

for KAN_TYPE in FasterKAN ReLU HardSwish PWLO TeLU; do
  LOG_FILE="outputs/terminal_${KAN_TYPE}.log"

  COMMON_ARGS=(
    --name "bdd100k_${KAN_TYPE}"
    --model_name UKAN
    --kan_type "$KAN_TYPE"
    --batch_size "$BATCH_SIZE"
    --grad_accum_steps "$GRAD_ACCUM"
    --use_amp True
    --epochs "$EPOCHS"
    --resume "$RESUME"
    --num_workers 4
    --compile_model True
  )

  if [ "$RESUME" = "True" ]; then
    # Append to existing log so terminal history is continuous
    CUDA_VISIBLE_DEVICES=0 python train.py "${COMMON_ARGS[@]}" >>"$LOG_FILE" 2>&1 &
  else
    # Fresh start – overwrite log
    CUDA_VISIBLE_DEVICES=0 python train.py "${COMMON_ARGS[@]}" >"$LOG_FILE" 2>&1 &
  fi

  PID=$!
  PIDS+=("$PID")
  NAMES+=("$KAN_TYPE")
  LOGS+=("$LOG_FILE")
  JOB_COUNT=$((JOB_COUNT + 1))

  echo " [${JOB_COUNT}] Launched ${KAN_TYPE}-KAN  |  PID: ${PID}  |  log: ${LOG_FILE}"
done

echo ""
echo "========================================================"
echo " ${JOB_COUNT} jobs launched at $(date '+%H:%M:%S'). Waiting for completion…"
echo "========================================================"
echo ""
echo " Monitor tips:"
echo "   tail -f outputs/terminal_FasterKAN.log   # follow one job"
echo "   tail -n5 outputs/terminal_*.log          # check all latest"
echo ""

# Wait for all jobs and report results
FAILED=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"
  EXIT_CODE=$?
  TIMESTAMP=$(date '+%H:%M:%S')
  if [ $EXIT_CODE -eq 0 ]; then
    echo " [${TIMESTAMP}] ✓ ${NAMES[$i]}-KAN finished successfully  (PID ${PIDS[$i]})"
  else
    echo " [${TIMESTAMP}] ✗ ${NAMES[$i]}-KAN FAILED with exit code ${EXIT_CODE}  (PID ${PIDS[$i]})"
    echo "   └─ Check log: ${LOGS[$i]}"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "========================================================"
echo " All ${JOB_COUNT} experiments finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results: $((JOB_COUNT - FAILED)) succeeded, ${FAILED} failed"
echo "========================================================"

