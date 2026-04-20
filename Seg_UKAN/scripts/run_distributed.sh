#!/bin/bash
# Distributed Training Launch Script for UKAN
# Optimized for 2x Quadro RTX 4000 (8GB VRAM each)

# Configuration
NUM_GPUS=2
EXPERIMENT_NAME="bdd100k_UKAN_ddp_fast"

# Batch size per GPU - adjust based on VRAM
# With 8GB VRAM and mixed precision, you can typically use batch_size 8-16
BATCH_SIZE_PER_GPU=32

# Gradient accumulation to achieve larger effective batch size
# Effective batch size = BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACCUM
GRAD_ACCUM=2

# Number of workers per GPU for data loading
NUM_WORKERS=4

# Image size - reduce if running out of memory
INPUT_H=192
INPUT_W=256

# Training parameters
EPOCHS=100
LR=1e-4

echo "=============================================="
echo "Distributed Training Configuration"
echo "=============================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Gradient accumulation steps: ${GRAD_ACCUM}"
echo "Effective batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACCUM))"
echo "Input size: ${INPUT_H}x${INPUT_W}"
echo "=============================================="

# Set CUDA visible devices (adjust if your GPUs have different indices)
export CUDA_VISIBLE_DEVICES=0,1

# Enable cuDNN autotuning for speed
export CUDNN_BENCHMARK=1

# Limit CPU threads per worker to avoid contention
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Launch distributed training with torchrun
torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=29500 \
  train_ddp.py \
  --name ${EXPERIMENT_NAME} \
  --dataset bdd100k \
  --arch UKAN \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE_PER_GPU} \
  --grad_accum_steps ${GRAD_ACCUM} \
  --input_h ${INPUT_H} \
  --input_w ${INPUT_W} \
  --lr ${LR} \
  --num_workers ${NUM_WORKERS} \
  --use_amp True \
  --sync_bn True \
  --prefetch_factor 4 \
  --scheduler CosineAnnealingLR \
  --output_dir outputs \
  --input_list 64,128,256 \
  "$@" # Pass any additional arguments
