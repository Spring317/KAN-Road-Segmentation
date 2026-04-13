#!/bin/bash
# 8GB VRAM Constraint Automated KAN Deploy Script (SEQUENTIAL - 1 GPU)
# Iterates through KAN architectures automatically ONE AT A TIME.
# Uses Batch Size 4 and Gradient Accumulation 4 to match original Batch Size 16.
# Automatic Mixed Precision (AMP) is natively enabled via PyTorch AMP within train.py.

BATCH_SIZE=4
GRAD_ACCUM=4
EPOCHS=100

echo "Starting automated KAN permutation deployment on 1 GPU (Sequential)..."
mkdir -p outputs

for KAN_TYPE in FasterKAN ReLU HardSwish PWLO TeLU; do
    echo "=========================================="
    echo "Deploying Experiment: ${KAN_TYPE}-KAN on single GPU"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --name bdd100k_${KAN_TYPE} \
        --model_name UKAN \
        --kan_type $KAN_TYPE \
        --batch_size $BATCH_SIZE \
        --grad_accum_steps $GRAD_ACCUM \
        --use_amp True \
        --epochs $EPOCHS \
        --num_workers 4 \
        --compile_model True 2>&1 | tee "outputs/terminal_${KAN_TYPE}.log"
        
    echo "Finished ${KAN_TYPE}-KAN!"
done

echo "All permutations successfully deployed sequentially!"
