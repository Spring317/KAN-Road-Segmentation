#!/bin/bash
# 8GB VRAM Constraint Automated KAN Deploy Script
# Iterates through KAN architectures automatically.
# Uses Batch Size 4 and Gradient Accumulation 4 to match original Batch Size 16.
# Automatic Mixed Precision (AMP) is natively enabled via PyTorch AMP within train.py.

BATCH_SIZE=4
GRAD_ACCUM=4
EPOCHS=100

echo "Starting automated KAN permutation deployment on 8GB VRAM..."

for KAN_TYPE in FasterKAN ReLU HardSwish PWLO TeLU; do
    echo "=========================================="
    echo "Deploying Experiment: ${KAN_TYPE}-KAN"
    echo "=========================================="
    
    python train.py \
        --name bdd100k_${KAN_TYPE} \
        --model_name UKAN \
        --kan_type $KAN_TYPE \
        --batch_size $BATCH_SIZE \
        --grad_accum_steps $GRAD_ACCUM \
        --use_amp True \
        --epochs $EPOCHS \
        --num_workers 4 \
        --compile_model True
        
    echo "Completed ${KAN_TYPE}-KAN"
done

echo "All permutations completely deployed!"
