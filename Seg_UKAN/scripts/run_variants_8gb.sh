#!/bin/bash
# 8GB VRAM Constraint Automated KAN Deploy Script
# Iterates through KAN architectures automatically.
# Uses Batch Size 4 and Gradient Accumulation 4 to match original Batch Size 16.
# Automatic Mixed Precision (AMP) is natively enabled via PyTorch AMP within train.py.

BATCH_SIZE=4
GRAD_ACCUM=4
EPOCHS=100

echo "Starting automated KAN permutation deployment on 8GB VRAM..."
mkdir -p outputs

i=0
for KAN_TYPE in FasterKAN ReLU HardSwish PWLO TeLU; do
    GPU_ID=$(( i % 2 ))
    echo "=========================================="
    echo "Deploying Experiment: ${KAN_TYPE}-KAN on GPU $GPU_ID"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --name bdd100k_${KAN_TYPE} \
        --model_name UKAN \
        --kan_type $KAN_TYPE \
        --batch_size $BATCH_SIZE \
        --grad_accum_steps $GRAD_ACCUM \
        --use_amp True \
        --epochs $EPOCHS \
        --num_workers 4 \
        --compile_model True > "outputs/terminal_${KAN_TYPE}.log" 2>&1 &
        
    echo "Deployed ${KAN_TYPE}-KAN onto GPU $GPU_ID in the background (logs: outputs/terminal_${KAN_TYPE}.log)..."
    i=$(( i + 1 ))
done

echo "Waiting for all parallel permutations to finish..."
wait
echo "All parallel permutations completely deployed!"
