#!/bin/bash
# Deploy FP32 ONNX inference to Raspberry Pi

PI_USER="pi"
PI_IP="192.168.1.90"
PI_TARGET_DIR="~/kan_segmentation_deploy"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Deploying to ${PI_USER}@${PI_IP}:${PI_TARGET_DIR}"

# 1. Create target directories on the Pi
echo "Creating target directories..."
ssh "${PI_USER}@${PI_IP}" "mkdir -p ${PI_TARGET_DIR}/outputs"

# 2. Copy the inference script and requirements
echo "Copying scripts and requirements..."
scp "${PROJECT_DIR}/inference_onnx_fp32.py" "${PI_USER}@${PI_IP}:${PI_TARGET_DIR}/"
scp "${PROJECT_DIR}/deploy_requirements.txt" "${PI_USER}@${PI_IP}:${PI_TARGET_DIR}/"

# 3. Copy FP32 ONNX model(s)
echo "Copying FP32 ONNX model(s)..."
# Using rsync to maintain directory structure of outputs but explicitly filtering for FP32 ONNX
rsync -avz --include='*/' --include='model.onnx' --exclude='*' "${PROJECT_DIR}/outputs/" "${PI_USER}@${PI_IP}:${PI_TARGET_DIR}/outputs/"

echo ""
echo "======================================================================"
echo "  Deployment Complete!"
echo "======================================================================"
echo "To test on your Raspberry Pi, run:"
echo "  ssh ${PI_USER}@${PI_IP}"
echo "  cd ${PI_TARGET_DIR}"
echo "  python inference_onnx_fp32.py --model outputs/<exp>/model.onnx --image test.jpg"
echo "======================================================================"
