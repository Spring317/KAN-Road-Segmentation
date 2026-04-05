#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting U-FasterKAN Environment Setup for Quy..."

# 1. Create the conda environment from configs.yaml
echo "📦 Building conda environment from configs.yaml..."
conda env create -f configs.yaml -y

# 2. Source conda to allow activation inside the script
echo "🔄 Activating environment 'ukan'..."
# Adapt the path if your miniforge/anaconda is installed elsewhere
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ukan

# 3. Compile the custom CUDA kernels
echo "⚙️ Compiling FasterKAN CUDA extensions..."
cd cuda

# Ensure CUDA_HOME is properly set to the conda environment
export CUDA_HOME=$CONDA_PREFIX

# Build the extension inplace
python setup.py build_ext --inplace
cd ..

echo "✅ Setup Complete! The FasterKAN CUDA kernel is ready."
echo "👉 To start training, run:"
echo "conda activate ukan"
echo "python train_ddp.py --name test-ukan --batch_size 8"
