# I KAN DRIVE (Applied KAN for Road segmentation tasks)

## Introduction

* This is a PyTorch implementation of **U-KAN implementation on Road Segmentation for AUtoCAR
This implementation is mainly nspired by [U-KAN make strong backbones for Medical Image Segmentation and Generation](https://github.com/CUHK-AIM-Group/U-KAN) and [FasterKAN](https://github.com/AthanasiosDelis/faster-kan)

## Prerequisites

Before setting up the project, ensure the host machine has:

* A Linux operating system (Ubuntu, Fedora, RHEL, etc.)
* An NVIDIA GPU with CUDA support
* [Conda](https://docs.conda.io/en/latest/) or [Miniforge](https://github.com/conda-forge/miniforge) installed

## Installation

   ```bash
    git clone https://github.com/Spring317/KAN-Road-Segmentation.git
    cd KAN-Road-Segmentation
    chmod +x setup_env.sh
    ./setup_env.sh
    
    ```
## Start training:
  
   ```bash
    conda activate ukan
    python train_ddp.py --name test-ukan --batch_size 8
  ```

## Citation

```
@article{li2024ukan,
  title={U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation},
  author={Li, Chenxin and Liu, Xinyu and Li, Wuyang and Wang, Cheng and Liu, Hengyu and Yuan, Yixuan},
  journal={arXiv preprint arXiv:2406.02918},
  year={2024}
'''
}
