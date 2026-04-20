# I KAN DRIVE — Applied KAN for Road Segmentation

## Introduction

A PyTorch implementation of **U-KAN** adapted for road segmentation in the **AUtoCAR** project.  
Inspired by [U-KAN: Strong Backbone for Medical Image Segmentation and Generation](https://github.com/CUHK-AIM-Group/U-KAN) and [FasterKAN](https://github.com/AthanasiosDelis/faster-kan).

Five KAN activation variants are benchmarked end-to-end on the BDD100K segmentation dataset:

| Variant | KAN type |
|---------|----------|
| `bdd100k_FasterKAN` | FasterKAN (default) |
| `bdd100k_ReLU`      | ReLU-KAN |
| `bdd100k_HardSwish` | HardSwish-KAN |
| `bdd100k_PWLO`      | PWLO-KAN |
| `bdd100k_TeLU`      | TeLU-KAN |

---

## Prerequisites

- Linux (Ubuntu, Fedora, RHEL, …)
- NVIDIA GPU with CUDA support (≥ 8 GB VRAM recommended)
- [Conda](https://docs.conda.io/en/latest/) or [Miniforge](https://github.com/conda-forge/miniforge)

---

## Installation

```bash
git clone https://github.com/Spring317/KAN-Road-Segmentation.git
cd KAN-Road-Segmentation
chmod +x setup_env.sh
./setup_env.sh
```

---

## Training

### Train all KAN variants sequentially (1 GPU, 8 GB VRAM)

The script iterates over all five variants one at a time, using **batch size 4 + gradient accumulation 4** to match an effective batch size of 16, with AMP and `torch.compile` enabled.

```bash
conda activate ukan
cd Seg_UKAN

# Edit the two variables at the top of the script if needed:
#   RESUME=True   → resume from the latest checkpoint of each variant
#   RESUME=False  → fresh start (overwrites existing checkpoints and logs)
#   EPOCHS=200    → total target epochs

bash scripts/run_variants_1gpu.sh
```

Logs are written to `outputs/terminal_<KAN_TYPE>.log`.  
Checkpoints are saved to `outputs/bdd100k_<KAN_TYPE>/`.

### Train a single variant manually

```bash
conda activate ukan
cd Seg_UKAN

CUDA_VISIBLE_DEVICES=0 python train.py \
    --name            bdd100k_HardSwish \
    --model_name      UKAN              \
    --kan_type        HardSwish         \
    --batch_size      4                 \
    --grad_accum_steps 4               \
    --use_amp         True              \
    --epochs          200               \
    --resume          False             \
    --num_workers     4                 \
    --compile_model   True
```

### Multi-GPU Distributed Training

DDP is now natively integrated directly into `train.py`. You do not need a separate script — simply launch `train.py` using `torchrun`.

```bash
torchrun --nproc_per_node=4 train.py \
    --name        bdd100k_HardSwish \
    --kan_type    HardSwish \
    --batch_size  4
```

---

## Validation

> [!IMPORTANT]
> Validation requires access to the **BDD100K segmentation dataset**. Pass its path via `--data_path` (or the `DATA_PATH` positional arg in the batch script). The default value is the cluster path `/mnt/ssd-0/M2_internship/bdd100k_seg/bdd100k/seg`.

### Evaluate all trained variants at once

```bash
conda activate ukan
cd Seg_UKAN

# Signature:
#   bash scripts/eval_all_variants.sh [OUTPUT_DIR] [DATA_PATH] [--cpu]

# 1. All defaults (cluster paths, GPU)
bash scripts/eval_all_variants.sh

# 2. Custom output dir, default data path
bash scripts/eval_all_variants.sh /path/to/outputs

# 3. Custom output dir + custom dataset path (most common when running elsewhere)
bash scripts/eval_all_variants.sh \
    /path/to/outputs \
    /path/to/bdd100k/seg

# 4. CPU mode (no GPU required)
bash scripts/eval_all_variants.sh \
    /path/to/outputs \
    /path/to/bdd100k/seg \
    --cpu

# 5. CPU mode via environment variable
USE_CPU=1 bash scripts/eval_all_variants.sh

# 6. High-Performance CPU mode (force 100% CPU utilisation)
# Adjust NUM_THREADS to your cluster's physical core count
BATCH_SIZE=8 NUM_WORKERS=16 NUM_THREADS=32 bash scripts/eval_all_variants.sh \
    /path/to/outputs \
    /path/to/bdd100k/seg \
    --cpu

# 7. Explicit checkpoint for every variant (quick testing)
MODEL_PATH=/path/to/checkpoint_best.pth bash scripts/eval_all_variants.sh
```

Evaluation logs → `<OUTPUT_DIR>/eval_<KAN_TYPE>.log`  
Metrics & visualisations → `<OUTPUT_DIR>/bdd100k_<KAN_TYPE>/`

### Evaluate a single variant manually

```bash
conda activate ukan
cd Seg_UKAN

# GPU (default)
python val.py \
    --name       bdd100k_HardSwish \
    --output_dir /path/to/outputs  \
    --data_path  /path/to/bdd100k/seg \
    --batch_size 1

# CPU — Proper Evaluation without thrashing resources
# Locks exactly 4 cores for math operations and 2 for data loading
python val.py \
    --name        bdd100k_HardSwish \
    --output_dir  /path/to/outputs  \
    --data_path   /path/to/bdd100k/seg \
    --batch_size  1 \
    --cpu \
    --num_threads 4 \
    --num_workers 2


# Explicit checkpoint path
python val.py \
    --name        bdd100k_HardSwish \
    --output_dir  /path/to/outputs  \
    --data_path   /path/to/bdd100k/seg \
    --model_path  /path/to/outputs/bdd100k_HardSwish/checkpoint_best.pth \
    --batch_size  1
```

#### `val.py` flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | `bdd100k_UKAN` | Experiment name (must match the training `--name`) |
| `--output_dir` | `outputs` | Root directory containing experiment output folders |
| `--data_path` | *(cluster path)* | Path to the BDD100K `seg` directory (contains `images/` and `labels/`) |
| `--model_path` | *(auto)* | Explicit checkpoint path — overrides auto-discovered `checkpoint_best.pth` / `model_best.pth` |
| `--batch_size` | `1` | Batch size for the validation dataloader |
| `--num_vis` | `10` | Number of sample images to visualise |
| `--cpu` | `False` | Run on CPU instead of CUDA (no GPU required) |
| `--yolo_exp` | `None` | Validate a YOLO experiment instead of a KAN model |


---

## Output structure

```
Seg_UKAN/outputs/
└── bdd100k_<KAN_TYPE>/
    ├── config.yml            # Saved training config
    ├── checkpoint_best.pth   # Best validation checkpoint
    ├── checkpoint_last.pth   # Latest checkpoint (for resuming)
    ├── log.csv               # Per-epoch train/val metrics
    ├── val_metrics.txt       # Final evaluation summary
    ├── out_val/              # Per-image prediction masks
    └── visualizations/       # Side-by-side comparison plots
```

---

## Citation

```bibtex
@article{li2024ukan,
  title   = {U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation},
  author  = {Li, Chenxin and Liu, Xinyu and Li, Wuyang and Wang, Cheng and Liu, Hengyu and Yuan, Yixuan},
  journal = {arXiv preprint arXiv:2406.02918},
  year    = {2024}
}
```
