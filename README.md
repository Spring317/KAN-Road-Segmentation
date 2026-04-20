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

bash run_variants_1gpu.sh
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

---

## Validation

### Evaluate all trained variants at once

```bash
conda activate ukan
cd Seg_UKAN

# Default — GPU, looks for checkpoints in Seg_UKAN/outputs/
bash eval_all_variants.sh

# Custom output directory (e.g. on a cluster with a different mount point)
bash eval_all_variants.sh /mnt/ssd-0/M2_internship/KAN-Road-Segmentation/Seg_UKAN/outputs

# CPU mode — no GPU required (useful for machines without CUDA)
bash eval_all_variants.sh outputs --cpu

# CPU mode with a custom output directory
bash eval_all_variants.sh /path/to/outputs --cpu

# CPU mode via environment variable
USE_CPU=1 bash eval_all_variants.sh

# Provide an explicit checkpoint path (applied to every variant — for quick testing)
MODEL_PATH=/path/to/checkpoint_best.pth bash eval_all_variants.sh
```

Evaluation logs are written to `<OUTPUT_DIR>/eval_<KAN_TYPE>.log`.  
Metrics and visualisations are saved under `<OUTPUT_DIR>/bdd100k_<KAN_TYPE>/`.

### Evaluate a single variant manually

```bash
conda activate ukan
cd Seg_UKAN

# GPU inference (default)
python val.py \
    --name       bdd100k_HardSwish \
    --output_dir outputs            \
    --batch_size 1

# CPU inference — no GPU required
python val.py \
    --name       bdd100k_HardSwish \
    --output_dir outputs            \
    --batch_size 1                  \
    --cpu

# Override the checkpoint path explicitly
python val.py \
    --name       bdd100k_HardSwish \
    --output_dir outputs            \
    --batch_size 1                  \
    --model_path outputs/bdd100k_HardSwish/model_best.pth

# CPU + explicit checkpoint
python val.py \
    --name       bdd100k_HardSwish \
    --output_dir outputs            \
    --batch_size 1                  \
    --cpu                           \
    --model_path outputs/bdd100k_HardSwish/model_best.pth
```

#### `val.py` flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | `bdd100k_UKAN` | Experiment name (must match the training `--name`) |
| `--output_dir` | `outputs` | Root directory containing experiment folders |
| `--model_path` | *(auto)* | Explicit checkpoint path — overrides the auto-discovered `checkpoint_best.pth` / `model_best.pth` |
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
