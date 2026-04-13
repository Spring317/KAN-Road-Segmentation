"""
Distributed Data Parallel (DDP) Training Script for UKAN
Optimized for multi-GPU training with mixed precision support.

Usage:
    # Single GPU (fallback)
    python train_ddp.py --name experiment_name --batch_size 8

    # Multi-GPU with torchrun (recommended)
    torchrun --nproc_per_node=2 train_ddp.py --name experiment_name --batch_size 16

    # Or using the launch script
    ./run_distributed.sh
"""

import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import albumentations
from typing_extensions import Optional
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations.augmentations as transforms

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Mixed precision
from torch.cuda.amp import GradScaler, autocast

from albumentations.augmentations import geometric
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

import archs
import losses
from dataset import BDD100KDataset, BDD100K_NUM_CLASSES
from metrics import iou_score, indicators
from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter
import shutil
from ultralytics import YOLO

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append("BCEWithLogitsLoss")


def list_type(s):
    str_list = s.split(",")
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", default=None, help="model name: (default: arch+timestamp)"
    )
    parser.add_argument(
        "--model_name", default="UKAN", help="model name: UKAN or yolo (default: UKAN)"
    )
    parser.add_argument(
        "--epochs",
        default=400,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=16,
        type=int,
        metavar="N",
        help="mini-batch size per GPU (default: 16)",
    )
    parser.add_argument("--dataseed", default=2981, type=int)

    # model
    parser.add_argument("--arch", "-a", metavar="ARCH", default="UKAN")
    parser.add_argument("--deep_supervision", default=False, type=str2bool)
    parser.add_argument("--input_channels", default=3, type=int, help="input channels")
    parser.add_argument("--num_classes", default=20, type=int, help="number of classes")
    parser.add_argument("--input_w", default=256, type=int, help="image width")
    parser.add_argument("--input_h", default=192, type=int, help="image height")
    parser.add_argument("--input_list", type=list_type, default=[64, 128, 256])

    # loss
    parser.add_argument(
        "--loss",
        default="BCEDiceLoss",
        choices=LOSS_NAMES,
        help="loss: " + " | ".join(LOSS_NAMES) + " (default: BCEDiceLoss)",
    )

    # dataset
    parser.add_argument("--dataset", default="busi", help="dataset name")
    parser.add_argument("--data_dir", default="inputs", help="dataset dir")
    parser.add_argument("--output_dir", default="outputs", help="ouput dir")

    # optimizer
    parser.add_argument(
        "--optimizer",
        default="Adam",
        choices=["Adam", "SGD", "AdamW"],
        help="optimizer (default: Adam)",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=1e-4,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--nesterov", default=False, type=str2bool)
    parser.add_argument(
        "--kan_lr",
        default=1e-2,
        type=float,
        metavar="LR",
        help="KAN layer learning rate",
    )
    parser.add_argument("--kan_weight_decay", default=1e-4, type=float)

    # scheduler
    parser.add_argument(
        "--scheduler",
        default="CosineAnnealingLR",
        choices=[
            "CosineAnnealingLR",
            "ReduceLROnPlateau",
            "MultiStepLR",
            "ConstantLR",
            "OneCycleLR",
        ],
    )
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--factor", default=0.1, type=float)
    parser.add_argument("--patience", default=2, type=int)
    parser.add_argument("--milestones", default="1,2", type=str)
    parser.add_argument("--gamma", default=2 / 3, type=float)
    parser.add_argument(
        "--early_stopping",
        default=-1,
        type=int,
        metavar="N",
        help="early stopping (default: -1)",
    )
    parser.add_argument("--cfg", type=str, metavar="FILE", help="path to config file")
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="number of data loading workers per GPU",
    )
    parser.add_argument("--no_kan", action="store_true")
    parser.add_argument(
        "--resume", default=False, type=str2bool, help="resume from checkpoint"
    )

    # DDP and optimization arguments
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by torchrun)",
    )
    parser.add_argument(
        "--use_amp",
        default=True,
        type=str2bool,
        help="Use automatic mixed precision training",
    )
    parser.add_argument(
        "--grad_accum_steps",
        default=1,
        type=int,
        help="Gradient accumulation steps (increase effective batch size)",
    )
    parser.add_argument(
        "--sync_bn",
        default=True,
        type=str2bool,
        help="Use synchronized batch normalization for DDP",
    )
    parser.add_argument(
        "--prefetch_factor",
        default=4,
        type=int,
        help="Number of batches to prefetch per worker",
    )
    parser.add_argument(
        "--compile_model",
        default=False,
        type=str2bool,
        help="Use torch.compile for PyTorch 2.0+ (experimental)",
    )

    config = parser.parse_args()
    return config


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif torch.cuda.is_available():
        print("Not using distributed mode - single GPU training")
        return False, 0, 1, 0
    else:
        print("CUDA not available - cannot train")
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    # Synchronize all processes
    dist.barrier()

    return True, rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is main (rank 0)."""
    return rank == 0


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes."""
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def train_one_epoch(
    config,
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    epoch,
    rank,
    world_size,
    grad_accum_steps=1,
):
    """Train for one epoch with mixed precision and gradient accumulation."""
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}
    model.train()

    # Only show progress bar on main process
    if is_main_process(rank):
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")

    optimizer.zero_grad()

    for batch_idx, (input, target, _) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Mixed precision forward pass
        with autocast(enabled=config["use_amp"]):
            if config["deep_supervision"]:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

        # Mixed precision backward pass
        scaler.scale(loss).backward()

        # Update weights after accumulation steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Reduce metrics across GPUs
        loss_reduced = reduce_tensor(loss.data * grad_accum_steps, world_size)
        iou_reduced = reduce_tensor(torch.tensor(iou).cuda(), world_size)

        avg_meters["loss"].update(loss_reduced.item(), input.size(0))
        avg_meters["iou"].update(iou_reduced.item(), input.size(0))

        if is_main_process(rank):
            pbar.set_postfix(
                OrderedDict(
                    [
                        ("loss", f"{avg_meters['loss'].avg:.4f}"),
                        ("iou", f"{avg_meters['iou'].avg:.4f}"),
                    ]
                )
            )
            pbar.update(1)

    if is_main_process(rank):
        pbar.close()

    return OrderedDict(
        [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]
    )


def validate(config, val_loader, model, criterion, rank, world_size):
    """Validate the model."""
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter(), "dice": AverageMeter()}
    model.eval()

    if is_main_process(rank):
        pbar = tqdm(total=len(val_loader), desc="Validation")

    with torch.no_grad():
        for input, target, _ in val_loader:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with autocast(enabled=config["use_amp"]):
                if config["deep_supervision"]:
                    outputs = model(input)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                    iou, dice, _ = iou_score(outputs[-1], target)
                else:
                    output = model(input)
                    loss = criterion(output, target)
                    iou, dice, _ = iou_score(output, target)

            # Reduce metrics across GPUs
            loss_reduced = reduce_tensor(loss.data, world_size)
            iou_reduced = reduce_tensor(torch.tensor(iou).cuda(), world_size)
            dice_reduced = reduce_tensor(torch.tensor(dice).cuda(), world_size)

            avg_meters["loss"].update(loss_reduced.item(), input.size(0))
            avg_meters["iou"].update(iou_reduced.item(), input.size(0))
            avg_meters["dice"].update(dice_reduced.item(), input.size(0))

            if is_main_process(rank):
                pbar.set_postfix(
                    OrderedDict(
                        [
                            ("loss", f"{avg_meters['loss'].avg:.4f}"),
                            ("iou", f"{avg_meters['iou'].avg:.4f}"),
                            ("dice", f"{avg_meters['dice'].avg:.4f}"),
                        ]
                    )
                )
                pbar.update(1)

    if is_main_process(rank):
        pbar.close()

    return OrderedDict(
        [
            ("loss", avg_meters["loss"].avg),
            ("iou", avg_meters["iou"].avg),
            ("dice", avg_meters["dice"].avg),
        ]
    )


def seed_torch(seed=1029, rank=0):
    """Set random seeds for reproducibility."""
    seed = seed + rank  # Different seed per process for data augmentation diversity
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = (
        True  # Enable for speed (disable for reproducibility)
    )
    torch.backends.cudnn.deterministic = False  # Disable for speed


def load_model(config) -> nn.Module:
    """Load several models for experiment
    ######################################
    Parameters:
        model_name (str): name of the model to load
        config (str | None): path to config file (optional, used for loading KAN/FasterKAN)"""

    if config["model_name"] == "UKAN":
        model = archs.__dict__[config["arch"]](
            config["num_classes"],
            config["input_channels"],
            config["deep_supervision"],
            embed_dims=config["input_list"],
            no_kan=config["no_kan"],
        )

        return model

    elif config["model_name"] == "yolo":
        yolo11 = yolo_model("yolov11m-seg.pt")
        model = yolo11.load_model()
        return model


def main():
    config = vars(parse_args())
    print(config)
    # Setup distributed training
    distributed, rank, world_size, local_rank = setup_distributed()

    seed_torch(rank=rank)

    exp_name = config.get("name")
    output_dir = config.get("output_dir")

    # Only create directories and writers on main process
    if is_main_process(rank):
        os.makedirs(f"{output_dir}/{exp_name}", exist_ok=True)
        my_writer = SummaryWriter(f"{output_dir}/{exp_name}")
    else:
        my_writer = None

    if config["name"] is None:
        if config["deep_supervision"]:
            config["name"] = "%s_%s_wDS" % (config["dataset"], config["arch"])
        else:
            config["name"] = "%s_%s_woDS" % (config["dataset"], config["arch"])

    # Override num_classes for BDD100K dataset
    if config["dataset"] == "bdd100k":
        config["num_classes"] = BDD100K_NUM_CLASSES
        if is_main_process(rank):
            print(f"Using BDD100K dataset with {BDD100K_NUM_CLASSES} classes")

    if is_main_process(rank):
        print("-" * 50)
        print(f"Distributed Training: {distributed}")
        print(f"World Size (Total GPUs): {world_size}")
        print(f"Rank: {rank}, Local Rank: {local_rank}")
        print(f"Mixed Precision (AMP): {config['use_amp']}")
        print(f"Gradient Accumulation Steps: {config['grad_accum_steps']}")
        print(
            f"Effective Batch Size: {config['batch_size'] * world_size * config['grad_accum_steps']}"
        )
        print("-" * 50)
        for key in config:
            print("%s: %s" % (key, config[key]))
        print("-" * 50)

        with open(f"{output_dir}/{exp_name}/config.yml", "w") as f:
            yaml.dump(config, f)

    # Define loss function
    if config["loss"] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config["loss"]]().cuda()

    cudnn.benchmark = True

    # Create model
    # load args model_name into model:
    model = load_model(config)
    # Convert BatchNorm to SyncBatchNorm for DDP
    if distributed and config["sync_bn"]:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main_process(rank):
            print("Using Synchronized BatchNorm")

    model = model.cuda()

    # Optionally compile model (PyTorch 2.0+)
    if config["compile_model"] and hasattr(torch, "compile"):
        if is_main_process(rank):
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap model with DDP
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # Set to False for better performance
            gradient_as_bucket_view=False,
        )  # Fixes gradient stride mismatch warning

    # Calculate and print total parameters
    if is_main_process(rank):
        # Get base model for parameter counting
        base_model = model.module if distributed else model
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(
            p.numel() for p in base_model.parameters() if p.requires_grad
        )
        print("-" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("-" * 50)

    # Setup parameter groups with different learning rates
    base_model = model.module if distributed else model
    param_groups = []
    for name, param in base_model.named_parameters():
        if "layer" in name.lower() and "fc" in name.lower():
            param_groups.append(
                {
                    "params": param,
                    "lr": config["kan_lr"],
                    "weight_decay": config["kan_weight_decay"],
                }
            )
        else:
            param_groups.append(
                {
                    "params": param,
                    "lr": config["lr"],
                    "weight_decay": config["weight_decay"],
                }
            )

    # Optimizer
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(param_groups)
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(param_groups)
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            param_groups,
            lr=config["lr"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
        )
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented")

    # Mixed precision scaler
    scaler = GradScaler(enabled=config["use_amp"])

    # Learning rate scheduler
    if config["scheduler"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["min_lr"]
        )
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["factor"],
            patience=config["patience"],
            verbose=is_main_process(rank),
            min_lr=config["min_lr"],
        )
    elif config["scheduler"] == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config["milestones"].split(",")],
            gamma=config["gamma"],
        )
    elif config["scheduler"] == "OneCycleLR":
        # Will be created after dataloader is ready
        scheduler = None
    elif config["scheduler"] == "ConstantLR":
        scheduler = None
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented")

    # Copy training files (main process only)
    if is_main_process(rank):
        shutil.copy2("train_ddp.py", f"{output_dir}/{exp_name}/")
        shutil.copy2("archs.py", f"{output_dir}/{exp_name}/")

    # Resume from checkpoint
    start_epoch = 0
    best_iou = 0
    best_dice = 0

    if config["resume"]:
        checkpoint_path = f"{output_dir}/{exp_name}/checkpoint_best.pth"
        if os.path.exists(checkpoint_path):
            if is_main_process(rank):
                print(f"=> Loading checkpoint from {checkpoint_path}")

            # Map to correct device
            map_location = {"cuda:0": f"cuda:{local_rank}"}
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

            base_model = model.module if distributed else model
            base_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if (
                scheduler is not None
                and checkpoint.get("scheduler_state_dict") is not None
            ):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            best_iou = checkpoint["best_iou"]
            best_dice = checkpoint["best_dice"]

            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            if is_main_process(rank):
                print(
                    f"=> Resumed from epoch {checkpoint['epoch']}, "
                    f"best_iou: {best_iou:.4f}, best_dice: {best_dice:.4f}"
                )

    # Data augmentation
    train_transform = Compose(
        [
            RandomRotate90(),
            albumentations.HorizontalFlip(),
            Resize(config["input_h"], config["input_w"]),
            transforms.Normalize(),
        ]
    )

    val_transform = Compose(
        [
            Resize(config["input_h"], config["input_w"]),
            transforms.Normalize(),
        ]
    )

    # BDD100K dataset paths
    bdd100k_base = "/mnt/ssd-0/M2_internship/bdd100k_seg/bdd100k/seg"

    # Get image IDs
    train_mask_paths = sorted(
        glob(os.path.join(bdd100k_base, "labels", "train", "*.png"))
    )
    train_img_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in train_mask_paths
    ]

    val_mask_paths = sorted(glob(os.path.join(bdd100k_base, "labels", "val", "*.png")))
    val_img_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in val_mask_paths
    ]

    if is_main_process(rank):
        print(
            f"Training samples: {len(train_img_ids)}, Validation samples: {len(val_img_ids)}"
        )

    # Create datasets
    train_dataset = BDD100KDataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(bdd100k_base, "images", "train"),
        mask_dir=os.path.join(bdd100k_base, "labels", "train"),
        img_ext=".jpg",
        mask_ext=".png",
        num_classes=BDD100K_NUM_CLASSES,
        transform=train_transform,
        mask_suffix="_train_id",
    )

    val_dataset = BDD100KDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(bdd100k_base, "images", "val"),
        mask_dir=os.path.join(bdd100k_base, "labels", "val"),
        img_ext=".jpg",
        mask_ext=".png",
        num_classes=BDD100K_NUM_CLASSES,
        transform=val_transform,
        mask_suffix="_train_id",
    )

    # Create samplers for distributed training
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Create data loaders with optimizations
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        prefetch_factor=config["prefetch_factor"],
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
        prefetch_factor=config["prefetch_factor"],
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    # Create OneCycleLR scheduler if selected
    if config["scheduler"] == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"] * 10,
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )

    # Training log
    log = OrderedDict(
        [
            ("epoch", []),
            ("lr", []),
            ("loss", []),
            ("iou", []),
            ("val_loss", []),
            ("val_iou", []),
            ("val_dice", []),
        ]
    )

    # Load existing log if resuming
    log_path = f"{output_dir}/{exp_name}/log.csv"
    if config["resume"] and os.path.exists(log_path) and is_main_process(rank):
        existing_log = pd.read_csv(log_path)
        for key in log.keys():
            log[key] = existing_log[key].tolist()[:start_epoch]
        print(f"=> Loaded existing log with {len(log['epoch'])} entries")

    trigger = 0

    for epoch in range(start_epoch, config["epochs"]):
        # Set epoch for distributed sampler
        if distributed:
            train_sampler.set_epoch(epoch)

        if is_main_process(rank):
            print(f"\nEpoch [{epoch}/{config['epochs']}]")

        # Train
        train_log = train_one_epoch(
            config,
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            epoch,
            rank,
            world_size,
            config["grad_accum_steps"],
        )

        # Validate
        val_log = validate(config, val_loader, model, criterion, rank, world_size)

        # Update scheduler
        if config["scheduler"] == "CosineAnnealingLR":
            scheduler.step()
        elif config["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(val_log["loss"])
        elif config["scheduler"] == "OneCycleLR":
            pass  # OneCycleLR steps per batch, already done

        if is_main_process(rank):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"loss {train_log['loss']:.4f} - iou {train_log['iou']:.4f} - "
                f"val_loss {val_log['loss']:.4f} - val_iou {val_log['iou']:.4f} - "
                f"val_dice {val_log['dice']:.4f} - lr {current_lr:.6f}"
            )

            # Log metrics
            log["epoch"].append(epoch)
            log["lr"].append(current_lr)
            log["loss"].append(train_log["loss"])
            log["iou"].append(train_log["iou"])
            log["val_loss"].append(val_log["loss"])
            log["val_iou"].append(val_log["iou"])
            log["val_dice"].append(val_log["dice"])

            pd.DataFrame(log).to_csv(f"{output_dir}/{exp_name}/log.csv", index=False)

            # TensorBoard logging
            my_writer.add_scalar("train/loss", train_log["loss"], global_step=epoch)
            my_writer.add_scalar("train/iou", train_log["iou"], global_step=epoch)
            my_writer.add_scalar("val/loss", val_log["loss"], global_step=epoch)
            my_writer.add_scalar("val/iou", val_log["iou"], global_step=epoch)
            my_writer.add_scalar("val/dice", val_log["dice"], global_step=epoch)
            my_writer.add_scalar("lr", current_lr, global_step=epoch)

            trigger += 1

            # Get base model for saving
            base_model = model.module if distributed else model

            # Save last checkpoint
            checkpoint_last = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler_state_dict": scaler.state_dict(),
                "best_iou": best_iou,
                "best_dice": best_dice,
                "config": config,
            }
            torch.save(checkpoint_last, f"{output_dir}/{exp_name}/checkpoint_last.pth")

            # Save best model
            if val_log["iou"] > best_iou:
                torch.save(
                    base_model.state_dict(), f"{output_dir}/{exp_name}/model_best.pth"
                )
                best_iou = val_log["iou"]
                best_dice = val_log["dice"]

                checkpoint_best = {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "scaler_state_dict": scaler.state_dict(),
                    "best_iou": best_iou,
                    "best_dice": best_dice,
                    "config": config,
                }
                torch.save(
                    checkpoint_best, f"{output_dir}/{exp_name}/checkpoint_best.pth"
                )

                print(
                    f"=> Saved best model - IoU: {best_iou:.4f}, Dice: {best_dice:.4f}"
                )
                trigger = 0

            # Early stopping
            if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
                print("=> Early stopping")
                break

        # Synchronize all processes
        if distributed:
            dist.barrier()

        torch.cuda.empty_cache()

    cleanup_distributed()


if __name__ == "__main__":
    main()
