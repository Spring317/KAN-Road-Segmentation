import argparse
import os
import random
import shutil
from collections import OrderedDict
from glob import glob
from pathlib import Path

import albumentations
import albumentations.augmentations as transforms
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations import RandomRotate90, Resize
from albumentations.core.composition import Compose
from tensorboardX import SummaryWriter
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from ultralytics import YOLO
import difflib

import archs
import losses
from dataset import BDD100KDataset, BDD100K_NUM_CLASSES
from metrics import iou_score
from utils import AverageMeter, str2bool
from yolo_data_prep import BDD100KYOLOPreparer

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__ + ["BCEWithLogitsLoss"]


def resolve_yolo_weights(weight_arg: str, project_root: Path) -> str:
    if not weight_arg or not str(weight_arg).strip():
        raise ValueError("YOLO weights argument cannot be empty.")

    w = str(weight_arg).strip()
    aliases = {
        "yolov11n-seg.pt": "yolo11n-seg.pt",
        "yolov11s-seg.pt": "yolo11s-seg.pt",
        "yolov11m-seg.pt": "yolo11m-seg.pt",
        "yolov11l-seg.pt": "yolo11l-seg.pt",
        "yolov11x-seg.pt": "yolo11x-seg.pt",
    }
    w = aliases.get(w.lower(), w)

    # 1) absolute/relative file path exists
    p = Path(w).expanduser()
    if p.is_file():
        return str(p.resolve())

    # 2) resolve relative to project root
    p2 = (project_root / w).resolve()
    if p2.is_file():
        return str(p2)

    # 3) allow official Ultralytics names (auto-download)
    official = {f"yolo11{sz}-seg.pt" for sz in ("n", "s", "m", "l", "x")}
    if w in official:
        return w

    # friendly suggestion
    candidates = sorted(official | set(aliases.keys()))
    suggestion = difflib.get_close_matches(w, candidates, n=1)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    raise FileNotFoundError(
        f"YOLO weights not found: '{weight_arg}'. "
        f"Provide an existing .pt path or official model name.{hint}"
    )


def list_type(s):
    return [int(a) for a in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default=None, help="experiment name")
    parser.add_argument("--model_name", default="UKAN", choices=["UKAN", "yolo"])
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("--dataseed", default=2981, type=int)

    # Model
    parser.add_argument("--arch", "-a", default="UKAN", choices=ARCH_NAMES)
    parser.add_argument("--deep_supervision", default=False, type=str2bool)
    parser.add_argument("--input_channels", default=3, type=int)
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--input_w", default=256, type=int)
    parser.add_argument("--input_h", default=192, type=int)
    parser.add_argument("--input_list", type=list_type, default=[64, 128, 256])
    parser.add_argument("--no_kan", action="store_true")
    parser.add_argument("--kan_type", default="FasterKAN", choices=["FasterKAN", "ReLU", "HardSwish", "PWLO", "TeLU"])

    # Loss
    parser.add_argument("--loss", default="BCEDiceLoss", choices=LOSS_NAMES)

    # Dataset
    parser.add_argument("--dataset", default="bdd100k")
    parser.add_argument("--data_dir", default="inputs")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument(
        "--bdd100k_base", default="/mnt/ssd-0/M2_internship/bdd100k_seg/bdd100k/seg"
    )

    # Optimizer
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD", "AdamW"])
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--nesterov", default=False, type=str2bool)
    parser.add_argument("--kan_lr", default=1e-2, type=float)
    parser.add_argument("--kan_weight_decay", default=1e-4, type=float)

    # Scheduler
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
    parser.add_argument("--early_stopping", default=-1, type=int)

    parser.add_argument("--cfg", type=str, metavar="FILE")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--resume", default=False, type=str2bool)

    # DDP/AMP
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--use_amp", default=True, type=str2bool)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--sync_bn", default=True, type=str2bool)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--compile_model", default=False, type=str2bool)

    # YOLO specific
    parser.add_argument("--yolo_weights", default="yolov11m-seg.pt")
    parser.add_argument("--yolo_data", default=None, help="YOLO dataset YAML path")
    parser.add_argument("--yolo_rebuild_labels", default=False, type=str2bool)
    parser.add_argument(
        "--yolo_task", default="segment", choices=["detect", "segment", "pose", "obb"]
    )
    return vars(parser.parse_args())


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
        return True, rank, world_size, local_rank
    if torch.cuda.is_available():
        return False, 0, 1, 0
    raise RuntimeError("CUDA not available")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def reduce_tensor(tensor, world_size):
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def seed_torch(seed=1029, rank=0):
    seed = seed + rank
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_base_model(model, distributed):
    return model.module if distributed else model


def load_ukan_model(config) -> nn.Module:
    return archs.__dict__[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config["deep_supervision"],
        embed_dims=config["input_list"],
        no_kan=config.get("no_kan", False),
        kan_type=config.get("kan_type", "FasterKAN"),
    )


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
    scheduler=None,
):
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}
    model.train()
    pbar = (
        tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        if is_main_process(rank)
        else None
    )
    optimizer.zero_grad()

    for batch_idx, (inp, target, _) in enumerate(train_loader):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with amp.autocast("cuda", enabled=config["use_amp"]):
            if config["deep_supervision"]:
                outputs = model(inp)
                loss = sum(criterion(output, target) for output in outputs) / len(
                    outputs
                )
                iou, _, _ = iou_score(outputs[-1], target)
            else:
                output = model(inp)
                loss = criterion(output, target)
                iou, _, _ = iou_score(output, target)

            loss = loss / config["grad_accum_steps"]

        scaler.scale(loss).backward()

        if (batch_idx + 1) % config["grad_accum_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if config["scheduler"] == "OneCycleLR" and scheduler is not None:
                scheduler.step()

        loss_reduced = reduce_tensor(
            loss.detach() * config["grad_accum_steps"], world_size
        )
        iou_reduced = reduce_tensor(torch.tensor(iou, device=inp.device), world_size)

        avg_meters["loss"].update(loss_reduced.item(), inp.size(0))
        avg_meters["iou"].update(iou_reduced.item(), inp.size(0))

        if pbar is not None:
            pbar.set_postfix(
                OrderedDict(
                    loss=f"{avg_meters['loss'].avg:.4f}",
                    iou=f"{avg_meters['iou'].avg:.4f}",
                )
            )
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return OrderedDict(loss=avg_meters["loss"].avg, iou=avg_meters["iou"].avg)


def validate(config, val_loader, model, criterion, rank, world_size):
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter(), "dice": AverageMeter()}
    model.eval()
    pbar = (
        tqdm(total=len(val_loader), desc="Validation")
        if is_main_process(rank)
        else None
    )

    with torch.no_grad():
        for inp, target, _ in val_loader:
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with amp.autocast("cuda", enabled=config["use_amp"]):
                if config["deep_supervision"]:
                    outputs = model(inp)
                    loss = sum(criterion(output, target) for output in outputs) / len(
                        outputs
                    )
                    iou, dice, _ = iou_score(outputs[-1], target)
                else:
                    output = model(inp)
                    loss = criterion(output, target)
                    iou, dice, _ = iou_score(output, target)

            loss_reduced = reduce_tensor(loss.detach(), world_size)
            iou_reduced = reduce_tensor(
                torch.tensor(iou, device=inp.device), world_size
            )
            dice_reduced = reduce_tensor(
                torch.tensor(dice, device=inp.device), world_size
            )

            avg_meters["loss"].update(loss_reduced.item(), inp.size(0))
            avg_meters["iou"].update(iou_reduced.item(), inp.size(0))
            avg_meters["dice"].update(dice_reduced.item(), inp.size(0))

            if pbar is not None:
                pbar.set_postfix(
                    OrderedDict(
                        loss=f"{avg_meters['loss'].avg:.4f}",
                        iou=f"{avg_meters['iou'].avg:.4f}",
                        dice=f"{avg_meters['dice'].avg:.4f}",
                    )
                )
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return OrderedDict(
        loss=avg_meters["loss"].avg,
        iou=avg_meters["iou"].avg,
        dice=avg_meters["dice"].avg,
    )


def make_bdd100k_yolo_yaml(config, exp_dir):
    """
    Auto-create YOLO YAML using your existing bdd100k_seg structure.
    NOTE: YOLO expects labels in YOLO format (.txt), not semantic PNG train_id masks.
    """
    yolo_yaml = Path(exp_dir) / "bdd100k_yolo.yaml"
    train_images = os.path.join(config["bdd100k_base"], "images", "train")
    val_images = os.path.join(config["bdd100k_base"], "images", "val")

    # class names as numeric strings
    names = [str(i) for i in range(config["num_classes"])]

    data = {
        "path": "/",
        "train": train_images,
        "val": val_images,
        "names": names,
    }
    with open(yolo_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(yolo_yaml)


def run_yolo_training(config, exp_dir):
    project_root = Path(__file__).resolve().parent
    weights = resolve_yolo_weights(config["yolo_weights"], project_root)
    model = YOLO(weights)
    if config["yolo_data"]:
        data_yaml = config["yolo_data"]
    else:
        preparer = BDD100KYOLOPreparer(
            bdd100k_base=config["bdd100k_base"],
            num_classes=config["num_classes"],
            rebuild_labels=config["yolo_rebuild_labels"],
            ignore_index=config["num_classes"] - 1,
        )
        data_yaml = preparer.prepare(exp_dir)

    model.train(
        task=config["yolo_task"],
        data=data_yaml,
        epochs=config["epochs"],
        batch=config["batch_size"],
        imgsz=[config["input_h"], config["input_w"]],
        project=config["output_dir"],
        name=config["name"],
        workers=config["num_workers"],
        lr0=config["lr"],
        optimizer=config["optimizer"],
        device=0 if torch.cuda.is_available() else "cpu",
    )


def main():
    config = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    seed_torch(seed=config["dataseed"], rank=rank)
    if config["dataset"] == "bdd100k":
        seed_torch(seed=config["dataseed"], rank=rank)  # different seed for data split
    if config["name"] is None:
        config["name"] = f"{config['dataset']}_{config['model_name']}"
    exp_name = config["name"]
    exp_dir = os.path.join(config["output_dir"], exp_name)

    if is_main_process(rank):
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "config.yml"), "w") as f:
            yaml.safe_dump(config, f)

    # --- YOLO branch ---
    if config["model_name"].lower() == "yolo":
        if distributed:
            if is_main_process(rank):
                print("YOLO branch must be run without torchrun in this script.")
            cleanup_distributed()
            return
        run_yolo_training(config, exp_dir)
        cleanup_distributed()
        return

    # --- UKAN branch ---
    if config["dataset"] == "bdd100k":
        config["num_classes"] = BDD100K_NUM_CLASSES

    writer = SummaryWriter(exp_dir) if is_main_process(rank) else None

    if config["loss"] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config["loss"]]().cuda()

    model = load_ukan_model(config)
    if distributed and config["sync_bn"]:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    if config["compile_model"] and hasattr(torch, "compile"):
        model = torch.compile(model)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    base_model = get_base_model(model, distributed)
    param_groups = []
    for name, p in base_model.named_parameters():
        if "layer" in name.lower() and "fc" in name.lower():
            param_groups.append(
                {
                    "params": p,
                    "lr": config["kan_lr"],
                    "weight_decay": config["kan_weight_decay"],
                }
            )
        else:
            param_groups.append(
                {
                    "params": p,
                    "lr": config["lr"],
                    "weight_decay": config["weight_decay"],
                }
            )

    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(param_groups)
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(param_groups)
    else:
        optimizer = optim.SGD(
            param_groups,
            lr=config["lr"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
        )

    scaler = amp.GradScaler("cuda", enabled=config["use_amp"])

    if config["scheduler"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["min_lr"]
        )
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["factor"],
            patience=config["patience"],
            min_lr=config["min_lr"],
        )
    elif config["scheduler"] == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config["milestones"].split(",")],
            gamma=config["gamma"],
        )
    elif config["scheduler"] == "OneCycleLR":
        scheduler = None
    else:
        scheduler = None

    if is_main_process(rank):
        shutil.copy2("train_ddp.py", os.path.join(exp_dir, "train_ddp.py"))
        shutil.copy2("archs.py", os.path.join(exp_dir, "archs.py"))

    train_transform = Compose(
        [
            RandomRotate90(),
            albumentations.HorizontalFlip(),
            Resize(config["input_h"], config["input_w"]),
            transforms.Normalize(),
        ]
    )
    val_transform = Compose(
        [Resize(config["input_h"], config["input_w"]), transforms.Normalize()]
    )

    bdd = config["bdd100k_base"]
    train_img_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(bdd, "labels", "train", "*.png")))
    ]
    val_img_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(bdd, "labels", "val", "*.png")))
    ]

    train_dataset = BDD100KDataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(bdd, "images", "train"),
        mask_dir=os.path.join(bdd, "labels", "train"),
        img_ext=".jpg",
        mask_ext=".png",
        num_classes=BDD100K_NUM_CLASSES,
        transform=train_transform,
        mask_suffix="_train_id",
    )
    val_dataset = BDD100KDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(bdd, "images", "val"),
        mask_dir=os.path.join(bdd, "labels", "val"),
        img_ext=".jpg",
        mask_ext=".png",
        num_classes=BDD100K_NUM_CLASSES,
        transform=val_transform,
        mask_suffix="_train_id",
    )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if distributed else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        prefetch_factor=config["prefetch_factor"]
        if config["num_workers"] > 0
        else None,
        persistent_workers=(config["num_workers"] > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
        prefetch_factor=config["prefetch_factor"]
        if config["num_workers"] > 0
        else None,
        persistent_workers=(config["num_workers"] > 0),
    )

    if config["scheduler"] == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"] * 10,
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )

    best_iou = 0.0
    trigger = 0
    log = OrderedDict(
        epoch=[], lr=[], loss=[], iou=[], val_loss=[], val_iou=[], val_dice=[]
    )

    for epoch in range(config["epochs"]):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

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
            scheduler=scheduler,
        )
        val_log = validate(config, val_loader, model, criterion, rank, world_size)

        if config["scheduler"] == "CosineAnnealingLR" and scheduler is not None:
            scheduler.step()
        elif config["scheduler"] == "ReduceLROnPlateau" and scheduler is not None:
            scheduler.step(val_log["loss"])
        elif config["scheduler"] == "MultiStepLR" and scheduler is not None:
            scheduler.step()

        if is_main_process(rank):
            lr_now = optimizer.param_groups[0]["lr"]
            log["epoch"].append(epoch)
            log["lr"].append(lr_now)
            log["loss"].append(train_log["loss"])
            log["iou"].append(train_log["iou"])
            log["val_loss"].append(val_log["loss"])
            log["val_iou"].append(val_log["iou"])
            log["val_dice"].append(val_log["dice"])
            pd.DataFrame(log).to_csv(os.path.join(exp_dir, "log.csv"), index=False)

            if writer is not None:
                writer.add_scalar("train/loss", train_log["loss"], epoch)
                writer.add_scalar("train/iou", train_log["iou"], epoch)
                writer.add_scalar("val/loss", val_log["loss"], epoch)
                writer.add_scalar("val/iou", val_log["iou"], epoch)
                writer.add_scalar("val/dice", val_log["dice"], epoch)
                writer.add_scalar("lr", lr_now, epoch)

            base_model = get_base_model(model, distributed)
            checkpoint_last = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler_state_dict": scaler.state_dict(),
                "best_iou": best_iou,
                "config": config,
            }
            torch.save(checkpoint_last, os.path.join(exp_dir, "checkpoint_last.pth"))

            if val_log["iou"] > best_iou:
                best_iou = val_log["iou"]
                torch.save(
                    base_model.state_dict(), os.path.join(exp_dir, "model_best.pth")
                )
                torch.save(
                    checkpoint_last, os.path.join(exp_dir, "checkpoint_best.pth")
                )
                trigger = 0
            else:
                trigger += 1

            if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
                break

        if distributed:
            dist.barrier()
        torch.cuda.empty_cache()

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
