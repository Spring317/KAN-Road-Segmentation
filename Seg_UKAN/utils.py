# utils.py — compatibility shim
import argparse

import torch.nn as nn

from src.utils.meters import AverageMeter  # noqa: F401
from src.utils.seed import seed_torch  # noqa: F401
from src.utils.checkpoint import load_checkpoint  # noqa: F401


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform (kept for legacy compat)."""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
