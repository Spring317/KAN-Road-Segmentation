"""Deterministic seeding utilities."""

import os
import random

import numpy as np
import torch


def seed_torch(seed: int = 1029, rank: int = 0) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Base seed value.
        rank: DDP rank — added to *seed* so each rank gets a unique but
            deterministic seed.
    """
    seed = seed + rank
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
