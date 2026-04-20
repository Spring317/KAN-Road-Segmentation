"""Loss functions for semantic segmentation training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge  # type: ignore
except ImportError:
    lovasz_hinge = None  # type: ignore

__all__ = ["BCEDiceLoss", "LovaszHingeLoss"]


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy + Dice loss.

    ``loss = 0.5 * BCE + Dice``
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        inp = torch.sigmoid(input)
        num = target.size(0)
        inp = inp.view(num, -1)
        tgt = target.view(num, -1)
        intersection = (inp * tgt)
        dice = (2.0 * intersection.sum(1) + smooth) / (inp.sum(1) + tgt.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    """Lovász-Hinge loss (requires the ``LovaszSoftmax`` package)."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if lovasz_hinge is None:
            raise ImportError(
                "LovaszSoftmax is not installed. "
                "Run: pip install git+https://github.com/bermanmaxim/LovaszSoftmax"
            )
        return lovasz_hinge(input.squeeze(1), target.squeeze(1), per_image=True)
