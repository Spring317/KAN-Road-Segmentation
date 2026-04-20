"""Segmentation evaluation metrics."""

import numpy as np
import torch

__all__ = ["iou_score", "dice_coef"]


def iou_score(
    output: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, float]:
    """Compute IoU, Dice, and (stub) HD95 for a batch.

    Args:
        output: Raw logits ``(B, C, H, W)``.
        target: One-hot ground truth ``(B, C, H, W)``.

    Returns:
        ``(iou, dice, hd95)`` — hd95 is 0.0 (stub).
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    return iou, dice, 0.0


def dice_coef(output: torch.Tensor, target: torch.Tensor) -> float:
    """Scalar Dice coefficient for a batch."""
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)
