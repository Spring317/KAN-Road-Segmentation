"""Mask transform utilities.

Provides pure functions operating on NumPy arrays — no dataset or dataset-
specific constants are imported here (Interface Segregation Principle).
"""

import numpy as np

__all__ = ["mask_to_onehot", "onehot_to_mask", "colorize_mask"]


def mask_to_onehot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a single-channel class mask to a multi-channel one-hot array.

    Args:
        mask: ``(H, W)`` array with integer class labels ``[0, num_classes)``.
        num_classes: Total number of classes.

    Returns:
        ``(H, W, num_classes)`` float32 one-hot array.
    """
    h, w = mask.shape[:2]
    one_hot = np.zeros((h, w, num_classes), dtype=np.float32)
    for c in range(num_classes):
        one_hot[:, :, c] = (mask == c).astype(np.float32)
    return one_hot


def onehot_to_mask(one_hot: np.ndarray) -> np.ndarray:
    """Convert a multi-channel one-hot array back to a single-channel class mask.

    Args:
        one_hot: ``(H, W, C)`` or ``(C, H, W)`` array.

    Returns:
        ``(H, W)`` array with argmax class labels.
    """
    if one_hot.shape[0] < one_hot.shape[-1]:
        one_hot = one_hot.transpose(1, 2, 0)
    return np.argmax(one_hot, axis=-1)


def colorize_mask(
    mask: np.ndarray, color_dict: dict
) -> np.ndarray:
    """Render a class-label mask as an RGB image.

    Args:
        mask: ``(H, W)`` array with integer class labels.
        color_dict: ``{class_id: (r, g, b)}`` with values in ``[0, 1]``.

    Returns:
        ``(H, W, 3)`` uint8 RGB array.
    """
    if len(mask.shape) > 2:
        mask = np.squeeze(mask)
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_dict.items():
        colored[mask == class_id] = [int(c * 255) for c in color]
    return colored
