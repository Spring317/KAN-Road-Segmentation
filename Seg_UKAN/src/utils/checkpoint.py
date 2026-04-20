"""Checkpoint loading utilities.

Single place that handles all checkpoint format variations:
- Raw ``state_dict`` (old format)
- Full training snapshot with ``model_state_dict`` key (new format)
- ``torch.compile`` artefacts with ``_orig_mod.`` key prefix
"""

import os

import torch
import torch.nn as nn

__all__ = ["load_checkpoint"]


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load *path* into *model*, handling all known checkpoint formats.

    Formats handled automatically:

    1. **Raw state dict** — ``torch.save(model.state_dict(), path)``
    2. **Full training snapshot** — contains ``model_state_dict``, ``epoch``,
       ``optimizer_state_dict``, etc.
    3. **torch.compile artefacts** — keys prefixed with ``_orig_mod.``

    Args:
        model: Uninitialised or randomly-initialised model instance.
        path: Absolute or relative path to the checkpoint file.
        device: Target device for ``map_location``.

    Returns:
        *model* with weights loaded in-place.

    Raises:
        FileNotFoundError: *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # --- unwrap full training snapshots ---
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        epoch = ckpt.get("epoch", "?")
        best_iou = ckpt.get("best_iou", float("nan"))
        print(
            f"  Full checkpoint — epoch {epoch}, "
            f"best_iou {best_iou:.4f}"
            if isinstance(best_iou, float) and not __import__("math").isnan(best_iou)
            else f"  Full checkpoint — epoch {epoch}"
        )
        ckpt = ckpt["model_state_dict"]

    # --- strip torch.compile prefix ---
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}

    # --- load ---
    try:
        model.load_state_dict(ckpt)
        print("  Weights loaded successfully (strict).")
    except RuntimeError as exc:
        print(f"  Strict load failed: {exc}\n  Retrying with strict=False …")
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"  Missing  ({len(missing)}): {missing[:5]}{' …' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected ({len(unexpected)}): {unexpected[:5]}{' …' if len(unexpected) > 5 else ''}")

    return model
