"""BDD100K segmentation dataset.

Contains dataset constants and the :class:`BDD100KDataset` PyTorch Dataset.
The original generic :class:`Dataset` class is preserved in the legacy shim
``dataset.py`` and is not re-exported here.
"""

import os

import cv2
import numpy as np
import torch
import torch.utils.data

from src.data.transforms import mask_to_onehot

# Disable OpenCV internal threading (let PyTorch DataLoader handle parallelism)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

__all__ = [
    "BDD100K_CLASSES",
    "BDD100K_COLOR_DICT",
    "BDD100K_NUM_CLASSES",
    "BDD100KDataset",
]

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

BDD100K_CLASSES: dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    19: "unknown",
}

BDD100K_COLOR_DICT: dict[int, tuple] = {
    0: (0.7, 0.7, 0.7),
    1: (0.9, 0.9, 0.2),
    2: (1.0, 0.4980392156862745, 0.054901960784313725),
    3: (1.0, 0.7333333333333333, 0.47058823529411764),
    4: (0.8, 0.5, 0.1),
    5: (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    6: (0.325, 0.196, 0.361),
    7: (1.0, 0.596078431372549, 0.5882352941176471),
    8: (0.2, 0.6, 0.2),
    9: (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    10: (0.5, 0.7, 1.0),
    11: (1.0, 0.0, 0.0),
    12: (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    13: (0.0, 0.0, 1.0),
    14: (0.0, 0.0, 1.0),
    15: (0.0, 0.0, 1.0),
    16: (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    17: (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    18: (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    19: (0, 0, 0),
}

BDD100K_NUM_CLASSES: int = 20


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BDD100KDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for the BDD100K semantic segmentation split.

    Masks are single-channel PNG images where each pixel value is a class
    label ``[0, 19]``.  Pixel value 255 (BDD100K "ignore") is remapped to
    ``ignore_index`` (default 19 — the *unknown* class).

    Expected structure::

        <data_path>/          (the ``seg`` directory)
        ├── images/
        │   ├── train/*.jpg
        │   └── val/*.jpg
        └── labels/
            ├── train/*_train_id.png
            └── val/*_train_id.png

    Args:
        img_ids: List of image IDs (filenames without extension).
        img_dir: Directory containing ``*.jpg`` images.
        mask_dir: Directory containing mask PNGs.
        img_ext: Image file extension (default ``".jpg"``).
        mask_ext: Mask file extension (default ``".png"``).
        num_classes: Number of classes (default 20).
        transform: ``albumentations.Compose`` transform applied jointly to
            image and mask.
        ignore_index: Class to assign to BDD100K ignore pixels (default 19).
        mask_suffix: Suffix appended to ``img_id`` when building the mask
            path, e.g. ``"_train_id"`` for BDD100K.
    """

    def __init__(
        self,
        img_ids: list,
        img_dir: str,
        mask_dir: str,
        img_ext: str = ".jpg",
        mask_ext: str = ".png",
        num_classes: int = BDD100K_NUM_CLASSES,
        transform=None,
        ignore_index: int = 19,
        mask_suffix: str = "",
    ):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.ignore_index = ignore_index
        self.mask_suffix = mask_suffix

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(
            self.mask_dir, img_id + self.mask_suffix + self.mask_ext
        )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask[mask == 255] = self.ignore_index

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        mask_onehot = mask_to_onehot(mask, self.num_classes)  # (H, W, C)

        img = img.astype("float32") / 255.0
        img = img.transpose(2, 0, 1)          # (C, H, W)
        mask_onehot = mask_onehot.transpose(2, 0, 1)  # (C, H, W)

        return img, mask_onehot, {"img_id": img_id}
