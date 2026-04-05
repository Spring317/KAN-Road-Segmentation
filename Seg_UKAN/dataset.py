import os

import cv2
import numpy as np
import torch
import torch.utils.data

# Optimize OpenCV for faster image loading
cv2.setNumThreads(0)  # Disable OpenCV threading (let PyTorch handle it)
cv2.ocl.setUseOpenCL(False)  # Disable OpenCL for consistency


# BDD100K segmentation class labels and colors
BDD100K_CLASSES = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
    19: 'unknown'
}

BDD100K_COLOR_DICT = {
    0: (0.7, 0.7, 0.7),     # road - gray
    1: (0.9, 0.9, 0.2),     # sidewalk - light yellow
    2: (1.0, 0.4980392156862745, 0.054901960784313725),
    3: (1.0, 0.7333333333333333, 0.47058823529411764),
    4: (0.8, 0.5, 0.1),     # Fence - rust orange
    5: (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    6: (0.325, 0.196, 0.361),
    7: (1.0, 0.596078431372549, 0.5882352941176471),
    8: (0.2, 0.6, 0.2),     # vegetation - green
    9: (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    10: (0.5, 0.7, 1.0),    # sky - light blue
    11: (1.0, 0.0, 0.0),    # person - red
    12: (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    13: (0.0, 0.0, 1.0),    # Car - blue
    14: (0.0, 0.0, 1.0),    # Truck - blue
    15: (0.0, 0.0, 1.0),    # Bus - blue
    16: (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    17: (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    18: (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    19: (0, 0, 0)           # unknown - black
}

BDD100K_NUM_CLASSES = 20


def mask_to_onehot(mask, num_classes=BDD100K_NUM_CLASSES):
    """
    Convert a single-channel class mask to multi-channel one-hot encoded mask.
    
    Args:
        mask: numpy array of shape (H, W) with class labels (0 to num_classes-1)
        num_classes: number of classes
    
    Returns:
        one_hot: numpy array of shape (H, W, num_classes) with one-hot encoding
    """
    h, w = mask.shape[:2]
    one_hot = np.zeros((h, w, num_classes), dtype=np.float32)
    
    for c in range(num_classes):
        one_hot[:, :, c] = (mask == c).astype(np.float32)
    
    return one_hot


def onehot_to_mask(one_hot):
    """
    Convert a multi-channel one-hot encoded mask back to single-channel class mask.
    
    Args:
        one_hot: numpy array of shape (H, W, num_classes) or (num_classes, H, W)
    
    Returns:
        mask: numpy array of shape (H, W) with class labels
    """
    if one_hot.shape[0] < one_hot.shape[-1]:
        # Shape is (num_classes, H, W), transpose to (H, W, num_classes)
        one_hot = one_hot.transpose(1, 2, 0)
    
    return np.argmax(one_hot, axis=-1)


def colorize_mask(mask, color_dict=BDD100K_COLOR_DICT):
    """
    Convert a single-channel class mask to a colored RGB image for visualization.
    
    Args:
        mask: numpy array of shape (H, W) with class labels
        color_dict: dictionary mapping class labels to RGB colors (0-1 range)
    
    Returns:
        colored: numpy array of shape (H, W, 3) with RGB values (0-255)
    """
    if len(mask.shape) > 2:
        mask = np.squeeze(mask)
    
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_dict.items():
        colored[mask == class_id] = [int(c * 255) for c in color]
    
    return colored


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):

            # print(os.path.join(self.mask_dir, str(i),
            #             img_id + self.mask_ext))

            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        if mask.max()<1:
            mask[mask>0] = 1.0

        return img, mask, {'img_id': img_id}


class BDD100KDataset(torch.utils.data.Dataset):
    """
    Dataset class for BDD100K semantic segmentation dataset.
    
    The masks are single-channel images where each pixel value represents a class label (0-19).
    Value 255 in the original masks represents unknown/ignore and is mapped to class 19.
    
    Args:
        img_ids (list): Image ids (filenames without extension).
        img_dir: Image file directory.
        mask_dir: Mask file directory.
        img_ext (str): Image file extension (default: '.jpg').
        mask_ext (str): Mask file extension (default: '.png').
        num_classes (int): Number of classes (default: 20).
        transform (Compose, optional): Compose transforms of albumentations.
        ignore_index (int): Value to use for unknown/ignore pixels (default: 19).
    
    Note:
        Expected structure for BDD100K segmentation:
        bdd100k_seg/
        └── bdd100k/
            └── seg/
                ├── images/
                │   ├── train/
                │   │   ├── image1.jpg
                │   │   ├── image2.jpg
                │   │   └── ...
                │   └── val/
                │       └── ...
                └── labels/
                    ├── train/
                    │   ├── image1.png
                    │   ├── image2.png
                    │   └── ...
                    └── val/
                        └── ...
    """
    
    def __init__(self, img_ids, img_dir, mask_dir, img_ext='.jpg', mask_ext='.png', 
                 num_classes=BDD100K_NUM_CLASSES, transform=None, ignore_index=19, mask_suffix=''):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.ignore_index = ignore_index
        self.mask_suffix = mask_suffix  # e.g., '_train_id' for BDD100K training masks

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Load image (BGR format from cv2)
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask (single channel with class labels)
        # Mask files may have a suffix (e.g., _train_id for BDD100K)
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_suffix + self.mask_ext)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Map 255 (unknown in BDD100K) to ignore_index (usually 19)
        mask[mask == 255] = self.ignore_index
        
        # Apply augmentations if provided
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Convert mask to one-hot encoding (H, W) -> (H, W, num_classes)
        mask_onehot = mask_to_onehot(mask, self.num_classes)
        
        # Normalize image and transpose to (C, H, W)
        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)
        
        # Transpose mask to (num_classes, H, W)
        mask_onehot = mask_onehot.transpose(2, 0, 1)
        
        return img, mask_onehot, {'img_id': img_id}

