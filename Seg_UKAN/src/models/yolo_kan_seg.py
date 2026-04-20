"""YOLO-KAN hybrid segmentation network.

Combines a YOLOv11 pretrained backbone with a FasterKAN-based
segmentation head for semantic segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import KANBlock

__all__ = ["YOLOKANSeg"]


class KANSegHead(nn.Module):
    """Semantic segmentation head using KANBlock on FPN features."""

    def __init__(self, in_channels_list, num_classes, kan_type, no_kan, target_dim=256):
        super().__init__()
        # Projections to unify dimensions of FPN levels
        self.projs = nn.ModuleList([
            nn.Conv2d(c, target_dim, 1) for c in in_channels_list
        ])
        
        self.kan_block = KANBlock(dim=target_dim, kan_type=kan_type, no_kan=no_kan)
        self.final_conv = nn.Conv2d(target_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x is a list of features from YOLO neck
        target_size = x[0].shape[2:]
        
        feats = []
        for feat, proj in zip(x, self.projs):
            p = proj(feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            feats.append(p)
            
        # Sum FPN features
        out = sum(feats)
        
        B, C, H, W = out.shape
        out_tokens = out.flatten(2).transpose(1, 2)
        out_tokens = self.kan_block(out_tokens, H, W)
        out = out_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        return self.final_conv(out)


class YOLOKANSeg(nn.Module):
    """YOLOv11 backbone with KAN segmentation head.

    Args:
        num_classes: Number of segmentation output channels.
        input_channels: Ignored (kept for compatibility with UKAN factory).
        kan_type: KAN variant to use.
        no_kan: Standard nn.Linear instead of KAN.
        yolo_weights: Pretrained YOLOv11 weights path or format.
        freeze_backbone: Freeze YOLO layers, train only KAN head.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        deep_supervision: bool = False,
        kan_type: str = "FasterKAN",
        no_kan: bool = False,
        yolo_weights: str = "yolo11m-seg.pt",
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        from ultralytics import YOLO
        
        self.num_classes = num_classes
        self.kan_type = kan_type
        self.freeze_backbone = freeze_backbone
        
        yolo = YOLO(yolo_weights)
        self.yolo = yolo.model.model
        
        original_head = self.yolo[-1]
        
        # Dry run to find FPN feature dimensions entering the head
        in_channels_list = []
        try:
            device = next(self.yolo.parameters()).device
            dummy = torch.randn(1, 3, 256, 256).to(device)
            def hook(module, input):
                for t in input[0]:
                    in_channels_list.append(t.shape[1])
            h = original_head.register_forward_pre_hook(hook)
            with torch.no_grad():
                self.yolo(dummy)
            h.remove()
        except Exception:
            # Fallback to common dimensions if dry run fails
            in_channels_list = [256, 512, 1024]
            
        target_dim = 256
        kan_head = KANSegHead(in_channels_list, num_classes, kan_type, no_kan, target_dim=target_dim)
        
        # Copy Ultralytics structural attributes to keep forward pass working
        kan_head.f = getattr(original_head, 'f', -1)
        kan_head.i = getattr(original_head, 'i', -1)
        
        self.yolo[-1] = kan_head
        self.kan_head = self.yolo[-1]
        
        if freeze_backbone:
            for name, p in self.yolo.named_parameters():
                if id(p) not in [id(hp) for hp in self.kan_head.parameters()]:
                    p.requires_grad = False

    def forward(self, x):
        input_shape = x.shape[2:]
        out = self.yolo(x)
        
        if isinstance(out, tuple):
            out = out[0]
        
        if out.shape[2:] != input_shape:
            out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
            
        return out

    def print_parameter_stats(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.kan_head.parameters() if p.requires_grad)
        backbone_params = total_params - head_params
        
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)

        print(f"==========================================")
        print(f"YOLOKANSeg Parameter Summary (Trainable):")
        print(f"Backbone (YOLO):     {backbone_params:,}")
        print(f"Head (KAN):          {head_params:,}")
        print(f"Total Trainable:     {total_params:,}")
        print(f"Frozen Parameters:   {frozen_params:,}")
        print(f"KAN Variant:         {self.kan_type}")
        print(f"==========================================")

