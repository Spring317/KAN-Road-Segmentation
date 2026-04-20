"""U-KAN segmentation network.

Implements the full UKAN encoder-bottleneck-decoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import (
    ConvLayer,
    D_ConvLayer,
    KANBlock,
    PatchEmbed,
)

__all__ = ["UKAN"]


class UKAN(nn.Module):
    """U-shaped KAN segmentation network.

    Args:
        num_classes: Number of segmentation output channels.
        input_channels: Number of input image channels (default: 3).
        deep_supervision: Enable auxiliary outputs (not used in current impl).
        img_size: Spatial size of the input (single int, assumed square).
        embed_dims: Channel dims for the three encoder stages ``[C1, C2, C3]``.
        no_kan: Replace KAN linears with standard ``nn.Linear`` (ablation).
        drop_rate: Token dropout rate.
        drop_path_rate: Stochastic depth drop rate.
        norm_layer: Normalisation layer class.
        depths: Number of KANBlocks per KAN stage.
        kan_type: KAN variant — see :func:`~src.models.kan.get_kan_linear`.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        deep_supervision: bool = False,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dims: list | None = None,
        no_kan: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        depths: list | None = None,
        kan_type: str = "FasterKAN",
        **kwargs,
    ):
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 256]
        if depths is None:
            depths = [1, 1, 1]

        kan_input_dim = embed_dims[0]

        # --- Encoder (CNN stages) ---
        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        # --- Layer norms ---
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _block_kw = dict(drop=drop_rate, norm_layer=norm_layer, no_kan=no_kan, kan_type=kan_type)

        # --- KAN encoder blocks ---
        self.block1 = nn.ModuleList([KANBlock(dim=embed_dims[1], drop_path=dpr[0], **_block_kw)])
        self.block2 = nn.ModuleList([KANBlock(dim=embed_dims[2], drop_path=dpr[1], **_block_kw)])

        # --- KAN decoder blocks ---
        self.dblock1 = nn.ModuleList([KANBlock(dim=embed_dims[1], drop_path=dpr[0], **_block_kw)])
        self.dblock2 = nn.ModuleList([KANBlock(dim=embed_dims[0], drop_path=dpr[1], **_block_kw)])

        # --- Patch embeddings ---
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1],
        )
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2],
        )

        # --- Decoder conv stages ---
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # ── Encoder CNN ──────────────────────────────────────────────────────
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); t3 = out

        # ── KAN stage 4 (encoder) ────────────────────────────────────────────
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous(); t4 = out

        # ── Bottleneck ───────────────────────────────────────────────────────
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # ── Decoder ─────────────────────────────────────────────────────────
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode="bilinear"))

        return self.final(out)
