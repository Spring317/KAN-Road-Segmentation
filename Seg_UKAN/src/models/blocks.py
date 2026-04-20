"""Building blocks shared across UKAN encoder/decoder stages.

Each class has a single responsibility (SRP):
- :class:`DWConv` — depthwise convolution for spatial mixing
- :class:`DW_bn_relu` — depthwise conv + BN + ReLU
- :class:`PatchEmbed` — image-to-token patch embedding
- :class:`ConvLayer` — double conv encoder block
- :class:`D_ConvLayer` — double conv decoder block
- :class:`KANLayer` — KAN MLP with spatial depthwise convs
- :class:`KANBlock` — KAN transformer block (norm + KANLayer + skip)
"""

import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from src.models.kan import get_kan_linear

__all__ = [
    "DWConv",
    "DW_bn_relu",
    "PatchEmbed",
    "ConvLayer",
    "D_ConvLayer",
    "KANLayer",
    "KANBlock",
]


# ---------------------------------------------------------------------------
# Weight initialisation helper
# ---------------------------------------------------------------------------

def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


# ---------------------------------------------------------------------------
# Spatial mixing primitives
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    """Depthwise convolution for token-space spatial mixing."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class DW_bn_relu(nn.Module):
    """Depthwise Conv → BN → ReLU block for token-space spatial mixing."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image → patch tokens via a strided Conv2d projection."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# Conv encoder/decoder blocks
# ---------------------------------------------------------------------------

class ConvLayer(nn.Module):
    """Double Conv encoder block: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class D_ConvLayer(nn.Module):
    """Double Conv decoder block: in→in conv then in→out conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# KAN blocks
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    """KAN MLP with three FC stages interleaved with depthwise convolutions.

    Args:
        in_features: Input token dimension.
        hidden_features: Hidden dimension (defaults to ``in_features``).
        out_features: Output dimension (defaults to ``in_features``).
        drop: Dropout rate.
        no_kan: If ``True``, replace KAN linears with standard ``nn.Linear``.
        kan_type: Which KAN variant to use (see :func:`~src.models.kan.get_kan_linear`).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
        no_kan: bool = False,
        kan_type: str = "FasterKAN",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 8

        if not no_kan:
            kan_class = get_kan_linear(kan_type)
            self.fc1 = kan_class(in_features, hidden_features, grid_size=grid_size)
            self.fc2 = kan_class(hidden_features, hidden_features, grid_size=grid_size)
            self.fc3 = kan_class(hidden_features, out_features, grid_size=grid_size)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)

        x = self.fc2(x.reshape(B * N, C)).reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)

        x = self.fc3(x.reshape(B * N, C)).reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x


class KANBlock(nn.Module):
    """Transformer-style block: LayerNorm → KANLayer → skip connection."""

    def __init__(
        self,
        dim: int,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        no_kan: bool = False,
        kan_type: str = "FasterKAN",
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.layer = KANLayer(
            in_features=dim,
            hidden_features=int(dim),
            act_layer=act_layer,
            drop=drop,
            no_kan=no_kan,
            kan_type=kan_type,
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return x + self.drop_path(self.layer(self.norm2(x), H, W))
