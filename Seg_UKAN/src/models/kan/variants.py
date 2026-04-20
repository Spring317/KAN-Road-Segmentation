"""KAN activation variant implementations.

Each class follows the same ``(in_features, out_features, grid_size, …)``
constructor signature as :class:`~src.models.kan.fasterkan.KANLinear`
(Liskov Substitution Principle) so they can be swapped transparently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TeLU",
    "ReLUKANLinear",
    "HardSwishKANLinear",
    "PWLOKANLinear",
    "TeLUKANLinear",
]

# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

class TeLU(nn.Module):
    """Hyperbolic Tangent Exponential Linear Unit.

    ``f(x) = x · tanh(exp(x))``
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.exp(x))


# ---------------------------------------------------------------------------
# KAN variant linears
# ---------------------------------------------------------------------------

class ReLUKANLinear(nn.Module):
    """ReLU-KAN — piecewise-linear basis via ReLU for maximal inference speed.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        grid_size: Number of grid knots.
        grid_range: ``[min, max]`` of the grid.
        All other args kept for API compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list | None = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.layernorm = nn.LayerNorm(in_features)
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.linear = nn.Linear(in_features * grid_size, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        x = self.layernorm(x)
        basis = F.relu(x.unsqueeze(-1) - self.grid)
        basis = basis.view(x.shape[0], -1)
        return self.linear(basis)


class HardSwishKANLinear(nn.Module):
    """Hard-Swish KAN — smooth piecewise-linear basis preventing dying neurons.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        grid_size: Number of grid knots.
        grid_range: ``[min, max]`` of the grid.
        All other args kept for API compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list | None = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.layernorm = nn.LayerNorm(in_features)
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.linear = nn.Linear(in_features * grid_size, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        x = self.layernorm(x)
        basis = F.hardswish(x.unsqueeze(-1) - self.grid)
        basis = basis.view(x.shape[0], -1)
        return self.linear(basis)


class PWLOKANLinear(nn.Module):
    """Piecewise Linear Optimisation KAN (PWLO-KAN).

    Uses lookup-table embeddings (``y = ax + b`` per segment) for efficient
    shift-and-add inference.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        grid_size: Number of piecewise-linear segments.
        grid_range: ``[min, max]`` of the input domain.
        All other args kept for API compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list | None = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]

        self.in_features = in_features
        self.out_features = out_features
        self.num_segments = grid_size

        self.layernorm = nn.LayerNorm(in_features)
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.step = (self.grid_max - self.grid_min) / self.num_segments

        # LUT as embedding — maximises execution efficiency
        self.a_embed = nn.Embedding(in_features * self.num_segments, out_features)
        self.b_embed = nn.Embedding(in_features * self.num_segments, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.a_embed.weight)
        nn.init.zeros_(self.b_embed.weight)

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        x = self.layernorm(x)
        float_idx = (x - self.grid_min) / self.step
        segment_idx = torch.clamp(float_idx.long(), 0, self.num_segments - 1)

        feature_offsets = (
            torch.arange(self.in_features, device=x.device).unsqueeze(0)
            * self.num_segments
        )
        global_idx = segment_idx + feature_offsets

        a_vals = self.a_embed(global_idx)  # (B, in_features, out_features)
        b_vals = self.b_embed(global_idx)

        return (a_vals * x.unsqueeze(-1) + b_vals).sum(dim=1)


class TeLUKANLinear(nn.Module):
    """TeLU-KAN — near-linear basis with stable gradients.

    Uses the :class:`TeLU` activation ``f(x) = x · tanh(exp(x))``.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        grid_size: Number of grid knots.
        grid_range: ``[min, max]`` of the grid.
        All other args kept for API compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list | None = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.layernorm = nn.LayerNorm(in_features)
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.telu = TeLU()
        self.linear = nn.Linear(in_features * grid_size, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        x = self.layernorm(x)
        basis = self.telu(x.unsqueeze(-1) - self.grid)
        basis = basis.view(x.shape[0], -1)
        return self.linear(basis)
