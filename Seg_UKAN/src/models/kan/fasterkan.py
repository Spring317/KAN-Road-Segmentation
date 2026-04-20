"""FasterKAN linear layer — CUDA-accelerated RSWAf basis function.

This module contains the production KAN linear implementation that wraps the
compiled ``cuda/faster_ops`` extension.  On CPU it falls back to a pure-PyTorch
loop so that the package can be imported even without the CUDA build (e.g.
during unit testing or CPU-only evaluation).
"""

import torch
import torch.nn as nn
from torch.autograd import Function

# ---------------------------------------------------------------------------
# CUDA extension
# ---------------------------------------------------------------------------
try:
    from cuda import faster_ops  # type: ignore[import]
    _HAS_CUDA_EXT = True
except ImportError:
    _HAS_CUDA_EXT = False


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class RSWAFFunction(Function):
    """Reflectional Switch Weighted Activation Function (RSWAf)."""

    @staticmethod
    def forward(ctx, input, grid, inv_denominator, train_grid, train_inv_denominator):
        if input.is_cuda and _HAS_CUDA_EXT:
            res, th = faster_ops.forward(input, grid, inv_denominator)
        else:
            batchsize, in_feats = input.shape
            gridsize = grid.shape[0]

            grid_expanded = grid.view(gridsize, 1, 1).to(input.device)
            input_expanded = input.view(1, batchsize, in_feats)

            z = torch.tanh((input_expanded - grid_expanded) * inv_denominator)
            r = 1.0 - z * z

            res = r.permute(1, 0, 2).reshape(batchsize, -1)
            th = z.permute(1, 0, 2).reshape(batchsize, -1)

        ctx.save_for_backward(input, grid, inv_denominator, th)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, inv_denominator, th = ctx.saved_tensors
        if input.is_cuda and _HAS_CUDA_EXT:
            grad_input = faster_ops.backward(
                grad_output.contiguous(), th.contiguous(), input, grid, inv_denominator
            )
        else:
            batchsize, in_feats = input.shape
            gridsize = grid.shape[0]

            gx = -2.0 * th * (1.0 - th * th) * grad_output
            gx = gx.view(batchsize, gridsize, in_feats)
            grad_input = gx.sum(dim=1)

        return grad_input, None, None, None, None


# ---------------------------------------------------------------------------
# nn.Module wrappers
# ---------------------------------------------------------------------------

class ReflectionalSwitchFunction(nn.Module):
    """Learnable RSWAf basis — wraps :class:`RSWAFFunction`."""

    def __init__(
        self,
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        inv_denominator: float = 0.5,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.train_grid = torch.tensor(False, dtype=torch.bool)
        self.train_inv_denominator = torch.tensor(False, dtype=torch.bool)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.inv_denominator = nn.Parameter(
            torch.tensor(inv_denominator, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RSWAFFunction.apply(
            x,
            self.grid,
            self.inv_denominator,
            self.train_grid,
            self.train_inv_denominator,
        )


class SplineLinear(nn.Linear):
    """Linear projection with Xavier-uniform initialisation (no bias)."""

    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)


class KANLinear(nn.Module):
    """FasterKAN linear layer using the RSWAf basis function.

    Drop-in replacement for the original KANLinear; uses the CUDA-accelerated
    RSWAf implementation when available and falls back to pure PyTorch otherwise.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        grid_size: Number of RSWAf grid points (``num_grids``).
        grid_range: ``[min, max]`` range for grid initialisation.
        All other args are kept for API compatibility with the original KAN
        interface but are ignored.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,       # kept for API compat
        scale_noise: float = 0.1,    # kept for API compat
        scale_base: float = 1.0,     # kept for API compat
        scale_spline: float = 1.0,   # kept for API compat
        base_activation=nn.SiLU,     # kept for API compat
        grid_eps: float = 0.02,      # kept for API compat
        grid_range: list | None = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]

        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = grid_size

        self.layernorm = nn.LayerNorm(in_features)
        self.rbf = ReflectionalSwitchFunction(
            grid_min=grid_range[0], grid_max=grid_range[1], num_grids=grid_size
        )
        self.spline_linear = SplineLinear(
            in_features * grid_size, out_features, init_scale=0.667
        )

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        x = self.layernorm(x)
        spline_basis = self.rbf(x).view(x.shape[0], -1)
        return self.spline_linear(spline_basis)

    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Stub — prevents crashes when training loops call this."""

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        """Dummy regularisation loss for API compat."""
        l1_fake = self.spline_linear.weight.abs().mean(-1)
        return regularize_activation * l1_fake.sum()
