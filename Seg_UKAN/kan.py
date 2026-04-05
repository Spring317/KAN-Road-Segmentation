import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Ensure the CUDA extension is compiled and available in the 'cuda' folder
try:
    from cuda import faster_ops
except ImportError:
    raise ImportError(
        "Could not import 'faster_ops'. Please ensure the CUDA extension "
        "is compiled correctly and the 'cuda' folder is accessible."
    )


class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator, train_grid, train_inv_denominator):
        res, th = faster_ops.forward(input, grid, inv_denominator)
        ctx.save_for_backward(input, grid, inv_denominator, th)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, inv_denominator, th = ctx.saved_tensors
        grad_input = faster_ops.backward(
            grad_output.contiguous(), th.contiguous(), input, grid, inv_denominator
        )
        return grad_input, None, None, None, None


class ReflectionalSwitchFunction(nn.Module):
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
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.inv_denominator = torch.nn.Parameter(
            torch.tensor(inv_denominator, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        return RSWAFFunction.apply(
            x,
            self.grid,
            self.inv_denominator,
            self.train_grid,
            self.train_inv_denominator,
        )


class SplineLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kw
    ):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)


class KANLinear(nn.Module):
    """
    Drop-in replacement for the original KANLinear, utilizing
    the FasterKAN CUDA RSWAf implementation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,  # Maps to num_grids
        spline_order: int = 3,  # Ignored, kept for API compatibility
        scale_noise: float = 0.1,  # Ignored
        scale_base: float = 1.0,  # Ignored
        scale_spline: float = 1.0,  # Ignored
        base_activation=nn.SiLU,  # Ignored
        grid_eps: float = 0.02,  # Ignored
        grid_range: list = [-1, 1],  # Handled via grid_min/grid_max
    ):
        super().__init__()
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

    def forward(self, x: torch.Tensor, update_grid=False):
        # Ignore the update_grid flag from legacy code
        x = self.layernorm(x)
        spline_basis = self.rbf(x).view(x.shape[0], -1)
        ret = self.spline_linear(spline_basis)
        return ret

    def update_grid(self, x: torch.Tensor, margin=0.01):
        # Stub to prevent crashes if training loop calls it
        pass

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # Fallback dummy loss
        l1_fake = self.spline_linear.weight.abs().mean(-1)
        return regularize_activation * l1_fake.sum()
