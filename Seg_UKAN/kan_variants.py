import torch
import torch.nn as nn
import torch.nn.functional as F

class TeLU(nn.Module):
    """
    Hyperbolic Tangent Exponential Linear Unit (TeLU)
    f(x) = x * tanh(exp(x))
    """
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

class ReLUKANLinear(nn.Module):
    """
    ReLU-KAN architecture utilizing exclusively ReLU activations 
    for maximal inference speed.
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
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        self.layernorm = nn.LayerNorm(in_features)
        
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.grid = nn.Parameter(grid, requires_grad=False)
        
        self.linear = nn.Linear(in_features * grid_size, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, update_grid=False):
        x = self.layernorm(x)
        # Apply piecewise-linear basis function
        basis = F.relu(x.unsqueeze(-1) - self.grid)
        basis = basis.view(x.shape[0], -1)
        return self.linear(basis)


class HardSwishKANLinear(nn.Module):
    """
    Hard-Swish KAN architecture utilizing an optimized smooth piecewise-linear
    function, preventing dying neurons while keeping arithmetic simple.
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
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        self.layernorm = nn.LayerNorm(in_features)
        
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.grid = nn.Parameter(grid, requires_grad=False)
        
        self.linear = nn.Linear(in_features * grid_size, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, update_grid=False):
        x = self.layernorm(x)
        basis = F.hardswish(x.unsqueeze(-1) - self.grid) 
        basis = basis.view(x.shape[0], -1)
        return self.linear(basis)


class PWLOKANLinear(nn.Module):
    """
    Piecewise Linear Optimization (PWLO) KAN.
    Uses Look-Up Tables (LUTs) with embeddings to apply a single 
    shift-and-add equivalent operation (y = ax+b) per segment.
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
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_segments = grid_size
        
        self.layernorm = nn.LayerNorm(in_features)
        
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.step = (self.grid_max - self.grid_min) / self.num_segments
        
        # LUT mapped as an embedding layer for max execution efficiency
        # Input features span independent segments
        self.a_embed = nn.Embedding(in_features * self.num_segments, out_features)
        self.b_embed = nn.Embedding(in_features * self.num_segments, out_features)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.a_embed.weight)
        nn.init.zeros_(self.b_embed.weight)
        
    def forward(self, x: torch.Tensor, update_grid=False):
        x = self.layernorm(x)
        B = x.shape[0]
        
        float_idx = (x - self.grid_min) / self.step
        segment_idx = torch.clamp(float_idx.long(), 0, self.num_segments - 1)
        
        feature_offsets = torch.arange(self.in_features, device=x.device).unsqueeze(0) * self.num_segments
        global_idx = segment_idx + feature_offsets
        
        a_vals = self.a_embed(global_idx) # (B, in_features, out_features)
        b_vals = self.b_embed(global_idx) # (B, in_features, out_features)
        
        # y = a_vals * x + b_vals
        y = a_vals * x.unsqueeze(-1) + b_vals 
        
        return y.sum(dim=1)


class TeLUKANLinear(nn.Module):
    """
    TeLU (Hyperbolic Tangent Exponential Linear Unit) KAN.
    Near-linearity approach approximating exact linearity for positive inputs
    with highly stable gradient behavior.
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
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        self.layernorm = nn.LayerNorm(in_features)
        
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.telu = TeLU()
        self.linear = nn.Linear(in_features * grid_size, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, update_grid=False):
        x = self.layernorm(x)
        basis = self.telu(x.unsqueeze(-1) - self.grid)
        basis = basis.view(x.shape[0], -1)
        return self.linear(basis)
