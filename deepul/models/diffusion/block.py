import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.diffusion.utils import default
from deepul.models.nn.attention import SimpleAttention
from deepul.models.nn.mlp import FeedForward


class ScaleShiftConvLayer(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift

        x = self.act(x)
        return x


class ScaleShiftResBlock(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = ScaleShiftConvLayer(dim, dim_out, kernel_size=kernel_size, groups=groups)
        self.block2 = ScaleShiftConvLayer(dim_out, dim_out, kernel_size=kernel_size, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb[..., None][..., None]
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class ScaleShiftPreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, *args, **kwargs) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, *args, **kwargs)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, scale: torch.FloatTensor = None, shift: torch.FloatTensor = None, **kwargs) -> torch.FloatTensor:
        x = self.norm(x)
        if scale is not None:
            x = x * (1 + scale)
        if shift is not None:
            x = x + shift
        return self.fn(x, **kwargs)


class ScaleShiftTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int = None,
                 time_emb_dim: int = None, out_dim: int = None, out_norm: bool = True) -> None:
        super().__init__()
        self.depth = depth

        mlp_dim = default(mlp_dim, 4 * dim)
        out_dim = default(out_dim, dim)

        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([ScaleShiftPreNorm(dim, SimpleAttention(dim, heads=heads, dim_head=dim_head),
                                                     elementwise_affine=False, eps=1e-6),
                                   ScaleShiftPreNorm(dim, FeedForward(dim, mlp_dim, activation=lambda: nn.GELU(approximate="tanh")),
                                                     elementwise_affine=False, eps=1e-6)])
            self.layers.append(layer)

        self.final_layer = ScaleShiftPreNorm(dim, nn.Linear(dim, out_dim), elementwise_affine=False, eps=1e-6)

        if time_emb_dim is not None:
            self.modulations = nn.ModuleList([])
            for idx in range(depth):
                modulation = nn.Sequential(nn.SiLU(),
                                           nn.Linear(time_emb_dim, 6 * dim, bias=True))
                self.modulations.append(modulation)

            self.final_modulation = nn.Sequential(nn.SiLU(),
                                                  nn.Linear(time_emb_dim, 2 * dim, bias=True))

        self.norm = nn.LayerNorm(out_dim) if out_norm else None

    def forward(self, x: torch.FloatTensor, time_emb: torch.FloatTensor = None) -> torch.FloatTensor:
        for i in range(self.depth):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulations[i](time_emb).unsqueeze(1).chunk(6, dim=2) if time_emb is not None else [None]*6
            )
            attn, ff = self.layers[i]
            x = x + gate_msa * attn(x, scale=scale_msa, shift=shift_msa)
            x = x + gate_mlp * ff(x, scale=scale_mlp, shift=shift_mlp)
        shift_mlp, scale_mlp = (
            self.final_modulation(time_emb).unsqueeze(1).chunk(2, dim=2) if time_emb is not None else [None]*2
        )
        x = self.final_layer(x, scale=scale_mlp, shift=shift_mlp)

        return self.norm(x) if self.norm is not None else x
