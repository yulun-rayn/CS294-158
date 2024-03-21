import torch
import torch.nn as nn

from deepul.models.nn.attention import CausalAttention
from deepul.models.utils import LayerNorm


class TransformerBlock(nn.Module):
    """A Transformer block."""

    def __init__(self, d_model=128, n_heads=4, n_dims=1, bottleneck_rate=4.):
        """Initializes a new TransformerBlock instance.
        """
        super().__init__()
        self._attn = CausalAttention(
            in_channels=d_model,
            embed_channels=d_model,
            out_channels=d_model,
            n_heads=n_heads,
            n_dims=n_dims,
        )

        mod = nn.Conv1d if n_dims==1 else nn.Conv2d if n_dims==2 else nn.Conv3d
        self._out = nn.Sequential(
            mod(d_model, int(bottleneck_rate*d_model), 1, 1, 0),
            nn.GELU(),
            mod(int(bottleneck_rate*d_model), d_model, 1, 1, 0),
        )

        self._ln = LayerNorm(d_model)

    def forward(self, x, kv_cache=None, pos=None):
        if kv_cache:
            out, kv_cache = self._attn(x, kv_cache=kv_cache, pos=pos)
        else:
            out = self._attn(x)
        x = x + out
        x = self._ln(x)
        x = x + self._out(x)
        if kv_cache:
            return x, kv_cache
        else:
            return x
