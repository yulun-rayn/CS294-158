import functools
from packaging import version
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.utils import print_once


class SimpleAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        b, n, _ = x.shape

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b, n, self.heads, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim = -1)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)

        return self.to_out(out)



AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


class RMSNorm(nn.Module):
    def __init__(self, dim, n_dims=2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, *[1]*n_dims))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class Attend(nn.Module):
    def __init__(self, dropout=0., flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        if self.scale is not None:
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if q.is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        if self.flash:
            return self.flash_attn(q, k, v)

        scale = q.shape[-1] ** -0.5 if self.scale is None else self.scale

        # similarity

        sim = torch.einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = torch.einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, n_dims=2, num_mem_kv=4, out_norm=False, flash=False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads

        mod = nn.Conv1d if n_dims==1 else nn.Conv2d if n_dims==2 else nn.Conv3d

        self.norm = RMSNorm(dim, n_dims)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = mod(dim, hidden_dim * 3, 1, bias=False)

        to_out = [mod(hidden_dim, dim, 1)]
        if out_norm:
            to_out.append(RMSNorm(dim, n_dims))
        self.to_out = nn.Sequential(*to_out)

    def forward(self, x):
        b, _, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, -1).transpose(2, 3), qkv)

        mk, mv = map(lambda t: t.repeat(b, 1, 1, 1), self.mem_kv)
        k, v = map(functools.partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = out.transpose(2, 3).contiguous().view(b, -1, h, w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, n_dims=2, num_mem_kv=4, out_norm=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads

        mod = nn.Conv1d if n_dims==1 else nn.Conv2d if n_dims==2 else nn.Conv3d

        self.norm = RMSNorm(dim, n_dims)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = mod(dim, hidden_dim * 3, 1, bias=False)

        to_out = [mod(hidden_dim, dim, 1)]
        if out_norm:
            to_out.append(RMSNorm(dim, n_dims))
        self.to_out = nn.Sequential(*to_out)

    def forward(self, x):
        b, _, *size = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, -1), qkv)

        mk, mv = map(lambda t: t.repeat(b, 1, *[1]*len(size)), self.mem_kv)
        k, v = map(functools.partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.contiguous().view(b, -1, *size)
        return self.to_out(out)



@functools.lru_cache(maxsize=32)
def _get_causal_mask(size, mask_current):
    """Generates causal masks for attention weights."""
    return torch.tril(torch.ones((size, size)), diagonal=-int(mask_current))


class CausalAttention(nn.Module):
    """Autoregresively masked, multihead self-attention layer.

    Autoregressive masking means that the current pixel can only attend to itself,
    pixels to the left, and pixels above. When mask_current=True, the current pixel does
    not attend to itself.

    This Module generalizes attention to use 1D, 2D or 3D convolutions. As such, the
    input is expected to be 2D, 3D or 4D tensors.
    """

    def __init__(
        self,
        in_channels,
        embed_channels=None,
        out_channels=None,
        n_heads=1,
        n_dims=2,
        mask_current=False,
        extra_input_channels=0
    ):
        """Initializes a new CausalAttention instance.

        Args:
            in_channels: Number of input channels.
            embed_channels: Number of embedding channels. Defaults to in_channels.
            out_channels: Number of output channels. Defaults to in_channels.
            n_dims: Number of dimensions of the object.
            n_heads: Number of causal self-attention heads.
            extra_input_channels: Extra input channels which are only used to compute
                the embeddings and not the attention weights since doing so may break
                the autoregressive property. For example, in [1] these channels include
                the original input image.
            mask_current: Whether to mask the center pixel of the attention matrices.
        """
        super().__init__()
        self._n_heads = n_heads
        self._embed_channels = embed_channels or in_channels
        self._out_channels = out_channels or in_channels
        self._mask_current = mask_current

        mod = nn.Conv1d if n_dims==1 else nn.Conv2d if n_dims==2 else nn.Conv3d

        self._q = mod(in_channels, self._embed_channels, 1, 1, 0)
        self._kv = mod(in_channels + extra_input_channels, self._embed_channels + self._out_channels, 1, 1, 0)

        self._proj = mod(self._out_channels, self._out_channels, 1, 1, 0)

    def forward(self, x, extra_x=None, kv_cache=None, pos=None):
        """Computes the forward pass.

        Args:
            x: An (N, C, ...) input tensor used to compute both embeddings and
                attention weights.
            extra_x: Extra channels concatenated with 'x' only used to compute the
                embeddings. See the 'extra_input_channels' argument for more info.
            kv_cache: A tuple (k, v) with cached keys and cached values from previous
                time steps.
            pos: A tuple (...) indicating the current position. only used when kv_cache
                is provided.
        Returns:
            The result of the forward pass.
        """
        if kv_cache:
            assert pos is not None
            n, _, *shape = kv_cache[0].shape
        else:
            n, _, *shape = x.shape

        # Compute the causal attention weights.
        mask = (
            _get_causal_mask(np.prod(shape), self._mask_current)
            .view(1, 1, np.prod(shape), np.prod(shape))
            .to(next(self.parameters()).device)
        )

        # Compute the query, key, and value.
        q = self._to_multihead(self._q(x))
        if kv_cache:
            k_cache, v_cache = kv_cache
            new_k, new_v = self._kv(x).split([self._embed_channels, self._out_channels], dim=1)
            k_cache[(slice(None),slice(None),)+pos] = new_k[(slice(None),slice(None),)+tuple([0]*len(pos))]
            v_cache[(slice(None),slice(None),)+pos] = new_v[(slice(None),slice(None),)+tuple([0]*len(pos))]
            k, v = k_cache, v_cache
            pos_flat = int(sum([pos[i]*np.prod(shape[(i+1):]) for i in range(len(pos))]))
            mask = mask[:, :, [pos_flat], :]
        else:
            if extra_x is not None:
                x = torch.cat((x, extra_x), dim=1)
            k, v = self._kv(x).split([self._embed_channels, self._out_channels], dim=1)
        k, v = self._to_multihead(k), self._to_multihead(v)

        attn = (q @ k.transpose(2, 3)) / np.sqrt(k.shape[-1])
        attn = attn.masked_fill(mask == 0, -np.inf)

        attn = F.softmax(attn, dim=-1).masked_fill(mask == 0, 0)

        # Attend to output for each head, stack, and project.
        out = (attn @ v).transpose(2, 3).contiguous().view(n, -1, *x.shape[2:])
        if kv_cache:
            return self._proj(out), (k_cache, v_cache)
        else:
            return self._proj(out)

    def _to_multihead(self, t):
        """Reshapes an (N, C, ...) tensor into (N, n_heads, prod(...), head_size)."""
        n, c, *_ = t.shape
        t = t.view(n, self._n_heads, c//self._n_heads, -1)
        return t.transpose(2, 3)
