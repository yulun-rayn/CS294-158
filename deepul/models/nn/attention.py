import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_encoding(d_model, max_len):
    """Generates the sinusoidal positional encodings introduced in [3].

    Copied from https://pytorch.org/tutorials/beginner/transformer_tutorial.html.

    Args:
        d_model: Dimension of the model (i.e. embedding dimension).
        max_len: Maximum possible sequence length.
    Return:
        Tensor of shape [max_len, 1, d_model] containing the positional encodings.
    """
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    positional_encoding = torch.zeros(max_len, 1, d_model)
    positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
    return positional_encoding


@functools.lru_cache(maxsize=32)
def image_positional_encoding(shape):
    """Generates positional encodings for 2d images.

    Copied from https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn/attention.py.

    The positional encoding is a Tensor of shape (N, 2, H, W) of (x, y) pixel
    coordinates scaled to be between -.5 and .5.

    Args:
        shape: NCHW shape of image for which to generate positional encodings.
    Returns:
        The positional encodings.
    """
    n, _, h, w = shape
    zeros = torch.zeros(n, 1, h, w)
    return torch.cat(
        (
            (torch.arange(-0.5, 0.5, 1 / h)[None, None, :, None] + zeros),
            (torch.arange(-0.5, 0.5, 1 / w)[None, None, None, :] + zeros),
        ),
        dim=1,
    )


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
        n_dims=2,
        n_heads=1,
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
