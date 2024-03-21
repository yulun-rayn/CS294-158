import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def mult(val, t):
    b, *_ = t.shape
    val = val.reshape(b, *[1]*(t.dim() - 1))
    return val * t

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def cosine_alpha_schedule(s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """

    alpha_0 = math.cos(s / (1 + s) * math.pi * 0.5) ** 2
    def sch(t):
        alpha = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        return alpha / alpha_0
    return sch

def sigmoid_alpha_schedule(start=-3, end=3, tau=1):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()

    alpha_0 = (-(start / tau).sigmoid() + v_end) / (v_end - v_start)
    def sch(t):
        alpha = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        return alpha / alpha_0
    return sch


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1000, theta=10000):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = self.scale * x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=(not is_random))

    def forward(self, x):
        x = x[..., None]
        freqs = x * self.weights[None, ...] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
