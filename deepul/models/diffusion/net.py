from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.diffusion.utils import (
    default, cast_tuple, SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb
)
from deepul.models.diffusion.block import ScaleShiftResBlock, ScaleShiftTransformer
from deepul.models.nn.convolution import Upsample, Downsample
from deepul.models.nn.attention import Attention, LinearAttention
from deepul.models.vit.utils import get_2d_sincos_pos_embed


class TimeMLP(nn.Module):
    def __init__(
        self,
        dim,
        in_dim=3,
        out_dim=None,
        emb_dim=None,
        num_layers=4,
        num_classes=1,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_scale=1000,
        sinusoidal_pos_emb_theta=10000
    ):
        super().__init__()

        # determine dimensions

        self.in_dim = in_dim
        self.self_condition = self_condition
        input_dim = self.in_dim * (2 if self_condition else 1)

        emb_dim = default(emb_dim, dim)
        self.init_emb = nn.Linear(input_dim, emb_dim)

        self.out_dim = default(out_dim, in_dim)
        output_dim = self.out_dim * (1 if not learned_variance else 2)

        # time embeddings

        time_dim = emb_dim

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(emb_dim, scale=sinusoidal_pos_emb_scale, theta=sinusoidal_pos_emb_theta)
            fourier_dim = emb_dim

        self.time_emb = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        if num_classes > 1:
            self.label_emb = nn.Embedding(num_classes + 1, time_dim, padding_idx=0)
        else:
            self.register_parameter('label_emb', None)

        # layers

        dims = [2*emb_dim, *[dim]*num_layers, output_dim]
        in_out = list(zip(dims[:-1], dims[1:]))

        layers = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            layers.extend([
                nn.ReLU(),
                nn.Linear(dim_in, dim_out)
            ])
        self.layers = nn.Sequential(*layers)

    @property
    def device(self):
        return self.init_emb.weight.device

    def forward(self, x, time, label=None, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_emb(x)
        t = self.time_emb(time)
        if self.label_emb is not None:
            label = torch.zeros_like(time, dtype=torch.long) if label is None else label + 1
            y = self.label_emb(label)
            t = t + y

        return self.layers(torch.cat([x, t], dim=1))


class TimeUnet(nn.Module):
    def __init__(
        self,
        dim,
        in_dim=3,
        out_dim=None,
        emb_dim=None,
        dim_mults=(1, 2, 4, 8),
        kernel=3,
        emb_kernel=7,
        num_classes=1,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_scale=1000,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        flash_attn=False
    ):
        super().__init__()

        # determine dimensions

        self.in_dim = in_dim
        self.self_condition = self_condition
        input_channels = self.in_dim * (2 if self_condition else 1)

        emb_dim = default(emb_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, emb_dim, emb_kernel, padding=emb_kernel//2)

        dims = [emb_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ScaleShiftResBlock, kernel_size=kernel, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, scale=sinusoidal_pos_emb_scale, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_emb = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        if num_classes > 1:
            self.label_emb = nn.Embedding(num_classes + 1, time_dim, padding_idx=0)
        else:
            self.register_parameter('label_emb', None)

        # attention

        full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        FullAttention = partial(Attention, flash=flash_attn)
        PartAttention = partial(LinearAttention, out_norm=True)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_class = FullAttention if layer_full_attn else PartAttention

            self.downs.append(nn.ModuleList([
                block_class(dim_in, dim_in, time_emb_dim=time_dim),
                block_class(dim_in, dim_in, time_emb_dim=time_dim),
                attn_class(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, kernel, padding=kernel//2)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_class = FullAttention if layer_full_attn else PartAttention

            self.ups.append(nn.ModuleList([
                block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                attn_class(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, kernel, padding=kernel//2)
            ]))

        self.out_dim = default(out_dim, in_dim)
        output_channels = self.out_dim * (1 if not learned_variance else 2)

        self.final_res_block = block_class(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

    @property
    def device(self):
        return self.init_conv.weight.device

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, label=None, x_self_cond=None):
        assert all([d % self.downsample_factor == 0 for d in x.shape[-2:]]), \
            f'input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_emb(time)
        if self.label_emb is not None:
            label = torch.zeros_like(time, dtype=torch.long) if label is None else label + 1
            y = self.label_emb(label)
            t = t + y

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class TimeViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        in_dim=3,
        out_dim=None,
        depth=12,
        kernel=3,
        num_classes=1,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_scale=1000,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=64,
        attn_heads=4
    ):
        super().__init__()

        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height, image_width // patch_width)
        self.patch_dim = in_dim * patch_height * patch_width

        # patchify

        self.in_dim = in_dim
        self.self_condition = self_condition
        input_channels = self.in_dim * (2 if self_condition else 1)

        self.patchify = nn.Sequential(
            nn.Conv2d(input_channels, dim, kernel_size=kernel, padding=kernel//2),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size)
        )

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, scale=sinusoidal_pos_emb_scale, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_emb = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        if num_classes > 1:
            self.label_emb = nn.Embedding(num_classes + 1, time_dim, padding_idx=0)
        else:
            self.register_parameter('label_emb', None)

        # positional embeddings

        self.pos_embedding = nn.Parameter(torch.from_numpy(pos_embedding).float().unsqueeze(0), requires_grad=False)

        # transformer

        self.transformer = ScaleShiftTransformer(dim, depth, attn_heads, attn_dim_head, time_emb_dim=time_dim)

        # unpatchify

        self.out_dim = default(out_dim, in_dim)
        output_channels = self.out_dim * (1 if not learned_variance else 2)

        self.unpatchify = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Conv2d(dim, output_channels, kernel_size=kernel, padding=kernel//2)
        )

    @property
    def device(self):
        return self.pos_embedding.data.device

    def forward(self, x, time, label=None, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.patchify(x)

        t = self.time_emb(time)
        if self.label_emb is not None:
            label = torch.zeros_like(time, dtype=torch.long) if label is None else label + 1
            y = self.label_emb(label)
            t = t + y

        b, c, _, _ = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        x = x + self.pos_embedding

        x = self.transformer(x, time_emb=t)

        b, _, c = x.shape
        x = x.permute(0, 2, 1).reshape(b, c, self.num_patches[0], -1)

        x = self.unpatchify(x)

        return x
