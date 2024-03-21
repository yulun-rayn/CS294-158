# credits: thuan h. nguyen https://github.com/thuanz123/enhancing-transformers/blob/main/enhancing/modules/stage1/layers.py

from typing import Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.nn.attention import SimpleAttention
from deepul.models.nn.mlp import FeedForward
from deepul.models.vit.utils import get_2d_sincos_pos_embed, init_weights
from deepul.models.utils import PreNorm


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, out_norm: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, SimpleAttention(dim, heads=heads, dim_head=dim_head)),
                                   PreNorm(dim, FeedForward(dim, mlp_dim, activation="tanh"))])
            self.layers.append(layer)

        self.norm = nn.LayerNorm(dim) if out_norm else None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x) if self.norm is not None else x


class ViTEncoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, latent_dim: int = None, channels: int = 3, dim_head: int = 64) -> None:
        super().__init__()
        if latent_dim is None:
            latent_dim = dim

        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        en_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height, image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.to_patch = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size)
        )
        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.to_latent = nn.Linear(dim, latent_dim)

        self.apply(init_weights)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_patch(x)
        b, c, _, _ = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        x = x + self.en_pos_embedding
        x = self.transformer(x)
        x = self.to_latent(x)

        return x


class ViTDecoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, latent_dim: int = None, channels: int = 3, dim_head: int = 64) -> None:
        super().__init__()
        if latent_dim is None:
            latent_dim = dim

        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height, image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.to_token = nn.Linear(latent_dim, dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.to_pixel = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Conv2d(dim, channels, kernel_size=(3, 3), padding=1)
        )

        self.apply(init_weights)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_token(x)
        x = x + self.de_pos_embedding
        x = self.transformer(x)
        b, _, c = x.shape
        x = x.permute(0, 2, 1).reshape(b, c, self.num_patches[0], -1)
        x = self.to_pixel(x)

        return x

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight
