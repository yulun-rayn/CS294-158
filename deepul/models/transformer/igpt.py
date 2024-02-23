from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.transformer.block import TransformerBlock
from deepul.models.nn.convolution import CausalConv2d
from deepul.models.utils import LayerNorm


class ImageGPT(nn.Module):
    """
    The ImageGPT Model.
    """
    def __init__(self, image_shape, n_colors, d_model=128, n_layers=2, n_heads=4, kernel_size=7):
        """Initializes a new ImageGPT instance.
        """
        super(ImageGPT, self).__init__()
        self.image_shape = image_shape
        self.d_model = d_model

        self._input = nn.Sequential(
			CausalConv2d(True, image_shape[0], d_model, kernel_size, 1, kernel_size//2),
			LayerNorm(d_model)
		)

        self._pos = nn.Parameter(torch.zeros(1, d_model, *image_shape[1:]))

        self._transformer = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(d_model, n_heads, n_dims=2),
                LayerNorm(d_model)
            ])
            for _ in range(n_layers)
        ])

        self._out = nn.Conv2d(d_model, n_colors*image_shape[0], 1, 1, 0)

    def forward(self, x, kv_cache=None, pos=None):
        x = x.to(next(self.parameters()).device)
        n, c, _, _ = x.size()

        x = x.float()
        x = self._input(x)
        if pos is None:
            x = x + self._pos
        else:
            x = x[:, :, pos[0], pos[1], None, None] + self._pos[:, :, pos[0], pos[1], None, None]
        if kv_cache:
            for i, ((block, norm), cache) in enumerate(zip(self._transformer, kv_cache)):
                x, cache = block(x, kv_cache=cache, pos=pos)
                kv_cache[i] = cache
                x = norm(x)
        else:
            for i, (block, norm) in enumerate(self._transformer):
                x = block(x)
                x = norm(x)
        x = self._out(x)
        x = x.view(n, -1, c, *x.shape[2:])
        if kv_cache:
            return x, kv_cache
        else:
            return x

    def loss(self, x):
        x = x.to(next(self.parameters()).device)
        return F.cross_entropy(self(x), x)

    def sample(self, n_samples, kv_cache=False, track_time=False):
        device = next(self.parameters()).device
        if kv_cache:
            kv_cache = [
                [torch.zeros((n_samples, self.d_model, *self.image_shape[1:]), device=device)
                    for _ in range(2)
                ]
                for _ in range(len(self._transformer))
            ]

        samples = torch.zeros((n_samples, *self.image_shape), device=device)
        if track_time:
            time_list = []
            start_time = datetime.now()
        with torch.no_grad():
            for r in range(self.image_shape[1]):
                for c in range(self.image_shape[2]):
                    for k in range(self.image_shape[0]):
                        if kv_cache:
                            logits, kv_cache = self(samples, kv_cache=kv_cache, pos=(r, c))
                            probs = F.softmax(logits[:, :, k, 0, 0], dim=1)
                        else:
                            logits = self(samples)
                            probs = F.softmax(logits[:, :, k, r, c], dim=1)
                        samples[:, k, r, c] = torch.multinomial(probs, 1)[:, 0]

                        if track_time:
                            end_time = datetime.now()
                            time_list.append((end_time-start_time).total_seconds())
                            start_time = end_time

        if track_time:
            return samples.cpu().numpy(), time_list
        else:
            return samples.cpu().numpy()
