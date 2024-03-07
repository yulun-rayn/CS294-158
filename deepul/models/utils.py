import torch
import torch.nn as nn


class Summation(nn.Module):
    def __init__(self, dim=None) -> None:
        super().__init__()
        self.dim=dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Embedding(nn.Embedding):
    """Generate embeddings for (N, L) tensors and output (N, C, L) tensors."""

    def forward(self, x):
        x = super().forward(x)
        return x.permute(0, 2, 1)


class LayerNorm(nn.LayerNorm):
    """Applies LayerNorm to the channel dimension of (N, C, ...) tensors."""

    def forward(self, x):
        x = x.permute(0, *range(2, len(x.shape)), 1)
        x = super().forward(x)
        return x.permute(0, len(x.shape)-1, *range(1, len(x.shape)-1))
