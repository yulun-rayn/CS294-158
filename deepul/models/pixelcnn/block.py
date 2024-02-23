import torch
import torch.nn as nn

from deepul.models.nn.convolution import CausalConv2d


class CausalResidualBlock(nn.Module):
    """A residual block masked to respect the autoregressive property."""

    def __init__(self, mask_center, n_filters, kernel_size):
        super().__init__()

        self._net = nn.Sequential(
            nn.Conv2d(n_filters, n_filters//2, 1, 1, 0),
            nn.ReLU(),
            CausalConv2d(mask_center=mask_center,
                in_channels=n_filters//2, out_channels=n_filters//2,
                kernel_size=kernel_size, stride=1, padding=kernel_size//2
            ),
            nn.ReLU(),
            nn.Conv2d(n_filters//2, n_filters, 1, 1, 0)
        )

    def forward(self, x):
        return x + self._net(x)
