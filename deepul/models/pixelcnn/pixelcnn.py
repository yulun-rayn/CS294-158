import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.nn.convolution import CausalConv2d
from deepul.models.pixelcnn.block import CausalResidualBlock
from deepul.models.utils import LayerNorm


class PixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al. 
    """
    def __init__(self, n_channels, n_colors, n_filters=64, n_layers=5, kernel_size=7, res_block=False):
        super(PixelCNN, self).__init__()
        self.n_channels = n_channels

        layers = [nn.Sequential(
            CausalConv2d(True, n_channels, n_filters, kernel_size, 1, kernel_size//2),
            LayerNorm(n_filters)
        )]
        for _ in range(n_layers):
            if res_block:
                layers.append(nn.Sequential(
                    CausalResidualBlock(False, n_filters, kernel_size),
                    LayerNorm(n_filters)
                ))
            else:
                layers.append(nn.Sequential(
                    CausalConv2d(False, n_filters, n_filters, kernel_size, 1, kernel_size//2),
                    LayerNorm(n_filters)
                ))
        layers.append(nn.Conv2d(n_filters, n_filters, 1, 1, 0))

        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.final_layer = nn.Conv2d(n_filters, n_colors*n_channels, 1, 1, 0)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = x.float()
        shape = x.size()

        for _, layer in enumerate(self.layers):
            x = self.act(layer(x))
        x = self.final_layer(x)

        return x.view(shape[0], -1, *shape[1:])

    def loss(self, x):
        x = x.to(next(self.parameters()).device)
        return F.cross_entropy(self(x), x)

    def sample(self, n_samples, image_shape):
        samples = torch.zeros(n_samples, self.n_channels, *image_shape).to(next(self.parameters()).device)
        with torch.no_grad():
            for r in range(image_shape[0]):
                for c in range(image_shape[1]):
                    for k in range(self.n_channels):
                        logits = self(samples)[:, :, k, r, c]
                        probs = F.softmax(logits, dim=-1)
                        samples[:, k, r, c] = torch.multinomial(probs, 1)[:, 0]
        return samples.cpu().numpy()
