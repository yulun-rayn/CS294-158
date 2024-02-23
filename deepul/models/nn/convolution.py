# credits: eugen hotaj https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn/convolution.py

import torch
import torch.nn as nn


class CausalConv2d(nn.Conv2d):
    """A Conv2d layer masked to respect the autoregressive property.

    Autoregressive masking means that the computation of the current pixel only
    depends on itself, pixels to the left, and pixels above. When mask_center=True, the
    computation of the current pixel does not depend on itself.

    E.g. for a 3x3 kernel, the following masks are generated for each channel:
                          [[1 1 1],                     [[1 1 1],
        mask_center=True   [1 0 0],    mask_center=False [1 1 0],
                           [0 0 0]]                      [0 0 0]
    In [1], they refer to the left masks as 'type A' and right as 'type B'.
    """

    def __init__(self, mask_center, *args, **kwargs):
        """Initializes a new CausalConv2d instance.

        Args:
            mask_center: Whether to mask the center pixel of the convolution filters.
        """
        super().__init__(*args, **kwargs)
        i, o, h, w = self.weight.shape
        mask = torch.zeros((i, o, h, w))
        mask.data[:, :, : h//2, :] = 1
        mask.data[:, :, h//2, : w//2+int(not mask_center)] = 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(CausalConv2d, self).forward(x)


class GatedActivation(nn.Module):
    """Gated activation function as introduced in [2].

    The function computes activation_fn(f) * sigmoid(g). The f and g correspond to the
    top 1/2 and bottom 1/2 of the input channels.
    """

    def __init__(self, activation_fn=torch.tanh):
        """Initializes a new GatedActivation instance.

        Args:
            activation_fn: Activation to use for the top 1/2 input channels.
        """
        super().__init__()
        self._activation_fn = activation_fn

    def forward(self, x):
        _, c, _, _ = x.shape
        assert c % 2 == 0, "x must have an even number of channels."
        x, gate = x[:, : c//2, :, :], x[:, c//2 :, :, :]
        return self._activation_fn(x) * torch.sigmoid(gate)
