import math

import torch
import torch.nn as nn


class MLConv(nn.Module):
    def __init__(self, sizes, resolutions, kernel_size=3, dim=2, batch_norm=False, final_act=None):
        super().__init__()
        assert len(sizes) == len(resolutions)
        if dim == 1:
            conv_model = nn.Conv1d
            if batch_norm:
                norm_model = nn.BatchNorm1d
        elif dim == 2:
            conv_model = nn.Conv2d
            if batch_norm:
                norm_model = nn.BatchNorm2d
        elif dim == 3:
            conv_model = nn.Conv3d
            if batch_norm:
                norm_model = nn.BatchNorm3d
        else:
            return ValueError("dim not recognized")

        layers = []
        for i in range(len(sizes) - 1):
            up_rate = ()
            down_rate = ()
            for res_in, res_out in zip(resolutions[i], resolutions[i+1]):
                assert (res_in % res_out == 0 or res_out % res_in == 0)
                up_rate += (res_out//res_in,)
                down_rate += (res_in//res_out,)

            layers += [
                conv_model(sizes[i], sizes[i+1], kernel_size, 1, kernel_size//2)
                    if min(resolutions[i]) >= kernel_size else
                conv_model(sizes[i], sizes[i+1], 1, 1, 0),

                nn.Upsample(scale_factor=up_rate)
                    if math.prod(up_rate) > 1 else
                nn.AvgPool2d(kernel_size=down_rate, stride=down_rate)
                    if math.prod(down_rate) > 1 else
                None,

                norm_model(sizes[i+1])
                    if batch_norm and i < len(sizes) - 2 else
                None,

                nn.GELU()
                    if i < len(sizes) - 2 else
                None
            ]
        if final_act is None:
            pass
        elif final_act == "gelu":
            layers += [nn.GELU()]
        elif final_act == "relu":
            layers += [nn.ReLU()]
        elif final_act == "sigmoid":
            layers += [nn.Sigmoid()]
        elif final_act == "softmax":
            layers += [nn.Softmax(dim=-3)]
        else:
            raise ValueError("final_act not recognized")

        layers = [l for l in layers if l is not None]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalConv2d(nn.Conv2d):
    """A Conv2d layer masked to respect the autoregressive property.

    Copied from https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn/attention.py.

    Autoregressive masking means that the computation of the current pixel only
    depends on itself, pixels to the left, and pixels above. When mask_center=True, the
    computation of the current pixel does not depend on itself.

    E.g. for a 3x3 kernel, the following masks are generated for each channel:
                          [[1 1 1],                     [[1 1 1],
        mask_center=True   [1 0 0],    mask_center=False [1 1 0],
                           [0 0 0]]                      [0 0 0]]
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

    Copied from https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn/attention.py.

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
