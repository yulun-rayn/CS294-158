import math

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


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


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(
            0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True):
        super(Upsample_Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, x):
        _x = torch.cat([x, x, x, x], dim=1)
        _x = self.depth_to_space(_x)
        _x = self.conv(_x)
        return _x


class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True):
        super(Downsample_Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.space_to_depth = SpaceToDepth(2)

    def forward(self, x):
        _x = self.space_to_depth(x)
        _x = sum(_x.chunk(4, dim=1)) / 4.0
        _x = self.conv(_x)
        return _x


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, n_filters=256, batch_norm=False):
        super(ResnetBlockUp, self).__init__()
        layers = [
            nn.BatchNorm2d(in_dim) if batch_norm else None,
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(n_filters) if batch_norm else None,
            nn.ReLU(),
            Upsample_Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1)
        ]

        layers = [l for l in layers if l is not None]
        self.layers = nn.Sequential(*layers)
        self.proj = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        return self.layers(x) + self.proj(x)


class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, n_filters=256, batch_norm=False):
        super(ResnetBlockDown, self).__init__()
        layers = [
            nn.BatchNorm2d(in_dim) if batch_norm else None,
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(n_filters) if batch_norm else None,
            nn.ReLU(),
            Downsample_Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1)
        ]

        layers = [l for l in layers if l is not None]
        self.layers = nn.Sequential(*layers)
        self.proj = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        return self.layers(x) + self.proj(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, n_filters=256, batch_norm=False, spectral_norm=False):
        super(ResnetBlock, self).__init__()
        conv1 = nn.Conv2d(in_dim, n_filters, kernel_size=(3, 3), padding=1)
        conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1)
        if spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
            conv2 = nn.utils.spectral_norm(conv2)

        layers = [
            nn.BatchNorm2d(in_dim) if batch_norm else None,
            nn.ReLU(),
            conv1,
            nn.BatchNorm2d(n_filters) if batch_norm else None,
            nn.ReLU(),
            conv2
        ]

        layers = [l for l in layers if l is not None]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + x
