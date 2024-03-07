import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.ae import VAEConv
from deepul.models.utils import LayerNorm

def conv_1x1(in_width, out_width, dim=2):
    if dim == 1:
        return nn.Conv1d(in_width, out_width, 1, 1, 0)
    elif dim == 2:
        return nn.Conv2d(in_width, out_width, 1, 1, 0)
    elif dim == 3:
        return nn.Conv3d(in_width, out_width, 1, 1, 0)
    else:
        return ValueError("dim not recognized")

def conv_3x3(in_width, out_width, dim=2):
    if dim == 1:
        return nn.Conv1d(in_width, out_width, 3, 1, 1)
    elif dim == 2:
        return nn.Conv2d(in_width, out_width, 3, 1, 1)
    elif dim == 3:
        return nn.Conv3d(in_width, out_width, 3, 1, 1)
    else:
        return ValueError("dim not recognized")


class ConvBlock(nn.Module):
    def __init__(self, in_width, emb_width, out_width, dim=2,
                 residual=True, layer_norm=False, use_3x3=True, final_act=None,
                 up_rate=None, down_rate=None, rescale_first=False):
        super().__init__()
        assert up_rate is None or down_rate is None

        self.up_rate = up_rate
        self.down_rate = down_rate
        self.rescale_first = rescale_first
        self.dim = dim
        self.residual = residual
        self.layer_norm = layer_norm

        layers = [
            conv_1x1(in_width, emb_width, dim),
            conv_3x3(emb_width, emb_width, dim)
                if use_3x3 else
            conv_1x1(emb_width, emb_width, dim),
            conv_1x1(emb_width, out_width, dim)
        ]

        layers, final_layer = layers[:-1], layers[-1]
        self.layers = nn.ModuleList(layers)
        self.final_layer = final_layer

        self.act = nn.GELU()
        if final_act is None:
            self.final_act = nn.Identity()
        elif final_act == "gelu":
            self.final_act = nn.GELU()
        elif final_act == "relu":
            self.final_act = nn.ReLU()
        elif final_act == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif final_act == "softmax":
            self.final_act = nn.Softmax(dim=-1-dim)
        else:
            raise ValueError("final_act not recognized")

        self.res_layer = None
        if residual and in_width != out_width:
            self.res_layer = conv_1x1(in_width, out_width, dim)

        self.norm_layer = None
        if layer_norm:
            self.norm_layer = LayerNorm(out_width)

    def rescale(self, x):
        if self.up_rate is not None:
            return F.interpolate(x, scale_factor=self.up_rate)
        elif self.down_rate is not None:
            if self.dim == 1:
                return F.avg_pool1d(x, kernel_size=self.down_rate, stride=self.down_rate)
            elif self.dim == 2:
                return F.avg_pool2d(x, kernel_size=self.down_rate, stride=self.down_rate)
            elif self.dim == 3:
                return F.avg_pool3d(x, kernel_size=self.down_rate, stride=self.down_rate)
        else:
            return x

    def forward(self, x):
        if self.rescale_first:
            x = self.rescale(x)

        out = x
        for layer in self.layers:
            out = self.act(layer(out))
        out = self.final_act(self.final_layer(out))

        if self.residual:
            if self.res_layer:
                x = self.res_layer(x)
            out = out + x

        if self.layer_norm:
            out = self.norm_layer(out)

        if not self.rescale_first:
            out = self.rescale(out)
        return out


class HConvEncoder(nn.Module):
    def __init__(self, sizes, resolutions, dim=2, bottleneck_ratio=0.25, n_outputs=2):
        super().__init__()
        assert len(sizes) == len(resolutions)

        down_rate = ()
        for res_in, res_out in zip(resolutions[0], resolutions[1]):
            assert res_in % res_out == 0
            down_rate += (res_in//res_out,)
        self.first_block = ConvBlock(sizes[0], math.ceil(sizes[0]*bottleneck_ratio), sizes[1],
            dim=dim, use_3x3=(min(resolutions[1]) > 2),
            final_act=("gelu" if 0 < len(sizes) - 2 else None),
            down_rate=(down_rate if math.prod(down_rate) > 1 else None)
        )

        groups = []
        blocks = []
        final_blocks = []
        for i in range(1, len(sizes) - 1):
            down_rate = ()
            for res_in, res_out in zip(resolutions[i], resolutions[i+1]):
                assert res_in % res_out == 0
                down_rate += (res_in//res_out,)

            if math.prod(down_rate) > 1 and len(blocks) > 0:
                group = nn.Sequential(*blocks)
                groups.append(group)
                blocks = []
                final_blocks.append(
                    ConvBlock(sizes[i], math.ceil(sizes[i]*bottleneck_ratio), sizes[i],
                        dim=dim, use_3x3=False, final_act=None
                    )
                )

            blocks.append(
                ConvBlock(sizes[i], math.ceil(sizes[i]*bottleneck_ratio), sizes[i+1],
                    dim=dim, use_3x3=(min(resolutions[i+1]) > 2),
                    final_act=("gelu" if i < len(sizes) - 2 else None),
                    down_rate=(down_rate if math.prod(down_rate) > 1 else None)
                )
            )
        if len(blocks) > 0:
            group = nn.Sequential(*blocks)
            groups.append(group)
            final_blocks.append(
                ConvBlock(sizes[len(sizes)-1], math.ceil(sizes[len(sizes)-1]*bottleneck_ratio), sizes[len(sizes)-1],
                    dim=dim, use_3x3=False, final_act=None
                )
            )

        self.groups = nn.ModuleList(groups)
        self.final_blocks = nn.ModuleList(final_blocks[(-n_outputs):])

    def forward(self, x):
        x = self.first_block(x)

        xs = []
        for group in self.groups:
            x = group(x)
            xs.append(x)

        hs = []
        for x, final_block in zip(xs[(-len(self.final_blocks)):], self.final_blocks):
            h = final_block(x)
            hs.append(h)
        return hs


class HConvDecoder(nn.Module):
    def __init__(self, sizes, resolutions, dim=2, bottleneck_ratio=0.25, n_inputs=2):
        super().__init__()
        assert len(sizes) == len(resolutions)

        self.x_init = nn.Parameter(torch.zeros(sizes[0], *resolutions[0]))

        q_groups = []
        p_groups = []
        groups = []
        q_blocks = []
        p_blocks = []
        blocks = []
        h_size = sizes[0]
        h_count = 1
        for i in range(len(sizes) - 2):
            up_rate = ()
            for res_in, res_out in zip(resolutions[i], resolutions[i+1]):
                assert res_out % res_in == 0
                up_rate += (res_out//res_in,)

            q_blocks.append(
                ConvBlock(sizes[i]+h_size, math.ceil(sizes[i]*bottleneck_ratio), sizes[i+1]*2,
                    dim=dim, use_3x3=(min(resolutions[i+1]) > 2),
                    final_act=None, up_rate=None
                )
            )
            p_blocks.append(
                ConvBlock(sizes[i], math.ceil(sizes[i]*bottleneck_ratio), sizes[i+1],
                    dim=dim, use_3x3=(min(resolutions[i+1]) > 2),
                    final_act=None, up_rate=None
                )
            )
            blocks.append(
                ConvBlock(sizes[i+1], math.ceil(sizes[i+1]*bottleneck_ratio), sizes[i+1],
                    dim=dim, use_3x3=(min(resolutions[i+1]) > 2),
                    final_act=("gelu" if i < len(sizes) - 2 else None),
                    up_rate=(up_rate if math.prod(up_rate) > 1 else None)
                )
            )

            if math.prod(up_rate) > 1:
                q_group = nn.ModuleList(q_blocks)
                p_group = nn.ModuleList(p_blocks)
                group = nn.ModuleList(blocks)
                q_groups.append(q_group)
                p_groups.append(p_group)
                groups.append(group)
                q_blocks = []
                p_blocks = []
                blocks = []
                h_size = sizes[i+1]
                h_count += 1
                if h_count > n_inputs:
                    break
        if len(blocks) > 0:
            q_group = nn.ModuleList(q_blocks)
            p_group = nn.ModuleList(p_blocks)
            group = nn.ModuleList(blocks)
            q_groups.append(q_group)
            p_groups.append(p_group)
            groups.append(group)

        self.q_groups = nn.ModuleList(q_groups)
        self.p_groups = nn.ModuleList(p_groups)
        self.groups = nn.ModuleList(groups)

        final_group = []
        for j in range(i + 1, len(sizes) - 1):
            up_rate = ()
            for res_in, res_out in zip(resolutions[j], resolutions[j+1]):
                assert res_out % res_in == 0
                up_rate += (res_out//res_in,)

            final_group.append(
                ConvBlock(sizes[j], math.ceil(sizes[j]*bottleneck_ratio), sizes[j+1],
                    dim=dim, use_3x3=(min(resolutions[j+1]) > 2),
                    final_act=("gelu" if j < len(sizes) - 2 else None),
                    up_rate=(up_rate if math.prod(up_rate) > 1 else None)
                )
            )

        self.final_group = nn.Sequential(*final_group)

    def forward(self, hs, eps=1e-3):
        qs = []
        ps = []
        x = self.x_init[None, ...].expand(hs[0].shape[0], *self.x_init.size())
        for h, q_group, p_group, group in zip(reversed(hs), self.q_groups, self.p_groups, self.groups):
            for q_block, p_block, block in zip(q_group, p_group, group):
                q = q_block(torch.cat((x, h), dim=1)).chunk(2, dim=1)
                p = p_block(x)

                q_mean, q_std = q[0], F.softplus(q[1]).add(eps)
                p_mean, p_std = p, torch.ones_like(p)
                qs.append((q_mean, q_std))
                ps.append((p_mean, p_std))

                x = q_mean + q_std*torch.randn_like(q_std)
                x = block(x)

        x = self.final_group(x)
        return x, qs, ps

    def generate(self, size=1):
        x = self.x_init[None, ...].expand(size, *self.x_init.size())
        with torch.no_grad():
            for p_group, group in zip(self.p_groups, self.groups):
                for p_block, block in zip(p_group, group):
                    p = p_block(x)

                    p_mean, p_std = p, torch.ones_like(p)

                    x = p_mean + p_std*torch.randn_like(p_std)
                    x = block(x)

        x = self.final_group(x)
        return x


class HVAEConv(VAEConv):
    def __init__(self, *args, 
                 enc_hidden_sizes=None, enc_hidden_reses=None,
                 dec_hidden_sizes=None, dec_hidden_reses=None,
                 n_levels=2, **kwargs):
        assert enc_hidden_sizes is not None or dec_hidden_sizes is not None
        assert enc_hidden_reses is not None or dec_hidden_reses is not None
        if enc_hidden_sizes is None:
            enc_hidden_sizes = list(reversed(dec_hidden_sizes))
        elif dec_hidden_sizes is None:
            dec_hidden_sizes = list(reversed(enc_hidden_sizes))
        else:
            assert enc_hidden_sizes == list(reversed(dec_hidden_sizes))
        if enc_hidden_reses is None:
            enc_hidden_reses = list(reversed(dec_hidden_reses))
        elif dec_hidden_reses is None:
            dec_hidden_reses = list(reversed(enc_hidden_reses))
        else:
            assert enc_hidden_reses == list(reversed(dec_hidden_reses))

        self.n_levels = n_levels
        super().__init__(*args,
            enc_hidden_sizes=enc_hidden_sizes, enc_hidden_reses=enc_hidden_reses,
            dec_hidden_sizes=dec_hidden_sizes, dec_hidden_reses=dec_hidden_reses,
            **kwargs)

    def encode(self, x):
        return self.encoder(x)

    def sample(self, hs, size=1):
        hs = [h.repeat(size, *[1]*(h.ndim-1)) for h in hs]

        return self.decode(hs)

    def predict(self, x):
        x = x.to(next(self.parameters()).device)

        with torch.no_grad():
            hs = self.encode(x)
            x_mean, _, _ = self.decode(hs)
        return x_mean.cpu().numpy()

    def generate(self, size=1):
        with torch.no_grad():
            x_mean = self.decoder.generate(size=size)
        return x_mean.cpu().numpy()

    def interpolate(self, x1, x2, n_interps=10):
        x = torch.cat((x1, x2), dim=0)
        x = x.to(next(self.parameters()).device)

        with torch.no_grad():
            hs = self.encode(x)
            for i, h_mean in enumerate(hs):
                h1, h2 = h_mean.chunk(2, dim=0)
                h = [h1 * (1 - alpha) + h2 * alpha for alpha in torch.linspace(0, 1, n_interps)]
                h = torch.cat(h, dim=0)
                hs[i] = h
            x_mean, _, _ = self.decode(hs)
            return x_mean.view(-1, n_interps, *x.shape[1:]).cpu().numpy()

    def forward(self, x):
        x = x.to(next(self.parameters()).device)

        hs = self.encode(x)
        x_mean, qs, ps = self.sample(hs, size=self.mc_sample_size)

        return x_mean, qs, ps

    def loss_kldivs(self, qs, ps):
        return sum([self.loss_kldiv(q[0], q[1], p[0], p[1]) for q, p in zip(qs, ps)])

    def loss(self, x):
        x = x.to(next(self.parameters()).device)

        x_mean, qs, ps = self(x)

        recon_loss = self.loss_recon(x_mean, x.repeat(self.mc_sample_size, *[1]*(x.ndim-1)))
        kldiv_loss = self.loss_kldivs(qs, ps)

        return OrderedDict(loss=recon_loss+self.beta*kldiv_loss,
            recon_loss=recon_loss, kldiv_loss=kldiv_loss)

    def init_encoder(self):
        return HConvEncoder(
            sizes=[self.input_dim]+self.enc_hidden_sizes+[self.latent_dim],
            resolutions=[self.input_res]+self.enc_hidden_reses+[self.latent_res],
            n_outputs=self.n_levels
        )

    def init_decoder(self):
        return HConvDecoder(
            sizes=[self.latent_dim]+self.dec_hidden_sizes+[self.input_dim],
            resolutions=[self.latent_res]+self.dec_hidden_reses+[self.input_res],
            n_inputs=self.n_levels
        )
