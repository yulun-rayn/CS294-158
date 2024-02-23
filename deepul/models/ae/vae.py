from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from deepul.models.nn.mlp import MLP


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=None, enc_hidden_sizes=[],
                 dec_hidden_sizes=[], beta=1.0, mc_sample_size=5, loss_reduce="all"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = ([input_dim]+enc_hidden_sizes)[-1] if latent_dim is None else latent_dim
        self.enc_hidden_sizes = enc_hidden_sizes
        self.dec_hidden_sizes = dec_hidden_sizes
        self.beta = beta
        self.mc_sample_size = mc_sample_size
        self.loss_reduce = loss_reduce

        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    def distributionize(self, constructions, eps=1e-3):
        locs = constructions[0]
        scales = F.softplus(constructions[1]).add(eps)
        return Normal(loc=locs, scale=scales)

    def encode(self, x, distributionize=True):
        z = self.encoder(x).chunk(2, dim=1)
        return self.distributionize(z) if distributionize else z

    def decode(self, z, distributionize=True):
        x = self.decoder(z).chunk(2, dim=1)
        return self.distributionize(x) if distributionize else x

    def sample(self, mu=None, sigma=None, size=1, distributionize=True):
        if mu is None:
            mu = torch.zeros((1, self.latent_dim), device=next(self.parameters()).device)
        if sigma is None:
            sigma = torch.ones((1, self.latent_dim), device=next(self.parameters()).device)
        mu = mu.repeat(size, *[1]*(mu.ndim-1))
        sigma = sigma.repeat(size, *[1]*(sigma.ndim-1))

        latents = self.reparameterize(mu, sigma)
        return self.decode(latents, distributionize=distributionize)

    def predict(self, x):
        x = x.to(next(self.parameters()).device)

        with torch.no_grad():
            z_dist = self.encode(x)
            x_dist = self.decode(z_dist.mean)
        return x_dist.mean.cpu().numpy()

    def generate(self, size=1, noise=False):
        with torch.no_grad():
            x_dist = self.sample(size=size)
        return x_dist.sample().cpu().numpy() if noise else x_dist.mean.cpu().numpy()

    def interpolate(self, x1, x2, n_interps=10):
        x = torch.cat((x1, x2), dim=0)
        x = x.to(next(self.parameters()).device)

        with torch.no_grad():
            z_dist = self.encode(x)
            z1, z2 = z_dist.mean.chunk(2, dim=0)
            z = [z1 * (1 - alpha) + z2 * alpha for alpha in torch.linspace(0, 1, n_interps)]
            z = torch.cat(z, dim=0)
            x_dist = self.decode(z)
            return x_dist.mean.view(-1, n_interps, *x.shape[1:]).cpu().numpy()

    def forward(self, x):
        x = x.to(next(self.parameters()).device)

        z_dist = self.encode(x)
        x_dist = self.sample(z_dist.mean, z_dist.stddev, size=self.mc_sample_size)

        return x_dist, z_dist

    def loss_recon(self, x_dist, x):
        nllhs = -x_dist.log_prob(x)

        if self.loss_reduce == "all":
            return nllhs.mean()
        else:
            return nllhs.sum(tuple(range(1, nllhs.ndim))).mean()

    def loss_kldiv(self, q_mean, q_std, p_mean=None, p_std=None):
        if p_mean is None:
            p_mean = torch.zeros_like(q_mean)
        if p_std is None:
            p_std = torch.ones_like(q_std)

        var_ratio = (q_std / p_std).pow(2)
        t1 = ((q_mean - p_mean) / p_std).pow(2)
        kldivs = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

        if self.loss_reduce == "all":
            return kldivs.mean()
        else:
            return kldivs.sum(tuple(range(1, kldivs.ndim))).mean()

    def loss(self, x):
        x = x.to(next(self.parameters()).device)

        x_dist, z_dist = self(x)

        recon_loss = self.loss_recon(x_dist, x.repeat(self.mc_sample_size, *[1]*(x.ndim-1)))
        kldiv_loss = self.loss_kldiv(z_dist.mean, z_dist.stddev)

        return OrderedDict(loss=recon_loss+self.beta*kldiv_loss,
            recon_loss=recon_loss, kldiv_loss=kldiv_loss)

    def init_encoder(self):
        return MLP([self.input_dim]+self.enc_hidden_sizes+[self.latent_dim*2])

    def init_decoder(self):
        return MLP([self.latent_dim]+self.dec_hidden_sizes+[self.input_dim*2])
