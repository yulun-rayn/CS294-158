import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.ae import VAE
from deepul.models.nn.convolution import MLConv


class VAEConv(VAE):
    def __init__(self, input_dim, input_res, latent_dim=None, latent_res=None,
                 enc_hidden_sizes=[], enc_hidden_reses=[],
                 dec_hidden_sizes=[], dec_hidden_reses=[],
                 beta=1.0, mc_sample_size=5, loss_reduce="all"):
        self.input_res = input_res
        self.latent_res = ([input_res]+enc_hidden_reses)[-1] if latent_res is None else latent_res
        self.enc_hidden_reses = enc_hidden_reses
        self.dec_hidden_reses = dec_hidden_reses
        super().__init__(input_dim, latent_dim, enc_hidden_sizes, dec_hidden_sizes, beta, mc_sample_size, loss_reduce)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, mu=None, sigma=None, size=1):
        if mu is None:
            mu = torch.zeros((1, self.latent_dim, *self.latent_res), device=next(self.parameters()).device)
        if sigma is None:
            sigma = torch.ones((1, self.latent_dim, *self.latent_res), device=next(self.parameters()).device)
        mu = mu.repeat(size, *[1]*(mu.ndim-1))
        sigma = sigma.repeat(size, *[1]*(sigma.ndim-1))

        latents = self.reparameterize(mu, sigma)
        return self.decode(latents)

    def predict(self, x):
        x = x.to(next(self.parameters()).device)

        with torch.no_grad():
            z_dist = self.encode(x)
            x_mean = self.decode(z_dist.mean)
        return x_mean.cpu().numpy()

    def generate(self, size=1):
        with torch.no_grad():
            x_mean = self.sample(size=size)
        return x_mean.cpu().numpy()

    def interpolate(self, x1, x2, n_interps=10):
        x = torch.cat((x1, x2), dim=0)
        x = x.to(next(self.parameters()).device)

        with torch.no_grad():
            z_dist = self.encode(x)
            z1, z2 = z_dist.mean.chunk(2, dim=0)
            z = [z1 * (1 - alpha) + z2 * alpha for alpha in torch.linspace(0, 1, n_interps)]
            z = torch.cat(z, dim=0)
            x_mean = self.decode(z)
            return x_mean.view(-1, n_interps, *x.shape[1:]).cpu().numpy()

    def loss_recon(self, x_mean, x):
        ses = F.mse_loss(x_mean, x, reduction="none")

        if self.loss_reduce == "all":
            return ses.mean()
        elif self.loss_reduce == "batch":
            return ses.mean(0).sum()
        else:
            return ses.mean(self.loss_reduce).sum()

    def init_encoder(self):
        return MLConv(
            sizes=[self.input_dim]+self.enc_hidden_sizes+[self.latent_dim*2],
            resolutions=[self.input_res]+self.enc_hidden_reses+[self.latent_res]
        )

    def init_decoder(self):
        return MLConv(
            sizes=[self.latent_dim]+self.dec_hidden_sizes+[self.input_dim],
            resolutions=[self.latent_res]+self.dec_hidden_reses+[self.input_res]
        )
