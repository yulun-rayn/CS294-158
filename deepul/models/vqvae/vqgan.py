import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.vqvae import VQVAE
from deepul.models.gan import GAN
from deepul.models.nn.utils import LPIPS


class VQGenerator(nn.Module):
    def __init__(self, h_dim=256, res_h_dim=32, n_res_layers=3, n_embeddings=1024, embedding_dim=64):
        super().__init__()
        self.vqvae = VQVAE(h_dim=h_dim, res_h_dim=res_h_dim, n_res_layers=n_res_layers, n_embeddings=n_embeddings, embedding_dim=embedding_dim,
                           data_dim=3, min_val=-1, max_val=1, channel_first=True, loss_reduce=(0, 2, 3))

        self.loss_perceptual = LPIPS() # takes images normalized to [-1, 1], average reduce along batch and spatial dimensions

    def forward(self, x):
        return self.vqvae.forward(x)

    def predict(self, x):
        return self.vqvae.predict(x)

    def loss_recon(self, x_hat, x):
        return self.vqvae.loss_recon(x_hat, x)

    def loss_emb(self, z_e, e):
        return self.vqvae.loss_emb(z_e, e)

    def loss_percept(self, x_hat, x):
        return self.loss_perceptual(x_hat, x).mean()


class VQDiscriminator(nn.Module):
    """PatchGAN Discriminator

    Credit: https://github.com/dome272/VQGAN-pytorch/blob/main/discriminator.py.
    """
    def __init__(self, num_filters_last=64, n_downs=2):
        super().__init__()

        layers = [nn.Conv2d(3, num_filters_last, 3, 1, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_downs + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult,
                          4, 2, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def critic(self, x):
        with torch.no_grad():
            p = self.forward(x)
        return p.cpu().numpy()


class VQGAN(GAN, nn.Module):
    def __init__(self, h_dim=256, res_h_dim=32, n_res_layers=3, n_embeddings=1024, embedding_dim=64,
                 min_val=0, max_val=1, num_filters_last=64, n_downs=2, beta_percept=0.5, beta_gan=0.1):
        nn.Module.__init__(self)
        self.beta_percept = beta_percept
        self.beta_gan = beta_gan
        self.shift = (min_val + max_val) / 2.
        self.scale = (max_val - min_val) / 2.

        self.generator = VQGenerator(h_dim=h_dim, res_h_dim=res_h_dim, n_res_layers=n_res_layers,
            n_embeddings=n_embeddings, embedding_dim=embedding_dim)
        self.discriminator = VQDiscriminator(num_filters_last=num_filters_last, n_downs=n_downs)

    def predict(self, x):
        x = self.transform(x)
        x = self.generator.predict(x)
        x = self.reverse(x)
        return x

    def loss_generator(self, x):
        x = self.transform(x)

        x_fake, z_e, e = self.forward(x)
        loss_recon = self.generator.loss_recon(x_fake, x)
        loss_emb = self.generator.loss_emb(z_e, e)
        loss = loss_recon + loss_emb
        loss_dict = {"recon_loss": loss_recon.item(), "emb_loss": loss_emb.item()}
        if self.beta_percept > 0:
            loss_percept = self.generator.loss_percept(x_fake, x)
            loss = loss + self.beta_percept*loss_percept
            loss_dict.update({"percept_loss": loss_percept.item()})
        if self.beta_gan > 0:
            loss_gan = -self.discriminator(x_fake).mean()
            loss = loss + self.beta_gan*loss_gan
            loss_dict.update({"g_loss": loss_gan.item()})
        return loss, loss_dict

    def loss_discriminator(self, x):
        x = self.transform(x)

        x_fake, _, _ = self.forward(x)
        loss = F.softplus(-self.discriminator(x)).mean() + F.softplus(self.discriminator(x_fake)).mean()
        return loss, {"d_loss": loss.item()}

    def loss(self, x):
        x = self.transform(x)

        x_fake, z_e, e = self.forward(x)
        loss_recon = self.generator.loss_recon(x_fake, x)
        loss_emb = self.generator.loss_emb(z_e, e)
        loss = loss_recon + loss_emb
        loss_dict = {"recon_loss": loss_recon.item(), "emb_loss": loss_emb.item()}
        if self.beta_percept > 0:
            loss_percept = self.generator.loss_percept(x_fake, x)
            loss = loss + self.beta_percept*loss_percept
            loss_dict.update({"percept_loss": loss_percept.item()})
        return loss, loss_dict

    def transform(self, x):
        return (x - self.shift)/self.scale

    def reverse(self, x):
        return x*self.scale + self.shift
