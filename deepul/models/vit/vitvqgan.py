import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.nn.convolution import ResnetBlock
from deepul.models.vit import ViTVQVAE
from deepul.models.gan import Discriminator
from deepul.models.vqvae import VQGenerator, VQGAN
from deepul.models.nn.utils import LPIPS


class ViTVQGenerator(VQGenerator, nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, n_embeddings=1024, embedding_dim=64):
        nn.Module.__init__(self)
        self.loss_reduce = (0, 2, 3)

        self.vqvae = ViTVQVAE(image_size, patch_size, dim, depth, heads, mlp_dim, n_embeddings=n_embeddings, embedding_dim=embedding_dim,
                           data_dim=3, min_val=-1, max_val=1, channel_first=True, loss_reduce=self.loss_reduce)

        self.loss_perceptual = LPIPS() # takes images normalized to [-1, 1], average reduce along batch and spatial dimensions

    def loss_abs(self, x_hat, x):
        ses = torch.abs(x_hat - x)

        if self.loss_reduce == "all":
            return ses.mean()
        elif self.loss_reduce == "batch":
            return ses.mean(0).sum()
        else:
            return ses.mean(self.loss_reduce).sum()


class ViTVQDiscriminator(Discriminator, nn.Module):
    def __init__(self, data_dim=3, data_res=(32, 32), n_filters=128, n_downs=2, n_res=1, spectral_norm=True):
        nn.Module.__init__(self)
        base_res = (data_res[0] // 2**n_downs, data_res[1] // 2**n_downs)

        emb_conv = nn.Conv2d(data_dim, n_filters, kernel_size=(3, 3), padding=1)
        out_linear = nn.Linear(n_filters, 1)
        if spectral_norm:
            emb_conv = nn.utils.spectral_norm(emb_conv)
            out_linear = nn.utils.spectral_norm(out_linear)

        net  = [emb_conv]
        for _ in range(n_downs):
            net.extend([
                ResnetBlock(n_filters, n_filters=n_filters, spectral_norm=spectral_norm),
                nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
            ])
        for _ in range(n_res):
            net.append(ResnetBlock(n_filters, n_filters=n_filters, spectral_norm=spectral_norm))
        net.extend([
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=base_res, stride=base_res),
            nn.Flatten(start_dim=1, end_dim=-1),
            out_linear,
            #nn.Sigmoid()
        ])

        net = [l for l in net if l is not None]
        self.net = nn.Sequential(*net)


class ViTVQGAN(VQGAN, nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, n_embeddings=1024, embedding_dim=64,
                 min_val=0, max_val=1, n_filters=64, n_downs=2, n_res=1, beta_percept=0.5, beta_abs=0.1, beta_gan=0.1):
        nn.Module.__init__(self)
        self.beta_percept = beta_percept
        self.beta_abs = beta_abs
        self.beta_gan = beta_gan
        self.shift = (min_val + max_val) / 2.
        self.scale = (max_val - min_val) / 2.

        self.generator = ViTVQGenerator(image_size, patch_size, dim, depth, heads, mlp_dim,
            n_embeddings=n_embeddings, embedding_dim=embedding_dim)
        self.discriminator = ViTVQDiscriminator(data_res=image_size, n_filters=n_filters, n_downs=n_downs, n_res=n_res)

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
        if self.beta_abs > 0:
            loss_abs = self.generator.loss_abs(x_fake, x)
            loss = loss + self.beta_abs*loss_abs
            loss_dict.update({"abs_loss": loss_abs.item()})
        if self.beta_gan > 0:
            loss_gan = -self.discriminator(x_fake).mean()
            loss = loss + self.beta_gan*loss_gan
            loss_dict.update({"g_loss": loss_gan.item()})
        return loss, loss_dict

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
        if self.beta_abs > 0:
            loss_abs = self.generator.loss_abs(x_fake, x)
            loss = loss + self.beta_abs*loss_abs
            loss_dict.update({"abs_loss": loss_abs.item()})
        return loss, loss_dict
