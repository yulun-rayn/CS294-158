import torch
import torch.nn as nn

from deepul.models.nn.convolution import ResnetBlockUp, ResnetBlockDown, ResnetBlock
from deepul.models.gan import Generator, Discriminator, GAN
from deepul.models.utils import Summation


class GeneratorConv(Generator, nn.Module):
    def __init__(self, latent_dim=128, n_filters=256, n_ups=3, n_res=0, data_dim=3, base_res=(4,4), batch_norm=False):
        nn.Module.__init__(self)
        self.latent_dim = latent_dim

        net = [
            nn.Linear(latent_dim, n_filters*base_res[0]*base_res[1]),
            nn.Unflatten(1, (n_filters, base_res[0], base_res[1]))
        ]
        for _ in range(n_ups):
            net.append(ResnetBlockUp(in_dim=n_filters, n_filters=n_filters, batch_norm=batch_norm))
        for _ in range(n_res):
            net.append(ResnetBlock(in_dim=n_filters, n_filters=n_filters, batch_norm=batch_norm))
        net.extend([
            nn.BatchNorm2d(n_filters) if batch_norm else None,
            nn.ReLU(),
            nn.Conv2d(n_filters, data_dim, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        ])

        net = [l for l in net if l is not None]
        self.net = nn.Sequential(*net)


class DiscriminatorConv(Discriminator, nn.Module):
    def __init__(self, n_filters=128, n_downs=2, n_res=1, data_dim=3, batch_norm=False):
        nn.Module.__init__(self)

        net = [nn.Conv2d(data_dim, n_filters, kernel_size=(3, 3), padding=1)]
        for _ in range(n_downs):
            net.append(ResnetBlockDown(in_dim=n_filters, n_filters=n_filters, batch_norm=batch_norm))
        for _ in range(n_res):
            net.append(ResnetBlock(in_dim=n_filters, n_filters=n_filters, batch_norm=batch_norm))
        net.extend([
            nn.BatchNorm2d(n_filters) if batch_norm else None,
            nn.ReLU(),
            Summation((2,3)),
            nn.Linear(n_filters, 1)
            #nn.Sigmoid()
        ])

        net = [l for l in net if l is not None]
        self.net = nn.Sequential(*net)


class GANConv(GAN, nn.Module):
    def __init__(self, data_dim=3, datas_res=(32, 32), latent_dim=128, g_filters=256, d_filters=128, g_ups=3, d_downs=2, g_res=0, d_res=1, grad_penalty=10.0):
        nn.Module.__init__(self)
        self.grad_penalty = grad_penalty
        base_res = (datas_res[0] // 2**g_ups, datas_res[0] // 2**g_ups)

        self.generator = GeneratorConv(latent_dim=latent_dim, n_filters=g_filters, n_ups=g_ups, n_res=g_res, data_dim=data_dim, base_res=base_res, batch_norm=True)
        self.discriminator = DiscriminatorConv(n_filters=d_filters, n_downs=d_downs, n_res=d_res, data_dim=data_dim, batch_norm=grad_penalty==0)

    def loss_generator(self, x):
        x_fake = self.forward(x.shape[0])
        loss = -self.discriminator(x_fake).mean()
        return loss, {"g_loss": loss.item()}

    def loss_discriminator(self, x):
        x_fake = self.forward(x.shape[0])
        loss = self.discriminator(x_fake).mean() - self.discriminator(x).mean()
        if self.grad_penalty > 0:
            loss = loss + self.grad_penalty*self.gradient_penalty(x, x_fake)
        return loss, {"d_loss": loss.item()}

    def gradient_penalty(self, x, x_fake, eps=1e-12):
        batch_size = x.shape[0]

        _eps = torch.rand(batch_size, 1, 1, 1).to(next(self.parameters()).device)
        _eps = _eps.expand_as(x)
        interpolated = _eps * x.data + (1 - _eps) * x_fake.data
        interpolated.requires_grad = True

        score = self.discriminator(interpolated)
        gradients = torch.autograd.grad(outputs=score, inputs=interpolated,
                                        grad_outputs=torch.ones(score.size()).to(next(self.parameters()).device),
                                        create_graph=True, retain_graph=True)[0]

        gradients = gradients.reshape(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + eps)
        return ((gradients_norm - 1) ** 2).mean()
