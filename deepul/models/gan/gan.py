import torch
import torch.nn as nn

from deepul.models.nn.mlp import MLP


class Generator(nn.Module):
    def __init__(self, latent_dim, n_hidden, hidden_size, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = MLP([latent_dim] + [hidden_size]*n_hidden + [output_dim],
            batch_norm=False, act="leaky_relu", final_act="tanh")

    def forward(self, n):
        z = torch.normal(
            mean=torch.zeros(n, self.latent_dim),
            std=torch.ones(n, self.latent_dim)
        ).to(next(self.parameters()).device)
        return self.net.forward(z)

    def sample(self, n):
        with torch.no_grad():
            x = self.forward(n)
        return x.cpu().numpy()


class Discriminator(nn.Module):
    def __init__(self, input_dim, n_hidden, hidden_size):
        super().__init__()
        self.net = MLP([input_dim] + [hidden_size]*n_hidden + [1],
            batch_norm=False, act="leaky_relu", final_act="sigmoid")
    
    def forward(self, x):
        return self.net.forward(x).squeeze(-1)

    def critic(self, x):
        with torch.no_grad():
            p = self.forward(x)
        return p.cpu().numpy()


class GAN(nn.Module):
    def __init__(self, data_dim, latent_dim, n_hidden, hidden_size, saturation=True):
        super().__init__()
        self.saturation = saturation

        self.generator = Generator(latent_dim, n_hidden, hidden_size, data_dim)
        self.discriminator = Discriminator(data_dim, n_hidden, hidden_size)

    def forward(self, n):
        x_fake = self.generator.forward(n)
        return x_fake

    def sample(self, n):
        return self.generator.sample(n)

    def critic(self, x):
        return self.discriminator.critic(x)

    def loss_generator(self, x):
        x_fake = self.forward(x.shape[0])
        if self.saturation:
            loss = (1 - self.discriminator(x_fake)).log().mean()
        else:
            loss = -self.discriminator(x_fake).log().mean()
        return loss, {"g_loss": loss.item()}

    def loss_discriminator(self, x):
        x_fake = self.forward(x.shape[0])
        loss = -(
            self.discriminator(x).log().mean() +
            (1 - self.discriminator(x_fake)).log().mean()
        )
        return loss, {"d_loss": loss.item()}
