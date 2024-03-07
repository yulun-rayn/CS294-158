import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.vit import ViTEncoder, ViTDecoder
from deepul.models.vqvae.quantizer import VectorQuantizer
from deepul.models.vqvae import VQVAE


class ViTVectorQuantizer(VectorQuantizer):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    """

    def __init__(self, n_e, e_dim):
        super().__init__(n_e, e_dim)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, length, channel)

        quantization pipeline:

            1. get encoder input (B,N,C)
            2. flatten input to (B*N,C)

        """
        # reshape z -> (batch, length, channel) and flatten
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(
            self.embedding.weight.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        e = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        min_encoding_indices = min_encoding_indices.view(z.shape[:-1])

        # preserve gradients
        z_q = z + (e - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return e, z_q, perplexity, min_encodings, min_encoding_indices


class ViTVQVAE(VQVAE, nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        n_embeddings,
        embedding_dim,
        beta=1.0,
        data_dim=3,
        min_val=0,
        max_val=3,
        channel_first=False,
        loss_reduce="all"
    ):
        nn.Module.__init__(self)
        # encode image into continuous latent space
        self.encoder = ViTEncoder(image_size, patch_size, dim, depth, heads, mlp_dim, latent_dim=embedding_dim, channels=data_dim)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = ViTVectorQuantizer(n_embeddings, embedding_dim)
        # decode the discrete latent representation
        self.decoder = ViTDecoder(image_size, patch_size, dim, depth, heads, mlp_dim, latent_dim=embedding_dim, channels=data_dim)

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.channel_first = channel_first
        self.loss_reduce = loss_reduce
        self.shift = (min_val + max_val) / 2.

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize an image x.

        Args:
            x (np.ndarray, dtype=int): Image to quantize. shape=(batch_size, 28, 28, 3).

        Returns:
            np.ndarray: Quantized image. shape=(batch_size, 7, 7, 3)
        """
        x = torch.FloatTensor(x).to(next(self.parameters()).device)
        x = self.transform(x)
        with torch.no_grad():
            z_e = self.encoder(x)
            _, _, _, _, z_index = self.vector_quantization(z_e)
        return z_index

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.transform(x)

        z_e = self.encoder(x)
        e, z_q, _, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        x_hat = self.reverse(x_hat)
        return x_hat, z_e, e
