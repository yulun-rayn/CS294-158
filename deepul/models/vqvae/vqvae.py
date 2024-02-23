from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.vqvae.decoder import Decoder
from deepul.models.vqvae.encoder import Encoder
from deepul.models.vqvae.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        n_embeddings,
        embedding_dim,
        beta=1.0,
        min_val=0,
        max_val=3,
        save_img_embedding_map=False,
        loss_reduce="all"
    ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1
        )
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.loss_reduce = loss_reduce
        self.offset = (max_val - min_val) / 2.

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
            z_e = self.pre_quantization_conv(z_e)
            _, _, _, _, z_index = self.vector_quantization(z_e)
        return z_index

    def decode(self, z_index: np.ndarray) -> np.ndarray:
        """Decode a quantized image.

        Args:
            z_index (np.ndarray, dtype=int): Quantized image. shape=(batch_size, 7, 7). Values in [0, n_embeddings].

        Returns:
            np.ndarray: Decoded image. shape=(batch_size, 28, 28, 3).
        """
        z_index = torch.LongTensor(z_index).to(next(self.parameters()).device)
        with torch.no_grad():
            z_q = self.vector_quantization.embedding(z_index).permute(0, 3, 1, 2)
            x_hat = self.decoder(z_q)
        x_hat = self.reverse(x_hat)
        return x_hat.cpu().numpy()

    def predict(self, x):
        with torch.no_grad():
            x_hat, _, _ = self(x)
        return x_hat.cpu().numpy()

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.transform(x)

        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        e, z_q, _, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        x_hat = self.reverse(x_hat)
        return x_hat, z_e, e

    def loss_recon(self, x_hat, x):
        ses = (x_hat - x)**2

        if self.loss_reduce == "all":
            return ses.mean()
        else:
            return ses.sum(tuple(range(1, ses.ndim))).mean()
    
    def loss_emb(self, z_e, e):
        ses = (z_e.detach() - e)**2 + self.beta * (z_e - e.detach())**2

        if self.loss_reduce == "all":
            return ses.mean()
        else:
            return ses.sum(tuple(range(1, ses.ndim))).mean()

    def loss(self, x):
        x_hat, z_e, e = self(x)

        recon_loss = self.loss_recon(x_hat, x)
        emb_loss = self.loss_emb(z_e, e)

        return OrderedDict(loss=recon_loss+emb_loss,
            recon_loss=recon_loss, kldiv_loss=emb_loss)

    def transform(self, x):
        return x.permute(0, 3, 1, 2) - self.offset

    def reverse(self, x):
        return x.permute(0, 2, 3, 1) + self.offset
