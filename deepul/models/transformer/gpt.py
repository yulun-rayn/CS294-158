from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.transformer.block import TransformerBlock
from deepul.models.utils import Embedding, LayerNorm


class GPT(nn.Module):
    """
    The GPT Model.
    """
    def __init__(self, n_classes, bos_ind=None, eos_ind=-1, max_len=1000, d_model=128, n_layers=2, n_heads=4):
        """Initializes a new GPT instance.
        """
        super().__init__()
        if bos_ind is None:
            bos_ind = eos_ind
        self.bos_ind = bos_ind
        self.eos_ind = eos_ind
        self.max_len = max_len
        self.d_model = d_model

        self._input = nn.Sequential(
			Embedding(n_classes, d_model),
			LayerNorm(d_model)
		)

        self._pos = nn.Parameter(torch.zeros(1, d_model, max_len+1))

        self._transformer = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(d_model, n_heads),
                LayerNorm(d_model)
            ])
            for _ in range(n_layers)
        ])

        self._out = nn.Conv1d(d_model, n_classes, 1, 1, 0)

    def forward(self, x, kv_cache=None, pos=None):
        x = x.to(next(self.parameters()).device)
        _, l = x.size()

        x = self._input(x)
        if pos is None:
            x = x + self._pos[:, :, :l]
        else:
            x = x[:, :, pos, None] + self._pos[:, :, pos, None]
        if kv_cache:
            for i, ((block, norm), cache) in enumerate(zip(self._transformer, kv_cache)):
                x, cache = block(x, kv_cache=cache, pos=(pos,))
                kv_cache[i] = cache
                x = norm(x)
        else:
            for i, (block, norm) in enumerate(self._transformer):
                x = block(x)
                x = norm(x)
        x = self._out(x)
        if kv_cache:
            return x, kv_cache
        else:
            return x

    def loss(self, x):
        x = x.to(next(self.parameters()).device)

        nll = F.cross_entropy(self(x[:, :-1]), x[:, 1:], reduction='none')
        masked_nll = nll * (x[:, :-1] != self.eos_ind).float()
        return masked_nll.mean()

    def sample(self, n_samples, max_len=500, kv_cache=False, track_time=False):
        assert max_len <= self.max_len

        device = next(self.parameters()).device
        if kv_cache:
            kv_cache = [
                [torch.zeros((n_samples, self.d_model, max_len+1), device=device)
                    for _ in range(2)
                ]
                for _ in range(len(self._transformer))
            ]

        samples = torch.cat(
            [torch.full((n_samples, 1), self.bos_ind, device=device, dtype=torch.long),
             torch.full((n_samples, max_len+1), self.eos_ind, device=device, dtype=torch.long)],
            dim=1
        )
        eos_flags = torch.zeros(n_samples, device=device, dtype=torch.long)
        if track_time:
            time_list = []
            start_time = datetime.now()
        with torch.no_grad():
            for i in range(max_len+1):
                if kv_cache:
                    logits, kv_cache = self(samples, kv_cache=kv_cache, pos=i)
                    probs = F.softmax(logits[:, :, 0], dim=1)
                else:
                    logits = self(samples[:, :-1])
                    probs = F.softmax(logits[:, :, i], dim=1)
                samples[:, (i+1)] = torch.multinomial(probs, 1)[:, 0]

                if track_time:
                    end_time = datetime.now()
                    time_list.append((end_time-start_time).total_seconds())
                    start_time = end_time

                eos_flags = eos_flags | (samples[:, (i+1)]==self.eos_ind)
                if sum(~eos_flags) == 0:
                    break
            i = i + 1

        samples = samples[:, :(i+1)]
        if sum(~eos_flags) > 0:
            samples[:, i] = torch.full((n_samples, 1), self.eos_ind, device=device, dtype=torch.long)
        if track_time:
            return samples.cpu().numpy(), time_list
        else:
            return samples.cpu().numpy()
