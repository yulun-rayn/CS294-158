import numpy as np

import torch
import torch.nn as nn


class LogisticsMixture(nn.Module):
    def __init__(self, m, d):
        super().__init__()
        self.d = d

        self.logits = nn.Parameter(torch.zeros(m), requires_grad=True)
        self.means = nn.Parameter(torch.FloatTensor(np.linspace(start=0, stop=d, num=m)), requires_grad=True)
        self.log_scales = nn.Parameter(torch.randn(m), requires_grad=True)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = x[..., None]
        means = self.means[None, ...]
        scales = torch.exp(self.log_scales)[None, ...]

        prob_upper = torch.sigmoid((x + 0.5 - means)/scales) * (x < self.d) + torch.ones_like(means) * (x == self.d)
        prob_lower = torch.sigmoid((x - 0.5 - means)/scales) * (x > 0) #+ torch.zeros_like(means) * (x == 0)

        return ((prob_upper - prob_lower) @ torch.softmax(self.logits, dim=0))

    def loss(self, x):
        x = x.to(next(self.parameters()).device)
        return -torch.mean(torch.log(self(x)))

    def get_probs(self):
        with torch.no_grad():
            probs = self(torch.arange(self.d).to(self.logits.device))
        return (probs/torch.sum(probs)).cpu().numpy()
