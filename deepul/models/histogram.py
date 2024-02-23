import torch
import torch.nn as nn
import torch.nn.functional as F


class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(d), requires_grad=True)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        return self.logits[None, ...].repeat(x.shape[0], *self.logits.size())

    def loss(self, x):
        x = x.to(next(self.parameters()).device)
        return F.cross_entropy(self(x), x)

    def get_probs(self):
        return F.softmax(self.logits, dim=0).detach().cpu().numpy()
