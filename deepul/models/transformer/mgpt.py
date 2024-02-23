import copy
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.transformer.block import TransformerBlock
from deepul.models.nn.convolution import CausalConv2d
from deepul.models.utils import Embedding, LayerNorm


class MultimodalGPT(nn.Module):
    """
    The MultimodalGPT Model.
    """
    def __init__(self, image_shape, n_colors, n_classes, bos_ind=None, eos_ind=-1,
                 max_len=1000, d_model=128, n_layers=2, n_heads=4, kernel_size=7):
        """Initializes a new MultimodalGPT instance.
        """
        super().__init__()
        self.image_shape = image_shape
        if bos_ind is None:
            bos_ind = eos_ind
        self.bos_ind = bos_ind
        self.eos_ind = eos_ind
        self.max_len = max_len
        self.d_model = d_model

        self._input = nn.ModuleDict({
            "img": nn.Sequential(
                    CausalConv2d(True, image_shape[0], d_model, kernel_size, 1, kernel_size//2),
                    LayerNorm(d_model)
                ),
            "txt": nn.Sequential(
                    Embedding(n_classes, d_model),
                    LayerNorm(d_model)
                )
        })

        self._pos = nn.ParameterDict({
            "img": nn.Parameter(torch.zeros(1, d_model, *image_shape[1:])),
            "txt": nn.Parameter(torch.zeros(1, d_model, max_len+2))
        })

        self._transformer = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(d_model, n_heads),
                LayerNorm(d_model)
            ])
            for _ in range(n_layers)
        ])

        self._out = nn.ModuleDict({
            "img": nn.Conv2d(d_model, n_colors*image_shape[0], 1, 1, 0),
            "txt": nn.Conv1d(d_model, n_classes, 1, 1, 0)
        })

    def forward(self, x: OrderedDict, kv_cache=None, pos=None):
        x["img"] = x["img"].to(next(self.parameters()).device)
        x["txt"] = x["txt"].to(next(self.parameters()).device)
        n, c, h, w = x["img"].size()
        _, l = x["txt"].size()
        first = list(x.items())[0][0]

        x = OrderedDict(x.items())
        x["img"] = x["img"].float()
        if pos is None:
            x["img"] = self._input["img"](x["img"])
            x["txt"] = self._input["txt"](x["txt"])

            x["img"] = x["img"] + self._pos["img"]
            x["txt"] = x["txt"] + self._pos["txt"][:, :, :l]
            
            if first == "img":
                x = torch.cat([x["img"].view(n, -1, h*w), x["txt"]], dim=-1)
            else:
                x = torch.cat([x["txt"], x["img"].view(n, -1, h*w)], dim=-1)
        else:
            if first == "img":
                if pos < h*w:
                    x = self._input["img"](x["img"])
                    x = x[:, :, pos//h, pos%h, None] + self._pos["img"][:, :, pos//h, pos%h, None]
                    pos = ("img", pos)
                else:
                    x = self._input["txt"](x["txt"])
                    x = x[:, :, pos-h*w, None] + self._pos["txt"][:, :, pos-h*w, None]
                    pos = ("txt", pos)
            else:
                if pos < l:
                    x = self._input["txt"](x["txt"])
                    x = x[:, :, pos, None] + self._pos["txt"][:, :, pos, None]
                    pos = ("txt", pos)
                else:
                    x = self._input["img"](x["img"])
                    x = x[:, :, (pos-l)//h, (pos-l)%h, None] + self._pos["img"][:, :, (pos-l)//h, (pos-l)%h, None]
                    pos = ("img", pos)

        if kv_cache:
            for i, ((block, norm), cache) in enumerate(zip(self._transformer, kv_cache)):
                x, cache = block(x, kv_cache=cache, pos=(pos[1],))
                kv_cache[i] = cache
                x = norm(x)
        else:
            for i, (block, norm) in enumerate(self._transformer):
                x = block(x)
                x = norm(x)

        if pos is None:
            if first == "img":
                x = OrderedDict([("img", x[:, :, :(h*w)].view(n,-1,h,w)), ("txt", x[:, :, (h*w):])])
            else:
                x = OrderedDict([("txt", x[:, :, :l]), ("img", x[:, :, l:].view(n,-1,h,w))])

            x["img"] = self._out["img"](x["img"]).view(n, -1, c, h, w)
            x["txt"] = self._out["txt"](x["txt"])
        else:
            if pos[0] == "img":
                x = x[:, :, :, None]
            x = self._out[pos[0]](x)
            if pos[0] == "img":
                x = x.view(n, -1, c, *x.shape[2:])
        if kv_cache:
            return x, kv_cache
        else:
            return x

    def loss(self, x):
        x["img"] = x["img"].to(next(self.parameters()).device)
        x["txt"] = x["txt"].to(next(self.parameters()).device)

        logits = self(x)
        # image
        image_nll = F.cross_entropy(logits["img"], x["img"])
        # text
        text_nll = F.cross_entropy(logits["txt"][:, :, :-1], x["txt"][:, 1:], reduction='none')
        text_nll = text_nll * (x["txt"][:, :-1] != self.eos_ind).float()
        return (image_nll + text_nll.mean()) / 2.

    def sample(self, n_samples=None, image_prompt=None, text_prompt=None, max_len=500, image_first=True):
        assert max_len <= self.max_len
        assert (n_samples is not None)+(image_prompt is not None)+(text_prompt is not None) == 1
        if image_prompt is not None:
            n_samples = len(image_prompt)
        if text_prompt is not None:
            n_samples = len(text_prompt)
        full_len = self.image_shape[1]*self.image_shape[2]+max_len

        device = next(self.parameters()).device
        kv_cache = [
            [torch.zeros((n_samples, self.d_model, full_len+2), device=device)
                for _ in range(2)
            ]
            for _ in range(len(self._transformer))
        ]

        if image_prompt is None:
            img_samples = torch.zeros((n_samples, *self.image_shape), device=device)
        else:
            img_samples = image_prompt
        if text_prompt is None:
            txt_samples = torch.cat(
                [torch.full((n_samples, 1), self.bos_ind, device=device, dtype=torch.long),
                torch.full((n_samples, max_len+1), self.eos_ind, device=device, dtype=torch.long)],
                dim=1
            )
        else:
            txt_samples = text_prompt

        eos_flags = torch.zeros(n_samples, device=device, dtype=torch.long)
        if (image_prompt is not None) or (image_prompt is None and text_prompt is None and image_first):
            samples = OrderedDict([("img", img_samples), ("txt", txt_samples)])
            pos = 0
            with torch.no_grad():
                for r in range(self.image_shape[1]):
                    for c in range(self.image_shape[2]):
                        for k in range(self.image_shape[0]):
                            logits, kv_cache = self(samples, kv_cache=kv_cache, pos=pos)
                            if image_prompt is None:
                                probs = F.softmax(logits[:, :, k, 0, 0], dim=1)
                                samples["img"][:, k, r, c] = torch.multinomial(probs, 1)[:, 0]
                        pos = pos + 1
                for i in range(max_len+1):
                    logits, kv_cache = self(samples, kv_cache=kv_cache, pos=pos)
                    if text_prompt is None:
                        probs = F.softmax(logits[:, :, 0], dim=1)
                        samples["txt"][:, (i+1)] = torch.multinomial(probs, 1)[:, 0]
                    pos = pos + 1
                    eos_flags = eos_flags | (samples["txt"][:, (i+1)]==self.eos_ind)
                    if sum(~eos_flags) == 0:
                        break
                i = i + 1
                samples["txt"] = samples["txt"][:, :(i+1)]
                if sum(~eos_flags) > 0:
                    samples["txt"][:, i] = torch.full((n_samples, 1), self.eos_ind, device=device, dtype=torch.long)
                pos = pos + 1
        if (text_prompt is not None) or (image_prompt is None and text_prompt is None and not image_first):
            samples = OrderedDict([("txt", txt_samples), ("img", img_samples)])
            pos = 0
            with torch.no_grad():
                for i in range(max_len+1):
                    logits, kv_cache = self(samples, kv_cache=kv_cache, pos=pos)
                    if text_prompt is None:
                        probs = F.softmax(logits[:, :, 0], dim=1)
                        samples["txt"][:, (i+1)] = torch.multinomial(probs, 1)[:, 0]
                    pos = pos + 1
                    eos_flags = eos_flags | (samples["txt"][:, (i+1)]==self.eos_ind)
                    if sum(~eos_flags) == 0:
                        break
                i = i + 1
                samples["txt"] = samples["txt"][:, :(i+1)]
                if sum(~eos_flags) > 0:
                    samples["txt"][:, i] = torch.full((n_samples, 1), self.eos_ind, device=device, dtype=torch.long)
                pos = pos + 1
                for r in range(self.image_shape[1]):
                    for c in range(self.image_shape[2]):
                        for k in range(self.image_shape[0]):
                            logits, kv_cache = self(samples, kv_cache=kv_cache, pos=pos)
                            probs = F.softmax(logits[:, :, k, 0, 0], dim=1)
                            if image_prompt is None:
                                samples["img"][:, k, r, c] = torch.multinomial(probs, 1)[:, 0]
                        pos = pos + 1

        return samples["img"].cpu().numpy(), samples["txt"].cpu().numpy()
