from functools import partial
from random import random

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.diffusion.utils import default


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        size=(),
        timesteps=None,            # int: discrete, None: continuous
        objective='pred_v',
        offset_noise_strength=0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        clip_range=None
    ):
        super().__init__()
        self.model = model
        assert not (type(self) == GaussianDiffusion and self.model.in_dim != self.model.out_dim)
        assert not hasattr(self.model, 'random_or_learned_sinusoidal_cond') or not self.model.random_or_learned_sinusoidal_cond

        self.dim = self.model.in_dim
        self.self_condition = self.model.self_condition

        self.size = size
        self.objective = objective
        self.offset_noise_strength = offset_noise_strength

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict x0) or pred_v (predict velocity)'

        self.num_timesteps = int(timesteps) if timesteps is not None else None

        # derive loss weight
        # snr - signal noise ratio

        if objective == 'pred_noise':
            self.snr_fn = lambda alpha, sigma: 1.
        elif objective == 'pred_x0':
            self.snr_fn = lambda alpha, sigma: alpha**2 / sigma**2
        elif objective == 'pred_v':
            self.snr_fn = lambda alpha, sigma: alpha**2

        if clip_range is not None:
            clip_min, clip_max = clip_range
            self.clip_fn = partial(torch.clamp, min=clip_min, max=clip_max)
        else:
            self.clip_fn = None

    @property
    def device(self):
        return self.model.device

    def _get_alpha_sigma(self, t):
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

    def _expand(self, t):
        for _ in range(len(self.size)+1):
            t = t[..., None]
        return t

    def get_x(self, x_0, noise, alpha_t, sigma_t):
        return alpha_t * x_0 + sigma_t * noise

    def get_v(self, x_0, noise, alpha_t, sigma_t):
        return alpha_t * noise - sigma_t * x_0

    def predict_start_from_v(self, x_t, v, alpha_t, sigma_t):
        return alpha_t * x_t - sigma_t * v

    def predict_noise_from_v(self, x_t, v, alpha_t, sigma_t):
        return sigma_t * x_t + alpha_t * v

    def predict_start_from_noise(self, x_t, noise, alpha_t, sigma_t):
        return (x_t - sigma_t * noise) / alpha_t

    def predict_noise_from_start(self, x_t, x_0, alpha_t, sigma_t):
        return (x_t - alpha_t * x_0) / sigma_t

    def forward(self, x, model_output, alpha_t, sigma_t, rederive_pred_noise=False):
        if self.objective == 'pred_noise':
            pred_noise = model_output

            x_0 = self.predict_start_from_noise(x, pred_noise, alpha_t, sigma_t)
            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
                if rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(x, x_0, alpha_t, sigma_t)

        elif self.objective == 'pred_x0':
            x_0 = model_output

            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
            pred_noise = self.predict_noise_from_start(x, x_0, alpha_t, sigma_t)

        elif self.objective == 'pred_v':
            v = model_output

            x_0 = self.predict_start_from_v(x, v, alpha_t, sigma_t)
            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
            pred_noise = self.predict_noise_from_start(x, x_0, alpha_t, sigma_t)

        return pred_noise, x_0

    @torch.inference_mode()
    def ddim_sample(self, n, labels=None, steps=512, eta=1., eps=1e-4, return_all_steps=False):
        if self.num_timesteps is not None:
            steps = default(steps, self.num_timesteps)
            assert steps <= self.num_timesteps

        ts = torch.linspace(1 - eps, eps, steps=steps+1)

        x = torch.randn(n, self.dim, *self.size, device=self.device)
        xs = [x]

        x_0 = None
        for i in range(steps):
            t_curr = torch.full((n,), ts[i], dtype=torch.float32, device=self.device)
            t_next = torch.full((n,), ts[i+1], dtype=torch.float32, device=self.device)

            alpha_cur, sigma_cur = self._get_alpha_sigma(t_curr)
            alpha_next, sigma_next = self._get_alpha_sigma(t_next)

            alpha_cur, sigma_cur = self._expand(alpha_cur), self._expand(sigma_cur)
            alpha_next, sigma_next = self._expand(alpha_next), self._expand(sigma_next)

            self_cond = x_0 if self.self_condition else None
            model_output = self.model(x, t_curr, label=labels, x_self_cond=self_cond)

            pred_noise, x_0 = self.forward(x, model_output, alpha_cur, sigma_cur, rederive_pred_noise=True)

            ddim_sigma = eta * (sigma_next / sigma_cur) * torch.sqrt(1 - alpha_cur**2 / alpha_next**2)

            noise = torch.randn_like(x)

            x = alpha_next * x_0 + \
                torch.sqrt((sigma_next ** 2 - ddim_sigma ** 2).clamp(min=0)) * pred_noise + \
                ddim_sigma * noise

            xs.append(x)

        ret = x if not return_all_steps else torch.stack(xs, dim = 1)
        return ret

    @torch.inference_mode()
    def sample(self, n=16, label=None, steps=512, eta=1., return_all_steps=False):
        if label is not None:
            if isinstance(label, int):
                labels = torch.LongTensor([label]*n)
            else:
                labels = label.repeat_interleave(n, dim=0)
            labels = labels.to(self.device)
            n_samples = len(labels)
        else:
            labels = None
            n_samples = n

        samples = self.ddim_sample(n_samples, labels=labels, steps=steps,
            eta=eta, return_all_steps=return_all_steps)
        if label is not None and not isinstance(label, int):
            samples = samples.reshape(-1, n, *samples.shape[1:])
        return samples.cpu().numpy()

    def p_losses(self, x_0, t, noise=None, y=None, offset_noise_strength=None):
        noise = default(noise, lambda: torch.randn_like(x_0))

        alpha_t, sigma_t = self._get_alpha_sigma(t)
        snr_t = self.snr_fn(alpha_t, sigma_t)

        alpha_t, sigma_t = self._expand(alpha_t), self._expand(sigma_t)

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_0.shape[:2], device = self.device)
            while offset_noise.dim() < x_0.dim():
                offset_noise = offset_noise[..., None]
            noise += offset_noise_strength * offset_noise

        # noise sample
        x = self.get_x(x_0, noise, alpha_t, sigma_t)

        # if doing self-conditioning, 50% of the time, predict x0 from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                model_output = self.model(x, t, label=y)

                if self.objective == 'pred_noise':
                    pred_noise = model_output

                    x_self_cond = self.predict_start_from_noise(x, pred_noise, alpha_t, sigma_t)
                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)
                elif self.objective == 'pred_x0':
                    x_self_cond = model_output

                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)
                elif self.objective == 'pred_v':
                    v = model_output

                    x_self_cond = self.predict_start_from_v(x, v, alpha_t, sigma_t)
                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)

                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, label=y, x_self_cond=x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_0
        elif self.objective == 'pred_v':
            target = self.get_v(x_0, noise, alpha_t, sigma_t)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = torch.mean(loss, dim=tuple(range(1, loss.dim())))

        loss = loss * snr_t
        return loss.mean()

    def loss(self, x, y=None, *args, **kwargs):
        b, _, *size = x.shape
        for s, ss in zip(size, self.size):
            assert s == ss, f'size must be {self.size}'
        if self.num_timesteps is None:
            t = torch.rand((b,), device=x.device)
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=x.device) / self.num_timesteps

        return self.p_losses(x, t, y=y, *args, **kwargs)


class ConditionalGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, cfg_dropout=0., **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cfg_dropout = cfg_dropout

    def forward_cfg(self, x, model_output, model_output_y, alpha_t, sigma_t,
                cfg=1., rederive_pred_noise=False):
        if self.objective == 'pred_noise':
            pred_noise = model_output
            pred_noise_y = model_output_y

        elif self.objective == 'pred_x0':
            x_0 = model_output
            x_0_y = model_output_y
            pred_noise = self.predict_noise_from_start(x, x_0, alpha_t, sigma_t)
            pred_noise_y = self.predict_noise_from_start(x, x_0_y, alpha_t, sigma_t)

        elif self.objective == 'pred_v':
            v = model_output
            v_y = model_output_y
            pred_noise = self.predict_noise_from_v(x, v, alpha_t, sigma_t)
            pred_noise_y = self.predict_noise_from_v(x, v_y, alpha_t, sigma_t)

        pred_noise = pred_noise + cfg * (pred_noise_y - pred_noise)

        x_0 = self.predict_start_from_noise(x, pred_noise, alpha_t, sigma_t)
        if self.clip_fn is not None:
            x_0 = self.clip_fn(x_0)
            if rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, x_0, alpha_t, sigma_t)

        return pred_noise, x_0

    @torch.inference_mode()
    def ddim_sample(self, n, labels=None, steps=512, eta=1., cfg=0., eps=1e-4, return_all_steps=False):
        if self.num_timesteps is not None:
            steps = default(steps, self.num_timesteps)
            assert steps <= self.num_timesteps

        ts = torch.linspace(1 - eps, eps, steps=steps+1)

        x = torch.randn(n, self.dim, *self.size, device=self.device)
        xs = [x]

        x_0 = None
        for i in range(steps):
            t_curr = torch.full((n,), ts[i], dtype=torch.float32, device=self.device)
            t_next = torch.full((n,), ts[i+1], dtype=torch.float32, device=self.device)

            alpha_cur, sigma_cur = self._get_alpha_sigma(t_curr)
            alpha_next, sigma_next = self._get_alpha_sigma(t_next)

            alpha_cur, sigma_cur = self._expand(alpha_cur), self._expand(sigma_cur)
            alpha_next, sigma_next = self._expand(alpha_next), self._expand(sigma_next)

            self_cond = x_0 if self.self_condition else None
            if cfg is not None and cfg > 0:
                model_output, model_output_y = self.model(
                    torch.cat([x, x]), torch.cat([t_curr, t_curr]),
                    label=torch.cat([torch.full_like(t_curr, -1, dtype=torch.long), labels]),
                    x_self_cond=torch.cat([self_cond, self_cond]) if self_cond is not None else None
                ).chunk(2)

                pred_noise, x_0 = self.forward_cfg(x, model_output, model_output_y, alpha_cur, sigma_cur,
                                                   cfg=cfg, rederive_pred_noise=True)
            else:
                model_output = self.model(x, t_curr, label=labels, x_self_cond=self_cond)

                pred_noise, x_0 = self.forward(x, model_output, alpha_cur, sigma_cur, rederive_pred_noise=True)

            ddim_sigma = eta * (sigma_next / sigma_cur) * torch.sqrt(1 - alpha_cur**2 / alpha_next**2)

            noise = torch.randn_like(x)

            x = alpha_next * x_0 + \
                torch.sqrt((sigma_next ** 2 - ddim_sigma ** 2).clamp(min=0)) * pred_noise + \
                ddim_sigma * noise

            xs.append(x)

        ret = x if not return_all_steps else torch.stack(xs, dim = 1)
        return ret

    @torch.inference_mode()
    def sample(self, n=16, label=None, steps=512, eta=1., cfg=0., return_all_steps=False):
        if label is not None:
            if isinstance(label, int):
                labels = torch.LongTensor([label]*n)
            else:
                labels = label.repeat_interleave(n, dim=0)
            labels = labels.to(self.device)
            n_samples = len(labels)
        else:
            labels = None
            n_samples = n

        samples = self.ddim_sample(n_samples, labels=labels, steps=steps,
            eta=eta, cfg=cfg, return_all_steps=return_all_steps)
        if label is not None and not isinstance(label, int):
            samples = samples.reshape(-1, n, *samples.shape[1:])
        return samples.cpu().numpy()

    def loss(self, x, y=None, *args, **kwargs):
        b, _, *size = x.shape
        for s, ss in zip(size, self.size):
            assert s == ss, f'size must be {self.size}'
        if self.num_timesteps is None:
            t = torch.rand((b,), device=x.device)
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=x.device) / self.num_timesteps

        if self.cfg_dropout > 0:
            drop_ids = torch.rand(y.shape[0], device=y.device) < self.cfg_dropout
            y = torch.where(drop_ids, -1, y)
        return self.p_losses(x, t, y=y, *args, **kwargs)
