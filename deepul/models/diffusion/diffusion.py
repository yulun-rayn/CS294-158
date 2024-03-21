from functools import partial
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul.models.diffusion.utils import (
    default, mult, cosine_alpha_schedule, sigmoid_alpha_schedule
)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        size=(),
        timesteps=None,            # int: discrete, None: continuous
        objective='pred_v',
        alpha_schedule='sigmoid',
        schedule_fn_kwargs=dict(),
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

        if alpha_schedule == 'cosine':
            self.alpha_schedule_fn = cosine_alpha_schedule(**schedule_fn_kwargs)
        elif alpha_schedule == 'sigmoid':
            self.alpha_schedule_fn = sigmoid_alpha_schedule(**schedule_fn_kwargs)
        else:
            raise ValueError(f'unknown alpha schedule {alpha_schedule}')

        self.num_timesteps = int(timesteps) if timesteps is not None else None

        # derive loss weight
        # snr - signal noise ratio

        if objective == 'pred_noise':
            self.snr_fn = lambda alpha: 1.
        elif objective == 'pred_x0':
            self.snr_fn = lambda alpha: alpha / (1 - alpha)
        elif objective == 'pred_v':
            self.snr_fn = lambda alpha: alpha

        if clip_range is not None:
            clip_min, clip_max = clip_range
            self.clip_fn = partial(torch.clamp, min=clip_min, max=clip_max)
        else:
            self.clip_fn = None

    @property
    def device(self):
        return self.model.device

    def get_x(self, x_0, noise, alpha_t):
        return mult(torch.sqrt(alpha_t), x_0) + mult(torch.sqrt(1-alpha_t), noise)

    def get_v(self, x_0, noise, alpha_t):
        return mult(torch.sqrt(alpha_t), noise) - mult(torch.sqrt(1-alpha_t), x_0)

    def predict_start_from_v(self, x_t, v, alpha_t):
        return mult(torch.sqrt(alpha_t), x_t) - mult(torch.sqrt(1-alpha_t), v)

    def predict_noise_from_v(self, x_t, v, alpha_t):
        return mult(torch.sqrt(1-alpha_t), x_t) + mult(torch.sqrt(alpha_t), v)

    def predict_start_from_noise(self, x_t, noise, alpha_t):
        return mult(torch.sqrt(1/alpha_t), x_t) - mult(torch.sqrt((1-alpha_t)/alpha_t), noise)

    def predict_noise_from_start(self, x_t, x_0, alpha_t):
        return mult(torch.sqrt(1/(1-alpha_t)), x_t) - mult(torch.sqrt(alpha_t/(1-alpha_t)), x_0)

    def forward(self, x, t, y=None, x_self_cond=None, rederive_pred_noise=False):
        model_output = self.model(x, t, label=y, x_self_cond=x_self_cond)

        alpha_t = self.alpha_schedule_fn(t)

        if self.objective == 'pred_noise':
            pred_noise = model_output

            x_0 = self.predict_start_from_noise(x, pred_noise, alpha_t)
            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
                if rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(x, x_0, alpha_t)

        elif self.objective == 'pred_x0':
            x_0 = model_output

            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
            pred_noise = self.predict_noise_from_start(x, x_0, alpha_t)

        elif self.objective == 'pred_v':
            v = model_output

            x_0 = self.predict_start_from_v(x, v, alpha_t)
            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
            pred_noise = self.predict_noise_from_start(x, x_0, alpha_t)

        return pred_noise, x_0

    @torch.inference_mode()
    def ddim_sample(self, shape, labels=None, steps=500, eta=1, eps=1e-8, return_all_steps=False):
        if self.num_timesteps is not None:
            steps = default(steps, self.num_timesteps)
            assert steps <= self.num_timesteps
        b, *_ = shape

        times = torch.linspace(0, 1, steps=steps+1)
        alphas = self.alpha_schedule_fn(times)
        times = list(reversed(times.tolist()))
        alphas = list(reversed(alphas.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        alpha_pairs = list(zip(alphas[:-1], alphas[1:]))

        x = torch.randn(shape, device=self.device)
        xs = [x]

        x_0 = None

        for (time, time_next), (alpha, alpha_next) in zip(time_pairs, alpha_pairs):
            time_cond = torch.full((b,), time, device=self.device)
            self_cond = x_0 if self.self_condition else None
            pred_noise, x_0 = self.forward(x, time_cond, y=labels, x_self_cond=self_cond, rederive_pred_noise=True)

            if time_next < 0:
                x = x_0
                xs.append(x)
                continue

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))**0.5

            noise = torch.randn_like(x)

            x = alpha_next**0.5 * x_0 + \
                (1 - alpha_next - sigma**2 + eps)**0.5 * pred_noise + \
                sigma * noise

            xs.append(x)

        ret = x if not return_all_steps else torch.stack(xs, dim = 1)
        return ret

    @torch.inference_mode()
    def sample(self, n=16, label=None, steps=500, eta=1, return_all_steps=False):
        labels = torch.LongTensor([label]*n).to(self.device) if label is not None else None

        samples = self.ddim_sample((n, self.dim, *self.size), labels=labels,
            steps=steps, eta=eta, return_all_steps=return_all_steps)
        return samples.cpu().numpy()

    def p_losses(self, x_0, t, noise=None, y=None, offset_noise_strength=None):
        noise = default(noise, lambda: torch.randn_like(x_0))

        alpha_t = self.alpha_schedule_fn(t)
        snr_t = self.snr_fn(alpha_t)

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_0.shape[:2], device = self.device)
            while offset_noise.dim() < x_0.dim():
                offset_noise = offset_noise[..., None]
            noise += offset_noise_strength * offset_noise

        # noise sample
        x = self.get_x(x_0, noise, alpha_t)

        # if doing self-conditioning, 50% of the time, predict x0 from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                model_output = self.model(x, t, label=y)

                if self.objective == 'pred_noise':
                    pred_noise = model_output

                    x_self_cond = self.predict_start_from_noise(x, pred_noise, alpha_t)
                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)
                elif self.objective == 'pred_x0':
                    x_self_cond = model_output

                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)
                elif self.objective == 'pred_v':
                    v = model_output

                    x_self_cond = self.predict_start_from_v(x, v, alpha_t)
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
            target = self.get_v(x_0, noise, alpha_t)
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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x, t, y=None, x_self_cond=None,
                cfg=None, rederive_pred_noise=False):
        if y is not None and cfg is not None:
            model_output, model_output_y = self.model(
                torch.cat([x, x]), torch.cat([t, t]),
                label=torch.cat([torch.full_like(t, -1, dtype=torch.long), y]),
                x_self_cond=torch.cat([x_self_cond, x_self_cond]) if x_self_cond is not None else None
            ).chunk(2)
        else:
            model_output = self.model(x, t, label=y, x_self_cond=x_self_cond)

        alpha_t = self.alpha_schedule_fn(t)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            if y is not None and cfg is not None:
                pred_noise_y = model_output_y
                pred_noise = pred_noise + cfg * (pred_noise_y - pred_noise)

            x_0 = self.predict_start_from_noise(x, pred_noise, alpha_t)
            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
                if rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(x, x_0, alpha_t)

        elif self.objective == 'pred_x0':
            x_0 = model_output

            if y is not None and cfg is not None:
                x_0_y = model_output_y
                pred_noise = self.predict_noise_from_start(x, x_0, alpha_t)
                pred_noise_y = self.predict_noise_from_start(x, x_0_y, alpha_t)
                pred_noise = pred_noise + cfg * (pred_noise_y - pred_noise)

                x_0 = self.predict_start_from_noise(x, pred_noise, alpha_t)
                if self.clip_fn is not None:
                    x_0 = self.clip_fn(x_0)
                    if rederive_pred_noise:
                        pred_noise = self.predict_noise_from_start(x, x_0, alpha_t)
            else:
                if self.clip_fn is not None:
                    x_0 = self.clip_fn(x_0)
                pred_noise = self.predict_noise_from_start(x, x_0, alpha_t)

        elif self.objective == 'pred_v':
            v = model_output

            if y is not None and cfg is not None:
                v_y = model_output_y
                pred_noise = self.predict_noise_from_v(x, v, alpha_t)
                pred_noise_y = self.predict_noise_from_v(x, v_y, alpha_t)
                pred_noise = pred_noise + cfg * (pred_noise_y - pred_noise)

                x_0 = self.predict_start_from_noise(x, pred_noise, alpha_t)
                if self.clip_fn is not None:
                    x_0 = self.clip_fn(x_0)
                    if rederive_pred_noise:
                        pred_noise = self.predict_noise_from_start(x, x_0, alpha_t)
            else:
                x_0 = self.predict_start_from_v(x, v, alpha_t)
                if self.clip_fn is not None:
                    x_0 = self.clip_fn(x_0)
                pred_noise = self.predict_noise_from_start(x, x_0, alpha_t)

        return pred_noise, x_0

    @torch.inference_mode()
    def ddim_sample(self, shape, labels=None, steps=500, eta=1,
                    cfg=None, eps=1e-8, return_all_steps=False):
        if self.num_timesteps is not None:
            steps = default(steps, self.num_timesteps)
            assert steps <= self.num_timesteps
        b, *_ = shape

        times = torch.linspace(0, 1, steps=steps+1)
        alphas = self.alpha_schedule_fn(times)
        times = list(reversed(times.tolist()))
        alphas = list(reversed(alphas.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        alpha_pairs = list(zip(alphas[:-1], alphas[1:]))

        x = torch.randn(shape, device=self.device)
        xs = [x]

        x_0 = None

        for (time, time_next), (alpha, alpha_next) in zip(time_pairs, alpha_pairs):
            time_cond = torch.full((b,), time, device=self.device)
            self_cond = x_0 if self.self_condition else None
            pred_noise, x_0 = self.forward(x, time_cond, y=labels, x_self_cond=self_cond, 
                                           cfg=cfg, rederive_pred_noise=True)

            if time_next < 0:
                x = x_0
                xs.append(x)
                continue

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))**0.5

            noise = torch.randn_like(x)

            x = alpha_next**0.5 * x_0 + \
                (1 - alpha_next - sigma**2 + eps)**0.5 * pred_noise + \
                sigma * noise

            xs.append(x)

        ret = x if not return_all_steps else torch.stack(xs, dim = 1)
        return ret

    @torch.inference_mode()
    def sample(self, n=16, label=None, steps=500, eta=1,
               cfg=None, return_all_steps=False):
        labels = torch.LongTensor([label]*n).to(self.device) if label is not None else None

        samples = self.ddim_sample((n, self.dim, *self.size), labels=labels,
            steps=steps, eta=eta, cfg=cfg, return_all_steps=return_all_steps)
        return samples.cpu().numpy()

    def p_losses(self, x_0, t, noise=None, y=None, offset_noise_strength=None):
        noise = default(noise, lambda: torch.randn_like(x_0))

        alpha_t = self.alpha_schedule_fn(t)
        snr_t = self.snr_fn(alpha_t)

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_0.shape[:2], device = self.device)
            while offset_noise.dim() < x_0.dim():
                offset_noise = offset_noise[..., None]
            noise += offset_noise_strength * offset_noise

        # noise sample
        x = self.get_x(x_0, noise, alpha_t)

        # if doing self-conditioning, 50% of the time, predict x0 from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                model_output = self.model(x, t, label=y)

                if self.objective == 'pred_noise':
                    pred_noise = model_output

                    x_self_cond = self.predict_start_from_noise(x, pred_noise, alpha_t)
                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)
                elif self.objective == 'pred_x0':
                    x_self_cond = model_output

                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)
                elif self.objective == 'pred_v':
                    v = model_output

                    x_self_cond = self.predict_start_from_v(x, v, alpha_t)
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
            target = self.get_v(x_0, noise, alpha_t)
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
