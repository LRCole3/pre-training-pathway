"""Diffusion models on the 8x8 digits (notebook 07 onwards): every training choice is an argument.

    Diffusion(schedule, pred, weighting)   the forward process, the training target and the loss weight
        schedule   "linear" (Ho et al., 2020) | "sd-linear" (the scaled-linear schedule of Stable Diffusion) |
                   "cosine" (Nichol & Dhariwal, 2021) | any of these + "-zt" for zero terminal SNR (Lin et al., 2023) |
                   "cosine-short" (stops early, to show a train / sample mismatch)
        pred       what the network outputs: "eps" (the noise) | "x0" (the clean image) | "v" (Salimans & Ho, 2022)
        weighting  "uniform" (plain squared error on the output) | "min-snr" (Hang et al., 2023)
    Denoiser(width, depth, time)           residual MLP; time = "sinusoidal" | "scalar" | "none"
    train(model, diff, X, y, ...)          denoising loss, label dropout, EMA of the weights
    diff.sample(model, labels, steps, w)   DDIM / DDPM sampler with classifier-free guidance
    Hybrid(high, low, switch)              two trained denoisers in one sampling run, switching at a timestep
    EDM(precondition)                      the sigma-based formulation of Karras et al. (2022) with its Heun sampler

Images are vectors of 64 numbers in [-1, 1]; labels 0..9, NULL = 10 is "no label".
"""
import copy, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import set_seed

T = 1000
NULL = 10


# ----------------------------------------------------------------------------- noise schedules
def alpha_bar(schedule="cosine", T=T):
    """alpha_bar[t] for t = 0..T-1: the share of the signal's variance that is left after t+1 noising steps."""
    name = schedule.replace("-zt", "")
    if name == "linear":
        ab = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T, dtype=torch.float64), 0)
    elif name == "sd-linear":
        ab = torch.cumprod(1 - torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, T, dtype=torch.float64) ** 2, 0)
    elif name in ("cosine", "cosine-short"):
        t = torch.arange(T + 1, dtype=torch.float64) / T
        if name == "cosine-short":
            t = t * 0.6                                        # stop the same curve at 60% of the way: noise never takes over
        f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        ab = (f / f[0])[1:].clamp(1e-5, 0.9999)
    else:
        raise ValueError(schedule)
    if schedule.endswith("-zt"):                               # shift and rescale sqrt(alpha_bar) so that the last value is exactly 0
        s = ab.sqrt()
        s = (s - s[-1]) * s[0] / (s[0] - s[-1])
        ab = s ** 2
    return ab.float()


class Diffusion:
    def __init__(self, schedule="cosine", pred="eps", weighting="uniform", gamma=5.0):
        self.schedule, self.pred, self.weighting, self.gamma = schedule, pred, weighting, gamma
        self.ab = alpha_bar(schedule)
        self.snr = self.ab / (1 - self.ab)                     # signal-to-noise ratio at each timestep

    # -- forward process and targets
    def q_sample(self, x0, t, eps):
        ab = self.ab[t][:, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def target(self, x0, t, eps):
        if self.pred == "eps":
            return eps
        if self.pred == "x0":
            return x0
        ab = self.ab[t][:, None]
        return ab.sqrt() * eps - (1 - ab).sqrt() * x0          # v

    def to_x0_eps(self, out, xt, t):
        """Turn the network output into an estimate of the clean image and of the noise."""
        ab = self.ab[t][:, None]
        if self.pred == "eps":
            return (xt - (1 - ab).sqrt() * out) / ab.sqrt().clamp_min(1e-6), out
        if self.pred == "x0":
            return out, (xt - ab.sqrt() * out) / (1 - ab).sqrt()
        return ab.sqrt() * xt - (1 - ab).sqrt() * out, (1 - ab).sqrt() * xt + ab.sqrt() * out

    def weight(self, t):
        """Weight on the squared error of the network's own output."""
        if self.weighting == "uniform":
            return torch.ones(len(t))
        snr = self.snr[t]
        w = snr.clamp(max=self.gamma)                          # min-SNR: weight min(SNR, gamma) on the error of the clean image ...
        if self.pred == "eps":
            return w / snr.clamp_min(1e-8)                     # ... converted to the space of the network's output
        if self.pred == "v":
            return w / (snr + 1)
        return w

    def loss(self, model, x0, c, t=None):
        t = torch.randint(0, T, (len(x0),)) if t is None else t
        eps = torch.randn_like(x0)
        out = model(self.q_sample(x0, t, eps), t, c)
        return (self.weight(t) * ((out - self.target(x0, t, eps)) ** 2).mean(-1)).mean()

    # -- sampling
    def predict(self, model, x, t, c, w=0.0):
        """Network output with classifier-free guidance: (1 + w) * conditional - w * unconditional."""
        if w == 0:
            return model(x, t, c)
        oc, ou = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([c, torch.full_like(c, NULL)])).chunk(2)
        return (1 + w) * oc - w * ou

    @torch.no_grad()
    def sample(self, model, labels, steps=50, w=0.0, eta=0.0, seed=0, x_T=None, clip=True, record=False):
        """DDIM (Song et al., 2020). eta = 0 is deterministic; eta = 1 with steps = 1000 is the DDPM sampler of Ho et al.
        Network evaluations per image: steps (twice that with guidance)."""
        g = torch.Generator().manual_seed(seed)
        n = len(labels)
        x = torch.randn(n, 64, generator=g) if x_T is None else x_T.clone()
        ts = torch.linspace(T - 1, 0, steps).round().long()
        frames = []
        for i, t in enumerate(ts):
            tt = torch.full((n,), int(t))
            x0, eps = self.to_x0_eps(self.predict(model, x, tt, labels, w), x, tt)
            if clip:
                x0 = x0.clamp(-1, 1)
                eps = (x - self.ab[t].sqrt() * x0) / (1 - self.ab[t]).sqrt()
            if record:
                frames.append((int(t), x.clone(), x0.clone()))
            if i == steps - 1:
                x = x0
                break
            ab_t, ab_s = self.ab[t], self.ab[ts[i + 1]]
            sigma = eta * ((1 - ab_s) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_s).clamp_min(0).sqrt()
            x = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp_min(0).sqrt() * eps + sigma * torch.randn(n, 64, generator=g)
        return (x, frames) if record else x

    @torch.no_grad()
    def error_by_time(self, model, X, y, bins=10, repeats=4, seed=0):
        """Mean squared error of the model's estimate of the clean image and of the noise, per band of timesteps.
        Returns (bin centres, error of x0 estimate, error of eps estimate)."""
        g = torch.Generator().manual_seed(seed)
        edges = torch.linspace(0, T, bins + 1).long()
        ex, ee = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            a, b = [], []
            for _ in range(repeats):
                t = torch.randint(int(lo), int(hi), (len(X),), generator=g)
                eps = torch.randn(X.shape, generator=g)
                xt = self.q_sample(X, t, eps)
                x0h, epsh = self.to_x0_eps(model(xt, t, y), xt, t)
                a.append(((x0h - X) ** 2).mean().item()); b.append(((epsh - eps) ** 2).mean().item())
            ex.append(np.mean(a)); ee.append(np.mean(b))
        return ((edges[:-1] + edges[1:]) / 2).numpy(), np.array(ex), np.array(ee)


# ----------------------------------------------------------------------------- the network
def timestep_embedding(t, dim=128, max_period=10000):
    """Sinusoidal features of the timestep: the position encoding of notebook 03, applied to time."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class Denoiser(nn.Module):
    """Residual MLP: (noisy image, timestep, label) -> prediction with 64 numbers."""

    def __init__(self, width=128, depth=3, time="sinusoidal", n_classes=11):
        super().__init__()
        self.time = time
        self.inp = nn.Linear(64, width)
        self.temb = nn.Sequential(nn.Linear(128 if time == "sinusoidal" else 1, width), nn.SiLU(), nn.Linear(width, width))
        self.cemb = nn.Embedding(n_classes, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)) for _ in range(depth))
        self.mods = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)

    def forward(self, x, t, c):
        e = self.cemb(c)
        if self.time == "sinusoidal":
            e = e + self.temb(timestep_embedding(t))
        elif self.time == "scalar":
            e = e + self.temb(t.float()[:, None] / T)          # the timestep as one number between 0 and 1
        h = self.inp(x)
        for blk, mod in zip(self.blocks, self.mods):
            h = h + blk(h + mod(F.silu(e)))
        return self.out(h)


class Hybrid(nn.Module):
    """Two trained denoisers used in one sampling run: `high` for timesteps >= switch, `low` below.
    high and low are (Diffusion, model) pairs. The output is an estimate of the clean image, so sample with
    Diffusion(schedule, "x0").sample(Hybrid(...), labels)."""

    def __init__(self, high, low, switch):
        super().__init__()
        self.high, self.low, self.switch = high, low, switch

    def forward(self, x, t, c):
        diff, model = self.high if int(t[0]) >= self.switch else self.low
        return diff.to_x0_eps(model(x, t, c), x, t)[0]


class EMA:
    """Exponential moving average of the weights, with the usual warm-up of the decay."""

    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.lerp_(p, 1 - d)


def train(model, diff, X, y, steps=1500, batch=128, lr=2e-3, p_uncond=0.1, ema_decay=0.995, seed=0, callback=None, every=250, lr_schedule="cosine"):
    """Train `model` on the loss of `diff` (a Diffusion or an EDM). Labels are replaced by NULL with probability
    p_uncond, which is what makes classifier-free guidance possible. callback(step, raw model, EMA model) every `every` steps.
    The learning rate warms up for 100 steps and then follows a cosine to zero (lr_schedule="cosine") or stays constant.
    Returns dict(model=EMA weights, raw=last weights, losses, seconds)."""
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * (0.5 * (1 + math.cos(math.pi * s / steps)) if lr_schedule == "cosine" else 1.0))
    ema = EMA(model, ema_decay)
    losses, t0 = [], time.time()
    for step in range(steps):
        model.train()
        idx = torch.randint(0, len(X), (batch,))
        c = y[idx].clone()
        c[torch.rand(batch) < p_uncond] = NULL
        loss = diff.loss(model, X[idx], c)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); ema.update(model)
        losses.append(loss.item())
        if callback is not None and (step + 1) % every == 0:
            model.eval(); callback(step + 1, model, ema.model)
    model.eval()
    return dict(model=ema.model, raw=model, losses=np.array(losses), seconds=time.time() - t0)


# ----------------------------------------------------------------------------- EDM (Karras et al., 2022)
class EDM:
    """Noise level sigma instead of a timestep: x = x0 + sigma * noise. The network F is wrapped as
        D(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F(c_in(sigma) * x, sigma)
    so that its input and its training target have unit variance at every noise level (precondition=True).
    precondition=False uses the network output directly as the estimate of the clean image."""

    def __init__(self, precondition=True, sigma_data=0.5, p_mean=-1.2, p_std=1.2, sigma_min=0.002, sigma_max=80.0):
        self.precondition, self.sd, self.p_mean, self.p_std = precondition, sigma_data, p_mean, p_std
        self.sigma_min, self.sigma_max = sigma_min, sigma_max

    def denoise(self, model, x, sigma, c):
        s = sigma[:, None]
        t_in = 250 * sigma.log()                               # the network sees log(sigma), scaled to the range of a timestep
        if not self.precondition:
            return model(x, t_in, c)
        c_skip = self.sd ** 2 / (s ** 2 + self.sd ** 2)
        c_out = s * self.sd / (s ** 2 + self.sd ** 2).sqrt()
        c_in = 1 / (s ** 2 + self.sd ** 2).sqrt()
        return c_skip * x + c_out * model(c_in * x, t_in, c)

    def loss(self, model, x0, c):
        sigma = (self.p_mean + self.p_std * torch.randn(len(x0))).exp()
        d = self.denoise(model, x0 + sigma[:, None] * torch.randn_like(x0), sigma, c)
        w = (sigma ** 2 + self.sd ** 2) / (sigma * self.sd) ** 2 if self.precondition else torch.ones(len(x0))
        return (w * ((d - x0) ** 2).mean(-1)).mean()

    def sigmas(self, steps, rho=7.0):
        i = torch.arange(steps) / max(steps - 1, 1)
        s = (self.sigma_max ** (1 / rho) + i * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        return torch.cat([s, torch.zeros(1)])

    @torch.no_grad()
    def sample(self, model, labels, steps=18, w=0.0, seed=0, heun=True, clip=True):
        """Deterministic sampler of the paper: Euler steps with a second-order (Heun) correction.
        Network evaluations per image: 2 * steps - 1 with the correction, steps without."""
        g = torch.Generator().manual_seed(seed)
        n = len(labels)
        sig = self.sigmas(steps)
        x = torch.randn(n, 64, generator=g) * sig[0]

        def D(x, s):
            ss = torch.full((n,), float(s))
            d = self.denoise(model, x, ss, labels)
            if w:
                d = (1 + w) * d - w * self.denoise(model, x, ss, torch.full_like(labels, NULL))
            return d.clamp(-1, 1) if clip else d

        for i in range(steps):
            d = (x - D(x, sig[i])) / sig[i]
            x_next = x + (sig[i + 1] - sig[i]) * d
            if heun and sig[i + 1] > 0:
                d2 = (x_next - D(x_next, sig[i + 1])) / sig[i + 1]
                x_next = x + (sig[i + 1] - sig[i]) * (d + d2) / 2
            x = x_next
        return x
