"""Image generators before diffusion (notebook 06): autoencoder, VAE, GAN / WGAN-GP, pixel autoregression.

Images are the 8x8 digits of prelib.images: vectors of 64 numbers in [-1, 1] (17 grey levels).

    VAE(z_dim)                    encoder q(z|x), decoder p(x|z); beta=0 and deterministic=True gives a plain autoencoder
    train_vae(model, X, beta, anneal)
    GAN: Generator, Critic, train_gan(..., loss="gan" | "wgan-gp")
    ring(n)                       eight Gaussians on a circle, the standard 2-D test of mode coverage
    PixelAR, train_pixel          a causal transformer over the 64 pixels, each a 17-way choice
    class_shares, contrast        two simple read-outs of a set of samples
"""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import set_seed, transformer as T

LEVELS = 17                                   # grey levels of the digits data: 0 .. 16


def mlp(sizes, act=nn.GELU, last=None):
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(act())
    if last is not None:
        layers.append(last)
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------- autoencoder and VAE
class VAE(nn.Module):
    """Variational autoencoder (Kingma & Welling, 2013) with MLP encoder and decoder."""

    def __init__(self, z_dim=8, hidden=128, x_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.enc = mlp([x_dim, hidden, hidden, 2 * z_dim])
        self.dec = mlp([z_dim, hidden, hidden, x_dim], last=nn.Tanh())

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, -1)
        return mu, logvar

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else mu + torch.randn_like(mu) * (0.5 * logvar).exp()      # the reparameterisation trick
        return self.dec(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        return self.dec(torch.randn(n, self.z_dim, generator=g))


def kl_per_dim(mu, logvar):
    """KL( q(z|x) || N(0, I) ) for each latent dimension, averaged over the batch: array (z_dim,)."""
    return (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)).mean(0)


def train_vae(model, X, steps=2000, beta=1.0, anneal=0, batch=128, lr=2e-3, seed=0, deterministic=False, log_every=0, callback=None, at=()):
    """Minimise  reconstruction + beta * KL.  reconstruction = squared error summed over the 64 pixels.
    anneal > 0 raises the KL weight linearly from 0 to beta over that many steps (Bowman et al., 2015).
    deterministic=True with beta=0 trains a plain autoencoder (no noise, no prior).
    callback(step, model) is called before training and after each step listed in `at`."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    log = []
    if callback is not None:
        callback(0, model)
    for step in range(steps):
        x = X[torch.randint(0, len(X), (batch,))]
        xh, mu, logvar = model(x, deterministic)
        rec = ((x - xh) ** 2).sum(-1).mean()
        kl = kl_per_dim(mu, logvar).sum()
        w = beta * min(1.0, (step + 1) / anneal) if anneal else beta
        loss = rec + w * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and (step + 1) % log_every == 0:
            log.append(dict(step=step + 1, reconstruction=rec.item(), kl=kl.item(), weight=w))
        if callback is not None and step + 1 in at:
            callback(step + 1, model)
    return log


@torch.no_grad()
def vae_report(model, X, threshold=0.01):
    """Reconstruction error, total KL and number of latent dimensions in use (KL above `threshold` nats) on X."""
    xh, mu, logvar = model(X, deterministic=True)
    kl = kl_per_dim(mu, logvar)
    return {"reconstruction error": float(((X - xh) ** 2).sum(-1).mean()), "KL (nats)": float(kl.sum()),
            "latent dimensions in use": int((kl > threshold).sum())}


# ----------------------------------------------------------------------------- GAN
class Generator(nn.Module):
    def __init__(self, z_dim=16, hidden=128, x_dim=64, squash=True):
        super().__init__()
        self.z_dim = z_dim
        self.net = mlp([z_dim, hidden, hidden, x_dim], last=nn.Tanh() if squash else None)

    def forward(self, z):
        return self.net(z)

    @torch.no_grad()
    def sample(self, n, seed=0):
        return self.net(torch.randn(n, self.z_dim, generator=torch.Generator().manual_seed(seed)))


class Critic(nn.Module):
    """The discriminator of a GAN (outputs a logit) or the critic of a WGAN (outputs a score)."""

    def __init__(self, hidden=128, x_dim=64, spectral=False):
        super().__init__()
        sn = nn.utils.spectral_norm if spectral else (lambda m: m)
        self.net = nn.Sequential(sn(nn.Linear(x_dim, hidden)), nn.LeakyReLU(0.2), sn(nn.Linear(hidden, hidden)), nn.LeakyReLU(0.2), sn(nn.Linear(hidden, 1)))

    def forward(self, x):
        return self.net(x)[:, 0]


def train_gan(G, D, X, steps=3000, loss="gan", batch=64, seed=0, n_critic=None, gp=10.0, lr=None, callback=None, every=100):
    """Adversarial training.
    loss="gan":      discriminator: real vs fake cross-entropy; generator: the non-saturating loss (Goodfellow et al., 2014)
    loss="wgan-gp":  critic estimates a Wasserstein distance, kept 1-Lipschitz by a gradient penalty (Gulrajani et al., 2017);
                     the critic takes n_critic steps per generator step.
    callback(step, G) is called every `every` generator steps."""
    set_seed(seed)
    wg = loss == "wgan-gp"
    n_critic = n_critic or (5 if wg else 1)
    lr = lr or (1e-4 if wg else 2e-4)
    betas = (0.0, 0.9) if wg else (0.5, 0.999)
    og, od = torch.optim.Adam(G.parameters(), lr, betas=betas), torch.optim.Adam(D.parameters(), lr, betas=betas)
    log = []
    for step in range(steps):
        for _ in range(n_critic):
            x = X[torch.randint(0, len(X), (batch,))]
            fake = G(torch.randn(batch, G.z_dim)).detach()
            if wg:
                eps = torch.rand(batch, 1)
                mix = (eps * x + (1 - eps) * fake).requires_grad_(True)
                grad = torch.autograd.grad(D(mix).sum(), mix, create_graph=True)[0]
                d_loss = D(fake).mean() - D(x).mean() + gp * ((grad.norm(dim=1) - 1) ** 2).mean()
            else:
                d_loss = F.binary_cross_entropy_with_logits(D(x), torch.ones(batch)) + F.binary_cross_entropy_with_logits(D(fake), torch.zeros(batch))
            od.zero_grad(); d_loss.backward(); od.step()
        out = D(G(torch.randn(batch, G.z_dim)))
        g_loss = -out.mean() if wg else F.binary_cross_entropy_with_logits(out, torch.ones(batch))
        og.zero_grad(); g_loss.backward(); og.step()
        log.append((d_loss.item(), g_loss.item()))
        if callback is not None and (step + 1) % every == 0:
            callback(step + 1, G)
    return log


RING_CENTRES = torch.tensor([[2 * math.cos(2 * math.pi * k / 8), 2 * math.sin(2 * math.pi * k / 8)] for k in range(8)])


def ring(n, seed=0, std=0.05):
    """n points from a mixture of eight narrow Gaussians on a circle of radius 2."""
    g = torch.Generator().manual_seed(seed)
    return RING_CENTRES[torch.randint(0, 8, (n,), generator=g)] + std * torch.randn(n, 2, generator=g)


def ring_modes(points, radius=0.3):
    """(number of the eight modes that receive at least 2% of the points, share of points within `radius` of any mode)."""
    d = torch.cdist(points, RING_CENTRES)
    near = d.min(1).values < radius
    share = torch.bincount(d.argmin(1)[near], minlength=8).float() / len(points)
    return int((share > 0.02).sum()), float(near.float().mean())


# ----------------------------------------------------------------------------- pixel autoregression
def to_levels(X):
    """Images in [-1, 1] -> integer grey levels 0..16."""
    return torch.round((X + 1) * 8).long().clamp(0, LEVELS - 1)


def from_levels(L):
    return L.float() / 8 - 1


class PixelAR(nn.Module):
    """p(image) = product over the 64 pixels, in reading order, of p(pixel | earlier pixels) (van den Oord et al., 2016).
    Each pixel is a 17-way choice; the network is the causal transformer of notebook 03 with one token per pixel."""
    START = LEVELS

    def __init__(self, d=64, n_layer=2, n_head=4):
        super().__init__()
        self.net = T.Transformer(d=d, n_layer=n_layer, n_head=n_head, block=65, pos="learned", norm="pre", norm_type="rms", ffn="swiglu", vocab=LEVELS + 1)
        self.cfg = self.net.cfg

    def tokens(self, X):
        L = to_levels(X)
        return torch.cat([torch.full((len(L), 1), self.START), L], 1)

    def forward(self, tokens):
        return self.net(tokens)

    @torch.no_grad()
    def bits_per_pixel(self, X):
        self.eval()
        t = self.tokens(X)
        lp = F.log_softmax(self.net(t)[..., :LEVELS], -1)
        return float(-lp[:, 1:].gather(-1, t[:, 1:, None]).mean() / math.log(2))

    @torch.no_grad()
    def sample(self, n, seed=0, temperature=1.0, record=False):
        """Generate n images one pixel at a time: 64 network evaluations per image."""
        self.eval(); g = torch.Generator().manual_seed(seed)
        t = torch.full((n, 1), self.START); frames = []
        for i in range(64):
            logits = self.net.out(self.net.hidden(t)[:, -1])[:, :LEVELS] / temperature
            t = torch.cat([t, torch.multinomial(torch.softmax(logits, -1), 1, generator=g)], 1)
            if record:
                frames.append(t[:, 1:].clone())
        x = from_levels(t[:, 1:])
        return (x, frames) if record else x


    @torch.no_grad()
    def complete(self, X, keep=32, n=1, seed=0, temperature=1.0):
        """Keep the first `keep` pixels of each image in X and sample the rest, n times per image."""
        self.eval(); g = torch.Generator().manual_seed(seed)
        t = self.tokens(X)[:, :keep + 1].repeat_interleave(n, 0)
        for i in range(keep, 64):
            logits = self.net.out(self.net.hidden(t)[:, -1])[:, :LEVELS] / temperature
            t = torch.cat([t, torch.multinomial(torch.softmax(logits, -1), 1, generator=g)], 1)
        return from_levels(t[:, 1:])

    @torch.no_grad()
    def next_pixel(self, X):
        """p(pixel i | pixels before i) for every pixel of every image in X: array (n, 64, 17)."""
        self.eval()
        return torch.softmax(self.net(self.tokens(X))[:, 1:, :LEVELS], -1)


def train_pixel(model, X, X_test=None, steps=700, batch=64, lr=2e-3, weight_decay=0.0, seed=0, log_every=100):
    """Maximum likelihood: cross-entropy of each pixel's grey level given the pixels before it.
    Returns a log of (step, train bits per pixel, test bits per pixel)."""
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    tok, log = model.tokens(X), []
    for step in range(steps):
        model.train()
        t = tok[torch.randint(0, len(tok), (batch,))]
        loss = F.cross_entropy(model(t)[:, 1:, :LEVELS].reshape(-1, LEVELS), t[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and (step + 1) % log_every == 0:
            log.append(dict(step=step + 1, train=model.bits_per_pixel(X[:len(X_test)] if X_test is not None else X),
                            test=model.bits_per_pixel(X_test) if X_test is not None else float("nan")))
    return log


def independent_pixels(X):
    """The model with no dependence between pixels: a table (64, 17) of how often each pixel takes each grey level."""
    L = to_levels(X)
    P = torch.stack([torch.bincount(L[:, i], minlength=LEVELS).float() + 1 for i in range(64)])      # +1: no zero probabilities
    return P / P.sum(1, keepdim=True)


def independent_bits(P, X):
    return float(-torch.log2(P)[torch.arange(64), to_levels(X)].mean())


def independent_sample(P, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return from_levels(torch.stack([torch.multinomial(P[i], n, replacement=True, generator=g) for i in range(64)], 1))


# ----------------------------------------------------------------------------- read-outs of a set of samples
@torch.no_grad()
def class_shares(judge, X):
    """Share of the samples that the frozen judge assigns to each of the ten digits."""
    return (torch.bincount(judge(X.clamp(-1, 1)).argmax(1), minlength=10).float() / len(X)).numpy()


def contrast(X):
    """Mean absolute difference between neighbouring pixels: lower than the data's value means blurrier images."""
    im = X.view(-1, 8, 8)
    return float(((im[:, 1:] - im[:, :-1]).abs().mean() + (im[:, :, 1:] - im[:, :, :-1]).abs().mean()) / 2)
