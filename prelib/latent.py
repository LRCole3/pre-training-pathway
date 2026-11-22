"""Latent diffusion at toy scale (notebook 08): an autoencoder first, a diffusion model on its codes second.

    Autoencoder(z_dim, kind)       kind = "kl" (a VAE with a small KL weight) or "vq" (vector-quantised codes)
    train_autoencoder(ae, X, ...)  reconstruction + regulariser
    Latents(ae, X)                 codes of a dataset, rescaled to unit standard deviation (the 'scale factor')
    caption_table / train_text_diffusion / sample_text    a denoiser conditioned on text embeddings
"""
import time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import set_seed, generative as G, diffusion as D, vision as V


class Autoencoder(nn.Module):
    """MLP encoder and decoder around a code of z_dim numbers.
    kind="kl": the encoder outputs a Gaussian per image and pays `weight` times its KL to N(0, I) (Rombach et al.
               use a very small weight). weight=0 gives a plain autoencoder.
    kind="vq": the code is split into z_dim / code_dim tokens and each token is replaced by the nearest entry of a
               learned codebook (van den Oord et al., 2017). Gradients pass straight through the replacement."""

    def __init__(self, z_dim=16, kind="kl", hidden=128, n_codes=64, code_dim=4):
        super().__init__()
        self.z_dim, self.kind, self.code_dim = z_dim, kind, code_dim
        self.enc = G.mlp([64, hidden, hidden, 2 * z_dim if kind == "kl" else z_dim])
        self.dec = G.mlp([z_dim, hidden, hidden, 64], last=nn.Tanh())
        if kind == "vq":
            self.codebook = nn.Parameter(torch.randn(n_codes, code_dim) * 0.5)

    def encode(self, x):
        """The code that the second-stage model is trained on (for "vq": before quantisation)."""
        h = self.enc(x)
        return h[:, :self.z_dim]

    def quantise(self, z):
        tok = z.reshape(len(z), -1, self.code_dim)
        idx = torch.cdist(tok, self.codebook[None].expand(len(z), -1, -1)).argmin(-1)
        return self.codebook[idx].reshape(len(z), -1), idx

    def decode(self, z):
        """Code -> image. For "vq" the code is quantised first, so quantisation is part of the decoder."""
        if self.kind == "vq":
            z = self.quantise(z)[0]
        return self.dec(z)

    def loss(self, x, weight):
        h = self.enc(x)
        if self.kind == "kl":
            mu, logvar = h.chunk(2, -1)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if weight > 0 else mu
            rec = ((x - self.dec(z)) ** 2).sum(-1).mean()
            return rec + weight * G.kl_per_dim(mu, logvar).sum(), rec
        q, _ = self.quantise(h)
        rec = ((x - self.dec(h + (q - h).detach())) ** 2).sum(-1).mean()            # straight-through estimator
        return rec + ((q - h.detach()) ** 2).sum(-1).mean() + weight * ((q.detach() - h) ** 2).sum(-1).mean(), rec

    @torch.no_grad()
    def reconstruct(self, x):
        self.eval()
        return self.decode(self.encode(x))


def train_autoencoder(ae, X, weight=1e-3, steps=2000, batch=128, lr=2e-3, seed=0):
    set_seed(seed)
    opt = torch.optim.Adam(ae.parameters(), lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    t0 = time.time()
    for step in range(steps):
        ae.train()
        loss, rec = ae.loss(X[torch.randint(0, len(X), (batch,))], weight)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    ae.eval()
    return dict(seconds=time.time() - t0)


class Latents:
    """Codes of the training images, divided by one number so that their overall standard deviation is 1
    (latent diffusion models ship this number as their 'scale factor'). scale=False leaves the codes as they are."""

    def __init__(self, ae, X, scale=True):
        self.ae = ae
        with torch.no_grad():
            z = ae.encode(X)
        self.factor = float(z.std()) if scale else 1.0
        self.z = z / self.factor

    @torch.no_grad()
    def decode(self, z):
        return self.ae.decode(z * self.factor)


# ----------------------------------------------------------------------------- text conditioning
def caption_table(numerals=True, synonyms=(0, 1, 2)):
    """Every caption the generator of prelib.vision can produce for each (digit, style): tokens (10, 2, M, CAP_LEN).
    numerals=False keeps only captions that spell the digit as a word; synonyms selects which of the three style
    words may appear (0: thin / bold, 1: fine / thick, 2: light / heavy)."""
    table = []
    for d in range(10):
        row = []
        for s in (0, 1):
            forms = (V.DIGIT_WORDS[d], str(d)) if numerals else (V.DIGIT_WORDS[d],)
            caps = sorted({t.format(d=dw, s=V.STYLE_WORDS[s][k]) for t in V.TEMPLATES for dw in forms for k in synonyms})
            row.append(torch.stack([V.encode_caption(c) for c in caps]))
        table.append(torch.stack(row))
    return torch.stack(table)


def train_text_diffusion(model, diff, Z, digits, styles, embed, steps=3000, batch=128, lr=2e-3, p_uncond=0.1, seed=0, text_encoder=None, table=None):
    """Denoiser conditioned on the embedding of a caption of each training image.
    embed(tokens) -> vectors. If text_encoder is given, its parameters are trained together with the denoiser
    (a text encoder learned from scratch); otherwise the embeddings come from a frozen, pretrained encoder.
    table: the captions used for training (default: all of them, see caption_table)."""
    set_seed(seed)
    table = caption_table() if table is None else table
    params = list(model.parameters()) + (list(text_encoder.parameters()) if text_encoder is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    ema = D.EMA(model, 0.995)
    if text_encoder is None:
        with torch.no_grad():
            frozen = embed(table.reshape(-1, table.shape[-1])).reshape(*table.shape[:3], -1)
    t0 = time.time()
    for step in range(steps):
        model.train()
        idx = torch.randint(0, len(Z), (batch,))
        j = torch.randint(0, table.shape[2], (batch,))
        c = frozen[digits[idx], styles[idx], j].clone() if text_encoder is None else embed(table[digits[idx], styles[idx], j])
        c = c * (torch.rand(batch, 1) >= p_uncond)                              # caption dropped -> the zero vector
        loss = diff.loss(model, Z[idx], c)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); ema.update(model)
    model.eval()
    return dict(model=ema.model, seconds=time.time() - t0)
