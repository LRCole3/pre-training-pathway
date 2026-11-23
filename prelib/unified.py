"""One transformer backbone, three ways to pretrain it, two kinds of data (notebook 10).

    Backbone(pos, d, depth, heads)        transformer blocks; knows nothing about what its tokens are
    Discrete(vocab, pos, causal)          token ids in, logits out. causal=True: autoregressive; causal=False: masked
    Continuous(in_dim, out_dim, pos)      vectors in, vectors or logits out, told a timestep: a denoiser
    PatchDenoiser(pos)                    Continuous on the 16 patches of an 8x8 image (a diffusion transformer)

    Autoregressive()                      predict each token from the tokens before it
    MaskedDiffusion()                     hide a random share of the tokens and predict them (the share is drawn anew
                                          for every example; with weight 1 / share the loss bounds the likelihood)
    EmbeddingDiffusion(vocab)             Gaussian diffusion on fixed token embeddings; the network predicts the token
    prelib.diffusion.Diffusion            Gaussian diffusion on pixels (notebook 07)

    PatchTokenizer(X, k)                  turns each 2x2 patch into one of k codes (k-means), and back
    sample_ar, sample_masked              the two samplers for Discrete models
All objectives have .loss(model, x, c), so every model trains with prelib.diffusion.train.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import timestep_embedding, alpha_bar, T, NULL
from .dit import patchify, unpatchify, sincos_2d, rotate


def sincos_1d(n, d):
    """Fixed position vectors for a sequence of n tokens (the encoding of the 2017 transformer)."""
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    freqs = torch.exp(-math.log(10000.0) * torch.arange(d // 2, dtype=torch.float32) / (d // 2))
    return torch.cat([torch.sin(pos * freqs), torch.cos(pos * freqs)], -1)


class Rotary:
    """Rotary positions (RoPE; notebook 03) for a sequence of n tokens. Nothing is added to the tokens. Inside every
    attention layer the queries and keys are rotated by an angle that grows with the position, so the score of a
    pair of tokens depends on how far apart they are. Pass Rotary(n) wherever a table of position vectors is expected."""

    def __init__(self, n, d=64, heads=4):
        hd = d // heads
        freqs = torch.exp(-math.log(10000.0) * torch.arange(hd // 2, dtype=torch.float32) / (hd // 2))
        self.angles = torch.arange(n, dtype=torch.float32)[:, None] * freqs[None]        # (n, head_dim / 2)


# ----------------------------------------------------------------------------- the shared backbone
class Block(nn.Module):
    """Pre-norm transformer block. A condition vector e sets a shift, a scale and a gate for the attention layer and
    for the feed-forward layer (adaLN-Zero, notebook 09). causal=True hides later tokens from earlier ones."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, h, e, causal, angles=None):
        B, n, d = h.shape
        s1, c1, g1, s2, c2, g2 = (z[:, None] for z in self.ada(e).chunk(6, -1))
        q, k, v = (z.view(B, n, self.heads, d // self.heads).transpose(1, 2) for z in self.qkv(self.n1(h) * (1 + c1) + s1).chunk(3, -1))
        if angles is not None:                                  # rotary positions: rotate queries and keys, not values
            q, k = rotate(q, angles[:n]), rotate(k, angles[:n])
        a = F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2).reshape(B, n, d)
        h = h + g1 * self.proj(a)
        return h + g2 * self.mlp(self.n2(h) * (1 + c2) + s2)


class Backbone(nn.Module):
    """pos: a table of position vectors (n, d) that is added to the tokens, or Rotary(n) for rotary positions."""

    def __init__(self, pos, d=64, depth=3, heads=4):
        super().__init__()
        self.d, self.rotary = d, isinstance(pos, Rotary)
        self.register_buffer("pos", pos.angles if self.rotary else pos)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, h, e, causal=False):
        """h: embedded tokens (B, n, d); e: condition (B, d). Returns the hidden states (B, n, d)."""
        if not self.rotary:
            h = h + self.pos[:h.shape[1]]
        for blk in self.blocks:
            h = blk(h, e, causal, self.pos if self.rotary else None)
        shift, scale = (z[:, None] for z in self.ada(e).chunk(2, -1))
        return self.norm(h) * (1 + scale) + shift


# ----------------------------------------------------------------------------- the two kinds of input and output
class Discrete(nn.Module):
    """Token ids in, logits over the vocabulary out.
    causal=True:  the input is shifted one place to the right, so position t is predicted from the tokens before t.
    causal=False: the input may contain the mask token; every position sees the whole sequence.
    n_classes > 0 adds a class label as the condition (the last class index means 'no label')."""

    def __init__(self, vocab, pos, causal, n_classes=0, mask_id=None, d=64, depth=3, heads=4):
        super().__init__()
        self.vocab, self.causal, self.d = vocab, causal, d
        self.start_id, self.mask_id = vocab, (vocab + 1 if mask_id is None else mask_id)
        self.emb = nn.Embedding(vocab + 2, d)
        self.cemb = nn.Embedding(n_classes, d) if n_classes else None
        self.backbone = Backbone(pos, d, depth, heads)
        self.out = nn.Linear(d, vocab)

    def hidden(self, x, c=None):
        if self.causal:
            x = torch.cat([torch.full_like(x[:, :1], self.start_id), x[:, :-1]], 1)
        e = self.cemb(c) if self.cemb is not None else torch.zeros(len(x), self.d)
        return self.backbone(self.emb(x), e, self.causal)

    def forward(self, x, c=None):
        return self.out(self.hidden(x, c))


class Continuous(nn.Module):
    """Vectors in (B, n, in_dim), vectors out (B, n, out_dim), given a timestep and optionally a class label."""

    def __init__(self, in_dim, out_dim, pos, n_classes=0, d=64, depth=3, heads=4):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(in_dim, d)
        self.temb = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.cemb = nn.Embedding(n_classes, d) if n_classes else None
        self.backbone = Backbone(pos, d, depth, heads)
        self.out = nn.Linear(d, out_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def hidden(self, x, t, c=None):
        e = self.temb(timestep_embedding(t))
        if self.cemb is not None:
            e = e + self.cemb(c)
        return self.backbone(self.inp(x), e)

    def forward(self, x, t, c=None):
        return self.out(self.hidden(x, t, c))


class PatchDenoiser(nn.Module):
    """Continuous on the patches of an 8x8 image: (noisy image, timestep, label) -> prediction, as in notebook 09."""

    def __init__(self, patch=2, n_classes=11, d=64, depth=3, heads=4):
        super().__init__()
        self.patch, self.d_in = patch, 64
        self.net = Continuous(patch * patch, patch * patch, sincos_2d(8 // patch, d), n_classes, d, depth, heads)

    def forward(self, x, t, c):
        return unpatchify(self.net(patchify(x, self.patch), t, c), self.patch)

    def hidden(self, x, t, c):
        return self.net.hidden(patchify(x, self.patch), t, c)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- image tokens
class PatchTokenizer:
    """Each 2x2 patch becomes the index of its nearest entry in a codebook of k patches (k-means on the training
    patches). An image is then 16 integers. Decoding looks the patches up again; it loses a little detail."""

    def __init__(self, X, k=256, patch=2, seed=0):
        from sklearn.cluster import KMeans
        self.patch, self.k = patch, k
        km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(patchify(X, patch).reshape(-1, patch * patch).numpy())
        self.codebook = torch.tensor(km.cluster_centers_, dtype=torch.float32)

    def encode(self, X):
        P = patchify(X, self.patch)
        return torch.cdist(P, self.codebook[None].expand(len(P), -1, -1)).argmin(-1)

    def decode(self, tokens):
        return unpatchify(self.codebook[tokens], self.patch)


# ----------------------------------------------------------------------------- three objectives
class Autoregressive:
    """Cross-entropy of every token given the tokens before it. first_fixed: position 0 is always the same token
    (the start-of-document token of the language data) and is not scored."""

    def __init__(self, first_fixed=False):
        self.s = 1 if first_fixed else 0

    def loss(self, model, x, c=None):
        return F.cross_entropy(model(x, c)[:, self.s:].reshape(-1, model.vocab), x[:, self.s:].reshape(-1))


class MaskedDiffusion:
    """Hide each token with probability r, where r is drawn uniformly between 0 and 1 for every example, and
    predict the hidden tokens. With the weight 1 / r this is the loss of masked (absorbing-state) diffusion
    (Austin et al., 2021; Sahoo et al., 2024), an upper bound on the negative log-likelihood per token.
    weighting="none" gives the plain average over hidden tokens (MaskGIT; Chang et al., 2022)."""

    def __init__(self, first_fixed=False, weighting="elbo"):
        self.s, self.weighting = (1 if first_fixed else 0), weighting

    def corrupt(self, model, x, r, generator=None):
        hide = torch.rand(x.shape, generator=generator) < r[:, None]
        hide[:, :self.s] = False
        return torch.where(hide, torch.full_like(x, model.mask_id), x), hide

    def loss(self, model, x, c=None, r=None, generator=None):
        if r is None:                                          # evenly spread shares, shifted by one random number: lower variance
            r = ((torch.rand(1, generator=generator) + torch.arange(len(x)) / len(x)) % 1).clamp_min(1e-3)
        xin, hide = self.corrupt(model, x, r, generator)
        nll = F.cross_entropy(model(xin, c).transpose(1, 2), x, reduction="none") * hide
        if self.weighting == "elbo":
            return (nll.sum(1) / r).mean() / (x.shape[1] - self.s)
        return nll.sum() / hide.sum().clamp_min(1)

    @torch.no_grad()
    def bound(self, model, X, c=None, repeats=8, chunk=500, seed=0):
        """Monte-Carlo estimate of the bound on the negative log-likelihood, in nats per token."""
        g = torch.Generator().manual_seed(seed)
        keep, self.weighting = self.weighting, "elbo"
        vals = [float(self.loss(model, X[i:i + chunk], None if c is None else c[i:i + chunk], generator=g)) * len(X[i:i + chunk])
                for _ in range(repeats) for i in range(0, len(X), chunk)]
        self.weighting = keep
        return sum(vals) / (repeats * len(X))


class EmbeddingDiffusion:
    """Gaussian diffusion for tokens: every token is replaced by a fixed random vector, noise is added as in
    notebook 07, and the network is trained with cross-entropy to say which token each noisy vector was.
    The estimate of the clean vectors is the average embedding under the predicted probabilities."""

    def __init__(self, vocab, dim=16, schedule="cosine", first_fixed=True, seed=0):
        g = torch.Generator().manual_seed(7919 + seed)         # its own stream: noise drawn elsewhere from seed 0, 1, ...
        E = torch.randn(vocab, dim, generator=g)                # must not be correlated with the table
        self.E = E / E.norm(dim=-1, keepdim=True) * math.sqrt(dim)            # every coordinate has unit variance
        self.ab, self.s, self.dim = alpha_bar(schedule), (1 if first_fixed else 0), dim

    def noised(self, x, t, generator=None):
        ab = self.ab[t][:, None, None]
        return ab.sqrt() * self.E[x] + (1 - ab).sqrt() * torch.randn(*x.shape, self.dim, generator=generator)

    def loss(self, model, x, c=None):
        t = torch.randint(0, T, (len(x),))
        logits = model(self.noised(x, t), t)
        return F.cross_entropy(logits[:, self.s:].reshape(-1, logits.shape[-1]), x[:, self.s:].reshape(-1))

    @torch.no_grad()
    def sample(self, model, n, length, steps=50, seed=0, first=None, forbid=(), record=False):
        """Deterministic (DDIM) sampling on the embeddings; the tokens are read off at the end."""
        g = torch.Generator().manual_seed(100_003 + seed)      # not the stream that drew the embedding table
        x = torch.randn(n, length, self.dim, generator=g)
        ts = torch.linspace(T - 1, 0, steps).round().long()
        frames = []
        for i, t in enumerate(ts):
            logits = model(x, torch.full((n,), int(t)))
            logits[..., list(forbid)] = float("-inf")
            x0 = torch.softmax(logits, -1) @ self.E
            tokens = logits.argmax(-1)
            if first is not None:
                tokens[:, 0] = first; x0[:, 0] = self.E[first]
            if record:
                frames.append(tokens.clone())
            if i == steps - 1:
                break
            ab_t, ab_s = self.ab[t], self.ab[ts[i + 1]]
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
            x = ab_s.sqrt() * x0 + (1 - ab_s).sqrt() * eps
        return (tokens, frames) if record else tokens


# ----------------------------------------------------------------------------- samplers for Discrete models
def _draw(logits, temperature, forbid, generator):
    logits = logits / temperature
    if len(forbid):
        logits[..., list(forbid)] = float("-inf")
    p = torch.softmax(logits, -1)
    tok = torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=generator).reshape(p.shape[:-1])
    return tok, p.gather(-1, tok[..., None])[..., 0]


@torch.no_grad()
def sample_ar(model, n, length, c=None, seed=0, temperature=1.0, first=None, forbid=(), record=False):
    """One token per network evaluation, left to right. Returns token ids (n, length)."""
    model.eval(); g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, length, dtype=torch.long)
    frames = []
    for t in range(length):
        if t == 0 and first is not None:
            x[:, 0] = first
        else:
            x[:, t] = _draw(model(x[:, :t + 1], c)[:, t], temperature, forbid, g)[0]
        if record:
            f = x.clone(); f[:, t + 1:] = -1; frames.append(f)
    return (x, frames) if record else x


@torch.no_grad()
def sample_masked(model, n, length, steps, c=None, seed=0, order="random", temperature=1.0, first=None, forbid=(), record=False):
    """Start from a sequence of mask tokens. In each of `steps` rounds, predict every hidden token, draw a value for
    each, and keep an equal share of them: a random share (order="random", the sampler of masked diffusion) or the
    draws the model gave the highest probability (order="confidence", as in MaskGIT). Kept tokens are never changed."""
    model.eval(); g = torch.Generator().manual_seed(seed)
    x = torch.full((n, length), model.mask_id)
    if first is not None:
        x[:, 0] = first
    todo = int((x[0] == model.mask_id).sum())
    frames = []
    for s in range(steps):
        k = int(math.ceil(todo * (s + 1) / steps)) - int(math.ceil(todo * s / steps))
        if k == 0:
            continue
        tok, conf = _draw(model(x, c), temperature, forbid, g)
        if order == "random":
            conf = torch.rand(x.shape, generator=g)
        conf[x != model.mask_id] = float("-inf")
        pick = conf.topk(k, -1).indices
        x.scatter_(1, pick, tok.gather(1, pick))
        if record:
            f = x.clone(); f[f == model.mask_id] = -1; frames.append(f)
    return (x, frames) if record else x


# ----------------------------------------------------------------------------- read-outs
def logprobs_ar(model, chunk=500):
    """For prelib.world.evaluate_lm: tokens (n, L) -> log p(token t | tokens before t), array (n, L, vocab)."""
    @torch.no_grad()
    def fn(tokens):
        model.eval(); t = torch.as_tensor(tokens)
        return torch.cat([F.log_softmax(model(t[i:i + chunk]), -1) for i in range(0, len(t), chunk)]).numpy()
    return fn


def logprobs_masked(model, chunk=500):
    """The same for a masked model, used left to right: to predict token t, tokens t and later are all masked.
    One network evaluation per position."""
    @torch.no_grad()
    def fn(tokens):
        model.eval(); tok = torch.as_tensor(tokens); out = torch.zeros(*tok.shape, model.vocab)
        for i in range(0, len(tok), chunk):
            x = tok[i:i + chunk]
            for t in range(1, x.shape[1]):
                xin = x.clone(); xin[:, t:] = model.mask_id
                out[i:i + chunk, t] = F.log_softmax(model(xin)[:, t], -1)
        return out.numpy()
    return fn


@torch.no_grad()
def mean_hidden(model, x, *args, chunk=500):
    """Mean hidden state over the positions of each sequence: the features used for linear probes."""
    model.eval()
    return torch.cat([model.hidden(x[i:i + chunk], *[a[i:i + chunk] for a in args]).mean(1) for i in range(0, len(x), chunk)])
