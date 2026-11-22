"""Diffusion transformers on 8x8 digits (notebook 09 onwards): every design choice is an argument.

    DiT(d, depth, heads, patch, cond, pos)
        patch   1, 2 or 4 pixels per side: 64, 16 or 4 tokens
        cond    how timestep and label reach the tokens:
                "adaln-zero"  shift, scale and a gate on every block, the gate starting at zero (Peebles & Xie, 2022)
                "adaln"       shift and scale only (FiLM; Perez et al., 2017)
                "cross"       cross-attention to two condition tokens
                "in-context"  the two condition tokens are appended to the sequence (U-ViT; Bao et al., 2022)
        pos     "learned" | "sincos" (fixed 2-D sinusoids) | "rope" (2-D rotary, inside attention) | "none"
    Flow(pred, t_sampling)          flow matching / rectified flow (Lipman et al., 2022; Liu et al., 2022)
    Masked(diff, ratio)             train on a random subset of the tokens (after Zheng et al., 2023; Gao et al., 2023)
    Repa(diff, encoder, weight)     align hidden tokens with a pretrained encoder (Yu et al., 2024)

A DiT takes (x, t, c) like prelib.diffusion.Denoiser, so it trains with prelib.diffusion.train.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import timestep_embedding, no_condition, Diffusion, T, NULL


def patchify(x, p):
    n, g = len(x), 8 // p
    return x.view(n, g, p, g, p).permute(0, 1, 3, 2, 4).reshape(n, g * g, p * p)


def unpatchify(tok, p):
    n, g = len(tok), 8 // p
    return tok.view(n, g, g, p, p).permute(0, 1, 3, 2, 4).reshape(n, 64)


def grid_positions(g):
    """(row, column) of each of the g*g tokens, in reading order."""
    r, c = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    return r.reshape(-1).float(), c.reshape(-1).float()


def sincos_2d(g, d):
    """Fixed position vectors: half of the dimensions encode the row, half the column."""
    r, c = grid_positions(g)
    freqs = torch.exp(-math.log(100.0) * torch.arange(d // 4, dtype=torch.float32) / (d // 4))
    return torch.cat([f(p[:, None] * freqs[None]) for p in (r, c) for f in (torch.sin, torch.cos)], -1)


def rope_angles(g, head_dim, extra=0):
    """Rotation angle of every pair of coordinates for every token: (extra + g*g, head_dim / 2).
    The first half of the pairs rotates with the row, the second half with the column (2-D RoPE).
    `extra` leading tokens (condition tokens) are not rotated."""
    r, c = grid_positions(g)
    freqs = torch.exp(-math.log(100.0) * torch.arange(head_dim // 4, dtype=torch.float32) / (head_dim // 4))
    ang = torch.cat([r[:, None] * freqs[None], c[:, None] * freqs[None]], -1)
    return torch.cat([torch.zeros(extra, ang.shape[1]), ang])


def rotate(x, ang):
    """Rotate consecutive pairs of coordinates of x (..., tokens, head_dim) by the angles (tokens, head_dim / 2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = ang.cos(), ang.sin()
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, heads, angles=None):
        super().__init__()
        self.heads, self.hd = heads, d // heads
        self.q, self.kv, self.proj = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.angles, self.record, self.last = angles, False, None

    def forward(self, x, context=None):
        B, n, _ = x.shape
        src = x if context is None else context
        q = self.q(x).view(B, n, self.heads, self.hd).transpose(1, 2)
        k, v = (z.view(B, src.shape[1], self.heads, self.hd).transpose(1, 2) for z in self.kv(src).chunk(2, -1))
        if self.angles is not None and context is None:                 # rotary positions act on queries and keys only
            q, k = rotate(q, self.angles), rotate(k, self.angles)
        att = (q @ k.transpose(-1, -2) / math.sqrt(self.hd)).softmax(-1)
        if self.record:
            self.last = att.detach()
        return self.proj((att @ v).transpose(1, 2).reshape(B, n, -1))


def modulate(h, shift, scale):
    return h * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, d, heads, cond, angles=None):
        super().__init__()
        self.cond = cond
        ada = cond.startswith("adaln")
        self.n1, self.n2 = nn.LayerNorm(d, elementwise_affine=not ada), nn.LayerNorm(d, elementwise_affine=not ada)
        self.attn = Attention(d, heads, angles)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        if ada:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d if cond == "adaln-zero" else 4 * d))
            if cond == "adaln-zero":                                    # every block starts as the identity
                nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)
        if cond == "cross":
            self.n3, self.cross = nn.LayerNorm(d), Attention(d, heads)

    def forward(self, h, e, ctx):
        if self.cond == "adaln-zero":
            s1, c1, g1, s2, c2, g2 = self.ada(e).chunk(6, -1)
            h = h + g1[:, None] * self.attn(modulate(self.n1(h), s1, c1))
            return h + g2[:, None] * self.mlp(modulate(self.n2(h), s2, c2))
        if self.cond == "adaln":
            s1, c1, s2, c2 = self.ada(e).chunk(4, -1)
            h = h + self.attn(modulate(self.n1(h), s1, c1))
            return h + self.mlp(modulate(self.n2(h), s2, c2))
        h = h + self.attn(self.n1(h))
        if self.cond == "cross":
            h = h + self.cross(self.n3(h), ctx)
        return h + self.mlp(self.n2(h))


class DiT(nn.Module):
    def __init__(self, d=64, depth=3, heads=4, patch=2, cond="adaln-zero", pos="learned", n_classes=11, hidden_layer=None, align_dim=None):
        super().__init__()
        self.d_in, self.d, self.patch, self.cond, self.pos_kind = 64, d, patch, cond, pos
        self.g, self.n_tok = 8 // patch, (8 // patch) ** 2
        self.extra = 2 if cond == "in-context" else 0
        self.embed = nn.Linear(patch * patch, d)
        if pos == "learned":
            self.pos = nn.Parameter(torch.randn(self.n_tok, d) * 0.02)
        elif pos == "sincos":
            self.register_buffer("pos", sincos_2d(self.g, d))
        angles = rope_angles(self.g, d // heads, self.extra) if pos == "rope" else None
        self.temb = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.cemb = nn.Embedding(n_classes, d)
        self.blocks = nn.ModuleList(Block(d, heads, cond, angles) for _ in range(depth))
        ada = cond.startswith("adaln")
        self.norm = nn.LayerNorm(d, elementwise_affine=not ada)
        if ada:
            self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
            nn.init.zeros_(self.ada_out[1].weight); nn.init.zeros_(self.ada_out[1].bias)
        self.out = nn.Linear(d, patch * patch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.hidden_layer, self.hidden = hidden_layer, None             # keep the tokens after this block (for Repa and probes)
        if align_dim:
            self.align = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, align_dim))

    def position(self):
        return self.pos if self.pos_kind in ("learned", "sincos") else 0.0

    def forward(self, x, t, c, keep=None):
        """keep: (B, k) indices of the image tokens the blocks may see (masked training). The last block always
        sees all positions, with a learned mask token wherever a token was dropped."""
        B = len(x)
        te, ce = self.temb(timestep_embedding(t)), self.cemb(c)
        e, ctx = te + ce, torch.stack([te, ce], 1)
        h = self.embed(patchify(x, self.patch)) + self.position()
        if keep is not None:
            assert self.pos_kind != "rope" and not self.extra, "masked training is implemented for additive positions only"
            h = torch.gather(h, 1, keep[..., None].expand(-1, -1, self.d))
        if self.extra:
            h = torch.cat([ctx, h], 1)
        for i, blk in enumerate(self.blocks):
            if keep is not None and i == len(self.blocks) - 1:
                full = (self.mask_token + self.position()).expand(B, self.n_tok, self.d)
                h = full.scatter(1, keep[..., None].expand(-1, -1, self.d), h)
            h = blk(h, e, ctx)
            if i == self.hidden_layer:
                self.hidden = h[:, self.extra:]
        h = self.norm(h[:, self.extra:])
        if self.cond.startswith("adaln"):
            h = modulate(h, *self.ada_out(e).chunk(2, -1))
        return unpatchify(self.out(h), self.patch)

    def n_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("align"))

    def cost(self):
        """A rough count of multiplications per forward pass, in millions: tokens * depth * (12 d^2 + tokens * d)."""
        n = self.n_tok + self.extra
        return n * len(self.blocks) * (12 * self.d ** 2 + n * self.d) / 1e6

    @torch.no_grad()
    def attention(self, x, t, c, block=-1):
        """Attention of one block, averaged over heads: (B, tokens, tokens) over the image tokens."""
        a = self.blocks[block].attn
        a.record = True
        self(x, t, c)
        a.record = False
        return a.last.mean(1)[:, self.extra:, self.extra:]


# ----------------------------------------------------------------------------- flow matching
class Flow:
    """x_s = (1 - s) * data + s * noise for s in [0, 1]: a straight line from each image to its noise sample.
    pred="v": the network predicts the velocity along the line, noise - data. pred="x0": it predicts the data.
    t_sampling: how s is drawn in training, "uniform" or "logit-normal" (more weight on the middle; Esser et al., 2024).
    The network's time input is 999 * s, so the same models as for Diffusion can be used."""

    def __init__(self, pred="v", t_sampling="uniform"):
        self.pred, self.t_sampling = pred, t_sampling

    def pair(self, x0):
        s = torch.sigmoid(torch.randn(len(x0))) if self.t_sampling == "logit-normal" else torch.rand(len(x0))
        eps = torch.randn_like(x0)
        xs = (1 - s[:, None]) * x0 + s[:, None] * eps
        return xs, s * (T - 1), (eps - x0 if self.pred == "v" else x0), torch.ones(len(x0))

    def loss(self, model, x0, c):
        xs, t, target, w = self.pair(x0)
        return ((model(xs, t, c) - target) ** 2).mean()

    def predict(self, model, x, s, c, w=0.0):
        t = torch.full((len(x),), s * (T - 1))
        if w == 0:
            return model(x, t, c)
        oc, ou = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([c, no_condition(c)])).chunk(2)
        return (1 + w) * oc - w * ou

    @torch.no_grad()
    def sample(self, model, labels, steps=50, w=0.0, seed=0, x_T=None, record=False):
        """Euler steps along the learned velocity from s = 1 (noise) to s = 0 (data). Network evaluations: steps."""
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(len(labels), 64, generator=g) if x_T is None else x_T.clone()
        grid = torch.linspace(1, 0, steps + 1)
        frames = []
        for i in range(steps):
            s, s_next = float(grid[i]), float(grid[i + 1])
            out = self.predict(model, x, s, labels, w)
            x0 = x - s * out if self.pred == "v" else out                # the current estimate of the clean image
            if record:
                frames.append((s, x.clone(), x0.clone()))
            x = x0 + (s_next / s) * (x - x0)                             # for pred="v" this is x - (s - s_next) * velocity
        return (x, frames) if record else x


def training_pair(diff, x0):
    """(noisy input, time input, target, weight per example) for a Flow or a Diffusion."""
    if isinstance(diff, Flow):
        return diff.pair(x0)
    t = torch.randint(0, T, (len(x0),))
    eps = torch.randn_like(x0)
    return diff.q_sample(x0, t, eps), t, diff.target(x0, t, eps), diff.weight(t)


# ----------------------------------------------------------------------------- two training accelerators
class Masked:
    """Masked training: all blocks but the last see a random subset of the image tokens, so each step is cheaper.
    The loss is the usual one on the visible patches plus `hidden_weight` times the same loss on the hidden patches.
    After `unmask_after` calls, training continues on all tokens (a short unmasked phase, as in Zheng et al., 2023).
    A simplified variant of MaskDiT, not a reimplementation."""

    def __init__(self, diff, ratio=0.5, hidden_weight=0.1, unmask_after=None):
        self.diff, self.ratio, self.hidden_weight, self.unmask_after, self.calls = diff, ratio, hidden_weight, unmask_after, 0

    def loss(self, model, x0, c):
        self.calls += 1
        if self.unmask_after is not None and self.calls > self.unmask_after:
            return self.diff.loss(model, x0, c)
        x, t, target, w = training_pair(self.diff, x0)
        k = max(1, int(round(model.n_tok * (1 - self.ratio))))
        keep = torch.rand(len(x0), model.n_tok).argsort(1)[:, :k]
        hidden = torch.ones(len(x0), model.n_tok).scatter_(1, keep, 0.0)
        err = ((patchify(model(x, t, c, keep=keep), model.patch) - patchify(target, model.patch)) ** 2).mean(-1)      # (B, tokens)
        per = (err * (1 - hidden)).sum(1) / (1 - hidden).sum(1) + self.hidden_weight * (err * hidden).sum(1) / hidden.sum(1)
        return (w * per).mean()

    def sample(self, *a, **k):
        return self.diff.sample(*a, **k)


class Repa:
    """Representation alignment: the tokens after block `model.hidden_layer`, passed through the small MLP
    `model.align`, are pulled towards the patch features that a frozen pretrained encoder computes for the CLEAN image
    (cosine similarity, weight `weight`), in addition to the usual loss."""

    def __init__(self, diff, encoder, weight=0.5):
        self.diff, self.encoder, self.weight = diff, encoder, weight

    def loss(self, model, x0, c):
        x, t, target, w = training_pair(self.diff, x0)
        out = model(x, t, c)
        with torch.no_grad():
            feats = self.encoder.tokens(x0)
        align = F.cosine_similarity(model.align(model.hidden), feats, dim=-1).mean()
        return (w * ((out - target) ** 2).mean(-1)).mean() - self.weight * align

    def sample(self, *a, **k):
        return self.diff.sample(*a, **k)


@torch.no_grad()
def hidden_features(model, X, y, diff, level=0.3, seed=0):
    """Mean hidden token (after block model.hidden_layer) for images noised to `level` of the way to pure noise:
    the features a linear probe is fitted on."""
    g = torch.Generator().manual_seed(seed)
    eps = torch.randn(X.shape, generator=g)
    if isinstance(diff, Flow):
        x, t = (1 - level) * X + level * eps, torch.full((len(X),), level * (T - 1))
    else:
        t = torch.full((len(X),), int(level * (T - 1)))
        x = diff.q_sample(X, t, eps)
    model(x, t, torch.full((len(X),), NULL))
    return model.hidden.mean(1)
