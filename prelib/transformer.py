"""A decoder-only transformer whose design choices are switches (notebooks 03-05, 10).

Defaults reproduce the 2017 block (post-LayerNorm, sinusoidal positions, ReLU feed-forward, multi-head attention).
Each keyword swaps one choice for what later models use:

    norm="pre", norm_type="rms"      pre-normalisation with RMSNorm
    pos="rope" | "alibi" | "relative" | "learned" | "none"
    ffn="gelu" | "swiglu"
    n_kv_head=1 or 2                 grouped-query attention (fewer key/value heads than query heads)
    moe=dict(n_experts=4, top_k=1)   mixture-of-experts feed-forward layer with a learned router
    causal=False                     every position sees the whole sequence (encoders, notebook 04)
    tie=True                         share the input embedding with the output layer (notebook 05)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import world as W

V = W.VOCAB


class RMSNorm(nn.Module):
    """x / rms(x) * gain: LayerNorm without the mean subtraction and bias (Zhang & Sennrich, 2019)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gain, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gain


def sinusoidal(T, d):
    """The fixed position vectors of Vaswani et al. (2017): sines and cosines of geometrically spaced frequencies."""
    pos = torch.arange(T)[:, None]
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2) / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe


def rope_rotate(x, base=10000.0):
    """Rotary position embedding (Su et al., 2021). x: (..., T, head_dim). Each consecutive pair of coordinates is
    rotated by an angle proportional to the position, so q_m . k_n depends only on the offset m - n."""
    T, hd = x.shape[-2], x.shape[-1]
    freq = base ** (-torch.arange(0, hd, 2) / hd)
    ang = torch.arange(T)[:, None] * freq[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, n_head, n_kv_head=None, pos="sinusoidal", max_rel=32, causal=True):
        super().__init__()
        self.H, self.Hkv, self.hd, self.pos, self.causal = n_head, n_kv_head or n_head, d // n_head, pos, causal
        assert n_head % self.Hkv == 0
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, self.Hkv * self.hd, bias=False)
        self.v = nn.Linear(d, self.Hkv * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ablate = None                          # set to a head index to remove that head (see notebook 03, section 8)
        if pos == "relative":                       # one learned bias per head and (clipped) distance back;
            self.max_rel = max_rel                  # a bidirectional model also needs entries for "ahead"
            self.rel = nn.Parameter(torch.zeros(n_head, max_rel + 1 if causal else 2 * max_rel + 1))
        if pos == "alibi":                          # fixed, not learned: a different slope per head
            self.register_buffer("slopes", torch.tensor([2 ** (-8 * (h + 1) / n_head) for h in range(n_head)]))

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.H, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.Hkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.Hkv, self.hd).transpose(1, 2)
        if self.pos == "rope":
            q, k = rope_rotate(q), rope_rotate(k)
        if self.Hkv != self.H:                      # each key/value head is shared by a group of query heads
            k = k.repeat_interleave(self.H // self.Hkv, 1); v = v.repeat_interleave(self.H // self.Hkv, 1)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.hd)
        back = (torch.arange(T)[:, None] - torch.arange(T)[None]).abs()                    # distance between positions i and j
        if self.pos == "relative" and self.causal:
            s = s + self.rel[:, back.clamp(max=self.max_rel)]
        elif self.pos == "relative":                # signed offset i - j, clipped, shifted to index from 0
            s = s + self.rel[:, (torch.arange(T)[:, None] - torch.arange(T)[None]).clamp(-self.max_rel, self.max_rel) + self.max_rel]
        if self.pos == "alibi":
            s = s - self.slopes[:, None, None] * back
        if self.causal:                             # a language model must not see the tokens it is asked to predict
            s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        w = torch.softmax(s, -1)
        self.last_weights = w.detach()              # (B, heads, T, T), kept for plotting
        out = w @ v                                 # (B, heads, T, head_dim)
        if self.ablate is not None:                 # experiments only: silence one head by zeroing what it writes
            out = out.clone(); out[:, self.ablate] = 0
        return self.o(out.transpose(1, 2).reshape(B, T, -1))


class FeedForward(nn.Module):
    def __init__(self, d, hidden, kind="relu"):
        super().__init__()
        self.kind = kind
        if kind == "swiglu":                        # SwiGLU (Shazeer, 2020): a gated unit; 2/3 width keeps the parameter count
            hidden = int(2 * hidden / 3)
            self.w1, self.w3, self.w2 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False), nn.Linear(hidden, d, bias=False)
        else:
            self.w1, self.w2 = nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.w2(F.relu(self.w1(x)) if self.kind == "relu" else F.gelu(self.w1(x)))


class MoE(nn.Module):
    """Mixture-of-experts feed-forward layer (Shazeer et al., 2017; Fedus et al., 2021).

    A router scores the experts for every token; only the top_k experts are used, weighted by a softmax over their
    scores. `balance` weights the auxiliary loss that keeps the experts equally loaded.
    For clarity this implementation evaluates every expert and masks; real systems run only the chosen ones."""

    def __init__(self, d, hidden, n_experts=4, top_k=1, balance=0.01, kind="gelu"):
        super().__init__()
        self.E, self.top_k, self.balance = n_experts, top_k, balance
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(FeedForward(d, hidden, kind) for _ in range(n_experts))

    def forward(self, x):
        logits = self.router(x)                                           # (B, T, E)
        top = logits.topk(self.top_k, -1)
        gate = torch.zeros_like(logits).scatter(-1, top.indices, torch.softmax(top.values, -1))
        out = sum(gate[..., e:e + 1] * expert(x) for e, expert in enumerate(self.experts))
        probs = torch.softmax(logits, -1)
        load = (gate > 0).float().mean((0, 1)) / self.top_k               # share of routed tokens per expert
        self.aux = self.balance * self.E * (load.detach() * probs.mean((0, 1))).sum()     # = balance at uniform routing
        self.last_gate, self.last_load, self.last_logits = gate.detach(), load.detach(), logits.detach()
        return out


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d"]
        Norm = RMSNorm if cfg["norm_type"] == "rms" else nn.LayerNorm
        self.pre = cfg["norm"] == "pre"
        self.n1, self.n2 = Norm(d), Norm(d)
        self.attn = Attention(d, cfg["n_head"], cfg["n_kv_head"], cfg["pos"], causal=cfg.get("causal", True))
        hidden = cfg["ffn_mult"] * d
        self.ffn = MoE(d, hidden, kind=cfg["ffn"], **cfg["moe"]) if cfg["moe"] else FeedForward(d, hidden, cfg["ffn"])

    def forward(self, x):
        if self.pre:                                 # pre-norm: the residual path is never normalised
            x = x + self.attn(self.n1(x))
            return x + self.ffn(self.n2(x))
        x = self.n1(x + self.attn(x))                # post-norm (2017): normalise after adding
        return self.n2(x + self.ffn(x))


class Transformer(nn.Module):
    def __init__(self, d=64, n_layer=2, n_head=4, block=64, pos="sinusoidal", norm="post", norm_type="layer", ffn="relu",
                 ffn_mult=4, n_kv_head=None, moe=None, vocab=V, causal=True, tie=False):
        super().__init__()
        self.cfg = dict(d=d, n_layer=n_layer, n_head=n_head, block=block, pos=pos, norm=norm, norm_type=norm_type, ffn=ffn,
                        ffn_mult=ffn_mult, n_kv_head=n_kv_head, moe=moe, vocab=vocab, causal=causal, tie=tie)
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.1)
        if pos == "learned":
            self.pos_emb = nn.Embedding(block, d); nn.init.normal_(self.pos_emb.weight, std=0.1)
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(n_layer))
        self.final = (RMSNorm if norm_type == "rms" else nn.LayerNorm)(d) if norm == "pre" else nn.Identity()
        self.out = nn.Linear(d, vocab)
        if tie:                                     # weight tying (Press & Wolf, 2016): one matrix reads tokens in and scores them out
            self.out.weight = self.emb.weight

    def hidden(self, tokens):
        """Final hidden states (B, T, d); position t has seen tokens 0..t."""
        B, T = tokens.shape
        x = self.emb(tokens)
        if self.cfg["pos"] == "sinusoidal":
            x = x + sinusoidal(T, self.cfg["d"])
        elif self.cfg["pos"] == "learned":          # no entry exists beyond the training length: reuse the last one
            x = x + self.pos_emb(torch.arange(T).clamp(max=self.cfg["block"] - 1))
        for blk in self.blocks:
            x = blk(x)
        return self.final(x)

    @torch.no_grad()
    def residuals(self, tokens):
        """The residual stream after the embedding and after each block: list of arrays (B, T, d)."""
        self.eval()
        tokens = torch.as_tensor(tokens)
        x = self.emb(tokens)
        if self.cfg["pos"] == "sinusoidal":
            x = x + sinusoidal(tokens.shape[1], self.cfg["d"])
        elif self.cfg["pos"] == "learned":
            x = x + self.pos_emb(torch.arange(tokens.shape[1]).clamp(max=self.cfg["block"] - 1))
        out = [x.numpy()]
        for blk in self.blocks:
            x = blk(x); out.append(x.numpy())
        return out

    @property
    def aux_loss(self):
        return sum(b.ffn.aux for b in self.blocks) if self.cfg["moe"] else 0.0

    def token_logits(self, tokens):
        """Logits for the token AT each position (no shift): used by masked-language-model training (notebook 04)."""
        return self.out(self.hidden(tokens))

    def forward(self, tokens):
        logits = self.out(self.hidden(tokens))                                 # position t predicts token t+1
        return torch.cat([torch.zeros_like(logits[:, :1]), logits[:, :-1]], 1)

    @torch.no_grad()
    def logprobs(self, tokens, chunk=250):
        self.eval()
        t = torch.as_tensor(tokens)
        return torch.cat([F.log_softmax(self(t[i:i + chunk]), -1) for i in range(0, len(t), chunk)]).numpy()

    @torch.no_grad()
    def features(self, tokens):
        self.eval()
        t = torch.as_tensor(tokens)
        return torch.cat([self.hidden(t[i:i + 500]).mean(1) for i in range(0, len(t), 500)]).numpy()

    @torch.no_grad()
    def attention_maps(self, tokens):
        """Attention weights for ONE document: array (n_layer, n_head, T, T); [l, h, t, j] = position t reading j."""
        self.eval()
        self.hidden(torch.as_tensor(tokens)[None])
        return np.stack([b.attn.last_weights[0].numpy() for b in self.blocks])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def kv_cache_floats_per_token(self):
        """Numbers stored per generated token when keys and values are cached for fast generation."""
        c = self.cfg
        return 2 * c["n_layer"] * (c["n_kv_head"] or c["n_head"]) * (c["d"] // c["n_head"])
