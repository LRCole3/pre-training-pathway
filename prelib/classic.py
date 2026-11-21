"""Language models from before recurrent networks (notebook 01): smoothed n-grams, a fixed-window neural
language model, and skip-gram word embeddings. Every model exposes `logprobs(tokens) -> (n, L, VOCAB)` so that
`world.evaluate_lm` can score it against the oracle floor."""
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import world as W, set_seed

V = W.VOCAB


def context_ids(tokens, m):
    """For each position t, one integer encoding the m-1 tokens before it (PAD-filled at the document start)."""
    n, L = tokens.shape
    pad = np.concatenate([np.zeros((n, m - 1), dtype=np.int64), tokens], 1)
    cid = np.zeros((n, L), dtype=np.int64)
    for j in range(m - 1):
        cid = cid * V + pad[:, j:j + L]
    return cid


class NGram:
    """n-gram model with add-k smoothing or interpolated Kneser-Ney (Chen & Goodman, 1996)."""

    def __init__(self, n, smoothing="kn", k=0.1, discount=0.75):
        self.n, self.smoothing, self.k, self.D = n, smoothing, k, discount

    def fit(self, tokens):
        self.ctx, self.C = {}, {}
        w = tokens[:, 1:].ravel()
        for m in range(1, self.n + 1):
            cid = context_ids(tokens, m)[:, 1:].ravel()
            u, inv = np.unique(cid, return_inverse=True)
            self.ctx[m] = u
            self.C[m] = sp.csr_matrix((np.ones(len(w)), (inv, w)), shape=(len(u), V))
        # Kneser-Ney: lower orders count in how many distinct contexts a word appears, not how often
        self.K = {}
        for m in range(1, self.n):
            hi = self.C[m + 1].tocoo()
            lower = self.ctx[m + 1][hi.row] % (V ** (m - 1)) if m > 1 else np.zeros(len(hi.row), dtype=np.int64)
            rows = np.searchsorted(self.ctx[m], lower)
            self.K[m] = sp.csr_matrix((np.ones(len(rows)), (rows, hi.col)), shape=(len(self.ctx[m]), V))
        return self

    def _rows(self, mat, m, cid):
        idx = np.searchsorted(self.ctx[m], cid).clip(max=len(self.ctx[m]) - 1)
        found = self.ctx[m][idx] == cid
        out = np.zeros((len(cid), V))
        out[found] = mat[idx[found]].toarray()
        return out

    def probs_flat(self, tokens):
        n_, L = tokens.shape
        if self.smoothing == "add-k":
            A = self._rows(self.C[self.n], self.n, context_ids(tokens, self.n).ravel())
            return (A + self.k) / (A.sum(1, keepdims=True) + self.k * V)
        A = self._rows(self.K[1] if self.n > 1 else self.C[1], 1, np.zeros(n_ * L, dtype=np.int64))
        p = (np.maximum(A - self.D, 0) + self.D * (A > 0).sum(1, keepdims=True) / V) / A.sum(1, keepdims=True)
        for m in range(2, self.n + 1):
            A = self._rows(self.C[m] if m == self.n else self.K[m], m, context_ids(tokens, m).ravel())
            tot = A.sum(1, keepdims=True)
            pm = (np.maximum(A - self.D, 0) + self.D * (A > 0).sum(1, keepdims=True) * p) / np.maximum(tot, 1)
            p = np.where(tot > 0, pm, p)
        return p

    def logprobs(self, tokens, chunk=400):
        out = [np.log(np.maximum(self.probs_flat(tokens[i:i + chunk]), 1e-300)).reshape(-1, tokens.shape[1], V)
               for i in range(0, len(tokens), chunk)]
        return np.concatenate(out)

    def unseen_fraction(self, tokens):
        """Share of test positions whose (n-1)-token context never occurred in training."""
        cid = context_ids(tokens, self.n)[:, 1:].ravel()
        idx = np.searchsorted(self.ctx[self.n], cid).clip(max=len(self.ctx[self.n]) - 1)
        return float((self.ctx[self.n][idx] != cid).mean())

    def sample(self, length=24, seed=0):
        rng = np.random.default_rng(seed)
        tok = np.full((1, length), W.PAD, dtype=np.int64); tok[0, 0] = W.BOS
        for t in range(1, length):
            p = self.probs_flat(tok[:, :t + 1])[t]
            tok[0, t] = rng.choice(V, p=p / p.sum())
        return tok[0]


class WindowMLP(nn.Module):
    """Fixed-window neural language model: embed the previous `window` tokens, concatenate, MLP, softmax."""

    def __init__(self, window=4, d=32, hidden=256):
        super().__init__()
        self.window = window
        self.emb = nn.Embedding(V, d)
        self.net = nn.Sequential(nn.Linear(window * d, hidden), nn.Tanh(), nn.Linear(hidden, V))

    def forward(self, tokens):
        """tokens (B, L) -> logits (B, L, V); logits[:, t] uses tokens t-window .. t-1."""
        B, L = tokens.shape
        pad = torch.cat([torch.zeros(B, self.window, dtype=torch.long), tokens], 1)[:, :L + self.window - 1]
        ctx = pad.unfold(1, self.window, 1)                              # (B, L, window)
        return self.net(self.emb(ctx).flatten(2))

    @torch.no_grad()
    def logprobs(self, tokens):
        self.eval()
        t = torch.as_tensor(tokens)
        return torch.cat([F.log_softmax(self(t[i:i + 500]), -1) for i in range(0, len(t), 500)]).numpy()


def train_lm(model, tokens, steps=1500, batch=64, lr=2e-3, seed=0, log_every=0, clip=1.0, callback=None):
    """Next-token training loop shared by the window model, RNNs and transformers in this series."""
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    it = W.batches(tokens, batch, seed)
    losses, t0 = [], time.time()
    model.train()
    for step in range(steps):
        x = next(it)
        logits = model(x)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(loss.item())
        if callback is not None:
            callback(step, model)
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:5d}  loss {np.mean(losses[-log_every:]):.3f}  ({time.time() - t0:.0f}s)")
    return losses


def skipgram_pairs(tokens, window=2):
    """(centre, context) pairs within +-window, over the whole corpus."""
    c, o = [], []
    for off in range(1, window + 1):
        c += [tokens[:, :-off].ravel(), tokens[:, off:].ravel()]
        o += [tokens[:, off:].ravel(), tokens[:, :-off].ravel()]
    return torch.as_tensor(np.concatenate(c)), torch.as_tensor(np.concatenate(o))


class SkipGram(nn.Module):
    def __init__(self, vocab=V, d=24):
        super().__init__()
        self.inp, self.out = nn.Embedding(vocab, d), nn.Embedding(vocab, d)
        nn.init.normal_(self.inp.weight, std=0.1); nn.init.zeros_(self.out.weight)

    def loss_softmax(self, c, o):
        """Full softmax: one score for every word in the vocabulary (cost grows with the vocabulary)."""
        return F.cross_entropy(self.inp(c) @ self.out.weight.T, o)

    def loss_negative(self, c, o, neg):
        """Negative sampling (Mikolov et al., 2013): tell the true context word from k random ones."""
        v = self.inp(c)
        pos = F.logsigmoid((v * self.out(o)).sum(-1))
        negs = F.logsigmoid(-(self.out(neg) @ v[..., None])[..., 0]).sum(-1)
        return -(pos + negs).mean()


def train_skipgram(tokens, method="negative", d=24, steps=1500, batch=1024, k=5, lr=0.01, seed=0, snapshot_every=0):
    set_seed(seed)
    c, o = skipgram_pairs(tokens)
    model = SkipGram(d=d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    freq = torch.as_tensor(np.bincount(tokens.ravel(), minlength=V) ** 0.75, dtype=torch.float)
    snaps, t0 = [], time.time()
    for step in range(steps):
        idx = torch.randint(0, len(c), (batch,))
        if method == "softmax":
            loss = model.loss_softmax(c[idx], o[idx])
        else:
            loss = model.loss_negative(c[idx], o[idx], torch.multinomial(freq, batch * k, replacement=True).view(batch, k))
        opt.zero_grad(); loss.backward(); opt.step()
        if snapshot_every and step % snapshot_every == 0:
            snaps.append(model.inp.weight.detach().numpy().copy())
    return model.inp.weight.detach().numpy().copy(), snaps, time.time() - t0


def neighbours(E, word_id, k=5, ids=None):
    """Indices of the k nearest words by cosine similarity (restricted to `ids` if given)."""
    ids = np.arange(len(E)) if ids is None else np.asarray(ids)
    X = E[ids] / np.linalg.norm(E[ids], axis=1, keepdims=True)
    q = E[word_id] / np.linalg.norm(E[word_id])
    order = np.argsort(-(X @ q))
    return [int(ids[i]) for i in order if ids[i] != word_id][:k]


def class_purity(E, k=5):
    """Share of each filler word's k nearest filler words that belong to its own class."""
    fill = np.arange(W.FILL0, V)
    return float(np.mean([np.mean(W.WORD_CLASS[neighbours(E, i, k, fill)] == W.WORD_CLASS[i]) for i in fill]))
