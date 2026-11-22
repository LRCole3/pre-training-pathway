"""What it takes to make pretraining scale (notebook 05): tokenisers, the optimiser recipe, size and data.

    train_bpe(word_counts, n_merges)   byte-pair encoding, learned from word frequencies
    Tokenizer                          text <-> sub-word ids for the synthetic language (characters ... whole words)
    fit(model, ids, ...)               one training loop with the choices of the recipe as arguments:
                                       optimiser, weight decay, warm-up, schedule, clipping
    per_word_loss(model, ids, ...)     loss per WORD, comparable across tokenisers
    word_losses(model, ids, spans, which)  loss of one chosen word per document
"""
import collections, math, time
import numpy as np
import torch
import torch.nn.functional as F
from . import world as W, set_seed

EOW = "_"                                    # end-of-word marker, so that "is_" (a suffix) differs from "is" inside a word


# ----------------------------------------------------------------------------- byte-pair encoding
def word_counts(tokens):
    """How often each word of the synthetic language occurs in an array of word-level token ids."""
    c = np.bincount(np.asarray(tokens).reshape(-1), minlength=W.VOCAB)
    return {W.WORDS[t]: int(c[t]) for t in range(W.SET, W.VOCAB) if t != W.MASK and c[t] > 0}


def _merge(seq, a, b):
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(seq[i]); i += 1
    return tuple(out)


def train_bpe(counts, n_merges):
    """Byte-pair encoding (Sennrich et al., 2015). Start from characters; repeatedly merge the adjacent pair of
    symbols that occurs most often. Returns the list of merges [(left, right, count)] in the order they were made."""
    seqs = {w: tuple(w) + (EOW,) for w in counts}
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, s in seqs.items():
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += counts[w]
        if not pairs:
            break
        (a, b), c = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append((a, b, c))
        seqs = {w: _merge(s, a, b) for w, s in seqs.items()}
    return merges


class Tokenizer:
    """Sub-word tokeniser for the synthetic language. n_merges=0 gives characters; enough merges give whole words.
    Ids: 0 padding, 1 start of document, then one id per symbol."""

    def __init__(self, counts, n_merges):
        self.merges = train_bpe(counts, n_merges)
        chars = sorted({ch for w in counts for ch in w} | {EOW})
        self.symbols = chars + [a + b for a, b, _ in self.merges]
        self.id = {s: i + 2 for i, s in enumerate(self.symbols)}
        self.vocab = len(self.symbols) + 2
        self._cache = {}

    def split(self, word):
        """A word as a tuple of symbols, applying the merges in the order they were learned."""
        if word not in self._cache:
            s = tuple(word) + (EOW,)
            for a, b, _ in self.merges:
                s = _merge(s, a, b)
            self._cache[word] = s
        return self._cache[word]

    def encode_words(self, words):
        return [self.id[s] for w in words for s in self.split(w)]

    def encode(self, word_tokens, n_words=None, length=None):
        """Word-level documents (array of ids from prelib.world, start token first) -> padded array of sub-word ids,
        plus the number of sub-word tokens in each document."""
        docs = np.asarray(word_tokens)
        docs = docs[:, :1 + n_words] if n_words else docs
        rows = [[1] + self.encode_words([W.WORDS[t] for t in d[1:]]) for d in docs]
        length = length or max(map(len, rows))
        out = np.zeros((len(rows), length), dtype=np.int64)
        for i, r in enumerate(rows):
            out[i, :len(r)] = r[:length]
        return out, np.array([len(r) - 1 for r in rows])

    def encode_lists(self, word_lists, length=None):
        """Documents given as lists of word STRINGS (any strings, including words never seen) -> padded ids, and for
        each document the (start, end) token positions of every word."""
        rows, spans = [], []
        for words in word_lists:
            ids, sp = [1], []
            for w in words:
                piece = [self.id[x] for x in self.split(w)]
                sp.append((len(ids), len(ids) + len(piece))); ids += piece
            rows.append(ids); spans.append(sp)
        length = length or max(map(len, rows))
        out = np.zeros((len(rows), length), dtype=np.int64)
        for i, r in enumerate(rows):
            out[i, :len(r)] = r[:length]
        return out, spans

    def tokens_per_word(self, counts):
        return sum(len(self.split(w)) * c for w, c in counts.items()) / sum(counts.values())

    def used(self, counts):
        """Symbols that actually appear when the corpus is tokenised (many intermediate merges do not)."""
        return {s for w in counts for s in self.split(w)}


# ----------------------------------------------------------------------------- the training recipe
def fit(model, ids, steps=1500, batch=64, lr=2e-3, optimizer="adamw", weight_decay=0.0, warmup=0, schedule="constant",
        clip=1.0, seed=0, pad=None, log_every=0, log_fn=None, callback=None):
    """Next-token training with the choices of the recipe exposed.

    optimizer     "adamw": weight decay applied directly to the weights (Loshchilov & Hutter, 2017);
                  "adam": the same decay added to the gradient as an L2 penalty (the older way);  "sgd": plain SGD
    warmup        number of steps over which the learning rate rises linearly from zero
    schedule      "constant" or "cosine" (the rate falls to zero along half a cosine after the warm-up)
    clip          maximum gradient norm, or None
    pad           id to ignore in the loss (sub-word documents are padded)
    log_fn(model) is called every log_every steps; its results are returned alongside the training losses."""
    set_seed(seed)
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    def rate(s):
        if callable(schedule):                       # a custom schedule: function step -> multiplier of lr
            return schedule(s)
        if warmup and s < warmup:
            return (s + 1) / warmup
        if schedule == "cosine":
            return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    it = W.batches(ids, batch, seed)
    V = model.cfg["vocab"]
    losses, log, norms = [], [], []
    model.train()
    for step in range(steps):
        x = next(it)
        logits = model(x)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, V), x[:, 1:].reshape(-1), ignore_index=-100 if pad is None else pad)
        opt.zero_grad(); loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf"))))
        opt.step(); sched.step(); losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0 and log_fn is not None:
            log.append({"step": step + 1, **log_fn(model)}); model.train()
        if callback is not None and callback(step, model):
            break
    return {"loss": losses, "grad_norm": norms, "log": log, "lr": [lr * rate(s) for s in range(len(losses))]}


@torch.no_grad()
def per_word_loss(model, ids, n_tokens, n_words, chunk=250):
    """Average negative log-likelihood per WORD: the total over a document's sub-word tokens divided by its number of
    words. This is comparable across tokenisers; loss per token is not."""
    model.eval()
    x = torch.as_tensor(ids); total = 0.0
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        lp = F.log_softmax(model(xb), -1)
        nll = -lp.gather(-1, xb[..., None])[..., 0]
        mask = torch.arange(xb.shape[1])[None] <= torch.as_tensor(n_tokens[i:i + chunk])[:, None]
        mask[:, 0] = False
        total += float((nll * mask).sum())
    return total / (len(x) * n_words)


@torch.no_grad()
def word_losses(model, ids, spans, which):
    """Negative log-likelihood of ONE chosen word per document (summed over its sub-word tokens).
    which[i] is the index of the word in document i."""
    model.eval()
    x = torch.as_tensor(ids)
    nll = torch.cat([-F.log_softmax(model(x[i:i + 250]), -1).gather(-1, x[i:i + 250, :, None])[..., 0] for i in range(0, len(x), 250)]).numpy()
    return np.array([nll[i, spans[i][w][0]:spans[i][w][1]].sum() for i, w in enumerate(which)])


def floor_per_word(split="test", n_words=None):
    """The oracle's loss per word on the first n_words words of each document of a split."""
    d = W.corpus(split)
    lp = d["logp_bayes"][:, 1:1 + n_words] if n_words else d["logp_bayes"][:, 1:]
    return float(-lp.mean())
