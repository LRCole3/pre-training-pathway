"""Pretraining objectives beyond next-token prediction (notebook 04).

    corrupt(tokens, rate, ...)      choose positions to hide and how to hide them (BERT's 80/10/10 rule, span masking)
    train_mlm(model, tokens, ...)   masked-language-model training: predict the hidden tokens from both sides
    span_corrupt(tokens, ...)       T5-style: replace spans by sentinel tokens; the target is the missing spans
    Seq2Seq                         a small encoder-decoder transformer for that objective
    Electra / train_electra         replaced-token detection: a signal from every position
    probe_curve                     linear-probe accuracy against the number of labelled documents
    Classifier / fine_tune          a linear head on mean-pooled states, for transfer experiments
    sample_causal / sample_masked   generating documents from a causal model and from a masked model
    document_checks                 how well-formed a set of generated documents is
"""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import world as W, transformer as T, set_seed

V = W.VOCAB
N_SENTINEL = 12
V_SEQ = V + N_SENTINEL                       # vocabulary of the encoder-decoder: the usual tokens plus sentinels
SENT0 = V


# ----------------------------------------------------------------------------- masking
def corrupt(tokens, rate=0.15, rng=None, rule="80/10/10", span=False, mean_span=3.0):
    """Hide a fraction `rate` of the tokens (never the start token).

    Returns (corrupted tokens, hidden): `hidden` is a boolean array marking the positions the model must predict.
    rule="80/10/10" is BERT's: of the chosen positions 80% become <mask>, 10% a random token, 10% stay as they are.
    rule="mask" always writes <mask>. span=True hides contiguous runs (SpanBERT) with geometric lengths."""
    rng = rng or np.random.default_rng()
    x = np.array(tokens); n, L = x.shape
    if span:
        hidden = np.zeros((n, L), dtype=bool)
        budget = max(1, int(round(rate * (L - 1))))
        for i in range(n):
            while hidden[i].sum() < budget:
                length = min(int(rng.geometric(1.0 / mean_span)), 10, budget - int(hidden[i].sum()))
                start = int(rng.integers(1, L - length + 1))
                hidden[i, start:start + length] = True
    else:
        hidden = rng.random((n, L)) < rate
        hidden[:, 0] = False
    out = x.copy()
    u = rng.random((n, L))
    if rule == "80/10/10":
        out[hidden & (u < 0.8)] = W.MASK
        swap = hidden & (u >= 0.8) & (u < 0.9)
        out[swap] = rng.integers(W.KEY0, V, swap.sum())
    else:
        out[hidden] = W.MASK
    return out, hidden


def train_mlm(model, tokens, steps=1500, batch=64, lr=2e-3, seed=0, rate=0.15, rule="80/10/10", span=False, static=False,
              clip=1.0, callback=None):
    """Masked-language-model training. static=True fixes one corruption per document for the whole run (original BERT);
    otherwise a new one is drawn every time a document is seen (RoBERTa's dynamic masking)."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    data = np.asarray(tokens)
    if static:
        fixed_x, fixed_h = corrupt(data, rate, np.random.default_rng(seed + 1), rule, span)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    model.train()
    perm, at = rng.permutation(len(data)), 0
    for step in range(steps):
        if at + batch > len(data):
            perm, at = rng.permutation(len(data)), 0
        idx = perm[at:at + batch]; at += batch
        if static:
            x, hidden = fixed_x[idx], fixed_h[idx]
        else:
            x, hidden = corrupt(data[idx], rate, rng, rule, span)
        logits = model.token_logits(torch.as_tensor(x))
        h = torch.as_tensor(hidden)
        loss = F.cross_entropy(logits[h], torch.as_tensor(data[idx])[h])
        opt.zero_grad(); loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); losses.append(loss.item())
        if callback is not None and callback(step, model):
            break
    return losses


@torch.no_grad()
def cloze(model, tokens, hidden, chunk=500):
    """Fill-in-the-blank test: write <mask> at the `hidden` positions and return the log-probabilities the model gives
    at every position, array (n, L, vocabulary). Works for any model with token_logits (a bidirectional encoder)."""
    model.eval()
    x = np.array(tokens); x[hidden] = W.MASK
    x = torch.as_tensor(x)
    return torch.cat([F.log_softmax(model.token_logits(x[i:i + chunk])[..., :V], -1) for i in range(0, len(x), chunk)]).numpy()


def cloze_table(logp, tokens, hidden, kind):
    """Accuracy and loss on the hidden tokens, overall and by kind of token."""
    import pandas as pd
    tok = np.asarray(tokens)
    correct = logp.argmax(-1) == tok
    nll = -np.take_along_axis(logp, tok[..., None], -1)[..., 0]
    rows = {"all hidden tokens": hidden}
    for k in (W.K_FILL, W.K_BKEY, W.K_BVAL, W.K_QKEY, W.K_QVAL):
        rows[W.KIND_NAMES[k]] = hidden & (kind == k)
    return pd.DataFrame({name: {"accuracy": correct[m].mean(), "loss": nll[m].mean(), "tokens": int(m.sum())} for name, m in rows.items()}).T


# ----------------------------------------------------------------------------- span corruption and the encoder-decoder
def span_corrupt(tokens, rate=0.15, mean_span=3.0, rng=None, target_len=24):
    """T5-style span corruption for a batch. Each hidden span is replaced by ONE sentinel token in the input; the
    target lists, for each sentinel in order, the sentinel followed by the tokens it replaced.
    Returns (inputs padded to the document length, decoder inputs, decoder targets); padding is W.PAD."""
    rng = rng or np.random.default_rng()
    x = np.asarray(tokens); n, L = x.shape
    inp = np.full((n, L), W.PAD); dec_in = np.full((n, target_len), W.PAD); dec_out = np.full((n, target_len), W.PAD)
    budget = max(1, int(round(rate * (L - 1))))
    for i in range(n):
        hidden = np.zeros(L, dtype=bool); starts = []
        while hidden.sum() < budget and len(starts) < N_SENTINEL:
            length = min(int(rng.geometric(1.0 / mean_span)), 10, budget - int(hidden.sum()))
            start = int(rng.integers(1, L - length + 1))
            if hidden[max(start - 1, 0):start + length + 1].any():       # keep spans apart so each gets its own sentinel
                continue
            hidden[start:start + length] = True; starts.append((start, length))
        starts.sort()
        src, tgt, t = [], [], 0
        for s, (start, length) in enumerate(starts):
            src += list(x[i, t:start]) + [SENT0 + s]; t = start + length
            tgt += [SENT0 + s] + list(x[i, start:start + length])
        src += list(x[i, t:])
        tgt = tgt[:target_len - 1]
        inp[i, :len(src)] = src
        dec_in[i, :len(tgt) + 1] = [W.BOS] + tgt
        dec_out[i, :len(tgt)] = tgt
        dec_out[i, len(tgt)] = W.BOS                                      # the start token doubles as "end of target"
    return inp, dec_in, dec_out


class CrossAttention(nn.Module):
    """Attention from decoder positions (queries) to the encoder's output (keys and values)."""

    def __init__(self, d, n_head):
        super().__init__()
        self.H, self.hd = n_head, d // n_head
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x, memory, memory_pad):
        B, T_, _ = x.shape; S = memory.shape[1]
        q = self.q(x).view(B, T_, self.H, self.hd).transpose(1, 2)
        k = self.k(memory).view(B, S, self.H, self.hd).transpose(1, 2)
        v = self.v(memory).view(B, S, self.H, self.hd).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.hd)
        w = torch.softmax(s.masked_fill(memory_pad[:, None, None, :], float("-inf")), -1)
        self.last_weights = w.detach()
        return self.o((w @ v).transpose(1, 2).reshape(B, T_, -1))


class DecoderBlock(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n1, self.n2, self.n3 = T.RMSNorm(d), T.RMSNorm(d), T.RMSNorm(d)
        self.self_attn = T.Attention(d, n_head, pos="relative", causal=True)
        self.cross = CrossAttention(d, n_head)
        self.ffn = T.FeedForward(d, 4 * d, "swiglu")

    def forward(self, x, memory, memory_pad):
        x = x + self.self_attn(self.n1(x))
        x = x + self.cross(self.n2(x), memory, memory_pad)
        return x + self.ffn(self.n3(x))


class Seq2Seq(nn.Module):
    """Encoder-decoder transformer (the T5 / BART layout): a bidirectional encoder reads the corrupted document,
    a causal decoder writes the missing pieces while reading the encoder's output through cross-attention."""

    def __init__(self, d=64, n_enc=2, n_dec=2, n_head=4, pos="relative"):
        super().__init__()
        self.encoder = T.Transformer(d=d, n_layer=n_enc, n_head=n_head, pos=pos, norm="pre", norm_type="rms", ffn="swiglu",
                                     causal=False, vocab=V_SEQ)
        self.emb = self.encoder.emb                                       # decoder shares the token embedding
        self.blocks = nn.ModuleList(DecoderBlock(d, n_head) for _ in range(n_dec))
        self.final = T.RMSNorm(d)
        self.out = nn.Linear(d, V_SEQ)

    def forward(self, inp, dec_in):
        memory = self.encoder.hidden(inp)
        pad = inp == W.PAD
        x = self.emb(dec_in)
        for blk in self.blocks:
            x = blk(x, memory, pad)
        return self.out(self.final(x))

    @torch.no_grad()
    def features(self, tokens):
        """Mean-pooled ENCODER states of uncorrupted documents: the part that is reused downstream."""
        self.eval()
        t = torch.as_tensor(tokens)
        return torch.cat([self.encoder.hidden(t[i:i + 500]).mean(1) for i in range(0, len(t), 500)]).numpy()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_span(model, tokens, steps=1500, batch=64, lr=2e-3, seed=0, rate=0.15, mean_span=3.0, clip=1.0, callback=None):
    """Span-corruption training of a Seq2Seq model."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    data = np.asarray(tokens)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(data), batch)
        inp, dec_in, dec_out = (torch.as_tensor(a) for a in span_corrupt(data[idx], rate, mean_span, rng))
        logits = model(inp, dec_in)
        loss = F.cross_entropy(logits.reshape(-1, V_SEQ), dec_out.reshape(-1), ignore_index=W.PAD)
        opt.zero_grad(); loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); losses.append(loss.item())
        if callback is not None and callback(step, model):
            break
    return losses


# ----------------------------------------------------------------------------- replaced-token detection (ELECTRA)
class Electra(nn.Module):
    """ELECTRA (Clark et al., 2020): a small masked model (the generator) fills the hidden positions with plausible
    tokens; the main model (the discriminator) reads the result and says, for EVERY position, whether the token is
    the original or a replacement. Only the discriminator's body (`encoder`) is kept for downstream use."""

    def __init__(self, d=64, d_gen=32, **kw):
        super().__init__()
        self.generator = T.Transformer(d=d_gen, n_layer=1, causal=False, **kw)
        self.encoder = T.Transformer(d=d, causal=False, **kw)
        self.head = nn.Linear(d, 1)
        self.cfg = self.encoder.cfg

    def hidden(self, tokens):
        return self.encoder.hidden(tokens)

    def features(self, tokens):
        return self.encoder.features(tokens)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_electra(model, tokens, steps=1500, batch=64, lr=2e-3, seed=0, rate=0.15, weight=50.0, clip=1.0, callback=None):
    """Joint training: generator on the masked-token loss, discriminator on original-or-replaced for all positions."""
    set_seed(seed)
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    data = np.asarray(tokens)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    log = []
    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(data), batch)
        original = torch.as_tensor(data[idx])
        masked, hidden = corrupt(data[idx], rate, rng, rule="mask")
        h = torch.as_tensor(hidden)
        gen_logits = model.generator.token_logits(torch.as_tensor(masked))[h]
        gen_loss = F.cross_entropy(gen_logits, original[h])
        with torch.no_grad():
            p = torch.softmax(gen_logits[:, :V], -1); p[:, [W.PAD, W.BOS, W.MASK]] = 0
            fill = torch.multinomial(p, 1, generator=g)[:, 0]
        replaced = original.clone(); replaced[h] = fill
        target = (replaced != original).float()                       # a correct guess by the generator counts as original
        logit = model.head(model.encoder.hidden(replaced))[..., 0]
        disc_loss = F.binary_cross_entropy_with_logits(logit[:, 1:], target[:, 1:])
        loss = gen_loss + weight * disc_loss
        opt.zero_grad(); loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); log.append((gen_loss.item(), disc_loss.item()))
        if callback is not None and callback(step, model):
            break
    return log


# ----------------------------------------------------------------------------- transfer
@torch.no_grad()
def states(model, tokens, chunk=500):
    """Per-token final states (n, L, d) of a Transformer or of a Seq2Seq's encoder."""
    enc = model.encoder if isinstance(model, (Seq2Seq, Electra)) else model
    enc.eval()
    t = torch.as_tensor(tokens)
    return torch.cat([enc.hidden(t[i:i + chunk]) for i in range(0, len(t), chunk)]).numpy()


def probe_curve(features, label="style", label_counts=(15, 30, 100, 300), draws=3):
    """Linear-probe accuracy for several numbers of labelled documents, averaged over `draws` random choices of which
    documents are labelled. `features` is an array (4000, D) for the probe corpus or a function tokens -> features.
    Returns {number of labels: accuracy on the 1,000 held-out probe documents}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    d = W.corpus("probe")
    X = np.asarray(features(d["tokens"])) if callable(features) else np.asarray(features)
    y = d[label]; out = {}
    for n in label_counts:
        acc = []
        for seed in range(draws):
            idx = np.random.default_rng(seed).permutation(3000)[:n]
            if len(set(y[idx])) < len(set(y)):                         # a draw that misses a class entirely: skip it
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[idx], y[idx])
            acc.append(clf.score(X[3000:], y[3000:]))
        out[n] = float(np.mean(acc))
    return out


class Classifier(nn.Module):
    """A pretrained encoder with a linear head on its mean-pooled states."""

    def __init__(self, model, n_classes=W.N_STYLE):
        super().__init__()
        self.encoder = model.encoder if isinstance(model, (Seq2Seq, Electra)) else model
        self.head = nn.Linear(self.encoder.cfg["d"], n_classes)

    def forward(self, tokens):
        return self.head(self.encoder.hidden(tokens).mean(1))


def fine_tune(model, tokens, labels, steps=300, batch=32, lr=2e-3, lr_encoder=2e-4, seed=0):
    """Train a Classifier on a small labelled set; the pretrained part gets its own (smaller) learning rate."""
    torch.manual_seed(seed)
    x, y = torch.as_tensor(tokens), torch.as_tensor(labels)
    opt = torch.optim.Adam([{"params": model.head.parameters(), "lr": lr}, {"params": model.encoder.parameters(), "lr": lr_encoder}])
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(x), (min(batch, len(x)),))
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def accuracy(model, tokens, labels):
    model.eval()
    t = torch.as_tensor(tokens)
    pred = torch.cat([model(t[i:i + 500]).argmax(1) for i in range(0, len(t), 500)]).numpy()
    return float((pred == np.asarray(labels)).mean())


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def sample_causal(model, n=200, length=W.DOC_LEN, seed=0, temperature=1.0):
    """Generate documents left to right from a causal language model."""
    model.eval(); g = torch.Generator().manual_seed(seed)
    x = torch.full((n, 1), W.BOS)
    for _ in range(length - 1):
        logits = model.out(model.hidden(x)[:, -1]) / temperature
        logits[:, [W.PAD, W.BOS, W.MASK]] = float("-inf")
        x = torch.cat([x, torch.multinomial(torch.softmax(logits, -1), 1, generator=g)], 1)
    return x.numpy()


@torch.no_grad()
def sample_masked(model, n=200, length=W.DOC_LEN, steps=16, seed=0, order="confidence", record=False):
    """Generate documents from a masked model: start from all <mask>, and in each of `steps` rounds fill in some
    positions with samples from the model's predictions, keeping them fixed afterwards.
    order="confidence" fills the positions the model is surest about first; "left-to-right" fills from the left."""
    model.eval(); g = torch.Generator().manual_seed(seed)
    x = torch.full((n, length), W.MASK); x[:, 0] = W.BOS
    todo = length - 1; history = [x.clone()]
    for s in range(steps):
        k = int(math.ceil(todo * (s + 1) / steps)) - int(math.ceil(todo * s / steps))     # how many to fill this round
        if k == 0:
            continue
        logits = model.token_logits(x)[..., :V]
        logits[..., [W.PAD, W.BOS, W.MASK]] = float("-inf")
        p = torch.softmax(logits, -1)
        draw = torch.multinomial(p.reshape(-1, V), 1, generator=g).reshape(n, length)
        conf = p.gather(-1, draw[..., None])[..., 0]
        open_ = x == W.MASK
        if order == "left-to-right":
            conf = -torch.arange(length).float().expand(n, length).clone()
        conf[~open_] = float("-inf")
        pick = conf.topk(k, -1).indices
        x.scatter_(1, pick, draw.gather(1, pick))
        history.append(x.clone())
    return (x.numpy(), [h.numpy() for h in history]) if record else x.numpy()


def document_checks(docs):
    """How well-formed are these documents? Returns a dict of rates between 0 and 1:
    grammar       'set' and 'ask' are followed by a key and then a value (and keys/values appear nowhere else)
    asks bound    a queried key was bound earlier in the document
    recall        the value after 'ask K' is the one bound to K (among queries of bound keys)
    word order    share of word-to-word steps that follow the favourite class order of the document's best-fitting style"""
    docs = np.asarray(docs)
    fav = W.T_STYLE.argmax(-1)                                        # (style, previous class) -> favourite next class
    is_key = lambda t: W.KEY0 <= t < W.KEY0 + W.N_KEY
    is_val = lambda t: W.VAL0 <= t < W.VAL0 + W.N_VAL
    gram_ok = gram_n = bound_ok = bound_n = rec_ok = rec_n = 0
    order = []
    for d in docs:
        bound = {}; L = len(d); t = 1
        cls = []
        while t < L:
            tok = d[t]
            if tok in (W.SET, W.ASK):
                if t + 2 >= L:
                    break                                                # a statement cut off by the end of the document
                if True:
                    gram_n += 1
                    ok = is_key(d[t + 1]) and is_val(d[t + 2])
                    gram_ok += ok
                    if ok and tok == W.SET:
                        bound[d[t + 1]] = d[t + 2]
                    if ok and tok == W.ASK:
                        bound_n += 1; bound_ok += d[t + 1] in bound
                        if d[t + 1] in bound:
                            rec_n += 1; rec_ok += bound[d[t + 1]] == d[t + 2]
                    t += 3 if ok else 1
                    continue
            elif is_key(tok) or is_val(tok):
                gram_n += 1                                              # a key or value outside a statement
            elif W.WORD_CLASS[tok] >= 0:
                cls.append(W.WORD_CLASS[tok])
            t += 1
        cls = np.array(cls)
        if len(cls) > 5:
            order.append(max(float((fav[s][cls[:-1]] == cls[1:]).mean()) for s in range(W.N_STYLE)))
    return {"grammar": gram_ok / max(gram_n, 1), "asks bound": bound_ok / max(bound_n, 1), "recall": rec_ok / max(rec_n, 1),
            "word order": float(np.mean(order)) if order else float("nan")}
