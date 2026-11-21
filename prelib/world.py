"""A synthetic language with a known generator (notebooks 01-05 and 10).

Every document is 64 tokens. Three things are built in, each with a ground truth we can measure against:

1. LOCAL STRUCTURE. Filler words belong to 6 classes (think parts of speech; the class is visible in the word's
   suffix). The next class follows a transition matrix chosen by the document's hidden STYLE (3 styles), and the
   word within the class is drawn with preferences set by the document's hidden TOPIC (4 topics).
2. LONG-RANGE RECALL. A document can bind a key to a value ("set C 5") and ask for it later ("ask C" -> "5").
   The answer is deterministic but can only be read from the earlier binding, any number of tokens back.
3. A HIDDEN LABEL. The style changes word ORDER only (every style uses every class equally often), so a bag of
   words cannot recover it; a model that reads context can. It is the downstream task for probing.

Because we own the generator, the best achievable loss is computed, not guessed: `corpus(..., oracle=True)`
returns the log-probability of every token under the Bayes-optimal predictor, which infers topic and style from
the tokens seen so far. No model can beat that floor on average.
"""
import functools
import numpy as np
import torch

N_CLASS, N_WORD = 6, 10
N_KEY, N_VAL = 8, 8
N_TOPIC, N_STYLE = 4, 3
P_BIND, P_QUERY, MAX_BOUND = 0.08, 0.10, 3
DOC_LEN = 64

PAD, BOS, SET, ASK, MASK = 0, 1, 2, 3, 4
KEY0 = 5
VAL0 = KEY0 + N_KEY
FILL0 = VAL0 + N_VAL
VOCAB = FILL0 + N_CLASS * N_WORD            # 81 token ids

# token kinds, for analysis and plots
K_BOS, K_FILL, K_SET, K_BKEY, K_BVAL, K_ASK, K_QKEY, K_QVAL = range(8)
KIND_NAMES = ["start", "filler word", "'set'", "bound key", "bound value", "'ask'", "queried key", "recalled value"]

SUFFIX = ["a", "ek", "is", "om", "ur", "ly"]


def _build():
    rng = np.random.default_rng(0)
    cons, vow = "bdfgklmnprstvz", "aeiou"
    stems = set()
    while len(stems) < N_CLASS * N_WORD:
        stems.add(rng.choice(list(cons)) + rng.choice(list(vow)) + rng.choice(list(cons)))
    stems = sorted(stems); rng.shuffle(stems)
    words = ["<pad>", "<s>", "set", "ask", "<mask>"] + [chr(65 + i) for i in range(N_KEY)] + [str(i + 1) for i in range(N_VAL)]
    words += [stems[c * N_WORD + w] + SUFFIX[c] for c in range(N_CLASS) for w in range(N_WORD)]
    # styles: each style sends a class to its own 'next' class with prob 0.75, else uniform. Doubly stochastic,
    # so every style uses every class equally often in the long run.
    T = np.zeros((N_STYLE, N_CLASS, N_CLASS))
    nexts = [[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 1, 0], [5, 0, 1, 2, 3, 4]]      # three different 6-cycles over the classes
    for s, nxt in enumerate(nexts):
        for c in range(N_CLASS):
            T[s, c] = 0.25 / N_CLASS
            T[s, c, nxt[c]] += 0.75
    logits = rng.normal(size=(N_TOPIC, N_CLASS, N_WORD))
    E = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    return words, T, E


WORDS, T_STYLE, E_TOPIC = _build()
WORD_CLASS = np.full(VOCAB, -1)
WORD_CLASS[FILL0:] = np.repeat(np.arange(N_CLASS), N_WORD)


def to_text(tokens):
    return " ".join(WORDS[int(t)] for t in tokens if int(t) != PAD)


def generate(n, seed=0, length=DOC_LEN, oracle=False, force_distance=None):
    """Sample n documents. Returns a dict of arrays (tokens, topic, style, kind, ...).

    force_distance=d builds recall tests: one binding at the start and its query placed so that the recalled value
    is exactly d tokens after the bound value, with only filler words in between."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, N_TOPIC, n); s = rng.integers(0, N_STYLE, n)
    tok = np.zeros((n, length), dtype=np.int64); tok[:, 0] = BOS
    kind = np.zeros((n, length), dtype=np.int8)
    logp = np.zeros((n, length))                  # log p(token | history, true topic and style)
    log_stay = np.zeros((n, length))              # log P(filler rather than a statement) at filler steps
    pc_arr = np.zeros((n, length), dtype=np.int64); c_arr = np.zeros((n, length), dtype=np.int64)
    w_arr = np.zeros((n, length), dtype=np.int64)
    prev_class = np.zeros(n, dtype=np.int64)
    bound = np.full((n, N_KEY), -1); bound_pos = np.full((n, N_KEY), -1)
    mode = np.zeros(n, dtype=np.int64); pending = np.zeros(n, dtype=np.int64)
    dist = np.zeros((n, length), dtype=np.int64)  # for recalled values: tokens since the bound value
    rows = np.arange(n)
    q_start = None if force_distance is None else 3 + force_distance - 2
    for t in range(1, length):
        nb = (bound >= 0).sum(1)
        m0 = mode == 0
        if force_distance is None:
            pb = np.where(nb < MAX_BOUND, P_BIND, 0.0); pq = np.where(nb >= 1, P_QUERY, 0.0)
            u = rng.random(n)
            act = np.where(u < pb, 1, np.where(u < pb + pq, 2, 0))
        else:
            pb = np.zeros(n); pq = np.zeros(n)
            if t == 1: act = np.ones(n, dtype=int)
            elif t == q_start: act = np.full(n, 2)
            elif t < q_start + 3: act = np.zeros(n, dtype=int)
            else:
                pq = np.full(n, P_QUERY); u = rng.random(n); act = np.where(u < pq, 2, 0)
        # statements start
        i = m0 & (act == 1); tok[i, t] = SET; kind[i, t] = K_SET; logp[i, t] = np.log(np.maximum(pb[i], 1e-12)); mode[i] = 1
        j = m0 & (act == 2); tok[j, t] = ASK; kind[j, t] = K_ASK; logp[j, t] = np.log(np.maximum(pq[j], 1e-12)); mode[j] = 3
        # filler word
        f = m0 & (act == 0)
        if f.any():
            cdf = T_STYLE[s[f], prev_class[f]].cumsum(1)
            c = (rng.random(f.sum())[:, None] > cdf).sum(1).clip(max=N_CLASS - 1)
            wcdf = E_TOPIC[z[f], c].cumsum(1)
            w = (rng.random(f.sum())[:, None] > wcdf).sum(1).clip(max=N_WORD - 1)
            tok[f, t] = FILL0 + c * N_WORD + w; kind[f, t] = K_FILL
            log_stay[f, t] = np.log(1 - pb[f] - pq[f])
            logp[f, t] = log_stay[f, t] + np.log(T_STYLE[s[f], prev_class[f], c]) + np.log(E_TOPIC[z[f], c, w])
            pc_arr[f, t] = prev_class[f]; c_arr[f, t] = c; w_arr[f, t] = w
            prev_class[f] = c
        # inside statements (modes set on an earlier step)
        for mm, kk in [(1, K_BKEY), (3, K_QKEY)]:
            i = (mode == mm) & ~m0
            if i.any():
                avail = (bound[i] < 0) if mm == 1 else (bound[i] >= 0)
                pick = (rng.random(i.sum())[:, None] * avail.sum(1, keepdims=True) > avail.cumsum(1) - 1e-9).sum(1)
                pick = pick.clip(max=N_KEY - 1)
                tok[i, t] = KEY0 + pick; kind[i, t] = kk; logp[i, t] = -np.log(avail.sum(1)); pending[i] = pick
        i = (mode == 2) & ~m0
        if i.any():
            v = rng.integers(0, N_VAL, i.sum())
            tok[i, t] = VAL0 + v; kind[i, t] = K_BVAL; logp[i, t] = -np.log(N_VAL)
            bound[rows[i], pending[i]] = v; bound_pos[rows[i], pending[i]] = t
        i = (mode == 4) & ~m0
        if i.any():
            tok[i, t] = VAL0 + bound[rows[i], pending[i]]; kind[i, t] = K_QVAL
            dist[i, t] = t - bound_pos[rows[i], pending[i]]
        nxt = {1: 2, 2: 0, 3: 4, 4: 0}
        adv = ~m0
        mode[adv] = np.vectorize(nxt.get)(mode[adv]) if adv.any() else mode[adv]
    out = dict(tokens=tok, topic=z, style=s, kind=kind, logp_latent=logp, dist=dist)
    if oracle:
        fill = kind == K_FILL
        ll = np.log(E_TOPIC)[:, c_arr, w_arr][:, None] + np.log(T_STYLE)[:, pc_arr, c_arr][None]      # (K, S, n, L)
        ll = np.where(fill[None, None], ll, 0.0)
        cum = np.cumsum(ll, -1) - ll                                                                   # evidence before t
        post = np.exp(cum - cum.max((0, 1), keepdims=True)); post /= post.sum((0, 1), keepdims=True)
        mix = np.log((post * np.exp(ll)).sum((0, 1)))
        out["logp_bayes"] = np.where(fill, log_stay + mix, logp)
        final = cum[..., -1] + ll[..., -1]
        out["style_bayes"] = np.exp(final - final.max((0, 1), keepdims=True)).sum(0).argmax(0)
    return out


@functools.lru_cache(maxsize=None)
def corpus(split="train", n=None, length=DOC_LEN):
    """Standard splits. train: 20,000 docs; test: 2,000 docs with the oracle; probe: 4,000 docs for the label task."""
    seed, default_n, oracle = {"train": (0, 20000, False), "test": (1, 2000, True), "probe": (2, 4000, False)}[split]
    return generate(n or default_n, seed=seed, length=length, oracle=oracle)


def floor():
    """Best achievable loss (nats per token) on the test split: the Bayes-optimal predictor's cross-entropy."""
    return float(-corpus("test")["logp_bayes"][:, 1:].mean())


RECALL_DISTANCES = (3, 4, 6, 8, 12, 16, 24, 32, 48, 56)


@functools.lru_cache(maxsize=None)
def recall_set(distance, n=500, length=DOC_LEN):
    d = generate(n, seed=1000 + distance, length=max(length, distance + 8), force_distance=distance)
    return d["tokens"], 3 + distance                    # tokens, position of the value to recall


def evaluate_lm(logprob_fn, distances=RECALL_DISTANCES, name=None):
    """Score any language model. logprob_fn(tokens: int array (n, L)) -> array (n, L, VOCAB) where [:, t] is the
    model's log-distribution for token t given tokens before t (position 0 is ignored)."""
    test = corpus("test")
    tok = test["tokens"]
    lp = np.asarray(logprob_fn(tok))
    tl = np.take_along_axis(lp, tok[..., None], -1)[..., 0][:, 1:]
    kind = test["kind"][:, 1:]
    res = dict(loss=float(-tl.mean()), floor=floor())
    res["gap"] = res["loss"] - res["floor"]
    bay = -test["logp_bayes"][:, 1:]
    for k in (K_FILL, K_QVAL):
        if (kind == k).any():
            res[f"gap[{KIND_NAMES[k]}]"] = float(-tl[kind == k].mean() - bay[kind == k].mean())
    rec = {}
    for d in distances:
        t, pos = recall_set(d)
        rec[d] = float((np.asarray(logprob_fn(t[:, :pos + 1]))[:, pos].argmax(-1) == t[:, pos]).mean())
    res["recall"] = rec
    res["recall_mean"] = float(np.mean(list(rec.values())))
    if name:
        res["model"] = name
    return res


def probe(feature_fn, n_labels=200, label="style", seed=0):
    """Downstream task: fit logistic regression on features of n_labels labelled documents, test on 1,000 others.
    feature_fn(tokens (n, L)) -> array (n, D)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    d = corpus("probe")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(3000)[:n_labels]
    Xtr, ytr = np.asarray(feature_fn(d["tokens"][idx])), d[label][idx]
    Xte, yte = np.asarray(feature_fn(d["tokens"][3000:])), d[label][3000:]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def batches(tokens, batch=64, seed=0):
    """Endless shuffled mini-batches of documents as torch tensors."""
    g = torch.Generator().manual_seed(seed)
    data = torch.as_tensor(tokens)
    while True:
        perm = torch.randperm(len(data), generator=g)
        for i in range(0, len(data) - batch + 1, batch):
            yield data[perm[i:i + batch]]


def scorecard_row(name, res, **extra):
    """One row for results/scorecard.csv from an evaluate_lm result."""
    return dict(model=name, family="language", loss=res["loss"], gap=res["gap"], recall_mean=res["recall_mean"],
                recall_d8=res["recall"][8], recall_d32=res["recall"][32], **extra)
