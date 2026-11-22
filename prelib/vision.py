"""Pretrained parts of a modern image generator, at toy scale (notebook 08).

    ViT(d, depth, patch)                 vision transformer on 8x8 digits: 16 patches of 2x2 pixels
    train_mae(vit, X, mask_ratio)        masked autoencoding (He et al., 2021)
    train_contrastive(vit, X, ...)       SimCLR (Chen et al., 2020): two augmented views of an image should agree
    probe(features_fn, ...)              logistic regression on frozen features with few labels
    captions / TextEncoder / train_clip  a CLIP-style model (Radford et al., 2021) on captioned digits, with the
                                         softmax loss of CLIP or the sigmoid loss of SigLIP (Zhai et al., 2023)
"""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import set_seed, images


# ----------------------------------------------------------------------------- patches and the vision transformer
def patchify(X, patch=2):
    """(n, 64) images -> (n, 64 / patch^2, patch^2): the patches in reading order."""
    n, g = len(X), 8 // patch
    return X.view(n, g, patch, g, patch).permute(0, 1, 3, 2, 4).reshape(n, g * g, patch * patch)


def unpatchify(P, patch=2):
    n, g = len(P), 8 // patch
    return P.view(n, g, g, patch, patch).permute(0, 1, 3, 2, 4).reshape(n, 64)


class Block(nn.Module):
    def __init__(self, d, heads=4, mult=2):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    """Patches -> linear embedding + learned positions -> transformer blocks (Dosovitskiy et al., 2020)."""

    def __init__(self, d=64, depth=3, heads=4, patch=2):
        super().__init__()
        self.patch, self.n_tok, self.d = patch, (8 // patch) ** 2, d
        self.embed = nn.Linear(patch * patch, d)
        self.pos = nn.Parameter(torch.randn(self.n_tok, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)

    def tokens(self, X, keep=None):
        """Encode the patches listed in `keep` (n, k indices); all patches if keep is None. Returns (n, k, d)."""
        h = self.embed(patchify(X, self.patch)) + self.pos
        if keep is not None:
            h = torch.gather(h, 1, keep[..., None].expand(-1, -1, self.d))
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def forward(self, X):
        """One vector per image: the mean over its patch tokens."""
        return self.tokens(X).mean(1)

    @torch.no_grad()
    def features(self, X):
        self.eval()
        return self(X)


# ----------------------------------------------------------------------------- masked autoencoder
class MAE(nn.Module):
    """The encoder sees only the visible patches. A small decoder receives the encoded patches plus a learned
    'mask token' at every hidden position and predicts the pixels of the hidden patches."""

    def __init__(self, vit, d_dec=32, depth_dec=1):
        super().__init__()
        self.vit = vit
        self.to_dec = nn.Linear(vit.d, d_dec)
        self.mask_token = nn.Parameter(torch.zeros(d_dec))
        self.pos_dec = nn.Parameter(torch.randn(vit.n_tok, d_dec) * 0.02)
        self.dec = nn.ModuleList(Block(d_dec, 4) for _ in range(depth_dec))
        self.out = nn.Sequential(nn.LayerNorm(d_dec), nn.Linear(d_dec, vit.patch ** 2))

    def split(self, n, mask_ratio, generator=None):
        """Random choice of visible patches for each image: (indices kept, mask with True = hidden)."""
        n_tok = self.vit.n_tok
        k = max(1, int(round(n_tok * (1 - mask_ratio))))
        order = torch.rand(n, n_tok, generator=generator).argsort(1)
        keep = order[:, :k]
        hidden = torch.ones(n, n_tok, dtype=torch.bool).scatter_(1, keep, False)
        return keep, hidden

    def forward(self, X, keep):
        n, n_tok = len(X), self.vit.n_tok
        enc = self.to_dec(self.vit.tokens(X, keep))
        full = self.mask_token.expand(n, n_tok, -1).clone()
        full = full.scatter(1, keep[..., None].expand(-1, -1, enc.shape[-1]), enc) + self.pos_dec
        for blk in self.dec:
            full = blk(full)
        return self.out(full)                                             # (n, n_tok, patch^2): predicted pixels of every patch

    @torch.no_grad()
    def reconstruct(self, X, mask_ratio, seed=0):
        """Returns (what the encoder saw, the reconstruction with visible patches pasted back, hidden mask)."""
        self.eval()
        keep, hidden = self.split(len(X), mask_ratio, torch.Generator().manual_seed(seed))
        P, pred = patchify(X, self.vit.patch), self(X, keep)
        seen = torch.where(hidden[..., None], torch.zeros_like(P), P)
        return unpatchify(seen, self.vit.patch), unpatchify(torch.where(hidden[..., None], pred, P), self.vit.patch), hidden


def train_mae(mae, X, mask_ratio=0.75, steps=1500, batch=128, lr=2e-3, seed=0):
    """Squared error on the hidden patches only."""
    set_seed(seed)
    opt = torch.optim.AdamW(mae.parameters(), lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    losses, t0 = [], time.time()
    for step in range(steps):
        mae.train()
        x = X[torch.randint(0, len(X), (batch,))]
        keep, hidden = mae.split(batch, mask_ratio)
        err = ((mae(x, keep) - patchify(x, mae.vit.patch)) ** 2).mean(-1)
        loss = (err * hidden).sum() / hidden.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    return dict(losses=np.array(losses), seconds=time.time() - t0)


# ----------------------------------------------------------------------------- contrastive learning
def augment(X, shift=1, noise=0.2, ink=0.3, erase=0, generator=None):
    """A random 'view' of each image: shifted by up to `shift` pixels, ink scaled by 1 +- `ink`, Gaussian noise
    added, and (erase > 0) a random erase x erase square blanked."""
    g, n = generator, len(X)
    img = X.view(n, 8, 8)
    if shift:
        pad = F.pad(img, (shift, shift, shift, shift), value=-1.0)
        dx = torch.randint(0, 2 * shift + 1, (n,), generator=g); dy = torch.randint(0, 2 * shift + 1, (n,), generator=g)
        idx = torch.arange(8)
        img = pad[torch.arange(n)[:, None, None], (dy[:, None] + idx)[:, :, None], (dx[:, None] + idx)[:, None, :]]
    if ink:
        scale = 1 + ink * (2 * torch.rand(n, 1, 1, generator=g) - 1)
        img = (img + 1) * scale - 1                                        # background stays at -1
    if erase:
        ex = torch.randint(0, 9 - erase, (n,), generator=g); ey = torch.randint(0, 9 - erase, (n,), generator=g)
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        box = (yy[None] >= ey[:, None, None]) & (yy[None] < ey[:, None, None] + erase) & (xx[None] >= ex[:, None, None]) & (xx[None] < ex[:, None, None] + erase)
        img = torch.where(box, torch.full_like(img, -1.0), img)
    out = img.reshape(n, 64)
    if noise:
        out = out + noise * torch.randn(out.shape, generator=g)
    return out.clamp(-1.5, 1.5)


def info_nce(a, b, temperature=0.2):
    """Each row of a should be most similar to the same row of b, among all rows of b (and the reverse)."""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    target = torch.arange(len(a))
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def train_contrastive(vit, X, steps=1500, batch=128, lr=2e-3, temperature=0.2, head=True, seed=0, **aug):
    """SimCLR: two augmented views of each image; the encoder (plus a projection head that is thrown away
    afterwards) is trained so that the two views of one image are closer than views of different images."""
    set_seed(seed)
    proj = nn.Sequential(nn.Linear(vit.d, vit.d), nn.ReLU(), nn.Linear(vit.d, 32)) if head else nn.Identity()
    params = list(vit.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    losses, t0 = [], time.time()
    for step in range(steps):
        vit.train()
        x = X[torch.randint(0, len(X), (batch,))]
        loss = info_nce(proj(vit(augment(x, **aug))), proj(vit(augment(x, **aug))), temperature)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    return dict(losses=np.array(losses), seconds=time.time() - t0)


def train_supervised(vit, X, y, steps=300, batch=64, lr=2e-3, seed=0):
    """The same encoder trained from scratch on labels only (the baseline that pretraining has to beat)."""
    set_seed(seed)
    head = nn.Linear(vit.d, 10)
    opt = torch.optim.AdamW(list(vit.parameters()) + list(head.parameters()), lr, weight_decay=0.05)
    for step in range(steps):
        vit.train()
        idx = torch.randint(0, len(X), (min(batch, len(X)),))
        loss = F.cross_entropy(head(vit(augment(X[idx], noise=0.1))), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    vit.eval()
    return lambda Z: head(vit(Z)).argmax(1)


def few_labels(y, per_class, seed=0):
    """Indices of `per_class` labelled examples of each digit."""
    g = torch.Generator().manual_seed(seed)
    return torch.cat([torch.nonzero(y == c)[:, 0][torch.randperm(int((y == c).sum()), generator=g)[:per_class]] for c in range(10)])


def probe(features, Xtr, ytr, Xte, yte, per_class=5, draws=5):
    """Linear probe: logistic regression on frozen features, trained on `per_class` labels per digit.
    Returns the mean held-out accuracy over `draws` random choices of the labelled images."""
    from sklearn.linear_model import LogisticRegression
    ftr, fte = features(Xtr).numpy(), features(Xte).numpy()
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    accs = []
    for seed in range(draws):
        idx = few_labels(ytr, per_class, seed).numpy() if per_class else np.arange(len(ytr))
        clf = LogisticRegression(C=1.0, max_iter=2000).fit((ftr[idx] - mu) / sd, ytr.numpy()[idx])
        accs.append(clf.score((fte - mu) / sd, yte.numpy()))
        if not per_class:
            break
    return float(np.mean(accs))


# ----------------------------------------------------------------------------- captioned digits and a CLIP-style model
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
STYLE_WORDS = {0: ["thin", "fine", "light"], 1: ["bold", "thick", "heavy"]}
FILLER = ["a", "the", "digit", "drawn", "picture", "of", "in", "strokes", "number"]
VOCAB = ["<pad>"] + DIGIT_WORDS + [str(i) for i in range(10)] + STYLE_WORDS[0] + STYLE_WORDS[1] + FILLER
WORD = {w: i for i, w in enumerate(VOCAB)}
CAP_LEN = 7
TEMPLATES = ["a {s} {d}", "the digit {d} drawn in {s} strokes", "a picture of a {s} {d}", "{s} number {d}", "{d} in {s} strokes", "a {d}", "the number {d}"]


def encode_caption(text):
    ids = [WORD[w] for w in text.split()][:CAP_LEN]
    return torch.tensor(ids + [0] * (CAP_LEN - len(ids)))


def caption(digit, style, rng):
    """A random caption for a digit (0-9) in a style (0 thin, 1 bold). The digit is written as a word or a numeral,
    the style by one of three synonyms, in one of seven templates; two templates do not mention the style."""
    d = DIGIT_WORDS[digit] if rng.random() < 0.7 else str(digit)
    return TEMPLATES[rng.integers(len(TEMPLATES))].format(d=d, s=STYLE_WORDS[style][rng.integers(3)])


def thicken(X):
    """The 'bold' style: every stroke is widened by one pixel to the right."""
    img = X.view(-1, 1, 8, 8)
    return torch.maximum(img, F.pad(img, (1, 0, 0, 0), value=-1.0)[..., :8]).reshape(-1, 64)


def captioned_digits(X, y, seed=0, holdout=()):
    """Every image in two styles: as it is ('thin') and with thickened strokes ('bold').
    Returns images (2n, 64), digit labels, style labels. Pairs (digit, style) in `holdout` are left out."""
    Xs = torch.cat([X, thicken(X)])
    ds, ss = torch.cat([y, y]), torch.cat([torch.zeros(len(y), dtype=torch.long), torch.ones(len(y), dtype=torch.long)])
    keep = torch.ones(len(Xs), dtype=torch.bool)
    for d, s in holdout:
        keep &= ~((ds == d) & (ss == s))
    return Xs[keep], ds[keep], ss[keep]


class TextEncoder(nn.Module):
    """Caption tokens -> one vector: embeddings + positions, one transformer block, mean over the real tokens."""

    def __init__(self, d=64, out=32):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(CAP_LEN, d) * 0.02)
        self.block = Block(d, 4)
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, out)

    def forward(self, tokens):
        real = (tokens != 0).float()[..., None]
        h = self.block(self.emb(tokens) + self.pos)
        return self.proj(self.norm((h * real).sum(1) / real.sum(1).clamp_min(1)))


class CLIP(nn.Module):
    def __init__(self, d=64, out=32, loss="softmax"):
        super().__init__()
        self.image, self.text, self.loss_kind = ViT(d=d, depth=3), TextEncoder(d, out), loss
        self.img_proj = nn.Linear(d, out)
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))                 # learned inverse temperature
        self.bias = nn.Parameter(torch.tensor(-10.0 if loss == "sigmoid" else 0.0))

    def embed_image(self, X):
        return F.normalize(self.img_proj(self.image(X)), dim=-1)

    def embed_text(self, tokens):
        return F.normalize(self.text(tokens), dim=-1)

    def loss(self, X, tokens):
        logits = self.embed_image(X) @ self.embed_text(tokens).T * self.scale.exp()
        if self.loss_kind == "softmax":                                         # CLIP: pick the right caption among the batch, and the right image
            target = torch.arange(len(X))
            return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        labels = 2 * torch.eye(len(X)) - 1                                      # SigLIP: every image-caption pair is its own yes / no question
        return -F.logsigmoid(labels * (logits + self.bias)).sum() / len(X)

    @torch.no_grad()
    def similarity(self, X, texts):
        self.eval()
        return self.embed_image(X) @ self.embed_text(torch.stack([encode_caption(t) for t in texts])).T


def train_clip(model, X, digits, styles, steps=1500, batch=128, lr=2e-3, seed=0):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    losses, t0 = [], time.time()
    for step in range(steps):
        model.train()
        idx = torch.randint(0, len(X), (batch,))
        toks = torch.stack([encode_caption(caption(int(digits[i]), int(styles[i]), rng)) for i in idx])
        loss = model.loss(augment(X[idx], noise=0.1), toks)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    model.eval()
    return dict(losses=np.array(losses), seconds=time.time() - t0)


@torch.no_grad()
def zero_shot(model, X, digits, styles=None, prompt="a {d}"):
    """Classify images by the most similar prompt. With styles given, the prompts name digit and style (20 classes)."""
    if styles is None:
        sim = model.similarity(X, [prompt.format(d=w) for w in DIGIT_WORDS])
        return float((sim.argmax(1) == digits).float().mean())
    texts = [f"a {STYLE_WORDS[s][0]} {w}" for s in (0, 1) for w in DIGIT_WORDS]
    pred = model.similarity(X, texts).argmax(1)
    return float(((pred % 10 == digits) & (pred // 10 == styles)).float().mean())
