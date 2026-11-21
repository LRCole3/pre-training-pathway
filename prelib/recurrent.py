"""Recurrent language models with optional attention over their own past states (notebook 02)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import world as W

V = W.VOCAB


class RecurrentLM(nn.Module):
    """Embedding -> RNN / LSTM / GRU -> next-token distribution.

    attention=None      : the prediction for token t+1 uses only the hidden state h_t.
    attention='dot'     : h_t also looks back over h_0..h_{t-1} with scaled dot-product scores (Luong et al., 2015).
    attention='additive': the same with a small network as the score (Bahdanau et al., 2014)."""

    def __init__(self, cell="lstm", d=128, attention=None, d_att=64):
        super().__init__()
        self.cell, self.attention, self.d_att = cell, attention, d_att
        self.emb = nn.Embedding(V, d)
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell](d, d, batch_first=True)
        if attention:
            self.q, self.k = nn.Linear(d, d_att, bias=False), nn.Linear(d, d_att, bias=False)
            if attention == "additive":
                self.v = nn.Linear(d_att, 1, bias=False)
        self.out = nn.Linear(2 * d if attention else d, V)

    def states(self, tokens):
        """Hidden state after reading each token: (B, L, d)."""
        return self.rnn(self.emb(tokens))[0]

    def attend(self, H):
        """Each position t attends over strictly earlier positions. Returns (context (B, L, d), weights (B, L, L))."""
        L = H.shape[1]
        if self.attention == "dot":
            s = self.q(H) @ self.k(H).transpose(1, 2) / math.sqrt(self.d_att)
        else:
            s = self.v(torch.tanh(self.q(H)[:, :, None] + self.k(H)[:, None]))[..., 0]
        earlier = torch.ones(L, L, dtype=torch.bool).tril(-1)
        w = torch.softmax(s.masked_fill(~earlier, -1e9), -1) * earlier        # row 0 has nothing to attend to
        return w @ H, w

    def forward(self, tokens, return_weights=False):
        H = self.states(tokens)
        if self.attention:
            ctx, w = self.attend(H)
            feat = torch.cat([H, ctx], -1)
        else:
            feat, w = H, None
        logits = self.out(feat)                                               # position t predicts token t+1
        logits = torch.cat([torch.zeros_like(logits[:, :1]), logits[:, :-1]], 1)
        return (logits, w) if return_weights else logits

    @torch.no_grad()
    def logprobs(self, tokens):
        self.eval()
        t = torch.as_tensor(tokens)
        return torch.cat([F.log_softmax(self(t[i:i + 250]), -1) for i in range(0, len(t), 250)]).numpy()

    @torch.no_grad()
    def features(self, tokens):
        """Document feature for probing: the average hidden state (contextual, unlike a bag of word vectors)."""
        self.eval()
        t = torch.as_tensor(tokens)
        return torch.cat([self.states(t[i:i + 500]).mean(1) for i in range(0, len(t), 500)]).numpy()


def gradient_reach(model, tokens):
    """How strongly the loss at the LAST position depends on the input at each earlier position.
    Returns mean gradient norm per distance back (index 0 = the token just before the target)."""
    model.eval()
    t = torch.as_tensor(tokens)
    x = model.emb(t).detach().requires_grad_(True)
    H = model.rnn(x)[0]
    feat = torch.cat([H, model.attend(H)[0]], -1) if model.attention else H
    loss = F.cross_entropy(model.out(feat)[:, -2], t[:, -1])
    g = torch.autograd.grad(loss, x)[0].norm(dim=-1).mean(0)[:-1]             # inputs 0..L-2
    return g.flip(0).numpy()


@torch.no_grad()
def lstm_trace(model, tokens):
    """Run the LSTM step by step and return its gates and cell state: dict of arrays (L, d) for ONE document."""
    assert model.cell == "lstm"
    r = model.rnn
    x = model.emb(torch.as_tensor(tokens)[None])[0]
    h = torch.zeros(r.hidden_size); c = torch.zeros(r.hidden_size)
    out = {k: [] for k in ["input gate", "forget gate", "output gate", "cell state", "hidden state"]}
    for xt in x:
        i, f, g, o = (xt @ r.weight_ih_l0.T + r.bias_ih_l0 + h @ r.weight_hh_l0.T + r.bias_hh_l0).chunk(4)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        for k, val in zip(out, [i, f, o, c, h]):
            out[k].append(val.numpy())
    return {k: np.stack(v) for k, v in out.items()}


class Classifier(nn.Module):
    """A recurrent encoder with a linear head on the average hidden state, for the downstream label."""

    def __init__(self, encoder, n_classes=W.N_STYLE):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.rnn.hidden_size, n_classes)

    def forward(self, tokens):
        return self.head(self.encoder.states(tokens).mean(1))


def train_classifier(model, tokens, labels, steps=300, batch=32, lr=2e-3, seed=0, lr_encoder=None):
    """Supervised training on a small labelled set. lr_encoder sets a separate (usually smaller) rate for the encoder."""
    torch.manual_seed(seed)
    x, y = torch.as_tensor(tokens), torch.as_tensor(labels)
    groups = [{"params": model.head.parameters(), "lr": lr},
              {"params": model.encoder.parameters(), "lr": lr if lr_encoder is None else lr_encoder}]
    opt = torch.optim.Adam(groups)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(x), (min(batch, len(x)),))
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def classifier_accuracy(model, tokens, labels):
    model.eval()
    return float((model(torch.as_tensor(tokens)).argmax(1).numpy() == labels).mean())
