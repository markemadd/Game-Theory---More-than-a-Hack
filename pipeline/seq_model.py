"""Sequence expert: a bidirectional GRU over the same features at 5 Hz.

The tree model sees one frame plus fixed rolling windows; the GRU reads whole stretches, so it
can learn "how long has this team had it" and "a touch makes a change of hands more likely".
Same training data, degradation, perspective swap and leave-one-match-out split as cv.py, so its
out-of-fold probabilities can be ensembled with the trees' and judged on the same matches.

    python -m pipeline.seq_model --trial 3           # out-of-fold: cache/oof_seq_t3_*.npz
    python -m pipeline.seq_model --trial 3 --final   # all matches -> cache/bar_prob_seq_t3.npz
"""

import os
import sys
import time

import numpy as np
import torch
from torch import nn

from . import config, degrade, features, io_atd, io_idsse, model

STEP = 5
WIN = 300          # 1 minute at 5 Hz per training window
BATCH = 16
EPOCH_STEPS = 300
EPOCHS = 4
SEED = 2027


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def to_steps(X, y=None):
    """Block means of features (NaN-aware) and block-majority labels at 5 Hz."""
    n = len(X) // STEP * STEP
    arr = X[:n].reshape(-1, STEP, X.shape[1])
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        xs = np.nanmean(arr, 1)
    if y is None:
        return xs
    ys = y[:n].reshape(-1, STEP)
    ys = np.array([np.bincount(r[r >= 0], minlength=3).argmax() if (r >= 0).any() else -1
                   for r in ys])
    return xs, ys


def sequences(match, profile, seed):
    rng = np.random.default_rng(seed)
    target = degrade.relabel(match["label"], model.MIN_CONTROL)
    out = []
    for _ in range(model.N_COPIES):
        noisy = degrade.degrade(match, profile, rng, anchor_frac=float(rng.uniform(0.0, 0.6)))
        for swap in (False, True):
            X = features.build(noisy, swap=swap).to_numpy(np.float32)
            y = model.SWAP[target] if swap else target
            y = np.where(target < 0, -1, y)
            for lo, hi in match["halves"]:
                out.append(to_steps(X[lo:hi], y[lo:hi]))
    return out


class Net(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, 96), nn.ReLU(), nn.Dropout(0.1))
        self.gru = nn.GRU(96, hidden, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=0.1)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h)


class Scaler:
    def fit(self, seqs):
        allx = np.concatenate([s[0] for s in seqs])
        self.mu = np.nanmean(allx, 0)
        self.sd = np.nanstd(allx, 0) + 1e-6
        return self

    def __call__(self, x):
        z = (x - self.mu) / self.sd
        return np.nan_to_num(np.clip(z, -6, 6), nan=0.0).astype(np.float32)


def spell_weights(y, power):
    """Per-step loss weight (run length / 5 s) ** -power for H/A runs, clipped to [0.5, 4].

    Plain cross-entropy counts frames, so a 2-second possession weighs a tenth of a 20-second
    one, while spellF1 counts them equally. This moves the loss towards the metric.
    """
    w = np.ones(len(y), np.float32)
    change = np.flatnonzero(y[1:] != y[:-1]) + 1
    for a, b in zip(np.r_[0, change], np.r_[change, len(y)]):
        if y[a] in (0, 1):
            w[a:b] = np.clip(((b - a) / 25.0) ** -power, 0.5, 4.0)
    return w


def train(seqs, seed=SEED):
    epochs = config.CFG.get("seq_epochs", EPOCHS)
    hidden = config.CFG.get("seq_hidden", 64)
    power = config.CFG.get("seq_spell_power", 0.0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = Scaler().fit(seqs)
    data = [(scaler(x), y, spell_weights(y, power) if power else None)
            for x, y in seqs if len(x) > WIN]
    dev = device()
    net = Net(data[0][0].shape[1], hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * EPOCH_STEPS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="none" if power else "mean")
    net.train()
    for _ in range(epochs * EPOCH_STEPS):
        xb, yb, wb = [], [], []
        for _ in range(BATCH):
            x, y, w = data[rng.integers(len(data))]
            s = rng.integers(0, len(x) - WIN)
            xb.append(x[s:s + WIN])
            yb.append(y[s:s + WIN])
            if power:
                wb.append(w[s:s + WIN])
        xb = torch.from_numpy(np.stack(xb)).to(dev)
        yb = torch.from_numpy(np.stack(yb)).long().to(dev)
        loss = loss_fn(net(xb).reshape(-1, 3), yb.reshape(-1))
        if power:
            wt = torch.from_numpy(np.stack(wb)).to(dev).reshape(-1) * (yb.reshape(-1) >= 0)
            loss = (loss * wt).sum() / wt.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    net.eval()
    return net, scaler


def predict(net, scaler, match):
    """(n, 3) probabilities at 25 Hz, symmetrised over both team perspectives."""
    n = len(match["ball"])
    prob = np.full((n, 3), 1.0 / 3, np.float32)
    dev = device()
    parts = []
    for swap in (False, True):
        X = features.build(match, swap=swap).to_numpy(np.float32)
        p = np.full((n, 3), 1.0 / 3, np.float32)
        for lo, hi in match["halves"]:
            xs = scaler(to_steps(X[lo:hi]))
            with torch.no_grad():
                logits = net(torch.from_numpy(xs)[None].to(dev))[0]
            ps = torch.softmax(logits, -1).cpu().numpy()
            up = np.repeat(ps, STEP, 0)
            seg = hi - lo
            if len(up) < seg:
                up = np.vstack([up, np.repeat(up[-1:], seg - len(up), 0)])
            p[lo:hi] = up[:seg]
        parts.append(p[:, model.SWAP] if swap else p)
    prob[:] = (parts[0] + parts[1]) / 2
    return prob


def weights_path():
    return os.path.join(io_atd.MODELS_DIR, "gru_t{}.pt".format(config.TRIAL))


def save(net, scaler):
    os.makedirs(io_atd.MODELS_DIR, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()},
                "mu": torch.from_numpy(scaler.mu), "sd": torch.from_numpy(scaler.sd)},
               weights_path())


def load():
    blob = torch.load(weights_path(), map_location="cpu", weights_only=True)
    scaler = Scaler()
    scaler.mu, scaler.sd = blob["mu"].numpy(), blob["sd"].numpy()
    net = Net(len(scaler.mu), config.CFG.get("seq_hidden", 64))
    net.load_state_dict(blob["state"])
    return net.to(device()).eval(), scaler


def final():
    """Train on all external matches, predict the provided match."""
    profile = degrade.atd_gap_profile(io_atd.load_match())
    ids = io_idsse.available()
    seqs = [s for i, m in enumerate(ids) for s in sequences(io_idsse.load_match(m), profile, SEED + i)]
    net, scaler = train(seqs)
    save(net, scaler)
    prob = predict(net, scaler, io_atd.load_match())
    np.savez(os.path.join(io_atd.CACHE_DIR, "bar_prob_seq_t{}.npz".format(config.TRIAL)), prob=prob)
    print("saved GRU probabilities for BAR-ATH")


def main():
    if "--trial" in sys.argv:
        config.set_trial(sys.argv[sys.argv.index("--trial") + 1])
    if "--final" in sys.argv:
        return final()
    from . import cv
    profile = degrade.atd_gap_profile(io_atd.load_match())
    ids = io_idsse.available()
    matches = {m: io_idsse.load_match(m) for m in ids}
    seqs = {m: sequences(matches[m], profile, SEED + i) for i, m in enumerate(ids)}
    for i, held in enumerate(ids):
        cache = os.path.join(io_atd.CACHE_DIR, "oof_seq_t{}_{}.npz".format(config.TRIAL, held))
        if os.path.isfile(cache):
            continue
        t0 = time.time()
        net, scaler = train([s for m in ids if m != held for s in seqs[m]])
        test = degrade.degrade(matches[held], profile, np.random.default_rng(cv.SEED + 100 + i),
                               anchor_frac=0.3)
        np.savez(cache, prob=predict(net, scaler, test))
        print("seq fold {} done in {:.0f}s".format(held, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
