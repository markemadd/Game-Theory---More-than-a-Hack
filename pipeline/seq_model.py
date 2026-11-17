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
    def __init__(self, n_in, hidden=64, change_head=False):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, 96), nn.ReLU(), nn.Dropout(0.1))
        self.gru = nn.GRU(96, hidden, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=0.1)
        self.out = nn.Linear(2 * hidden, 3)
        self.change = nn.Linear(2 * hidden, 1) if change_head else None

    def forward(self, x, with_change=False):
        h, _ = self.gru(self.inp(x))
        if with_change:
            return self.out(h), self.change(h)[..., 0]
        return self.out(h)


def change_target(y, dilate=1):
    """1 at the 5 Hz steps where a new H or A run starts (a change of possession, or play
    resuming with a team in possession), dilated by +/- `dilate` steps; -1 where unlabelled."""
    t = np.zeros(len(y), np.float32)
    start = np.flatnonzero((y[1:] != y[:-1]) & np.isin(y[1:], (0, 1)) & (y[:-1] >= 0)) + 1
    for k in range(-dilate, dilate + 1):
        idx = np.clip(start + k, 0, len(y) - 1)
        t[idx] = 1.0
    t[y < 0] = -1.0
    return t


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
    cp_w = config.CFG.get("seq_change_weight", 0.0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = Scaler().fit(seqs)
    data = [(scaler(x), y, spell_weights(y, power) if power else None,
             change_target(y) if cp_w else None)
            for x, y in seqs if len(x) > WIN]
    dev = device()
    net = Net(data[0][0].shape[1], hidden, change_head=bool(cp_w)).to(dev)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(4.0, device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * EPOCH_STEPS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="none" if power else "mean")
    net.train()
    for _ in range(epochs * EPOCH_STEPS):
        xb, yb, wb, cb = [], [], [], []
        for _ in range(BATCH):
            x, y, w, c = data[rng.integers(len(data))]
            s = rng.integers(0, len(x) - WIN)
            xb.append(x[s:s + WIN])
            yb.append(y[s:s + WIN])
            if power:
                wb.append(w[s:s + WIN])
            if cp_w:
                cb.append(c[s:s + WIN])
        xb = torch.from_numpy(np.stack(xb)).to(dev)
        yb = torch.from_numpy(np.stack(yb)).long().to(dev)
        if cp_w:
            logits, cl = net(xb, with_change=True)
        else:
            logits = net(xb)
        loss = loss_fn(logits.reshape(-1, 3), yb.reshape(-1))
        if power:
            wt = torch.from_numpy(np.stack(wb)).to(dev).reshape(-1) * (yb.reshape(-1) >= 0)
            loss = (loss * wt).sum() / wt.sum()
        if cp_w:
            ct = torch.from_numpy(np.stack(cb)).to(dev).reshape(-1)
            m = (ct >= 0).float()
            lc = bce(cl.reshape(-1), ct.clamp(min=0))
            loss = loss + cp_w * (lc * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    net.eval()
    return net, scaler


def _up(a, seg):
    up = np.repeat(a, STEP, 0)
    if len(up) < seg:
        up = np.concatenate([up, np.repeat(up[-1:], seg - len(up), 0)])
    return up[:seg]


def predict(net, scaler, match, with_change=False):
    """(n, 3) probabilities at 25 Hz, symmetrised over both team perspectives; with_change also
    returns (n,) the probability that a new possession starts here (NaN outside the halves)."""
    n = len(match["ball"])
    prob = np.full((n, 3), 1.0 / 3, np.float32)
    dev = device()
    parts, cps = [], []
    for swap in (False, True):
        X = features.build(match, swap=swap).to_numpy(np.float32)
        p = np.full((n, 3), 1.0 / 3, np.float32)
        c = np.full(n, np.nan, np.float32)
        for lo, hi in match["halves"]:
            xs = scaler(to_steps(X[lo:hi]))
            with torch.no_grad():
                out = net(torch.from_numpy(xs)[None].to(dev), with_change=with_change)
            logits = out[0][0] if with_change else out[0]
            p[lo:hi] = _up(torch.softmax(logits, -1).cpu().numpy(), hi - lo)
            if with_change:
                c[lo:hi] = _up(torch.sigmoid(out[1][0]).cpu().numpy(), hi - lo)
        parts.append(p[:, model.SWAP] if swap else p)
        cps.append(c)
    prob[:] = (parts[0] + parts[1]) / 2
    if with_change:
        return prob, (cps[0] + cps[1]) / 2
    return prob


def seeds():
    """Training seeds; trials with several average the GRUs' probabilities."""
    return list(config.CFG.get("seq_seeds", (SEED,)))


def weights_path(seed=None):
    if len(seeds()) == 1:
        return os.path.join(io_atd.MODELS_DIR, "gru_t{}.pt".format(config.TRIAL))
    return os.path.join(io_atd.MODELS_DIR, "gru_t{}_s{}.pt".format(config.TRIAL, seed))


def save(net, scaler, seed=None):
    os.makedirs(io_atd.MODELS_DIR, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()},
                "mu": torch.from_numpy(scaler.mu), "sd": torch.from_numpy(scaler.sd)},
               weights_path(seed))


def load_all():
    """[(net, scaler)] for every training seed of the trial."""
    return [load(sd) for sd in seeds()]


def predict_all(models, match):
    return np.mean([predict(net, scaler, match) for net, scaler in models], 0)


def load(seed=None):
    blob = torch.load(weights_path(seed), map_location="cpu", weights_only=True)
    scaler = Scaler()
    scaler.mu, scaler.sd = blob["mu"].numpy(), blob["sd"].numpy()
    net = Net(len(scaler.mu), config.CFG.get("seq_hidden", 64),
              change_head=bool(config.CFG.get("seq_change_weight", 0)))
    net.load_state_dict(blob["state"])
    return net.to(device()).eval(), scaler


def final(only=None):
    """Train on all external matches (once per training seed), predict the provided match.
    only: train just this seed (the others must already be in models/)."""
    profile = degrade.atd_gap_profile(io_atd.load_match())
    ids = io_idsse.available()
    seqs = [s for i, m in enumerate(ids) for s in sequences(io_idsse.load_match(m), profile, SEED + i)]
    for sd in seeds():
        if only is None or sd == only:
            net, scaler = train(seqs, seed=sd)
            save(net, scaler, sd)
            print("trained GRU seed", sd, flush=True)
    prob = predict_all(load_all(), io_atd.load_match())
    np.savez(os.path.join(io_atd.CACHE_DIR, "bar_prob_seq_t{}.npz".format(config.TRIAL)), prob=prob)
    print("saved GRU probabilities for BAR-ATH")


def main():
    if "--trial" in sys.argv:
        config.set_trial(sys.argv[sys.argv.index("--trial") + 1])
    if "--final" in sys.argv:
        return final(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else None)
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
        test = degrade.degrade(matches[held], profile, np.random.default_rng(cv.SEED + 100 + i),
                               anchor_frac=0.3)
        probs = [predict(*train([s for m in ids if m != held for s in seqs[m]], seed=sd), test)
                 for sd in seeds()]
        np.savez(cache, prob=np.mean(probs, 0))
        print("seq fold {} done in {:.0f}s".format(held, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
