"""Possession-level decoder: an explicit-duration (semi-Markov) HMM.

decode.py charges a fixed cost per change of state, which implies geometric run lengths: every
extra second of a possession is as likely as the last, so a 2-second possession is never
"expected". Here a whole run is scored at once:

    score(run of state s, frames a..b) = alpha * sum log p_s  +  lam * log P_s(duration)
                                         - cost(previous state -> s)

with P_s(duration) learned from the external matches' relabelled possessions and stoppages
(H and A share one distribution). Decoding is exact Viterbi over runs at 5 Hz up to D_MAX steps;
longer runs continue as a new run of the same state (allowed, with the duration prior applied
again). Change points are then snapped at 25 Hz and runs under 25 frames absorbed, as in decode.py.
"""

import numpy as np

from .decode import STEP, _snap, absorb_short

D_MAX = 600  # 5 Hz steps = 120 s


def duration_logpdf(codes_list, halves_list, bw=2.0, floor=1e-4):
    """{0/1 (pooled possession), 2 (dead)} -> log pdf over 1..D_MAX steps (5 Hz), Gaussian-smoothed
    histogram in log-duration space so short and long runs are both resolved."""
    runs = {0: [], 2: []}
    for codes, halves in zip(codes_list, halves_list):
        for lo, hi in halves:
            c = codes[lo:hi]
            ch = np.flatnonzero(c[1:] != c[:-1]) + 1
            for a, b in zip(np.r_[0, ch], np.r_[ch, len(c)]):
                if c[a] in (0, 1):
                    runs[0].append(b - a)
                elif c[a] == 2:
                    runs[2].append(b - a)
    d = np.arange(1, D_MAX + 1)
    out = {}
    for k, r in runs.items():
        steps = np.clip(np.round(np.array(r) / STEP), 1, D_MAX)
        x, mu = np.log(d)[:, None], np.log(steps)[None, :]
        dens = np.exp(-0.5 * ((x - mu) / (bw * 0.1)) ** 2).sum(1) / d  # density in d, not log d
        dens = dens / dens.sum()
        out[k] = np.log(dens + floor / D_MAX)
    return np.stack([out[0], out[0], out[2]])  # (3, D_MAX) for states H, A, D


def _viterbi_semi(em, logdur, trans, lam):
    """em: (T, 3) per-step emission scores; logdur: (3, D_MAX); trans: (3, 3) costs.
    Returns the state path (T,)."""
    T, S = em.shape
    C = np.vstack([np.zeros((1, S)), np.cumsum(em, 0)])          # C[t] = sum em[:t]
    V = np.full((T + 1, S), -np.inf)
    U = np.full((T + 1, S), -np.inf)                              # best entry score at t into s
    U[0] = 0.0
    best_d = np.zeros((T + 1, S), np.int32)
    best_r = np.zeros((T + 1, S), np.int8)
    W = U - C                                                     # updated as U fills
    ldur = lam * logdur
    tr = -trans.copy()
    for t in range(1, T + 1):
        lo = max(0, t - D_MAX)
        n = t - lo
        for s in range(S):
            cand = W[t - 1:lo - 1 if lo > 0 else None:-1, s][:n] + ldur[s, :n]
            k = int(np.argmax(cand))
            V[t, s] = cand[k] + C[t, s]
            best_d[t, s] = k + 1
        # entry into s at t: from any r (self allowed: a long run continuing)
        e = V[t][:, None] + tr                                    # (r, s)
        best_r[t] = e.argmax(0)
        U[t] = e.max(0)
        W[t] = U[t] - C[t]
    path = np.empty(T, np.int8)
    t, s = T, int(np.argmax(V[T]))
    while t > 0:
        d = best_d[t, s]
        path[t - d:t] = s
        r = best_r[t - d, s] if t - d > 0 else s
        t, s = t - d, int(r)
    return path


def decode(prob, halves, logdur, alpha=1.0, lam=1.0, c_switch=4.0, c_in=4.0, c_out=4.0,
           c_self=8.0, min_run=25):
    n = len(prob)
    logp = np.log(np.clip(prob, 1e-6, 1.0))
    trans = np.array([[c_self, c_switch, c_in], [c_switch, c_self, c_in], [c_out, c_out, c_self]],
                     np.float64)
    codes = np.full(n, 3, np.int8)
    for lo, hi in halves:
        seg = logp[lo:hi]
        m = (hi - lo) // STEP * STEP
        blocks = seg[:m].reshape(-1, STEP, 3).sum(1)
        if m < hi - lo:
            blocks = np.vstack([blocks, seg[m:].sum(0, keepdims=True)])
        path5 = _viterbi_semi(alpha * blocks, logdur, trans, lam)
        path = np.repeat(path5, STEP)[:hi - lo]
        c = _snap(path, logp, lo)
        codes[lo:hi] = absorb_short(c, min_run) if min_run else c
    return codes
