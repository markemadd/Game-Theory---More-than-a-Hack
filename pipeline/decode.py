"""Turn per-frame H/A/D probabilities into possessions.

A 3-state HMM decoded with Viterbi: emission = alpha * log p, and a fixed cost for every state
change (H<->A, into dead ball, out of dead ball). Decoding runs at 5 Hz (block means of log p) for
speed, then each change point is snapped to the best frame within its block at 25 Hz.

All labels are written explicitly: the decoder never abstains on a graded frame, because the
scorer's forward fill makes X equal to repeating the previous label on spellF1 and a sure miss
on macroF1.
"""

import numpy as np

STEP = 5


def viterbi(score, trans):
    """score: (T, S) per-step rewards; trans: (S, S) costs from->to. Returns best path."""
    T, S = score.shape
    back = np.zeros((T, S), np.int8)
    acc = score[0].astype(np.float64).copy()
    for t in range(1, T):
        cand = acc[:, None] - trans
        back[t] = cand.argmax(0)
        acc = cand[back[t], np.arange(S)] + score[t]
    path = np.empty(T, np.int8)
    path[-1] = acc.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _snap(path, logp, lo):
    """Move each change point to the frame that maximises the two adjacent states' evidence."""
    change = np.flatnonzero(path[1:] != path[:-1]) + 1
    out = path.copy()
    for c in change:
        a, b = out[c - 1], out[c]
        left, right = max(1, c - STEP), min(len(out) - 1, c + STEP)
        if right <= left:
            continue
        diff = logp[lo + left - 1:lo + right + 1, a] - logp[lo + left - 1:lo + right + 1, b]
        # choose split s in [left, right]: frames < s are a, >= s are b
        gain = np.cumsum(diff)
        best = left + int(np.argmax(gain[:right - left + 1]))
        lo_i, hi_i = min(c, best), max(c, best)
        out[lo_i:hi_i] = a if best > c else b
    return out


def absorb_short(codes, min_run):
    """Give every H/A run shorter than min_run frames the label before it (the one after it at
    the start of a half). CHALLENGE.md does not count such a touch as a possession."""
    out = codes.copy()
    change = np.flatnonzero(out[1:] != out[:-1]) + 1
    starts, stops = np.r_[0, change], np.r_[change, len(out)]
    for k, (a, b) in enumerate(zip(starts, stops)):
        if out[a] in (0, 1) and b - a < min_run:
            out[a:b] = out[a - 1] if k else out[min(b, len(out) - 1)]
    return out


def decode(prob, halves, alpha=1.0, c_switch=8.0, c_in=8.0, c_out=6.0, min_run=0):
    """prob: (n, 3) H/A/D probabilities. Returns int8 codes 0/1/2 inside halves, 3 (X) elsewhere."""
    n = len(prob)
    logp = np.log(np.clip(prob, 1e-6, 1.0))
    trans = np.array([[0, c_switch, c_in], [c_switch, 0, c_in], [c_out, c_out, 0]], np.float64)
    codes = np.full(n, 3, np.int8)
    for lo, hi in halves:
        seg = logp[lo:hi]
        m = (hi - lo) // STEP * STEP
        blocks = seg[:m].reshape(-1, STEP, 3).sum(1)
        if m < hi - lo:
            blocks = np.vstack([blocks, seg[m:].sum(0, keepdims=True)])
        path5 = viterbi(alpha * blocks, trans)
        path = np.repeat(path5, STEP)[:hi - lo]
        seg_codes = _snap(path, logp, lo)
        codes[lo:hi] = absorb_short(seg_codes, min_run) if min_run else seg_codes
    return codes
