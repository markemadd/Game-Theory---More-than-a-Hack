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
    """score: (T, S) per-step rewards; trans: (S, S) costs from->to, or (T, S, S) costs into
    step t. Returns best path."""
    T, S = score.shape
    back = np.zeros((T, S), np.int8)
    acc = score[0].astype(np.float64).copy()
    for t in range(1, T):
        cand = acc[:, None] - (trans[t] if trans.ndim == 3 else trans)
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


def decode(prob, halves, alpha=1.0, c_switch=8.0, c_in=8.0, c_out=6.0, min_run=0, change=None,
           beta=0.0, gamma=0.0, base_rate=0.035):
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
        if change is not None and gamma > 0:
            # Additive form: entering H or A costs gamma * (logit p - logit base rate) less, so
            # a confident change-point pays for a boundary on its own; never below 0.5.
            cp = np.clip(np.nan_to_num(change[lo:hi], nan=base_rate), 1e-4, 1 - 1e-4)
            cp5 = np.maximum.reduceat(cp, np.arange(0, hi - lo, STEP))[:len(blocks)]
            bonus = gamma * (np.log(cp5 / (1 - cp5)) - np.log(base_rate / (1 - base_rate)))
            tt = np.repeat(trans[None], len(blocks), 0)
            for r in range(3):
                for s_ in (0, 1):
                    if r != s_:
                        tt[:, r, s_] = np.maximum(tt[:, r, s_] - bonus, 0.5)
            path5 = viterbi(alpha * blocks, tt)
        elif change is not None and beta > 0:
            # Possession-change evidence: entering H or A from another state is cheaper where
            # the model says a new possession starts (cost x (1 - beta * p), block maximum).
            cp = np.nan_to_num(change[lo:hi], nan=0.0)
            cp5 = np.maximum.reduceat(cp, np.arange(0, hi - lo, STEP))[:len(blocks)]
            tt = np.repeat(trans[None], len(blocks), 0)
            scale = 1.0 - beta * np.clip(cp5, 0, 1)
            for r in range(3):
                for s_ in (0, 1):
                    if r != s_:
                        tt[:, r, s_] *= scale
            path5 = viterbi(alpha * blocks, tt)
        else:
            path5 = viterbi(alpha * blocks, trans)
        path = np.repeat(path5, STEP)[:hi - lo]
        seg_codes = _snap(path, logp, lo)
        codes[lo:hi] = absorb_short(seg_codes, min_run) if min_run else seg_codes
    return codes


# -- adaptive switch cost ----------------------------------------------------------------------
# The switch cost is tuned where the frame model is sharp. On a match where it is less confident,
# the same cost removes more of the model's own team changes and merges possessions. The rule
# below needs no labels: it measures what share of the model's (1 s smoothed) team changes the
# decoder keeps, and lowers the switch cost until that share matches the one measured on the
# external matches at the tuned cost.

ADAPT_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24)


def raw_team_changes(prob, halves, smooth=25):
    """Team changes in the frame argmax after a 1 s moving average, dead frames skipped."""
    n = 0
    kernel = np.ones(smooth) / smooth
    for lo, hi in halves:
        avg = np.stack([np.convolve(prob[lo:hi, j], kernel, "same") for j in range(3)], 1)
        a = avg.argmax(1)
        ha = a[a < 2]
        n += int((ha[1:] != ha[:-1]).sum())
    return n


def decoded_team_changes(codes, halves):
    n = 0
    for lo, hi in halves:
        x = codes[lo:hi]
        runs = x[np.flatnonzero(np.r_[True, x[1:] != x[:-1]])]
        ha = runs[runs < 2]
        n += int((ha[1:] != ha[:-1]).sum())
    return n


def kept_share(prob, halves, min_run=25, **costs):
    return decoded_team_changes(decode(prob, halves, min_run=min_run, **costs), halves) / max(
        raw_team_changes(prob, halves), 1)


def adaptive_switch(prob, halves, target, floor=4, min_run=25, **costs):
    """The switch cost, at or below the tuned one, whose kept share is closest to target
    (ties go to the higher cost)."""
    top = costs["c_switch"]
    grid = [c for c in ADAPT_GRID if floor <= c <= top]
    shares = {c: kept_share(prob, halves, min_run=min_run, **dict(costs, c_switch=c)) for c in grid}
    return min(grid, key=lambda c: (abs(shares[c] - target), -c)), shares
