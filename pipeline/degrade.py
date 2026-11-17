"""Make clean external tracking look like the provided ATD, and relabel it to the challenge's
possession definition.

Degradation is fitted to the ATD itself (no ground truth involved):
  * the ball disappears in long blocks, with visible/missing run lengths bootstrapped from the
    ATD's own graded frames (median missing run ~10 s, not short flicker);
  * a fraction of missing runs is anchored to the start of dead-ball stretches, because a
    stoppage is when the camera and the tracker most often lose the ball; that fraction is
    randomised per copy so the model cannot lean on one value;
  * players vanish in short runs until ~20.5 objects remain per frame, as in the ATD;
  * the ball is smoothed (Gaussian, sigma per trial in config.py) plus low-frequency noise so its acceleration
    profile matches the ATD's, whose ball track is heavily filtered (99th pct ~37 m/s^2 vs ~220
    in DFL). Players are left as they are: their speed profile already matches the ATD's.

Relabelling follows CHALLENGE.md: "a simple touch of the ball without control" is not a change of
possession. DFL's raw BallPossession flips on touches, so team runs shorter than `min_control`
frames are absorbed into the surrounding possession. The threshold is a documented choice, never
fitted to the public ground truth.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import CFG

H, A, D = 0, 1, 2


def run_lengths(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return e[::2], e[1::2] - e[::2]


def atd_gap_profile(atd):
    """Visible and missing ball run lengths on the graded frames of the provided match."""
    miss = np.isnan(atd["ball"][:, 0])
    g = atd["graded"]
    _, miss_runs = run_lengths(miss & g)
    _, vis_runs = run_lengths(~miss & g)
    n_tracked = ((~np.isnan(atd["home"][..., 0])).sum(1)
                 + (~np.isnan(atd["away"][..., 0])).sum(1))[g].mean()
    return {"miss": miss_runs, "vis": vis_runs, "n_tracked": float(n_tracked),
            "miss_rate": float(miss[g].mean())}


def relabel(label, min_control=25):
    """Absorb H/A runs shorter than min_control into the neighbouring possession, shortest first."""
    out = label.copy()
    while True:
        starts = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
        stops = np.r_[starts[1:], len(out)]
        lens = stops - starts
        labs = out[starts]
        short = np.flatnonzero(((labs == H) | (labs == A)) & (lens < min_control))
        if not len(short):
            return out
        done_any = False
        touched = set()
        for i in short[np.argsort(lens[short], kind="stable")]:
            if i in touched or i - 1 in touched or i + 1 in touched:
                continue  # a neighbour changed this sweep; revisit next sweep
            a, b = starts[i], stops[i]
            prev_lab = labs[i - 1] if i > 0 else -1
            next_lab = labs[i + 1] if i + 1 < len(labs) else -1
            fill = prev_lab if prev_lab in (H, A) else next_lab if next_lab in (H, A) else -1
            if fill < 0:
                continue
            out[a:b] = fill
            touched.add(i)
            done_any = True
        if not done_any:
            return out


def _atd_ball_behaviour(raw, ball, home, away, pitch, rng, still_mask, oob_mask):
    """Make DFL's dead-time ball behave like the ATD's.

    DFL keeps reporting a still ball (23% of dead frames < 0.5 m/s) and a ball held outside the
    lines; the ATD almost never does (2.6% still, rarely off the pitch) -- its tracker loses
    the ball instead. Without this, the model learns "still/off-pitch ball = dead", a cue that
    never fires on the provided match. Still runs are hidden (prob still_mask) or made to follow
    the nearest player, as a carried ball; off-pitch runs are hidden with prob oob_mask.
    Returns the mask of frames to hide; `ball` is modified in place for carried runs.
    """
    n = len(raw)
    hide = np.zeros(n, bool)
    step = np.r_[0.0, np.hypot(*np.diff(raw, axis=0).T)] * 25
    still = step < 0.3
    starts, lens = run_lengths(still)
    players = np.concatenate([home, away], axis=1)
    for s, length in zip(starts, lens):
        if length < 10:
            continue
        if rng.random() < still_mask:
            hide[s:s + length] = True
            continue
        d = np.hypot(players[s, :, 0] - raw[s, 0], players[s, :, 1] - raw[s, 1])
        if np.all(np.isnan(d)):
            hide[s:s + length] = True
            continue
        k = int(np.nanargmin(d))
        offset = rng.normal(0, 0.4, 2)
        track = players[s:s + length, k] + offset
        noise = gaussian_filter1d(rng.normal(0, 0.3, (length, 2)), 3, axis=0)
        ball[s:s + length] = (track + noise).astype(np.float32)
        hide[s:s + length] |= np.isnan(track[:, 0])
    length, width = pitch
    off = (np.abs(raw[:, 0]) > length / 2) | (np.abs(raw[:, 1]) > width / 2)
    starts, lens = run_lengths(off)
    for s, run in zip(starts, lens):
        if rng.random() < oob_mask:
            hide[s:s + run] = True
    return hide


def degrade(match, profile, rng, anchor_frac=0.3, ball_sigma=None, ball_noise_m=None):
    """Return a degraded copy of an external match (labels untouched)."""
    ball_sigma = CFG["ball_sigma"] if ball_sigma is None else ball_sigma
    ball_noise_m = CFG["ball_noise_m"] if ball_noise_m is None else ball_noise_m
    n = len(match["ball"])
    ball = gaussian_filter1d(match["ball"].astype(np.float64), ball_sigma, axis=0, mode="nearest")
    ball += gaussian_filter1d(rng.normal(0, ball_noise_m, ball.shape), 3, axis=0)
    ball = ball.astype(np.float32)
    home = match["home"].copy()
    away = match["away"].copy()
    label = match["label"]
    behave = np.zeros(n, bool)
    if CFG.get("dead_ball_behaviour"):
        behave = _atd_ball_behaviour(match["ball"], ball, home, away, match["pitch"], rng,
                                     CFG["still_mask"], CFG["oob_mask"])

    # Ball: alternating renewal process with ATD run lengths.
    miss = np.zeros(n, bool)
    t = int(rng.integers(0, int(np.median(profile["vis"]))))
    while t < n:
        t += int(rng.choice(profile["vis"]))
        length = int(rng.choice(profile["miss"]))
        miss[t:t + length] = True
        t += length
    # Anchor some gaps to stoppages: a missing run starting near the beginning of a dead stretch.
    d_starts, d_lens = run_lengths(label == D)
    for s, length in zip(d_starts, d_lens):
        if rng.random() < anchor_frac:
            offset = int(rng.integers(0, max(1, min(50, length))))
            miss[s + offset:s + offset + int(rng.choice(profile["miss"]))] = True
    # Thin the random part back toward the ATD miss rate if anchoring overshot.
    g = match["graded"]
    excess = (miss | behave)[g].mean() - profile["miss_rate"]
    if excess > 0.02:
        starts, lens = run_lengths(miss)
        drop = rng.random(len(starts)) < excess / max(miss[g].mean(), 1e-6)
        for s, length, d in zip(starts, lens, drop):
            if d:
                miss[s:s + length] = False
    ball[miss | behave] = np.nan

    # Players: drop short runs until the tracked count matches the ATD.
    n_now = ((~np.isnan(home[..., 0])).sum(1) + (~np.isnan(away[..., 0])).sum(1))[g].mean()
    p_drop = max(0.0, 1.0 - profile["n_tracked"] / max(n_now, 1.0))
    for arr in (home, away):
        for k in range(arr.shape[1]):
            n_runs = int(p_drop * n / 75)
            for s in rng.integers(0, n, n_runs):
                arr[s:s + int(rng.integers(25, 126)), k] = np.nan

    out = dict(match)
    out.update(ball=ball, home=home, away=away)
    return out
