"""Broadcast feed -> per-tactical-frame "the TV is not showing live play" signals.

The broadcast is not frame-aligned to the tactical video, but live TV does not pause: within a
continuous recording the two timelines differ by a constant offset. The offset is found from
the two ATDs alone (both are in pitch coordinates): sample broadcast frames, find the tactical
frame whose ball and players match, and take the dominant offsets. No video is decoded.

Signals per tactical frame t (broadcast frame b = t - offset):
  bc_away      the broadcast has no pitch homography at b: replay, close-up, crowd, graphics
  bc_mismatch  the broadcast's tracked players disagree with the tactical ones at t (a replay
               shot from a wide camera still has a homography, but shows another moment)
  bc_covered   1 where a broadcast frame maps to t at all
"""

import os

import numpy as np
import pandas as pd

from . import io_atd

SAMPLE_EVERY = 250
BALL_TOL_M = 1.5


def _players(m, f):
    p = np.concatenate([m["home"][f], m["away"][f]], 0)
    return p[~np.isnan(p[:, 0])]


def _nn_score(pb, pt):
    """Mean distance from each broadcast player to the nearest tactical player."""
    if len(pb) < 6 or len(pt) < 6:
        return np.inf
    d = np.hypot(pb[:, None, 0] - pt[None, :, 0], pb[:, None, 1] - pt[None, :, 1])
    return float(np.median(d.min(1)))


def match_samples(bc, tac, valid):
    """[(broadcast frame, best tactical frame, score)] for sampled broadcast frames."""
    tball = tac["ball"]
    out = []
    for b in range(0, len(bc["ball"]), SAMPLE_EVERY):
        if not valid[b] or np.isnan(bc["ball"][b, 0]):
            continue
        pb = _players(bc, b)
        if len(pb) < 10:
            continue
        d = np.hypot(tball[:, 0] - bc["ball"][b, 0], tball[:, 1] - bc["ball"][b, 1])
        cand = np.flatnonzero(d < BALL_TOL_M)
        if not len(cand):
            continue
        scores = [_nn_score(pb, _players(tac, t)) for t in cand]
        k = int(np.argmin(scores))
        out.append((b, int(cand[k]), scores[k]))
    return out


def offset_segments(samples, max_score=1.0, min_support=5):
    """Constant-offset segments: runs of consecutive good samples sharing one offset (+/-2)."""
    good = [(b, t - b) for b, t, s in samples if s < max_score]
    segs = []
    for b, off in good:
        if segs and abs(off - segs[-1]["offset"]) <= 2:
            segs[-1]["end"] = b
            segs[-1]["n"] += 1
            segs[-1]["offs"].append(off)
        else:
            segs.append({"start": b, "end": b, "offset": off, "n": 1, "offs": [off]})
    segs = [s for s in segs if s["n"] >= min_support]
    for s in segs:
        s["offset"] = int(np.median(s["offs"]))
        del s["offs"]
    return segs


def refine_offset(bc, tac, seg, valid, search=3):
    """Frame-exact offset: minimise the player score over +/-search frames."""
    best = None
    frames = [b for b in range(seg["start"], seg["end"] + 1, 25) if valid[b]][:200]
    for off in range(seg["offset"] - search, seg["offset"] + search + 1):
        s = [_nn_score(_players(bc, b), _players(tac, b + off)) for b in frames
             if 0 <= b + off < len(tac["ball"])]
        s = [x for x in s if np.isfinite(x)]
        score = float(np.median(s)) if s else np.inf
        if best is None or score < best[1]:
            best = (off, score)
    return best


def signals(tac, bc, valid, segs):
    n = len(tac["ball"])
    covered = np.zeros(n, np.float32)
    away = np.full(n, np.nan, np.float32)
    mismatch = np.full(n, np.nan, np.float32)
    # Each segment's offset holds from its first to its last supporting sample; extend to the
    # neighbouring segment boundaries so a replay inside a segment is still covered.
    bounds = [0] + [(a["end"] + b["start"]) // 2 for a, b in zip(segs, segs[1:])] + [len(valid)]
    for seg, lo, hi in zip(segs, bounds[:-1], bounds[1:]):
        b = np.arange(lo, hi)
        t = b + seg["offset"]
        ok = (t >= 0) & (t < n)
        b, t = b[ok], t[ok]
        covered[t] = 1
        away[t] = 1 - valid[b]
        for k in range(0, len(b), 5):
            bb, tt = b[k], t[k]
            if valid[bb]:
                s = _nn_score(_players(bc, bb), _players(tac, tt))
                mismatch[tt:tt + 5] = s if np.isfinite(s) else np.nan
    return {"bc_covered": covered, "bc_away": away, "bc_mismatch": mismatch}


def build(use_cache=True):
    cache = os.path.join(io_atd.CACHE_DIR, "broadcast_signals.npz")
    if use_cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    tac = io_atd.load_match(raw_ball=True)
    bc = io_atd.load_match(source="broadcast")
    hom = pd.read_csv(os.path.join(io_atd.ROOT, "data", "broadcast", "BAR-ATH_broadcast_homography.csv"),
                      usecols=["valid"])
    valid = hom["valid"].to_numpy().astype(np.float32)
    samples = match_samples(bc, tac, valid)
    segs = offset_segments(samples)
    for s in segs:
        s["offset"], s["score"] = refine_offset(bc, tac, s, valid)
    out = signals(tac, bc, valid, segs)
    out["segments"] = np.array([(s["start"], s["end"], s["offset"], s["n"], s["score"]) for s in segs])
    out["samples"] = np.array(samples)
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    out = build(use_cache=False)
    samples = out["samples"]
    print("matched samples", len(samples), " good (<1 m)", int((samples[:, 2] < 1.0).sum()))
    print("segments (broadcast start, end, offset, support, score):")
    for row in out["segments"]:
        print("  {:>7.0f} {:>7.0f} {:>+7.0f} {:>4.0f} {:.2f}".format(*row))
    tac = io_atd.load_match(raw_ball=True)
    g = tac["graded"]
    print("graded frames covered by the broadcast: {:.1%}".format(out["bc_covered"][g].mean()))
    print("of those, broadcast away from live play: {:.1%}".format(np.nanmean(out["bc_away"][g])))
