"""Read the open IDSSE / DFL matches (Bassek et al. 2025, CC-BY 4.0) into the same arrays as io_atd.

These are the external training and validation matches. Their ball frames carry
BallPossession (1 home, 2 guest) and BallStatus (1 alive, 0 dead) on every frame, which is the
same kind of label our ground truth is derived from: a per-frame owner and alive flag on
corrected tracking. Labels here are produced by DFL's own pipeline, not by us.

Files are expected in ext_data/idsse/ as meta_<ID>.xml and pos_<ID>.xml (see fetch_idsse.sh).
Both halves are concatenated with a 1-frame NaN gap so a half boundary never looks like play.
"""

import os
import re

import numpy as np

from .io_atd import CACHE_DIR, ROOT

IDSSE_DIR = os.path.join(ROOT, "ext_data", "idsse")
MATCH_IDS = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")

_ATTR = re.compile(r'(\w+)="([^"]*)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"(.*?)/>')


def _home_id(meta_path):
    with open(meta_path, encoding="utf-8") as handle:
        text = handle.read()
    return re.search(r'HomeTeamId="([^"]+)"', text).group(1)


def _parse(pos_path, home_id):
    """{section: {"ball": (N, X, Y, poss, status), "home": {pid: (N, X, Y)}, "away": {...}}}"""
    sections = {}
    with open(pos_path, encoding="utf-8") as handle:
        text = handle.read()
    for block in text.split("<FrameSet ")[1:]:
        header = dict(_ATTR.findall(block[:block.index(">")]))
        team = header["TeamId"]
        if team == "referee":
            continue
        sec = sections.setdefault(header["GameSection"], {"home": {}, "away": {}})
        rows = _FRAME.findall(block)
        n = np.array([int(r[0]) for r in rows])
        x = np.array([float(r[1]) for r in rows], np.float32)
        y = np.array([float(r[2]) for r in rows], np.float32)
        if team == "BALL":
            poss = np.array([int(re.search(r'BallPossession="(\d)"', r[3]).group(1)) for r in rows])
            status = np.array([int(re.search(r'BallStatus="(\d)"', r[3]).group(1)) for r in rows])
            sec["ball"] = (n, x, y, poss, status)
        else:
            side = "home" if team == home_id else "away"
            sec[side][header["PersonId"]] = (n, x, y)
    return sections


def _densify(sections):
    """Stack both halves on one timeline. Labels: 0 H, 1 A, 2 D, -1 between halves."""
    parts = []
    for name in ("firstHalf", "secondHalf"):
        sec = sections[name]
        n_ball, bx, by, poss, status = sec["ball"]
        start, stop = n_ball.min(), n_ball.max() + 1
        count = stop - start
        ball = np.full((count, 2), np.nan, np.float32)
        ball[n_ball - start] = np.stack([bx, by], 1)
        label = np.full(count, -1, np.int8)
        label[n_ball - start] = np.where(status == 0, 2, np.where(poss == 1, 0, 1))
        teams = {}
        for side in ("home", "away"):
            arr = np.full((count, len(sec[side]), 2), np.nan, np.float32)
            for k, (n, x, y) in enumerate(sec[side].values()):
                keep = (n >= start) & (n < stop)
                arr[n[keep] - start, k] = np.stack([x[keep], y[keep]], 1)
            teams[side] = arr
        parts.append((ball, teams["home"], teams["away"], label))

    k_home = max(p[1].shape[1] for p in parts)
    k_away = max(p[2].shape[1] for p in parts)

    def pad(a, k):
        out = np.full((a.shape[0] + 1, k, 2), np.nan, np.float32)  # +1 NaN separator row
        out[:a.shape[0], :a.shape[1]] = a
        return out

    ball = np.concatenate([np.vstack([p[0], np.full((1, 2), np.nan, np.float32)]) for p in parts])
    home = np.concatenate([pad(p[1], k_home) for p in parts])
    away = np.concatenate([pad(p[2], k_away) for p in parts])
    label = np.concatenate([np.r_[p[3], np.int8(-1)] for p in parts]).astype(np.int8)
    return ball, home, away, label


def load_match(match_id, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "idsse_{}.npz".format(match_id))
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        out = {k: z[k] for k in z.files}
    else:
        home_id = _home_id(os.path.join(IDSSE_DIR, "meta_{}.xml".format(match_id)))
        sections = _parse(os.path.join(IDSSE_DIR, "pos_{}.xml".format(match_id)), home_id)
        ball, home, away, label = _densify(sections)
        out = dict(ball=ball, home=home, away=away, label=label)
        np.savez(cache, **out)
    label = out["label"]
    graded = label >= 0
    edges = np.flatnonzero(np.diff(np.r_[0, graded.astype(np.int8), 0]))
    out.update(pitch=np.array([105.0, 68.0]), graded=graded,
               halves=list(zip(edges[::2], edges[1::2])))
    return out


def available():
    return [m for m in MATCH_IDS
            if os.path.isfile(os.path.join(IDSSE_DIR, "pos_{}.xml".format(m)))
            and os.path.getsize(os.path.join(IDSSE_DIR, "pos_{}.xml".format(m))) > 1e8]
