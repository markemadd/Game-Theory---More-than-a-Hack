"""Read the provided ATD into dense numpy arrays, cached as .npz.

Output of load_match():
    ball    (n, 2)  metres, NaN where the tracker lost it
    home    (n, Kh, 2) metres, NaN for empty slots
    away    (n, Ka, 2) metres
    pitch   (length, width)
    graded  (n,) bool, the two scored halves
    halves  list of (lo, hi) half-open frame ranges

Coordinates are converted from the 0-1 convention to metres with the origin at the pitch centre,
the same frame the external (IDSSE) data uses, so every downstream feature is provider-agnostic.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

CACHE_DIR = os.environ.get("MTAH_CACHE") or os.path.join(ROOT, "cache")
MODELS_DIR = os.path.join(ROOT, "models")  # trained weights, committed


def _slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")])


def _parse_raw(raw_path, n_slots):
    """One pass through pandas' C parser after turning every separator into a comma."""
    with open(raw_path) as handle:
        text = handle.read().translate(str.maketrans(":;", ",,"))
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN"]).to_numpy()
    frame_id = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 3 * n_slots].reshape(len(table), n_slots, 3)
    ball = table[:, 1 + 3 * n_slots:1 + 3 * n_slots + 3]
    return frame_id, players, ball


def load_match(game=GAME, source="tactical", use_cache=True, raw_ball=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "{}_{}_atd.npz".format(game, source))
    length, width = GAMES[game]["pitch"]
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        out = {k: z[k] for k in z.files}
    else:
        paths = asset_paths(game, source=source)
        teams = _slot_teams(paths["atd_meta"])
        frame_id, players, ball = _parse_raw(paths["atd_raw"], len(teams))
        n = int(frame_id.max())
        assert np.array_equal(frame_id, np.arange(1, n + 1)), "ATD frame ids are not 1..n"

        def to_m(xy):
            out = np.empty_like(xy[..., :2])
            out[..., 0] = (xy[..., 0] - 0.5) * length
            out[..., 1] = (xy[..., 1] - 0.5) * width
            return out.astype(np.float32)

        out = {
            "ball": to_m(ball),
            "home": to_m(players[:, teams == 0]),
            "away": to_m(players[:, teams == 1]),
        }
        np.savez(cache, **out)
    from .config import CFG
    filled = os.path.join(CACHE_DIR, "ball", "ball_filled.npz")
    if source == "tactical" and CFG.get("filled_ball") and not raw_ball:
        out["ball"] = np.load(filled)["ball"]
    n = len(out["ball"])
    graded = np.zeros(n, bool)
    halves = []
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
        halves.append((first, last + 1))
    out.update(pitch=np.array([length, width]), graded=graded, halves=halves)
    return out


if __name__ == "__main__":
    m = load_match(use_cache=False)
    g = m["graded"]
    miss = np.isnan(m["ball"][:, 0])
    print("frames", len(miss), "home slots", m["home"].shape[1], "away slots", m["away"].shape[1])
    print("ball missing: all {:.1%}  graded {:.1%}".format(miss.mean(), miss[g].mean()))
    n_tr = (~np.isnan(m["home"][..., 0])).sum(1) + (~np.isnan(m["away"][..., 0])).sum(1)
    print("tracked objects per graded frame: mean {:.1f}".format(n_tr[g].mean()))
