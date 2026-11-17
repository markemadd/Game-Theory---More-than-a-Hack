"""Metrica Smart Tagging -> per-frame categorical observation, fused with team-aware likelihoods.

Smart Tagging is Metrica's automatic tactical annotation (FAQ.md: generated without manual review,
allowed). Its phases carry a Team tag, so unlike the broadcast it can say *which* team has the
ball, not just whether play is alive. It exists only for the provided match, so it cannot be
learned from the external matches: its likelihoods P(obs | H), P(obs | A), P(obs | D) are
EM-estimated on the match from the model's own probabilities, as in pipeline/fuse.py, but with
separate H and A rows. Nothing assumes what a tag means (e.g. whether the Team on a defending
phase is the team without the ball); EM learns it.

    python -m pipeline.smart_tagging       # print the category counts and the learned table
"""

import os
import sys

import numpy as np

from . import io_atd

sys.path.insert(0, os.path.join(io_atd.ROOT, "starter"))
import load_smart_tagging as lst  # noqa: E402

ATTACK = {"build up", "progression", "ball in final third", "ball in the box", "players in the box",
          "creating chances", "long ball", "goals"}
TRANSITION = {"attacking transition"}
DEFEND = {"defending in defensive third", "defending in middle third",
          "defending in attacking third", "defensive transition"}
SET_PIECE = "set pieces"
TEAMS = ("barcelona", "atletic club")
PRE_SET_PIECE_S = 3.0

NAMES = ["none", "attack Barcelona", "attack Athletic", "att.transition Barcelona",
         "att.transition Athletic", "defend Barcelona", "defend Athletic",
         "set piece Barcelona", "set piece Athletic", "before set piece", "phase, no team"]
N_CAT = len(NAMES)


def _family(code):
    c = (code or "").strip().lower()
    if c == SET_PIECE:
        return "sp"
    if c in TRANSITION:
        return "tr"
    if c in ATTACK:
        return "att"
    if c in DEFEND:
        return "def"
    return None


def observations(n=None, fps=25):
    """(n,) int category per video frame. Overlaps resolved by priority:
    before set piece > set piece > attacking transition > attack > defend > phase without team."""
    events = lst.load()
    n = n or len(io_atd.load_match(raw_ball=True)["ball"])
    rank = np.zeros(n, np.int8)          # higher wins
    obs = np.zeros(n, np.int64)
    base = {"att": 1, "tr": 3, "def": 5, "sp": 7}
    prio = {"def": 2, "att": 3, "tr": 4, "sp": 5}

    def put(lo, hi, cat, p):
        lo, hi = max(0, lo), min(n, hi)
        sel = np.arange(lo, hi)
        sel = sel[rank[sel] < p]
        obs[sel] = cat
        rank[sel] = p

    for e in events:
        fam = _family(e["code"])
        if fam is None:
            continue
        team = (e["tags"].get("Team") or "").strip().lower()
        if team in TEAMS:
            put(e["start_frame"], e["end_frame"], base[fam] + TEAMS.index(team), prio[fam])
        else:
            put(e["start_frame"], e["end_frame"], 10, 1)
        if fam == "sp":
            put(e["start_frame"] - int(PRE_SET_PIECE_S * fps), e["start_frame"], 9, 6)
    return obs


def em_likelihoods(prior, obs, graded, iters=50, smooth=1.0):
    """(3, N_CAT) table P(obs | H/A/D), EM on graded frames with the model's probs as prior."""
    m = graded & (obs >= 0)
    p, o = prior[m].astype(np.float64), obs[m]
    table = np.full((3, N_CAT), 1.0 / N_CAT)
    for _ in range(iters):
        post = p * table[:, o].T
        post /= post.sum(1, keepdims=True)
        new = np.stack([np.bincount(o, weights=post[:, s], minlength=N_CAT) for s in range(3)]) + smooth
        new /= new.sum(1, keepdims=True)
        done = np.abs(new - table).max() < 1e-6
        table = new
        if done:
            break
    return table


def apply(prior, obs, table, weight=1.0):
    ok = obs >= 0
    out = prior.astype(np.float64).copy()
    out[ok] *= table[:, obs[ok]].T ** weight
    out /= out.sum(1, keepdims=True)
    return out.astype(np.float32)


if __name__ == "__main__":
    bar = io_atd.load_match()
    g = bar["graded"]
    obs = observations(len(g))
    counts = np.bincount(obs[g], minlength=N_CAT)
    for k, name in enumerate(NAMES):
        print("{:>2} {:<26} {:6.1%}".format(k, name, counts[k] / g.sum()))
