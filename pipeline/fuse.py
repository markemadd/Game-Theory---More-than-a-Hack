"""Product-of-experts fusion: frame-model probabilities x likelihoods of extra observations.

Signals that exist only for the provided match (broadcast, video) cannot be learned from the
external matches. Each one is turned into a categorical observation per frame and combined as

    log p(state | everything) = log p_model(state) + sum_k  w_k * log P(obs_k | state) + const

with P(obs_k | state) estimated by EM on the provided match itself: the model's probabilities
are the prior, the posterior is the soft label, and the likelihood tables are re-estimated from
it until they stop moving. No human label and no ground truth is involved, which keeps this on
the "pseudo-labels produced by an algorithm" side of RULES.md. Live states (H, A) share one
likelihood row: an observation about the camera says whether play is alive, not who has it.
"""

import numpy as np

LIVE, DEAD = 0, 1


def em_likelihoods(prior, obs, n_cat, graded, iters=20, smooth=1.0):
    """prior: (n, 3) model probs; obs: (n,) int category or -1 (missing).
    Returns (2, n_cat) table P(obs | live/dead)."""
    m = graded & (obs >= 0)
    p_dead = prior[m, 2]
    o = obs[m]
    table = np.full((2, n_cat), 1.0 / n_cat)
    for _ in range(iters):
        like_live = table[LIVE, o]
        like_dead = table[DEAD, o]
        post_dead = p_dead * like_dead / (p_dead * like_dead + (1 - p_dead) * like_live)
        new = np.empty_like(table)
        for c in range(n_cat):
            hit = o == c
            new[DEAD, c] = post_dead[hit].sum() + smooth
            new[LIVE, c] = (1 - post_dead[hit]).sum() + smooth
        new /= new.sum(1, keepdims=True)
        if np.abs(new - table).max() < 1e-5:
            table = new
            break
        table = new
    return table


def apply(prior, obs, table, weight=1.0):
    """Multiply the likelihood into the prior and renormalise. Missing obs (-1) leave it as is."""
    out = prior.copy()
    ok = obs >= 0
    out[ok, 0] *= table[LIVE, obs[ok]] ** weight
    out[ok, 1] *= table[LIVE, obs[ok]] ** weight
    out[ok, 2] *= table[DEAD, obs[ok]] ** weight
    out /= out.sum(1, keepdims=True)
    return out


def broadcast_obs(sig, mismatch_m=1.5):
    """0 TV live and consistent, 1 TV live but showing another moment, 2 TV away, -1 no feed."""
    obs = np.full(len(sig["bc_covered"]), -1, np.int64)
    cov = sig["bc_covered"] == 1
    away = sig["bc_away"] == 1
    mism = np.nan_to_num(sig["bc_mismatch"], nan=0.0) > mismatch_m
    obs[cov & ~away] = 0
    obs[cov & ~away & mism] = 1
    obs[cov & away] = 2
    return obs
