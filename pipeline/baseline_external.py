"""Reference points on the external held-out matches, same degraded copies as cv.py:

  * the official baseline (nearest player, majority vote +/-50 frames, missing ball = D)
  * our frame model with plain argmax (no decoder)
  * our frame model + HMM decoder (the trial 1 method)

This tells us whether each stage earns its place, without touching the provided match.
"""

import json
import os

import numpy as np

from . import config, cv, decode, degrade, features, io_atd, io_idsse


def baseline_codes(match, window=50):
    f = features.build(match)
    vis = ~np.isnan(match["ball"][:, 0])
    near = np.where(f["d1"].to_numpy() <= f["d2"].to_numpy(), 0, 1)
    codes = np.where(vis, near, -1)
    home = np.r_[0, np.cumsum(codes == 0)]
    away = np.r_[0, np.cumsum(codes == 1)]
    idx = np.arange(len(codes))
    lo, hi = np.maximum(0, idx - window), np.minimum(len(codes), idx + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    out = np.where(h + a > 0, np.where(h >= a, 0, 1), 2).astype(np.int8)
    out[~match["graded"]] = 3
    return out


def main():
    import sys
    if "--trial" in sys.argv:
        config.set_trial(sys.argv[sys.argv.index("--trial") + 1])
    profile = degrade.atd_gap_profile(io_atd.load_match())
    with open(os.path.join(io_atd.CACHE_DIR, "decoder_t{}.json".format(config.TRIAL))) as handle:
        setting = json.load(handle)
    rows = []
    for i, m in enumerate(io_idsse.available()):
        match = io_idsse.load_match(m)
        test = degrade.degrade(match, profile, np.random.default_rng(cv.SEED + 100 + i),
                               anchor_frac=0.3)
        gt = cv.gt_codes(match)
        prob = np.load(os.path.join(io_atd.CACHE_DIR, "oof_t{}_{}.npz".format(config.TRIAL, m)))["prob"]
        arg = np.where(match["graded"], prob.argmax(1), 3).astype(np.int8)
        dec = decode.decode(prob, match["halves"], **setting)
        rows.append([cv.score(gt, baseline_codes(test)), cv.score(gt, arg), cv.score(gt, dec)])
    r = np.array(rows, float)
    for k, name in enumerate(("baseline smooth50", "model argmax", "model + decoder")):
        print("{:<18} spellF1 {:.4f}  macroF1 {:.4f}  poss {:.0f} / {:.0f}".format(
            name, r[:, k, 0].mean(), r[:, k, 1].mean(), r[:, k, 2].mean(), r[:, k, 3].mean()))


if __name__ == "__main__":
    main()
