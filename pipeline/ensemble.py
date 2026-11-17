"""Compare experts and their ensemble on the external out-of-fold predictions.

    python -m pipeline.ensemble --trial 3 [--adaptive]

For each candidate (trees, GRU, weighted average), the decoder costs are re-tuned on the same
out-of-fold matches, so every row is the best that candidate can do. Saves the winning weight
and decoder setting to cache/ensemble_t<N>.json.
"""

import itertools
import json
import os
import sys

import numpy as np

from . import config, cv, decode, io_atd, io_idsse


def tune(probs, matches, gts):
    best = None
    for cs, ci, co in itertools.product((4, 6, 8, 12), (8, 16, 24, 32), (8, 16, 24, 32)):
        s = np.array([cv.score(gts[m], decode.decode(probs[m], matches[m]["halves"], 1.0, cs, ci, co))
                      for m in probs], float)
        row = ((1.0, cs, ci, co), s[:, 0].mean(), s[:, 1].mean(), s[:, 0].std(), s[:, 2].mean())
        if best is None or row[1] > best[1]:
            best = row
    return best


def main():
    if "--trial" in sys.argv:
        config.set_trial(sys.argv[sys.argv.index("--trial") + 1])
    t = config.TRIAL
    ids = io_idsse.available()
    matches = {m: io_idsse.load_match(m) for m in ids}
    gts = {m: cv.gt_codes(matches[m]) for m in ids}
    tree = {m: np.load(os.path.join(io_atd.CACHE_DIR, "oof_t{}_{}.npz".format(t, m)))["prob"] for m in ids}
    seq = {m: np.load(os.path.join(io_atd.CACHE_DIR, "oof_seq_t{}_{}.npz".format(t, m)))["prob"] for m in ids}
    rows = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):  # weight on the GRU
        probs = {m: (1 - w) * tree[m] + w * seq[m] for m in ids}
        acc = np.mean([(probs[m].argmax(1)[gts[m] < 3] == gts[m][gts[m] < 3]).mean() for m in ids])
        best = tune(probs, matches, gts)
        rows.append((w, acc) + best)
        print("GRU weight {:.2f} | frame acc {:.3f} | spellF1 {:.4f} (sd {:.3f}) macroF1 {:.4f} | "
              "poss {:.0f} | decoder {}".format(w, acc, best[1], best[3], best[2], best[4], best[0]),
              flush=True)
    w, _, setting, sf, mf, _, _ = max(rows, key=lambda r: r[3])
    out = {"gru_weight": w, "decoder": dict(zip(("alpha", "c_switch", "c_in", "c_out"), setting)),
           "spell_f1": sf, "macro_f1": mf}
    if "--adaptive" in sys.argv:
        # share of the model's own team changes the chosen decoder keeps on these matches; the
        # submission lowers its switch cost on the provided match until it keeps the same share
        probs = {m: (1 - w) * tree[m] + w * seq[m] for m in ids}
        share = float(np.mean([decode.kept_share(probs[m], matches[m]["halves"], **out["decoder"])
                               for m in ids]))
        out["adaptive"] = {"target": round(share, 4), "floor": 4}
        print("adaptive switch cost: kept-share target", round(share, 4))
    with open(os.path.join(io_atd.CACHE_DIR, "ensemble_t{}.json".format(t)), "w") as handle:
        json.dump(out, handle)
    print("best: GRU weight", w, "decoder", setting)


if __name__ == "__main__":
    main()
