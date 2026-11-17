"""Leave-one-match-out validation (per trial: --trial N) on the external matches, then decoder tuning on those
out-of-fold predictions. The provided match and its public ground truth are not touched here.

    python -m pipeline.cv --trial 2  # writes cache/oof_t2_*.npz and cache/decoder_t2.json
"""

import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import config, decode, degrade, io_atd, io_idsse, model

sys.path.insert(0, os.path.join(io_atd.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

SEED = 2027


def gt_codes(match):
    lab = degrade.relabel(match["label"], model.MIN_CONTROL)
    return np.where(lab < 0, 3, lab).astype(np.int8)


def score(gt, pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main():
    if "--trial" in sys.argv:
        config.set_trial(sys.argv[sys.argv.index("--trial") + 1])
    print("trial", config.TRIAL, config.CFG)
    profile = degrade.atd_gap_profile(io_atd.load_match())
    ids = io_idsse.available()
    matches = {m: io_idsse.load_match(m) for m in ids}

    t0 = time.time()
    rows = {}
    for i, m in enumerate(ids):
        rows[m] = model.training_frames(matches[m], profile, SEED + i)
    print("features built in {:.0f}s".format(time.time() - t0))

    oof = {}
    for i, held in enumerate(ids):
        cache = os.path.join(io_atd.CACHE_DIR, "oof_t{}_{}.npz".format(config.TRIAL, held))
        if os.path.isfile(cache):
            oof[held] = np.load(cache)["prob"]
            continue
        t0 = time.time()
        X = pd.concat([rows[m][0] for m in ids if m != held], ignore_index=True)
        y = np.concatenate([rows[m][1] for m in ids if m != held])
        clf = model.fit(X, y)
        rng = np.random.default_rng(SEED + 100 + i)
        test = degrade.degrade(matches[held], profile, rng, anchor_frac=0.3)
        oof[held] = model.predict(clf, test)
        np.savez(cache, prob=oof[held])
        print("fold {} done in {:.0f}s".format(held, time.time() - t0))

    # Frame-model quality before decoding
    for m in ids:
        gt = gt_codes(matches[m])
        g = gt < 3
        acc = (oof[m].argmax(1)[g] == gt[g]).mean()
        print("{}  frame argmax acc {:.3f}".format(m, acc))

    # Decoder grid on out-of-fold predictions, chosen on the mean spellF1 across matches
    grid = list(itertools.product((1.0,), (3, 4, 5, 6, 8, 12), (4, 8, 16, 24, 32), (4, 8, 16, 24, 32)))
    results = []
    for alpha, cs, ci, co in grid:
        s = []
        for m in ids:
            pred = decode.decode(oof[m], matches[m]["halves"], alpha, cs, ci, co)
            s.append(score(gt_codes(matches[m]), pred))
        s = np.array(s, float)
        results.append(((alpha, cs, ci, co), s[:, 0].mean(), s[:, 1].mean(),
                        s[:, 2].mean(), s[:, 3].mean()))
    results.sort(key=lambda r: -r[1])
    print("\nalpha c_sw c_in c_out | spellF1 macroF1 | pred poss / gt poss")
    for (a, cs, ci, co), sf, mf, npred, ngt in results[:10]:
        print("{:4.1f} {:4.0f} {:4.0f} {:5.0f} | {:.4f}  {:.4f} | {:6.0f} / {:.0f}".format(
            a, cs, ci, co, sf, mf, npred, ngt))

    # Reference: the official baseline idea (nearest player + vote) on the same degraded data
    best = results[0][0]
    per = []
    for m in ids:
        pred = decode.decode(oof[m], matches[m]["halves"], *best)
        per.append((m,) + score(gt_codes(matches[m]), pred))
    print("\nper match at best setting:")
    for m, sf, mf, npred, ngt in per:
        print("  {}  spellF1 {:.4f}  macroF1 {:.4f}  poss {} / {}".format(m, sf, mf, npred, ngt))

    with open(os.path.join(io_atd.CACHE_DIR, "decoder_t{}.json".format(config.TRIAL)), "w") as handle:
        json.dump(dict(zip(("alpha", "c_switch", "c_in", "c_out"), best)), handle)
    print("\nsaved decoder setting", best)


if __name__ == "__main__":
    main()
