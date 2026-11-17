"""One trial, end to end: train on all external matches, label BAR-ATH, write the submission.

    python -m pipeline.cv --trial N         # once: out-of-fold validation + decoder setting
    python -m pipeline.run_trial --trial N  # writes evaluation/results/trialN.csv

Deterministic: fixed seeds, pinned dependencies (requirements-pipeline.txt).
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from . import config, decode, degrade, io_atd, io_idsse, model

SEED = 2027


def main():
    if "--trial" in sys.argv:
        config.set_trial(sys.argv[sys.argv.index("--trial") + 1])
    out = os.path.join(io_atd.ROOT, "evaluation", "results", "trial{}.csv".format(config.TRIAL))
    bar = io_atd.load_match()
    profile = degrade.atd_gap_profile(bar)
    with open(os.path.join(io_atd.CACHE_DIR, "decoder_t{}.json".format(config.TRIAL))) as handle:
        setting = json.load(handle)

    xs, ys = [], []
    for i, m in enumerate(io_idsse.available()):
        X, y = model.training_frames(io_idsse.load_match(m), profile, SEED + i)
        xs.append(X)
        ys.append(y)
    clf = model.fit(pd.concat(xs, ignore_index=True), np.concatenate(ys))

    os.makedirs(io_atd.MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(io_atd.MODELS_DIR, "trees_t{}.joblib".format(config.TRIAL)), compress=3)

    prob = model.predict(clf, bar)
    np.savez(os.path.join(io_atd.CACHE_DIR, "bar_prob_t{}.npz".format(config.TRIAL)), prob=prob)
    codes = decode.decode(prob, bar["halves"], **setting)
    labels = np.array(["H", "A", "D", "X"])[codes]

    with open(out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))

    g = bar["graded"]
    seg = labels[g]
    starts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
    runs = seg[starts]
    lens = np.diff(np.r_[starts, len(seg)])
    poss = lens[(runs == "H") | (runs == "A")]
    print("decoder", setting)
    print("graded shares  H {:.1%}  A {:.1%}  D {:.1%}".format(
        *[(seg == k).mean() for k in "HAD"]))
    print("possessions {}  mean {:.1f}s  median {:.1f}s".format(
        len(poss), poss.mean() / 25, np.median(poss) / 25))
    print("dead stretches {}  median {:.1f}s".format(
        int((runs == "D").sum()), np.median(lens[runs == "D"]) / 25))
    print("wrote", os.path.relpath(out, io_atd.ROOT))


if __name__ == "__main__":
    sys.exit(main())
