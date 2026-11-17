"""Provided data -> the submitted possession file, with the trained weights in models/.

    python -m pipeline.make_submission                    # evaluation/results/submission.csv
    python -m pipeline.make_submission --out my_team.csv

Steps (nothing is trained here; see README-pipeline.md for the training commands):
  1. fill the ATD's ball gaps with our detector on the tactical video   (models/ball_detector.pt)
  2. align the broadcast ATD to the tactical one -> "TV is away from live play" signal
  3. H/A/D probabilities per frame from the trees and the GRU           (models/trees_t6.joblib,
                                                                         models/gru_t8_s*.pt)
  4. mix them, fuse the broadcast signal, decode possessions, write the CSV

Step 1 decodes about 13,000 video frames and is the slow part; its result is cached in
cache/ball/, so a second run takes a few minutes. Delete cache/ to redo everything.
"""

import argparse
import json
import os

import joblib
import numpy as np

from . import broadcast, config, decode, fuse, io_atd, model, seq_model

TRIAL = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(io_atd.ROOT, "evaluation", "results", "submission.csv"))
    ap.add_argument("--settings", help="settings JSON (default: models/settings_t<TRIAL>.json)")
    args = ap.parse_args()
    config.set_trial(TRIAL)
    models = io_atd.MODELS_DIR

    filled = os.path.join(io_atd.CACHE_DIR, "ball", "ball_filled.npz")
    if not os.path.isfile(filled):
        from . import ball_detector
        print("1/4 ball detector on the ATD's ball gaps", flush=True)
        ball_detector.fill(os.path.join(models, "ball_detector.pt"))
    print("2/4 broadcast alignment", flush=True)
    signals = broadcast.build()

    print("3/4 frame probabilities", flush=True)
    bar = io_atd.load_match()
    with open(args.settings or os.path.join(models, "settings_t{}.json".format(TRIAL))) as handle:
        settings = json.load(handle)
    prob = seq_model.predict_all(seq_model.load_all(), bar)  # averaged over the training seeds
    w = settings["gru_weight"]
    if w < 1:
        # trial 7 changed only the GRU, so it reuses trial 6's trees (same features, same fit)
        trees = joblib.load(os.path.join(models, settings.get("trees", "trees_t{}.joblib".format(TRIAL))))
        prob = (1 - w) * model.predict(trees, bar) + w * prob

    print("4/4 fuse, decode, write", flush=True)
    obs = fuse.broadcast_obs(signals)
    prob = fuse.apply(prob, obs, fuse.em_likelihoods(prob, obs, 3, bar["graded"]))
    costs = dict(settings["decoder"])
    if "adaptive" in settings:
        a = settings["adaptive"]
        costs["c_switch"], shares = decode.adaptive_switch(prob, bar["halves"], a["target"], a["floor"],
                                                           **costs)
        print("adaptive switch cost {} (kept share {:.3f}, target {:.3f})".format(
            costs["c_switch"], shares[costs["c_switch"]], a["target"]))
    codes = decode.decode(prob, bar["halves"], min_run=25, **costs)
    labels = np.array(["H", "A", "D", "X"])[codes]
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    seg = labels[bar["graded"]]
    print("graded shares H {:.1%} A {:.1%} D {:.1%}".format(*[(seg == k).mean() for k in "HAD"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
