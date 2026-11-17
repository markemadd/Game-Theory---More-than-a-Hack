"""Fuse match-only signals into a trial's frame probabilities, then decode.

    python -m pipeline.run_fused --trial 3 --signals broadcast     # trial3_broadcast.csv

Signals: broadcast (pipeline/broadcast.py). Likelihoods are EM-estimated on the provided match
from the model's own probabilities (pipeline/fuse.py); decoder costs are the trial's, tuned on
the external matches.
"""

import argparse
import json
import os

import numpy as np

from . import broadcast, config, decode, fuse, io_atd, pose_cues, smart_tagging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, default=config.LATEST)
    ap.add_argument("--signals", default="broadcast")
    ap.add_argument("--ensemble", action="store_true",
                    help="use the tree/GRU mix and decoder chosen by pipeline.ensemble")
    ap.add_argument("--name", default=None, help="output file stem")
    ap.add_argument("--settings-from", type=int, default=None,
                    help="reuse the decoder/ensemble setting chosen for another trial")
    args = ap.parse_args()
    config.set_trial(args.trial)
    bar = io_atd.load_match()
    g = bar["graded"]
    ap_settings = args.settings_from or args.trial
    if args.ensemble:
        with open(os.path.join(io_atd.CACHE_DIR, "ensemble_t{}.json".format(ap_settings))) as handle:
            ens = json.load(handle)
        prob = np.load(os.path.join(io_atd.CACHE_DIR, "bar_prob_seq_t{}.npz".format(args.trial)))["prob"]
        if ens["gru_weight"] < 1:
            tree = np.load(os.path.join(io_atd.CACHE_DIR, "bar_prob_t{}.npz".format(args.trial)))["prob"]
            prob = (1 - ens["gru_weight"]) * tree + ens["gru_weight"] * prob
        setting = ens["decoder"]
        print("ensemble: GRU weight", ens["gru_weight"], "decoder", setting,
              "(settings from trial {})".format(ap_settings))
    else:
        prob = np.load(os.path.join(io_atd.CACHE_DIR, "bar_prob_t{}.npz".format(args.trial)))["prob"]
        with open(os.path.join(io_atd.CACHE_DIR, "decoder_t{}.json".format(ap_settings))) as handle:
            setting = json.load(handle)

    names = [n for n in args.signals.split(",") if n]
    for name in names:
        if name == "broadcast":
            obs = fuse.broadcast_obs(broadcast.build())
            n_cat = 3
        elif name == "pose":
            obs = pose_cues.observations()
            n_cat = 3
        elif name == "smarttag":
            obs = smart_tagging.observations(len(prob))
            table = smart_tagging.em_likelihoods(prob, obs, g)
            print("smarttag: P(obs|H,A,D)\n", table.round(3))
            prob = smart_tagging.apply(prob, obs, table)
            continue
        else:
            raise ValueError("unknown signal " + name)
        table = fuse.em_likelihoods(prob, obs, n_cat, g)
        print("{}: P(obs|live) {}  P(obs|dead) {}".format(
            name, table[0].round(3), table[1].round(3)))
        prob = fuse.apply(prob, obs, table)

    codes = decode.decode(prob, bar["halves"], min_run=25 if args.trial >= 6 else 0, **setting)
    labels = np.array(["H", "A", "D", "X"])[codes]
    out = os.path.join(io_atd.ROOT, "evaluation", "results",
                       (args.name or "trial{}_{}".format(args.trial, "_".join(names))) + ".csv")
    with open(out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    seg = labels[g]
    print("graded shares H {:.1%} A {:.1%} D {:.1%}".format(*[(seg == k).mean() for k in "HAD"]))
    print("wrote", os.path.relpath(out, io_atd.ROOT))


if __name__ == "__main__":
    main()
