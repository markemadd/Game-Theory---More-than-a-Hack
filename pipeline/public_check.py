"""Public 5-minute window: per-possession diagnostics for one or more submissions.

A pipeline check, not a tuning set (RULES.md). It prints every true segment next to what each
submission said over the same frames, so errors can be read rather than just counted.

    python -m pipeline.public_check trial1 trial2
"""

import os
import sys

import numpy as np
import pandas as pd

from .io_atd import ROOT

GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def segments(labels, lo, hi):
    seg = labels[lo:hi]
    starts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
    stops = np.r_[starts[1:], len(seg)]
    return [(lo + a, lo + b, seg[a]) for a, b in zip(starts, stops)]


def summary(pred, a, b):
    """Majority label and the run pattern of a prediction inside [a, b)."""
    runs = segments(pred, a, b)
    pattern = " ".join("{}{:.1f}".format(lab, (y - x) / 25) for x, y, lab in runs)
    major = pd.Series(pred[a:b]).value_counts().idxmax()
    return major, pattern


def main(names):
    gt = pd.read_csv(GT)["label"].to_numpy()
    graded = np.flatnonzero(gt != "X")
    lo, hi = graded[0], graded[-1] + 1
    preds = {n: pd.read_csv(os.path.join(ROOT, "evaluation", "results", n + ".csv"))["label"]
             .to_numpy() for n in names}
    print("{:>6} {:>6} {:>5} {:>3} | ".format("start", "end", "secs", "gt")
          + " | ".join("{:<34}".format(n) for n in names))
    for a, b, lab in segments(gt, lo, hi):
        cells = []
        for n in names:
            major, pattern = summary(preds[n], a, b)
            mark = " " if major == lab else "*"
            cells.append("{}{:<33}".format(mark, pattern[:33]))
        print("{:>6} {:>6} {:>5.1f} {:>3} | ".format(a, b, (b - a) / 25, lab) + " | ".join(cells))
    for n in names:
        p = preds[n][lo:hi]
        g = gt[lo:hi]
        print("{}: frame accuracy {:.3f} | D recall {:.3f} D precision {:.3f}".format(
            n, (p == g).mean(), ((p == "D") & (g == "D")).sum() / max(1, (g == "D").sum()),
            ((p == "D") & (g == "D")).sum() / max(1, (p == "D").sum())))


if __name__ == "__main__":
    main(sys.argv[1:] or ["trial1"])
