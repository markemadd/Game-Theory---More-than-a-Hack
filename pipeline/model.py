"""Frame model: gradient-boosted trees on team-level features, trained on degraded external matches.

Training rows come from every external match twice over: each degraded copy is featurised from
both teams' perspectives (labels swapped to match), so the model is symmetric by construction.
Prediction averages the two perspectives the same way.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import degrade, features

TRAIN_STRIDE = 4
N_COPIES = 2
MIN_CONTROL = 25  # frames; CHALLENGE.md: a touch without control is not a change of possession
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
              l2_regularization=1.0, early_stopping=False, random_state=0)

SWAP = np.array([1, 0, 2])  # label map when the perspective is swapped


def training_frames(match, profile, seed):
    """(X, y) rows from one external match: N_COPIES degradations x 2 perspectives."""
    rng = np.random.default_rng(seed)
    target = degrade.relabel(match["label"], MIN_CONTROL)
    keep = np.flatnonzero(match["graded"])[::TRAIN_STRIDE]
    xs, ys = [], []
    for _ in range(N_COPIES):
        noisy = degrade.degrade(match, profile, rng, anchor_frac=float(rng.uniform(0.0, 0.6)))
        for swap in (False, True):
            xs.append(features.build(noisy, swap=swap).iloc[keep])
            ys.append(SWAP[target[keep]] if swap else target[keep])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def fit(X, y):
    model = HistGradientBoostingClassifier(**PARAMS)
    model.fit(X, y)
    return model


def predict(model, match):
    """(n, 3) H/A/D probabilities, symmetrised over the two team perspectives."""
    p1 = model.predict_proba(features.build(match, swap=False))
    p2 = model.predict_proba(features.build(match, swap=True))[:, SWAP]
    return (p1 + p2) / 2
