"""Per-trial settings, so every earlier submission stays reproducible from the same code.

Select with the MTAH_TRIAL environment variable or `--trial N` on the scripts (default: latest).
"""

import os

TRIALS = {
    1: dict(ball_sigma=5.0, ball_noise_m=0.3, touch_dv=4.0, restart_features=False),
    # Trial 2: ball smoothing fitted to the ATD's whole touch-rate curve (1-4 m/s thresholds),
    # touch threshold 2.5 m/s where both datasets agree (~11 touches per visible-ball minute),
    # plus stoppage/restart features.
    2: dict(ball_sigma=7.0, ball_noise_m=0.3, touch_dv=2.5, restart_features=True),
    # Trial 3: DFL's dead-time ball made to behave like the ATD's (still and off-pitch balls
    # hidden or carried), calibrated to the ATD's own shares: still 2.6%, off-pitch 2.5%,
    # ball visible 59.3% (DFL after degradation: 2.4%, 2.0%, 58.7%).
    3: dict(ball_sigma=7.0, ball_noise_m=0.3, touch_dv=2.5, restart_features=True,
            dead_ball_behaviour=True, still_mask=0.1, oob_mask=0.4),
    # Trial 5: trial 3 + the provided match's ball gaps filled by our fine-tuned detector
    # (pipeline/ball_detector.py). The degradation reads its gap profile from the filled track.
    5: dict(ball_sigma=7.0, ball_noise_m=0.3, touch_dv=2.5, restart_features=True,
            dead_ball_behaviour=True, still_mask=0.1, oob_mask=0.4, filled_ball=True),
    # Trial 6: trial 5 with the sequence model trained twice as long and its loss weighted
    # towards short possessions (pipeline/seq_model.py: spell_weights). Trees are unchanged.
    6: dict(ball_sigma=7.0, ball_noise_m=0.3, touch_dv=2.5, restart_features=True,
            dead_ball_behaviour=True, still_mask=0.1, oob_mask=0.4, filled_ball=True,
            seq_epochs=8, seq_spell_power=0.5),
}
LATEST = max(TRIALS)

TRIAL = int(os.environ.get("MTAH_TRIAL", LATEST))
CFG = dict(TRIALS[TRIAL])


def set_trial(n):
    global TRIAL
    TRIAL = int(n)
    CFG.clear()
    CFG.update(TRIALS[TRIAL])
