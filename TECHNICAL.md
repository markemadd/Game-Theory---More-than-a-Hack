# Game Theory — technical document

Possession labels (H / A / D) for every frame of BAR-ATH, produced by a model trained only on
open data from other matches, plus a ball detector and a broadcast signal that use the provided
match without any human label. Code: `pipeline/`. Full experiment log, including everything
that failed: `pipeline/TRIALS.md`.

## Approach

The provided tracking (ATD) has no possession label, and hand labels on the game are not
allowed. So the model is trained on other matches that do carry per-frame possession, after
making their tracking look like ours.

1. **External training data.** Seven open Bundesliga matches (IDSSE, see *Training data*), with
   per-frame ball possession and ball status.
2. **Relabel to the challenge definition.** DFL possession flips on any touch; CHALLENGE.md says
   a touch without control is not a change. Team runs shorter than 25 frames (1 s) are absorbed
   into the surrounding possession.
3. **Degrade the external tracking to look like the ATD.** Every statistic is measured on the ATD
   itself, never on ground truth: the ball is hidden in long blocks with the ATD's own gap
   lengths; the ball is smoothed until touch rates match; players are dropped to the ATD's
   average of 20.5 tracked; and during stoppages a still or off-pitch ball is hidden or carried
   by a player, because the ATD loses the ball there while DFL keeps reporting it.
4. **Features** (129 per frame, `features.py`): ball position, speed, visibility and gap timing;
   nearest players per team; touch events; team shape and speed; stoppage and restart geometry
   (distance to lines, stillness, players around the ball, kick-off shape). The pitch is flipped
   so the same team always attacks the same way, and every match is used from both teams'
   perspectives, so the models cannot learn which club is which.
5. **Two frame models.** Gradient-boosted trees on single frames, and a bidirectional GRU that
   reads the features at 5 Hz. The GRU is trained with a loss weighted towards short
   possessions: plain cross-entropy counts frames, while the ranking metric counts possessions.
   It also has a second output trained to predict where a new possession starts; that output
   is not used at inference, but learning it makes the H/A/D probabilities better.
   Final probabilities are 0.75 GRU + 0.25 trees.
6. **Our own ball detector** (`ball_detector.py`). The ATD has no ball on 40.7% of graded frames.
   A public pre-trained YOLO11s was fine-tuned on the tactical video with labels produced by
   agreement of two algorithms (see *Training data*), then run on the gap frames. Detections are
   projected to the pitch with the provided homography and accepted only if a neighbouring
   sample agrees. Ball visibility on graded frames rises from 59.3% to 75.1%.
7. **Broadcast signal** (`broadcast.py`, `fuse.py`). The broadcast ATD is aligned to the tactical
   one from player positions alone (two constant offsets, players 0.27 m apart). Each frame gets
   one of: TV live, TV live but showing another moment, TV away (replay, close-up). Its
   likelihood under live and dead play is estimated by EM on the match from the model's own
   probabilities, then multiplied in. TV away turns out about 10 times more likely in dead ball.
8. **Decoder** (`decode.py`). A 3-state HMM decoded with Viterbi at 5 Hz, change points snapped
   to the best frame at 25 Hz, H/A runs under 25 frames absorbed. Costs were chosen on the
   external matches by spellF1. The file never contains `X` on a graded frame.

Why this rather than rules on the tracking: nearest-player rules fragment badly (official
baseline spellF1 about 0.2 on our external setting) and have no notion of dead ball when the
ball is missing, which is most of the stoppage time in this ATD.

## Assets used

| asset | use |
|---|---|
| tactical ATD | all tracking features |
| tactical video | ball detector (training crops and gap inference) |
| tactical homography | pixel <-> pitch for the detector |
| broadcast ATD + broadcast homography `valid` flag | broadcast live/away signal |

Tried and dropped: pose cues from the tactical video (throw-in posture and raised arms from a
public pose model; real but changed 0.23% of frames); the broadcast ATD's ball as a gap filler
(covers 9% of the remaining gaps and is noisier than ours); the referee's whistle from the
tactical video's audio (dominated by crowd noise). Smart Tagging was fused like the broadcast (team per phase, set-piece restarts, likelihoods by EM): its team evidence is coherent, but its ~20 s phases merge short possessions, so it was dropped. A duration-aware (semi-Markov) decoder matched but did not beat the fixed-cost one (external spellF1 0.743 both): most missed possessions are absent from the probabilities, not lost in decoding. Shirt colours at the ball (a home/away colour classifier trained on the tracking's own team assignment, 98.5% on held-out minutes) mostly repeat what the nearest tracked player already says, and nearest-to-the-ball is often the pressing defender, so it adds little the features do not already have. Both of these signals exist only for the provided match, so the external matches cannot score them; they were dropped for these structural reasons, and their public-window numbers in `pipeline/TRIALS.md` are recorded as a check only. An adaptive decoder switch cost (lowered on a match whose probabilities are less sharp, see *Limitations*) helped in one training seed and not in a second, so it is not used. Not used: broadcast video.

## Training data

| dataset | source | labels | amount |
|---|---|---|---|
| IDSSE / DFL open tracking | Bassek et al., *Scientific Data* 2025, CC-BY 4.0, fetched by `pipeline/fetch_idsse.sh` | per-frame `BallPossession` and `BallStatus` from the league's official data collection; we labelled nothing | 7 full matches (2 Bundesliga, 5 2. Bundesliga) |
| ball crops from the provided tactical video | produced by `ball_detector.py label_agree` | **algorithmic only**: a crop is kept where the public pre-trained YOLO11s ("sports ball") finds a ball within 25 px of the ATD ball projected through the homography; the box is the detector's own | 3,008 ball crops + 1,430 background crops; every 5th minute held out |

No person labelled, corrected or selected anything on the provided game. No ground truth of the
provided game was used for training or tuning. The public 5-minute window was used only as a
pipeline check after each trial; decoder costs, the model mix and every design choice were
selected on the external matches. Pre-trained weights: Ultralytics `yolo11s.pt` (COCO).

## Possession rules implemented

- **Change of possession** needs control: an opposing run shorter than 1 s (25 frames) does not
  count, in the training labels and again in the output.
- **Loose balls and duels** have no state of their own. The decoder keeps the previous team
  until the evidence for the other team outweighs the switching cost, so a contested ball stays
  with the last team in control.
- **Dead ball** follows the external `BallStatus`: from the moment play stops to the restart.
  Evidence comes from the tracking (ball leaving the pitch or disappearing, players walking,
  restart shapes) and from the broadcast going to replays.
- **Ball not visible**: the label comes from player behaviour and context on both sides of the
  gap (the GRU is bidirectional); nothing is left as `X` inside the graded halves.

## Results

Leave-one-match-out on the seven external matches, each degraded like the ATD:

| | spellF1 | macroF1 |
|---|---|---|
| official baseline (smooth50) | 0.21 | 0.45 |
| trees + decoder | 0.647 | 0.910 |
| GRU + decoder (ATD ball only, trial 3) | 0.703 | 0.916 |
| GRU + decoder (with the filled-ball gap profile, trial 5) | 0.736 | 0.929 |
| weighted GRU 0.75 + trees 0.25 (trial 6) | 0.743 | 0.931 |
| **submitted: same, GRU also trained to predict possession starts (trial 7)** | **0.753** | **0.933** |

Retrained with a second seed, trial 6 scores 0.741 / 0.930 and trial 7 0.745 / 0.934: trial 7 is ahead with both
seeds (+0.010 and +0.004 spellF1) and at every decoder setting tried, so the choice does not rest on one lucky run.

Public 5-minute window (17 possessions, a check only, never used to choose): spellF1 0.671, macroF1
0.922, frame accuracy 93.2%, 17 possessions found, dead-ball recall 0.82 at precision 0.97. The
evaluation tool's bootstrap puts the noise band on this window at about 0.33 spellF1.

## Limitations

- **Short possessions.** On the external matches the model finds 94% of possessions longer than
  15 s but about a quarter of those under 2 s; it predicts roughly 10% fewer possessions than
  there are. The loss weighting helped only a little.
- **The external validation cannot see transfer errors.** Train and test matches are degraded
  the same way, so a mismatch between our degradation and the real ATD is invisible there. The
  only check on the real match is 5 public minutes, whose confidence interval is very wide.
- **Possession definitions differ.** DFL's possession and this competition's ground truth are
  not guaranteed to draw the line in the same place for duels and restarts.
- **The ball is still missing on 25% of graded frames**, mostly during stoppages and long aerial
  balls. The detector was trained on labels that inherit the ATD's own blind spots.
- **Stoppages with the ball in play** (fouls, offside) are the dead-ball frames still missed.
- **Fewer possessions on the provided match than in validation.** Without any labels: the output
  has 2.9 possessions per minute on the provided match against 3.4 on the external matches (true
  3.2-4.2). Before decoding, the probabilities change team almost as often as on the external
  matches, but more of those changes are sub-second flickers that the decoder removes. Part of
  this may be real (long Barcelona spells); part is a transfer gap the external validation cannot
  see. A label-free adaptive switch cost did not hold up across training seeds, so the decoder
  is left as validated.
- With more time: a possession-level loss or a segment model for short spells, a detector
  trained on external football video, and a whistle detector robust to crowd noise.

## Reproduction

Python 3.12, ffmpeg not required. Developed on an Apple M1 Pro (PyTorch MPS).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-pipeline.txt
# unpack the data bundle into data/ as described in DATA.md, then:
.venv/bin/python -m pipeline.make_submission --out submission.csv
.venv/bin/python evaluation/validate_submission.py submission.csv
```

`make_submission` uses the trained weights in `models/` and trains nothing: it runs the ball
detector on the gap frames of the tactical video (the slow step), aligns the broadcast, predicts,
fuses and decodes. Seeds are fixed and dependencies pinned. Re-running it on the development
machine reproduces the submitted file byte for byte; on other hardware the detector's floating
point results can differ in the last digits, which may move a small number of frames.

To retrain everything from scratch (several hours):

```bash
bash pipeline/fetch_idsse.sh                                   # external matches, ~2.6 GB
.venv/bin/python -m pipeline.ball_detector label_agree         # algorithmic ball labels
.venv/bin/python -m pipeline.ball_detector train --base yolo11s.pt --epochs 20
.venv/bin/python -m pipeline.ball_detector fill --weights cache/ball/runs/ball_agree/weights/best.pt
.venv/bin/python -m pipeline.cv --trial 7                      # trees: validation (same trees as trial 6)
.venv/bin/python -m pipeline.seq_model --trial 7               # GRU: validation
.venv/bin/python -m pipeline.ensemble --trial 7                # mix + decoder -> cache/ensemble_t7.json
.venv/bin/python -m pipeline.run_trial --trial 6               # trees on all matches -> models/trees_t6.joblib
.venv/bin/python -m pipeline.seq_model --trial 7 --final       # GRU on all matches -> models/gru_t7.pt
# models/settings_t7.json = cache/ensemble_t7.json plus "trees": "trees_t6.joblib"
cp cache/ball/runs/ball_agree/weights/best.pt models/ball_detector.pt
```
