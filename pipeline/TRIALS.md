# Trials: learned frame model + HMM decoding

Settings per trial live in `pipeline/config.py`; every trial regenerates byte for byte with
`--trial N`.

# Trial 1

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-pipeline.txt
bash pipeline/fetch_idsse.sh                  # 7 open DFL matches, ~2.6 GB, into ext_data/idsse/
.venv/bin/python -m pipeline.cv --trial 1     # out-of-fold validation + decoder setting (~25 min)
.venv/bin/python -m pipeline.run_trial --trial 1  # writes evaluation/results/trial1.csv (~4 min)
.venv/bin/python -m pipeline.baseline_external --trial 1  # stage-by-stage reference
```

Data the match needs: `data/tactical/BAR-ATH_tactical_atd.{xml,txt}` only. Smart Tagging, the
homography, the videos and the broadcast feed are not used yet.

## Method

1. **External training data.** The seven IDSSE matches (Bassek et al., *Sci Data* 2025, CC-BY 4.0)
   carry per-frame `BallPossession` and `BallStatus` from DFL's own pipeline: the same kind of
   owner/alive label our ground truth is derived from. No human labels on the provided match.
2. **Relabel to the challenge definition.** DFL possession flips on any touch. CHALLENGE.md says a
   touch without control is not a change, so team runs under 1 s are absorbed into the surrounding
   possession (`degrade.relabel`, `MIN_CONTROL = 25`). Fixed by the documented rule, not fitted.
3. **Degrade the external data to look like our ATD**, with every statistic measured on the ATD
   itself (no ground truth):
   - ball lost in long blocks, run lengths bootstrapped from the ATD (median gap ~10.7 s);
   - a random 0-60% of gaps anchored to stoppages;
   - ball smoothed (Gaussian sigma 5 frames + low-frequency noise) to match the ATD's
     acceleration profile; the ATD ball is heavily filtered;
   - players dropped until ~20.5 tracked objects per frame.
4. **Team-level features** (`features.py`, 106 per frame): ball visibility/position/speed/out of
   pitch and gap timing; nearest-player distances and counts per team; touch events from
   ball-velocity peaks; team centroid, spread, speed and share of walking players; rolling context
   at 0.5/2/5 s. Pitch flipped so team 1 always attacks +x.
5. **Frame model.** Gradient-boosted trees (`HistGradientBoostingClassifier`) on H/A/D. Each
   training match is featurised from both teams' perspectives with swapped labels, and prediction
   averages both, so the model has no notion of which club is which.
6. **Decoder.** 3-state HMM, Viterbi at 5 Hz, change points snapped to the best frame at 25 Hz.
   Switch costs chosen by mean spellF1 over the seven out-of-fold matches: H<->A 6, into/out of
   dead ball 24 (log-probability units). No `X` on graded frames.

## Results

Out-of-fold on the seven external matches (each held out, degraded with a fixed seed):

| | spellF1 | macroF1 | possessions (truth 364 avg) |
|---|---|---|---|
| official baseline (smooth50) | 0.206 | 0.453 | 430 |
| frame model, argmax | 0.347 | 0.896 | 834 |
| **frame model + decoder** | **0.611** | **0.898** | 290 |

The baseline scores 0.21 here and 0.24 on the real full match, so the degraded external setting
is a plausible proxy for the real one. Per-match spellF1 0.590-0.637 (sd 0.02).

On BAR-ATH: 350 possessions (mean 10.7 s), 145 dead stretches, dead share 41.3% of graded
frames (the external matches run 41-52%).

Public 5-minute window, **pipeline check only, nothing tuned on it**: spellF1 0.659, macroF1
0.756, 17 predicted possessions against 17 true.

## Known gaps

- Touch detection fires 1.7/min on the ATD vs 4.8/min on degraded DFL: the ATD ball is smoother
  still. Touch features transfer imperfectly.
- External possession is under-segmented (290 vs 364): the decoder's costs are a single global
  trade-off.
- DFL "possession" and Metrica "owning team" are not guaranteed to be the same definition.

# Trial 2: touch transfer + stoppage features

Changes (all in `config.py` / `features.py`):

- **Touch transfer.** Ball smoothing re-fitted to the ATD's whole touch-rate curve at 1-4 m/s
  thresholds (sigma 7 frames + 0.3 m low-frequency noise), touch threshold 2.5 m/s, where both
  datasets give ~11 touches per visible-ball minute (trial 1: 1.7 on the ATD vs 4.8 external).
- **Stoppage/restart features** (+23): ball distance to touchline/goal line (and at last/next
  sighting), ball stillness over 1-2 s, players within 9.15 m per team, own-half share and
  kickoff shape, players at the touchline/goal line, play slowing down vs speeding up, 10 s
  context for walking share, visibility, speed and out-of-pitch.

| | external spellF1 (7 matches OOF) | external macroF1 | public spellF1 [90% CI] | public macroF1 | public D recall / precision |
|---|---|---|---|---|---|
| baseline smooth50 | 0.200 | 0.451 | 0.326 [0.21, 0.45] | 0.621 | 0.21 / 0.66 |
| trial 1 | 0.611 | 0.898 | 0.659 [0.48, 0.81] | 0.756 | 0.36 / 0.94 |
| trial 2 | 0.618 | 0.899 | 0.555 [0.36, 0.75] | 0.774 | 0.41 / 1.00 |

The two trials agree on 96.1% of graded frames of the full match. Neither the external gain
(+0.006) nor the public gap is outside its noise band: the public difference is one or two
possessions (the 8.1 s H spell at 11351 and the H/A flicker inside the dead stretch at 15889).
The external validation cannot measure transfer fixes by construction (train and test copies are
degraded the same way), so the touch change is justified by matching the ATD, not by a score.

Still missed in the public window: three of four stoppages are only partly labelled D
(12851, 15889, 18027): the frame model sees live-looking movement there.

# Trial 3: make DFL's dead-time ball behave like the ATD's

**Why.** On the external matches trial 2 already finds dead ball almost perfectly (recall 0.97,
precision 0.97), but only 41% of it in the public window. Comparing the two datasets on all
graded frames (no ground truth) showed why: player signals match closely, the ball does not.

| | ATD | DFL (degraded) | DFL dead frames |
|---|---|---|---|
| ball still (< 0.5 m/s) | 2.6% | 10.8% | 23% |
| ball outside the lines | rare | often | most |

DFL keeps reporting a still or off-pitch ball during stoppages; the ATD loses it instead. The
model had learned a dead-ball cue that never fires on the provided match.

**Change** (`degrade._atd_ball_behaviour`): still runs are hidden (10%) or carried by the
nearest player (90%); off-pitch runs are hidden (40%). Calibrated to the ATD's own shares:
still 2.4% vs 2.6%, off-pitch 2.0% vs 2.5%, visible 58.7% vs 59.3%.

External OOF: spellF1 0.611 (unchanged; this validation cannot see a transfer fix).
On BAR-ATH the dead share rises to 44.9%, inside the external 41-52% band.

## Broadcast fusion (`pipeline/broadcast.py`, `pipeline/fuse.py`)

- The broadcast is aligned to the tactical timeline from the two ATDs alone: two constant
  offsets, +4014 frames (1st half) and +26920 (2nd), matched players 0.27 m apart (median).
  98.3% of graded frames are covered; the TV is away from live play on 32% of them.
- Per tactical frame: TV live / TV live but showing another moment / TV away.
- Fused as a product of experts, with P(observation | live, dead) estimated by EM on the
  provided match from the model's own probabilities (algorithmic pseudo-labels, no human
  label). Estimated: TV away is 9.5x more likely in dead ball, TV live 0.31x.

# Trial 4: sequence expert (bidirectional GRU) + broadcast

`pipeline/seq_model.py`: 2-layer bidirectional GRU (64 units) over the same 129 features at
5 Hz, same degraded data, perspective swap and leave-one-match-out split. Deterministic on
this machine (MPS and CPU, bit-identical reruns).

External OOF, decoder re-tuned per candidate (`pipeline/ensemble.py`):

| GRU weight | frame acc | spellF1 | macroF1 |
|---|---|---|---|
| 0 (trees, trial 3) | 0.910 | 0.611 | 0.897 |
| 0.5 | 0.927 | 0.676 | 0.914 |
| 0.75 | 0.929 | 0.694 | 0.917 |
| **1.0** | 0.927 | **0.703** | 0.916 |

+0.09 spellF1 over the trees, far outside the 0.02 bar. Selected: GRU only, decoder
switch 6 / into dead 8 / out of dead 32.

## Summary so far

| | external spellF1 | public spellF1 [90% CI] | public macroF1 | public D recall / precision |
|---|---|---|---|---|
| baseline smooth50 | 0.25 | 0.326 [0.21, 0.45] | 0.621 | 0.21 / 0.66 |
| trial 1 | 0.611 | 0.659 [0.48, 0.81] | 0.756 | 0.36 / 0.94 |
| trial 2 | 0.618 | 0.555 [0.36, 0.75] | 0.774 | 0.41 / 1.00 |
| trial 3 + broadcast | 0.611 | 0.525 [0.34, 0.72] | 0.875 | 0.79 / 0.90 |
| trial 4 GRU | 0.703 | 0.688 [0.50, 0.84] | 0.875 | - |
| **trial 4 GRU + broadcast** | **0.703** | 0.667 [0.48, 0.82] | **0.908** | 0.78 / 0.96 |

```bash
.venv/bin/python -m pipeline.cv --trial 3 && .venv/bin/python -m pipeline.run_trial --trial 3
.venv/bin/python -m pipeline.seq_model --trial 3 && .venv/bin/python -m pipeline.ensemble --trial 3
.venv/bin/python -m pipeline.seq_model --trial 3 --final
.venv/bin/python -m pipeline.broadcast
.venv/bin/python -m pipeline.run_fused --trial 3 --ensemble --signals broadcast --name trial4
```

# Trial 5: our own ball detector fills the tracker's gaps

`pipeline/ball_detector.py`, all local, labels produced by algorithms only:

1. **Labels by agreement.** A training crop is kept only where the public pre-trained YOLO
   (`yolo11s`, COCO "sports ball") finds a ball within 25 px of the ATD ball projected through
   the homography; the box is the detector's own. 3,008 of 6,000 tried frames agreed (+1,430
   background crops). Every 5th minute of the match is held out and never labelled.
   (A first attempt with boxes placed on the projected ATD ball failed, mAP50 ~0.03: the
   projection is off by a median 7.8 px for an 8.5 px ball.)
2. **Fine-tune** `yolo11s` 20 epochs on 640 px crops (MPS). Validation mAP50 0.977.
3. **Held-out minutes, top detection mapped to the pitch:**

   | detector | frames with a detection | within 1.5 m of the ATD ball | median error |
   |---|---|---|---|
   | pre-trained | 87.2% | 39.5% | 4.60 m |
   | fine-tuned | 87.5% | 63.4% | 0.96 m |

4. **Gap fill** on every 5th frame where the ATD lost the ball; a detection is accepted only if
   a neighbouring sample (detection or ATD ball) agrees; accepted samples <= 10 frames apart are
   bridged. Ball visible on graded frames: **59.3% -> 75.1%**.
5. GRU retrained with the filled track (the degradation's gap profile follows it); decoder and
   ensemble settings reused from trial 3 (no new external CV before the deadline).

| | public spellF1 [90% CI] | public macroF1 | frame acc | D recall / precision |
|---|---|---|---|---|
| trial 4 | 0.667 [0.48, 0.82] | 0.908 | 0.920 | 0.78 / 0.96 |
| **trial 5** | **0.679** [0.50, 0.84] | **0.933** | **0.940** | **0.86 / 0.97** |

Trials 4 and 5 agree on 94.3% of graded frames of the full match.

```bash
.venv/bin/python -m pipeline.ball_detector label_agree
.venv/bin/python -m pipeline.ball_detector train --base yolo11s.pt --epochs 20
.venv/bin/python -m pipeline.ball_detector evalpitch --weights cache/ball/runs/ball_agree/weights/best.pt
.venv/bin/python -m pipeline.ball_detector fill --weights cache/ball/runs/ball_agree/weights/best.pt
.venv/bin/python -m pipeline.seq_model --trial 5 --final
.venv/bin/python -m pipeline.run_fused --trial 5 --ensemble --settings-from 3 --signals broadcast --name trial5
```

## Pose cues (`pipeline/pose_cues.py`): tried, not kept

Public pre-trained `yolo11s-pose` on 640 px tiles upscaled 2x (it finds no players at native
scale: ~70 px tall), one frame per second (6,354 frames, ~55 min on the M1). Per frame:
throw-in posture (both wrists above the head, feet within 2.5 m of a touchline) and arms
raised anywhere. 6.5 people per sampled frame, arms raised on 12.2%, throw-in posture on 1.1%.

EM-estimated likelihood ratio dead:live: throw-in posture 11x, arms raised 1.6x. Fused into
trial 5 it changes 369 of 158,857 graded frames (0.23%), none of them in the public window
(identical public scores). Not kept in the submission: an hour of extra GPU per regeneration
for no measurable effect.

## 16 Nov: last-day checks

**Trial 5 validated on the external matches** (leave-one-match-out, never run before): sequence
model spellF1 0.736, macroF1 0.929 (trial 3: 0.703 / 0.916); trees alone 0.647. Decoder retuned
for trial 5: switch 4, into dead 32, out of dead 32 (`cache/ensemble_t5.json`).

**Where spellF1 is lost** (trial 3 out-of-fold): 301 predicted possessions per match against 364
true ones. Recall at IoU 0.5 by possession length: under 2 s 27%, 2-5 s 57%, 5-15 s 81%, over
15 s 94%. Possessions under 5 s are half of all possessions.

**Broadcast ball as a gap filler: tried, not kept.** The broadcast ATD ball is present on 52% of
graded frames, mostly where the tactical ball is too; it covers 9% of the remaining gaps
(visibility 75.1% -> 77.4%). Where both exist it is within 3 m of the tactical ball 82% of the
time, so it is also the noisier of the two. As a cross-check, our detector's accepted fills are
within 3 m of it 73% of the time.

**Referee whistle from the audio (`pipeline/whistle.py`): tried, not kept.** Tonal-peak score in
3-4.8 kHz per frame. Tonal events are dominated by the crowd: the model's dead-ball probability
is as high before an event as after it. Narrowing to 3.6-4.0 kHz gives about 0.45 events per
minute with a faint step (P(dead) 0.43 before -> 0.53 after), too weak to move labels.

**Trial 6: sequence model trained twice as long, loss weighted towards short possessions.**
Screened separately on three held-out matches (each with its own best decoder): baseline 0.719,
weighting (power 0.5) 0.737, doubled training 0.736. Combined and validated on all seven:

| leave-one-match-out, 7 matches | spellF1 | macroF1 |
|---|---|---|
| trial 5, GRU only, decoder (4, 32, 32) | 0.736 | 0.929 |
| trial 6, GRU only, decoder (12, 8, 8) | 0.736 | 0.928 |
| trial 6, 0.75 GRU + 0.25 trees, decoder (12, 8, 16) | 0.742 | 0.931 |
| same + H/A runs under 25 frames absorbed | 0.743 | 0.931 |

The combination did not keep the screening gain: against trial 5 the mix wins spellF1 on 4 of 7
matches (paired differences -0.035 to +0.050) and macroF1 on 6 of 7. A small edge, inside the
match-to-match noise (sd 0.025).

Public window (sanity check, 17 possessions): trial5b (trial 5, own decoder) spellF1 0.679,
macroF1 0.933, identical to trial5; trial6 spellF1 0.788, macroF1 0.931. The files differ on
3.8% of graded frames.
