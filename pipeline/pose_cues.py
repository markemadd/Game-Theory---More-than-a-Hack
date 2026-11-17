"""Dead-ball cues from the image, with a public pre-trained pose model (no labels at all).

Per sampled frame of the tactical video:
  throwin   people with both wrists above the head whose feet are within 2.5 m of a touchline
  arms_up   people with at least one wrist above the head anywhere (referee signal, appeal)
  people    people detected (sanity / normaliser)

Feet are projected to the pitch with the provided homography. The counts become one
categorical observation per frame and are fused like the broadcast (pipeline/fuse.py): their
likelihoods under live and dead play are EM-estimated on the match from the model's own
probabilities, so no person decides any label.

    python -m pipeline.pose_cues run [--every 10]   # cache/pose_cues.npz
"""

import argparse
import os
import sys
import time

import numpy as np

from . import ball_detector as bd
from . import io_atd

lh = bd.lh
NOSE, L_SH, R_SH, L_WR, R_WR, L_AN, R_AN = 0, 5, 6, 9, 10, 15, 16
KP_CONF = 0.35
OUT = os.path.join(io_atd.CACHE_DIR, "pose_cues.npz")


def people_in_frame(model, frame):
    """[(feet_px_x, feet_px_y, both_up, any_up)] deduplicated across overlapping tiles."""
    tiles = bd.tiles(frame.shape[1], frame.shape[0])
    crops = [frame[y:y + bd.TILE, x:x + bd.TILE] for x, y in tiles]
    # Players are ~70 px tall; the public pose model only finds them upscaled 2x (verified:
    # 0 people per frame at 640 or full-frame 1920, 7-25 at 1280 with ankle conf ~0.95).
    res = model.predict(crops, imgsz=2 * bd.TILE, conf=0.3, verbose=False, device=bd._device())
    people = []
    for (x0, y0), r in zip(tiles, res):
        if r.keypoints is None or r.keypoints.data is None:
            continue
        for kp in r.keypoints.data.cpu().numpy():
            xy, c = kp[:, :2] + (x0, y0), kp[:, 2]
            ank = [k for k in (L_AN, R_AN) if c[k] > KP_CONF]
            if not ank:
                continue
            feet = xy[ank].mean(0)
            head_y = xy[NOSE, 1] if c[NOSE] > KP_CONF else (
                xy[[L_SH, R_SH], 1].min() - 15 if min(c[L_SH], c[R_SH]) > KP_CONF else None)
            if head_y is None:
                continue
            up = [c[w] > KP_CONF and xy[w, 1] < head_y for w in (L_WR, R_WR)]
            people.append((feet[0], feet[1], all(up), any(up)))
    dedup = []
    for p in people:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 20 for q in dedup):
            dedup.append(p)
    return dedup


def run(every=10, weights="yolo11s-pose.pt"):
    from ultralytics import YOLO
    model = YOLO(os.path.join(bd.OUT, weights) if os.path.isfile(os.path.join(bd.OUT, weights))
                 else weights)
    match = io_atd.load_match(raw_ball=True)
    length, width = match["pitch"]
    H, _, valid = lh.load()
    frames = np.flatnonzero(match["graded"] & valid)
    frames = frames[frames % every == 0]
    n = len(match["ball"])
    throwin = np.full(n, np.nan, np.float32)
    arms = np.full(n, np.nan, np.float32)
    count = np.full(n, np.nan, np.float32)
    t0 = time.time()
    for k, (i, frame) in enumerate(bd.frames_at(frames)):
        ps = people_in_frame(model, frame)
        t = a = 0
        for fx, fy, both, anyup in ps:
            nx, ny = lh.unproject(H[i], fx, fy)
            y_m = (ny - 0.5) * width
            near_touch = abs(abs(y_m) - width / 2) < 2.5
            t += both and near_touch
            a += anyup
        throwin[i], arms[i], count[i] = t, a, len(ps)
        if k % 1000 == 0:
            print("  {}/{} frames, {:.0f}s".format(k, len(frames), time.time() - t0), flush=True)
    np.savez(OUT, throwin=throwin, arms_up=arms, people=count, every=every)
    g = match["graded"] & np.isfinite(count)
    print("sampled {} frames | people/frame {:.1f} | frames with throw-in posture {:.1%} | "
          "with arms up {:.1%}".format(int(g.sum()), count[g].mean(), (throwin[g] > 0).mean(),
                                      (arms[g] > 0).mean()))


def observations(every=None):
    """Categorical obs per frame: 0 none, 1 arms up (no throw-in posture), 2 throw-in posture,
    -1 not sampled. Each sample is held for the frames around it."""
    z = np.load(OUT)
    step = int(z["every"]) if every is None else every
    n = len(z["people"])
    base = np.full(n, -1, np.int64)
    s = np.flatnonzero(np.isfinite(z["people"]))
    cat = np.where(z["throwin"][s] > 0, 2, np.where(z["arms_up"][s] > 0, 1, 0))
    for off in range(-(step // 2), step - step // 2):
        idx = np.clip(s + off, 0, n - 1)
        base[idx] = cat
    return base


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    run(args.every)
