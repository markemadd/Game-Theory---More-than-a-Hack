"""Our own ball detector on the tactical video, to fill the ATD's ball gaps.

Everything here runs locally. Labels are produced by an algorithm, never by a person
(RULES.md: "pseudo-labels produced by an algorithm, provided no person decided any of them"):
the ATD's own visible ball, projected into the image through the provided homography.

    python -m pipeline.ball_detector eval  [--weights yolo11s.pt]   # recall vs the ATD ball
    python -m pipeline.ball_detector label                           # pseudo-label crops
    python -m pipeline.ball_detector train                           # fine-tune on the crops
    python -m pipeline.ball_detector fill  --weights <best.pt>       # detect in the gaps

Outputs live in cache/ball/ (gitignored): crops, runs/, and ball_filled.npz.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from . import io_atd

sys.path.insert(0, os.path.join(io_atd.ROOT, "starter"))
import load_homography as lh  # noqa: E402

OUT = os.path.join(io_atd.CACHE_DIR, "ball")
os.makedirs(OUT, exist_ok=True)
SPORTS_BALL = 32       # COCO class id in the public pre-trained weights
TILE = 640
HIT_PX = 15            # a detection within this many pixels of the projected ATD ball is a hit
BALL_M = 0.22


def video_path():
    from games import asset_paths
    return asset_paths()["video"]


def ball_pixels(match, H, valid):
    """ATD ball projected into tactical pixels (NaN where missing or no homography)."""
    length, width = match["pitch"]
    nx = match["ball"][:, 0] / length + 0.5
    ny = match["ball"][:, 1] / width + 0.5
    px = np.full(len(nx), np.nan)
    py = np.full(len(nx), np.nan)
    ok = np.isfinite(nx) & valid
    for i in np.flatnonzero(ok):
        px[i], py[i] = lh.project(H[i], nx[i], ny[i])
    return px, py


def ball_size_px(H, nx, ny, length):
    """Apparent ball diameter in pixels at a pitch point (local homography scale)."""
    x0, y0 = lh.project(H, nx, ny)
    x1, y1 = lh.project(H, nx + 1.0 / length, ny)
    return max(6.0, float(np.hypot(x1 - x0, y1 - y0)) * BALL_M)


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return 0 if torch.cuda.is_available() else "cpu"


def tiles(width=1920, height=1080, size=TILE, overlap=64):
    xs = list(range(0, width - size, size - overlap)) + [width - size]
    ys = list(range(0, height - size, size - overlap)) + [height - size]
    return [(x, y) for y in ys for x in xs]


def detect(model, frame, conf=0.05, classes=None):
    """Tiled inference -> list of (cx, cy, conf) in full-frame pixels."""
    crops = [frame[y:y + TILE, x:x + TILE] for x, y in tiles(frame.shape[1], frame.shape[0])]
    res = model.predict(crops, imgsz=TILE, conf=conf, classes=classes, verbose=False, device=_device())
    out = []
    for (x, y), r in zip(tiles(frame.shape[1], frame.shape[0]), res):
        for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append((x + b[0], y + b[1], float(c)))
    return out


def frames_at(indices):
    """Yield (index, BGR frame) for sorted frame indices, decoding sequentially."""
    cap = cv2.VideoCapture(video_path())
    want = iter(sorted(indices))
    target = next(want, None)
    i = 0
    while target is not None:
        ok = cap.grab()
        if not ok:
            break
        if i == target:
            _, frame = cap.retrieve()
            yield i, frame
            target = next(want, None)
        i += 1
    cap.release()


def check_video():
    cap = cv2.VideoCapture(video_path())
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def evaluate(weights, n_frames=400, seed=0, classes=(SPORTS_BALL,), frame_pool=None):
    """Recall/precision of the detector against the ATD ball on frames where the ATD sees it."""
    from ultralytics import YOLO
    model = YOLO(weights)
    match = io_atd.load_match(raw_ball=True)
    H, _, valid = lh.load()
    px, py = ball_pixels(match, H, valid)
    pool = np.flatnonzero(match["graded"] & np.isfinite(px)) if frame_pool is None else frame_pool
    pick = np.sort(np.random.default_rng(seed).choice(pool, n_frames, replace=False))
    hits, top_hits, n_det = 0, 0, 0
    for i, frame in frames_at(pick):
        dets = detect(model, frame, classes=list(classes) if classes else None)
        n_det += len(dets)
        if not dets:
            continue
        d = np.array(dets)
        dist = np.hypot(d[:, 0] - px[i], d[:, 1] - py[i])
        hits += bool((dist < HIT_PX).any())
        top_hits += bool(dist[np.argmax(d[:, 2])] < HIT_PX)
    print("frames {} | any detection within {} px: {:.1%} | top-confidence detection is the "
          "ball: {:.1%} | detections per frame {:.2f}".format(
              len(pick), HIT_PX, hits / len(pick), top_hits / len(pick), n_det / len(pick)))


def held_out(frames):
    """Every 5th minute of the match is never used for training; detector metrics use it."""
    return (np.asarray(frames) // 1500) % 5 == 0


def label(n_frames=5000, seed=0):
    """Pseudo-labelled 640 px crops from frames where the ATD sees the ball (training minutes)."""
    match = io_atd.load_match(raw_ball=True)
    length = match["pitch"][0]
    H, _, valid = lh.load()
    px, py = ball_pixels(match, H, valid)
    width, height = 1920, 1080
    inside = np.isfinite(px) & (px > 4) & (px < width - 4) & (py > 4) & (py < height - 4)
    pool = np.flatnonzero(match["graded"] & inside & ~held_out(np.arange(len(px))))
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(pool, min(n_frames, len(pool)), replace=False))
    root = os.path.join(OUT, "dataset")
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
    n_pos = n_neg = 0
    for i, frame in frames_at(pick):
        split = "val" if rng.random() < 0.1 else "train"
        nx = match["ball"][i, 0] / length + 0.5
        ny = match["ball"][i, 1] / match["pitch"][1] + 0.5
        size = 1.6 * ball_size_px(H[i], nx, ny, length)
        x0 = int(np.clip(px[i] - rng.integers(48, TILE - 48), 0, width - TILE))
        y0 = int(np.clip(py[i] - rng.integers(48, TILE - 48), 0, height - TILE))
        name = "f{:06d}".format(i)
        cv2.imwrite(os.path.join(root, "images", split, name + ".jpg"),
                    frame[y0:y0 + TILE, x0:x0 + TILE], [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(root, "labels", split, name + ".txt"), "w") as f:
            f.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                (px[i] - x0) / TILE, (py[i] - y0) / TILE, size / TILE, size / TILE))
        n_pos += 1
        if rng.random() < 0.5:  # background tile away from the ball
            for _ in range(10):
                bx = int(rng.integers(0, width - TILE))
                by = int(rng.integers(0, height - TILE))
                if not (bx - 100 < px[i] < bx + TILE + 100 and by - 100 < py[i] < by + TILE + 100):
                    cv2.imwrite(os.path.join(root, "images", split, name + "_bg.jpg"),
                                frame[by:by + TILE, bx:bx + TILE], [cv2.IMWRITE_JPEG_QUALITY, 92])
                    open(os.path.join(root, "labels", split, name + "_bg.txt"), "w").close()
                    n_neg += 1
                    break
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n".format(root))
    print("pseudo-labelled crops: {} with ball, {} background".format(n_pos, n_neg))


AGREE_PX = 25


def label_agree(n_frames=6000, seed=0, weights="yolo11s.pt", conf=0.05):
    """Labels where two algorithms agree: the ATD ball (projected) and the pre-trained detector.

    A crop is kept only if the public detector finds a sports ball within AGREE_PX of the
    projected ATD ball; the box is the detector's own box, so it sits on the ball in the image
    even where the ATD position is smoothed or the ball is in the air. Background crops are kept
    only if they are far from the ATD ball and the detector sees no confident ball in them.
    Training minutes only (held_out() minutes are never labelled).
    """
    from ultralytics import YOLO
    model = YOLO(os.path.join(OUT, weights) if os.path.isfile(os.path.join(OUT, weights)) else weights)
    match = io_atd.load_match(raw_ball=True)
    H, _, valid = lh.load()
    px, py = ball_pixels(match, H, valid)
    width, height = 1920, 1080
    inside = np.isfinite(px) & (px > 4) & (px < width - 4) & (py > 4) & (py < height - 4)
    pool = np.flatnonzero(match["graded"] & inside & ~held_out(np.arange(len(px))))
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(pool, min(n_frames, len(pool)), replace=False))
    root = os.path.join(OUT, "dataset_agree")
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
    n_try = n_pos = n_neg = 0
    offsets, sizes = [], []
    for i, frame in frames_at(pick):
        n_try += 1
        x0 = int(np.clip(px[i] - rng.integers(64, TILE - 64), 0, width - TILE))
        y0 = int(np.clip(py[i] - rng.integers(64, TILE - 64), 0, height - TILE))
        crop = frame[y0:y0 + TILE, x0:x0 + TILE]
        r = model.predict(crop, imgsz=TILE, conf=conf, classes=[SPORTS_BALL], verbose=False,
                          device=_device())[0]
        boxes = r.boxes.xywh.cpu().numpy()
        if not len(boxes):
            continue
        d = np.hypot(boxes[:, 0] + x0 - px[i], boxes[:, 1] + y0 - py[i])
        k = int(np.argmin(d))
        if d[k] > AGREE_PX:
            continue
        bx, by, bw, bh = boxes[k]
        split = "val" if rng.random() < 0.1 else "train"
        name = "f{:06d}".format(i)
        cv2.imwrite(os.path.join(root, "images", split, name + ".jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(root, "labels", split, name + ".txt"), "w") as f:
            f.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(bx / TILE, by / TILE, bw / TILE, bh / TILE))
        n_pos += 1
        offsets.append(d[k])
        sizes.append(max(bw, bh))
        if rng.random() < 0.5:
            for _ in range(10):
                gx = int(rng.integers(0, width - TILE))
                gy = int(rng.integers(0, height - TILE))
                if gx - 150 < px[i] < gx + TILE + 150 and gy - 150 < py[i] < gy + TILE + 150:
                    continue
                bg = frame[gy:gy + TILE, gx:gx + TILE]
                rb = model.predict(bg, imgsz=TILE, conf=0.3, classes=[SPORTS_BALL], verbose=False,
                                   device=_device())[0]
                if len(rb.boxes):
                    continue
                cv2.imwrite(os.path.join(root, "images", split, name + "_bg.jpg"), bg,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                open(os.path.join(root, "labels", split, name + "_bg.txt"), "w").close()
                n_neg += 1
                break
        if n_try % 1000 == 0:
            print("  {} frames tried, {} agreed".format(n_try, n_pos), flush=True)
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n".format(root))
    offsets, sizes = np.array(offsets), np.array(sizes)
    print("frames tried {} | agreed {} ({:.1%}) | background {} | detector-vs-ATD offset median "
          "{:.1f} px | box size median {:.1f} px (p10 {:.1f}, p90 {:.1f})".format(
              n_try, n_pos, n_pos / max(n_try, 1), n_neg, np.median(offsets), np.median(sizes),
              np.percentile(sizes, 10), np.percentile(sizes, 90)))


def train(base="yolo11s.pt", epochs=25, dataset="dataset_agree"):
    from ultralytics import YOLO
    model = YOLO(os.path.join(OUT, base) if os.path.isfile(os.path.join(OUT, base)) else base)
    model.train(data=os.path.join(OUT, dataset, "data.yaml"), imgsz=TILE, epochs=epochs,
                batch=16, device=_device(), project=os.path.join(OUT, "runs"), name="ball_agree",
                exist_ok=True, seed=0, deterministic=True, workers=4, plots=False,
                mosaic=0.5, degrees=0, fliplr=0.5, close_mosaic=5, patience=8)


def to_pitch(H, cx, cy, match):
    nx, ny = lh.unproject(H, cx, cy)
    return (nx - 0.5) * match["pitch"][0], (ny - 0.5) * match["pitch"][1]


def eval_pitch(weights, n_frames=400, conf=0.25, seed=1):
    """Held-out minutes: does the top detection land on the ATD ball, in metres on the pitch?"""
    from ultralytics import YOLO
    model = YOLO(weights)
    match = io_atd.load_match(raw_ball=True)
    H, _, valid = lh.load()
    seen = match["graded"] & np.isfinite(match["ball"][:, 0]) & valid
    pool = np.flatnonzero(seen & held_out(np.arange(len(seen))))
    pick = np.sort(np.random.default_rng(seed).choice(pool, n_frames, replace=False))
    found, close = 0, []
    for i, frame in frames_at(pick):
        dets = [d for d in detect(model, frame, conf=conf) if d[2] >= conf]
        if not dets:
            continue
        cx, cy, _ = max(dets, key=lambda d: d[2])
        x, y = to_pitch(H[i], cx, cy, match)
        found += 1
        close.append(np.hypot(x - match["ball"][i, 0], y - match["ball"][i, 1]))
    close = np.array(close)
    print("held-out frames {} | detection made {:.1%} | of those within 1.5 m of the ATD ball "
          "{:.1%} (median error {:.2f} m)".format(
              len(pick), found / len(pick), (close < 1.5).mean() if len(close) else 0,
              np.median(close) if len(close) else float("nan")))


def fill(weights, conf=0.25, every=5, max_jump_mps=35.0):
    """Detect the ball where the ATD lost it; keep detections that link into a plausible track."""
    from ultralytics import YOLO
    model = YOLO(weights)
    match = io_atd.load_match(raw_ball=True)
    H, _, valid = lh.load()
    gap = match["graded"] & np.isnan(match["ball"][:, 0]) & valid
    frames = np.flatnonzero(gap)
    frames = frames[frames % every == 0]
    det = np.full((len(match["ball"]), 3), np.nan, np.float32)
    for k, (i, frame) in enumerate(frames_at(frames)):
        dets = [d for d in detect(model, frame, conf=conf) if d[2] >= conf]
        if dets:
            cx, cy, c = max(dets, key=lambda d: d[2])
            x, y = to_pitch(H[i], cx, cy, match)
            det[i] = (x, y, c)
        if k % 2000 == 0:
            print("  {}/{} frames".format(k, len(frames)), flush=True)
    np.savez(os.path.join(OUT, "raw_detections.npz"), det=det)
    return link(match, det, every, max_jump_mps)


def link(match, det, every=5, max_jump_mps=35.0):
    """Accept a detection only if a neighbouring sample (detection or ATD ball) agrees with it."""
    length, width = match["pitch"]
    ball = match["ball"].astype(np.float64)
    known = np.where(np.isfinite(ball[:, :1]), ball, np.nan)
    x, y = det[:, 0].astype(np.float64), det[:, 1].astype(np.float64)
    inside = (np.abs(x) < length / 2 + 2) & (np.abs(y) < width / 2 + 2)
    idx = np.flatnonzero(np.isfinite(x) & inside)
    keep = np.zeros(len(x), bool)
    for j, i in enumerate(idx):
        for nb in (idx[j - 1] if j else None, idx[j + 1] if j + 1 < len(idx) else None):
            if nb is not None and abs(nb - i) <= 3 * every:
                if np.hypot(x[nb] - x[i], y[nb] - y[i]) <= max_jump_mps * abs(nb - i) / 25:
                    keep[i] = True
        for nb in (i - 1, i + 1):
            if 0 <= nb < len(x) and np.isfinite(known[nb, 0]):
                keep[i] |= np.hypot(known[nb, 0] - x[i], known[nb, 1] - y[i]) <= 3.0
    filled = ball.copy()
    filled[keep, 0], filled[keep, 1] = x[keep], y[keep]
    # bridge accepted samples that are at most 2 sampling steps apart
    for a, b in zip(np.flatnonzero(keep)[:-1], np.flatnonzero(keep)[1:]):
        if b - a <= 2 * every and np.isnan(filled[a + 1:b, 0]).all():
            t = (np.arange(a + 1, b) - a) / (b - a)
            filled[a + 1:b] = filled[a] + t[:, None] * (filled[b] - filled[a])
    g = match["graded"]
    print("ball visible on graded frames: ATD {:.1%} -> filled {:.1%} ({} detections accepted "
          "of {})".format(np.isfinite(ball[g, 0]).mean(), np.isfinite(filled[g, 0]).mean(),
                          int(keep.sum()), len(idx)))
    np.savez(os.path.join(OUT, "ball_filled.npz"), ball=filled.astype(np.float32), accepted=keep)
    return filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "eval", "label", "label_agree", "train", "evalpitch", "fill"))
    ap.add_argument("--base", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.cmd == "check":
        print("frames {} fps {} size {}x{}".format(*check_video()))
    elif args.cmd == "eval":
        evaluate(args.weights, args.n)
    elif args.cmd == "label":
        label()
    elif args.cmd == "label_agree":
        label_agree()
    elif args.cmd == "train":
        train(args.base, args.epochs)
    elif args.cmd == "evalpitch":
        eval_pitch(args.weights, args.n, args.conf)
    else:
        fill(args.weights, args.conf)


if __name__ == "__main__":
    main()
