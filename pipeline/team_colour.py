"""Team colours at the ball, read from the tactical video. Labels by algorithm only.

The ATD knows which team every tracked player belongs to, so a shirt-colour classifier can be
trained with no human input: project each tracked player's feet into the image through the
homography, take the torso pixels above them, and learn home vs away from their colours. It is
then applied where the tracking cannot help: to whoever is standing at the ball, tracked or not.

Nothing here is ever viewed by a person; only aggregate statistics are printed.

    python -m pipeline.team_colour calibrate   # player height in the image, from pixel profiles
    python -m pipeline.team_colour train       # torso colour classifier, held-out accuracy
    python -m pipeline.team_colour read        # cache/team_colour.npz: colour at the ball
"""

import os
import sys

import cv2
import joblib
import numpy as np

from . import ball_detector as bd
from . import io_atd

lh = bd.lh
OUT = os.path.join(io_atd.CACHE_DIR, "team_colour.npz")
MODEL = os.path.join(io_atd.CACHE_DIR, "team_colour_clf.joblib")
GEOM = os.path.join(io_atd.CACHE_DIR, "team_colour_geom.npz")
HUE_BINS = 18


def grass_mask(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 30) & (h <= 90) & (s >= 40) & (v >= 30)


def scale_px(Hm, nx, ny, length):
    """Image pixels per metre along the ground at a pitch point."""
    x0, y0 = lh.project(Hm, nx, ny)
    x1, y1 = lh.project(Hm, nx + 1.0 / length, ny)
    x2, y2 = lh.project(Hm, nx, ny + 1.0 / length)
    return float(np.mean([np.hypot(x1 - x0, y1 - y0), np.hypot(x2 - x0, y2 - y0)]))


def players_px(match, Hm, f):
    """[(px, py, scale, team)] for tracked players at frame f (team 0 home, 1 away)."""
    length, width = match["pitch"]
    out = []
    for team, key in ((0, "home"), (1, "away")):
        p = match[key][f]
        ok = np.isfinite(p[:, 0])
        for x, y in p[ok]:
            nx, ny = x / length + 0.5, y / width + 0.5
            px, py = lh.project(Hm, nx, ny)
            out.append((float(px), float(py), scale_px(Hm, nx, ny, length), team, x, y))
    return out


def torso(frame_hsv, frame_lab, px, py, s, geom):
    """Colour feature of the torso box above a ground point; None if off-image or mostly grass."""
    lo, hi, half_w = geom
    y0, y1 = int(py - hi * s), int(py - lo * s)
    x0, x1 = int(px - half_w * s), int(px + half_w * s) + 1
    H, W = frame_hsv.shape[:2]
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W or y1 - y0 < 3 or x1 - x0 < 2:
        return None
    hsv = frame_hsv[y0:y1, x0:x1].reshape(-1, 3)
    lab = frame_lab[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    keep = ~grass_mask(hsv)
    if keep.mean() < 0.3:
        return None
    hsv, lab = hsv[keep], lab[keep]
    hist = np.bincount((hsv[:, 0].astype(int) * HUE_BINS) // 180, weights=hsv[:, 1] / 255.0,
                       minlength=HUE_BINS)
    hist = hist / max(hist.sum(), 1e-6)
    return np.concatenate([lab.mean(0) / 255.0, lab.std(0) / 255.0, np.percentile(lab[:, 0], [20, 80]) / 255.0,
                           [np.mean(hsv[:, 1] < 50), np.mean(hsv[:, 2] < 60), keep.mean()], hist]).astype(np.float32)


def calibrate(n_frames=300):
    """Player height in units of the ground scale, from the non-grass profile above isolated
    tracked players. Prints the profile; saves the torso geometry."""
    match = io_atd.load_match(raw_ball=True)
    Hs, _, valid = lh.load()
    frames = np.flatnonzero(match["graded"] & valid)
    frames = frames[np.linspace(0, len(frames) - 1, n_frames).astype(int)]
    offs = np.arange(0.0, 4.01, 0.1)
    prof, cnt = np.zeros(len(offs)), 0
    for i, frame in bd.frames_at(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ps = players_px(match, Hs[i], i)
        xy = np.array([(p[4], p[5]) for p in ps])
        for k, (px, py, s, _, _, _) in enumerate(ps):
            d = np.hypot(xy[:, 0] - xy[k, 0], xy[:, 1] - xy[k, 1])
            if len(d) < 2 or np.sort(d)[1] < 4.0:
                continue
            col = []
            for o in offs:
                y = int(py - o * s)
                xa, xb = int(px - 0.2 * s), int(px + 0.2 * s) + 1
                if y < 0 or y >= hsv.shape[0] or xa < 0 or xb > hsv.shape[1]:
                    break
                col.append(1 - grass_mask(hsv[y:y + 1, xa:xb]).mean())
            if len(col) == len(offs):
                prof += col
                cnt += 1
    prof /= max(cnt, 1)
    top = offs[np.flatnonzero(prof > 0.5 * prof.max())[-1]]
    print("isolated players {} | non-grass fraction by height (ground-scale units):".format(cnt))
    print(" ".join("{:.1f}:{:.2f}".format(o, p) for o, p in zip(offs[::3], prof[::3])))
    print("estimated player height {:.2f} units".format(top))
    geom = np.array([0.50 * top, 0.80 * top, 0.12 * top])
    np.savez(GEOM, geom=geom, profile=prof, offs=offs)
    return geom


def collect(every=50):
    """Torso features of tracked players on sampled frames, with team and minute."""
    match = io_atd.load_match(raw_ball=True)
    geom = np.load(GEOM)["geom"]
    Hs, _, valid = lh.load()
    frames = np.flatnonzero(match["graded"] & valid)
    frames = frames[frames % every == 0]
    X, y, minute = [], [], []
    for i, frame in bd.frames_at(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        ps = players_px(match, Hs[i], i)
        xy = np.array([(p[4], p[5]) for p in ps])
        for k, (px, py, s, team, _, _) in enumerate(ps):
            d = np.hypot(xy[:, 0] - xy[k, 0], xy[:, 1] - xy[k, 1])
            if len(d) > 1 and np.sort(d)[1] < 1.5:     # overlapping players: the box would mix shirts
                continue
            f = torso(hsv, lab, px, py, s, geom)
            if f is not None:
                X.append(f)
                y.append(team)
                minute.append(i // 1500)
    return np.array(X), np.array(y), np.array(minute)


def train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y, minute = collect()
    test = minute % 5 == 0          # every 5th minute held out, as for the ball detector
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    clf.fit(X[~test], y[~test])
    p = clf.predict_proba(X[test])[:, 1]
    acc = ((p > 0.5) == y[test]).mean()
    conf = np.abs(p - 0.5) > 0.4
    print("torso samples {} (train {}, held-out minutes {}) | home share {:.2f}".format(
        len(y), int((~test).sum()), int(test.sum()), 1 - y.mean()))
    print("held-out accuracy {:.3f} | confident (p<0.1 or >0.9) on {:.1%} of players, accuracy there {:.3f}".format(
        acc, conf.mean(), ((p[conf] > 0.5) == y[test][conf]).mean()))
    clf.fit(X, y)
    joblib.dump(clf, MODEL)


def read(every=5, radius_m=2.0, step_m=0.5, sigma_m=1.0):
    """Colour at the ball on every `every`-th graded frame with a ball (filled track).

    Candidate ground points on a grid around the ball; a point whose torso box is mostly
    non-grass is a person. Each person-point is classified home/away and weighted by its distance
    to the ball. Saved per frame: home and away colour weight, distance to the nearest person,
    and that person's home probability; plus, for comparison, the team of the nearest *tracked*
    player within radius_m."""
    from .config import set_trial
    set_trial(6)
    clf = joblib.load(MODEL)
    geom = np.load(GEOM)["geom"]
    match = io_atd.load_match()                      # filled ball (trial 6)
    length, width = match["pitch"]
    Hs, _, valid = lh.load()
    ball = match["ball"]
    frames = np.flatnonzero(match["graded"] & valid & np.isfinite(ball[:, 0]))
    frames = frames[frames % every == 0]
    n = len(ball)
    w_home = np.full(n, np.nan, np.float32)
    w_away = np.full(n, np.nan, np.float32)
    near_d = np.full(n, np.nan, np.float32)
    near_p = np.full(n, np.nan, np.float32)
    atd_team = np.full(n, -1, np.int8)
    g = np.arange(-radius_m, radius_m + 1e-6, step_m)
    gx, gy = np.meshgrid(g, g)
    keep = np.hypot(gx, gy) <= radius_m
    gx, gy = gx[keep], gy[keep]
    gd = np.hypot(gx, gy)
    for k, (i, frame) in enumerate(bd.frames_at(frames)):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        bx, by = ball[i]
        feats, dist = [], []
        for x, y, d in zip(bx + gx, by + gy, gd):
            nx, ny = x / length + 0.5, y / width + 0.5
            px, py = lh.project(Hs[i], nx, ny)
            f = torso(hsv, lab, float(px), float(py), scale_px(Hs[i], nx, ny, length), geom)
            if f is not None:
                feats.append(f)
                dist.append(d)
        if feats:
            p_away = clf.predict_proba(np.array(feats))[:, 1]
            dist = np.array(dist)
            w = np.exp(-0.5 * (dist / sigma_m) ** 2)
            w_home[i], w_away[i] = float((w * (1 - p_away)).sum()), float((w * p_away).sum())
            j = int(np.argmin(dist))
            near_d[i], near_p[i] = dist[j], 1 - p_away[j]
        else:
            w_home[i] = w_away[i] = 0.0
        best = (radius_m, -1)
        for team, key in ((0, "home"), (1, "away")):
            p = match[key][i]
            ok = np.isfinite(p[:, 0])
            if ok.any():
                d = np.hypot(p[ok, 0] - bx, p[ok, 1] - by).min()
                if d < best[0]:
                    best = (d, team)
        atd_team[i] = best[1]
        if k % 5000 == 0:
            print("  {}/{} frames".format(k, len(frames)), flush=True)
    np.savez(OUT, w_home=w_home, w_away=w_away, near_d=near_d, near_p=near_p, atd_team=atd_team,
             every=every)
    s = np.isfinite(w_home)
    tot = w_home[s] + w_away[s]
    print("sampled {} | a person at the ball on {:.1%} | clear colour (>80% one team) on {:.1%}".format(
        int(s.sum()), (tot > 0.3).mean(), ((tot > 0.3) & (np.maximum(w_home[s], w_away[s]) > 0.8 * tot)).mean()))


N_CAT = 4  # nobody at the ball, home colour, away colour, mixed


def observations(min_mass=0.3, clear=0.8):
    """(n,) category per sampled frame (-1 elsewhere), and the sampling step."""
    z = np.load(OUT)
    wh, wa = z["w_home"], z["w_away"]
    obs = np.full(len(wh), -1, np.int64)
    s = np.isfinite(wh)
    tot = wh + wa
    obs[s] = 3
    obs[s & (tot < min_mass)] = 0
    obs[s & (tot >= min_mass) & (wh >= clear * tot)] = 1
    obs[s & (tot >= min_mass) & (wa >= clear * tot)] = 2
    return obs, int(z["every"])


def em_likelihoods(prior, obs, graded, n_cat=N_CAT, iters=100, smooth=1.0):
    """(3, n_cat) P(obs | H, A, D), EM on the match from the model's own probabilities."""
    m = graded & (obs >= 0)
    p, o = prior[m].astype(np.float64), obs[m]
    table = np.full((3, n_cat), 1.0 / n_cat)
    for _ in range(iters):
        post = p * table[:, o].T
        post /= post.sum(1, keepdims=True)
        new = np.stack([np.bincount(o, weights=post[:, k], minlength=n_cat) for k in range(3)]) + smooth
        new /= new.sum(1, keepdims=True)
        done = np.abs(new - table).max() < 1e-7
        table = new
        if done:
            break
    return table


def apply(prior, obs, table, every, weight=1.0):
    """Each sample is spread over its `every` frames with weight/every, so it counts once."""
    held = np.full(len(obs), -1, np.int64)
    s = np.flatnonzero(obs >= 0)
    for off in range(-(every // 2), every - every // 2):
        idx = np.clip(s + off, 0, len(obs) - 1)
        held[idx] = obs[s]
    ok = held >= 0
    out = prior.astype(np.float64).copy()
    out[ok] *= table[:, held[ok]].T ** (weight / every)
    out /= out.sum(1, keepdims=True)
    return out.astype(np.float32)


if __name__ == "__main__":
    {"calibrate": calibrate, "train": train, "read": read}[sys.argv[1]]()
