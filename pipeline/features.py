"""Team-level, provider-agnostic frame features.

Everything is computed from "team 1" (T1) vs "team 2" (T2) perspective, with x flipped per half so
T1 always attacks +x. Calling build(match, swap=True) swaps the teams, which is how training is
augmented and how prediction is symmetrised: the model never learns anything about a club.

Evidence families:
  ball      visibility, position, speed, out of pitch, gap timing (offline: before and after)
  control   nearest-player distances per team, players within 2/5/10 m, signed control score
  touches   sudden ball-velocity changes assigned to the nearest team; last/next touch team
  shape     team centroids, spread, speed, share of walking players (stoppage signature)
  context   centred rolling means of the above at 0.5 s, 2 s and 5 s
"""

import numpy as np
import pandas as pd

from .config import CFG

FPS = 25
TOUCH_RADIUS = 2.0  # m, nearest player must be this close
WINDOWS = (12, 50, 125)


def _nan_rolling_mean(x, half):
    s = pd.Series(x)
    return s.rolling(2 * half + 1, center=True, min_periods=1).mean().to_numpy()


def _interp_short(x, max_gap):
    s = pd.Series(x)
    return s.interpolate(limit=max_gap, limit_area="inside").to_numpy()


def _attack_sign(match, t1, t2):
    """+1 where T1 attacks +x in that half: T1's average x is the lower one."""
    sign = np.ones(len(match["ball"]), np.float32)
    for lo, hi in match["halves"]:
        c1 = np.nanmean(t1[lo:hi, :, 0])
        c2 = np.nanmean(t2[lo:hi, :, 0])
        sign[lo:hi] = 1.0 if c1 < c2 else -1.0
    return sign


def _team_stats(p, sign, prefix):
    """Shape and movement of one team. p: (n, K, 2) metres."""
    x = p[..., 0] * sign[:, None]
    y = p[..., 1]
    valid = ~np.isnan(x)
    cnt = valid.sum(1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"), \
            np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        cx = np.nanmean(x, 1)
        cy = np.nanmean(y, 1)
        sx = np.nanstd(x, 1)
        sy = np.nanstd(y, 1)
        last_line = np.nanmin(x, 1)
        # per-slot speed from 0.2 s differences; slots are tracklets so NaN breaks are fine
        vx = (np.roll(p[..., 0], -2, 0) - np.roll(p[..., 0], 2, 0)) * FPS / 4
        vy = (np.roll(p[..., 1], -2, 0) - np.roll(p[..., 1], 2, 0)) * FPS / 4
        sp = np.hypot(vx, vy)
        sp[sp > 12] = np.nan
        mean_sp = np.nanmean(sp, 1)
        slow = np.nanmean((sp < 1.0).astype(np.float32) + np.where(np.isnan(sp), np.nan, 0), 1)
        fast = np.nanmean((sp > 4.0).astype(np.float32) + np.where(np.isnan(sp), np.nan, 0), 1)
    cvx = np.gradient(_nan_rolling_mean(cx, 5)) * FPS
    return {
        prefix + "cnt": cnt, prefix + "cx": cx, prefix + "cy_abs": np.abs(cy),
        prefix + "sx": sx, prefix + "sy": sy, prefix + "area": sx * sy,
        prefix + "line": last_line, prefix + "speed": mean_sp,
        prefix + "slow": slow, prefix + "fast": fast, prefix + "cvx": cvx,
    }


def _nearest(ball, p):
    d = np.hypot(p[..., 0] - ball[:, None, 0], p[..., 1] - ball[:, None, 1])
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        nearest = np.nanmin(d, 1)
    within = {r: np.nansum(d < r, 1).astype(np.float32) for r in (2.0, 5.0, 10.0)}
    nearest[np.isnan(ball[:, 0])] = np.nan
    return nearest, within


def _gap_timing(vis):
    """Frames since the ball was last seen and until it is next seen (0 when visible)."""
    n = len(vis)
    idx = np.arange(n)
    last = np.where(vis, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(vis, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.where(last >= 0, idx - last, 10 * FPS).astype(np.float32)
    until = np.where(nxt < n, nxt - idx, 10 * FPS).astype(np.float32)
    return np.minimum(since, 30 * FPS), np.minimum(until, 30 * FPS)


def _carry(values, vis, direction):
    """Value at the last (direction=+1) or next (-1) visible frame."""
    s = pd.Series(np.where(vis, values, np.nan))
    return (s.ffill() if direction > 0 else s.bfill()).to_numpy()


def _touches(ball_m, d1, d2, graded):
    """Touch frames and their team (+1 T1, -1 T2) from ball-velocity change peaks."""
    x = _interp_short(ball_m[:, 0].astype(np.float64), 5)
    y = _interp_short(ball_m[:, 1].astype(np.float64), 5)
    x = _nan_rolling_mean(x, 1)
    y = _nan_rolling_mean(y, 1)
    k = 4
    vbx = (x - np.roll(x, k)) * FPS / k
    vby = (y - np.roll(y, k)) * FPS / k
    vax = (np.roll(x, -k) - x) * FPS / k
    vay = (np.roll(y, -k) - y) * FPS / k
    dv = np.nan_to_num(np.hypot(vax - vbx, vay - vby), nan=0.0)
    dv[dv > 40] = 0.0  # teleports are tracker jumps, not kicks
    peak = dv == pd.Series(dv).rolling(13, center=True, min_periods=1).max().to_numpy()
    n1 = pd.Series(d1).rolling(5, center=True, min_periods=1).min().to_numpy()
    n2 = pd.Series(d2).rolling(5, center=True, min_periods=1).min().to_numpy()
    near = np.fmin(n1, n2)
    cand = peak & (dv >= CFG["touch_dv"]) & graded & (near <= TOUCH_RADIUS)
    team = np.where(np.nan_to_num(n1, nan=99) <= np.nan_to_num(n2, nan=99), 1.0, -1.0)
    return np.flatnonzero(cand), team, dv


def _touch_context(idx, team, n):
    """Signed team and frames since the last touch / until the next touch."""
    t_last = np.zeros(n, np.float32)
    t_next = np.zeros(n, np.float32)
    since = np.full(n, 30 * FPS, np.float32)
    until = np.full(n, 30 * FPS, np.float32)
    if len(idx):
        frames = np.arange(n)
        j = np.searchsorted(idx, frames, side="right") - 1
        ok = j >= 0
        t_last[ok] = team[idx[j[ok]]]
        since[ok] = np.minimum(frames[ok] - idx[j[ok]], 30 * FPS)
        j2 = np.searchsorted(idx, frames, side="left")
        ok2 = j2 < len(idx)
        t_next[ok2] = team[idx[j2[ok2]]]
        until[ok2] = np.minimum(idx[j2[ok2]] - frames[ok2], 30 * FPS)
    # touch balance over the last/next 5 s: who is playing the ball
    ev = np.zeros(n, np.float32)
    if len(idx):
        ev[idx] = team[idx]
    cs = np.r_[0, np.cumsum(ev)]
    frames = np.arange(n)
    past = cs[frames + 1] - cs[np.maximum(0, frames - 125)]
    future = cs[np.minimum(n, frames + 126)] - cs[frames]
    return t_last, since, t_next, until, past, future


def _restart_features(ball, t1, t2, sign, length, width, vis, f):
    """Stoppage and restart signatures: ball near a line and still, opponents kept 9.15 m away,
    kickoff shape, players at the touchline, play slowing down or speeding up."""
    r = {}
    to_touch = width / 2 - np.abs(ball[:, 1])
    to_goal = length / 2 - np.abs(ball[:, 0])
    r["to_touch"] = to_touch
    r["to_goal"] = to_goal
    r["to_line"] = np.fmin(to_touch, to_goal)
    r["to_line_last_seen"] = _carry(r["to_line"], vis, +1)
    r["to_line_next_seen"] = _carry(r["to_line"], vis, -1)
    bsp = pd.Series(f["bspeed"])
    r["bspeed_med1"] = bsp.rolling(25, center=True, min_periods=5).median().to_numpy()
    r["bspeed_max2"] = bsp.rolling(50, center=True, min_periods=5).max().to_numpy()
    for name, p in (("t1", t1), ("t2", t2)):
        d = np.hypot(p[..., 0] - ball[:, None, 0], p[..., 1] - ball[:, None, 1])
        r[name + "_w9"] = np.where(vis, np.nansum(d < 9.15, 1), np.nan).astype(np.float32)
        x = p[..., 0] * sign[:, None]
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            r[name + "_own_half"] = np.nanmean(np.where(np.isnan(x), np.nan, x < 0), 1)
            r[name + "_at_touchline"] = np.nansum(np.abs(p[..., 1]) > width / 2 - 1.5, 1)
            r[name + "_at_goalline"] = np.nansum(np.abs(p[..., 0]) > length / 2 - 1.5, 1)
    r["kickoff_shape"] = r["t1_own_half"] + (1 - r["t2_own_half"])
    sp = pd.Series(f["all_speed"])
    r["speed_past3"] = sp.rolling(75, min_periods=1).mean().to_numpy()
    r["speed_fut3"] = sp[::-1].rolling(75, min_periods=1).mean().to_numpy()[::-1]
    r["speed_trend"] = r["speed_fut3"] - r["speed_past3"]
    for name in ("all_slow", "all_speed", "vis", "oob"):
        r[name + "_r250"] = _nan_rolling_mean(f[name], 250)
    return r


def build(match, swap=False):
    """DataFrame of features for every frame of a match (T1 = home unless swap)."""
    t1, t2 = (match["away"], match["home"]) if swap else (match["home"], match["away"])
    ball = match["ball"]
    n = len(ball)
    length, width = match["pitch"]
    sign = _attack_sign(match, t1, t2)
    vis = ~np.isnan(ball[:, 0])

    f = {}
    bx = ball[:, 0] * sign
    f["vis"] = vis.astype(np.float32)
    f["bx"] = bx
    f["by_abs"] = np.abs(ball[:, 1])
    f["oob"] = ((np.abs(ball[:, 0]) > length / 2 + 0.5)
                | (np.abs(ball[:, 1]) > width / 2 + 0.5)).astype(np.float32)
    f["oob"][~vis] = np.nan
    bxs = _nan_rolling_mean(_interp_short(ball[:, 0].astype(np.float64), 3), 2)
    bys = _nan_rolling_mean(_interp_short(ball[:, 1].astype(np.float64), 3), 2)
    bsp = np.hypot(np.gradient(bxs), np.gradient(bys)) * FPS
    bsp[bsp > 40] = np.nan
    f["bspeed"] = bsp
    since, until = _gap_timing(vis)
    f["gap_since"] = since / FPS
    f["gap_until"] = until / FPS
    f["gap_len"] = (since + until) / FPS

    d1, w1 = _nearest(ball, t1)
    d2, w2 = _nearest(ball, t2)
    f["d1"], f["d2"] = d1, d2
    f["ddiff"] = d2 - d1
    for r in (2.0, 5.0, 10.0):
        f["w1_%d" % r] = np.where(vis, w1[r], np.nan)
        f["w2_%d" % r] = np.where(vis, w2[r], np.nan)
    ctrl = np.tanh((d2 - d1) / 1.5) * np.exp(-np.fmin(d1, d2) / 4.0)
    ctrl = np.nan_to_num(ctrl, nan=0.0).astype(np.float32)
    f["ctrl"] = ctrl
    f["ctrl_last_seen"] = _carry(ctrl, vis, +1)
    f["ctrl_next_seen"] = _carry(ctrl, vis, -1)
    f["bx_last_seen"] = _carry(bx, vis, +1)
    f["bx_next_seen"] = _carry(bx, vis, -1)
    f["oob_last_seen"] = _carry(np.nan_to_num(f["oob"]), vis, +1)
    f["oob_next_seen"] = _carry(np.nan_to_num(f["oob"]), vis, -1)

    idx, team_at, dv = _touches(ball, d1, d2, match["graded"])
    t_last, t_since, t_next, t_until, t_past, t_future = _touch_context(idx, team_at, n)
    f["dv"] = dv
    f["touch_last"], f["touch_since"] = t_last, t_since / FPS
    f["touch_next"], f["touch_until"] = t_next, t_until / FPS
    f["touch_past5"], f["touch_next5"] = t_past, t_future

    f.update(_team_stats(t1, sign, "t1_"))
    f.update(_team_stats(t2, sign, "t2_"))
    f["cx_diff"] = f["t1_cx"] - f["t2_cx"]
    f["area_ratio"] = np.log((f["t1_area"] + 1) / (f["t2_area"] + 1))
    f["all_slow"] = (f["t1_slow"] + f["t2_slow"]) / 2
    f["all_speed"] = (f["t1_speed"] + f["t2_speed"]) / 2

    rolled = ("vis", "ctrl", "bspeed", "oob", "all_slow", "all_speed", "t1_speed", "t2_speed",
              "cx_diff", "area_ratio", "t1_cvx", "t2_cvx", "ddiff", "dv")
    for w in WINDOWS:
        for name in rolled:
            f["%s_r%d" % (name, w)] = _nan_rolling_mean(f[name], w)
    # direction-aware: how play evolves (2 s back vs 2 s ahead)
    for name in ("ctrl", "all_slow", "vis"):
        s = pd.Series(f[name])
        f[name + "_past2"] = s.rolling(50, min_periods=1).mean().to_numpy()
        f[name + "_fut2"] = s[::-1].rolling(50, min_periods=1).mean().to_numpy()[::-1]
    if CFG["restart_features"]:
        f.update(_restart_features(ball, t1, t2, sign, length, width, vis, f))
    f["half_pos"] = np.zeros(n, np.float32)
    for lo, hi in match["halves"]:
        f["half_pos"][lo:hi] = (np.arange(hi - lo) / FPS / 60).astype(np.float32)

    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
