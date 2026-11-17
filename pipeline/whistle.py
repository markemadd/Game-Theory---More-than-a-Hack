"""Referee whistle from the tactical video's audio track -> a dead-ball cue per frame.

A referee's whistle is a loud, narrow tone between about 3 and 4.8 kHz that lasts a few tenths
of a second. Per video frame (40 ms hop) the score is how far the strongest bin in that band
stands above the frame's broadband level, in dB. A whistle event is a run of high-score frames.
Crowd whistling is tonal too, so the cue is noisy; like the broadcast signal it is not trusted
directly but fused with likelihoods EM-estimated on the match (pipeline/fuse.py).

Needs ffmpeg on the PATH (audio decoding only).

    python -m pipeline.whistle        # cache/whistle.npz
"""

import os
import subprocess

import numpy as np

from . import io_atd

RATE = 16000
HOP = RATE // 25           # one video frame
NFFT = 1024
BAND = (3000.0, 4800.0)
REF = (1000.0, 7000.0)
OUT = os.path.join(io_atd.CACHE_DIR, "whistle.npz")


def audio():
    video = os.path.join(io_atd.ROOT, "data", "tactical", "BAR-ATH_tactical.mp4")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(RATE),
                          "-f", "s16le", "-"], check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def scores(x, n_frames):
    """(n_frames,) tonal peak in the whistle band over the broadband median, dB; and its Hz."""
    freqs = np.fft.rfftfreq(NFFT, 1.0 / RATE)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    ref = (freqs >= REF[0]) & (freqs <= REF[1])
    win = np.hanning(NFFT).astype(np.float32)
    score = np.full(n_frames, np.nan, np.float32)
    peak = np.full(n_frames, np.nan, np.float32)
    level = np.full(n_frames, np.nan, np.float32)
    for lo in range(0, n_frames, 5000):
        idx = np.arange(lo, min(lo + 5000, n_frames))
        start = idx * HOP
        ok = start + NFFT <= len(x)
        idx, start = idx[ok], start[ok]
        if not len(idx):
            break
        seg = x[start[:, None] + np.arange(NFFT)[None]] * win
        p = np.abs(np.fft.rfft(seg, axis=1)) ** 2 + 1e-12
        k = p[:, band].argmax(1)
        med = np.median(p[:, ref], 1)
        score[idx] = 10 * np.log10(p[:, band][np.arange(len(idx)), k] / med)
        peak[idx] = freqs[band][k]
        level[idx] = 10 * np.log10(med)
    return score, peak, level


def events(score, thresh, min_len=4, merge=12):
    """[(start, stop)] runs of score > thresh at least min_len frames long, merged over short gaps."""
    hot = np.nan_to_num(score, nan=-99.0) > thresh
    change = np.flatnonzero(hot[1:] != hot[:-1]) + 1
    bounds = np.r_[0, change, len(hot)]
    runs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if hot[a]]
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= merge:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_len]


def build(use_cache=True):
    if use_cache and os.path.isfile(OUT):
        z = np.load(OUT)
        return {k: z[k] for k in z.files}
    n = len(io_atd.load_match(raw_ball=True)["ball"])
    score, peak, level = scores(audio(), n)
    out = {"score": score, "peak_hz": peak, "level": level}
    np.savez(OUT, **out)
    return out


if __name__ == "__main__":
    out = build(use_cache=False)
    g = io_atd.load_match(raw_ball=True)["graded"]
    s = out["score"][g]
    print("graded frames with audio {:.1%} | score dB percentiles 50/90/99/99.9: {}".format(
        np.isfinite(s).mean(), np.nanpercentile(s, [50, 90, 99, 99.9]).round(1)))
