#!/usr/bin/env python3
"""Select highlight segments from a video's analyzed excitement signals.

Reads outputs/<video_basename>/signals.npz + meta.json (produced by
analyze.py), fuses the five signals into one robust-z excitement score,
grows threshold-crossing cores into clean segments (snapping cuts into
calm moments), greedily fills to the target duration, and writes
segments.json + timeline.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from common import load_config  # noqa: E402

SIGNAL_NAMES = ("loudness", "onset", "speech", "motion", "scenecut")
SIGNAL_COLORS = {
    "loudness": "steelblue",
    "onset": "orange",
    "speech": "red",
    "motion": "green",
    "scenecut": "gray",
}


def robust_z(x: np.ndarray, z_clip: float) -> np.ndarray:
    """Clipped robust z-score. A flat signal (MAD ~ 0) yields zeros, not garbage."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        # Majority-flat signal (e.g. motion in a mostly-static screen
        # recording): MAD collapses to 0 even though real spikes exist.
        # Fall back to the mean absolute deviation so the spikes survive;
        # a truly constant signal still yields zeros.
        scale = float(np.mean(np.abs(x - med)))
    if scale < 1e-12:
        return np.zeros(len(x), dtype=np.float32)
    z = (x - med) / (scale + 1e-9)
    return np.clip(z, 0.0, z_clip).astype(np.float32)


def fuse(zs: dict[str, np.ndarray], weights: dict) -> np.ndarray:
    total_w = sum(float(weights.get(name, 0.0)) for name in SIGNAL_NAMES)
    if total_w <= 0:
        sys.exit("all preset weights are zero -- nothing to fuse")
    n = len(next(iter(zs.values())))
    fused = np.zeros(n, dtype=np.float32)
    for name in SIGNAL_NAMES:
        fused += float(weights.get(name, 0.0)) * zs[name]
    return fused / total_w


def find_cores(mask: np.ndarray) -> list[list[int]]:
    """Contiguous runs of True as [first_idx, last_idx] (inclusive)."""
    cores: list[list[int]] = []
    for i in np.flatnonzero(mask):
        if cores and i == cores[-1][1] + 1:
            cores[-1][1] = i
        else:
            cores.append([int(i), int(i)])
    return cores


def merge_cores(cores: list[list[int]], gap_hops: int) -> list[list[int]]:
    if not cores:
        return []
    merged = [cores[0]]
    for i0, i1 in cores[1:]:
        if i0 - merged[-1][1] - 1 <= gap_hops:
            merged[-1][1] = i1
        else:
            merged.append([i0, i1])
    return merged


def snap_edges(i0: int, i1: int, fused: np.ndarray, snap_hops: int) -> tuple[int, int]:
    """Walk each edge outward up to snap_hops and land it on the calmest hop."""
    lo = max(0, i0 - snap_hops)
    left = lo + int(np.argmin(fused[lo:i0 + 1]))
    hi = min(len(fused) - 1, i1 + snap_hops)
    right = i1 + int(np.argmin(fused[i1:hi + 1]))
    return left, right


def enforce_durations(segs: list[dict], min_clip: float, max_clip: float,
                      duration: float) -> list[dict]:
    out = []
    for s in segs:
        start, end, peak_t = s["start"], s["end"], s["peak_t"]
        dur = end - start
        if dur < min_clip:
            half = min_clip / 2
            start = max(0.0, min(start, peak_t - half))
            end = min(duration, max(end, peak_t + half))
        elif dur > max_clip:
            new_start = min(max(peak_t - max_clip / 2, start), end - max_clip)
            start, end = new_start, new_start + max_clip
        out.append({"start": start, "end": end, "peak_t": peak_t})
    return out


def merge_close(segs: list[dict], min_gap: float = 0.5) -> list[dict]:
    """Fuse segments that overlap or sit closer than min_gap after growth."""
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s["start"])
    merged = [dict(segs[0])]
    for s in segs[1:]:
        if s["start"] - merged[-1]["end"] < min_gap:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(dict(s))
    return merged


def score_segments(segs: list[dict], fused: np.ndarray, hop_s: float) -> None:
    n = len(fused)
    for s in segs:
        i0 = max(0, min(int(np.floor(s["start"] / hop_s)), n - 1))
        i1 = max(i0 + 1, min(int(np.ceil(s["end"] / hop_s)), n))
        window = fused[i0:i1]
        s["score"] = float(window.mean())
        s["peak"] = float(window.max())


def greedy_select(segs: list[dict], target: float) -> list[dict]:
    """Rank by 0.7*peak + 0.3*score, accept until the target duration is met."""
    order = sorted(segs, key=lambda s: 0.7 * s["peak"] + 0.3 * s["score"], reverse=True)
    selected: list[dict] = []
    total = 0.0
    for rank, s in enumerate(order, 1):
        if target > 0 and total >= target:
            break
        s["rank"] = rank
        selected.append(s)
        total += s["end"] - s["start"]
    return sorted(selected, key=lambda s: s["start"])


def write_timeline(out_path: Path, t: np.ndarray, fused: np.ndarray, thr: float,
                   zs: dict[str, np.ndarray], selected: list[dict],
                   percentile_floor: float, video_name: str) -> None:
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(20, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_top.plot(t, fused, lw=0.8, color="steelblue", label="fused score")
    ax_top.axhline(thr, ls="--", lw=0.8, color="orange",
                   label=f"threshold (p{percentile_floor:.0f})")
    for seg in selected:
        ax_top.axvspan(seg["start"], seg["end"], color="green", alpha=0.2)
        ax_top.text((seg["start"] + seg["end"]) / 2, 0.92, f"#{seg['rank']}",
                    transform=ax_top.get_xaxis_transform(),
                    ha="center", color="green", fontsize=8)
    ax_top.set_ylabel("excitement (z)")
    ax_top.set_title(f"{video_name}: excitement timeline")
    ax_top.legend(loc="upper right")

    for name in SIGNAL_NAMES:
        ax_bot.plot(t, zs[name], lw=0.5, color=SIGNAL_COLORS[name], label=name)
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("signal z")
    ax_bot.legend(loc="upper right", ncol=5, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--config", default=str(here / "config.toml"))
    parser.add_argument("--out-root", default=str(here / "outputs"))
    parser.add_argument("--preset", default=None, help="override the config's preset")
    parser.add_argument("--target", type=float, default=None,
                        help="override target_total_s (seconds)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    sel = cfg["selection"]
    video = Path(args.video).resolve()
    out_dir = Path(args.out_root) / video.stem

    signals_path = out_dir / "signals.npz"
    meta_path = out_dir / "meta.json"
    if not signals_path.exists() or not meta_path.exists():
        sys.exit(f"signals not found in {out_dir} (run analyze.py first)")

    preset = args.preset if args.preset is not None else cfg["preset"]
    presets = cfg.get("presets", {})
    if preset not in presets:
        sys.exit(f"unknown preset '{preset}' (available: {', '.join(sorted(presets))})")
    weights = presets[preset]

    target = float(args.target) if args.target is not None else float(sel["target_total_s"])

    meta = json.loads(meta_path.read_text())
    duration = float(meta["duration_s"])
    hop_s = float(meta["hop_s"])
    npz = np.load(signals_path)
    t = npz["t"]
    print(f"[select] {len(t)} hops over {duration:.1f}s, preset '{preset}'", flush=True)

    z_clip = float(sel["z_clip"])
    zs = {name: robust_z(npz[name], z_clip) for name in SIGNAL_NAMES}
    fused = fuse(zs, weights)
    fused = gaussian_filter1d(fused, sigma=float(sel["smooth_s"]) / hop_s).astype(np.float32)

    thr = float(np.percentile(fused, float(sel["percentile_floor"])))
    print(f"[select] threshold p{sel['percentile_floor']} = {thr:.2f}", flush=True)

    gap_hops = int(round(float(sel["merge_gap_s"]) / hop_s))
    snap_hops = int(round(float(sel["boundary_snap_s"]) / hop_s))
    cores = merge_cores(find_cores(fused >= thr), gap_hops)

    pre = float(sel["pre_roll_s"])
    post = float(sel["post_roll_s"])
    candidates = []
    for i0, i1 in cores:
        peak_idx = i0 + int(np.argmax(fused[i0:i1 + 1]))
        left, right = snap_edges(i0, i1, fused, snap_hops)
        candidates.append({
            "start": max(0.0, left * hop_s - pre),
            "end": min(duration, (right + 1) * hop_s + post),
            "peak_t": float(t[peak_idx]),
        })

    candidates = enforce_durations(candidates, float(sel["min_clip_s"]),
                                   float(sel["max_clip_s"]), duration)
    candidates = merge_close(candidates)
    score_segments(candidates, fused, hop_s)
    print(f"[select] {len(cores)} cores above threshold -> {len(candidates)} candidates", flush=True)

    selected = greedy_select(candidates, target)
    if not selected:
        print("[select] no segments found -- try lowering percentile_floor in config.toml", flush=True)

    selected_s = sum(s["end"] - s["start"] for s in selected)
    pct = 100.0 * selected_s / max(duration, 1e-9)
    segments = {
        "video": str(video),
        "preset": preset,
        "duration_s": round(duration, 3),
        "target_s": round(target, 3),
        "selected": [
            {
                "rank": s["rank"],
                "start": round(s["start"], 3),
                "end": round(s["end"], 3),
                "duration": round(s["end"] - s["start"], 3),
                "score": round(s["score"], 3),
                "peak": round(s["peak"], 3),
            }
            for s in selected
        ],
        "rejected_count": len(candidates) - len(selected),
        "coverage": {
            "original_s": round(duration, 3),
            "selected_s": round(selected_s, 3),
            "pct": round(pct, 1),
        },
    }
    (out_dir / "segments.json").write_text(json.dumps(segments, indent=2))
    print(f"[select] segments -> {out_dir/'segments.json'}", flush=True)

    write_timeline(out_dir / "timeline.png", t, fused, thr, zs, selected,
                   float(sel["percentile_floor"]), video.name)
    print(f"[select] timeline -> {out_dir/'timeline.png'}", flush=True)

    print(f"[select] kept {selected_s/60:.1f} min of {duration/60:.1f} min "
          f"({pct:.1f}%) in {len(selected)} segments", flush=True)


if __name__ == "__main__":
    main()
