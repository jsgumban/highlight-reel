#!/usr/bin/env python3
"""Score a video's raw excitement signals on a fixed hop grid.

Pipeline: ffmpeg -> mono PCM + tiny grayscale frames -> five per-hop
signals (loudness, onset flux, speech-band energy, motion, scenecut
density) -> outputs/<video_stem>/signals.npz + meta.json.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal

from common import load_config, resolve_ffmpeg

STFT_NPERSEG = 1024
STFT_NOVERLAP = 512
STFT_CHUNK_S = 600  # STFT processed ~10 minutes at a time so long videos fit in RAM
DIFF_CHUNK = 2048   # frame pairs diffed per batch


def decode_audio(ffmpeg: str, src: Path, sr: int) -> np.ndarray:
    """Mono float32 PCM in [-1, 1]; empty array if the file has no usable audio."""
    cmd = [
        ffmpeg, "-v", "error", "-i", str(src),
        "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def decode_frames(ffmpeg: str, src: Path, fps: float, w: int, h: int) -> np.ndarray:
    """Grayscale frames (n, h, w) uint8; empty array if the file has no usable video."""
    cmd = [
        ffmpeg, "-v", "error", "-i", str(src),
        "-vf", f"fps={fps},scale={w}:{h}",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    n = len(proc.stdout) // (w * h)
    if proc.returncode != 0 or n == 0:
        return np.zeros((0, h, w), dtype=np.uint8)
    return np.frombuffer(proc.stdout[: n * w * h], dtype=np.uint8).reshape(-1, h, w)


def fit_to_hops(x: np.ndarray, n_hops: int) -> np.ndarray:
    """Pad (with the array's minimum) or trim so the signal is exactly n_hops long."""
    x = x.astype(np.float32)
    if len(x) >= n_hops:
        return x[:n_hops]
    fill = float(x.min()) if len(x) else 0.0
    return np.concatenate([x, np.full(n_hops - len(x), fill, dtype=np.float32)])


def hop_bins(times: np.ndarray, hop_s: float, n_hops: int) -> np.ndarray:
    return np.clip((times / hop_s).astype(np.int64), 0, n_hops - 1)


def aggregate_mean(values: np.ndarray, times: np.ndarray, hop_s: float,
                   n_hops: int, empty_fill_min: bool = False) -> np.ndarray:
    """Mean of `values` per hop bin. Empty bins are 0, or the array minimum."""
    out = np.zeros(n_hops, dtype=np.float32)
    if len(values) == 0:
        return out
    bins = hop_bins(times, hop_s, n_hops)
    sums = np.zeros(n_hops, dtype=np.float64)
    counts = np.zeros(n_hops, dtype=np.int64)
    np.add.at(sums, bins, values.astype(np.float64))
    np.add.at(counts, bins, 1)
    filled = counts > 0
    out[filled] = (sums[filled] / counts[filled]).astype(np.float32)
    if empty_fill_min and filled.any():
        out[~filled] = out[filled].min()
    return out


def per_hop_rms_db(audio: np.ndarray, sr: int, hop_s: float, n_hops: int) -> np.ndarray:
    """RMS per hop -> 20*log10(rms + 1e-6) dB, aligned to the hop grid."""
    hop = max(1, int(round(sr * hop_s)))
    n_full = math.ceil(len(audio) / hop)
    padded = np.zeros(n_full * hop, dtype=np.float32)
    padded[: len(audio)] = audio
    rms = np.sqrt(np.mean(padded.reshape(n_full, hop) ** 2, axis=1))
    return fit_to_hops(20.0 * np.log10(rms + 1e-6), n_hops)


def speech_band(audio: np.ndarray, sr: int) -> np.ndarray:
    """300-3400 Hz telephone band, zero-phase filtered."""
    sos = signal.butter(4, [300.0, 3400.0], btype="band", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def onset_flux(audio: np.ndarray, sr: int, hop_s: float, n_hops: int) -> np.ndarray:
    """Spectral flux per STFT frame, mean-aggregated into hop bins.

    The STFT runs over ~10-minute chunks (each extended by one STFT hop so the
    frame grid stays continuous), carrying the last magnitude column across the
    boundary, so hour-long audio never holds a full spectrogram in memory.
    """
    step = STFT_NPERSEG - STFT_NOVERLAP
    chunk = max(step, int(STFT_CHUNK_S * sr) - int(STFT_CHUNK_S * sr) % step)
    flux_parts: list[np.ndarray] = []
    prev_col: np.ndarray | None = None
    for start in range(0, len(audio), chunk):
        seg = audio[start : start + chunk + step]
        if prev_col is not None and len(seg) <= STFT_NOVERLAP:
            # final sliver lies inside the previous chunk's last frame; a
            # single-pass STFT emits no frame here, so neither do we
            break
        if len(seg) < STFT_NPERSEG:
            seg = np.pad(seg, (0, STFT_NPERSEG - len(seg)))
        _, _, spec = signal.stft(seg, fs=sr, nperseg=STFT_NPERSEG,
                                 noverlap=STFT_NOVERLAP, boundary=None, padded=True)
        mag = np.abs(spec).astype(np.float32)
        lead = mag[:, :1] if prev_col is None else prev_col  # first-ever frame: flux 0
        d = np.diff(np.concatenate([lead, mag], axis=1), axis=1)
        flux_parts.append(np.maximum(d, 0.0).sum(axis=0).astype(np.float32))
        prev_col = mag[:, -1:]
    flux = np.concatenate(flux_parts)
    centers = (np.arange(len(flux), dtype=np.float64) * step + STFT_NPERSEG / 2) / sr
    return aggregate_mean(flux, centers, hop_s, n_hops, empty_fill_min=True)


def motion_signals(frames: np.ndarray, fps: float, hop_s: float,
                   n_hops: int) -> tuple[np.ndarray, np.ndarray]:
    """(motion, scenecut): per-hop mean frame diff, and hard-cut count per hop.

    A hard cut is a frame whose diff exceeds a global robust threshold
    (median + 5*MAD over all diffs).
    """
    zeros = np.zeros(n_hops, dtype=np.float32)
    n = len(frames)
    if n < 2:
        return zeros, zeros.copy()

    diffs = np.empty(n - 1, dtype=np.float32)
    for a in range(0, n - 1, DIFF_CHUNK):
        b = min(n - 1, a + DIFF_CHUNK)
        d = np.abs(frames[a + 1 : b + 1].astype(np.int16) - frames[a:b].astype(np.int16))
        diffs[a:b] = d.mean(axis=(1, 2)).astype(np.float32) / 255.0
    times = np.arange(1, n, dtype=np.float64) / fps  # diff i lands on frame i+1

    motion = aggregate_mean(diffs, times, hop_s, n_hops)

    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    thr = med + 5.0 * mad
    cut = diffs > thr
    scenecut = np.zeros(n_hops, dtype=np.float32)
    if cut.any():
        np.add.at(scenecut, hop_bins(times[cut], hop_s, n_hops), 1.0)
    return motion, scenecut


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--config", default=str(here / "config.toml"))
    parser.add_argument("--out-root", default=str(here / "outputs"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))["analysis"]
    sr = int(cfg["sample_rate"])
    hop_s = float(cfg["hop_s"])
    frame_fps = float(cfg["frame_fps"])
    w, h = int(cfg["frame_size"][0]), int(cfg["frame_size"][1])

    video = Path(args.video).resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    out_dir = Path(args.out_root) / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = resolve_ffmpeg()

    print(f"[analyze] decoding audio: {video.name}", flush=True)
    audio = decode_audio(ffmpeg, video, sr)
    has_audio = len(audio) > 0
    if has_audio:
        print(f"[analyze] {len(audio)/sr:.1f}s of audio at {sr} Hz", flush=True)
    else:
        print("[analyze] warning: no usable audio; audio signals will be zeros", flush=True)

    print(f"[analyze] decoding frames at {frame_fps:g} fps ({w}x{h} gray)", flush=True)
    frames = decode_frames(ffmpeg, video, frame_fps, w, h)
    has_video = len(frames) > 0
    if has_video:
        print(f"[analyze] {len(frames)} frames ({len(frames)/frame_fps:.1f}s of video)", flush=True)
    else:
        print("[analyze] warning: no usable video; motion/scenecut will be zeros", flush=True)

    if not has_audio and not has_video:
        sys.exit(f"could not decode any audio or video from {video}")

    duration_s = max(len(audio) / sr, len(frames) / frame_fps)
    n_hops = math.ceil(duration_s / hop_s)
    t = ((np.arange(n_hops) + 0.5) * hop_s).astype(np.float32)
    print(f"[analyze] {duration_s:.1f}s -> {n_hops} hops of {hop_s:g}s", flush=True)

    zeros = np.zeros(n_hops, dtype=np.float32)
    if has_audio:
        print("[analyze] loudness: per-hop RMS", flush=True)
        loudness = per_hop_rms_db(audio, sr, hop_s, n_hops)
        print("[analyze] speech: 300-3400 Hz band energy", flush=True)
        speech = per_hop_rms_db(speech_band(audio, sr), sr, hop_s, n_hops)
        print("[analyze] onset: spectral flux (chunked STFT)", flush=True)
        onset = onset_flux(audio, sr, hop_s, n_hops)
    else:
        loudness, speech, onset = zeros, zeros.copy(), zeros.copy()

    if has_video:
        print("[analyze] motion + scenecut density", flush=True)
        motion, scenecut = motion_signals(frames, frame_fps, hop_s, n_hops)
    else:
        motion, scenecut = zeros.copy(), zeros.copy()

    np.savez(out_dir / "signals.npz", t=t, loudness=loudness, onset=onset,
             speech=speech, motion=motion, scenecut=scenecut)
    print(f"[analyze] signals -> {out_dir/'signals.npz'}", flush=True)

    meta = {
        "video": str(video),
        "duration_s": float(duration_s),
        "hop_s": hop_s,
        "sample_rate": sr,
        "frame_fps": frame_fps,
        "frame_size": [w, h],
        "n_hops": int(n_hops),
        "has_audio": has_audio,
        "has_video": has_video,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[analyze] meta -> {out_dir/'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
