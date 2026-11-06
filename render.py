#!/usr/bin/env python3
"""Render highlight outputs from selected segments.

Reads outputs/<video_basename>/segments.json (produced by select.py) and
emits ranked clips (clips/highlight_NN_t<start>s.mp4), a chronological
supercut.mp4, chapters.txt mapping supercut time to source time, and a
contactsheet.jpg thumbnail grid.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import load_config, resolve_ffmpeg  # noqa: E402


def fmt_ts(seconds: float) -> str:
    """M:SS, or H:MM:SS past an hour."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def stream_maps(has_video: bool, has_audio: bool) -> list[str]:
    maps: list[str] = []
    if has_video:
        maps += ["-map", "0:v:0"]
    if has_audio:
        maps += ["-map", "0:a:0"]
    return maps


def cut_copy(ffmpeg: str, src: Path, start: float, dur: float,
             maps: list[str], out_path: Path) -> None:
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{dur:.3f}",
        *maps,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def cut_encode(ffmpeg: str, src: Path, start: float, dur: float, maps: list[str],
               has_video: bool, has_audio: bool, out_path: Path) -> None:
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{dur:.3f}",
        *maps,
    ]
    if has_video:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True)


def cut(ffmpeg: str, src: Path, start: float, dur: float, maps: list[str],
        has_video: bool, has_audio: bool, out_path: Path) -> None:
    """Stream-copy cut, falling back to a re-encode when the source's codecs
    can't be muxed into .mp4 (e.g. vp8/vorbis webm)."""
    try:
        cut_copy(ffmpeg, src, start, dur, maps, out_path)
    except subprocess.CalledProcessError:
        out_path.unlink(missing_ok=True)
        print(f"[render] stream copy at {start:.1f}s failed (codecs don't fit "
              ".mp4); re-encoding this cut", flush=True)
        cut_encode(ffmpeg, src, start, dur, maps, has_video, has_audio, out_path)


def concat_copy(ffmpeg: str, parts: list[Path], list_path: Path, out_path: Path) -> None:
    quoted = ("file '" + str(p).replace("'", "'\\''") + "'\n" for p in parts)
    list_path.write_text("".join(quoted))
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def write_chapters(segments: list[dict], out_path: Path) -> None:
    lines = []
    pos = 0.0
    for seg in segments:
        lines.append(
            f"{fmt_ts(pos)} #{seg['rank']} "
            f"(source {fmt_ts(seg['start'])}–{fmt_ts(seg['end'])}, "
            f"score {seg['score']:.1f})"
        )
        pos += seg["duration"]
    out_path.write_text("\n".join(lines) + "\n")


def grab_frame(ffmpeg: str, src: Path, at_s: float, out_png: Path) -> bool:
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-ss", f"{at_s:.3f}",
        "-i", str(src),
        "-frames:v", "1",
        "-vf", "scale=320:-2",
        str(out_png),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        return False
    return out_png.exists() and out_png.stat().st_size > 0


def write_contactsheet(ffmpeg: str, video: Path, segments: list[dict],
                       out_path: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="contactsheet-") as td:
        tiles = []
        for seg in segments:
            mid = (seg["start"] + seg["end"]) / 2
            png = Path(td) / f"tile_{seg['rank']:02d}.png"
            if grab_frame(ffmpeg, video, mid, png):
                tiles.append((seg, png))
            else:
                print(f"[render] contactsheet: frame grab at {mid:.1f}s failed, skipping tile", flush=True)
        if not tiles:
            print("[render] contactsheet skipped (no frames grabbed)", flush=True)
            return False

        cols = min(4, len(tiles))
        rows = math.ceil(len(tiles) / cols)
        fig = plt.figure(figsize=(cols * 3.2, rows * 2.2))
        for i, (seg, png) in enumerate(tiles):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(plt.imread(str(png)))
            ax.set_axis_off()
            ax.set_title(f"#{seg['rank']}  {fmt_ts(seg['start'])}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return True


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--config", default=str(here / "config.toml"))
    parser.add_argument("--out-root", default=str(here / "outputs"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    rcfg = cfg["render"]
    video = Path(args.video).resolve()
    out_dir = Path(args.out_root) / video.stem

    segments_path = out_dir / "segments.json"
    if not segments_path.exists():
        sys.exit(f"segments not found: {segments_path} (run select.py first)")
    data = json.loads(segments_path.read_text())
    segments = data["selected"]  # chronological
    if not segments:
        sys.exit("no segments selected -- nothing to render "
                 "(lower percentile_floor in config.toml and re-run select.py)")

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    has_video = bool(meta.get("has_video", True))
    has_audio = bool(meta.get("has_audio", True))
    ffmpeg = resolve_ffmpeg()
    maps = stream_maps(has_video, has_audio)

    clip_paths: list[Path] = []
    if rcfg["clips"]:
        clips_dir = out_dir / "clips"
        if clips_dir.exists():
            shutil.rmtree(clips_dir)
        clips_dir.mkdir(parents=True)
        for seg in segments:
            out = clips_dir / f"highlight_{seg['rank']:02d}_t{seg['start']:.0f}s.mp4"
            print(f"[render] clip #{seg['rank']:2d}: {seg['start']:.1f}-{seg['end']:.1f}s "
                  f"({seg['duration']:.1f}s) -> {out.name}", flush=True)
            cut(ffmpeg, video, seg["start"], seg["duration"], maps,
                has_video, has_audio, out)
            clip_paths.append(out)
        print(f"[render] clips -> {clips_dir}", flush=True)

    supercut_path = out_dir / "supercut.mp4"
    if rcfg["supercut"]:
        with tempfile.TemporaryDirectory(prefix="supercut-") as td_str:
            td = Path(td_str)
            if rcfg["reencode"]:
                parts = []
                for i, seg in enumerate(segments):
                    part = td / f"part_{i:03d}.mp4"
                    print(f"[render] re-encoding {seg['start']:.1f}-{seg['end']:.1f}s "
                          f"({i + 1}/{len(segments)})", flush=True)
                    cut_encode(ffmpeg, video, seg["start"], seg["duration"], maps,
                               has_video, has_audio, part)
                    parts.append(part)
            elif clip_paths:
                parts = clip_paths
            else:
                parts = []
                for i, seg in enumerate(segments):
                    part = td / f"part_{i:03d}.mp4"
                    cut(ffmpeg, video, seg["start"], seg["duration"], maps,
                        has_video, has_audio, part)
                    parts.append(part)
            concat_copy(ffmpeg, parts, td / "list.txt", supercut_path)
        print(f"[render] supercut -> {supercut_path}", flush=True)

    chapters_path = out_dir / "chapters.txt"
    write_chapters(segments, chapters_path)
    print(f"[render] chapters -> {chapters_path}", flush=True)

    contactsheet_path = out_dir / "contactsheet.jpg"
    if rcfg["contactsheet"]:
        if not has_video:
            print("[render] contactsheet skipped (no video stream)", flush=True)
        else:
            try:
                if write_contactsheet(ffmpeg, video, segments, contactsheet_path):
                    print(f"[render] contactsheet -> {contactsheet_path}", flush=True)
            except Exception as exc:  # never let the sheet kill the run
                print(f"[render] contactsheet failed ({exc}), continuing", flush=True)

    cov = data["coverage"]
    log_path = here / "LOG.md"
    if not log_path.exists():
        log_path.write_text("# Highlight Reel Log\n\n")
    log_line = (
        f"- {datetime.now().isoformat(timespec='seconds')} | {video.name} | "
        f"preset={data['preset']} segments={len(segments)} "
        f"supercut={cov['selected_s']:.0f}s of {cov['original_s']:.0f}s ({cov['pct']:.1f}%)\n"
    )
    with log_path.open("a") as f:
        f.write(log_line)

    print(f"[render] done: {len(segments)} segments, {cov['selected_s']:.0f}s "
          f"of {cov['original_s']:.0f}s ({cov['pct']:.1f}%) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
