#!/usr/bin/env python3
"""Shared helpers for the highlight-reel pipeline: config loading and ffmpeg lookup."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

_FFMPEG: str | None = None


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_ffmpeg() -> str:
    """Locate an ffmpeg binary: $FFMPEG_BIN -> PATH -> imageio-ffmpeg's bundled build.

    There is no system ffmpeg on some machines, so never call bare "ffmpeg";
    always run the path this returns. The result is cached for the process.
    """
    global _FFMPEG
    if _FFMPEG is not None:
        return _FFMPEG

    env = os.environ.get("FFMPEG_BIN")
    if env:
        try:
            probe = subprocess.run([env, "-version"], capture_output=True)
        except OSError:
            probe = None
        if probe is None or probe.returncode != 0 or not probe.stdout.startswith(b"ffmpeg"):
            sys.exit(
                f"$FFMPEG_BIN={env} is not a runnable ffmpeg binary "
                "(fix it or unset it to fall back to PATH / imageio-ffmpeg)"
            )
        _FFMPEG = env
        return _FFMPEG

    found = shutil.which("ffmpeg")
    if found:
        _FFMPEG = found
        return _FFMPEG

    try:
        import imageio_ffmpeg

        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG
    except Exception:
        sys.exit(
            "ffmpeg not found: set $FFMPEG_BIN to a binary, put ffmpeg on your "
            "PATH, or `pip install imageio-ffmpeg` (ships a static build)"
        )
