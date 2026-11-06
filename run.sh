#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <video> [--preset NAME] [--target SECONDS]"
  exit 1
fi

VIDEO="$1"
shift  # remaining flags are forwarded to select.py

PYTHON="$HERE/.venv/bin/python"
DEPS_OK="$HERE/.venv/.deps-ok"  # written only after pip succeeds, so a failed install retries

if [ ! -x "$PYTHON" ] || [ ! -f "$DEPS_OK" ]; then
  if [ ! -x "$PYTHON" ]; then
    BOOT=""
    for CAND in python3.14 python3.13 python3.12 python3.11 python3; do
      if command -v "$CAND" >/dev/null 2>&1 &&
         "$CAND" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
        BOOT="$CAND"
        break
      fi
    done
    if [ -z "$BOOT" ]; then
      echo "no python >= 3.11 on PATH (tried python3.14, python3.13, python3.12, python3.11, python3)"
      echo "install one, e.g.: brew install python@3.13"
      exit 1
    fi
    echo "[setup] creating venv with $("$BOOT" --version 2>&1) at $HERE/.venv"
    "$BOOT" -m venv "$HERE/.venv"
  fi
  echo "[setup] installing requirements"
  "$PYTHON" -m pip install --quiet -r "$HERE/requirements.txt"
  touch "$DEPS_OK"
fi

echo "=== $VIDEO ==="
"$PYTHON" "$HERE/analyze.py" "$VIDEO"
"$PYTHON" "$HERE/select.py" "$VIDEO" "$@"
"$PYTHON" "$HERE/render.py" "$VIDEO"
