#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv_csci1470_smoke}"
REQ_FILE="$ROOT_DIR/requirements-smoke.txt"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.pip_cache}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH"
  exit 2
fi

LOCK_DIR="${VENV_DIR}.lock"
READY_FILE="$VENV_DIR/.bootstrap_ready"

if mkdir "$LOCK_DIR" 2>/dev/null; then
  cleanup_lock() {
    rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
  }
  trap cleanup_lock EXIT

  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Creating venv at: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  echo "Using python: $(which python)"
  python --version

  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --cache-dir "$PIP_CACHE_DIR" -r "$REQ_FILE"

  touch "$READY_FILE"

  cleanup_lock
  trap - EXIT
else
  echo "Another job is preparing venv at: $VENV_DIR; waiting..."
  for _ in $(seq 1 1200); do
    if [[ -f "$READY_FILE" && -f "$VENV_DIR/bin/activate" ]]; then
      break
    fi
    sleep 1
  done

  if [[ ! -f "$READY_FILE" || ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: timed out waiting for venv bootstrap at $VENV_DIR"
    exit 3
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  echo "Using python: $(which python)"
  python --version
fi

echo "Venv ready: $VENV_DIR"
