#!/usr/bin/env bash
#SBATCH --job-name=csci1470_infer_best_gif
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

PIPE_TAG_VALUE="${PIPE_TAG:-}"
PIPE_DIR=""
if [[ -n "$PIPE_TAG_VALUE" ]]; then
  PIPE_DIR="$PWD/artifacts/pipelines/$PIPE_TAG_VALUE"
fi

LATEST_CKPT_PATH=$(ls -dt "$PWD"/artifacts/*/checkpoint_best.pt 2>/dev/null | head -n1 || true)

if [[ -n "$PIPE_TAG_VALUE" && ( -z "$PIPE_DIR" || ! -d "$PIPE_DIR" ) ]]; then
  echo "WARNING: pipeline dir not found for PIPE_TAG=$PIPE_TAG_VALUE; falling back to the newest checkpoint under artifacts/"
fi

if [[ -z "$PIPE_TAG_VALUE" ]]; then
  echo "INFO: PIPE_TAG not provided; falling back to the newest checkpoint under artifacts/"
fi

VENV_DIR="${VENV_DIR:-$PWD/.venv_csci1470_smoke}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pip_cache}"
export PIP_CACHE_DIR

bash ./bootstrap_venv.sh "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --cache-dir "$PIP_CACHE_DIR" -r requirements-train.txt

which python
python --version
nvidia-smi || true

if [[ -n "$PIPE_DIR" && -f "$PIPE_DIR/checkpoint_best_path.txt" ]]; then
  CKPT_PATH=$(cat "$PIPE_DIR/checkpoint_best_path.txt")
else
  if [[ -z "$LATEST_CKPT_PATH" ]]; then
    echo "ERROR: no checkpoint_best.pt found under $PWD/artifacts/"
    exit 2
  fi
  CKPT_PATH="$LATEST_CKPT_PATH"
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH"
  exit 3
fi

if [[ -n "$PIPE_DIR" && -f "$PIPE_DIR/train_run_dir.txt" ]]; then
  RUN_DIR=$(cat "$PIPE_DIR/train_run_dir.txt")
else
  RUN_DIR="$(dirname "$CKPT_PATH")"
fi

OUTDIR="$RUN_DIR/inference_best_gif"
mkdir -p "$OUTDIR"

echo "=== RUN INFERENCE + SETUP GIFS + BEST GIF ==="
echo "PIPE_TAG=${PIPE_TAG_VALUE:-<inferred>}"
echo "checkpoint=$CKPT_PATH"
echo "outdir=$OUTDIR"

python run_inference_best_gif.py \
  --checkpoint "$CKPT_PATH" \
  --num-setups "${NUM_SETUPS:-10}" \
  --max-steps "${MAX_STEPS:-320}" \
  --policy deterministic \
  --pos-threshold "${POS_THRESHOLD:-0.35}" \
  --vel-threshold "${VEL_THRESHOLD:-0.45}" \
  --seed "${SEED:-1234}" \
  --device auto \
  --outdir "$OUTDIR"

if [[ -n "$PIPE_DIR" ]]; then
  echo "$OUTDIR" > "$PIPE_DIR/inference_outdir.txt"
fi

echo "end_time=$(date)"
