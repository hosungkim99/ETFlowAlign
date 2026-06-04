#!/usr/bin/env bash
#SBATCH --job-name=pdbbind_shape
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
## --- cluster-specific: edit value, then delete ONE leading '#' to enable ---
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT
#
# Full-data PDBbind pocket-conditioned rigid (direction A) training + held-out evaluation.
#   stage 1: train on --train-dir with a held-out val split + in-training [val] RMSD monitoring
#   stage 2: evaluate the best checkpoint on the held-out val set
#   stage 3: evaluate on a small train subset (memorization vs generalization comparison)
#
# Runs both ways (the #SBATCH lines are plain comments to bash):
#   sbatch etflowalign/scripts/run_pdbbind_training.sh        # submit to Slurm  (squeue to watch)
#   bash   etflowalign/scripts/run_pdbbind_training.sh        # run directly
#
# IMPORTANT: submit from the repo root so $SLURM_SUBMIT_DIR points there:
#   cd /home/deepfold/users/hosung/work/ETFlowAlign
#   CONDA_ENV=etflowalign sbatch etflowalign/scripts/run_pdbbind_training.sh
#
# Override any setting via env vars (with sbatch, pass through --export):
#   sbatch --export=ALL,CONDA_ENV=etflowalign,STEPS=200000,LR=3e-4 \
#     etflowalign/scripts/run_pdbbind_training.sh
set -euo pipefail

# --- resolve repo root (works under both bash and sbatch) ---
if [ -n "${REPO_ROOT:-}" ]; then
  :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

# --- optional: activate a conda env inside the (non-interactive) job ---
if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# Stream python stdout live: when piped through `tee`, python block-buffers stdout, so
# [train]/[val] lines would otherwise sit in the buffer for thousands of steps before
# appearing in the log/.out file.
export PYTHONUNBUFFERED=1

# --- configuration (override via env) ---
SMOKE_ROOT="${SMOKE_ROOT:-$REPO_ROOT/etflowalign/smoke_tests}"
DIR="${DIR:-$SMOKE_ROOT/00_inputs/pdbbind_refined_pt}"
OUT="${OUT:-$SMOKE_ROOT/08_pdbbind_pocket}"

NAME="${NAME:-pdbbind_shape_split}"
STEPS="${STEPS:-100000}"
LR="${LR:-5e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOG_EVERY="${LOG_EVERY:-500}"

USE_POCKET="${USE_POCKET:-1}"            # 1 = --use-pocket-conditioning
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-0}"
VAL_EVERY="${VAL_EVERY:-2000}"
VAL_EVAL_LIMIT="${VAL_EVAL_LIMIT:-200}"

SOLVER="${SOLVER:-heun}"
EVAL_N_STEPS="${EVAL_N_STEPS:-50}"
TRAIN_EVAL_LIMIT="${TRAIN_EVAL_LIMIT:-50}"  # train-subset size for the comparison eval
RUN_EVAL="${RUN_EVAL:-1}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then DEVICE="${DEVICE:-cuda}"; else DEVICE="${DEVICE:-cpu}"; fi
PY="${PY:-python}"

POCKET_FLAG=""
[ "$USE_POCKET" = "1" ] && POCKET_FLAG="--use-pocket-conditioning"

SAVE="$OUT/checkpoints/etflowalign_${NAME}.pt"
BEST="$OUT/checkpoints/etflowalign_${NAME}_best.pt"
VAL_LIST="${SAVE}.val.txt"

mkdir -p "$OUT"/{checkpoints,logs,diagnostics,inference,sdf,eval}

hr() { printf '\n========== %s ==========\n' "$1"; }

# --- sanity ---
if ! "$PY" -m etflowalign.train --help 2>/dev/null | grep -q -- "--val-every"; then
  echo "ERROR: this checkout lacks --val-every / monitoring. Pull the latest code first." >&2
  exit 1
fi
if ! ls "$DIR"/*.pt >/dev/null 2>&1; then
  echo "ERROR: no .pt files in $DIR. Build the dataset with prepare_pdbbind_dataset first." >&2
  exit 1
fi
echo "[info] dataset: $(ls "$DIR"/*.pt | wc -l) complexes | device=$DEVICE | name=$NAME"

# --- stage 1: train (held-out val + in-training [val] RMSD monitoring) ---
hr "stage 1: train  ($NAME, steps=$STEPS, bs=$BATCH_SIZE, lr=$LR, val_fraction=$VAL_FRACTION)"
"$PY" -m etflowalign.train \
  --train-dir "$DIR" \
  --use-rigid-head --path-type rigid --source-type input_query $POCKET_FLAG \
  --conditioning pocket --batch-size "$BATCH_SIZE" \
  --steps "$STEPS" --lr "$LR" --hidden-dim "$HIDDEN_DIM" --num-blocks "$NUM_BLOCKS" \
  --val-fraction "$VAL_FRACTION" --split-seed "$SPLIT_SEED" \
  --val-every "$VAL_EVERY" --val-eval-limit "$VAL_EVAL_LIMIT" --val-eval-solver "$SOLVER" \
  --device "$DEVICE" --log-every "$LOG_EVERY" \
  --save-path "$SAVE" \
  2>&1 | tee "$OUT/logs/train_${NAME}.log"

if [ "$RUN_EVAL" != "1" ]; then hr "DONE (eval skipped)"; echo "outputs under: $OUT"; exit 0; fi

# --- stage 2: evaluate on the held-out val set (generalization) ---
hr "stage 2: evaluate held-out val"
"$PY" -m etflowalign.scripts.evaluate_dataset \
  --checkpoint "$BEST" --manifest "$VAL_LIST" \
  --conditioning pocket --solver "$SOLVER" --n-steps "$EVAL_N_STEPS" --device "$DEVICE" \
  --csv-out "$OUT/eval/eval_${NAME}_val.csv" \
  2>&1 | tee "$OUT/eval/eval_${NAME}_val.log"

# --- stage 3: evaluate on a train subset (memorization reference) ---
hr "stage 3: evaluate train subset (limit=$TRAIN_EVAL_LIMIT)"
"$PY" -m etflowalign.scripts.evaluate_dataset \
  --checkpoint "$BEST" --data-dir "$DIR" --limit "$TRAIN_EVAL_LIMIT" \
  --conditioning pocket --solver "$SOLVER" --n-steps "$EVAL_N_STEPS" --device "$DEVICE" \
  --csv-out "$OUT/eval/eval_${NAME}_trainsubset.csv" \
  2>&1 | tee "$OUT/eval/eval_${NAME}_trainsubset.log"

hr "DONE"
echo "compare: val (stage 2) vs train subset (stage 3) predicted RMSD"
echo "outputs under: $OUT"
