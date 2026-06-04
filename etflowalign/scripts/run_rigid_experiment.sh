#!/usr/bin/env bash
#SBATCH --job-name=rigid_a
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
## --- cluster-specific: edit value, then delete ONE leading '#' to enable ---
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT
#
# Run the full direction-A (rigid head + rigid SE(3) path) experiment pipeline:
#   train -> diagnose vector field -> inference (few-step Heun) -> trajectory diagnose -> export SDF
#
# This single file runs both ways (the #SBATCH lines are plain comments to bash):
#   sbatch etflowalign/scripts/run_rigid_experiment.sh     # submit to Slurm  (squeue to watch)
#   bash   etflowalign/scripts/run_rigid_experiment.sh     # run directly
#
# IMPORTANT: submit from the repo root so $SLURM_SUBMIT_DIR points there:
#   cd /home/deepfold/users/hosung/work/ETFlowAlign
#   CONDA_ENV=etflow sbatch etflowalign/scripts/run_rigid_experiment.sh
#
# Output layout follows the smoke_tests convention:
#   <SMOKE_ROOT>/07_rigid_head/{checkpoints,logs,diagnostics,inference,sdf}
#
# Override any setting via environment variables. With sbatch, pass them through --export:
#   sbatch --export=ALL,CONDA_ENV=etflow,STEPS=8000,LR=5e-4,NAME=rigid_head_8k \
#     etflowalign/scripts/run_rigid_experiment.sh
# With bash, just prefix them:
#   CONDA_ENV=etflow STEPS=8000 NAME=rigid_head_8k bash etflowalign/scripts/run_rigid_experiment.sh
#
set -euo pipefail

# --- resolve repo root (works under both `bash` and `sbatch`) ---
# Under Slurm the batch script is copied to a spool dir, so $BASH_SOURCE is unreliable.
# Slurm sets SLURM_SUBMIT_DIR to the directory you ran `sbatch` from (use the repo root).
if [ -n "${REPO_ROOT:-}" ]; then
  :                                       # explicit REPO_ROOT override wins
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"
# --- optional: activate a conda env inside the (non-interactive) job ---
# Non-interactive Slurm shells don't auto-init conda; set CONDA_ENV to activate one here.
if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
# --- configuration (override via env) ---
SMOKE_ROOT="${SMOKE_ROOT:-$REPO_ROOT/etflowalign/smoke_tests}"
INPUT_DIR="${INPUT_DIR:-$SMOKE_ROOT/00_inputs/diffalign_example_rigid_source}"
QUERY_SDF="${QUERY_SDF:-$SMOKE_ROOT/00_inputs/diffalign_example/original_files/query_original.sdf}"
OUT="${OUT:-$SMOKE_ROOT/07_rigid_head}"

NAME="${NAME:-rigid_head_5k}"
STEPS="${STEPS:-5000}"
LR="${LR:-5e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
LOG_EVERY="${LOG_EVERY:-100}"
USE_ATOM_INDEX="${USE_ATOM_INDEX:-1}"     # 1 = add --use-atom-index-embed

NUM_SAMPLES="${NUM_SAMPLES:-4}"
SOLVER="${SOLVER:-heun}"
N_STEPS_LIST="${N_STEPS_LIST:-10 20 50}"  # few-step sweep
TRAJ_N_STEPS="${TRAJ_N_STEPS:-50}"
TRAJ_SOLVERS="${TRAJ_SOLVERS:-euler heun}"   # trajectory diagnostic runs each: euler shows integrator drift, heun is tight
TRAJ_REPORT_EVERY="${TRAJ_REPORT_EVERY:-5}"  # finer than the default 50 so geometry evolution is visible
SDF_N_STEPS="${SDF_N_STEPS:-20}"          # which inference output to export to SDF

# default to cuda when Slurm allocated a GPU (CUDA_VISIBLE_DEVICES set), else cpu
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then DEVICE="${DEVICE:-cuda}"; else DEVICE="${DEVICE:-cpu}"; fi
PY="${PY:-python}"
RUN_PREPARE="${RUN_PREPARE:-0}"

TRAIN_PT="$INPUT_DIR/diffalign_example_train.pt"
INFER_PT="$INPUT_DIR/diffalign_example_infer.pt"
CKPT="$OUT/checkpoints/etflowalign_${NAME}.pt"
BEST_CKPT="$OUT/checkpoints/etflowalign_${NAME}_best.pt"

ATOM_INDEX_FLAG=""
[ "$USE_ATOM_INDEX" = "1" ] && ATOM_INDEX_FLAG="--use-atom-index-embed"

mkdir -p "$OUT"/{checkpoints,logs,diagnostics,inference,sdf}

hr() { printf '\n========== %s ==========\n' "$1"; }

# --- sanity: confirm the rigid flags exist in this checkout ---
if ! "$PY" -m etflowalign.train --help 2>/dev/null | grep -q -- "--path-type"; then
  echo "ERROR: this checkout has no --path-type / --use-rigid-head flags." >&2
  echo "       Pull the direction-A code changes before running." >&2
  exit 1
fi

# --- stage 0 (optional): rebuild rigid-source input batches ---
if [ "$RUN_PREPARE" = "1" ]; then
  hr "stage 0: prepare rigid-source input batches"
  mkdir -p "$INPUT_DIR"
  "$PY" -m etflowalign.scripts.prepare_diffalign_example_train_batch \
    --repo-root "$REPO_ROOT" --out "$TRAIN_PT"
  "$PY" -m etflowalign.scripts.prepare_diffalign_example_batch \
    --repo-root "$REPO_ROOT" --out "$INFER_PT"
fi

for f in "$TRAIN_PT" "$INFER_PT"; do
  [ -f "$f" ] || { echo "ERROR: missing input batch: $f (run with RUN_PREPARE=1)" >&2; exit 1; }
done

# --- stage 1: train (rigid head + rigid path) ---
hr "stage 1: train  ($NAME, steps=$STEPS, lr=$LR)"
"$PY" -m etflowalign.train \
  --train-data "$TRAIN_PT" \
  --use-rigid-head --path-type rigid \
  --source-type input_query --sigma 0.0 --source-noise-scale 0.0 \
  $ATOM_INDEX_FLAG \
  --steps "$STEPS" --lr "$LR" --hidden-dim "$HIDDEN_DIM" --num-blocks "$NUM_BLOCKS" \
  --log-every "$LOG_EVERY" --device "$DEVICE" \
  --save-path "$CKPT" \
  2>&1 | tee "$OUT/logs/train_${NAME}.log"

# --- stage 2: vector-field diagnostic (per-t RMSE) ---
hr "stage 2: diagnose vector field"
"$PY" -m etflowalign.scripts.diagnose_vector_field \
  --checkpoint "$BEST_CKPT" \
  --train-data "$TRAIN_PT" \
  --device "$DEVICE" \
  2>&1 | tee "$OUT/diagnostics/diagnose_${NAME}_best.txt"

# --- stage 3: few-step inference sweep (Heun) ---
hr "stage 3: inference sweep  (solver=$SOLVER, n_steps in: $N_STEPS_LIST)"
for N in $N_STEPS_LIST; do
  "$PY" -m etflowalign.inference \
    --checkpoint "$BEST_CKPT" \
    --input-batch "$INFER_PT" \
    --num-samples "$NUM_SAMPLES" --n-steps "$N" --solver "$SOLVER" \
    --device "$DEVICE" \
    --save-path "$OUT/inference/${NAME}_${SOLVER}_nsteps${N}_sample${NUM_SAMPLES}.pt"
done

# --- stage 4: trajectory diagnostic (per-step bond statistics), per solver ---
hr "stage 4: diagnose sampling trajectory  (solvers: $TRAJ_SOLVERS, n_steps=$TRAJ_N_STEPS, report-every=$TRAJ_REPORT_EVERY)"
for S in $TRAJ_SOLVERS; do
  "$PY" -m etflowalign.scripts.diagnose_sampling_trajectory \
    --checkpoint "$BEST_CKPT" \
    --input-batch "$INFER_PT" \
    --query-sdf "$QUERY_SDF" \
    --n-steps "$TRAJ_N_STEPS" \
    --solver "$S" \
    --report-every "$TRAJ_REPORT_EVERY" \
    --device "$DEVICE" \
    --csv-out "$OUT/diagnostics/trajectory_${NAME}_${S}_nsteps${TRAJ_N_STEPS}.csv" \
    2>&1 | tee "$OUT/diagnostics/trajectory_${NAME}_${S}_nsteps${TRAJ_N_STEPS}.log"
done

# --- stage 5: export candidates to SDF ---
hr "stage 5: export SDF  (from n_steps=$SDF_N_STEPS inference)"
"$PY" -m etflowalign.scripts.export_candidates_to_sdf \
  --input-pt "$OUT/inference/${NAME}_${SOLVER}_nsteps${SDF_N_STEPS}_sample${NUM_SAMPLES}.pt" \
  --out-dir "$OUT/sdf/${NAME}_${SOLVER}_nsteps${SDF_N_STEPS}_sample${NUM_SAMPLES}" \
  --prefix "${NAME}_${SOLVER}_nsteps${SDF_N_STEPS}"

hr "DONE"
echo "outputs under: $OUT"
