#!/bin/bash
# Gate B on Explorer: parity with the paper, and the pretrained Boston weights.
#
# The research repo must already be at /scratch/$USER/gnn-flu (Code/sweep/sync.sh
# push there) and this repo at /scratch/$USER/areaFluForecast
# (scripts/explorer/sync.sh push <nuid>). It reuses the research venv
# (~/envs/gnn-flu), which already has torch and torch-geometric.
#
#   # 27 parity tasks (3 seasons x 3 horizons x 3 models)
#   sbatch --export=ALL,MODE=parity --array=0-26 scripts/explorer/areaflu.sbatch
#   # pretrained weights, one task per horizon; then transfer once those finish
#   sbatch --export=ALL,MODE=pretrain --array=1-4 scripts/explorer/areaflu.sbatch
#   sbatch --export=ALL,MODE=transfer --array=1-4 --dependency=afterok:<pretrain job> \
#          scripts/explorer/areaflu.sbatch
#
# Live 2026-27 track (scripts/live_forecast.py). Push live/<city>/ first.
# 3 trust-check seasons x 4 horizons = 12 backtest tasks (skip once a combined
# backtest exists in live/runs/<city>/), 4 forecast tasks, then one assemble:
#   bt=$(sbatch --parsable --export=ALL,MODE=live_backtest,CITY=boston --array=0-11 scripts/explorer/areaflu.sbatch)
#   fc=$(sbatch --parsable --export=ALL,MODE=live_forecast,CITY=boston --array=1-4 scripts/explorer/areaflu.sbatch)
#   sbatch --export=ALL,MODE=live_assemble,CITY=boston --dependency=afterok:$bt:$fc scripts/explorer/areaflu.sbatch
# LIVE_ARGS goes through to live_forecast.py, e.g. a missed week or one new model:
#   sbatch --export=ALL,MODE=live_forecast,CITY=chicago,LIVE_ARGS="--as-of 2026-09-27" --array=1-4 ...
#
# Season-start graph models for --models-dir (live_forecast.py train), one task per horizon;
# they land in live/models/<season>/<city>/:
#   sbatch --export=ALL,MODE=live_train,CITY=boston --array=1-4 scripts/explorer/areaflu.sbatch
#
# Frozen-model check (scripts/frozen_check.py): 12 season tasks + one per 2025-26 flu week.
# Weekly tasks retrain all 4 horizons, so give them more time:
#   sbatch --export=ALL,MODE=exp_frozen --array=0-11 scripts/explorer/areaflu.sbatch
#   sbatch --export=ALL,MODE=exp_frozen --array=12-37 --time=10:00:00 scripts/explorer/areaflu.sbatch
#
# Training-window experiment (scripts/training_window.py): COVID in/out x refit, 32 tasks.
#   sbatch --export=ALL,MODE=exp_window --array=0-31 scripts/explorer/areaflu.sbatch
#
# Progress: logs print the Budget / seed / Epoch / Outputs lines that
# Code/run_progress.py --remote reads.
#
#SBATCH --job-name=areaflu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/areaflu_%x_%A_%a.log
#SBATCH --error=logs/areaflu_%x_%A_%a.err
set -euo pipefail

MODE="${MODE:?MODE=parity|pretrain|transfer|live_backtest|live_forecast|live_assemble|live_train|exp_window|exp_frozen}"
PKG="${AREAFLU_ROOT:-/scratch/$USER/areaFluForecast}"
export AREAFLU_RESEARCH_REPO="${AREAFLU_RESEARCH_REPO:-/scratch/$USER/gnn-flu}"
ENVDIR="${GNNFLU_ENV:-$HOME/envs/gnn-flu}"
OUT="${AREAFLU_OUT:-$PKG/gate_b}"

module purge
module load explorer anaconda3/2024.06
source "$ENVDIR/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export MPLBACKEND=Agg PYTHONWARNINGS=ignore

cd "$PKG"
mkdir -p logs "$OUT"
# The package itself is pure Python; put src/ on the path instead of pip
# installing on a compute node (pip needs the login-shell proxy variables).
export PYTHONPATH="$PKG/src:${PYTHONPATH:-}"

case "$MODE" in
  parity)   python -u scripts/parity/run_parity.py run --index "$SLURM_ARRAY_TASK_ID" --out "$OUT/parity" ;;
  pretrain) python -u scripts/pretrained/train_pretrained.py train --horizon "$SLURM_ARRAY_TASK_ID" --out "$OUT/pretrained" ;;
  transfer) python -u scripts/pretrained/train_pretrained.py transfer --horizon "$SLURM_ARRAY_TASK_ID" --out "$OUT/pretrained" ;;
  live_backtest) python -u scripts/live_forecast.py backtest-part --city "${CITY:?}" --index "$SLURM_ARRAY_TASK_ID" ${LIVE_ARGS:-} ;;
  live_forecast) python -u scripts/live_forecast.py forecast-part --city "${CITY:?}" --horizon "$SLURM_ARRAY_TASK_ID" ${LIVE_ARGS:-} ;;
  live_assemble) python -u scripts/live_forecast.py assemble --city "${CITY:?}" ${LIVE_ARGS:-} ;;
  live_train) python -u scripts/live_forecast.py train --city "${CITY:?}" --models gnn_st --horizon "$SLURM_ARRAY_TASK_ID" ${LIVE_ARGS:-} ;;
  exp_window) python -u scripts/training_window.py run-part --index "$SLURM_ARRAY_TASK_ID" ;;
  exp_frozen) python -u scripts/frozen_check.py run-part --index "$SLURM_ARRAY_TASK_ID" ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac
