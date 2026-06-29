#!/bin/bash
# Checkpoint 0 on Converge — the SAME run as notebooks/poc/bag_of_cells.ipynb and
# `python -m sp_ml.run`, submitted via SLURM. Submit from the repo root:
#
#     sbatch scripts/poc.sbatch                      # CPU (defq), offline W&B (default)
#     WANDB_MODE=online sbatch scripts/poc.sbatch    # live W&B sync
#     sbatch scripts/poc.sbatch model/graph=gcn      # forward Hydra overrides
#
# Defaults to CPU: the bag-of-cells logreg is light (~1 min) and CPU-runnable, and the
# `basic` tier allows only 1 A10G per user — a GPU job blocks (QOSMaxGRESPerUser) while
# the interactive GPU workspace holds it. To run on GPU when an A10G is free, override:
#     sbatch -p gpu --gres=gpu:a10g:1 scripts/poc.sbatch
#
#SBATCH --job-name=sp-poc
#SBATCH --account=basic
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=0-00:20
#SBATCH --output=logs/poc_%j.out
#SBATCH --error=logs/poc_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"                 # repo root (where sbatch was invoked)

source ../.venv/bin/activate
export WANDB_MODE="${WANDB_MODE:-offline}"  # offline-safe; sync later with `wandb sync`

srun python -m sp_ml.run "$@"
