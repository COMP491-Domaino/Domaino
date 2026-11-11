#!/usr/bin/env bash
#SBATCH --job-name=domaino-baselines
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/baselines-%j.out
#
# Train and score the minimum reference baselines (task T2.1, deliverable D2.1).
#
#   PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/run_baselines.slurm
#
# A GPU is requested because the baselines embed their subset with ESM-2. If a
# cache already exists for the split, the encoder is never loaded and this runs
# fine on CPU.

set -euo pipefail

CONFIG="${CONFIG:-configs/baselines_v1.json}"
PYTHON="${PYTHON:-python3}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "host:   $(hostname)"
echo "config: ${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

PYTHONPATH=src "${PYTHON}" -m domaino.model.run_baselines --config "${CONFIG}"
