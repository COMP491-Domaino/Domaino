#!/usr/bin/env bash
#SBATCH --job-name=domaino-t32
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/t32-%j.out
#
# Task T3.2: train one complete model on CATH.
#
#   ARCH=pairwise PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/train_t32.slurm
#   ARCH=query    SMOKE=1 PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/train_t32.slurm
#
# EXTRA passes anything else straight through, e.g. EXTRA=--source-aware for
# T2.6's composite loss or EXTRA=--no-classifier for segmentation only.
set -euo pipefail
PYTHON="${PYTHON:-python3}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# T3.1 requires every run to record its code revision, and git is not on PATH
# on the compute nodes, so the record would silently read "unknown".
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load git/2.9.5 2>/dev/null || true
command -v git >/dev/null || echo "WARNING: git unavailable; the run record cannot name its revision"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo
PYTHONPATH=src "${PYTHON}" -m domaino.train.run \
    --config configs/train_v1.json \
    --architecture "${ARCH:-pairwise}" \
    ${SMOKE:+--smoke} \
    ${EPOCHS:+--epochs "${EPOCHS}"} \
    ${DROPOUT:+--dropout "${DROPOUT}"} \
    ${EXTRA:-}
