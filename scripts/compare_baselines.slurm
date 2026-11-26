#!/usr/bin/env bash
#SBATCH --job-name=domaino-cmp
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cmp-%j.out
#
# Score a trained model against the T2.1 baselines on the baselines' own
# 2,000-chain evaluation set, so a progress report can put them side by side.
#
#   CKPT=data/.../t32-pairwise-...-best.pt \
#     PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/compare_baselines.slurm
set -euo pipefail
PYTHON="${PYTHON:-python3}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo
PYTHONPATH=src "${PYTHON}" -m domaino.train.compare_baselines \
    --config "${CONFIG:-configs/train_v1.json}" \
    --checkpoint "${CKPT}"
