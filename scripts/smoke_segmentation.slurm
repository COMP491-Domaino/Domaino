#!/usr/bin/env bash
#SBATCH --job-name=domaino-segsmoke
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/segsmoke-%j.out
#
# Run the sequence-to-domains path on real chains (T2.2 -> T2.3 -> T2.4).
#
#   PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/smoke_segmentation.slurm
set -euo pipefail
PYTHON="${PYTHON:-python3}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
echo
PYTHONPATH=src "${PYTHON}" scripts/smoke_segmentation.py \
    --config configs/segmentation_v1.json --chains "${1:-24}" --steps "${2:-300}"
