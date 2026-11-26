#!/usr/bin/env bash
#SBATCH --job-name=domaino-cal
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cal-%j.out
set -euo pipefail
PYTHON="${PYTHON:-python3}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo
PYTHONPATH=src "${PYTHON}" scripts/calibrate_finetune.py \
    --config "${CONFIG:-configs/train_v1.json}" ${ARGS:-}
