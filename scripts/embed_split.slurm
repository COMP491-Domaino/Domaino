#!/usr/bin/env bash
#SBATCH --job-name=domaino-embed
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/embed-%j.out
#
# Precompute frozen ESM-2 residue embeddings for one dataset split (task T2.2).
#
#   PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/embed_split.slurm train
#   PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/embed_split.slurm validation 4000
#
# Weights come from the TORCH_HOME cache named in the config, so the compute
# node never needs outbound internet.

set -euo pipefail

SPLIT="${1:-train}"
LIMIT="${2:-}"
CONFIG="${CONFIG:-configs/representation_v1.json}"
PYTHON="${PYTHON:-python3}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "host:   $(hostname)"
echo "gpu:    ${CUDA_VISIBLE_DEVICES:-none}"
echo "split:  ${SPLIT} ${LIMIT:+(limit ${LIMIT})}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

PYTHONPATH=src "${PYTHON}" -m domaino.model.embed \
    --config "${CONFIG}" --split "${SPLIT}" ${LIMIT:+--limit "${LIMIT}"}
