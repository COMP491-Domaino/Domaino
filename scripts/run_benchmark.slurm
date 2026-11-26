#!/usr/bin/env bash
#SBATCH --job-name=domaino-bench
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench-%j.out
#
# Task T3.7: score trained checkpoints on the Chainsaw CATH1363 benchmark.
#
#   PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/run_benchmark.slurm
#
# Scores every T3.2 checkpoint it finds unless CHECKPOINTS names them.
# Needs the benchmark embedding cache:
#   PYTHON=... sbatch scripts/embed_split.slurm benchmark_chainsaw
set -euo pipefail
PYTHON="${PYTHON:-python3}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs reports/benchmarks

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load git/2.9.5 2>/dev/null || true
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# With no CHECKPOINTS, the runner picks the best full (non-smoke) run per
# architecture out of the experiment registry.
ARGS=()
for path in ${CHECKPOINTS:-}; do ARGS+=(--checkpoint "${path}"); done

PYTHONPATH=src "${PYTHON}" -m domaino.train.run_benchmark \
    --config "${CONFIG:-configs/train_v1.json}" ${ARGS[@]+"${ARGS[@]}"}
