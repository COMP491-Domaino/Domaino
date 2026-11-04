#!/usr/bin/env bash
#SBATCH --job-name=domaino-dataset
#SBATCH --partition=kutem
#SBATCH --account=kutem
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_dataset-%j.out
#
# Build a versioned Domaino dataset (work package 1).
#
# The CATH stage holds the SEQRES index and the published domain sequences in
# memory; the Pfam and TED stages stream ~150M and ~214M rows respectively.
# None of that belongs on a shared login node, hence this wrapper.
#
#   sbatch scripts/build_dataset.slurm                      # every stage
#   sbatch scripts/build_dataset.slurm cath,splits,report   # a subset

set -euo pipefail

STAGES="${1:-cath,pfam,ted,splits,report}"
CONFIG="${CONFIG:-configs/dataset_v1.json}"

# The pipeline is pure standard library, so any interpreter from 3.9 up works.
# The cluster's default python3 is older than that, so point PYTHON at a conda
# environment, e.g. PYTHON=~/.conda/envs/domaino/bin/python sbatch ...
PYTHON="${PYTHON:-python3}"

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
mkdir -p logs

if ! "${PYTHON}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "ERROR: ${PYTHON} is $("${PYTHON}" -V 2>&1); Python 3.9 or newer is required."
    echo "Set PYTHON to a suitable interpreter, for example:"
    echo "  PYTHON=~/.conda/envs/domaino/bin/python sbatch scripts/build_dataset.slurm"
    exit 1
fi

echo "host:    $(hostname)"
echo "config:  ${CONFIG}"
echo "stages:  ${STAGES}"
echo "python:  $(${PYTHON} -V 2>&1)"
echo

PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -m domaino.data.build --config "${CONFIG}" --stages "${STAGES}"
