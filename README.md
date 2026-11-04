# Domaino

Protein sequence based domain identification and classification.

Domaino predicts, from an amino-acid sequence alone, where each domain of a
protein begins and ends -- including domains made of several separated sequence
intervals -- and assigns every detected domain a hierarchical CATH
classification (Class, Architecture, Topology, Homologous superfamily).

COMP 491 Computer Engineering Design, Koç University. Advisor: Prof. Dr. Attila
Gürsoy.

## Repository layout

```
configs/            versioned dataset build configurations
data/
  raw/              downloaded source releases      (git-ignored)
  processed/        generated record files          (git-ignored)
  manifests/        checksums, splits, audit reports, data cards (tracked)
references/         proposal and background papers
scripts/            data download, verification and Slurm wrappers
src/domaino/data/   work package 1: the data pipeline
tests/              unit tests
```

Everything under `data/manifests/` is deliberately tracked in git: the split id
lists, audit report, data card, checksums and lock file are what make an
experiment reconstructable later.

## Requirements

The data pipeline uses only the Python standard library, so any interpreter
from **Python 3.9** upwards works. The cluster's default `python3` is 3.6, so
point `PYTHON` at a newer one, for example a conda environment:

```bash
module load anaconda3/2025.06
conda create -n domaino python=3.11
```

## 1. Download the data

Downloads are long-running; start them inside tmux.

```bash
tmux new -s domaino-download
./scripts/download_raw_datasets.sh
```

The script is idempotent -- it skips files already present -- and records the
release, upstream URL and role of every source in
`data/manifests/source-versions.tsv`, with SHA-256 sums in
`data/manifests/raw-datasets.sha256`. Only newly fetched files are
checksum-verified, so re-running it is cheap. To re-hash everything and catch
corruption or upstream drift:

```bash
sbatch --partition=kutem --account=kutem --mem=4G --time=02:00:00 \
       --wrap "scripts/verify_raw_datasets.sh"
```

### Sources

| Source | Role | Provides | Note |
|---|---|---|---|
| CATH 4.4.0 | primary ground truth | 740,292 domain choppings over 493,969 chains; C/A/T/H labels and S35–S100 clusters for 601,328 | domain sequences only, no chain sequences |
| PDB SEQRES | chain sequences | the whole-chain sequences CATH's choppings index into | rebuilt weekly upstream; pinned by retrieval date and our checksum |
| Pfam 38.2 | auxiliary region evidence | ~149M family regions on UniProt sequences | no CATH labels, no sequences |
| TED v5 | lower-weight pretraining | per-chain choppings and predicted CATH labels from AlphaFold models | no sequences |
| Chainsaw CATH1363 | segmentation benchmark hold-out | exact train/validation/test chain lists | pinned to an upstream commit |
| CATHe2 | classification benchmark hold-out | test-domain identifiers | only the 36.8 GB Zenodo archive carries the identifiers |

## 2. Build a dataset

```bash
PYTHON=~/.conda/envs/domaino/bin/python sbatch scripts/build_dataset.slurm
```

Stages run individually, which matters because the Pfam and TED passes read
tens of gigabytes:

```bash
sbatch scripts/build_dataset.slurm cath,splits,report   # fast path
sbatch scripts/build_dataset.slurm pfam,ted             # the long scans
```

Expensive cross-source scans cache their results, so a rerun skips them unless
`--overwrite` is passed. Outputs land in `data/processed/<version>/` (records)
and `data/manifests/<version>/` (splits, audit report, data card, lock file).

## How the data is represented

One record is **one protein chain as a single source annotates it**. Every
domain segment is a **1-based inclusive interval into that record's own
`sequence`**. All three sources already number residues this way, so the
pipeline validates coordinates rather than shifting them.

Each record carries a `supervision` block saying which training targets it may
contribute to. The decisive field is `complete_partition`: CATH chops an entire
chain, so a residue outside every domain is genuine linker, whereas Pfam
annotates only the regions it recognises, so an unannotated residue there is
*unknown* and must never be used as a negative example. This is what keeps the
three sources from contradicting each other.

| Source | residue labels | pairwise | CATH labels | complete partition |
|---|---|---|---|---|
| CATH | yes | yes | yes, where classified | yes |
| Pfam | positive regions only | no | no | no |
| TED | yes | high/medium consensus only | predicted, weak | yes |
