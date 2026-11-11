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

## Reproducing the dataset from scratch

`data/raw/` and `data/processed/` are git-ignored, so a fresh clone has no data.
Run the four steps below to rebuild `dataset_v1` exactly. Budget **~67 GB of
disk** and **a few hours**, almost all of it download time.

### Step 0 -- prerequisites

The pipeline is pure Python standard library: no `pip install` needed, any
**Python 3.9+** works. The cluster's default `python3` is 3.6 and too old, so
make an environment once and keep its path handy:

```bash
module load anaconda3/2025.06
conda create -y -n domaino python=3.11
```

Everything below assumes `PYTHON=~/.conda/envs/domaino/bin/python`. On this
cluster `git` also needs `module load git/2.9.5`.

### Step 1 -- download the sources (~67 GB, several hours)

```bash
cd /path/to/Domaino
mkdir -p logs
setsid bash -c './scripts/download_raw_datasets.sh > logs/download.log 2>&1' &
```

Run it on the **login node**, not through Slurm -- compute nodes here may have
no outbound internet, and the job would sit in the queue anyway. `setsid`
detaches it so it survives your SSH session closing (`nohup` is disabled on
this cluster and `tmux` is not installed).

The script is **safe to re-run and resumes** where it left off, so if it dies
just start it again. It skips files already present, retries dropped
connections, verifies upstream MD5s, and writes
`data/manifests/source-versions.tsv` plus SHA-256 sums in
`data/manifests/raw-datasets.sha256`.

Watch progress — note that `squeue` will **not** show this, since it is not a
Slurm job:

```bash
tail -f logs/download.log
du -sh data/raw                    # should keep growing
```

The last file, CATHe2's `data.zip`, is 36.8 GB on its own and dominates the
wait. It is needed only for the classification benchmark hold-out, so you can
start Step 3 before it lands and rerun `splits,report` afterwards.

### Step 2 -- verify the downloads (optional, ~20 min)

Step 1 only checksums files it just fetched. To re-hash everything and catch
corruption or an upstream file changing underneath you:

```bash
sbatch --partition=kutem --account=kutem --mem=4G --time=02:00:00 \
       --wrap "scripts/verify_raw_datasets.sh"
```

### Step 3 -- build the dataset (~35 min)

```bash
PYTHON=~/.conda/envs/domaino/bin/python sbatch scripts/build_dataset.slurm
```

This runs all five stages and writes records to `data/processed/dataset_v1/`
and manifests to `data/manifests/dataset_v1/`. Stages can also run separately,
which is worth knowing because the Pfam and TED passes read tens of gigabytes:

| Stage | Reads | Time |
|---|---|---|
| `cath` | CATH + PDB SEQRES | ~2 min |
| `pfam` | 129M region rows | ~7 min |
| `ted` | 214M chain + 365M domain rows | ~23 min |
| `splits` | the CATH records | ~5 s |
| `report` | all records | ~2 min |

```bash
# just the fast path, or just the long scans
PYTHON=~/.conda/envs/domaino/bin/python sbatch scripts/build_dataset.slurm cath,splits,report
PYTHON=~/.conda/envs/domaino/bin/python sbatch scripts/build_dataset.slurm pfam,ted
```

Results are cached, so a rerun skips completed scans unless you pass
`--overwrite`. Check the job log in `logs/build_dataset-<jobid>.out`. Peak
memory is about 10 GB, well inside the 48 GB the Slurm script requests, so
nothing needs tuning.

### Step 4 -- confirm it worked

```bash
# 1. the leakage audit must say "clean", with no INACTIVE warnings
grep -E "verdict|INACTIVE" data/manifests/dataset_v1/audit_report.md

# 2. every split list must match the digests recorded in the lock file
PYTHONPATH=src ~/.conda/envs/domaino/bin/python - <<'PY'
import hashlib, json
d = json.load(open("data/manifests/dataset_v1/dataset.lock.json"))["split_digests"]
for part, expected in sorted(d.items()):
    ids = sorted(open(f"data/manifests/dataset_v1/splits/{part}.txt").read().split())
    got = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    print(("OK  " if got == expected else "FAIL"), part, len(ids))
PY

# 3. the unit tests
~/.conda/envs/domaino/bin/python -m unittest discover -s tests
```

A correct build reproduces these numbers exactly:

| | |
|---|---|
| records | 493,969 CATH · 165,231 TED · 9,063 Pfam |
| train / validation / test | 196,203 / 24,525 / 24,526 |
| benchmark_chainsaw / benchmark_cathe2 | 2,333 / 5,984 |
| excluded (homologous / no_cluster / unusable) | 117,877 / 85,697 / 36,824 |
| leakage audit | clean |
| tests | 84 passing |

If the CATHe2 archive was not downloaded, `benchmark_cathe2` is 0, the audit
report says the exclusion is INACTIVE, and the training set is larger
(262,991). That is a valid partial build, not a broken one.

### Sources

| Source | Role | Provides | Note |
|---|---|---|---|
| CATH 4.4.0 | primary ground truth | 740,292 domain choppings over 493,969 chains; C/A/T/H labels and S35–S100 clusters for 601,328 | domain sequences only, no chain sequences |
| PDB SEQRES | chain sequences | the whole-chain sequences CATH's choppings index into | rebuilt weekly upstream; pinned by retrieval date and our checksum |
| Pfam 38.2 | auxiliary region evidence | ~149M family regions on UniProt sequences | no CATH labels, no sequences |
| TED v5 | lower-weight pretraining | per-chain choppings and predicted CATH labels from AlphaFold models | no sequences |
| Chainsaw CATH1363 | segmentation benchmark hold-out | exact train/validation/test chain lists | pinned to an upstream commit |
| CATHe2 | classification benchmark hold-out | test-domain identifiers | only the 36.8 GB Zenodo archive carries the identifiers |

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
