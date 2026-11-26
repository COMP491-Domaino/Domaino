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
configs/            versioned build and model configurations
data/
  raw/              downloaded source releases      (git-ignored)
  processed/        generated records and embedding caches (git-ignored)
  manifests/        checksums, splits, audit reports, data cards (tracked)
references/         proposal and background papers
reports/            baseline and benchmark results  (tracked)
scripts/            download, verification and Slurm wrappers
src/domaino/
  data/             work package 1: the data pipeline
  model/            work package 2: representation, baselines, model, losses
  train/            work package 3: protocol, registry, trainer
  service/          work package 4: the backend inference service
  metrics.py        segmentation and classification measures
docs/
  api/              the versioned API contract (T4.1)
  wp4/              user flow and wireframes (T4.1)
web/mock/           the mocked interface (T4.1)
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

## Sequence representation and baselines (work package 2)

### ESM-2 residue embeddings (T2.2)

A frozen ESM-2 encoder turns a sequence into one contextual vector per
residue, which is the only input the model is allowed. Weights are read from
the `torch_home` cache named in `configs/representation_v1.json`, so compute
nodes need no internet.

```bash
PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/embed_split.slurm train
PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/embed_split.slurm validation 4000
```

Caching is optional -- a run can embed on the fly -- but the encoder is frozen,
so a chain's vectors never change and are worth storing once. Use
`--dry-run` to size a cache first; the training split at 650M is about 140 GB
in fp16. The cache is a few large memory-mapped fp16 shards plus an index,
because one file per chain would mean hundreds of thousands of tiny files,
which this filesystem handles badly.

Three details the pipeline handles, each easy to get wrong:

- fair-esm returns `(batch, 2 + max_len, dim)`; residue `i` sits at index `i`
  and the padding tail belongs to no residue. Embeddings are sliced back to
  each sequence's own length.
- ESM-2 was trained on 1022-residue crops but chains here reach 4,128, so
  longer ones are embedded in **overlapping windows** whose shared positions
  are averaged.
- Sequences are batched by a **token budget**, not a fixed count, so a batch of
  long chains cannot blow up memory and short chains are not padded out to the
  longest member.

A cache records which model and layer wrote it, and `EmbeddingStore.verify`
refuses a mismatch: silently mixing 650M and 150M vectors would train a model
on incoherent inputs.

### Minimum reference baselines (T2.1)

```bash
PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/run_baselines.slurm
```

Writes `reports/baselines_v1/baseline_results.{md,json}` (deliverable D2.1).
Three baselines, each answering a different question:

| Baseline | What it establishes |
|---|---|
| whole chain as one domain | the floor that is genuinely hard to beat, since most CATH chains are single-domain |
| residue-level boundary predictor | whether sequence context alone can locate a domain edge |
| pooled-embedding CATH classifier on **ground-truth** segments | classification performance isolated from segmentation error |

The third deliberately takes true segments: without that separation there is
no way to tell later whether an end-to-end mistake came from the parse or the
label. All three are linear models trained by plain minibatch descent, kept
simple so they stay a floor rather than a competitor.

Scores use the proposal's measures -- mean IoU, percentage correctly parsed at
IoU ≥ 0.8, boundary-position error and domain-count error for segmentation;
accuracy and macro F1 per CATH level for classification -- and are reported per
chain category (single-domain, multidomain, discontinuous, long), because a
good average hides failure on exactly the cases domain parsing is for.

### Domain-segmentation network (T2.3)

The proposal's first, lower-risk design. It reads frozen ESM-2 vectors and
emits two things:

- **residue logits** `(batch, length, 3)` — linker, inside a domain, or at a
  boundary
- **pair logits** `(batch, length, length)` — whether two residues are in the
  same domain

Both exist because boundary prediction alone cannot express a **discontinuous**
domain, and the T2.1 baseline showed the concrete failure: a residue-level
predictor over-segments (domain-count error 2.05 vs 0.30 for assuming one
domain) because nothing tells it which side of a cut belongs with which. A
same-domain relation fixes that and is invariant to how domains are numbered,
so the network never learns CATH's arbitrary ordering.

The trunk is a dilated convolutional stack: linear in length, with a receptive
field wider than a typical domain. The pair head is **low-rank** — two
`(length, rank)` projections whose product gives the matrix — because a
`(length, length, channels)` tensor at 4,128 residues is not affordable; the
proposal names a low-rank pair representation as the fallback for exactly this
risk. Training scores a bounded **sample** of pairs rather than all of them,
and `score_pairs` is tested to agree with the full matrix, since training on a
different function than inference uses would be a silent mismatch.

Pair targets are masked where either residue is linker: whether a linker
residue is "in the same domain" as anything has no answer, and supervising it
as negative would teach the model that linkers repel everything.

### Query-based segmentation, the second candidate (T2.3)

The proposal names two segmentation designs and picks between them on measured
performance in WP3. `src/domaino/model/query_segmentation.py` is the second:
a fixed set of learned **domain queries** attends to the ESM-2 residue vectors
through a small transformer decoder, and each query predicts a **residue
membership mask** plus a **domain / no-domain** score. **Bipartite (Hungarian)
matching** pairs the unordered predicted masks with the true domains during
training, so the model is never asked to reproduce CATH's arbitrary domain
numbering.

What it buys: a mask can select separated positions, so a discontinuous domain
is expressed directly — no pair matrix, and **no clustering step**; decoding is
thresholding plus the same length and gap tidying the other decoder applies.
Its output grows with queries × length rather than length².

What it costs, exactly as the proposal warns: a **fixed maximum number of
domains**. `num_queries` defaults to **20** because the most domains any chain
in dataset_v1 carries is 20 (`3vkgA`), so the limit does not bind on this data;
a chain with more domains than queries would lose some.

Both candidates read the same `trunk.ResidueTrunk`, so comparing them compares
the heads and nothing else. Run either:

```bash
ARCH=pairwise PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/smoke_segmentation.slurm 24 300
ARCH=query    PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/smoke_segmentation.slurm 24 300
```

### Domain assignment from network outputs (T2.4)

`DomainDecoder` turns those two outputs into domain intervals:

1. residues called linker are unassigned;
2. assigned stretches are cut at **boundary peaks** — essential, because
   adjacent CATH domains often touch with no linker (`1chmA01` is 2–156 and
   `1chmA02` is 157–402), so splitting only on linkers could never separate
   them;
3. the resulting blocks are grouped by average-linkage agglomerative clustering
   on the pairwise probabilities. Grouping *blocks* rather than cutting a
   sequence is what lets a domain be discontinuous — a cluster may hold blocks
   far apart. Average linkage with a similarity cut is used rather than a
   method needing the domain count up front, since the domain count is one of
   the things being predicted;
4. the proposal's four constraints are applied: **minimum domain length**
   (a fragment is absorbed into whichever domain it most resembles, not
   deleted), **acceptable linker gaps**, **residue coverage** (if too little of
   the chain is assigned, leftovers go to the nearest domain), and
   **consistency** (in range, sorted, no residue in two domains — an overlap is
   not expressible as a CATH chopping).

Every threshold is a post-processing knob the proposal tunes on validation data
in T3.5; the defaults are plain starting points, not tuned values. The decoder
always returns at least one domain, because every CATH chain has one.

Run the whole path on real chains:

```bash
PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/smoke_segmentation.slurm 24 300
```

### Hierarchical CATH classifier (T2.5)

`model/classifier.py` classifies a pooled domain embedding at all four CATH
levels at once. Two decisions matter:

* **Paths are decoded together, not per level.** The chosen Architecture must
  be a child of the chosen Class, and so on, so the answer is always a node
  CATH actually contains. Deciding each level independently produces
  combinations that do not exist.
* **H abstains.** A superfamily label is withheld unless that superfamily had
  at least `min_h_support` training domains *and* clears a confidence
  threshold. `h_coverage` is reported next to accuracy, because a model that
  answers only when sure otherwise looks better than one that always answers.

The label space is built from the **training split alone**. A classifier able
to emit a superfamily it never trained on would be scored on a label space the
validation split defined, which is a quiet form of leakage.

### Source-aware targets and losses (T2.6)

`model/losses.py` splits the training objective into the four separate losses
the proposal asks for, each masked by what the record's source is entitled to
supervise:

| Term | What it asks | CATH | Pfam | TED |
|---|---|---|---|---|
| residue assignment | is this residue inside a domain? | every residue | **annotated regions only** | every residue (high/medium consensus) |
| boundary | is this residue at a domain edge? | every residue | within annotated regions | every residue |
| pairwise grouping | do these two residues share a domain? | yes | **never** | high/medium consensus only |
| CATH classification | which superfamily? | yes | **never** | weakly, where TED resolved one |

Two consequences are worth knowing before reading a result:

* Pfam contributes **positives and no negatives** to the assignment term,
  because an unannotated residue is unknown rather than linker. That is a real
  bias, and it is why its record weight is 0.3 rather than 1.0.
* The **query architecture cannot use an incomplete source at all.** Set
  prediction supervises every unmatched query to answer "no domain", which is
  only true if the targets are every domain in the chain. So Pfam is a
  per-chain gate there, not a per-residue mask, while the pairwise design can
  still learn from it.

The assignment and boundary terms are binary cross-entropies on **marginals**
of the one 3-way residue distribution — P(in domain) = p(domain) + p(boundary),
P(boundary) = p(boundary) — so they need no extra parameters and do not change
the checkpoint format.

The switch is off by default (`"source_aware": false` in
`configs/train_v1.json`), because T3.2 trains on CATH alone, where every mask
is open, and turning it on would make the runs already in
`reports/experiments/` irreproducible from that file. T3.3's Pfam and TED arms
require it:

```bash
ARCH=pairwise EXTRA=--source-aware \
  PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/train_t32.slurm
```

## Two datasets, and why

`dataset_v1` reserves both benchmarks. `dataset_v2` differs in exactly one
way: it does not reserve the CATHe2 classification benchmark or its S35
homologues.

| | dataset_v1 | dataset_v2 |
|---|---|---|
| train | 196,203 | **262,991** |
| validation / test | 24,525 / 24,526 | 32,874 / 32,874 |
| `benchmark_chainsaw` | 2,333 | 2,333 |
| `benchmark_cathe2` | 5,984 | — not reserved |
| excluded as benchmark homologues | 117,877 | 40,192 |
| CHAINSAW segmentation benchmark | valid | valid |
| CATHe2 classification benchmark | valid | **not usable** |

Reserving CATHe2 costs 78,545 training chains — a quarter of the training set
— because its test domains are spread across CATH's superfamilies. `dataset_v2`
declines to pay that, which is the right trade for a segmentation-first model
and the wrong one for a classification claim. Both exist, so neither claim has
to be given up: the `dataset_v1` models remain the ones any CATHe2 number
rests on.

The Chainsaw reservation is **not** configurable. It and its 1-hop homologues
are held out of both datasets, which is what keeps `reports/benchmarks/`
comparable across them, and the holdout is asserted rather than assumed — zero
of the 1,363 CATH1363 chains appear in either dataset's train, validation or
test lists.

Two things `dataset_v2` does **not** do:

* It does not recover the 85,697 chains that have boundaries but no classified
  domain. Those have no S35 cluster, so their homology to the Chainsaw
  benchmark cannot be established; training on them would put the
  *segmentation* benchmark at risk rather than just the classification one.
  Recovering them needs an MMseqs2 clustering pass first.
* It does not make `val_mean_iou` comparable to `dataset_v1`'s. The re-split
  moved 16,379 chains out of train and brought 83,167 in, so the two
  validation sets are different sets of chains. Compare the two datasets'
  models on the benchmark, never on their validation scores.

Building it re-runs only the split stage, over the same records:

```bash
CONFIG=configs/dataset_v2.json PYTHON=~/.conda/envs/ted/bin/python \
  sbatch scripts/build_dataset.slurm splits,report
```

## Training the complete models (work package 3)

### Experiment protocol and registry (T3.1)

`train/protocol.py` holds everything a run may not change: the seeds, the
dataset version (`dataset_v1`), the checkpoint-selection metric
(`val_mean_iou`), early-stopping patience, the per-run and total GPU-hour
budgets, and the reporting groups every result is broken down by
(`single_domain`, `multidomain`, `continuous`, `discontinuous`, `long`). It is
Python rather than a config file precisely so that a run cannot quietly
override it.

Every run writes a record to `reports/experiments/` carrying its id, config
hash, seed, git revision and dirtiness, environment, per-epoch history, the
selected checkpoint and the final metrics. The registry also tracks the
budget, and a run refuses to start once it is exhausted.

### Training a model (T3.2)

```bash
# both candidate architectures on dataset_v1, under identical conditions
ARCH=pairwise PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/train_t32.slurm
ARCH=query    PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/train_t32.slurm

# a few minutes on a small subset, to check the whole path runs
ARCH=pairwise SMOKE=1 PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/train_t32.slurm

# the winning architecture on all the data dataset_v2 makes available
ARCH=pairwise PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/train_t32_v2.slurm
```

A run records its own `dataset_version` from its config, so the registry
cannot attribute a `dataset_v2` result to `dataset_v1`, and the benchmark
runner picks its checkpoints per (architecture, dataset) for the same reason.

Both architectures train through one file (`train/trainer.py`) on purpose:
T3.2 requires them trained under the same conditions, and the surest way to
mean that is for the optimiser, batching, precision, early stopping and
selection rule to be literally the same code. They also share
`model/trunk.py`, so the comparison isolates the heads.

The classifier is trained on **ground-truth** segments: early in training the
predicted domains are noise, and classifying noise teaches nothing about CATH.
Evaluation then reports both modes, which is how a segmentation error is kept
distinguishable from a classification error.

Requires the embedding caches from T2.2 — a run refuses to start without them
rather than embedding on the fly at unpredictable speed.

### Benchmark evaluation (T3.7)

```bash
# the benchmark chains need their own embedding cache first (~30 s)
PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/embed_split.slurm benchmark_chainsaw
PYTHON=~/.conda/envs/ted/bin/python sbatch scripts/run_benchmark.slurm
```

Writes [`reports/benchmarks/chainsaw_cath1363.md`](reports/benchmarks/chainsaw_cath1363.md).
With no `CHECKPOINTS` the runner takes the best **full** run per architecture
from the experiment registry, skipping smoke runs — a two-minute smoke
checkpoint sitting in a benchmark report next to real ones is a number
somebody would quote.

Chainsaw's repository publishes its per-chain predictions *and* its per-chain
metric values on CATH1363, which makes a careful comparison possible and
revealed a trap. Their published `iou` column **cannot be reproduced** from
their own choppings by a residue-level domain IoU: recomputing it with our
code gives 0.853 where they report 0.881, and the disagreements concentrate in
chains where a predicted domain covers residues their ground truth leaves
unassigned, so their IoU is the more forgiving definition. The metric details
are in Supplementary Material we do not have.

So the report scores **their predictions with our code** and compares Domaino
against that row, with their published figure shown separately. Quoting
Domaino's number against their published one would move the comparison by
about three IoU points for reasons that have nothing to do with either model.

Two further things the report states rather than hides: Chainsaw reads 3D
structure while Domaino reads sequence alone, and the two ground truths index
different coordinate spaces (observed structure residues versus SEQRES), so
each method is scored against its own. `ground_truth_agreement` measures how
far apart they are — they agree on the domain count for 86.5% of chains, with
a mean domain-size-distribution distance of 0.012.

## Web platform (work package 4)

### User flow, wireframes and API contract (T4.1)

* [`docs/wp4/user_flow.md`](docs/wp4/user_flow.md) — the workflow from
  submission to interpretation, the wireframes, and the conventions
* [`docs/api/domaino-api-v1.json`](docs/api/domaino-api-v1.json) — the
  versioned API contract, also served live at `GET /api/v1/schema`
* `web/mock/index.html` — the interactive interface

The contract is not a document that describes the code; the backend **loads it
at runtime** and validates every request and every response against it, so a
drift between the two fails a test instead of reaching a browser.

### Backend inference service (T4.2)

```bash
# mocked predictions, with the interface, at http://127.0.0.1:8080/
PYTHONPATH=src ~/.conda/envs/ted/bin/python -m domaino.service --static web/mock

# a trained model from WP3
PYTHONPATH=src ~/.conda/envs/ted/bin/python -m domaino.service \
    --checkpoint data/processed/dataset_v1/checkpoints/<run-id>-best.pt \
    --encoder-config configs/representation_v1.json \
    --static web/mock
```

`--checkpoint` is the only difference between the mocked system and the real
one. Without it the service serves clearly labelled simulated results
(`model.mode == "mock"`), which is what makes the interface demonstrable
before training finishes; if a checkpoint fails to load, the service says so
and falls back rather than refusing to start.

The service runs on the standard library. No web framework is installed in
this cluster's environment and the compute nodes that hold the model and the
ESM-2 weights have no outbound internet, so a framework dependency would make
the service unrunnable exactly where it has to run.

A submitted sequence lives in its job and nowhere else, is dropped the moment
the prediction finishes, and cannot be serialised by `Job.as_json` — asserted
in `tests/test_service.py`. Results expire after an hour by default, the store
is bounded, and `DELETE` forgets a job immediately.

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
