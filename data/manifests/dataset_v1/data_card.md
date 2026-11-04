# Domaino dataset dataset_v1 -- data card

## What this dataset is for

Training and evaluating a sequence-only model that segments a protein chain into domains and assigns each domain a CATH classification. One record is one protein chain as a single source annotates it.

## Sources

| source | version | role |
|---|---|---|
| CATH | 4.4.0 | primary ground truth |
| Pfam | 38.2 | auxiliary region supervision |
| TED | v5 | lower-confidence pretraining |

## Contents

| source | records | usable for training | domains |
|---|---|---|---|
| cath | 493969 | 456876 | 740266 |
| pfam | 9063 | 8786 | 66955 |
| ted | 165231 | 165231 | 249422 |

## Coordinate convention

Every domain segment is a 1-based inclusive interval into the record's own `sequence`. For CATH the sequence is the PDB SEQRES record of the chain and the intervals come from `cath-domain-boundaries-seqreschopping`; that pairing is canonical because it is the only one that yields a whole-chain sequence with coordinates defined on it.

## How supervision differs by source

Each record carries a `supervision` block stating which targets it may contribute to. The decisive field is `complete_partition`: CATH chops an entire chain, so residues outside every domain are genuine linker, whereas Pfam annotates only the regions it recognises, so an unannotated residue there is unknown and must never be used as a negative example.

| source | residue labels | pairwise | CATH labels | complete partition |
|---|---|---|---|---|
| cath | yes | yes | yes, where classified | yes |
| pfam | positive regions only | no | no | no |
| ted | yes | high/medium consensus only | predicted, weak | yes |

## Partitions

| partition | chains |
|---|---|
| benchmark_chainsaw | 2333 |
| excluded_homologous | 40192 |
| excluded_no_cluster | 85697 |
| excluded_unusable | 37008 |
| test | 32874 |
| train | 262991 |
| validation | 32874 |

Chains are grouped into homology components by shared CATH S35 sequence cluster and whole components are assigned together, so no close homologue spans a training and an evaluation partition. A chain that shares an S35 cluster with a published benchmark test chain -- that is, one homologous to it at 35% identity or better -- is never used for training.

## Known limitations

- CATH 4.4.0 references PDB entries that have since been obsoleted and removed from `pdb_seqres`; those chains have no sequence and are excluded from training.
- 85697 chains have domain boundaries but no classified domain, so they carry no S35 cluster and their homology to the rest of the dataset cannot be established. They are held out of every partition rather than risking leakage. Clustering them with MMseqs2, which the proposal names as the alternative to CATH's own clusters, would recover them as segmentation-only supervision.
- CATH's own files disagree about domain extent for a substantial minority of domains. Disagreements of a few residues at domain termini are kept and recorded; anything larger is excluded.
- Pfam and TED ship no sequences. Only their records whose sequence MD5 matches a chain we already hold carry a sequence and can train.
- TED boundaries derive from predicted structures and carry low weight.
- PDB and CATH over-represent proteins that are easy to crystallise, so results should be read per class, length and domain count.
