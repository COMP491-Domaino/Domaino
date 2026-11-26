# CHAINSAW CATH1363 benchmark (task T3.7)

Generated 2026-11-25T19:55:37Z

Chains: 1355 of Chainsaw's 1363 test chains scored. 8 are absent from `dataset_v1` (their PDB entries were obsoleted after CATH 4.4.0 was built) and 0 had no cached embedding.

## Headline, all scored chains

| Method | Input | mean IoU | correct@0.8 (per chain) | count MAE |
|---|---|---|---|---|
| Chainsaw v3, re-scored | structure | 0.8531 | 0.7648 | 0.1342 |
| Domaino pairwise | sequence | 0.6957 | 0.5758 | 0.4974 |
| Domaino query | sequence | 0.6037 | 0.4627 | 0.6723 |
| Chainsaw v3, as published | structure | 0.8812 | 0.7877 | — |

The last row is Chainsaw's own published column. The row above it is Chainsaw's **same predictions** re-scored with our metric code. The gap between the two is the metric definition, not the model, which is why Domaino is compared against the re-scored row.

## By chain category, mean IoU

| Method | all | single domain | multidomain | continuous | discontinuous |
|---|---|---|---|---|---|
| Chainsaw v3, re-scored | 0.8531 (n=1349) | 0.8715 (n=731) | 0.8314 (n=618) | 0.8696 (n=1131) | 0.7678 (n=218) |
| Domaino pairwise | 0.6957 (n=1355) | 0.9369 (n=653) | 0.4713 (n=702) | 0.791 (n=1037) | 0.3848 (n=318) |
| Domaino query | 0.6037 (n=1355) | 0.923 (n=653) | 0.3067 (n=702) | 0.6976 (n=1037) | 0.2975 (n=318) |

## How comparable are the two ground truths

On the 1349 chains both cover, the two ground truths agree on the domain count for 1167 (86.5%). Mean signed difference in domain count, ours minus theirs: 0.1364. Mean shape distance 0.0121.

Their choppings index observed structure residues and ours index SEQRES, so each method is scored against the ground truth in its own coordinate space. These figures are what makes that defensible or not.

## Caveats

- Chainsaw reads 3D structure (pairwise Ca distances and predicted secondary structure); Domaino reads sequence alone. The gap is the measurement this project exists to make, not a like-for-like contest.
- Chainsaw's published IoU column cannot be reproduced from its own choppings by a residue-level domain IoU, and the metric details are in Supplementary Material we do not have. Both their figure and our recomputation of it are reported.
- The two ground truths index different coordinate spaces (observed structure residues versus SEQRES), so each method is scored against its own; see ground_truth_agreement.
- Domaino's post-processing thresholds are untuned defaults. T3.5 tunes them on validation data, so these numbers are a floor.
