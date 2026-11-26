# Domaino against the T2.1 baselines

Generated 2026-11-26T07:24:57Z

Every column below is the **same 2000 validation chains** — the set `configs/baselines_v1.json` fixed with `eval_limit`. The T3.2 run records report the whole 24,525-chain split instead, so those numbers are not the ones to put beside a baseline.

Model: `pairwise` from run `t32-pairwise-a65bdee1b5ab`, epoch 13, selected on val_mean_iou 0.8543.

## Segmentation

| Segmentation metric | Baseline 1: one domain | Baseline 2: boundary predictor | Domaino |
|---|---|---|---|
| Mean IoU | 0.7138 | 0.6373 | **0.8858** |
| Correct at IoU >= 0.8 | 47.45% | 30.23% | **76.46%** |
| Domain-count error (MAE) | 0.295 | 2.053 | **0.132** |
| Boundary error (residues) | 33.9 | 6.9 | **7.7** |

## CATH classification on ground-truth segments

2534 domains. Domaino's H-level coverage is 64.9%: it abstains on the rest rather than forcing a superfamily label.

| CATH level | Baseline accuracy | Baseline macro F1 | Domaino accuracy | Domaino macro F1 |
|---|---|---|---|---|
| C | 78.59% | 46.74% | **88.95%** | **66.36%** |
| A | 40.71% | 7.13% | **77.35%** | **70.51%** |
| T | 18.18% | 0.53% | **63.30%** | **45.45%** |
| H | 5.02% | 0.46% | **53.08%** | **41.06%** |

The baseline was trained on a 4,000-chain subset, which left 850 of its evaluation domains (34%) in a superfamily it had never seen and capped its H accuracy near 66%. Domaino trained on the whole split, so that cap is the first thing the comparison removes; the remaining gap is the model.

## By chain category (Domaino)

| group | chains | mean IoU | correct@0.8 | count MAE | boundary |
|---|---|---|---|---|---|
| all | 2000 | 0.8858 | 0.7646 | 0.1315 | 7.656 |
| single_domain | 1586 | 0.9524 | 0.9571 | 0.0126 | 3.08 |
| multidomain | 414 | 0.6306 | 0.4531 | 0.5870 | 25.185 |
| continuous | 1819 | 0.9154 | 0.8547 | 0.0797 | 5.574 |
| discontinuous | 181 | 0.5877 | 0.3522 | 0.6519 | 28.576 |
| long | 115 | 0.5869 | 0.3038 | 0.8435 | 37.561 |
