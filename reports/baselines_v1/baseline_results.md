# Domaino baseline results (D2.1)

Representation: frozen `esm2_t33_650M_UR50D` (1280-dim per residue).
Trained on 4000 chains, scored on 2000 validation chains.

These are the floors every later design choice has to beat.

## Segmentation

| baseline | group | chains | mean IoU | %correct@0.8 | count MAE | boundary err |
|---|---|---|---|---|---|---|
| single_domain | all | 2000 | 0.7138 | 0.4745 | 0.295 | 33.914 |
| single_domain | single_domain | 1507 | 0.8609 | 0.8062 | 0.0 | 20.024 |
| single_domain | multidomain | 493 | 0.264 | 0.0129 | 1.1968 | 76.37 |
| single_domain | discontinuous | 202 | 0.2976 | 0.0285 | 1.0842 | 76.114 |
| single_domain | long | 193 | 0.3346 | 0.0785 | 0.7824 | 137.702 |
| boundary | all | 2000 | 0.6373 | 0.3023 | 2.053 | 6.874 |
| boundary | single_domain | 1507 | 0.6426 | 0.3212 | 1.8388 | 6.125 |
| boundary | multidomain | 493 | 0.6208 | 0.2761 | 2.7079 | 9.161 |
| boundary | discontinuous | 202 | 0.5653 | 0.1805 | 3.0347 | 10.968 |
| boundary | long | 193 | 0.5247 | 0.1483 | 6.7617 | 11.017 |

The learned boundary predictor reaches mean IoU 0.6373 against 0.7138 for assuming one domain, so on this subset it does **not** beat the trivial baseline overall -- most chains really are single-domain, and it over-segments them (domain-count error 2.053 against 0.295). It is far better at placing an edge once one exists (boundary error 6.874 against 33.914 residues) and clearly better on multidomain chains. Decision thresholds are deliberately left untuned here; the proposal tunes them on validation data in T3.5.

## CATH classification on ground-truth segments

Trained on 5356 domains, scored on 2508.

| level | classes | accuracy | macro F1 | eval labels unseen in training |
|---|---|---|---|---|
| C | 5 | 0.7859 | 0.4674 | 0 |
| A | 36 | 0.4071 | 0.0713 | 2 |
| T | 426 | 0.1818 | 0.0053 | 314 |
| H | 883 | 0.0502 | 0.0046 | 850 |

Accuracy far above macro F1 means the classifier is riding the common superfamilies and failing the rare ones, which is the imbalance the proposal plans class-balanced loss for.

Note the last column: 850 of 2508 evaluation domains (34%) belong to a homologous superfamily absent from this training subset, so they cannot be predicted at all and cap H-level accuracy at about 66%. That is a consequence of the small subset milestone M2.1 asks for combined with the homology-aware split, which keeps whole S35 clusters on one side. It is the concrete reason the proposal reports coverage next to accuracy and abstains on low-confidence H predictions rather than forcing a label.
