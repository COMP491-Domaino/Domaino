"""Train and score the minimum reference baselines (task T2.1, deliverable D2.1).

    python -m domaino.model.run_baselines --config configs/baselines_v1.json

Trains on a subset of the training split and scores on the validation split,
which is what milestone M2.1 asks for ("train and produce valid predictions on
a small data subset"). Writes a baseline-results report that later work has to
beat.

Results are reported per chain category -- single-domain, multidomain,
continuous, discontinuous, long -- because a good average hides failure on
exactly the cases domain parsing exists for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .. import metrics
from ..data.schema import ChainRecord, read_jsonl
from . import baselines as bl
from .embed import load_config, split_members
from .embedding_store import EmbeddingStore
from .representation import EncoderConfig, Esm2Encoder

LONG_CHAIN = 500


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def categories(record: ChainRecord) -> List[str]:
    """Which reporting groups a chain belongs to."""
    names = ["all"]
    names.append("multidomain" if record.is_multidomain else "single_domain")
    names.append(
        "discontinuous" if record.has_discontinuous_domain else "continuous"
    )
    if record.length >= LONG_CHAIN:
        names.append("long")
    return names


def select(
    records_path: str, members: Sequence[str], limit: Optional[int], seed: int = 0
) -> List[ChainRecord]:
    """Pick a reproducible subset of a split, keeping its length spread."""
    import random

    keep = set(members)
    chosen = [
        r for r in read_jsonl(records_path) if r.chain_id in keep and r.sequence
    ]
    chosen.sort(key=lambda r: r.chain_id)
    if limit and limit < len(chosen):
        random.Random(seed).shuffle(chosen)
        chosen = chosen[:limit]
        chosen.sort(key=lambda r: r.chain_id)
    return chosen


class Embedder:
    """Serve residue embeddings from a cache, or compute them on demand."""

    def __init__(
        self, encoder_config: EncoderConfig, cache_dir: Optional[str]
    ) -> None:
        self.config = encoder_config
        self.store: Optional[EmbeddingStore] = None
        self._encoder: Optional[Esm2Encoder] = None
        if cache_dir and os.path.exists(os.path.join(cache_dir, "index.json")):
            self.store = EmbeddingStore(cache_dir)
            self.store.verify(encoder_config.model_name)
            log(f"using cached embeddings: {len(self.store)} chains in {cache_dir}")

    @property
    def encoder(self) -> Esm2Encoder:
        if self._encoder is None:
            self._encoder = Esm2Encoder(self.config)
        return self._encoder

    def get(self, records: Sequence[ChainRecord]) -> List[object]:
        """Embeddings for a batch of records, in order."""
        out: List[object] = [None] * len(records)
        missing: List[int] = []
        for i, record in enumerate(records):
            if self.store is not None and record.chain_id in self.store:
                out[i] = self.store.get(record.chain_id)
            else:
                missing.append(i)
        if missing:
            computed = self.encoder.embed([records[i].sequence for i in missing])  # type: ignore[misc]
            for i, embedding in zip(missing, computed):
                out[i] = embedding.numpy()
        return out


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run(config: Dict[str, object]) -> Dict[str, object]:
    encoder_config = EncoderConfig(**dict(config["encoder"]))  # type: ignore[arg-type]
    records_path = str(config["records"])
    manifest_dir = str(config["manifest_dir"])
    dim = encoder_config.embed_dim

    train = select(
        records_path,
        split_members(manifest_dir, "train"),
        int(config.get("train_limit", 4000)) or None,
        seed=int(config.get("seed", 0)),
    )
    val = select(
        records_path,
        split_members(manifest_dir, "validation"),
        int(config.get("eval_limit", 2000)) or None,
        seed=int(config.get("seed", 0)),
    )
    log(f"train subset: {len(train)} chains | eval subset: {len(val)} chains")

    cache_root = str(config.get("embeddings_root", ""))
    embedder = Embedder(
        encoder_config,
        os.path.join(cache_root, f"{encoder_config.model_name}-train")
        if cache_root
        else None,
    )
    batch = int(config.get("chains_per_batch", 32))
    report: Dict[str, object] = {
        "encoder": {
            "model_name": encoder_config.model_name,
            "dim": dim,
            "frozen": True,
        },
        "subsets": {"train": len(train), "validation": len(val)},
    }

    # -- baseline 1: one domain per chain, no training ----------------------
    log("baseline 1/3: single domain")
    single = bl.SingleDomainBaseline().fit()
    truths = [[[s.as_tuple() for s in d.segments] for d in r.domains] for r in val]
    single_preds = [[single.predict(r.length)] for r in val]
    report["single_domain"] = grouped_segmentation(val, single_preds, truths)

    # -- baseline 2: residue-level boundary predictor -----------------------
    # Both learned baselines need the same residue vectors, so each split is
    # embedded once and the two feature sets are collected in that one pass.
    log("embedding the training subset (one pass, shared by both baselines)")
    tolerance = int(config.get("boundary_tolerance", 2))
    features, labels, train_pooled, train_labels = collect_features(
        train, embedder, batch, tolerance
    )

    log("baseline 2/3: boundary predictor")
    boundary = bl.BoundaryBaseline(
        dim=dim,
        epochs=int(config.get("boundary_epochs", 5)),
        learning_rate=float(config.get("boundary_learning_rate", 0.05)),
        threshold=float(config.get("boundary_threshold", 0.5)),
        min_domain_length=int(config.get("min_domain_length", 40)),
        seed=int(config.get("seed", 0)),
    ).fit(features, labels)
    del features, labels
    log(f"  final training loss: {boundary.history[-1]['loss']}")

    # One pass over the validation subset serves the boundary predictions and
    # the classifier's pooled vectors alike.
    log("embedding the validation subset (one pass)")
    boundary_preds: List[List[List[Tuple[int, int]]]] = []
    val_rows: List[object] = []
    val_labels: Dict[str, List[str]] = {"C": [], "A": [], "T": [], "H": []}
    for group in chunked(val, batch):
        for record, embedding in zip(group, embedder.get(group)):
            boundary_preds.append([[iv] for iv in boundary.predict(embedding)])
            add_pooled(record, embedding, val_rows, val_labels)

    report["boundary"] = grouped_segmentation(val, boundary_preds, truths)
    report["boundary"]["training_loss"] = boundary.history

    # -- baseline 3: pooled classifier on true segments ---------------------
    log("baseline 3/3: pooled CATH classifier on ground-truth segments")
    if train_pooled is None:
        log("  no labelled domains in the training subset; skipping")
    else:
        import numpy as np
        classifier = bl.PooledDomainClassifier(
            dim=dim,
            epochs=int(config.get("classifier_epochs", 15)),
            learning_rate=float(config.get("classifier_learning_rate", 0.2)),
            seed=int(config.get("seed", 0)),
        ).fit(train_pooled, train_labels)
        val_pooled = np.stack(val_rows)
        predicted = classifier.predict_hierarchical(val_pooled)
        # Labels unseen in training can never be predicted; report that rather
        # than letting it look like ordinary error.
        unseen = {
            level: sum(
                1 for l in val_labels[level] if l not in set(classifier.classes[level])
            )
            for level in classifier.classes
        }
        scores = metrics.score_classification(val_labels, predicted)
        report["pooled_classifier"] = {
            "train_domains": len(train_pooled),
            "eval_domains": len(val_pooled),
            "levels": scores.levels,
            "eval_labels_unseen_in_training": unseen,
            "training_loss": {k: v[-1] for k, v in classifier.history.items()},
        }
    return report


def add_pooled(
    record: ChainRecord,
    embedding,
    rows: List[object],
    labels: Dict[str, List[str]],
) -> None:
    """Append one pooled vector per fully labelled domain of a chain.

    Only domains CATH resolves all the way to the homologous-superfamily level
    are used, so every level has a real target rather than a truncated code
    standing in for one.
    """
    for domain in record.domains:
        if not domain.cath or domain.label_depth != "H":
            continue
        rows.append(
            bl.pool_segments(embedding, [s.as_tuple() for s in domain.segments])
        )
        for level in ("C", "A", "T", "H"):
            labels[level].append(domain.cath.code(level))  # type: ignore[arg-type]


def collect_features(
    records: Sequence[ChainRecord],
    embedder: Embedder,
    batch: int,
    tolerance: int,
) -> Tuple[List[object], List[List[int]], Optional[object], Dict[str, List[str]]]:
    """Embed a split once and build both baselines' training inputs from it.

    Embedding is by far the most expensive step, so collecting the residue
    features and the pooled domain vectors together halves the cost compared
    with giving each baseline its own pass.
    """
    import numpy as np

    features: List[object] = []
    labels: List[List[int]] = []
    rows: List[object] = []
    pooled_labels: Dict[str, List[str]] = {"C": [], "A": [], "T": [], "H": []}

    for group in chunked(list(records), batch):
        for record, embedding in zip(group, embedder.get(group)):
            features.append(embedding)
            labels.append(
                bl.boundary_targets(
                    record.length,
                    [[s.as_tuple() for s in d.segments] for d in record.domains],
                    tolerance=tolerance,
                )
            )
            add_pooled(record, embedding, rows, pooled_labels)

    pooled = np.stack(rows) if rows else None
    return features, labels, pooled, pooled_labels


def grouped_segmentation(
    records: Sequence[ChainRecord],
    predictions: Sequence[Sequence[Sequence[Tuple[int, int]]]],
    targets: Sequence[Sequence[Sequence[Tuple[int, int]]]],
) -> Dict[str, object]:
    """Score segmentation overall and per chain category."""
    keys: List[str] = []
    flat_pred: List[object] = []
    flat_true: List[object] = []
    for record, prediction, target in zip(records, predictions, targets):
        for name in categories(record):
            keys.append(name)
            flat_pred.append(prediction)
            flat_true.append(target)
    out: Dict[str, object] = {}
    for name, (preds, truths) in metrics.stratify(keys, flat_pred, flat_true).items():
        out[name] = metrics.score_segmentation(preds, truths).as_dict()
    return out


def render(report: Dict[str, object]) -> str:
    """Render the baseline-results report."""
    lines = ["# Domaino baseline results (D2.1)", ""]
    encoder = dict(report.get("encoder", {}))
    subsets = dict(report.get("subsets", {}))
    lines += [
        f"Representation: frozen `{encoder.get('model_name')}` "
        f"({encoder.get('dim')}-dim per residue).",
        f"Trained on {subsets.get('train')} chains, scored on "
        f"{subsets.get('validation')} validation chains.",
        "",
        "These are the floors every later design choice has to beat.",
        "",
        "## Segmentation",
        "",
        "| baseline | group | chains | mean IoU | %correct@0.8 | count MAE | boundary err |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("single_domain", "boundary"):
        groups = report.get(name)
        if not isinstance(groups, dict):
            continue
        for group in ("all", "single_domain", "multidomain", "discontinuous", "long"):
            stats = groups.get(group)
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"| {name} | {group} | {stats.get('chains')} | "
                f"{stats.get('mean_iou')} | {stats.get('pct_correct_at_080')} | "
                f"{stats.get('domain_count_mae')} | "
                f"{stats.get('mean_boundary_error')} |"
            )
    single = report.get("single_domain")
    boundary = report.get("boundary")
    if isinstance(single, dict) and isinstance(boundary, dict):
        a = dict(single.get("all", {}))
        b = dict(boundary.get("all", {}))
        lines += [
            "",
            f"The learned boundary predictor reaches mean IoU "
            f"{b.get('mean_iou')} against {a.get('mean_iou')} for assuming one "
            f"domain, so on this subset it does **not** beat the trivial "
            f"baseline overall -- most chains really are single-domain, and it "
            f"over-segments them (domain-count error {b.get('domain_count_mae')} "
            f"against {a.get('domain_count_mae')}). It is far better at placing "
            f"an edge once one exists (boundary error "
            f"{b.get('mean_boundary_error')} against "
            f"{a.get('mean_boundary_error')} residues) and clearly better on "
            "multidomain chains. Decision thresholds are deliberately left "
            "untuned here; the proposal tunes them on validation data in T3.5.",
        ]
    lines += ["", "## CATH classification on ground-truth segments", ""]
    pooled = report.get("pooled_classifier")
    if isinstance(pooled, dict):
        lines += [
            f"Trained on {pooled.get('train_domains')} domains, scored on "
            f"{pooled.get('eval_domains')}.",
            "",
            "| level | classes | accuracy | macro F1 | eval labels unseen in training |",
            "|---|---|---|---|---|",
        ]
        unseen = dict(pooled.get("eval_labels_unseen_in_training", {}))
        for level, stats in dict(pooled.get("levels", {})).items():
            lines.append(
                f"| {level} | {stats.get('classes')} | {stats.get('accuracy')} | "
                f"{stats.get('macro_f1')} | {unseen.get(level, 0)} |"
            )
        total = pooled.get("eval_domains") or 0
        unseen_h = unseen.get("H", 0)
        lines += [
            "",
            "Accuracy far above macro F1 means the classifier is riding the "
            "common superfamilies and failing the rare ones, which is the "
            "imbalance the proposal plans class-balanced loss for.",
        ]
        if total and unseen_h:
            lines += [
                "",
                f"Note the last column: {unseen_h} of {total} evaluation "
                f"domains ({100 * unseen_h / total:.0f}%) belong to a "
                "homologous superfamily absent from this training subset, so "
                "they cannot be predicted at all and cap H-level accuracy at "
                f"about {100 * (total - unseen_h) / total:.0f}%. That is a "
                "consequence of the small subset milestone M2.1 asks for "
                "combined with the homology-aware split, which keeps whole S35 "
                "clusters on one side. It is the concrete reason the proposal "
                "reports coverage next to accuracy and abstains on "
                "low-confidence H predictions rather than forcing a label.",
            ]
    else:
        lines.append("Not run: no labelled domains in the subset.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = run(config)

    out_dir = args.out_dir or str(config.get("report_dir", "."))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "baseline_results.json"), "wt", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    with open(os.path.join(out_dir, "baseline_results.md"), "wt", encoding="utf-8") as fh:
        fh.write(render(report))
    log(f"wrote baseline_results.json and .md to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
