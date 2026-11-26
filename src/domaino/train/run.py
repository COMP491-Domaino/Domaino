"""Train the CATH-only reference and complete models (task T3.2).

    python -m domaino.train.run --config configs/train_v1.json \
        --architecture pairwise

    python -m domaino.train.run --config configs/train_v1.json \
        --architecture query --smoke

Each invocation trains one model under the fixed protocol in `protocol.py`,
writes its run record to the experiment registry, and saves the checkpoint the
validation metric selected. `--smoke` runs the same code on a small subset, so
a failure shows up in minutes rather than hours.

The reference model is the T2.1 baseline set, already reported in
`reports/baselines_v1/`; this trains the complete models that have to beat it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from ..model import classifier as clf
from ..model.embed import load_config, split_members
from ..model.embedding_store import EmbeddingStore
from . import protocol
from .data import EmbeddingBatcher, load_examples
from .trainer import ARCHITECTURES, TrainConfig, train


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_hierarchy(examples) -> clf.CathHierarchy:
    """The label space, built from the training split only.

    Built from training labels alone on purpose: a classifier that could emit a
    superfamily it never trained on would be scored on a label space the
    validation split defined, which is a quiet form of leakage.
    """
    labels: Dict[str, List[str]] = {level: [] for level in ("C", "A", "T", "H")}
    for example in examples:
        for level in labels:
            labels[level].extend(c for c in (example.labels.get(level) or []) if c)
    return clf.CathHierarchy.from_labels(labels)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--seed", type=int, default=protocol.SEEDS[0])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--no-classifier", action="store_true",
        help="train segmentation only, leaving the hierarchical classifier off",
    )
    parser.add_argument(
        "--source-aware", action="store_true",
        help="use T2.6's source-aware composite loss (required for Pfam/TED)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="small subset and few epochs: exercises the whole path quickly",
    )
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    import torch

    file_config = load_config(args.config)
    records_path = str(file_config["records"])
    manifest_dir = str(file_config["manifest_dir"])
    embeddings_root = str(file_config["embeddings_root"])
    model_name = str(dict(file_config["encoder"])["model_name"])

    overrides: Dict[str, Any] = dict(file_config.get("training", {}))
    overrides["architecture"] = args.architecture
    overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.no_classifier:
        overrides["train_classifier"] = False
    if args.source_aware:
        overrides["source_aware"] = True
    if args.smoke:
        # Sized so the loss has to move: ~4000 chains at a smaller token
        # budget with no accumulation gives hundreds of optimiser steps, where
        # the first attempt managed about three and learned nothing.
        overrides.update(
            {
                "epochs": 6,
                "train_limit": 4000,
                "val_limit": 500,
                "max_tokens": 12000,
                "accumulate_steps": 1,
            }
        )
    config = TrainConfig(**overrides)

    # Caches were built per split by T2.2; a run refuses to start without them
    # rather than silently embedding on the fly at unpredictable speed.
    stores = {}
    for split in ("train", "validation"):
        path = os.path.join(embeddings_root, f"{model_name}-{split}")
        if not os.path.exists(os.path.join(path, "index.json")):
            log(f"missing embedding cache for {split} at {path}")
            log("build it first: sbatch scripts/embed_split.slurm <split>")
            return 1
        store = EmbeddingStore(path)
        store.verify(model_name)
        stores[split] = store
        log(f"{split} cache: {len(store)} chains, dim {store.meta.dim}")

    train_examples = load_examples(
        records_path, split_members(manifest_dir, "train"),
        stores["train"], config.train_limit,
    )
    val_examples = load_examples(
        records_path, split_members(manifest_dir, "validation"),
        stores["validation"], config.val_limit,
    )
    log(f"train {len(train_examples)} chains | validation {len(val_examples)} chains")
    log(
        "loss: "
        + (
            "source-aware composite, T2.6 (assignment "
            f"{config.assignment_weight} / boundary {config.boundary_weight} / "
            f"pairwise {config.pairwise_weight})"
            if config.source_aware
            else "combined residue+pair, T2.3"
        )
    )
    if not train_examples or not val_examples:
        log("no usable chains; check the caches and manifests")
        return 1

    hierarchy = build_hierarchy(train_examples) if config.train_classifier else None
    if hierarchy is not None:
        log(f"label space from the training split: {hierarchy.sizes()}")
        log(
            f"superfamilies with at least {config.label_min_h_support} domains: "
            f"{len(hierarchy.well_supported_h(config.label_min_h_support))}"
        )

    registry = protocol.ExperimentRegistry(str(file_config["registry_dir"]))
    remaining = registry.budget_remaining()
    log(
        f"budget: {registry.spent_gpu_hours()} of "
        f"{protocol.BUDGET_GPU_HOURS} GPU-hours spent, {remaining} left"
    )
    if remaining <= 0 and not args.smoke:
        log("the T3.2 computing budget is exhausted; not starting another run")
        return 1

    # Read from the config rather than taken from `protocol`: a run on a
    # different dataset must say so in its own record, or the registry quietly
    # attributes every result to dataset_v1.
    dataset_version = str(file_config.get("dataset_version", protocol.DATASET_VERSION))
    identity = {
        "config": config.as_dict(),
        "dataset_version": dataset_version,
        "encoder": model_name,
    }
    record = protocol.RunRecord(
        run_id=protocol.run_id(identity, prefix=f"t32-{args.architecture}"),
        name=args.name or f"T3.2 {args.architecture}"
        + (" (smoke)" if args.smoke else ""),
        architecture=args.architecture,
        dataset_version=dataset_version,
        seed=config.seed,
        config=identity,
        code=protocol.code_revision("."),
        environment=protocol.environment(),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    if record.code.get("dirty") == "yes":
        record.notes.append(
            "working tree was dirty at launch, so the recorded revision does "
            "not fully describe the code that ran"
        )
    log(f"run id {record.run_id} | revision {record.code.get('revision')[:12]}")
    log(f"dataset {dataset_version}")
    registry.save(record)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batcher = EmbeddingBatcher(stores["train"], device=device)
    val_batcher = EmbeddingBatcher(stores["validation"], device=device)

    record = train(
        config=config,
        train_examples=train_examples,
        val_examples=val_examples,
        batcher=batcher,
        val_batcher=val_batcher,
        hierarchy=hierarchy,
        record=record,
        registry=registry,
        checkpoint_dir=str(file_config["checkpoint_dir"]),
        log=log,
    )

    log("")
    log(f"selected: {json.dumps(record.selected)}")
    seg = dict(record.results.get("segmentation", {}))
    for group in protocol.REPORT_GROUPS:
        row = seg.get(group)
        if not row:
            continue
        log(
            f"  {group:<14}n={row['chains']:5d}  IoU={row['mean_iou']:.4f}  "
            f"correct@0.8={row['pct_correct_at_080']:.4f}  "
            f"countMAE={row['domain_count_mae']:.4f}  "
            f"boundary={row['mean_boundary_error']}"
        )
    classification = record.results.get("classification_true_segments")
    if classification:
        log(f"  classification on true segments ({classification['domains']} domains), "
            f"H coverage {classification['h_coverage']}")
        for level, row in dict(classification["levels"]).items():
            log(
                f"    {level}: accuracy={row['accuracy']:.4f} "
                f"macro_f1={row['macro_f1']:.4f} classes={row['classes']}"
            )
    log(f"runtime {record.runtime_seconds / 60:.1f} min, {record.gpu_hours} GPU-hours")
    log(f"record: {registry.path_for(record.run_id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
