"""Score trained checkpoints on the CHAINSAW CATH1363 benchmark (task T3.7).

    python -m domaino.train.run_benchmark --config configs/train_v1.json \
        --checkpoint data/.../t32-pairwise-...-best.pt \
        --checkpoint data/.../t32-query-...-best.pt

Writes `reports/benchmarks/chainsaw_cath1363.{json,md}`: our models and
Chainsaw's own published predictions, all re-scored with one metric
implementation so the numbers can be compared.

Evaluation is on the benchmark hold-out, which no model has trained on: those
chains sit in `benchmark_chainsaw`, and every chain sharing an S35 cluster with
one of them was dropped from training too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.benchmarks import load_chainsaw_splits
from ..data.schema import read_jsonl
from ..model.embed import load_config
from ..model.embedding_store import EmbeddingStore
from . import benchmark as bench
from . import protocol


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_truth(records_path: str, wanted: set) -> Dict[str, Dict[str, Any]]:
    """Our ground truth for the wanted chains: domains, length, labels."""
    found: Dict[str, Dict[str, Any]] = {}
    for record in read_jsonl(records_path):
        if record.chain_id not in wanted or not record.sequence:
            continue
        domains = [[s.as_tuple() for s in d.segments] for d in record.domains]
        if not domains:
            continue
        found[record.chain_id] = {
            "length": record.length,
            "domains": domains,
            "sequence": record.sequence,
        }
    return found


def predict_all(
    checkpoint_path: str,
    truth: Dict[str, Dict[str, Any]],
    store: EmbeddingStore,
    max_tokens: int,
    device_name: Optional[str] = None,
) -> Tuple[Dict[str, List[List[Tuple[int, int]]]], Dict[str, Any]]:
    """Run one checkpoint over every chain, returning its predicted domains."""
    import torch

    from ..model import classifier as clf
    from ..model import decode as dec
    from ..model import query_segmentation as qs
    from ..model import segmentation as seg
    from .data import ChainExample, EmbeddingBatcher, iter_epoch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["state"]
    architecture = state["architecture"]
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if architecture == "pairwise":
        config = seg.SegmentationConfig(**state["seg_config"])
        network = seg.build_network(config)
    else:
        config = qs.QuerySegmentationConfig(**state["seg_config"])
        network = qs.build_query_network(config)
    network.load_state_dict(state["segmentation"])
    network.eval().to(device)

    info = {
        "architecture": architecture,
        "run_id": payload.get("run_id"),
        "epoch": payload.get("epoch"),
        protocol.CHECKPOINT_METRIC: payload.get(protocol.CHECKPOINT_METRIC),
        "parameters": sum(p.numel() for p in network.parameters()),
    }

    examples = [
        ChainExample(chain_id, entry["length"], entry["domains"])
        for chain_id, entry in sorted(truth.items())
        if chain_id in store
    ]
    batcher = EmbeddingBatcher(store, device=device)
    decoder = dec.DomainDecoder()
    predictions: Dict[str, List[List[Tuple[int, int]]]] = {}

    with torch.no_grad():
        for batch in iter_epoch(examples, max_tokens, seed=0, shuffle=False):
            features, mask, _ = batcher.build(batch)
            out = network(features, mask=mask)
            if architecture == "pairwise":
                residue = torch.softmax(out["residue_logits"].float(), dim=-1)
                residue = residue.cpu().numpy()
                for row, example in enumerate(batch):
                    pair = torch.sigmoid(
                        network.pair_head(
                            out["features"][row : row + 1, : example.length]
                        )[0].float()
                    ).cpu().numpy()
                    result = decoder.decode(residue[row, : example.length], pair)
                    predictions[example.chain_id] = [list(d) for d in result.domains]
            else:
                for row, example in enumerate(batch):
                    predictions[example.chain_id] = [
                        list(d)
                        for d in qs.decode_queries(
                            out["class_logits"][row].float().cpu(),
                            out["mask_logits"][row].float().cpu(),
                            example.length,
                        )
                    ]
    return predictions, info


def full_run_checkpoints(registry_dir: str) -> List[Dict[str, Any]]:
    """The best full T3.2 checkpoint per (architecture, dataset), from the registry.

    Selected from run records rather than by globbing the checkpoint
    directory, which also holds two-minute smoke models. A smoke checkpoint
    sitting in a benchmark report next to real ones is a number somebody would
    quote.

    Grouped by dataset as well as architecture, because `val_mean_iou` is only
    comparable within one dataset: dataset_v2 re-split the chains, so its
    validation set is not dataset_v1's. Picking "the best pairwise run" across
    both datasets would be choosing on a metric measured on two different
    test sets. Every qualifying run is scored instead, and the benchmark --
    which both datasets reserve identically -- does the comparing.
    """
    best: Dict[tuple, Dict[str, Any]] = {}
    for name in sorted(os.listdir(registry_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(registry_dir, name), "rt", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("status") != "completed" or "smoke" in (record.get("name") or ""):
            continue
        selected = record.get("selected") or {}
        path = selected.get("checkpoint")
        value = selected.get(protocol.CHECKPOINT_METRIC)
        if not path or value is None or not os.path.exists(path):
            continue
        key = (
            record.get("architecture") or "unknown",
            record.get("dataset_version") or protocol.DATASET_VERSION,
        )
        if key not in best or value > best[key][protocol.CHECKPOINT_METRIC]:
            best[key] = {
                "checkpoint": path,
                "architecture": key[0],
                "dataset_version": key[1],
                protocol.CHECKPOINT_METRIC: value,
                "run_id": record.get("run_id"),
            }
    return [best[key] for key in sorted(best)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint", action="append", default=[],
        help="a T3.2 checkpoint; repeat to score several models at once",
    )
    parser.add_argument(
        "--splits",
        default="data/raw/benchmarks/chainsaw/chainsaw_cath1363_train_test_splits.json",
    )
    parser.add_argument(
        "--published",
        default="data/raw/benchmarks/chainsaw/chainsaw_model_v3_on_cath1363_test.csv",
    )
    parser.add_argument("--embedding-split", default="benchmark_chainsaw")
    parser.add_argument("--out", default="reports/benchmarks")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    file_config = load_config(args.config)
    records_path = str(file_config["records"])
    model_name = str(dict(file_config["encoder"])["model_name"])
    max_tokens = int(dict(file_config.get("training", {})).get("max_tokens", 24000))

    checkpoints = [{"checkpoint": path} for path in args.checkpoint]
    if not checkpoints:
        checkpoints = full_run_checkpoints(str(file_config["registry_dir"]))
        if not checkpoints:
            log("no completed full run in the registry and no --checkpoint given")
            return 1
        log(f"selected {len(checkpoints)} full-run checkpoints from the registry")
        for entry in checkpoints:
            log(
                f"  {entry['architecture']} on {entry['dataset_version']}: "
                f"{entry['run_id']} "
                f"({protocol.CHECKPOINT_METRIC} {entry[protocol.CHECKPOINT_METRIC]})"
            )

    test_ids = set(load_chainsaw_splits(args.splits)["test"])
    log(f"Chainsaw CATH1363 test split: {len(test_ids)} chains")

    truth = load_truth(records_path, test_ids)
    log(f"{len(truth)} of them are in dataset_v1 with a sequence and domains")

    cache_path = os.path.join(
        str(file_config["embeddings_root"]), f"{model_name}-{args.embedding_split}"
    )
    if not os.path.exists(os.path.join(cache_path, "index.json")):
        log(f"missing embedding cache at {cache_path}")
        log(f"build it first: sbatch scripts/embed_split.slurm {args.embedding_split}")
        return 1
    store = EmbeddingStore(cache_path)
    store.verify(model_name)
    embedded = {c: e for c, e in truth.items() if c in store}
    log(f"{len(embedded)} have a cached embedding and will be scored")
    if not embedded:
        return 1

    published = bench.load_published(args.published)
    # Only the chains we actually score, so the published average is over the
    # same set rather than over their full 1,363.
    shared_rows = [published[c] for c in embedded if c in published]
    log(f"{len(shared_rows)} of those carry a published Chainsaw result")

    methods: Dict[str, Any] = {}

    # Chainsaw's own predictions, re-scored with our code and their ground
    # truth, which is the only row Domaino can fairly be compared against.
    chainsaw_rows: List[bench.BenchmarkRow] = []
    their_truth: Dict[str, List[List[Tuple[int, int]]]] = {}
    for chain, row in published.items():
        if chain not in embedded:
            continue
        truth_domains = bench.parse_chopping(row.get("true_chopping", ""), offset=1)
        if not truth_domains:
            continue
        their_truth[chain] = truth_domains
        chainsaw_rows.append(
            bench.BenchmarkRow(
                chain_id=chain,
                length=max(e for d in truth_domains for _, e in d),
                truth=truth_domains,
                predicted=bench.parse_chopping(row.get("pred_chopping", ""), offset=1),
            )
        )
    if chainsaw_rows:
        methods["Chainsaw v3, re-scored"] = {
            "input": "structure",
            "source": os.path.basename(args.published),
            "groups": bench.grouped_scores(chainsaw_rows),
        }
        log(f"Chainsaw re-scored on {len(chainsaw_rows)} chains")

    for entry in checkpoints:
        path = entry["checkpoint"]
        if not os.path.exists(path):
            log(f"no such checkpoint: {path}")
            return 1
        log(f"running {path}")
        predictions, info = predict_all(
            path, embedded, store, max_tokens, args.device
        )
        info["dataset_version"] = entry.get("dataset_version", "unknown")
        rows = [
            bench.BenchmarkRow(
                chain_id=chain,
                length=entry["length"],
                truth=entry["domains"],
                predicted=predictions.get(chain, []),
            )
            for chain, entry in sorted(embedded.items())
        ]
        # The dataset is in the name: two rows reading "Domaino pairwise"
        # trained on different amounts of data would be unreadable.
        name = f"Domaino {info['architecture']} / {info['dataset_version']}"
        methods[name] = {
            "input": "sequence",
            "checkpoint": path,
            "model": info,
            "groups": bench.grouped_scores(rows),
        }
        row = methods[name]["groups"]["all"]
        log(
            f"  {name}: IoU {row['mean_iou']}  "
            f"correct@0.8/chain {row['pct_correct_at_080_per_chain']}  "
            f"countMAE {row['domain_count_mae']}"
        )

    result = {
        "benchmark": "chainsaw_cath1363",
        "task": "T3.7",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": protocol.DATASET_VERSION,
        "encoder": model_name,
        "code": protocol.code_revision("."),
        "coverage": {
            "chainsaw_test": len(test_ids),
            "in_dataset": len(truth),
            "scored": len(embedded),
            "missing_from_dataset": len(test_ids) - len(truth),
            "missing_embedding": len(truth) - len(embedded),
            "with_published_result": len(shared_rows),
        },
        "methods": methods,
        "chainsaw_published": bench.published_means(shared_rows),
        "ground_truth_agreement": bench.ground_truth_agreement(
            {c: e["domains"] for c, e in embedded.items()}, their_truth
        ),
        "caveats": [
            "Chainsaw reads 3D structure (pairwise Ca distances and predicted "
            "secondary structure); Domaino reads sequence alone. The gap is "
            "the measurement this project exists to make, not a like-for-like "
            "contest.",
            "Chainsaw's published IoU column cannot be reproduced from its own "
            "choppings by a residue-level domain IoU, and the metric details "
            "are in Supplementary Material we do not have. Both their figure "
            "and our recomputation of it are reported.",
            "The two ground truths index different coordinate spaces "
            "(observed structure residues versus SEQRES), so each method is "
            "scored against its own; see ground_truth_agreement.",
            "Domaino's post-processing thresholds are untuned defaults. T3.5 "
            "tunes them on validation data, so these numbers are a floor.",
        ],
    }

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "chainsaw_cath1363.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    report_path = os.path.join(args.out, "chainsaw_cath1363.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(bench.format_report(result))
    log(f"wrote {json_path} and {report_path}")
    print()
    print(bench.format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
