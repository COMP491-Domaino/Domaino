"""Score a trained model against the T2.1 baselines, on their own eval set.

    python -m domaino.train.compare_baselines --config configs/train_v1.json \
        --checkpoint data/.../t32-pairwise-...-best.pt

The T2.1 baselines were scored on the **first 2,000 validation chains**
(`eval_limit` in `configs/baselines_v1.json`), not on the whole split. The T3.2
runs report the whole 24,525-chain split. Putting those two numbers in one
table would compare a model and a baseline on different sets of proteins, and
the difference in chain mix alone moves mean IoU by more than the models
differ.

So this scores a checkpoint on exactly the baseline's chains, reads the
baseline numbers out of `reports/baselines_v1/`, and writes one table where
every column is the same 2,000 proteins. That table is what belongs in a
progress report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from ..model.embed import load_config, split_members
from ..model.embedding_store import EmbeddingStore
from . import protocol
from .data import EmbeddingBatcher, load_examples

# Published in reports/baselines_v1/baseline_results.md (deliverable D2.1).
BASELINE_SEGMENTATION = {
    "single_domain": {
        "label": "Baseline 1: one domain",
        "mean_iou": 0.7138,
        "pct_correct_at_080": 0.4745,
        "domain_count_mae": 0.295,
        "mean_boundary_error": 33.914,
    },
    "boundary": {
        "label": "Baseline 2: boundary predictor",
        "mean_iou": 0.6373,
        "pct_correct_at_080": 0.3023,
        "domain_count_mae": 2.053,
        "mean_boundary_error": 6.874,
    },
}
BASELINE_CLASSIFICATION = {
    "C": {"classes": 5, "accuracy": 0.7859, "macro_f1": 0.4674, "unseen": 0},
    "A": {"classes": 36, "accuracy": 0.4071, "macro_f1": 0.0713, "unseen": 2},
    "T": {"classes": 426, "accuracy": 0.1818, "macro_f1": 0.0053, "unseen": 314},
    "H": {"classes": 883, "accuracy": 0.0502, "macro_f1": 0.0046, "unseen": 850},
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_model(checkpoint_path: str, input_dim: int, device):
    """Rebuild a `CompleteModel` from a T3.2 checkpoint."""
    import torch

    from ..model import classifier as clf
    from .trainer import CompleteModel, TrainConfig

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["state"]
    # A checkpoint written before a config field existed simply lacks it, and
    # the dataclass default fills in, so an older run stays loadable.
    known = set(TrainConfig.__dataclass_fields__)
    config = TrainConfig(
        **{k: v for k, v in dict(payload["config"]).items() if k in known}
    )
    hierarchy = (
        clf.CathHierarchy.from_dict(state["hierarchy"])
        if "hierarchy" in state
        else None
    )
    model = CompleteModel(config, input_dim, hierarchy, device)
    model.segmentation.load_state_dict(state["segmentation"])
    if model.classifier is not None and "classifier" in state:
        model.classifier.load_state_dict(state["classifier"])
    model.eval()
    return model, config, payload


def format_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Domaino against the T2.1 baselines")
    lines.append("")
    lines.append(f"Generated {result['generated_at']}")
    lines.append("")
    lines.append(
        f"Every column below is the **same {result['chains']} validation "
        "chains** — the set `configs/baselines_v1.json` fixed with "
        "`eval_limit`. The T3.2 run records report the whole 24,525-chain "
        "split instead, so those numbers are not the ones to put beside a "
        "baseline."
    )
    lines.append("")
    model = result["model"]
    lines.append(
        f"Model: `{model['architecture']}` from run `{model['run_id']}`, "
        f"epoch {model['epoch']}, selected on "
        f"{protocol.CHECKPOINT_METRIC} {model['selection']}."
    )
    lines.append("")

    seg = result["segmentation"]
    lines.append("## Segmentation")
    lines.append("")
    lines.append(
        "| Segmentation metric | Baseline 1: one domain | "
        "Baseline 2: boundary predictor | Domaino |"
    )
    lines.append("|---|---|---|---|")
    rows = [
        ("Mean IoU", "mean_iou", "{:.4f}"),
        ("Correct at IoU >= 0.8", "pct_correct_at_080", "{:.2%}"),
        ("Domain-count error (MAE)", "domain_count_mae", "{:.3f}"),
        ("Boundary error (residues)", "mean_boundary_error", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        lines.append(
            f"| {label} | "
            f"{fmt.format(BASELINE_SEGMENTATION['single_domain'][key])} | "
            f"{fmt.format(BASELINE_SEGMENTATION['boundary'][key])} | "
            f"**{fmt.format(seg[key])}** |"
        )
    lines.append("")

    classification = result.get("classification")
    if classification:
        lines.append("## CATH classification on ground-truth segments")
        lines.append("")
        lines.append(
            f"{classification['domains']} domains. Domaino's H-level coverage "
            f"is {classification['h_coverage']:.1%}: it abstains on the rest "
            "rather than forcing a superfamily label."
        )
        lines.append("")
        lines.append(
            "| CATH level | Baseline accuracy | Baseline macro F1 | "
            "Domaino accuracy | Domaino macro F1 |"
        )
        lines.append("|---|---|---|---|---|")
        for level in ("C", "A", "T", "H"):
            base = BASELINE_CLASSIFICATION[level]
            ours = classification["levels"].get(level, {})
            lines.append(
                f"| {level} | {base['accuracy']:.2%} | {base['macro_f1']:.2%} | "
                f"**{ours.get('accuracy', 0):.2%}** | "
                f"**{ours.get('macro_f1', 0):.2%}** |"
            )
        lines.append("")
        lines.append(
            "The baseline was trained on a 4,000-chain subset, which left 850 "
            "of its evaluation domains (34%) in a superfamily it had never "
            "seen and capped its H accuracy near 66%. Domaino trained on the "
            "whole split, so that cap is the first thing the comparison "
            "removes; the remaining gap is the model."
        )
        lines.append("")

    lines.append("## By chain category (Domaino)")
    lines.append("")
    lines.append("| group | chains | mean IoU | correct@0.8 | count MAE | boundary |")
    lines.append("|---|---|---|---|---|---|")
    for group in protocol.REPORT_GROUPS:
        row = result["groups"].get(group)
        if not row:
            continue
        lines.append(
            f"| {group} | {row['chains']} | {row['mean_iou']:.4f} | "
            f"{row['pct_correct_at_080']:.4f} | "
            f"{row['domain_count_mae']:.4f} | {row['mean_boundary_error']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baselines-config", default="configs/baselines_v1.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import torch

    file_config = load_config(args.config)
    model_name = str(dict(file_config["encoder"])["model_name"])
    baselines = load_config(args.baselines_config)
    eval_limit = int(baselines.get("eval_limit", 2000))
    out_dir = args.out or str(baselines["report_dir"])

    cache = os.path.join(
        str(file_config["embeddings_root"]), f"{model_name}-validation"
    )
    store = EmbeddingStore(cache)
    store.verify(model_name)

    # Same call, same order, same limit the baselines used: the eval set is
    # reproduced rather than approximated.
    examples = load_examples(
        str(file_config["records"]),
        split_members(str(file_config["manifest_dir"]), "validation"),
        store,
        eval_limit,
    )
    log(f"scoring on the baseline evaluation set: {len(examples)} chains")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, payload = load_model(args.checkpoint, store.meta.dim, device)
    log(f"{config.architecture} from {payload.get('run_id')} epoch {payload.get('epoch')}")

    from .trainer import evaluate

    scores = evaluate(model, examples, EmbeddingBatcher(store, device=device), config)
    groups = scores["segmentation"]
    classification = scores.get("classification_true_segments")

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chains": len(examples),
        "eval_limit": eval_limit,
        "model": {
            "architecture": config.architecture,
            "run_id": payload.get("run_id"),
            "epoch": payload.get("epoch"),
            "selection": payload.get(protocol.CHECKPOINT_METRIC),
            "checkpoint": args.checkpoint,
        },
        "segmentation": groups["all"],
        "groups": groups,
        "classification": classification,
        "baselines": {
            "segmentation": BASELINE_SEGMENTATION,
            "classification": BASELINE_CLASSIFICATION,
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(
        os.path.join(out_dir, "domaino_vs_baselines.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    report = format_report(result)
    with open(
        os.path.join(out_dir, "domaino_vs_baselines.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write(report)
    log(f"wrote {out_dir}/domaino_vs_baselines.{{json,md}}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
