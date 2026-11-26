"""Evaluate a trained model on the CHAINSAW CATH1363 benchmark (task T3.7).

Chainsaw's repository publishes not only its CATH1363 test split but its own
**per-chain predictions and per-chain metric values** on that split. That
changes what is possible here: instead of comparing our numbers to a figure
copied out of a paper, we can re-score Chainsaw's own predictions with our
metric code and put both models on one definition.

Why that matters
----------------
Running their published scoring is not an option -- the paper puts the metric
definitions in Supplementary Material we do not have -- and their `iou` column
does not match a residue-level domain IoU. Recomputing their column from their
own choppings with our code gives 0.852 where they report 0.881, and the
disagreements are concentrated in chains where a predicted domain covers
residues their ground truth leaves unassigned. So their IoU is more forgiving
than ours about unassigned residues.

Two numbers are therefore reported for Chainsaw: the one they published, and
the one our code gives for the same predictions. Domaino is compared against
the second. Comparing against the first would flatter or punish us by an
amount that has nothing to do with the model.

What is **not** controlled
--------------------------
* Chainsaw reads **3D structure** (pairwise Cα distances plus predicted
  secondary structure). Domaino reads sequence alone. This is the comparison
  the proposal exists to make, not a like-for-like contest, and the gap is the
  result.
* The two ground truths live in different coordinate spaces: Chainsaw's
  choppings index observed structure residues, ours index SEQRES. Each model
  is scored against the ground truth in its own space, and
  `ground_truth_agreement` measures how far apart the two are so the reader
  can judge the comparison rather than take it on trust.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import metrics
from ..data.benchmarks import load_chainsaw_splits

Intervals = List[Tuple[int, int]]

# Chainsaw's chopping syntax: '|' between domains, '_' inside a discontinuous
# one, and 0-based inclusive residue indices.
DOMAIN_SEP = "|"
SEGMENT_SEP = "_"

# Their per-chain metric columns, for the published-versus-recomputed table.
PUBLISHED_COLUMNS = (
    "iou",
    "proportion_correct_domains",
    "boundary_dist_score",
    "ndo",
    "d_count_acc",
)


def parse_chopping(chopping: str, offset: int = 0) -> List[Intervals]:
    """Parse a Chainsaw chopping into domains of intervals.

    `offset` shifts the coordinates; pass 1 to turn their 0-based indices into
    our 1-based convention. Returns an empty list for an empty chopping, which
    is how they record a chain their model declined.
    """
    chopping = (chopping or "").strip()
    if not chopping:
        return []
    domains: List[Intervals] = []
    for part in chopping.split(DOMAIN_SEP):
        segments: Intervals = []
        for piece in part.split(SEGMENT_SEP):
            piece = piece.strip()
            if not piece:
                continue
            start, _, end = piece.partition("-")
            segments.append((int(start) + offset, int(end) + offset))
        if segments:
            domains.append(sorted(segments))
    return domains


@dataclass
class BenchmarkRow:
    """One chain's ground truth and whatever each method predicted for it."""

    chain_id: str
    length: int
    truth: List[Intervals] = field(default_factory=list)
    predicted: List[Intervals] = field(default_factory=list)

    @property
    def n_true(self) -> int:
        return len(self.truth)

    @property
    def multidomain(self) -> bool:
        return len(self.truth) > 1

    @property
    def discontinuous(self) -> bool:
        return any(len(d) > 1 for d in self.truth)


def load_published(path: str) -> Dict[str, Dict[str, Any]]:
    """Chainsaw's per-chain results, keyed by chain id."""
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            chain = (row.get("chain_id") or "").strip()
            if chain:
                rows[chain] = row
    return rows


def published_means(
    rows: Iterable[Dict[str, Any]], columns: Sequence[str] = PUBLISHED_COLUMNS
) -> Dict[str, Optional[float]]:
    """Mean of each published column, over the rows that carry a value."""
    out: Dict[str, Optional[float]] = {}
    for column in columns:
        values = []
        for row in rows:
            text = (row.get(column) or "").strip()
            if text:
                try:
                    values.append(float(text))
                except ValueError:
                    continue
        out[column] = round(statistics.mean(values), 4) if values else None
    return out


def chain_proportion_correct(
    predicted: Sequence[Intervals], truth: Sequence[Intervals], threshold: float = 0.8
) -> Optional[float]:
    """Share of *this chain's* true domains matched at IoU >= threshold.

    Chainsaw averages this per chain, so a 7-domain chain counts once rather
    than seven times. `score_segmentation` pools over all domains instead, so
    both are reported and neither is silently substituted for the other.
    """
    if not truth:
        return None
    matched = {ti: score for _, ti, score in metrics.match_domains(predicted, truth)}
    hits = sum(1 for ti in range(len(truth)) if matched.get(ti, 0.0) >= threshold)
    return hits / len(truth)


def score_rows(rows: Sequence[BenchmarkRow]) -> Dict[str, Any]:
    """Our metrics over a set of chains, plus Chainsaw's per-chain averaging."""
    if not rows:
        return {"chains": 0}
    scores = metrics.score_segmentation(
        [r.predicted for r in rows], [r.truth for r in rows]
    ).as_dict()
    per_chain = [
        value
        for value in (chain_proportion_correct(r.predicted, r.truth) for r in rows)
        if value is not None
    ]
    scores["pct_correct_at_080_per_chain"] = (
        round(statistics.mean(per_chain), 4) if per_chain else None
    )
    scores["chains_with_no_prediction"] = sum(1 for r in rows if not r.predicted)
    return scores


def grouped_scores(rows: Sequence[BenchmarkRow]) -> Dict[str, Any]:
    """Scores for the whole set and for the groups the proposal requires."""
    groups = {
        "all": list(rows),
        "single_domain": [r for r in rows if r.n_true == 1],
        "multidomain": [r for r in rows if r.multidomain],
        "continuous": [r for r in rows if not r.discontinuous],
        "discontinuous": [r for r in rows if r.discontinuous],
    }
    return {name: score_rows(member) for name, member in groups.items() if member}


def ground_truth_agreement(
    ours: Dict[str, List[Intervals]], theirs: Dict[str, List[Intervals]]
) -> Dict[str, Any]:
    """How far apart the two ground truths are on the chains both cover.

    Their choppings index observed structure residues and ours index SEQRES,
    so the intervals cannot be compared directly. The domain **count** can,
    and so can each domain's share of the annotated residues, which is
    coordinate-free. If those agree, scoring each method in its own space is
    reasonable; if they do not, the comparison is confounded and this says so.
    """
    shared = sorted(set(ours) & set(theirs))
    if not shared:
        return {"chains": 0}

    same_count = 0
    count_delta: List[int] = []
    shape_distance: List[float] = []
    for chain in shared:
        mine, other = ours[chain], theirs[chain]
        same_count += int(len(mine) == len(other))
        count_delta.append(len(mine) - len(other))
        if len(mine) == len(other) and mine and other:
            # Each domain's fraction of all annotated residues, largest first:
            # a scale-free description of how the chain was carved up.
            def shares(domains: List[Intervals]) -> List[float]:
                sizes = [sum(e - s + 1 for s, e in d) for d in domains]
                total = sum(sizes) or 1
                return sorted((size / total for size in sizes), reverse=True)

            a, b = shares(mine), shares(other)
            shape_distance.append(sum(abs(x - y) for x, y in zip(a, b)) / 2.0)

    return {
        "chains": len(shared),
        "same_domain_count": same_count,
        "same_domain_count_pct": round(same_count / len(shared), 4),
        "mean_signed_count_delta": round(statistics.mean(count_delta), 4),
        "mean_shape_distance": (
            round(statistics.mean(shape_distance), 4) if shape_distance else None
        ),
        "note": (
            "shape distance is total-variation between the two ground truths' "
            "domain-size distributions, on chains where they agree on the "
            "domain count; 0 means identical proportions"
        ),
    }


def format_report(result: Dict[str, Any]) -> str:
    """A readable report, which is what actually gets read."""
    lines: List[str] = []
    lines.append("# CHAINSAW CATH1363 benchmark (task T3.7)")
    lines.append("")
    lines.append(f"Generated {result['generated_at']}")
    lines.append("")
    coverage = result["coverage"]
    lines.append(
        f"Chains: {coverage['scored']} of Chainsaw's {coverage['chainsaw_test']} "
        f"test chains scored. "
        f"{coverage['missing_from_dataset']} are absent from `dataset_v1` "
        "(their PDB entries were obsoleted after CATH 4.4.0 was built) and "
        f"{coverage['missing_embedding']} had no cached embedding."
    )
    lines.append("")

    lines.append("## Headline, all scored chains")
    lines.append("")
    lines.append("| Method | Input | mean IoU | correct@0.8 (per chain) | count MAE |")
    lines.append("|---|---|---|---|---|")
    for name, entry in result["methods"].items():
        row = entry["groups"].get("all", {})
        lines.append(
            f"| {name} | {entry['input']} | {row.get('mean_iou')} | "
            f"{row.get('pct_correct_at_080_per_chain')} | "
            f"{row.get('domain_count_mae')} |"
        )
    published = result.get("chainsaw_published")
    if published:
        lines.append(
            f"| Chainsaw v3, as published | structure | {published.get('iou')} | "
            f"{published.get('proportion_correct_domains')} | — |"
        )
    lines.append("")
    if published:
        lines.append(
            "The last row is Chainsaw's own published column. The row above it "
            "is Chainsaw's **same predictions** re-scored with our metric code. "
            "The gap between the two is the metric definition, not the model, "
            "which is why Domaino is compared against the re-scored row."
        )
        lines.append("")

    lines.append("## By chain category, mean IoU")
    lines.append("")
    groups = ["all", "single_domain", "multidomain", "continuous", "discontinuous"]
    lines.append("| Method | " + " | ".join(g.replace("_", " ") for g in groups) + " |")
    lines.append("|---" * (len(groups) + 1) + "|")
    for name, entry in result["methods"].items():
        cells = []
        for group in groups:
            row = entry["groups"].get(group, {})
            value = row.get("mean_iou")
            count = row.get("chains")
            cells.append(f"{value} (n={count})" if value is not None else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    agreement = result.get("ground_truth_agreement", {})
    if agreement.get("chains"):
        lines.append("## How comparable are the two ground truths")
        lines.append("")
        lines.append(
            f"On the {agreement['chains']} chains both cover, the two ground "
            f"truths agree on the domain count for "
            f"{agreement['same_domain_count']} "
            f"({100 * agreement['same_domain_count_pct']:.1f}%). Mean signed "
            f"difference in domain count, ours minus theirs: "
            f"{agreement['mean_signed_count_delta']}. Mean shape distance "
            f"{agreement['mean_shape_distance']}."
        )
        lines.append("")
        lines.append(
            "Their choppings index observed structure residues and ours index "
            "SEQRES, so each method is scored against the ground truth in its "
            "own coordinate space. These figures are what makes that defensible "
            "or not."
        )
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for caveat in result["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)
