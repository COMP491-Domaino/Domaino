"""Segmentation and classification metrics, as the proposal defines them.

Segmentation reuses CHAINSAW's published measures so Domaino's numbers can be
put beside theirs without re-deriving anything: mean intersection over union,
the share of domains parsed correctly at an IoU of at least 0.8, boundary
position error, and domain-count error.

Classification reports accuracy and macro F1 at each CATH level. Macro F1
carries the weight here: CATH is dominated by a few large superfamilies, so a
classifier that ignores rare ones can look accurate while being useless on the
classes that matter (proposal risk 3).

Every function takes plain interval lists, ``[(start, end), ...]`` 1-based and
inclusive, so metrics never depend on how a model represents a domain.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Intervals = Sequence[Tuple[int, int]]


def residues(intervals: Intervals) -> set:
    """The residue positions a domain covers."""
    return {p for start, end in intervals for p in range(start, end + 1)}


def iou(left: Intervals, right: Intervals) -> float:
    """Intersection over union of two domains' residue sets."""
    a, b = residues(left), residues(right)
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def match_domains(
    predicted: Sequence[Intervals], truth: Sequence[Intervals]
) -> List[Tuple[int, int, float]]:
    """Pair predicted domains with true ones by descending IoU.

    Greedy one-to-one matching: the best remaining pair is taken until nothing
    overlaps. Domain identity is arbitrary -- a model has no reason to number
    domains the way CATH does -- so comparing them requires a matching rather
    than an index-wise comparison.
    """
    scores = [
        (iou(p, t), pi, ti)
        for pi, p in enumerate(predicted)
        for ti, t in enumerate(truth)
    ]
    scores.sort(key=lambda s: (-s[0], s[1], s[2]))
    used_pred, used_truth = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for score, pi, ti in scores:
        if score <= 0.0 or pi in used_pred or ti in used_truth:
            continue
        used_pred.add(pi)
        used_truth.add(ti)
        matches.append((pi, ti, score))
    return matches


def boundary_positions(intervals: Intervals) -> List[int]:
    """Segment start and end positions of one domain."""
    out: List[int] = []
    for start, end in intervals:
        out.extend((start, end))
    return out


def boundary_error(
    predicted: Sequence[Intervals], truth: Sequence[Intervals]
) -> Optional[float]:
    """Mean distance from each true boundary to the nearest predicted one.

    Reported in residues. None when either side has no boundaries, which keeps
    an undefined case out of the average instead of scoring it as zero.
    """
    true_points = sorted({p for d in truth for p in boundary_positions(d)})
    pred_points = sorted({p for d in predicted for p in boundary_positions(d)})
    if not true_points or not pred_points:
        return None
    total = sum(min(abs(t - p) for p in pred_points) for t in true_points)
    return total / len(true_points)


@dataclass
class SegmentationScores:
    """Aggregated segmentation performance over a set of chains."""

    chains: int = 0
    mean_iou: float = 0.0
    domains_true: int = 0
    domains_predicted: int = 0
    correctly_parsed: int = 0
    pct_correct_at_080: float = 0.0
    domain_count_mae: float = 0.0
    domain_count_exact: float = 0.0
    mean_boundary_error: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def score_segmentation(
    predictions: Iterable[Sequence[Intervals]],
    targets: Iterable[Sequence[Intervals]],
) -> SegmentationScores:
    """Score predicted domain sets against true ones, CHAINSAW-style.

    `mean_iou` averages per-chain IoU, where a chain's IoU is the mean over its
    true domains of the IoU with its matched prediction (0 when unmatched), so
    missing a domain is penalised rather than ignored.
    """
    chain_ious: List[float] = []
    boundary_errors: List[float] = []
    count_errors: List[int] = []
    exact_counts = 0
    total_true = 0
    total_pred = 0
    correct = 0
    chains = 0

    for predicted, truth in zip(predictions, targets):
        chains += 1
        total_true += len(truth)
        total_pred += len(predicted)
        matches = match_domains(predicted, truth)
        by_truth = {ti: score for _, ti, score in matches}
        if truth:
            chain_ious.append(
                sum(by_truth.get(ti, 0.0) for ti in range(len(truth))) / len(truth)
            )
        correct += sum(1 for score in by_truth.values() if score >= 0.8)
        delta = len(predicted) - len(truth)
        count_errors.append(abs(delta))
        exact_counts += int(delta == 0)
        error = boundary_error(predicted, truth)
        if error is not None:
            boundary_errors.append(error)

    return SegmentationScores(
        chains=chains,
        mean_iou=round(sum(chain_ious) / len(chain_ious), 4) if chain_ious else 0.0,
        domains_true=total_true,
        domains_predicted=total_pred,
        correctly_parsed=correct,
        pct_correct_at_080=round(correct / total_true, 4) if total_true else 0.0,
        domain_count_mae=(
            round(sum(count_errors) / len(count_errors), 4) if count_errors else 0.0
        ),
        domain_count_exact=(
            round(exact_counts / chains, 4) if chains else 0.0
        ),
        mean_boundary_error=(
            round(sum(boundary_errors) / len(boundary_errors), 3)
            if boundary_errors
            else None
        ),
    )


def macro_f1(
    truth: Sequence[str], predicted: Sequence[str]
) -> Tuple[float, float, Dict[str, float]]:
    """Accuracy, macro F1 and per-class F1 over string labels.

    Macro F1 averages over the classes that actually occur in the truth, so a
    class the model never sees cannot inflate or deflate the score.
    """
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must be the same length")
    if not truth:
        return 0.0, 0.0, {}

    tp: collections.Counter = collections.Counter()
    fp: collections.Counter = collections.Counter()
    fn: collections.Counter = collections.Counter()
    for actual, guess in zip(truth, predicted):
        if actual == guess:
            tp[actual] += 1
        else:
            fp[guess] += 1
            fn[actual] += 1

    per_class: Dict[str, float] = {}
    for label in sorted(set(truth)):
        denominator = 2 * tp[label] + fp[label] + fn[label]
        per_class[label] = (2 * tp[label] / denominator) if denominator else 0.0

    accuracy = sum(tp.values()) / len(truth)
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return round(accuracy, 4), round(macro, 4), per_class


@dataclass
class ClassificationScores:
    """Per-CATH-level classification performance."""

    domains: int = 0
    levels: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"domains": self.domains, "levels": self.levels}


def score_classification(
    truth_by_level: Dict[str, Sequence[str]],
    predicted_by_level: Dict[str, Sequence[str]],
) -> ClassificationScores:
    """Score C, A, T and H predictions, reporting each level separately.

    Levels are reported apart because they are not equally learnable: Class has
    a handful of values while the homologous-superfamily level has thousands,
    most of them rare.
    """
    scores = ClassificationScores()
    for level in ("C", "A", "T", "H"):
        if level not in truth_by_level:
            continue
        actual = truth_by_level[level]
        guess = predicted_by_level.get(level, [])
        if not actual:
            continue
        scores.domains = max(scores.domains, len(actual))
        accuracy, macro, per_class = macro_f1(actual, guess)
        rare = [f1 for label, f1 in per_class.items() if label in set(actual)]
        scores.levels[level] = {
            "n": len(actual),
            "classes": len(per_class),
            "accuracy": accuracy,
            "macro_f1": macro,
            "worst_class_f1": round(min(rare), 4) if rare else 0.0,
        }
    return scores


def stratify(
    keys: Sequence[str], predictions: Sequence[object], targets: Sequence[object]
) -> Dict[str, Tuple[List[object], List[object]]]:
    """Split paired predictions and targets into named groups.

    The proposal requires results reported separately for single-domain,
    multidomain, continuous, discontinuous and long chains, because a good
    average can hide failure on exactly the cases domain parsing is for.
    """
    groups: Dict[str, Tuple[List[object], List[object]]] = {}
    for key, prediction, target in zip(keys, predictions, targets):
        bucket = groups.setdefault(key, ([], []))
        bucket[0].append(prediction)
        bucket[1].append(target)
    return groups
