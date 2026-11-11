"""Minimum reference baselines (task T2.1).

The proposal asks for three deliberately simple systems, implemented before the
real architecture, so that every later design choice has to justify itself
against a number rather than against intuition:

`SingleDomainBaseline`
    Treat the whole chain as one domain. It learns nothing, yet it is the one
    baseline that is genuinely hard to beat on average, because most CATH
    chains really are single-domain. Any segmentation model that cannot clear
    it is not doing useful work.

`BoundaryBaseline`
    A residue-level boundary predictor: logistic regression over ESM-2 residue
    vectors, decoded into cut points. This is the floor for "can sequence
    context alone locate a domain edge".

`PooledDomainClassifier`
    Mean-pools ESM-2 vectors over *ground-truth* segments and predicts CATH
    labels from them. Taking true segments on purpose isolates classification
    from segmentation, which is the only way to tell later whether an
    end-to-end error came from the parse or from the label.

The two learned baselines are linear softmax/logistic models trained by plain
minibatch gradient descent, kept that way so they stay a floor rather than a
competitor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Intervals = List[Tuple[int, int]]

# Residues within this distance of a true boundary count as positives, because
# asking a sequence-only model for the exact residue is harsher than the
# benchmark's own boundary-distance measure.
BOUNDARY_TOLERANCE = 2

# CATH levels, outermost first.
LEVELS = ("C", "A", "T", "H")


# -- baseline 1: everything is one domain -----------------------------------


class SingleDomainBaseline:
    """Predict exactly one domain spanning the whole chain."""

    name = "single_domain"

    def fit(self, *args, **kwargs) -> "SingleDomainBaseline":
        """Nothing to learn; present so every baseline shares an interface."""
        return self

    def predict(self, length: int) -> Intervals:
        if length < 1:
            raise ValueError(f"length must be positive, got {length}")
        return [(1, length)]

    def predict_many(self, lengths: Sequence[int]) -> List[List[Intervals]]:
        return [[self.predict(length)] for length in lengths]


# -- baseline 2: residue-level boundary predictor ----------------------------


def boundary_targets(
    length: int, domains: Sequence[Intervals], tolerance: int = BOUNDARY_TOLERANCE
) -> List[int]:
    """Per-residue 1/0 labels marking domain edges.

    Only internal edges are labelled. The chain's own first and last residue
    are boundaries for every chain, so training on them would teach the model
    nothing except where the sequence ends.
    """
    labels = [0] * length
    edges = set()
    for segments in domains:
        for start, end in segments:
            if start > 1:
                edges.add(start)
            if end < length:
                edges.add(end)
    for edge in edges:
        for position in range(edge - tolerance, edge + tolerance + 1):
            if 1 <= position <= length:
                labels[position - 1] = 1
    return labels


def decode_boundaries(
    scores: Sequence[float],
    threshold: float = 0.5,
    min_domain_length: int = 40,
) -> Intervals:
    """Turn per-residue boundary scores into contiguous domain intervals.

    Peaks above `threshold` become cut points, nearest-peak-first, and a cut is
    only accepted if both resulting pieces stay at least `min_domain_length`
    long. Without that guard a noisy score shreds a chain into fragments and
    the IoU collapses.
    """
    length = len(scores)
    if length == 0:
        return []

    candidates = [
        (scores[i], i + 1)
        for i in range(1, length - 1)
        if scores[i] >= threshold
        and scores[i] >= scores[i - 1]
        and scores[i] >= scores[i + 1]
    ]
    candidates.sort(key=lambda c: (-c[0], c[1]))

    cuts: List[int] = []
    for _, position in candidates:
        edges = sorted([0, *cuts, length])
        # The piece this cut would land in must survive the split on both sides.
        ok = True
        for left, right in zip(edges, edges[1:]):
            if left < position <= right:
                if position - left < min_domain_length or right - position < min_domain_length:
                    ok = False
                break
        if ok:
            cuts.append(position)
            cuts.sort()

    edges = sorted([0, *cuts, length])
    return [(left + 1, right) for left, right in zip(edges, edges[1:]) if right > left]


class BoundaryBaseline:
    """Logistic regression over residue embeddings, predicting domain edges."""

    name = "boundary"

    def __init__(
        self,
        dim: int,
        learning_rate: float = 0.05,
        epochs: int = 3,
        batch_size: int = 4096,
        threshold: float = 0.5,
        min_domain_length: int = 40,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.min_domain_length = min_domain_length
        self.seed = seed
        self.weights = None
        self.bias = 0.0
        self.history: List[Dict[str, float]] = []

    def fit(
        self, features: Iterable, labels: Iterable
    ) -> "BoundaryBaseline":
        """Train on stacked residue vectors and their 0/1 boundary labels."""
        import numpy as np

        x = np.concatenate([np.asarray(f, dtype=np.float32) for f in features], axis=0)
        y = np.concatenate([np.asarray(l, dtype=np.float32) for l in labels], axis=0)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and labels disagree on residue count")
        if x.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {x.shape[1]}")

        rng = np.random.default_rng(self.seed)
        weights = np.zeros(self.dim, dtype=np.float32)
        bias = 0.0

        # Boundaries are a small minority of residues, so positives are
        # up-weighted; otherwise predicting "never a boundary" is near-optimal
        # loss and useless.
        positives = float(y.sum())
        pos_weight = ((len(y) - positives) / positives) if positives else 1.0

        for epoch in range(self.epochs):
            order = rng.permutation(len(y))
            total_loss = 0.0
            batches = 0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb, yb = x[idx], y[idx]
                logits = xb @ weights + bias
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                sample_w = np.where(yb > 0, pos_weight, 1.0)
                error = (probs - yb) * sample_w
                weights -= self.learning_rate * (xb.T @ error) / len(idx)
                bias -= self.learning_rate * float(error.mean())
                eps = 1e-7
                total_loss += float(
                    -(
                        sample_w
                        * (yb * np.log(probs + eps) + (1 - yb) * np.log(1 - probs + eps))
                    ).mean()
                )
                batches += 1
            self.history.append(
                {"epoch": epoch, "loss": round(total_loss / max(batches, 1), 5)}
            )

        self.weights = weights
        self.bias = float(bias)
        return self

    def scores(self, features) -> List[float]:
        """Per-residue boundary probabilities for one chain."""
        import numpy as np

        if self.weights is None:
            raise RuntimeError("fit the baseline before predicting")
        x = np.asarray(features, dtype=np.float32)
        logits = x @ self.weights + self.bias
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))).tolist()

    def predict(self, features) -> Intervals:
        return decode_boundaries(
            self.scores(features),
            threshold=self.threshold,
            min_domain_length=self.min_domain_length,
        )


# -- baseline 3: pooled-embedding CATH classifier ----------------------------


def pool_segments(embedding, segments: Intervals):
    """Mean-pool residue vectors over a domain's segments.

    Pooling across all segments of a discontinuous domain gives it one vector,
    which is what lets the same classifier handle domains that are split in
    sequence but joined in space.
    """
    import numpy as np

    rows = [
        embedding[start - 1 : end]
        for start, end in segments
        if end >= start
    ]
    stacked = np.concatenate([np.asarray(r, dtype=np.float32) for r in rows], axis=0)
    if stacked.size == 0:
        raise ValueError("segments selected no residues")
    return stacked.mean(axis=0)


class PooledDomainClassifier:
    """Multinomial logistic regression per CATH level on pooled embeddings."""

    name = "pooled_classifier"

    def __init__(
        self,
        dim: int,
        levels: Sequence[str] = LEVELS,
        learning_rate: float = 0.2,
        epochs: int = 15,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.levels = tuple(levels)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.classes: Dict[str, List[str]] = {}
        self.weights: Dict[str, object] = {}
        self.history: Dict[str, List[Dict[str, float]]] = {}

    def fit(
        self, pooled, labels_by_level: Dict[str, Sequence[str]]
    ) -> "PooledDomainClassifier":
        """Train one softmax head per CATH level on pooled domain vectors."""
        import numpy as np

        x = np.asarray(pooled, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}), got {x.shape}")

        rng = np.random.default_rng(self.seed)
        for level in self.levels:
            labels = labels_by_level.get(level)
            if not labels:
                continue
            if len(labels) != len(x):
                raise ValueError(f"{level}: {len(labels)} labels for {len(x)} rows")

            classes = sorted(set(labels))
            index = {label: i for i, label in enumerate(classes)}
            y = np.fromiter((index[l] for l in labels), dtype=np.int64, count=len(labels))

            w = np.zeros((self.dim, len(classes)), dtype=np.float32)
            b = np.zeros(len(classes), dtype=np.float32)
            history: List[Dict[str, float]] = []

            for epoch in range(self.epochs):
                order = rng.permutation(len(y))
                total_loss = 0.0
                batches = 0
                for start in range(0, len(order), self.batch_size):
                    idx = order[start : start + self.batch_size]
                    xb, yb = x[idx], y[idx]
                    logits = xb @ w + b
                    logits -= logits.max(axis=1, keepdims=True)
                    exp = np.exp(logits)
                    probs = exp / exp.sum(axis=1, keepdims=True)
                    onehot = np.zeros_like(probs)
                    onehot[np.arange(len(yb)), yb] = 1.0
                    grad = (probs - onehot) / len(yb)
                    w -= self.learning_rate * (xb.T @ grad + self.weight_decay * w)
                    b -= self.learning_rate * grad.sum(axis=0)
                    total_loss += float(
                        -np.log(probs[np.arange(len(yb)), yb] + 1e-9).mean()
                    )
                    batches += 1
                history.append(
                    {"epoch": epoch, "loss": round(total_loss / max(batches, 1), 5)}
                )

            self.classes[level] = classes
            self.weights[level] = (w, b)
            self.history[level] = history
        return self

    def predict(self, pooled) -> Dict[str, List[str]]:
        """Predict a label per level for each pooled domain vector."""
        import numpy as np

        x = np.asarray(pooled, dtype=np.float32)
        out: Dict[str, List[str]] = {}
        for level, (w, b) in self.weights.items():  # type: ignore[misc]
            logits = x @ w + b
            best = logits.argmax(axis=1)
            classes = self.classes[level]
            out[level] = [classes[i] for i in best]
        return out

    def predict_hierarchical(self, pooled) -> Dict[str, List[str]]:
        """Predict levels, then force deeper labels to agree with shallower.

        A standalone H prediction can contradict the C prediction above it,
        which CATH cannot express. The deepest level that is consistent wins:
        where a deeper code does not extend the accepted prefix, it is replaced
        by the prefix, so the output is always a real point in the hierarchy.
        """
        raw = self.predict(pooled)
        if not raw:
            return raw
        count = len(next(iter(raw.values())))
        fixed: Dict[str, List[str]] = {level: [] for level in raw}
        for row in range(count):
            accepted = ""
            for level in self.levels:
                if level not in raw:
                    continue
                candidate = raw[level][row]
                if accepted and not candidate.startswith(accepted + "."):
                    candidate = accepted
                fixed[level].append(candidate)
                accepted = candidate
        return fixed
