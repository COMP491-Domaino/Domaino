"""Hierarchical CATH classifier (task T2.5).

Takes the residue vectors assigned to one domain, pools them into a
domain-level representation, and predicts the four CATH levels: Class,
Architecture, Topology and Homologous superfamily.

Two requirements shape the design.

Hierarchy consistency
    CATH is a tree, so ``3.40.50.300`` is only meaningful under ``3.40.50``,
    which is only meaningful under ``3.40``. Four independent softmaxes can
    easily return a path that does not exist. Rather than predict each level
    and patch up contradictions afterwards, `predict` scores whole **paths**
    through the real tree and returns the best one, so every output is a point
    CATH actually contains.

The H level abstains
    H holds 6,630 superfamilies in dataset v1, of which 1,027 have a single
    domain and 3,721 fewer than ten. Forcing a label on every domain would
    report confident nonsense for the rare ones. An H label is therefore
    returned only when the superfamily had enough training support *and* the
    calibrated confidence clears a threshold; otherwise C, A and T stand and H
    is reported unresolved. Coverage is reported beside accuracy so that
    abstaining cannot quietly inflate the score.

The class imbalance is severe enough that plain cross-entropy learns to answer
with the common superfamilies, so both class-balanced and focal losses are
implemented for the proposal to compare.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LEVELS = ("C", "A", "T", "H")

# What `predict` returns when it declines to name a superfamily.
UNRESOLVED = "unresolved"


@dataclass
class ClassifierConfig:
    """Shape of the classifier and the rules for abstaining."""

    domain_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1
    # Levels the model predicts. C, A and T are the primary outputs; H is the
    # conditional one.
    levels: Tuple[str, ...] = LEVELS
    # An H label is withheld unless its superfamily had at least this many
    # training domains.
    min_h_support: int = 10
    # ...and unless the predicted probability clears this.
    h_confidence_threshold: float = 0.5
    # "none" | "balanced" | "focal": the proposal compares the last two.
    loss_weighting: str = "balanced"
    focal_gamma: float = 2.0
    # Effective-number beta for class-balanced weights.
    balanced_beta: float = 0.999

    def __post_init__(self) -> None:
        if self.loss_weighting not in ("none", "balanced", "focal"):
            raise ValueError(f"unknown loss_weighting: {self.loss_weighting}")
        if not 0.0 <= self.h_confidence_threshold <= 1.0:
            raise ValueError("h_confidence_threshold must be a probability")
        if "H" in self.levels and "T" not in self.levels:
            raise ValueError("H cannot be predicted without its parent level T")


def parent_code(code: str) -> Optional[str]:
    """The code one level up, or None at the Class level."""
    head, _, tail = code.rpartition(".")
    return head or None


class CathHierarchy:
    """The CATH tree as the dataset actually contains it.

    Built from the training labels rather than from CATH's full release, so the
    classifier can only ever emit a node it was trained on, and so H support
    counts reflect this dataset.
    """

    def __init__(self, levels: Sequence[str] = LEVELS) -> None:
        self.levels = tuple(levels)
        self.codes: Dict[str, List[str]] = {level: [] for level in self.levels}
        self.index: Dict[str, Dict[str, int]] = {level: {} for level in self.levels}
        self.support: Dict[str, collections.Counter] = {
            level: collections.Counter() for level in self.levels
        }
        # Child indices of each parent, per level, for path scoring.
        self._children: Dict[str, Dict[str, List[int]]] = {
            level: {} for level in self.levels
        }

    @classmethod
    def from_labels(
        cls, labels_by_level: Dict[str, Sequence[str]], levels: Sequence[str] = LEVELS
    ) -> "CathHierarchy":
        """Build the tree from per-level label lists of equal length."""
        hierarchy = cls(levels)
        for level in hierarchy.levels:
            for code in labels_by_level.get(level, []):
                if code:
                    hierarchy.support[level][code] += 1
        for level in hierarchy.levels:
            hierarchy.codes[level] = sorted(hierarchy.support[level])
            hierarchy.index[level] = {
                code: i for i, code in enumerate(hierarchy.codes[level])
            }
        # Link each node to its parent's children list.
        for position, level in enumerate(hierarchy.levels):
            if position == 0:
                continue
            table: Dict[str, List[int]] = {}
            for code, i in hierarchy.index[level].items():
                table.setdefault(parent_code(code) or "", []).append(i)
            hierarchy._children[level] = table
        return hierarchy

    def size(self, level: str) -> int:
        return len(self.codes[level])

    def sizes(self) -> Dict[str, int]:
        return {level: self.size(level) for level in self.levels}

    def children_of(self, level: str, parent: str) -> List[int]:
        """Indices at `level` whose parent code is `parent`."""
        return self._children.get(level, {}).get(parent, [])

    def h_support(self, code: str) -> int:
        return int(self.support.get("H", {}).get(code, 0))

    def well_supported_h(self, minimum: int) -> List[str]:
        return [c for c in self.codes.get("H", []) if self.h_support(c) >= minimum]

    def encode(self, labels_by_level: Dict[str, Sequence[str]]):
        """Label indices per level, with -100 where a label is absent.

        -100 is torch's ignore index, which is how a missing label becomes an
        unsupervised position rather than a wrong one. TED resolves many
        domains only to T, so this is not a rare case.
        """
        import numpy as np

        count = max((len(v) for v in labels_by_level.values()), default=0)
        out = {}
        for level in self.levels:
            values = labels_by_level.get(level) or []
            column = np.full(count, -100, dtype=np.int64)
            for row, code in enumerate(values):
                if code and code in self.index[level]:
                    column[row] = self.index[level][code]
            out[level] = column
        return out

    def as_dict(self) -> Dict[str, object]:
        """Serialisable form, so a checkpoint carries its own label space."""
        return {
            "levels": list(self.levels),
            "codes": {level: self.codes[level] for level in self.levels},
            "support": {
                level: dict(self.support[level]) for level in self.levels
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "CathHierarchy":
        hierarchy = cls(list(payload["levels"]))  # type: ignore[arg-type]
        for level in hierarchy.levels:
            hierarchy.support[level] = collections.Counter(
                dict(payload["support"][level])  # type: ignore[index]
            )
            hierarchy.codes[level] = list(payload["codes"][level])  # type: ignore[index]
            hierarchy.index[level] = {
                code: i for i, code in enumerate(hierarchy.codes[level])
            }
        for position, level in enumerate(hierarchy.levels):
            if position == 0:
                continue
            table: Dict[str, List[int]] = {}
            for code, i in hierarchy.index[level].items():
                table.setdefault(parent_code(code) or "", []).append(i)
            hierarchy._children[level] = table
        return hierarchy


# -- pooling ----------------------------------------------------------------


def pool_domain(features, segments: Sequence[Tuple[int, int]], query_vector=None):
    """Pool a domain's residue features into one vector.

    Pooling across every segment gives a discontinuous domain a single
    representation, which is what lets one classifier handle domains that are
    split in sequence but joined in space. `query_vector`, when the query
    architecture supplies one, is concatenated: the proposal allows the learned
    domain-query representation to be included alongside the pooled residues.
    """
    import torch

    rows = [
        features[start - 1 : end]
        for start, end in segments
        if end >= start
    ]
    rows = [r for r in rows if r.shape[0] > 0]
    if not rows:
        raise ValueError("segments selected no residues")
    pooled = torch.cat(rows, dim=0).mean(dim=0)
    if query_vector is not None:
        pooled = torch.cat([pooled, query_vector], dim=-1)
    return pooled


# -- losses -----------------------------------------------------------------


def class_balanced_weights(counts: Sequence[int], beta: float = 0.999):
    """Effective-number class weights (Cui et al., 2019).

    A superfamily with one domain and one with fifty thousand cannot carry the
    same weight, or the rare one is never learned; weighting by the effective
    number of samples is gentler than inverse frequency, which would make
    singletons dominate.
    """
    import torch

    counts_t = torch.as_tensor(counts, dtype=torch.float32).clamp(min=1.0)
    effective = (1.0 - torch.pow(beta, counts_t)) / (1.0 - beta)
    weights = 1.0 / effective
    return weights * (len(counts_t) / weights.sum())


def focal_cross_entropy(logits, target, gamma: float = 2.0, weight=None, ignore_index: int = -100):
    """Cross-entropy down-weighted on examples already classified well.

    Focal loss is the alternative the proposal names: instead of reweighting by
    class frequency it reweights by how wrong the model currently is, which
    keeps gradient on the hard, usually rare, cases.
    """
    import torch
    from torch.nn import functional as F

    keep = target != ignore_index
    if keep.sum() == 0:
        return logits.sum() * 0.0
    logits, target = logits[keep], target[keep]
    log_probability = F.log_softmax(logits, dim=-1)
    picked = log_probability.gather(1, target[:, None]).squeeze(1)
    modulation = (1.0 - picked.exp()).pow(gamma)
    loss = -modulation * picked
    if weight is not None:
        loss = loss * weight.to(loss.device)[target]
    return loss.mean()


# -- network ----------------------------------------------------------------


def build_classifier(
    hierarchy: CathHierarchy, config: Optional[ClassifierConfig] = None
):
    """Build the hierarchical classifier for a given label space."""
    import torch
    from torch import nn

    config = config or ClassifierConfig()

    class HierarchicalClassifier(nn.Module):
        """One shared trunk, one softmax head per CATH level."""

        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.hierarchy = hierarchy
            self.shared = nn.Sequential(
                nn.LayerNorm(config.domain_dim),
                nn.Linear(config.domain_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.heads = nn.ModuleDict(
                {
                    level: nn.Linear(config.hidden_dim, max(hierarchy.size(level), 1))
                    for level in config.levels
                }
            )
            # Class-balanced weights are fixed by the training label counts, so
            # they are buffers: they travel with the checkpoint.
            for level in config.levels:
                counts = [
                    hierarchy.support[level].get(code, 0)
                    for code in hierarchy.codes[level]
                ] or [1]
                self.register_buffer(
                    f"weight_{level}",
                    class_balanced_weights(counts, config.balanced_beta),
                    persistent=True,
                )

        def forward(self, pooled):
            """Per-level logits for a batch of pooled domain vectors."""
            shared = self.shared(pooled)
            return {level: self.heads[level](shared) for level in config.levels}

        def level_weight(self, level: str):
            return getattr(self, f"weight_{level}")

        def loss(self, logits: Dict[str, object], targets: Dict[str, object]):
            """Summed per-level loss, skipping levels with no label.

            A level whose label is absent contributes nothing rather than a
            wrong target, so a domain CATH resolves only to T trains C, A and T
            and leaves H alone.
            """
            from torch.nn import functional as F

            total = None
            parts: Dict[str, float] = {}
            for level in config.levels:
                target = targets[level]
                if (target != -100).sum() == 0:
                    continue
                if config.loss_weighting == "focal":
                    value = focal_cross_entropy(
                        logits[level], target, config.focal_gamma
                    )
                elif config.loss_weighting == "balanced":
                    value = F.cross_entropy(
                        logits[level],
                        target,
                        weight=self.level_weight(level),
                        ignore_index=-100,
                    )
                else:
                    value = F.cross_entropy(
                        logits[level], target, ignore_index=-100
                    )
                total = value if total is None else total + value
                parts[level] = float(value.detach())
            if total is None:
                total = logits[config.levels[0]].sum() * 0.0
            return total, parts

        @torch.no_grad()
        def predict(self, pooled) -> List[Dict[str, object]]:
            """Best consistent path per domain, with H possibly unresolved.

            Levels are decoded together rather than independently: the chosen
            Architecture must be a child of the chosen Class, and so on, so the
            result is always a node CATH contains.
            """
            logits = self.forward(pooled)
            log_probability = {
                level: torch.log_softmax(logits[level], dim=-1).cpu()
                for level in config.levels
            }
            results: List[Dict[str, object]] = []
            for row in range(pooled.shape[0]):
                results.append(
                    self._decode_row(
                        {l: log_probability[l][row] for l in config.levels}
                    )
                )
            return results

        def _decode_row(self, scores) -> Dict[str, object]:
            import torch as _torch

            levels = list(config.levels)
            # Class level: no parent to respect.
            first = levels[0]
            if hierarchy.size(first) == 0:
                return {level: UNRESOLVED for level in levels}
            best_index = int(_torch.argmax(scores[first]))
            path = {first: hierarchy.codes[first][best_index]}
            confidence = {first: float(scores[first][best_index].exp())}

            for level in levels[1:]:
                parent = path[levels[levels.index(level) - 1]]
                if parent == UNRESOLVED:
                    path[level] = UNRESOLVED
                    confidence[level] = 0.0
                    continue
                candidates = hierarchy.children_of(level, parent)
                if not candidates:
                    # The tree has no node below the accepted parent, so the
                    # honest answer is that this level is unresolved.
                    path[level] = UNRESOLVED
                    confidence[level] = 0.0
                    continue
                picked = max(candidates, key=lambda i: float(scores[level][i]))
                path[level] = hierarchy.codes[level][picked]
                confidence[level] = float(scores[level][picked].exp())

            # H abstains unless it is both well supported and confident.
            if "H" in path and path["H"] != UNRESOLVED:
                code = path["H"]
                if (
                    hierarchy.h_support(code) < config.min_h_support
                    or confidence["H"] < config.h_confidence_threshold
                ):
                    path["H"] = UNRESOLVED
            return {"labels": path, "confidence": confidence}

    return HierarchicalClassifier()


def h_coverage(predictions: Iterable[Dict[str, object]]) -> float:
    """Fraction of domains that received an H label rather than abstaining.

    Reported next to accuracy because a model that answers only when sure can
    otherwise look better than one that answers always.
    """
    rows = list(predictions)
    if not rows:
        return 0.0
    named = sum(
        1
        for row in rows
        if dict(row.get("labels", {})).get("H", UNRESOLVED) != UNRESOLVED
    )
    return round(named / len(rows), 4)
