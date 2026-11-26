"""Source-aware training targets and loss functions (task T2.6).

The proposal asks for four *separate* losses -- residue assignment, boundaries,
pairwise grouping, and CATH classification -- with source masks deciding which
of them a given record may contribute to. That separation is not cosmetic: the
three annotation sources answer different questions, and a single combined loss
silently lets the weakest one corrupt the others.

What each source can actually supervise
---------------------------------------
CATH chops the **whole chain**, so a residue outside every domain really is
linker. Every term is available, and a residue outside a domain is a valid
negative.

Pfam annotates only the families it recognises. An unannotated residue is
**unknown**, not linker, so:

* assignment is supervised **only inside annotated regions**, which means Pfam
  contributes positives and no negatives to that term -- correct but biased,
  which is exactly why its record weight is 0.3 rather than 1.0;
* boundaries are still supervised with both classes, because within a region we
  know both that its edges are edges and that its interior is not;
* pairwise grouping is unavailable: two residues in two different Pfam regions
  may well be one CATH domain, so "different domain" has no answer.

TED is AlphaFold-derived and carries its own consensus level. High- and
medium-consensus chains are treated as complete partitions; low-consensus ones
are not, and `Supervision` already refuses pairwise supervision for them.

Why marginals rather than a second head
---------------------------------------
The segmentation network emits one 3-way residue distribution over {linker,
domain, boundary}. Assignment and boundary are two different binary questions
over that same distribution, so each term is the binary cross-entropy of the
corresponding **marginal** -- P(in domain) = p(domain) + p(boundary), and
P(boundary) = p(boundary). No extra parameters, no change to the checkpoint
format, and the two terms can be masked and weighted independently, which is
the whole point of the task.

The sum of the two marginal losses is not the 3-way cross-entropy it replaces;
both are proper objectives, and which trains better is a T3.4 ablation rather
than something to assume here. `TrainConfig.source_aware` selects between them
so the already-published T3.2 runs stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.schema import Supervision
from .decode import BOUNDARY, DOMAIN, LINKER

# Marginal groupings of the three residue classes. Each entry is
# (positive classes, negative classes) for one binary question.
IN_DOMAIN_CLASSES = (DOMAIN, BOUNDARY)
NOT_IN_DOMAIN_CLASSES = (LINKER,)
BOUNDARY_CLASSES = (BOUNDARY,)
NOT_BOUNDARY_CLASSES = (LINKER, DOMAIN)

# Record weights for a source whose own records did not carry one. Mirrors the
# defaults in `data/sources/`, so a hand-built example behaves like a real one.
DEFAULT_SOURCE_WEIGHTS = {"cath": 1.0, "pfam": 0.3, "ted": 0.15}


@dataclass
class LossWeights:
    """Relative weight of the three residue- and pair-level terms.

    The fourth term, CATH classification, is the classifier's own loss on
    pooled ground-truth segments. It has no residue-level mask to apply -- a
    domain either comes from a source that grants CATH labels or it is not
    pooled at all -- so it keeps its own weight, `TrainConfig.
    classifier_weight`, at the point where it is computed.

    `boundary` is below 1.0 because the boundary band covers a few percent of
    residues: at equal weight its gradient is dominated by the negatives, and
    the term mostly learns to say "not a boundary".
    """

    assignment: float = 1.0
    boundary: float = 0.5
    pairwise: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "assignment": self.assignment,
            "boundary": self.boundary,
            "pairwise": self.pairwise,
        }


@dataclass
class SourceMask:
    """Which losses one record may contribute to, and how much.

    `assignment` and `boundary` are per-residue weights, zero where the source
    has no answer. `pairwise` and `classification` are whole-record gates,
    because those terms are either defined for a record or not.
    """

    assignment: Any
    boundary: Any
    pairwise: bool
    classification: bool
    weight: float

    @property
    def length(self) -> int:
        return int(self.assignment.shape[0])


def supervision_for(
    source: str, weight: Optional[float] = None, consensus: str = "high"
) -> Supervision:
    """The supervision a source grants, for examples built without a record.

    Tests and the inference path construct examples by hand; this keeps them
    from inventing a policy the dataset would never produce. `consensus`
    applies to TED only, whose per-chain agreement level decides whether its
    chopping may be treated as a complete partition.
    """
    source = source.lower()
    if weight is None:
        weight = DEFAULT_SOURCE_WEIGHTS.get(source, 0.1)
    if source == "cath":
        return Supervision(
            residue_labels=True, pairwise=True, cath_labels=True,
            complete_partition=True, weight=weight,
        )
    if source == "pfam":
        return Supervision(
            residue_labels=True, pairwise=False, cath_labels=False,
            complete_partition=False, weight=weight,
        )
    if source == "ted":
        complete = consensus != "low"
        return Supervision(
            residue_labels=True, pairwise=complete, cath_labels=True,
            complete_partition=complete, weight=weight,
        )
    raise ValueError(f"unknown source: {source}")


# -- targets ----------------------------------------------------------------


def assignment_targets(length: int, domains: Sequence[Sequence[Tuple[int, int]]]):
    """1 where a residue lies inside some domain, 0 elsewhere."""
    import numpy as np

    target = np.zeros(length, dtype=np.float32)
    for segments in domains:
        for start, end in segments:
            lo, hi = max(1, start), min(length, end)
            if hi >= lo:
                target[lo - 1 : hi] = 1.0
    return target


def boundary_targets(
    length: int,
    domains: Sequence[Sequence[Tuple[int, int]]],
    boundary_width: int = 2,
):
    """1 within `boundary_width` residues of a domain edge, 0 elsewhere.

    A band rather than the single edge residue, for the same reason as in
    T2.3: the benchmark scores boundary *distance*, so demanding the exact
    residue is stricter than the measure the result is reported against.
    """
    import numpy as np

    target = np.zeros(length, dtype=np.float32)
    for segments in domains:
        for start, end in segments:
            for edge in (start, end):
                lo = max(1, edge - boundary_width)
                hi = min(length, edge + boundary_width)
                if hi >= lo:
                    target[lo - 1 : hi] = 1.0
    return target


def source_masks(
    length: int,
    domains: Sequence[Sequence[Tuple[int, int]]],
    supervision: Supervision,
    boundary_width: int = 2,
) -> SourceMask:
    """Per-residue and per-record masks implied by one record's supervision."""
    import numpy as np

    annotated = assignment_targets(length, domains)
    if not supervision.residue_labels:
        residue = np.zeros(length, dtype=np.float32)
        return SourceMask(
            assignment=residue,
            boundary=residue.copy(),
            pairwise=False,
            classification=bool(supervision.cath_labels),
            weight=float(supervision.weight),
        )

    if supervision.complete_partition:
        # Every residue has an answer: inside a domain or genuinely linker.
        assignment = np.ones(length, dtype=np.float32)
        boundary = np.ones(length, dtype=np.float32)
    else:
        # Outside an annotated region the source is silent, so neither term is
        # supervised there. Inside, both are.
        assignment = annotated.copy()
        boundary = annotated.copy()

    return SourceMask(
        assignment=assignment,
        boundary=boundary,
        pairwise=bool(supervision.pairwise),
        classification=bool(supervision.cath_labels),
        weight=float(supervision.weight),
    )


# -- terms ------------------------------------------------------------------


def _marginal_log_probabilities(residue_logits, positive, negative):
    """Log P(positive classes) and log P(negative classes), elementwise.

    Computed from the 3-way log-softmax so the two are the logs of a proper
    two-outcome distribution, and the term below is an honest binary
    cross-entropy rather than a rescaled logit.
    """
    import torch
    from torch.nn import functional as F

    log_probability = F.log_softmax(residue_logits.float(), dim=-1)
    pos = torch.logsumexp(log_probability[..., list(positive)], dim=-1)
    neg = torch.logsumexp(log_probability[..., list(negative)], dim=-1)
    return pos, neg


def _weighted_binary_loss(positive_log, negative_log, target, weight):
    """`sum(w * bce) / sum(w)`, or an exact zero when nothing is supervised.

    The zero is produced from the logits rather than as a constant so the term
    stays in the autograd graph: a batch where one term happens to be fully
    masked must not break backward for the others.
    """
    import torch

    total = weight.sum()
    loss = -(target * positive_log + (1.0 - target) * negative_log)
    if float(total) <= 0.0:
        return positive_log.sum() * 0.0
    return (loss * weight).sum() / total


def assignment_loss(residue_logits, target, weight):
    """Is this residue inside a domain? Masked where the source is silent."""
    pos, neg = _marginal_log_probabilities(
        residue_logits, IN_DOMAIN_CLASSES, NOT_IN_DOMAIN_CLASSES
    )
    return _weighted_binary_loss(pos, neg, target, weight)


def boundary_loss(residue_logits, target, weight):
    """Is this residue at a domain edge? Masked the same way."""
    pos, neg = _marginal_log_probabilities(
        residue_logits, BOUNDARY_CLASSES, NOT_BOUNDARY_CLASSES
    )
    return _weighted_binary_loss(pos, neg, target, weight)


def pairwise_loss(pair_logits, pair_target, weight):
    """Do these two residues share a domain? One weight per sampled pair.

    Records whose source cannot answer the question contribute no pairs at
    all, so the gate is applied when the pairs are collected rather than as a
    zero weight here.
    """
    import torch
    from torch.nn import functional as F

    if pair_logits is None or pair_logits.numel() == 0:
        return None
    flat = pair_logits.reshape(-1)
    total = weight.sum()
    if float(total) <= 0.0:
        return flat.sum() * 0.0
    element = F.binary_cross_entropy_with_logits(
        flat.float(), pair_target.reshape(-1).float(), reduction="none"
    )
    return (element * weight).sum() / total


# -- composite --------------------------------------------------------------


@dataclass
class ResidueBatchTargets:
    """Padded assignment and boundary targets plus their masks for a batch."""

    assignment: Any
    assignment_weight: Any
    boundary: Any
    boundary_weight: Any
    pair_weight: Any
    sources: List[str] = field(default_factory=list)

    def to(self, device):
        self.assignment = self.assignment.to(device)
        self.assignment_weight = self.assignment_weight.to(device)
        self.boundary = self.boundary.to(device)
        self.boundary_weight = self.boundary_weight.to(device)
        if self.pair_weight is not None:
            self.pair_weight = self.pair_weight.to(device)
        return self


def build_residue_targets(
    lengths: Sequence[int],
    domains: Sequence[Sequence[Sequence[Tuple[int, int]]]],
    supervisions: Sequence[Supervision],
    boundary_width: int = 2,
    longest: Optional[int] = None,
) -> ResidueBatchTargets:
    """Assemble one batch's residue targets and source-aware weights.

    Padding is weight zero everywhere, so a batch of mixed lengths scores
    exactly as the same chains would one at a time -- the property that makes
    a length-bucketed batch safe to use.
    """
    import numpy as np
    import torch

    width = longest or max(lengths)
    rows = len(lengths)
    assignment = np.zeros((rows, width), dtype=np.float32)
    assignment_weight = np.zeros((rows, width), dtype=np.float32)
    boundary = np.zeros((rows, width), dtype=np.float32)
    boundary_weight = np.zeros((rows, width), dtype=np.float32)

    for row, (length, chain, supervision) in enumerate(
        zip(lengths, domains, supervisions)
    ):
        mask = source_masks(length, chain, supervision, boundary_width)
        assignment[row, :length] = assignment_targets(length, chain)
        boundary[row, :length] = boundary_targets(length, chain, boundary_width)
        assignment_weight[row, :length] = mask.assignment * mask.weight
        boundary_weight[row, :length] = mask.boundary * mask.weight

    return ResidueBatchTargets(
        assignment=torch.from_numpy(assignment),
        assignment_weight=torch.from_numpy(assignment_weight),
        boundary=torch.from_numpy(boundary),
        boundary_weight=torch.from_numpy(boundary_weight),
        pair_weight=None,
    )


def source_aware_loss(
    residue_logits,
    targets: ResidueBatchTargets,
    pair_logits=None,
    pair_target=None,
    pair_weight=None,
    weights: Optional[LossWeights] = None,
) -> Dict[str, Any]:
    """Residue assignment, boundary and pairwise terms, masked by source."""
    weights = weights or LossWeights()

    assignment = assignment_loss(
        residue_logits, targets.assignment, targets.assignment_weight
    )
    boundary = boundary_loss(
        residue_logits, targets.boundary, targets.boundary_weight
    )
    total = weights.assignment * assignment + weights.boundary * boundary
    parts: Dict[str, float] = {
        "assignment": float(assignment.detach()),
        "boundary": float(boundary.detach()),
    }

    pair = None
    if pair_logits is not None and pair_target is not None:
        if pair_weight is None:
            import torch

            pair_weight = torch.ones_like(pair_logits.reshape(-1), dtype=torch.float32)
        pair = pairwise_loss(pair_logits, pair_target, pair_weight)
        if pair is not None:
            total = total + weights.pairwise * pair
            parts["pairwise"] = float(pair.detach())

    return {"loss": total, "parts": parts}
