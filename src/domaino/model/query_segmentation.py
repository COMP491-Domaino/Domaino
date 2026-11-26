"""Query-based domain segmentation: the second candidate architecture (T2.3).

The proposal names two segmentation designs and picks between them on measured
performance in work package 3. The first, in `segmentation.py`, predicts a
same-domain score for every residue pair and needs a clustering step to turn
those scores into domains. This one adapts the set-prediction approach of DETR
and MaskFormer to a one-dimensional sequence instead:

* a fixed set of learned **domain queries** attends to the ESM-2 residue
  vectors through a small transformer decoder;
* each query predicts a **residue-membership mask** over the chain and a
  **domain / no-domain** score;
* during training, **bipartite matching** pairs the unordered predicted masks
  with the true domains, so the model is never asked to reproduce CATH's
  arbitrary domain numbering.

What it buys and what it costs
------------------------------
A mask can select separated positions, so a discontinuous domain is expressed
directly, with no pair matrix and no clustering: decoding is thresholding plus
tidying. The cost is the one the proposal names: a **fixed maximum number of
domains**. `num_queries` defaults to 20 because the most domains any chain in
dataset v1 carries is exactly 20 (`3vkgA`), so at that setting the limit never
binds on this data; a chain with more domains than queries would lose some.

Both candidates read the same `trunk.ResidueTrunk`, so a comparison between
them is a comparison of the heads and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .decode import Domain, Interval, enforce_consistency
from .trunk import TrunkConfig, build_trunk

# The largest number of domains on any chain in dataset v1 (3vkgA). At this
# many queries the architecture's fixed-capacity limitation never binds here.
MAX_DOMAINS_IN_DATASET = 20


@dataclass
class QuerySegmentationConfig:
    """Shape of the query-based segmentation network."""

    input_dim: int = 1280
    hidden_dim: int = 256
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    kernel_size: int = 5
    dropout: float = 0.1

    num_queries: int = MAX_DOMAINS_IN_DATASET
    decoder_layers: int = 3
    num_heads: int = 8
    # Weights of the three terms, used both in the matching cost and the loss,
    # so the model is trained on the criterion it was matched under.
    cost_class: float = 1.0
    cost_mask: float = 5.0
    cost_dice: float = 5.0
    # Weight of the no-domain class in the classification loss. Most queries
    # go unmatched on a single-domain chain, so without this the model learns
    # to answer "no domain" to everything.
    no_domain_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.num_queries < 1:
            raise ValueError("num_queries must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must divide evenly into num_heads")

    def trunk_config(self) -> TrunkConfig:
        return TrunkConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dilations=self.dilations,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )


def sinusoidal_positions(length: int, channels: int, device=None, dtype=None):
    """Standard sinusoidal position encoding, ``(1, length, channels)``.

    Cross-attention is permutation-invariant over its keys, so without this a
    query could not express "attend to this stretch of the chain" and could
    only separate two domains by their content. On synthetic embeddings that
    hand the model domain identity outright it fits either way; on real ESM-2
    vectors, where two domains of a chain can look locally alike, it does not.
    DETR and MaskFormer add positions to the keys for the same reason.
    """
    import torch

    position = torch.arange(length, device=device, dtype=torch.float32)
    half = max(channels // 2, 1)
    frequency = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = position[:, None] * frequency[None, :]
    encoding = torch.zeros(length, channels, device=device, dtype=torch.float32)
    encoding[:, 0:half] = angles.sin()
    encoding[:, half : half * 2] = angles.cos()
    return encoding.unsqueeze(0).to(dtype) if dtype is not None else encoding.unsqueeze(0)


# -- targets ----------------------------------------------------------------


def mask_targets(length: int, domains: Sequence[Sequence[Interval]]):
    """One binary residue mask per true domain, shaped ``(domains, length)``.

    This is the whole target: no per-residue class and no pair matrix. A
    discontinuous domain is simply a mask with gaps in it.
    """
    import numpy as np

    masks = np.zeros((len(domains), length), dtype=np.float32)
    for index, segments in enumerate(domains):
        for start, end in segments:
            lo, hi = max(1, start), min(length, end)
            if hi >= lo:
                masks[index, lo - 1 : hi] = 1.0
    return masks


# -- matching ---------------------------------------------------------------


def dice_cost(pred_prob, target):
    """1 - Dice overlap between predicted mask probabilities and a target.

    Dice sits beside the per-residue loss because it scores the overlap of the
    whole mask. A domain covers a small part of a long chain, so a mask that is
    empty everywhere already scores well per residue; Dice does not reward it.
    """
    import torch

    numerator = 2.0 * torch.einsum("ql,tl->qt", pred_prob, target)
    denominator = pred_prob.sum(-1)[:, None] + target.sum(-1)[None, :]
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def mask_bce_cost(pred_logits, target):
    """Mean per-residue binary cross-entropy for every (query, target) pair."""
    import torch
    from torch.nn import functional as F

    positive = F.binary_cross_entropy_with_logits(
        pred_logits, torch.ones_like(pred_logits), reduction="none"
    )
    negative = F.binary_cross_entropy_with_logits(
        pred_logits, torch.zeros_like(pred_logits), reduction="none"
    )
    # cost[q, t] = mean over residues of (target ? positive : negative)
    cost = torch.einsum("ql,tl->qt", positive, target) + torch.einsum(
        "ql,tl->qt", negative, 1.0 - target
    )
    return cost / pred_logits.shape[-1]


def match_queries(
    mask_logits, class_logits, target_masks, config: QuerySegmentationConfig
) -> Tuple[List[int], List[int]]:
    """Pair predicted masks with true domains, one to one, at least total cost.

    Hungarian assignment on the same three terms the loss uses. Matching is
    what makes the prediction a *set*: the model is scored on which domains it
    found, never on the order it listed them in, so it has no reason to learn
    CATH's numbering.
    """
    import torch
    from scipy.optimize import linear_sum_assignment

    if target_masks.shape[0] == 0:
        return [], []

    with torch.no_grad():
        # Cast out of any autocast dtype: scipy's assignment wants float64-able
        # input, and a half-precision cost matrix silently loses resolution
        # between near-equal candidate matches.
        mask_logits = mask_logits.float()
        class_logits = class_logits.float()
        target_masks = target_masks.float()
        probability = mask_logits.sigmoid()
        # Cost of calling a query a domain: the lower its domain probability,
        # the more it costs to match it to a real one.
        class_cost = -class_logits.softmax(-1)[:, 1][:, None].expand(
            -1, target_masks.shape[0]
        )
        cost = (
            config.cost_class * class_cost
            + config.cost_mask * mask_bce_cost(mask_logits, target_masks)
            + config.cost_dice * dice_cost(probability, target_masks)
        )
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return list(rows), list(cols)


# -- network ----------------------------------------------------------------


def build_query_network(config: Optional[QuerySegmentationConfig] = None):
    """Build the query-based segmentation network."""
    import torch
    from torch import nn

    config = config or QuerySegmentationConfig()

    class DecoderLayer(nn.Module):
        """Queries attend to each other, then to the residues."""

        def __init__(self) -> None:
            super().__init__()
            self.self_attention = nn.MultiheadAttention(
                config.hidden_dim, config.num_heads, dropout=config.dropout,
                batch_first=True,
            )
            self.cross_attention = nn.MultiheadAttention(
                config.hidden_dim, config.num_heads, dropout=config.dropout,
                batch_first=True,
            )
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            )
            self.norm_self = nn.LayerNorm(config.hidden_dim)
            self.norm_cross = nn.LayerNorm(config.hidden_dim)
            self.norm_ffn = nn.LayerNorm(config.hidden_dim)

        def forward(self, hidden, query_pos, residues, residue_pos, key_padding_mask):
            # Positions are added to the attention inputs, not to the values:
            # they say *where* to look, while the value carries what is there.
            keyed = hidden + query_pos
            attended, _ = self.self_attention(keyed, keyed, hidden)
            hidden = self.norm_self(hidden + attended)
            # Padding is masked out, so a padded tail cannot pull a query.
            attended, _ = self.cross_attention(
                hidden + query_pos,
                residues + residue_pos,
                residues,
                key_padding_mask=key_padding_mask,
            )
            hidden = self.norm_cross(hidden + attended)
            return self.norm_ffn(hidden + self.ffn(hidden))

    class QuerySegmentationNetwork(nn.Module):
        """Learned domain queries over frozen ESM-2 residue vectors."""

        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.trunk = build_trunk(config.trunk_config())
            # Initial query state, and the learned positional code that
            # keeps the queries distinguishable at every attention.
            self.queries = nn.Embedding(config.num_queries, config.hidden_dim)
            self.query_pos = nn.Embedding(config.num_queries, config.hidden_dim)
            self.layers = nn.ModuleList(
                DecoderLayer() for _ in range(config.decoder_layers)
            )
            # Two classes: 0 = no domain, 1 = a domain.
            self.class_head = nn.Linear(config.hidden_dim, 2)
            self.mask_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.residue_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        @property
        def receptive_field(self) -> int:
            return self.trunk.receptive_field

        @property
        def num_queries(self) -> int:
            return config.num_queries

        def forward(self, embeddings, mask=None):
            """Return ``class_logits`` (B, Q, 2) and ``mask_logits`` (B, Q, L)."""
            if mask is None:
                mask = torch.ones(
                    embeddings.shape[:2],
                    dtype=embeddings.dtype,
                    device=embeddings.device,
                )
            features = self.trunk(embeddings, mask)

            batch = embeddings.shape[0]
            queries = self.queries.weight.unsqueeze(0).expand(batch, -1, -1)
            query_pos = self.query_pos.weight.unsqueeze(0).expand(batch, -1, -1)
            residue_pos = sinusoidal_positions(
                features.shape[1], features.shape[2], features.device, features.dtype
            )
            key_padding = mask <= 0
            for layer in self.layers:
                queries = layer(
                    queries, query_pos, features, residue_pos, key_padding
                )

            mask_embed = self.mask_head(queries)
            projected = self.residue_proj(features)
            mask_logits = torch.einsum("bqc,blc->bql", mask_embed, projected)
            # A padded position belongs to no domain, whatever the head says.
            mask_logits = mask_logits.masked_fill(
                key_padding.unsqueeze(1), float("-inf")
            )
            return {
                "class_logits": self.class_head(queries),
                "mask_logits": mask_logits,
                "features": features,
            }

    return QuerySegmentationNetwork()


# -- loss -------------------------------------------------------------------


def query_segmentation_loss(
    class_logits,
    mask_logits,
    target_masks,
    config: Optional[QuerySegmentationConfig] = None,
    lengths: Optional[Sequence[int]] = None,
    chain_weights: Optional[Sequence[float]] = None,
):
    """Set-prediction loss over a batch.

    `target_masks` is one ``(domains, length)`` array per chain, so chains with
    different domain counts batch together. Matched queries are trained on
    their domain's mask; every unmatched query is trained to answer no-domain.

    `chain_weights` is T2.6's source mask for this architecture: one weight per
    chain, and zero drops the chain entirely. The gate is per chain rather
    than per residue because set prediction needs the **complete** set of
    domains -- an unmatched query is supervised to say "no domain", which is
    only true if the targets really are every domain in the chain. A source
    that annotates some regions and stays silent about the rest, as Pfam does,
    therefore cannot train this architecture at all, whereas the pairwise
    design can still use it through per-residue masks.
    """
    import torch
    from torch.nn import functional as F

    config = config or QuerySegmentationConfig()
    device = class_logits.device
    batch, num_queries, _ = class_logits.shape

    class_weight = torch.tensor(
        [config.no_domain_weight, 1.0], dtype=class_logits.dtype, device=device
    )

    total_class = class_logits.new_zeros(())
    total_mask = class_logits.new_zeros(())
    total_dice = class_logits.new_zeros(())
    matched_total = 0
    class_divisor = 0.0
    matched_divisor = 0.0

    for index in range(batch):
        weight = 1.0 if chain_weights is None else float(chain_weights[index])
        if weight <= 0.0:
            continue
        length = int(lengths[index]) if lengths is not None else mask_logits.shape[-1]
        targets = torch.as_tensor(
            target_masks[index], dtype=class_logits.dtype, device=device
        )
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        targets = targets[:, :length]
        logits = mask_logits[index][:, :length]

        labels = torch.zeros(num_queries, dtype=torch.long, device=device)
        rows, cols = match_queries(logits, class_logits[index], targets, config)
        if rows:
            labels[torch.tensor(rows, device=device)] = 1
            picked = logits[torch.tensor(rows, device=device)]
            wanted = targets[torch.tensor(cols, device=device)]
            total_mask = total_mask + weight * F.binary_cross_entropy_with_logits(
                picked, wanted
            )
            overlap = dice_cost(picked.sigmoid(), wanted)
            total_dice = total_dice + weight * overlap.diagonal().mean()
            matched_total += len(rows)
            matched_divisor += weight

        total_class = total_class + weight * F.cross_entropy(
            class_logits[index], labels, weight=class_weight
        )
        class_divisor += weight

    if class_divisor <= 0:
        # Every chain was masked out by T2.6's source gate. The loss is zero,
        # but it has to stay attached to the graph or backward fails for the
        # rest of the epoch. It is built from a single logit rather than from
        # a reduction over the tensor, because `mask_logits` holds -inf at
        # padded positions and -inf * 0 is NaN.
        zero = class_logits.reshape(-1)[:1].sum() * 0.0
        return {
            "loss": zero,
            "class_loss": zero.detach(),
            "mask_loss": zero.detach(),
            "dice_loss": zero.detach(),
            "matched_queries": 0,
        }

    # Weighted means, so a batch's loss does not depend on how many chains a
    # source mask happened to drop from it.
    matched_batches = matched_divisor if matched_divisor > 0 else 1.0
    loss = (
        config.cost_class * total_class / class_divisor
        + config.cost_mask * total_mask / matched_batches
        + config.cost_dice * total_dice / matched_batches
    )
    return {
        "loss": loss,
        "class_loss": (total_class / class_divisor).detach(),
        "mask_loss": (total_mask / matched_batches).detach(),
        "dice_loss": (total_dice / matched_batches).detach(),
        "matched_queries": matched_total,
    }


# -- decoding ---------------------------------------------------------------


@dataclass
class QueryDecoderConfig:
    """Thresholds for turning query outputs into domains.

    No clustering step appears here, which is the architectural point: a mask
    already names a domain's residues. What remains is tidying, and it enforces
    the same constraints the pairwise decoder does so the two architectures are
    scored on comparable output.
    """

    domain_threshold: float = 0.5
    mask_threshold: float = 0.5
    min_domain_length: int = 40
    max_linker_gap: int = 20


def intervals_from_mask(flags: Sequence[bool]) -> List[Interval]:
    """Maximal 1-based inclusive runs of True."""
    out: List[Interval] = []
    start: Optional[int] = None
    for index, flag in enumerate(flags, start=1):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            out.append((start, index - 1))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def decode_queries(
    class_logits,
    mask_logits,
    length: int,
    config: Optional[QueryDecoderConfig] = None,
) -> List[Domain]:
    """Turn one chain's query outputs into domain intervals.

    A residue claimed by several queries goes to the most confident one, since
    overlapping domains cannot be written as a CATH chopping.
    """
    import numpy as np
    import torch

    config = config or QueryDecoderConfig()

    with torch.no_grad():
        scores = torch.softmax(torch.as_tensor(class_logits), dim=-1)[:, 1]
        probability = torch.sigmoid(torch.as_tensor(mask_logits)[:, :length])
    scores = scores.detach().cpu().numpy()
    probability = probability.detach().cpu().numpy()
    probability = np.nan_to_num(probability, nan=0.0)

    keep = [q for q in range(len(scores)) if scores[q] >= config.domain_threshold]
    if not keep:
        # Every CATH chain has at least one domain, so answering "none" is
        # never right; fall back to the single most confident query.
        keep = [int(np.argmax(scores))]

    # Assign each residue to the confident query that claims it most strongly.
    claim = probability[keep] >= config.mask_threshold
    strength = np.where(claim, probability[keep], -1.0)
    owner = np.argmax(strength, axis=0)
    claimed = claim.any(axis=0)

    domains: List[Domain] = []
    confidence: List[float] = []
    for slot, query in enumerate(keep):
        flags = claimed & (owner == slot)
        intervals = _merge_over_unclaimed(
            intervals_from_mask(flags.tolist()), claimed, config.max_linker_gap
        )
        total = sum(end - start + 1 for start, end in intervals)
        if intervals and total >= config.min_domain_length:
            domains.append(intervals)
            confidence.append(float(scores[query]))

    if not domains:
        return [[(1, length)]]

    # Gap merging can still leave two domains touching, so the usual parse
    # check runs last; the more confident query keeps a contested residue.
    order = sorted(range(len(domains)), key=lambda i: -confidence[i])
    domains = enforce_consistency(domains, length, order=order)
    return domains or [[(1, length)]]


def _merge_over_unclaimed(
    intervals: Sequence[Interval], claimed, max_gap: int
) -> List[Interval]:
    """Join a domain's intervals only across residues no query claimed.

    A short run the model was unsure about belongs inside the domain, but a
    run another query owns does not: bridging it would hand one domain's
    residues to another and produce an overlap no CATH chopping can express.
    """
    merged: List[Interval] = []
    for start, end in sorted(intervals):
        if merged:
            previous_start, previous_end = merged[-1]
            gap = start - previous_end - 1
            # Gap positions are previous_end+1 .. start-1, 0-based slice below.
            if 0 <= gap <= max_gap and not claimed[previous_end : start - 1].any():
                merged[-1] = (previous_start, max(previous_end, end))
                continue
        merged.append((start, end))
    return merged
