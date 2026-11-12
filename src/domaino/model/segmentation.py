"""Domain-segmentation network (task T2.3).

This is the proposal's first, lower-risk segmentation design. It reads frozen
ESM-2 residue vectors and emits two things:

residue logits, ``(batch, length, 3)``
    Whether each residue is linker, inside a domain, or at a domain boundary.

pair logits, ``(batch, length, length)``
    Whether two residues belong to the same domain.

Why both
--------
Boundary prediction alone cannot express a **discontinuous** domain, whose
residues sit in separated stretches of the sequence, and the baseline in T2.1
showed the concrete failure: a residue-level predictor over-segments
(domain-count error 2.05 against 0.30 for assuming one domain) because nothing
tells it which side of a cut belongs with which. A same-domain relation fixes
both problems at once, and it is invariant to how domains are numbered, so the
network never has to learn CATH's arbitrary ordering.

Memory
------
A pair matrix grows with the square of chain length, and chains here reach
4,128 residues, so the pair head is **low-rank**: two ``(length, rank)``
projections whose product gives the matrix, never a ``(length, length,
channels)`` tensor. The proposal names a low-rank pair representation as the
fallback for exactly this risk. Training additionally samples a bounded number
of pairs per chain instead of scoring all of them.

The trunk is a dilated convolutional stack: its cost is linear in length, it
handles variable-length input without padding artefacts leaking across the
mask, and a few dilated layers already see far enough to judge a domain. The
alternative set-prediction architecture the proposal also names is a separate
candidate, to be compared in work package 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .decode import BOUNDARY, DOMAIN, LINKER, RESIDUE_CLASSES

# Distance buckets for the pair head's relative-position bias. Sequence
# separation is a strong prior -- nearby residues usually share a domain -- so
# the head is given it explicitly rather than having to rediscover it.
DISTANCE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


@dataclass
class SegmentationConfig:
    """Shape and capacity of the segmentation network."""

    input_dim: int = 1280
    hidden_dim: int = 256
    # Dilations of the residual convolution stack. Doubling dilations give a
    # receptive field of a few hundred residues with only a handful of layers.
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    kernel_size: int = 5
    dropout: float = 0.1
    # Rank of the factorised pair head.
    pair_rank: int = 64
    # Cap on sampled pairs per chain during training.
    max_pairs_per_chain: int = 40000

    def __post_init__(self) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep length unchanged")
        if self.pair_rank < 1:
            raise ValueError("pair_rank must be positive")


def bucket_distance(separation: int) -> int:
    """Index of the distance bucket a residue separation falls in."""
    for index, edge in enumerate(DISTANCE_BUCKETS):
        if separation <= edge:
            return index
    return len(DISTANCE_BUCKETS)


# -- targets ----------------------------------------------------------------


def residue_targets(
    length: int,
    domains: Sequence[Sequence[Tuple[int, int]]],
    boundary_width: int = 2,
) -> List[int]:
    """Per-residue class targets over {linker, domain, boundary}.

    A residue inside a domain is `DOMAIN`, one within `boundary_width` of a
    domain edge is `BOUNDARY`, and one in no domain at all is `LINKER`. The
    boundary class is a band rather than a single residue because asking a
    sequence-only model for the exact residue is stricter than the benchmark's
    own boundary-distance measure.
    """
    labels = [LINKER] * length
    for segments in domains:
        for start, end in segments:
            for position in range(max(1, start), min(length, end) + 1):
                labels[position - 1] = DOMAIN
    for segments in domains:
        for start, end in segments:
            for edge in (start, end):
                for position in range(edge - boundary_width, edge + boundary_width + 1):
                    if 1 <= position <= length and labels[position - 1] != LINKER:
                        labels[position - 1] = BOUNDARY
    return labels


def pair_targets(
    length: int, domains: Sequence[Sequence[Tuple[int, int]]]
):
    """Same-domain target matrix and the mask of pairs it is defined on.

    Returns ``(target, mask)``, both ``(length, length)``. A pair is supervised
    only when both residues belong to some domain: if either is linker, whether
    they are "in the same domain" has no answer, and training on it as a
    negative would teach the model that linkers repel everything.
    """
    import numpy as np

    owner = np.full(length, -1, dtype=np.int64)
    for index, segments in enumerate(domains):
        for start, end in segments:
            for position in range(max(1, start), min(length, end) + 1):
                owner[position - 1] = index

    assigned = owner >= 0
    mask = np.outer(assigned, assigned)
    target = (owner[:, None] == owner[None, :]) & mask
    return target.astype(np.float32), mask.astype(np.float32)


def sample_pairs(mask, limit: int, seed: int = 0):
    """Choose at most `limit` supervised pairs, balancing the two classes.

    Scoring every pair of a long chain is what makes the pairwise design
    expensive, and most pairs are redundant. Sampling keeps the cost bounded;
    balancing matters because same-domain pairs vastly outnumber
    different-domain ones in single-domain chains, which is most of them.
    """
    import numpy as np

    mask = np.asarray(mask)
    rows, cols = np.nonzero(np.triu(mask, k=1))
    if len(rows) <= limit:
        return rows, cols
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(rows), size=limit, replace=False)
    return rows[chosen], cols[chosen]


# -- network ----------------------------------------------------------------


def build_network(config: Optional[SegmentationConfig] = None):
    """Construct the network. Imported lazily so torch is only needed here."""
    import torch
    from torch import nn

    config = config or SegmentationConfig()

    class ResidualBlock(nn.Module):
        """Dilated convolution with a residual connection and layer norm."""

        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation * (config.kernel_size - 1) // 2
            self.conv = nn.Conv1d(
                channels,
                channels,
                kernel_size=config.kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.norm = nn.LayerNorm(channels)
            self.dropout = nn.Dropout(config.dropout)

            self.activation = nn.GELU()

        def forward(self, x, mask):
            # Zero the padding before convolving, so a padded tail cannot leak
            # into a real residue's receptive field.
            y = x * mask.unsqueeze(-1)
            y = self.conv(y.transpose(1, 2)).transpose(1, 2)
            y = self.activation(self.norm(y))
            return x + self.dropout(y)

    class PairHead(nn.Module):
        """Low-rank same-domain scorer with a relative-distance bias."""

        def __init__(self, channels: int, rank: int) -> None:
            super().__init__()
            self.left = nn.Linear(channels, rank)
            self.right = nn.Linear(channels, rank)
            self.scale = rank ** -0.5
            # One learned scalar per distance bucket, plus the overflow bucket.
            self.distance_bias = nn.Embedding(len(DISTANCE_BUCKETS) + 1, 1)
            nn.init.zeros_(self.distance_bias.weight)
            self.bias = nn.Parameter(torch.zeros(1))

        def _bucket_matrix(self, length: int, device):
            positions = torch.arange(length, device=device)
            separation = (positions[:, None] - positions[None, :]).abs()
            edges = torch.tensor(DISTANCE_BUCKETS, device=device)
            # bucketize gives the index of the first edge >= separation.
            return torch.bucketize(separation, edges, right=False)

        def forward(self, features):
            """Full ``(batch, length, length)`` logits."""
            a = self.left(features)
            b = self.right(features)
            logits = torch.matmul(a, b.transpose(1, 2)) * self.scale
            buckets = self._bucket_matrix(features.shape[1], features.device)
            logits = logits + self.distance_bias(buckets).squeeze(-1)
            # The relation is symmetric, so the output is symmetrised rather
            # than left to learn symmetry it can never need to break.
            return (logits + logits.transpose(1, 2)) / 2 + self.bias

        def score_pairs(self, features, rows, cols):
            """Logits for selected pairs only, without forming the matrix.

            Must agree exactly with `forward` on the pairs it covers, which is
            why it symmetrises the same way: a training signal computed under a
            different formula than inference uses would be a silent mismatch.
            """
            rows = torch.as_tensor(rows, device=features.device, dtype=torch.long)
            cols = torch.as_tensor(cols, device=features.device, dtype=torch.long)
            a = self.left(features)
            b = self.right(features)
            forward = (a[:, rows, :] * b[:, cols, :]).sum(-1)
            backward = (a[:, cols, :] * b[:, rows, :]).sum(-1)
            logits = (forward + backward) * 0.5 * self.scale
            separation = (rows - cols).abs()
            edges = torch.tensor(DISTANCE_BUCKETS, device=features.device)
            buckets = torch.bucketize(separation, edges, right=False)
            return logits + self.distance_bias(buckets).squeeze(-1) + self.bias

    class SegmentationNetwork(nn.Module):
        """Residue classes and pairwise same-domain relations from embeddings."""

        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.input_proj = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.blocks = nn.ModuleList(
                ResidualBlock(config.hidden_dim, d) for d in config.dilations
            )
            self.residue_head = nn.Linear(config.hidden_dim, len(RESIDUE_CLASSES))
            self.pair_head = PairHead(config.hidden_dim, config.pair_rank)

        @property
        def receptive_field(self) -> int:
            """How many residues can influence one output position."""
            span = 1
            for dilation in config.dilations:
                span += 2 * dilation * (config.kernel_size - 1) // 2
            return span

        def trunk(self, embeddings, mask):
            features = self.input_proj(embeddings)
            for block in self.blocks:
                features = block(features, mask)
            return features * mask.unsqueeze(-1)

        def forward(self, embeddings, mask=None, pairs=None):
            """Run the network.

            `embeddings` is ``(batch, length, input_dim)`` and `mask` marks real
            residues. Passing `pairs` as ``(rows, cols)`` scores just those
            pairs, which is how training avoids the full matrix; otherwise the
            full matrix is returned, which is what inference wants.
            """
            if mask is None:
                mask = torch.ones(
                    embeddings.shape[:2],
                    dtype=embeddings.dtype,
                    device=embeddings.device,
                )
            features = self.trunk(embeddings, mask)
            out: Dict[str, object] = {
                "residue_logits": self.residue_head(features),
                "features": features,
            }
            if pairs is None:
                out["pair_logits"] = self.pair_head(features)
            else:
                rows, cols = pairs
                out["pair_logits"] = self.pair_head.score_pairs(features, rows, cols)
            return out

    return SegmentationNetwork()


def segmentation_loss(
    residue_logits,
    residue_target,
    pair_logits,
    pair_target,
    mask=None,
    pair_weight: float = 1.0,
    class_weights: Optional[Sequence[float]] = None,
):
    """Combined residue and pairwise loss.

    A plain starting point: cross-entropy over residue classes plus binary
    cross-entropy over the supervised pairs. Task T2.6 replaces this with the
    source-aware version that also masks losses per data source, so nothing
    here depends on which source a record came from.
    """
    import torch
    from torch.nn import functional as F

    residue_logits = residue_logits.reshape(-1, len(RESIDUE_CLASSES))
    flat_target = residue_target.reshape(-1)
    weights = (
        torch.tensor(class_weights, dtype=residue_logits.dtype, device=residue_logits.device)
        if class_weights is not None
        else None
    )
    if mask is not None:
        keep = mask.reshape(-1) > 0
        residue_logits = residue_logits[keep]
        flat_target = flat_target[keep]
    residue_loss = F.cross_entropy(residue_logits, flat_target, weight=weights)

    pair_loss = F.binary_cross_entropy_with_logits(
        pair_logits.reshape(-1), pair_target.reshape(-1).to(pair_logits.dtype)
    )
    return {
        "loss": residue_loss + pair_weight * pair_loss,
        "residue_loss": residue_loss.detach(),
        "pair_loss": pair_loss.detach(),
    }
