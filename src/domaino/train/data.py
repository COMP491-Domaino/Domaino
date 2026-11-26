"""Training batches from cached embeddings (tasks T2.7 and T3.2).

Holds the memory controls the proposal asks for in T2.7, on the training side:

length-based batching
    Chains are grouped so that a batch's padded size stays under a token
    budget. A fixed batch size would either waste memory on short chains or
    exceed it on the 4,128-residue ones.

pair sampling
    The pairwise architecture scores a bounded sample of residue pairs rather
    than all of them, since the full matrix grows with the square of length.

The embeddings themselves come from the cache built in T2.2, read through a
memory map, so an epoch touches only the rows it needs rather than loading
133 GB.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..data.schema import ChainRecord, Supervision, read_jsonl
from ..model.embedding_store import EmbeddingStore
from ..model.losses import supervision_for


@dataclass
class ChainExample:
    """One training chain: its identity, length and domain structure.

    `source` and `supervision` travel with the example because T2.6's losses
    are masked per source: without them a Pfam chain would be indistinguishable
    from a CATH one at the point where the loss is computed, and its silence
    about unannotated residues would be read as "linker".
    """

    chain_id: str
    length: int
    domains: List[List[Tuple[int, int]]]
    labels: Dict[str, List[str]] = field(default_factory=dict)
    source: str = "cath"
    supervision: Optional[Supervision] = None

    @property
    def n_domains(self) -> int:
        return len(self.domains)

    @property
    def discontinuous(self) -> bool:
        return any(len(d) > 1 for d in self.domains)

    @property
    def effective_supervision(self) -> Supervision:
        """The record's own supervision, or its source's default."""
        if self.supervision is not None:
            return self.supervision
        return supervision_for(self.source)


def load_examples(
    records_path: str,
    members: Sequence[str],
    store: Optional[EmbeddingStore] = None,
    limit: Optional[int] = None,
    require_labels: bool = False,
) -> List[ChainExample]:
    """Load one split's chains, keeping only those that can actually train.

    A chain is dropped when it has no cached embedding: training on it would
    mean embedding it on the fly mid-epoch, which would make epoch times
    unpredictable and the comparison between architectures unfair.
    """
    keep = set(members)
    examples: List[ChainExample] = []
    for record in read_jsonl(records_path):
        if record.chain_id not in keep or not record.sequence:
            continue
        if store is not None and record.chain_id not in store:
            continue
        domains = [[s.as_tuple() for s in d.segments] for d in record.domains]
        if not domains:
            continue
        labels: Dict[str, List[str]] = {level: [] for level in ("C", "A", "T", "H")}
        for domain in record.domains:
            for level in ("C", "A", "T", "H"):
                code = domain.cath.code(level) if domain.cath else None
                # A truncated code is not a label for a deeper level.
                depth = ("C", "A", "T", "H").index(level) + 1
                labels[level].append(
                    code if code and code.count(".") + 1 == depth else ""
                )
        if require_labels and not any(labels["H"]):
            continue
        examples.append(
            ChainExample(
                record.chain_id,
                record.length,
                domains,
                labels,
                source=record.source,
                supervision=record.supervision,
            )
        )
        if limit and len(examples) >= limit:
            break
    return examples


def batch_by_tokens(
    examples: Sequence[ChainExample],
    max_tokens: int,
    shuffle: bool = True,
    seed: int = 0,
) -> List[List[int]]:
    """Group example indices into batches under a padded-token budget.

    Sorting by length first keeps padding waste small; the batches are then
    shuffled so that epoch order is not correlated with chain length, which
    would otherwise make the loss curve a function of length.
    """
    order = sorted(range(len(examples)), key=lambda i: examples[i].length)
    batches: List[List[int]] = []
    current: List[int] = []
    longest = 0
    for index in order:
        candidate = max(longest, examples[index].length)
        if current and candidate * (len(current) + 1) > max_tokens:
            batches.append(current)
            current, longest = [index], examples[index].length
        else:
            current.append(index)
            longest = candidate
    if current:
        batches.append(current)
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


class EmbeddingBatcher:
    """Assembles padded embedding tensors for a batch of chains."""

    def __init__(self, store: EmbeddingStore, device=None) -> None:
        self.store = store
        self.device = device
        self.dim = store.meta.dim

    def build(self, examples: Sequence[ChainExample]):
        """Return ``(features, mask, lengths)`` for a batch.

        `mask` marks real residues, which every model uses to keep padding out
        of its receptive field and out of its loss.
        """
        import numpy as np
        import torch

        lengths = [e.length for e in examples]
        longest = max(lengths)
        features = np.zeros((len(examples), longest, self.dim), dtype=np.float32)
        mask = np.zeros((len(examples), longest), dtype=np.float32)
        for row, example in enumerate(examples):
            vectors = self.store.get(example.chain_id)
            # The cache was written from the same records, but a truncated or
            # over-long row would silently shift every residue, so it is cut
            # to the length the record declares.
            usable = min(example.length, vectors.shape[0])
            features[row, :usable] = vectors[:usable]
            mask[row, :usable] = 1.0
        out_features = torch.from_numpy(features)
        out_mask = torch.from_numpy(mask)
        if self.device is not None:
            out_features = out_features.to(self.device, non_blocking=True)
            out_mask = out_mask.to(self.device, non_blocking=True)
        return out_features, out_mask, lengths


def iter_epoch(
    examples: Sequence[ChainExample],
    max_tokens: int,
    seed: int,
    shuffle: bool = True,
) -> Iterator[List[ChainExample]]:
    """Yield one epoch's batches as lists of examples."""
    for batch in batch_by_tokens(examples, max_tokens, shuffle=shuffle, seed=seed):
        yield [examples[i] for i in batch]
