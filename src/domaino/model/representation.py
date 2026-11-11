"""ESM-2 protein-sequence representation pipeline (task T2.2).

Turns an amino-acid sequence into one contextual vector per residue, which is
the only input Domaino's model is allowed: the proposal's whole point is that
no structure is required.

The encoder stays frozen. The proposal keeps it that way to cut training cost
and only considers limited fine-tuning if validation clearly improves, so
`Esm2Encoder` never builds a gradient graph and leaves the model in `eval()`.

Three details matter and are easy to get wrong:

Token layout
    fair-esm returns ``(batch, 2 + max_len, dim)`` because it prepends a
    beginning-of-sequence token and appends an end-of-sequence token. Residue
    ``i`` of a sequence therefore sits at index ``i`` (1-based) in that tensor,
    and the trailing padding belongs to no residue at all. `embed` slices each
    sequence back to exactly its own length so callers never see either.

Length
    ESM-2 was trained on crops of 1024 tokens, i.e. 1022 residues. Chains in
    this dataset reach 4,128 residues, so anything longer is embedded in
    overlapping windows whose shared positions are averaged, which is the
    overlapping-window handling the proposal calls for.

Batching
    Padding a batch to its longest member wastes compute, so sequences are
    grouped by length under a token budget rather than a fixed batch size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

# Residue vector width per cached ESM-2 variant, so callers can size a head
# without loading the encoder.
EMBED_DIMS: Dict[str, int] = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
}

# ESM-2's training crop is 1024 tokens, two of which are the special tokens.
MAX_TRAINED_RESIDUES = 1022


@dataclass
class EncoderConfig:
    """How to run ESM-2.

    `model_name` is one of fair-esm's pretrained entry points. The proposal
    leaves the size to be chosen on memory, speed and validation score, so it
    is a plain configuration value; 650M is the default because it is the
    variant those comparisons usually start from.
    """

    model_name: str = "esm2_t33_650M_UR50D"
    # Which transformer layer to read. None means the last one, which is what
    # ESM-2's own downstream tasks use.
    layer: Optional[int] = None
    device: str = "auto"
    # fp16 on GPU. The encoder is frozen, so reduced precision costs nothing
    # that training could not absorb, and it halves both memory and cache size.
    half: bool = True
    # Padding budget per forward pass, counted in tokens rather than sequences
    # so a batch of long chains does not blow up memory.
    max_tokens_per_batch: int = 16384
    window_size: int = MAX_TRAINED_RESIDUES
    window_overlap: int = 256
    torch_home: Optional[str] = None

    def __post_init__(self) -> None:
        if self.window_size < 16:
            raise ValueError("window_size must leave room for a real window")
        if not 0 <= self.window_overlap < self.window_size:
            raise ValueError("window_overlap must be smaller than window_size")
        if self.max_tokens_per_batch < self.window_size + 2:
            raise ValueError(
                "max_tokens_per_batch must fit at least one full window"
            )

    @property
    def embed_dim(self) -> int:
        if self.model_name not in EMBED_DIMS:
            raise KeyError(f"unknown ESM-2 variant: {self.model_name}")
        return EMBED_DIMS[self.model_name]


def plan_windows(
    length: int, window_size: int, overlap: int
) -> List[Tuple[int, int]]:
    """Cover ``[0, length)`` with windows of at most `window_size`.

    Windows are returned as half-open ``[start, end)`` pairs. A sequence that
    fits yields a single window. The last window is pulled back to end exactly
    at `length` so no residue is embedded with less context than the others
    merely because it sits at the tail.
    """
    if length <= window_size:
        return [(0, length)]
    step = window_size - overlap
    starts = list(range(0, length - window_size + 1, step))
    if starts[-1] + window_size < length:
        starts.append(length - window_size)
    return [(start, start + window_size) for start in starts]


def batch_by_tokens(
    lengths: Sequence[int], max_tokens: int
) -> List[List[int]]:
    """Group indices into batches of similar length under a token budget.

    Returns lists of indices into `lengths`. Sorting by length first keeps the
    padding waste small; the budget counts the padded size a batch would
    actually occupy, which is what decides whether it fits in memory.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: List[List[int]] = []
    current: List[int] = []
    current_max = 0
    for index in order:
        candidate_max = max(current_max, lengths[index] + 2)
        if current and candidate_max * (len(current) + 1) > max_tokens:
            batches.append(current)
            current, current_max = [index], lengths[index] + 2
        else:
            current.append(index)
            current_max = candidate_max
    if current:
        batches.append(current)
    return batches


class Esm2Encoder:
    """Frozen ESM-2 encoder producing one vector per residue."""

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        self.config = config or EncoderConfig()
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._device = None

    # -- setup ---------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import os

        if self.config.torch_home:
            # Weights are read from here, which lets a compute node without
            # outbound internet use an already-populated cache.
            os.environ.setdefault("TORCH_HOME", self.config.torch_home)

        import esm
        import torch

        if not hasattr(esm.pretrained, self.config.model_name):
            raise KeyError(f"fair-esm has no model {self.config.model_name!r}")
        model, alphabet = getattr(esm.pretrained, self.config.model_name)()

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        model = model.eval().to(self._device)
        # Frozen: no parameter of the encoder is ever updated.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if self.config.half and self._device.type == "cuda":
            model = model.half()

        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        LOG.info(
            "loaded %s on %s (dim=%d, layers=%d)",
            self.config.model_name,
            self._device,
            self.embed_dim,
            model.num_layers,
        )

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim

    @property
    def layer(self) -> int:
        self._load()
        return (
            self.config.layer
            if self.config.layer is not None
            else self._model.num_layers  # type: ignore[union-attr]
        )

    @property
    def device(self):
        self._load()
        return self._device

    # -- embedding -----------------------------------------------------------

    def _forward(self, sequences: Sequence[str]):
        """Embed sequences that already fit in one window, in one batch."""
        import torch

        self._load()
        data = [(f"s{i}", s) for i, s in enumerate(sequences)]
        _, _, tokens = self._batch_converter(data)  # type: ignore[misc]
        tokens = tokens.to(self._device)
        with torch.no_grad():
            out = self._model(tokens, repr_layers=[self.layer])  # type: ignore[misc]
        reps = out["representations"][self.layer]
        # Drop the BOS token and the EOS/padding tail: residue i of a sequence
        # of length L is at index i, for i in 1..L.
        return [reps[i, 1 : len(s) + 1].float().cpu() for i, s in enumerate(sequences)]

    def embed(self, sequences: Sequence[str]):
        """Return one ``(length, dim)`` float tensor per input sequence.

        Order is preserved. Long sequences go through overlapping windows and
        their shared positions are averaged, so a residue near a window seam
        is not represented by whichever window happened to end there.
        """
        import torch

        self._load()
        if not sequences:
            return []

        # Flatten every (sequence, window) pair into one work list, so batching
        # sees uniform items regardless of which sequence they came from.
        jobs: List[Tuple[int, int, int]] = []
        for index, sequence in enumerate(sequences):
            if not sequence:
                raise ValueError(f"sequence {index} is empty")
            for start, end in plan_windows(
                len(sequence), self.config.window_size, self.config.window_overlap
            ):
                jobs.append((index, start, end))

        totals = [
            torch.zeros(len(s), self.embed_dim, dtype=torch.float32)
            for s in sequences
        ]
        counts = [torch.zeros(len(s), 1, dtype=torch.float32) for s in sequences]

        job_lengths = [end - start for _, start, end in jobs]
        for batch in batch_by_tokens(job_lengths, self.config.max_tokens_per_batch):
            pieces = [sequences[jobs[j][0]][jobs[j][1] : jobs[j][2]] for j in batch]
            for j, piece_repr in zip(batch, self._forward(pieces)):
                index, start, end = jobs[j]
                totals[index][start:end] += piece_repr
                counts[index][start:end] += 1.0

        return [total / count for total, count in zip(totals, counts)]

    def embed_one(self, sequence: str):
        """Convenience wrapper for a single sequence."""
        return self.embed([sequence])[0]
