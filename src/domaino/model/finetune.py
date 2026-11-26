"""ESM-2 inside the training graph, so the encoder can be fine-tuned.

Everything up to now read **precomputed** residue embeddings from the T2.2
cache, which is why the encoder was frozen: it was never in the graph at all.
The proposal plans it that way -- "the encoder will initially remain frozen to
reduce training cost; limited fine-tuning will be considered only if it
produces a benefit" -- so this module is that second step, not a correction.

What changes when the encoder is unfrozen
-----------------------------------------
The cost moves by two orders of magnitude. A frozen epoch reads 125 GB of
memmapped fp16 and spends its time on a 4.6M-parameter head. An unfrozen epoch
runs 651M parameters forward *and backward* over every residue, and pays it
again on the validation split. `scripts/calibrate_finetune.py` measures the
real rate rather than guessing at it.

Two controls make it fit on one 40 GB A100:

`trainable_layers`
    How many of the 33 transformer layers get gradients, counted from the top.
    Each layer is 19.7M parameters, and AdamW holds two moments per parameter,
    so the fixed cost is about 16 bytes per trainable parameter before a single
    activation is stored. Full fine-tuning is 651M parameters and 10.4 GB of
    that fixed cost; the top 8 layers are 157M and 2.5 GB.

`gradient_checkpointing`
    Recompute each transformer layer's activations during the backward pass
    instead of keeping them. Costs roughly a third more compute and turns
    activation memory from the binding constraint into a non-issue, which is
    what allows a token budget large enough to be worth running.

The embedding table, the language-model head and the contact head never get
gradients: the first is 42K parameters of lookup, and the other two are not on
the path to a residue representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .representation import EMBED_DIMS, batch_by_tokens, plan_windows

# Parts of ESM-2 that are never trained here, whatever `trainable_layers` says.
ALWAYS_FROZEN = ("embed_tokens", "lm_head", "contact_head")


@dataclass
class FinetuneConfig:
    """How the encoder participates in training."""

    model_name: str = "esm2_t33_650M_UR50D"
    # Representation layer; None means the last one, as in T2.2.
    layer: Optional[int] = None
    # Transformer layers to train, counted from the top. 0 means all of them.
    trainable_layers: int = 0
    gradient_checkpointing: bool = True
    # ESM-2's training crop, and the overlap used to stitch longer chains.
    window_size: int = 1022
    window_overlap: int = 256
    # Padded residues per encoder forward. Much smaller than the frozen path's
    # budget, because activations now have to be kept or recomputed.
    max_tokens_per_forward: int = 8192
    torch_home: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model_name not in EMBED_DIMS:
            raise KeyError(f"unknown ESM-2 variant: {self.model_name}")
        if self.trainable_layers < 0:
            raise ValueError("trainable_layers cannot be negative")
        if self.window_overlap >= self.window_size:
            raise ValueError("window_overlap must be smaller than window_size")

    @property
    def embed_dim(self) -> int:
        return EMBED_DIMS[self.model_name]


def build_trainable_encoder(config: Optional[FinetuneConfig] = None):
    """Load ESM-2 as a trainable module. Imported lazily, as elsewhere."""
    import os

    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    config = config or FinetuneConfig()
    if config.torch_home:
        os.environ.setdefault("TORCH_HOME", config.torch_home)

    import esm

    if not hasattr(esm.pretrained, config.model_name):
        raise KeyError(f"fair-esm has no model {config.model_name!r}")
    base, alphabet = getattr(esm.pretrained, config.model_name)()

    class CheckpointedLayer(nn.Module):
        """One transformer layer whose activations are recomputed on backward.

        Wrapping the layer rather than reimplementing ESM-2's forward loop
        keeps this correct if fair-esm changes: the layer's own signature is
        all this depends on. `use_reentrant=False` is required because the
        call takes a keyword argument and a bool.
        """

        def __init__(self, layer: nn.Module) -> None:
            super().__init__()
            self.layer = layer

        def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None,
                    need_head_weights=False):
            if not torch.is_grad_enabled() or not any(
                p.requires_grad for p in self.layer.parameters()
            ):
                # Nothing to recompute for: a frozen layer, or inference.
                return self.layer(
                    x,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_head_weights=need_head_weights,
                )

            def run(hidden, padding_mask):
                return self.layer(
                    hidden,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=need_head_weights,
                )

            return checkpoint(run, x, self_attn_padding_mask, use_reentrant=False)

    class TrainableEncoder(nn.Module):
        """ESM-2 producing residue representations that carry gradients."""

        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.esm = base
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()
            self.num_layers = base.num_layers
            self.repr_layer = (
                config.layer if config.layer is not None else base.num_layers
            )

            # Freeze first, then re-enable the top `trainable_layers`.
            for parameter in self.esm.parameters():
                parameter.requires_grad_(False)
            keep = (
                range(self.num_layers)
                if config.trainable_layers == 0
                else range(
                    max(0, self.num_layers - config.trainable_layers),
                    self.num_layers,
                )
            )
            self.trainable_layer_indices = list(keep)
            for index in self.trainable_layer_indices:
                for parameter in self.esm.layers[index].parameters():
                    parameter.requires_grad_(True)
            # The final layer norm sits on the representation path, so it is
            # trained whenever anything above it is.
            if self.trainable_layer_indices:
                for parameter in self.esm.emb_layer_norm_after.parameters():
                    parameter.requires_grad_(True)
            for name in ALWAYS_FROZEN:
                module = getattr(self.esm, name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)

            if config.gradient_checkpointing:
                for index in range(self.num_layers):
                    self.esm.layers[index] = CheckpointedLayer(
                        self.esm.layers[index]
                    )

        @property
        def embed_dim(self) -> int:
            return self.config.embed_dim

        def trainable_parameters(self):
            return [p for p in self.esm.parameters() if p.requires_grad]

        def trainable_count(self) -> int:
            return sum(p.numel() for p in self.trainable_parameters())

        def _run(self, pieces: Sequence[str], device):
            """Representations for sequences that each fit one window."""
            data = [(f"s{i}", piece) for i, piece in enumerate(pieces)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(device)
            out = self.esm(tokens, repr_layers=[self.repr_layer])
            reps = out["representations"][self.repr_layer]
            # Residue i of a sequence of length L is at token index i.
            return [reps[i, 1 : len(p) + 1] for i, p in enumerate(pieces)]

        def forward(self, sequences: Sequence[str], device=None):
            """Padded ``(batch, longest, dim)`` features plus a residue mask.

            Same shape contract as the frozen path's `EmbeddingBatcher.build`,
            so the trainer does not care which one produced its input. Long
            chains go through overlapping windows whose shared positions are
            averaged, exactly as the cache was built, and that averaging is
            differentiable.
            """
            device = device or next(self.esm.parameters()).device
            longest = max(len(s) for s in sequences)

            jobs: List[Tuple[int, int, int]] = []
            for index, sequence in enumerate(sequences):
                if not sequence:
                    raise ValueError(f"sequence {index} is empty")
                for start, end in plan_windows(
                    len(sequence),
                    self.config.window_size,
                    self.config.window_overlap,
                ):
                    jobs.append((index, start, end))

            counts = [
                torch.zeros(len(s), 1, device=device) for s in sequences
            ]
            # Accumulate into a list per sequence and sum once, so the graph
            # holds one add per window instead of an in-place scatter that
            # autograd would have to version-track.
            parts: List[List[Tuple[int, int, Any]]] = [[] for _ in sequences]

            lengths = [end - start for _, start, end in jobs]
            for group in batch_by_tokens(
                lengths, self.config.max_tokens_per_forward
            ):
                pieces = [
                    sequences[jobs[j][0]][jobs[j][1] : jobs[j][2]] for j in group
                ]
                for j, piece in zip(group, self._run(pieces, device)):
                    index, start, end = jobs[j]
                    parts[index].append((start, end, piece))
                    counts[index][start:end] += 1.0

            mask = torch.zeros(len(sequences), longest, device=device)
            rows = []
            for index, sequence in enumerate(sequences):
                total = None
                for start, end, piece in parts[index]:
                    padded = torch.nn.functional.pad(
                        piece, (0, 0, start, len(sequence) - end)
                    )
                    total = padded if total is None else total + padded
                row = total / counts[index]
                # Padded to the batch width here and stacked below, rather
                # than written into a preallocated tensor: an in-place write
                # into a tensor the graph already holds is the kind of thing
                # autograd versions-tracks and then refuses.
                rows.append(
                    torch.nn.functional.pad(
                        row, (0, 0, 0, longest - len(sequence))
                    )
                )
                mask[index, : len(sequence)] = 1.0
            return torch.stack(rows), mask, [len(s) for s in sequences]

    return TrainableEncoder()
