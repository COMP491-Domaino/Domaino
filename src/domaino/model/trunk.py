"""Shared residue trunk for both segmentation architectures.

The proposal puts two candidate segmentation designs against each other and
picks the better one on predictive performance (work package 3, T3.4). That
comparison is only about the two designs if everything before them is the same,
so both read the frozen ESM-2 vectors through this identical trunk and differ
only in the head: pairwise same-domain scores in `segmentation.py`, learned
domain queries in `query_segmentation.py`.

A dilated convolutional stack is used rather than self-attention over residues:
its cost is linear in chain length, which matters at 4,128 residues, and a
handful of doubling dilations already sees further than a typical domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass
class TrunkConfig:
    """Shape of the residue trunk, shared by both architectures."""

    input_dim: int = 1280
    hidden_dim: int = 256
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    kernel_size: int = 5
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep length unchanged")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

    @property
    def receptive_field(self) -> int:
        """How many residues can influence one output position."""
        span = 1
        for dilation in self.dilations:
            span += 2 * dilation * (self.kernel_size - 1) // 2
        return span


def build_trunk(config: TrunkConfig):
    """Build the trunk. torch is imported here so importing the module is cheap."""
    from torch import nn

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

    class ResidueTrunk(nn.Module):
        """Frozen ESM-2 vectors in, contextual residue features out."""

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

        @property
        def receptive_field(self) -> int:
            return config.receptive_field

        def forward(self, embeddings, mask):
            features = self.input_proj(embeddings)
            for block in self.blocks:
                features = block(features, mask)
            return features * mask.unsqueeze(-1)

    return ResidueTrunk()
