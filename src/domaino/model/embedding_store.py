"""Optional on-disk cache for ESM-2 residue embeddings (task T2.2).

Embedding is the slowest part of a training step while the encoder is frozen,
and the result never changes, so it is worth storing once. Caching the training
split at 650M costs roughly 140 GB in fp16, which the cluster's scratch
filesystem absorbs easily.

Layout: a few large shards of concatenated ``float16`` rows plus a JSON index
giving each chain's shard, row offset and length. One file per chain would mean
hundreds of thousands of tiny files, which this parallel filesystem handles
badly; large shards are read back through `numpy.memmap`, so a training step
touches only the rows it needs without loading a shard into memory.

fp16 is deliberate: it halves both size and read bandwidth, and the values feed
a head that trains in mixed precision anyway.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

INDEX_NAME = "index.json"
SHARD_TEMPLATE = "shard-{:04d}.f16"
DEFAULT_SHARD_ROWS = 20_000_000  # ~51 GB per shard at dim 1280


@dataclass
class StoreMeta:
    """What a cache needs to describe itself to be reusable."""

    model_name: str
    layer: int
    dim: int
    dtype: str = "float16"

    def as_dict(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "layer": self.layer,
            "dim": self.dim,
            "dtype": self.dtype,
        }


class EmbeddingWriter:
    """Append residue embeddings to sharded files, recording an index."""

    def __init__(
        self,
        root: str,
        meta: StoreMeta,
        shard_rows: int = DEFAULT_SHARD_ROWS,
    ) -> None:
        self.root = root
        self.meta = meta
        self.shard_rows = shard_rows
        os.makedirs(root, exist_ok=True)
        self._entries: Dict[str, List[int]] = {}
        self._shard_index = 0
        self._rows_in_shard = 0
        self._handle = None

    def _open_shard(self):
        if self._handle is None:
            path = os.path.join(self.root, SHARD_TEMPLATE.format(self._shard_index))
            self._handle = open(path, "ab")
        return self._handle

    def _rotate(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._shard_index += 1
        self._rows_in_shard = 0

    def add(self, key: str, array) -> None:
        """Store one chain's ``(length, dim)`` embedding under `key`."""
        import numpy as np

        values = np.asarray(array, dtype=np.float16)
        if values.ndim != 2 or values.shape[1] != self.meta.dim:
            raise ValueError(
                f"{key}: expected (length, {self.meta.dim}), got {values.shape}"
            )
        if key in self._entries:
            raise KeyError(f"{key} is already in this cache")

        if self._rows_in_shard and self._rows_in_shard + values.shape[0] > self.shard_rows:
            self._rotate()

        handle = self._open_shard()
        handle.write(values.tobytes(order="C"))
        self._entries[key] = [self._shard_index, self._rows_in_shard, values.shape[0]]
        self._rows_in_shard += values.shape[0]

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        payload = {
            "meta": self.meta.as_dict(),
            "shard_rows": self.shard_rows,
            "entries": self._entries,
        }
        with open(os.path.join(self.root, INDEX_NAME), "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def __enter__(self) -> "EmbeddingWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._entries)


class EmbeddingStore:
    """Read-only view over a cache written by `EmbeddingWriter`."""

    def __init__(self, root: str) -> None:
        self.root = root
        with open(os.path.join(root, INDEX_NAME), "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        meta = payload["meta"]
        self.meta = StoreMeta(
            model_name=meta["model_name"],
            layer=int(meta["layer"]),
            dim=int(meta["dim"]),
            dtype=meta.get("dtype", "float16"),
        )
        self._entries: Dict[str, List[int]] = payload["entries"]
        self._maps: Dict[int, object] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> Iterable[str]:
        return self._entries.keys()

    def _shard(self, index: int):
        import numpy as np

        if index not in self._maps:
            path = os.path.join(self.root, SHARD_TEMPLATE.format(index))
            # Memory-mapped so a read costs only the rows it touches.
            self._maps[index] = np.memmap(
                path, dtype=np.float16, mode="r"
            ).reshape(-1, self.meta.dim)
        return self._maps[index]

    def get(self, key: str):
        """Return one chain's ``(length, dim)`` embedding as float32."""
        import numpy as np

        if key not in self._entries:
            raise KeyError(key)
        shard_index, offset, length = self._entries[key]
        rows = self._shard(shard_index)[offset : offset + length]
        return np.asarray(rows, dtype=np.float32)

    def verify(self, config_model: str, config_layer: Optional[int] = None) -> None:
        """Fail loudly when a cache was written by a different encoder.

        Silently mixing 650M and 150M vectors, or two different layers, would
        produce a model that trains on incoherent inputs, so this is checked
        rather than assumed.
        """
        if self.meta.model_name != config_model:
            raise ValueError(
                f"cache was written with {self.meta.model_name}, "
                f"configuration asks for {config_model}"
            )
        if config_layer is not None and self.meta.layer != config_layer:
            raise ValueError(
                f"cache holds layer {self.meta.layer}, configuration asks for "
                f"{config_layer}"
            )


def estimate_bytes(lengths: Iterable[int], dim: int, dtype_bytes: int = 2) -> int:
    """Size a cache before writing it, so a run can be sanity-checked first."""
    return sum(lengths) * dim * dtype_bytes
