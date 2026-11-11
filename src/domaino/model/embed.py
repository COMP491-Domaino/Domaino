"""Precompute ESM-2 residue embeddings for a dataset split (task T2.2).

    python -m domaino.model.embed --config configs/representation_v1.json \
        --split train --limit 4000

Writes a cache under ``data/processed/<dataset>/embeddings/<model>-<split>/``
that `EmbeddingStore` reads back. Caching is optional: a training run can embed
on the fly instead, which is slower per step but needs no disk. It pays off
because the encoder is frozen, so a chain's vectors never change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..data.schema import ChainRecord, read_jsonl
from .embedding_store import EmbeddingStore, EmbeddingWriter, StoreMeta, estimate_bytes
from .representation import EncoderConfig, Esm2Encoder


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_config(path: str) -> Dict[str, object]:
    with open(path, "rt", encoding="utf-8") as handle:
        config = json.load(handle)
    root = str(config.get("repo_root") or os.getcwd())

    def resolve(value):
        if isinstance(value, str) and value.startswith("@/"):
            return os.path.join(root, value[2:])
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return {key: resolve(value) for key, value in config.items()}


def split_members(manifest_dir: str, split: str) -> List[str]:
    path = os.path.join(manifest_dir, "splits", f"{split}.txt")
    with open(path, "rt", encoding="utf-8") as handle:
        return handle.read().split()


def iter_split_records(
    records_path: str, wanted: Sequence[str], limit: Optional[int] = None
) -> Iterator[ChainRecord]:
    """Stream records belonging to a split, longest-first for early failure.

    Ordering by length means an out-of-memory failure happens on the first
    batch rather than hours in.
    """
    keep = set(wanted)
    selected = [
        record
        for record in read_jsonl(records_path)
        if record.chain_id in keep and record.sequence
    ]
    selected.sort(key=lambda r: -r.length)
    if limit:
        # Keep the length spread rather than only the longest chains.
        step = max(1, len(selected) // limit)
        selected = selected[::step][:limit]
    return iter(selected)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="report how much the cache would cost, then stop")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    encoder_config = EncoderConfig(**dict(config["encoder"]))  # type: ignore[arg-type]
    records_path = str(config["records"])
    manifest_dir = str(config["manifest_dir"])

    out_dir = os.path.join(
        str(config["embeddings_root"]),
        f"{encoder_config.model_name}-{args.split}",
    )
    if os.path.exists(os.path.join(out_dir, "index.json")) and not args.overwrite:
        store = EmbeddingStore(out_dir)
        log(f"cache already holds {len(store)} chains at {out_dir}; use --overwrite")
        return 0

    members = split_members(manifest_dir, args.split)
    log(f"{args.split}: {len(members)} chains in the split")
    records = list(iter_split_records(records_path, members, args.limit))
    if not records:
        log("no records selected; nothing to do")
        return 1
    lengths = [r.length for r in records]
    size = estimate_bytes(lengths, encoder_config.embed_dim)
    log(
        f"selected {len(records)} chains, {sum(lengths)} residues, "
        f"longest {max(lengths)}; cache will be about {size/1e9:.1f} GB"
    )
    if args.dry_run:
        return 0

    encoder = Esm2Encoder(encoder_config)
    meta = StoreMeta(
        model_name=encoder_config.model_name,
        layer=encoder.layer,
        dim=encoder_config.embed_dim,
    )
    log(f"encoding on {encoder.device} from layer {meta.layer}")

    done = 0
    residues = 0
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with EmbeddingWriter(out_dir, meta) as writer:
        # Chunk the records so one embed() call holds a bounded number of
        # sequences in memory at once.
        chunk: List[ChainRecord] = []
        chunk_residues = 0
        budget = int(config.get("residues_per_chunk", 60000))
        for record in records:
            chunk.append(record)
            chunk_residues += record.length
            if chunk_residues >= budget:
                _flush(writer, encoder, chunk)
                done += len(chunk)
                residues += chunk_residues
                rate = residues / max(time.time() - started, 1e-6)
                log(f"  {done}/{len(records)} chains ({rate:.0f} residues/s)")
                chunk, chunk_residues = [], 0
        if chunk:
            _flush(writer, encoder, chunk)
            done += len(chunk)

    elapsed = time.time() - started
    log(f"wrote {done} chains to {out_dir} in {elapsed/60:.1f} min")
    return 0


def _flush(writer: EmbeddingWriter, encoder: Esm2Encoder, chunk: List[ChainRecord]) -> None:
    embeddings = encoder.embed([r.sequence for r in chunk])  # type: ignore[misc]
    for record, embedding in zip(chunk, embeddings):
        writer.add(record.chain_id, embedding.numpy())


if __name__ == "__main__":
    sys.exit(main())
