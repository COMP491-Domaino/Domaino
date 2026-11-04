"""PDB SEQRES chain sequences.

CATH distributes per-domain sequences only, but Domaino is trained on whole
chains, and `cath-domain-boundaries-seqreschopping` indexes into the SEQRES
record of a PDB chain. This module provides those chain sequences.

File format, one record per two lines::

    >101m_A mol:protein length:154  MYOGLOBIN
    MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSH...

Only ``mol:protein`` records are of interest; the file also carries nucleic
acid chains.
"""

from __future__ import annotations

import gzip
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple

ChainKey = Tuple[str, str]


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(
        path, "rt", encoding="utf-8"
    )


def iter_seqres(path: str, protein_only: bool = True) -> Iterator[Tuple[ChainKey, str]]:
    """Stream ``((pdb_id, chain_id), sequence)`` pairs from pdb_seqres."""
    with _open(path) as handle:
        key: Optional[ChainKey] = None
        for line in handle:
            if line.startswith(">"):
                fields = line[1:].split()
                identifier = fields[0]
                mol = fields[1] if len(fields) > 1 else ""
                pdb_id, _, chain_id = identifier.partition("_")
                keep = (not protein_only) or mol == "mol:protein"
                key = (pdb_id.lower(), chain_id) if keep else None
            elif key is not None:
                yield key, line.strip().upper()
                key = None


def load_chain_sequences(
    path: str, wanted: Optional[Iterable[ChainKey]] = None
) -> Dict[ChainKey, str]:
    """Load chain sequences, optionally restricted to the chains we need.

    Passing `wanted` matters: the full file holds over a million protein chains
    while CATH references about half that, so filtering while streaming keeps
    the index small enough to build alongside everything else.
    """
    keep: Optional[Set[ChainKey]] = None
    if wanted is not None:
        keep = {(p.lower(), c) for p, c in wanted}

    sequences: Dict[ChainKey, str] = {}
    for key, sequence in iter_seqres(path):
        if keep is None or key in keep:
            sequences[key] = sequence
    return sequences


def chain_key(cath_chain_id: str) -> ChainKey:
    """Split a 5-character CATH chain name such as ``1oaiA`` into its parts.

    A chain character of ``0`` means the original PDB file had no chain field.
    It is kept verbatim rather than guessed at, so that unmatched chains show
    up in the audit instead of being silently paired with the wrong sequence.
    """
    return (cath_chain_id[:4].lower(), cath_chain_id[4:5])
