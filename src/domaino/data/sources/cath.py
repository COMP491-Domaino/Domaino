"""CATH 4.4.0 to the common schema.

CATH is Domaino's primary ground truth: it carries experimentally derived
domain boundaries and the full C/A/T/H hierarchy.

Canonical coordinate decision
-----------------------------
Three CATH files state a domain's extent and they do not agree:

* ``cath-domain-boundaries-seqreschopping`` gives residue intervals as 1-based
  indices into the chain's PDB SEQRES record,
* ``cath-domain-seqs`` gives a domain sequence,
* ``cath-domain-list`` column 11 gives a domain length.

Measured on CATH 4.4.0, the chopping disagrees with the domain-sequence length
for about 39% of domains and with the domain-list length for about 25%. The
disagreements are almost entirely small offsets at domain termini, plus
whole-chain (``00``) domains whose published sequence is the entire SEQRES
rather than the chopped region.

This pipeline therefore treats **SEQRES plus the chopping intervals as
canonical**, because that is the only pairing that yields a whole-chain
sequence with coordinates defined on it, and records the comparison against
the domain sequences as an audit signal (`agreement` in each domain's
`quality`). Records whose disagreement is not explained by terminal offsets
are flagged and excluded from training by default.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from ..schema import (
    CathLabel,
    ChainRecord,
    Domain,
    Segment,
    Supervision,
    sequence_md5,
)
from .pdb_seqres import ChainKey, chain_key

# Beyond this many residues of difference at a domain terminus we stop calling
# it an offset between CATH's own files and treat the record as suspect.
MAX_TERMINAL_OFFSET = 5


def parse_ranges(ranges: str) -> Tuple[List[Segment], List[str]]:
    """Parse one chopping string, separating valid segments from broken ones.

    CATH 4.4.0 contains a small number of genuinely inverted intervals, such as
    ``567-1`` for ``1ohfB01`` (31 of 857,827 segments). Those are upstream data
    errors, so they are collected and reported rather than allowed to abort the
    build or, worse, silently reinterpreted.
    """
    segments: List[Segment] = []
    malformed: List[str] = []
    for piece in ranges.split(","):
        piece = piece.strip()
        if not piece:
            continue
        start, _, end = piece.partition("-")
        try:
            segments.append(Segment(int(start), int(end)))
        except ValueError:
            malformed.append(piece)
    return segments, malformed


def parse_chopping_file(
    path: str,
) -> Tuple[Dict[str, List[Segment]], Dict[str, List[str]]]:
    """Parse seqreschopping: ``1oaiA00\t1-59`` or ``10gsA01\t2-78,187-208``.

    Returns the usable segments per domain and, separately, any malformed
    intervals found so the audit can account for them.
    """
    choppings: Dict[str, List[Segment]] = {}
    malformed: Dict[str, List[str]] = {}
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            domain_id, _, ranges = line.rstrip("\n").partition("\t")
            segments, broken = parse_ranges(ranges)
            choppings[domain_id] = segments
            if broken:
                malformed[domain_id] = broken
    return choppings, malformed


def parse_domain_list(path: str) -> Dict[str, Dict[str, object]]:
    """Parse CathDomainList (CLF 2.0): labels, sequence clusters, length."""
    entries: Dict[str, Dict[str, object]] = {}
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            entries[parts[0]] = {
                "cath": CathLabel(
                    int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                ),
                "clusters": {
                    "s35": int(parts[5]),
                    "s60": int(parts[6]),
                    "s95": int(parts[7]),
                    "s100": int(parts[8]),
                },
                "domain_length": int(parts[10]),
                "resolution": float(parts[11]),
            }
    return entries


def parse_domain_sequences(path: str) -> Dict[str, str]:
    """Parse ``cath-domain-seqs``, keyed by domain id.

    Headers look like ``>cath|4_4_0|10gsA01/2-78_187-208``. The range in the
    header is PDB author numbering and is deliberately ignored: it carries
    insertion codes and does not index the sequence.
    """
    sequences: Dict[str, str] = {}
    domain_id: Optional[str] = None
    chunks: List[str] = []
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if domain_id is not None:
                    sequences[domain_id] = "".join(chunks)
                fields = line[1:].strip().split("|")
                domain_id = fields[2].split("/")[0] if len(fields) > 2 else None
                chunks = []
            elif domain_id is not None:
                chunks.append(line.strip().upper())
    if domain_id is not None:
        sequences[domain_id] = "".join(chunks)
    return sequences


def parse_names(path: str) -> Dict[str, str]:
    """Parse CathNames: node code to human-readable description."""
    names: Dict[str, str] = {}
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            code, _, rest = line.partition(" ")
            _, _, description = rest.partition(":")
            names[code.strip()] = description.strip()
    return names


def _is_subsequence(small: str, big: str) -> bool:
    iterator = iter(big)
    return all(character in iterator for character in small)


def _classify_agreement(
    observed: str, published: Optional[str], full_chain: str, is_whole_chain: bool
) -> str:
    """Compare the chopped SEQRES against CATH's published domain sequence."""
    if published is None:
        return "no_published_sequence"
    if observed == published:
        return "exact"
    if is_whole_chain and published == full_chain:
        # Whole-chain domains publish the entire SEQRES, while the chopping
        # covers only the residues CATH actually assigned.
        return "published_is_full_chain"
    difference = abs(len(observed) - len(published))
    if difference <= MAX_TERMINAL_OFFSET and (
        _is_subsequence(observed, published) or _is_subsequence(published, observed)
    ):
        return "terminal_offset"
    if _is_subsequence(observed, published) or _is_subsequence(published, observed):
        return "residues_missing_one_side"
    return "mismatch"


def load_records(
    chopping_path: str,
    domain_list_path: str,
    chain_sequences: Dict[ChainKey, str],
    domain_sequences: Optional[Dict[str, str]] = None,
    source_version: str = "4.4.0",
    weight: float = 1.0,
) -> Iterator[ChainRecord]:
    """Yield one `ChainRecord` per CATH chain.

    Every chopping entry is used, including domains CATH leaves unclassified:
    their boundaries still supervise segmentation, and because CATH chops a
    whole chain, including them is what makes the leftover residues genuine
    linker rather than merely unannotated.
    """
    choppings, malformed = parse_chopping_file(chopping_path)
    classified = parse_domain_list(domain_list_path)

    by_chain: Dict[str, List[str]] = {}
    for domain_id in choppings:
        by_chain.setdefault(domain_id[:5], []).append(domain_id)

    for cath_chain_id in sorted(by_chain):
        domain_ids = sorted(by_chain[cath_chain_id])
        sequence = chain_sequences.get(chain_key(cath_chain_id))
        flags: List[str] = []

        if sequence is None:
            # CATH 4.4.0 references PDB entries that have since been obsoleted
            # and dropped from pdb_seqres. The annotations are kept so the
            # audit can count them, but without a sequence they cannot train.
            flags.append("chain_sequence_unavailable")
            ends = [
                segment.end for d in domain_ids for segment in choppings[d]
            ]
            length = max(ends) if ends else 0

        else:
            length = len(sequence)

        domains: List[Domain] = []
        for domain_id in domain_ids:
            segments = choppings[domain_id]
            entry = classified.get(domain_id)

            if domain_id in malformed:
                flags.append(f"malformed_chopping:{domain_id}")
            if not segments:
                # Every interval for this domain was unusable, so there is
                # nothing to place on the sequence.
                continue

            quality: Dict[str, object] = {}
            if domain_id in malformed:
                quality["malformed_intervals"] = malformed[domain_id]
            if entry is not None:
                quality["resolution"] = entry["resolution"]
                quality["published_domain_length"] = entry["domain_length"]

            if sequence is not None:
                if any(segment.end > length for segment in segments):
                    flags.append(f"segment_out_of_bounds:{domain_id}")
                    quality["agreement"] = "out_of_bounds"
                else:
                    observed = "".join(
                        sequence[s.start - 1 : s.end] for s in segments
                    )
                    quality["agreement"] = _classify_agreement(
                        observed,
                        (domain_sequences or {}).get(domain_id),
                        sequence,
                        is_whole_chain=domain_id[5:7] == "00",
                    )

            domains.append(
                Domain(
                    domain_id=domain_id,
                    segments=segments,
                    cath=entry["cath"] if entry else None,  # type: ignore[index]
                    clusters=dict(entry["clusters"]) if entry else {},  # type: ignore[arg-type]
                    quality=quality,
                )
            )

        domains.sort(key=lambda d: d.segments[0].start)

        if any(d.quality.get("agreement") == "mismatch" for d in domains):
            flags.append("domain_sequence_mismatch")
        if not any(d.label_depth != "none" for d in domains):
            flags.append("no_classified_domain")

        has_labels = any(d.label_depth != "none" for d in domains)
        record = ChainRecord(
            record_id=f"cath:{cath_chain_id}",
            source="cath",
            source_version=source_version,
            chain_id=cath_chain_id,
            length=length,
            domains=domains,
            supervision=Supervision(
                residue_labels=True,
                pairwise=True,
                # CATH chops the entire chain, so residues outside every
                # domain are linker or excluded fragments, not unknowns.
                complete_partition=True,
                cath_labels=has_labels,
                weight=weight,
            ),
            sequence=sequence,
            sequence_md5=sequence_md5(sequence) if sequence else None,
            flags=flags,
        )

        problems = record.validate()
        for problem in problems:
            if problem not in record.flags:
                record.flags.append(f"schema:{problem}")
        yield record


def wanted_chain_keys(chopping_path: str) -> List[ChainKey]:
    """Chain keys CATH references, for filtering the SEQRES index."""
    keys = set()
    with open(chopping_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            keys.add(chain_key(line[:5]))
    return sorted(keys)
