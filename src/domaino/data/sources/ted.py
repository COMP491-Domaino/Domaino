"""TED (The Encyclopedia of Domains) v5 to the common schema.

TED segments AlphaFold models by combining Merizo, UniDoc and Chainsaw, so its
boundaries come from predicted structures rather than experiment. The proposal
therefore treats it as lower-weight pretraining data only.

Two files matter, and which is authoritative is not obvious:

``ted_365m.domain_summary...tsv.gz``
    One row per accepted domain, with its consensus level, chopping, pLDDT and
    a predicted CATH code. **This is the accepted segmentation.**

``ted_214m_per_chain_segmentation.tsv.gz``
    One row per chain, with sequence length, sequence MD5, and choppings listed
    per consensus level. Its ``low`` column can hold mutually overlapping
    alternatives that TED did not adopt (``AF-A0A002`` lists eight overlapping
    low-consensus candidates but only one accepted domain), so this file is
    used for chain length and MD5, never for the domain set.

TED ships no sequences. Chains are matched to sequences we already hold by
sequence MD5; the rest are emitted as annotation-only records.
"""

from __future__ import annotations

import gzip
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from ..schema import CathLabel, ChainRecord, Domain, Segment, Supervision

# Confidence attached to a TED domain, by how many methods agreed.
CONSENSUS_WEIGHTS = {"high": 0.30, "medium": 0.15, "low": 0.05}

MISSING = {"na", "-", ""}


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(
        path, "rt", encoding="utf-8"
    )


def parse_chopping(text: str) -> List[Segment]:
    """TED writes a domain's segments joined by ``_``: ``11-41_290-389``."""
    return Segment.parse_many(text, segment_sep="_")


def iter_chain_index(
    path: str, keep_md5: Optional[Set[str]] = None, limit: Optional[int] = None
) -> Iterator[Tuple[str, str, int]]:
    """Stream ``(chain_id, sequence_md5, length)`` from the per-chain file.

    `keep_md5` restricts output to chains whose sequence we already have, which
    is what makes a pass over this 9 GB file worth doing.
    """
    with _open(path) as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            chain_id, md5, length = fields[0], fields[1], fields[2]
            if keep_md5 is not None and md5 not in keep_md5:
                continue
            try:
                yield chain_id, md5, int(length)
            except ValueError:
                continue


def iter_domain_rows(
    path: str, keep_chains: Optional[Set[str]] = None, limit: Optional[int] = None
) -> Iterator[Dict[str, object]]:
    """Stream accepted domains from the domain-summary file."""
    with _open(path) as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15:
                continue
            ted_id = fields[0]
            chain_id = ted_id.rsplit("_TED", 1)[0]
            if keep_chains is not None and chain_id not in keep_chains:
                continue
            cath_code = fields[13].strip()
            yield {
                "ted_id": ted_id,
                "chain_id": chain_id,
                "consensus_level": fields[2].strip(),
                "chopping": fields[3].strip(),
                "n_residues": fields[4],
                "plddt": fields[6],
                "cath_code": None if cath_code in MISSING else cath_code,
                "cath_depth": fields[14].strip(),
                "assignment_method": fields[15].strip() if len(fields) > 15 else "",
            }


def build_records(
    domain_rows: Iterable[Dict[str, object]],
    chain_lengths: Dict[str, int],
    chain_md5: Optional[Dict[str, str]] = None,
    sequences_by_md5: Optional[Dict[str, str]] = None,
    source_version: str = "v5",
) -> Iterator[ChainRecord]:
    """Group per-domain rows into one record per chain.

    Rows must arrive grouped by chain, which the TED files already are: they
    are sorted by identifier, so chains stream through without buffering the
    whole 365-million-domain table.
    """
    current: Optional[str] = None
    buffer: List[Dict[str, object]] = []

    def flush(chain_id: str, rows: List[Dict[str, object]]) -> Optional[ChainRecord]:
        if not rows:
            return None
        md5 = (chain_md5 or {}).get(chain_id)
        sequence = (sequences_by_md5 or {}).get(md5) if md5 else None
        length = chain_lengths.get(chain_id) or (len(sequence) if sequence else 0)

        domains: List[Domain] = []
        levels: List[str] = []
        flags: List[str] = []
        for row in rows:
            level = str(row["consensus_level"])
            levels.append(level)
            try:
                plddt = float(row["plddt"])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                plddt = None
            code = row["cath_code"]
            domains.append(
                Domain(
                    domain_id=str(row["ted_id"]),
                    segments=parse_chopping(str(row["chopping"])),
                    cath=CathLabel.parse(str(code)) if code else None,
                    quality={
                        "consensus_level": level,
                        "plddt": plddt,
                        "assignment_method": row["assignment_method"],
                    },
                )
            )

        if not length:
            length = max(s.end for d in domains for s in d.segments)
            flags.append("length_inferred_from_segments")

        domains.sort(key=lambda d: d.segments[0].start)
        if sequence is None:
            flags.append("sequence_unavailable")

        # One chain may mix levels; the weakest present sets the weight, and a
        # chain carrying any low-consensus domain is not trusted for pairwise
        # grouping even though TED's accepted domains do not overlap.
        weakest = min(levels, key=lambda l: CONSENSUS_WEIGHTS.get(l, 0.0))
        record = ChainRecord(
            record_id=f"ted:{chain_id}",
            source="ted",
            source_version=source_version,
            chain_id=chain_id,
            length=length,
            domains=domains,
            supervision=Supervision(
                residue_labels=True,
                pairwise=weakest != "low",
                complete_partition=weakest != "low",
                cath_labels=any(d.label_depth != "none" for d in domains),
                weight=CONSENSUS_WEIGHTS.get(weakest, 0.05),
            ),
            sequence=sequence,
            sequence_md5=md5,
            flags=flags,
            extra={"consensus_levels": sorted(set(levels))},
        )
        for problem in record.validate():
            tag = f"schema:{problem}"
            if tag not in record.flags:
                record.flags.append(tag)
        return record

    for row in domain_rows:
        chain_id = str(row["chain_id"])
        if chain_id != current:
            if current is not None:
                result = flush(current, buffer)
                if result is not None:
                    yield result
            current, buffer = chain_id, []
        buffer.append(row)

    if current is not None:
        result = flush(current, buffer)
        if result is not None:
            yield result
