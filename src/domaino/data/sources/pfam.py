"""Pfam 38.2 regions to the common schema.

Pfam marks sequence regions belonging to conserved families. A region may be a
whole structural domain, part of one, or occasionally span more than one, and
Pfam carries no CATH labels, so the proposal uses it strictly as auxiliary
boundary evidence. Two consequences are encoded in `Supervision`:

* `complete_partition` is False. Pfam annotates only what it recognises, so a
  residue outside every region is unknown, not linker. Treating it as a
  negative example is exactly the source confusion risk 1 warns about.
* `pairwise` is therefore False as well: two residues in different Pfam
  regions cannot be called a negative pair.

Pfam regions are defined on UniProt sequences, which the project does not
download (``Pfam-A.regions.tsv.gz`` alone is ~149 million rows over ~12.8 GB
uncompressed, and the sequences themselves are far larger). Pfam publishes the
MD5 of each sequence, so regions are attached to chains we already have by
matching that MD5 against the MD5 of a PDB SEQRES chain. Regions that match
become usable auxiliary supervision; the rest are summarised statistically.
"""

from __future__ import annotations

import gzip
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from ..schema import ChainRecord, Domain, Segment, Supervision

EXPECTED_HEADER = (
    "pfamseq_acc",
    "seq_version",
    "crc64",
    "md5",
    "pfamA_acc",
    "seq_start",
    "seq_end",
    "ali_start",
    "ali_end",
)


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(
        path, "rt", encoding="utf-8"
    )


def parse_clans(path: str) -> Dict[str, Dict[str, str]]:
    """Parse Pfam-A.clans.tsv: family accession to clan and description."""
    families: Dict[str, Dict[str, str]] = {}
    with _open(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            families[fields[0]] = {
                "clan_acc": fields[1],
                "clan_id": fields[2],
                "family_id": fields[3],
                "description": fields[4],
            }
    return families


def iter_regions(
    path: str,
    keep_md5: Optional[Set[str]] = None,
    limit: Optional[int] = None,
) -> Iterator[Dict[str, str]]:
    """Stream region rows, optionally keeping only sequences we can use.

    The file is ordered by family rather than by sequence, so callers that need
    regions grouped per sequence must accumulate them. That is affordable only
    because `keep_md5` reduces the output to a small subset.
    """
    with _open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if tuple(header[: len(EXPECTED_HEADER)]) != EXPECTED_HEADER:
            raise ValueError(f"unexpected Pfam regions header: {header}")
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < len(EXPECTED_HEADER):
                continue
            if keep_md5 is not None and fields[3] not in keep_md5:
                continue
            yield dict(zip(EXPECTED_HEADER, fields))


def group_by_sequence(
    rows: Iterable[Dict[str, str]]
) -> Dict[str, List[Dict[str, str]]]:
    """Collect region rows into ``{sequence_md5: [row, ...]}``."""
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["md5"], []).append(row)
    return grouped


def build_records(
    grouped: Dict[str, List[Dict[str, str]]],
    sequences_by_md5: Dict[str, str],
    families: Optional[Dict[str, Dict[str, str]]] = None,
    source_version: str = "38.2",
    weight: float = 0.3,
) -> Iterator[ChainRecord]:
    """Yield one record per sequence whose MD5 we could resolve."""
    for md5, rows in sorted(grouped.items()):
        sequence = sequences_by_md5.get(md5)
        if sequence is None:
            continue
        accession = rows[0]["pfamseq_acc"]
        flags: List[str] = []

        regions: List[Domain] = []
        for row in rows:
            start, end = int(row["seq_start"]), int(row["seq_end"])
            if end > len(sequence) or start < 1 or start > end:
                flags.append(f"region_out_of_bounds:{row['pfamA_acc']}")
                continue
            meta = (families or {}).get(row["pfamA_acc"], {})
            regions.append(
                Domain(
                    domain_id=f"{accession}/{row['pfamA_acc']}/{start}-{end}",
                    segments=[Segment(start, end)],
                    cath=None,
                    quality={
                        "pfam_family": row["pfamA_acc"],
                        "clan": meta.get("clan_acc") or None,
                        "family_id": meta.get("family_id") or None,
                        "alignment": [int(row["ali_start"]), int(row["ali_end"])],
                    },
                )
            )

        if not regions:
            continue
        regions.sort(key=lambda d: d.segments[0].start)

        record = ChainRecord(
            record_id=f"pfam:{accession}",
            source="pfam",
            source_version=source_version,
            chain_id=accession,
            length=len(sequence),
            domains=regions,
            supervision=Supervision(
                residue_labels=True,
                pairwise=False,
                complete_partition=False,
                cath_labels=False,
                weight=weight,
            ),
            sequence=sequence,
            sequence_md5=md5,
            flags=flags,
        )
        for problem in record.validate():
            # Pfam families legitimately overlap on a sequence, so that is
            # recorded rather than treated as corruption.
            tag = f"schema:{problem}"
            if tag not in record.flags:
                record.flags.append(tag)
        yield record


def scan_and_select(
    path: str,
    keep_md5: Set[str],
    cache_path: str,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    """Stream the regions file once: cache usable rows, return statistics.

    Statistics cover every row read, not just the kept ones. Doing both in one
    pass matters for correctness as well as cost: the file is ordered by family,
    so statistics taken from a prefix of rows would describe a handful of
    families rather than Pfam as a whole.
    """
    import statistics as stats_mod

    families: Set[str] = set()
    sequences: Set[str] = set()
    lengths: List[int] = []
    rows = 0
    kept = 0

    with gzip.open(cache_path, "wt", encoding="utf-8") as cache:
        cache.write("\t".join(EXPECTED_HEADER) + "\n")
        for row in iter_regions(path, limit=limit):
            rows += 1
            families.add(row["pfamA_acc"])
            sequences.add(row["md5"])
            lengths.append(int(row["seq_end"]) - int(row["seq_start"]) + 1)
            if row["md5"] in keep_md5:
                cache.write("\t".join(row[field] for field in EXPECTED_HEADER) + "\n")
                kept += 1

    lengths.sort()
    return {
        "rows_scanned": rows,
        "truncated": limit is not None and rows >= limit,
        "rows_kept_on_held_sequences": kept,
        "distinct_families": len(families),
        "distinct_sequences": len(sequences),
        "regions_per_sequence": round(rows / len(sequences), 3) if sequences else 0.0,
        "region_length": {
            "min": lengths[0] if lengths else 0,
            "median": int(stats_mod.median(lengths)) if lengths else 0,
            "mean": round(stats_mod.fmean(lengths), 1) if lengths else 0.0,
            "p95": lengths[int(0.95 * (len(lengths) - 1))] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
    }
