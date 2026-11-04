"""Common dataset schema shared by every Domaino data source.

One record describes one annotated protein chain as a single source sees it:
the chain sequence, the domains carved out of it, and what that source is
allowed to supervise. CATH, Pfam and TED do not define domains the same way
(proposal risk 1), so a record never loses track of where it came from, and
`Supervision` states exactly which training targets it may contribute to.

Coordinate convention
---------------------
Every segment is a 1-based inclusive interval indexing the record's own
`sequence`. All three sources already number residues this way -- CATH
seqreschopping into the PDB SEQRES record, Pfam into the UniProt sequence, TED
into the AlphaFold model sequence -- so the pipeline validates bounds rather
than shifting coordinates, and `validate()` rejects anything out of range.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "1.0"

# The twenty standard amino acids. Everything else is recorded as a flag rather
# than silently dropped, because PDB SEQRES legitimately contains X for
# modified or unknown residues.
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AA = frozenset("BJOUXZ")

# Depth of a CATH label, i.e. the most specific level that is actually known.
LABEL_DEPTHS = ("none", "C", "A", "T", "H")


@dataclass(frozen=True)
class Segment:
    """One continuous residue interval, 1-based and inclusive."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise ValueError(f"invalid segment: {self.start}-{self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: "Segment") -> bool:
        return self.start <= other.end and other.start <= self.end

    def as_tuple(self) -> Tuple[int, int]:
        return (self.start, self.end)

    @classmethod
    def parse_many(cls, text: str, segment_sep: str = ",") -> List["Segment"]:
        """Parse a chopping string such as ``2-78,187-208`` into segments.

        CATH separates the segments of one domain with ``,`` while TED uses
        ``_``; both write a segment as ``start-end``.
        """
        segments = []
        for piece in text.split(segment_sep):
            piece = piece.strip()
            if not piece:
                continue
            start, _, end = piece.partition("-")
            segments.append(cls(int(start), int(end)))
        return segments


@dataclass(frozen=True)
class CathLabel:
    """A CATH classification, truncated at whatever level is known."""

    c: Optional[int] = None
    a: Optional[int] = None
    t: Optional[int] = None
    h: Optional[int] = None

    @property
    def depth(self) -> str:
        for level, value in zip(("H", "T", "A", "C"), (self.h, self.t, self.a, self.c)):
            if value is not None:
                return level
        return "none"

    def code(self, depth: Optional[str] = None) -> Optional[str]:
        """Dotted code, e.g. ``3.40.640.10``, truncated to `depth`."""
        wanted = depth or self.depth
        if wanted == "none":
            return None
        levels = ("C", "A", "T", "H")
        values = (self.c, self.a, self.t, self.h)
        parts = []
        for level, value in zip(levels, values):
            if value is None:
                break
            parts.append(str(value))
            if level == wanted:
                break
        return ".".join(parts) if parts else None

    @classmethod
    def parse(cls, code: str) -> "CathLabel":
        """Parse a dotted code; missing trailing levels stay None."""
        parts = [p for p in str(code).strip().split(".") if p != ""]
        values: List[Optional[int]] = [None, None, None, None]
        for index, part in enumerate(parts[:4]):
            values[index] = int(part)
        return cls(*values)


@dataclass(frozen=True)
class Domain:
    """One domain: its residue intervals plus whatever labels the source has."""

    domain_id: str
    segments: List[Segment]
    cath: Optional[CathLabel] = None
    # Sequence-identity cluster ids (CATH S35/S60/S95/S100). The S35 cluster is
    # what the homology-aware split groups chains by.
    clusters: Dict[str, int] = field(default_factory=dict)
    # Source-specific quality signals, e.g. TED consensus level and pLDDT.
    quality: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def is_discontinuous(self) -> bool:
        return len(self.segments) > 1

    @property
    def length(self) -> int:
        return sum(s.length for s in self.segments)

    @property
    def label_depth(self) -> str:
        return self.cath.depth if self.cath else "none"

    @property
    def s35_key(self) -> Optional[str]:
        """``C.A.T.H.S35``, the homology group key used for splitting."""
        if not self.cath or "s35" not in self.clusters:
            return None
        code = self.cath.code("H")
        return f"{code}.{self.clusters['s35']}" if code else None


@dataclass
class Supervision:
    """Which training targets a record is allowed to contribute to.

    This is the concrete form of the proposal's source-specific loss masks. The
    important field is `complete_partition`: CATH chops a whole chain, so a
    residue outside every domain really is linker, whereas Pfam only annotates
    the regions it recognises, so an unannotated residue there is unknown and
    must never be used as a negative example.
    """

    residue_labels: bool
    pairwise: bool
    cath_labels: bool
    complete_partition: bool
    weight: float

    def __post_init__(self) -> None:
        if self.pairwise and not self.complete_partition:
            raise ValueError(
                "pairwise supervision needs a complete partition: without it, two "
                "residues in different annotated regions cannot be called a "
                "negative pair"
            )
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight out of range: {self.weight}")


@dataclass
class ChainRecord:
    """One protein chain as annotated by one source."""

    record_id: str
    source: str
    source_version: str
    chain_id: str
    length: int
    domains: List[Domain]
    supervision: Supervision
    sequence: Optional[str] = None
    sequence_md5: Optional[str] = None
    # Audit findings. A record with flags is still written out; the split stage
    # decides what is usable, so nothing is discarded invisibly.
    flags: List[str] = field(default_factory=list)
    # Filled in by the split stage.
    split: Optional[str] = None
    homology_group: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_domains(self) -> int:
        return len(self.domains)

    @property
    def is_multidomain(self) -> bool:
        return len(self.domains) > 1

    @property
    def has_discontinuous_domain(self) -> bool:
        return any(d.is_discontinuous for d in self.domains)

    @property
    def covered_residues(self) -> int:
        return sum(s.length for d in self.domains for s in d.segments)

    @property
    def coverage(self) -> float:
        """Fraction of the chain that sits inside some domain."""
        return self.covered_residues / self.length if self.length else 0.0

    def s35_keys(self) -> List[str]:
        keys = {d.s35_key for d in self.domains}
        keys.discard(None)
        return sorted(keys)  # type: ignore[arg-type]

    def residue_labels(self) -> List[int]:
        """Per-residue domain index, 1-based; 0 means no domain covers it.

        Only meaningful when `supervision.complete_partition` holds, where 0 is
        a genuine linker rather than an unannotated residue.
        """
        labels = [0] * self.length
        for index, domain in enumerate(self.domains, start=1):
            for segment in domain.segments:
                for position in range(segment.start, segment.end + 1):
                    if 1 <= position <= self.length:
                        labels[position - 1] = index
        return labels

    def overlapping_domains(self) -> List[Tuple[str, str]]:
        """Pairs of domains that claim the same residue; should be empty."""
        clashes = []
        spans = [(d.domain_id, s) for d in self.domains for s in d.segments]
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                (id_a, seg_a), (id_b, seg_b) = spans[i], spans[j]
                if id_a != id_b and seg_a.overlaps(seg_b):
                    clashes.append(tuple(sorted((id_a, id_b))))
        return sorted(set(clashes))

    def validate(self) -> List[str]:
        """Return a list of schema problems; empty means the record is clean."""
        problems = []
        if self.length <= 0:
            problems.append("non_positive_length")
        if self.sequence is not None and len(self.sequence) != self.length:
            problems.append("length_sequence_mismatch")
        if not self.domains:
            problems.append("no_domains")

        for domain in self.domains:
            if not domain.segments:
                problems.append(f"empty_segments:{domain.domain_id}")
            for segment in domain.segments:
                if segment.end > self.length:
                    problems.append(f"segment_out_of_bounds:{domain.domain_id}")
                    break

        if self.overlapping_domains():
            problems.append("overlapping_domains")

        if self.sequence is not None:
            unknown = set(self.sequence) - STANDARD_AA
            if unknown & AMBIGUOUS_AA:
                problems.append("ambiguous_residues")
            if unknown - AMBIGUOUS_AA:
                problems.append("non_standard_residues")

        if self.supervision.cath_labels and not any(
            d.label_depth != "none" for d in self.domains
        ):
            problems.append("cath_supervision_without_labels")

        return problems

    def domain_sequence(self, domain: Domain) -> Optional[str]:
        """Concatenate the chain sequence over a domain's segments."""
        if self.sequence is None:
            return None
        return "".join(
            self.sequence[s.start - 1 : s.end] for s in domain.segments
        )

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        # Segments round-trip as [start, end] pairs rather than objects, which
        # keeps the JSONL small enough to read by eye.
        for domain, source in zip(payload["domains"], self.domains):
            domain["segments"] = [s.as_tuple() for s in source.segments]
            domain["cath_code"] = source.cath.code() if source.cath else None
            domain["label_depth"] = source.label_depth
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChainRecord":
        data = dict(payload)
        data.pop("schema_version", None)
        domains = []
        for raw in data.pop("domains", []):
            raw = dict(raw)
            raw.pop("cath_code", None)
            raw.pop("label_depth", None)
            cath = raw.pop("cath", None)
            domains.append(
                Domain(
                    domain_id=raw["domain_id"],
                    segments=[Segment(int(a), int(b)) for a, b in raw["segments"]],
                    cath=CathLabel(**cath) if cath else None,
                    clusters=raw.get("clusters", {}) or {},
                    quality=raw.get("quality", {}) or {},
                )
            )
        supervision = Supervision(**data.pop("supervision"))
        return cls(domains=domains, supervision=supervision, **data)


def sequence_md5(sequence: str) -> str:
    """MD5 of an uppercase sequence.

    This is the join key across sources: Pfam publishes the MD5 of each UniProt
    sequence and TED the MD5 of each AlphaFold model sequence, so hashing PDB
    SEQRES the same way lets the audit find chains that several sources
    describe and compare their boundaries directly.
    """
    return hashlib.md5(sequence.strip().upper().encode("ascii")).hexdigest()


def write_jsonl(path: str, records: Iterable[ChainRecord]) -> int:
    """Write records as JSON lines, returning how many were written."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    count = 0
    with opener(path, "wt", encoding="utf-8") as handle:  # type: ignore[operator]
        for record in records:
            handle.write(json.dumps(record.to_dict(), separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str) -> Iterator[ChainRecord]:
    """Stream records back from a (optionally gzipped) JSONL file."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if line:
                yield ChainRecord.from_dict(json.loads(line))


def iou(left: Sequence[Tuple[int, int]], right: Sequence[Tuple[int, int]]) -> float:
    """Intersection over union of two sets of residue intervals.

    Used by the annotation-conflict report and, later, by the CHAINSAW-style
    segmentation metrics, so both read the same definition.
    """
    left_set = {p for a, b in left for p in range(a, b + 1)}
    right_set = {p for a, b in right for p in range(a, b + 1)}
    if not left_set and not right_set:
        return 1.0
    union = len(left_set | right_set)
    return len(left_set & right_set) / union if union else 0.0
