"""Dataset audit and annotation-conflict measurement (task T1.4).

Produces the numbers deliverable D1.2 asks for: summary statistics, an
annotation-conflict report between sources, and the evidence behind every
record the pipeline refuses to train on. Nothing here modifies records; it
only measures them, so the audit can be rerun on a frozen dataset.
"""

from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .schema import ChainRecord, iou

# A record carrying any of these flags is not used for training. Everything
# else, including CATH's small terminal-offset disagreements, stays usable.
BLOCKING_FLAGS = frozenset(
    {
        "chain_sequence_unavailable",
        "sequence_unavailable",
        "domain_sequence_mismatch",
        "malformed_chopping",
        "segment_out_of_bounds",
        "region_out_of_bounds",
        "schema:length_sequence_mismatch",
        "schema:non_positive_length",
        "schema:no_domains",
        "schema:overlapping_domains",
        "schema:segment_out_of_bounds",
    }
)


def flag_kind(flag: str) -> str:
    """Drop the per-domain suffix from a flag, keeping any ``schema:`` prefix.

    Flags are emitted either bare (``domain_sequence_mismatch``), with the
    offending domain appended (``segment_out_of_bounds:1abcA01``), or prefixed
    by ``schema:`` when they come from `ChainRecord.validate`.
    """
    if flag.startswith("schema:"):
        return "schema:" + flag[len("schema:") :].split(":")[0]
    return flag.split(":")[0]


def is_usable(record: ChainRecord) -> bool:
    """Whether a record may contribute to training, labels included."""
    return not any(flag_kind(flag) in BLOCKING_FLAGS for flag in record.flags)


# A chain can be a valid *input* even when its labels are not trustworthy. The
# published benchmarks bring their own ground truth, so evaluating on them needs
# only a sound sequence, not agreement with CATH's published domain sequences.
SEQUENCE_BLOCKING_FLAGS = frozenset(
    {
        "chain_sequence_unavailable",
        "sequence_unavailable",
        "malformed_chopping",
        "segment_out_of_bounds",
        "schema:segment_out_of_bounds",
        "schema:length_sequence_mismatch",
        "schema:non_positive_length",
    }
)


def has_valid_sequence(record: ChainRecord) -> bool:
    """Whether the record's sequence can be fed to the model.

    Keeping this separate from `is_usable` matters: 85 of the 1,363 Chainsaw
    CATH1363 test chains fail only the CATH domain-sequence agreement check.
    Dropping them would shrink the benchmark for a reason that does not apply
    to it, since Chainsaw publishes its own ground truth.
    """
    if record.sequence is None:
        return False
    return not any(
        flag_kind(flag) in SEQUENCE_BLOCKING_FLAGS for flag in record.flags
    )


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return round(float(ordered[int(fraction * (len(ordered) - 1))]), 3)
    return {
        "min": round(float(ordered[0]), 3),
        "p25": at(0.25),
        "median": at(0.5),
        "mean": round(statistics.fmean(ordered), 3),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": round(float(ordered[-1]), 3),
    }


@dataclass
class SourceAudit:
    """Statistics for one source."""

    source: str
    records: int = 0
    usable: int = 0
    domains: int = 0
    flags: Dict[str, int] = field(default_factory=dict)
    agreement: Dict[str, int] = field(default_factory=dict)
    label_depth: Dict[str, int] = field(default_factory=dict)
    domains_per_chain: Dict[str, int] = field(default_factory=dict)
    segments_per_domain: Dict[str, int] = field(default_factory=dict)
    multidomain_records: int = 0
    discontinuous_records: int = 0
    discontinuous_domains: int = 0
    chain_length: Dict[str, float] = field(default_factory=dict)
    domain_length: Dict[str, float] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)
    duplicate_sequences: int = 0
    distinct_sequences: int = 0

    def as_dict(self) -> Dict[str, object]:
        payload = dict(self.__dict__)
        return payload


def audit_records(records: Iterable[ChainRecord]) -> Tuple[
    Dict[str, SourceAudit], Dict[str, Set[str]]
]:
    """Audit a stream of records, grouped by source.

    Also returns, per source, the set of record ids that are not usable, which
    the split stage needs in order to exclude them.
    """
    audits: Dict[str, SourceAudit] = {}
    unusable: Dict[str, Set[str]] = {}
    lengths: Dict[str, List[int]] = collections.defaultdict(list)
    domain_lengths: Dict[str, List[int]] = collections.defaultdict(list)
    coverages: Dict[str, List[float]] = collections.defaultdict(list)
    md5_counts: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )

    for record in records:
        audit = audits.setdefault(record.source, SourceAudit(source=record.source))
        audit.records += 1
        audit.domains += record.n_domains

        for flag in record.flags:
            key = flag_kind(flag)
            audit.flags[key] = audit.flags.get(key, 0) + 1

        if is_usable(record):
            audit.usable += 1
        else:
            unusable.setdefault(record.source, set()).add(record.chain_id)

        if record.is_multidomain:
            audit.multidomain_records += 1
        if record.has_discontinuous_domain:
            audit.discontinuous_records += 1

        bucket = str(record.n_domains) if record.n_domains < 10 else "10+"
        audit.domains_per_chain[bucket] = audit.domains_per_chain.get(bucket, 0) + 1

        for domain in record.domains:
            segments = str(domain.n_segments) if domain.n_segments < 6 else "6+"
            audit.segments_per_domain[segments] = (
                audit.segments_per_domain.get(segments, 0) + 1
            )
            if domain.is_discontinuous:
                audit.discontinuous_domains += 1
            audit.label_depth[domain.label_depth] = (
                audit.label_depth.get(domain.label_depth, 0) + 1
            )
            agreement = domain.quality.get("agreement")
            if agreement:
                audit.agreement[str(agreement)] = (
                    audit.agreement.get(str(agreement), 0) + 1
                )
            domain_lengths[record.source].append(domain.length)

        lengths[record.source].append(record.length)
        if record.supervision.complete_partition:
            coverages[record.source].append(record.coverage)
        if record.sequence_md5:
            md5_counts[record.source][record.sequence_md5] += 1

    for source, audit in audits.items():
        audit.chain_length = _percentiles(lengths[source])
        audit.domain_length = _percentiles(domain_lengths[source])
        audit.coverage = _percentiles(coverages[source])
        counter = md5_counts[source]
        audit.distinct_sequences = len(counter)
        audit.duplicate_sequences = sum(n - 1 for n in counter.values() if n > 1)

    return audits, unusable


def label_space(records: Iterable[ChainRecord]) -> Dict[str, object]:
    """Size and balance of the CATH label space (proposal risk 3).

    Rare homologous superfamilies are the ones a plain classifier will quietly
    sacrifice, so their frequency distribution is recorded before the label
    space is fixed.
    """
    order = ("C", "A", "T", "H")
    levels: Dict[str, Set[str]] = {level: set() for level in order}
    h_counts: collections.Counter = collections.Counter()
    labelled_domains = 0

    for record in records:
        for domain in record.domains:
            if not domain.cath:
                continue
            labelled_domains += 1
            for depth, level in enumerate(order, start=1):
                code = domain.cath.code(level)
                # A truncated label (TED often resolves only to T) returns a
                # shorter code, which must not be counted as a deeper label.
                if code and code.count(".") + 1 == depth:
                    levels[level].add(code)
                    if level == "H":
                        h_counts[code] += 1

    sizes = sorted(h_counts.values(), reverse=True)
    total = sum(sizes)
    singletons = sum(1 for n in sizes if n == 1)
    return {
        "labelled_domains": labelled_domains,
        "distinct": {level: len(values) for level, values in levels.items()},
        "h_superfamilies": len(sizes),
        "h_singleton_superfamilies": singletons,
        "h_superfamilies_under_10_domains": sum(1 for n in sizes if n < 10),
        "h_share_of_top_1_percent": (
            round(sum(sizes[: max(1, len(sizes) // 100)]) / total, 4) if total else 0.0
        ),
        "h_domains_per_superfamily": _percentiles(sizes),
    }


def cathe2_label_coverage(
    records: Iterable[ChainRecord], cathe2_superfamilies: Sequence[str]
) -> Dict[str, object]:
    """How much of CATHe2's test label space our CATH labels cover.

    Checkable without CATHe2's domain identifiers, so it gives a usable signal
    on classification comparability even before their archive is unpacked.
    """
    if not cathe2_superfamilies:
        return {"available": False}
    ours = {
        domain.cath.code("H")
        for record in records
        for domain in record.domains
        if domain.cath and domain.cath.code("H")
    }
    wanted = collections.Counter(cathe2_superfamilies)
    missing = {sf: n for sf, n in wanted.items() if sf not in ours}
    return {
        "available": True,
        "cathe2_test_domains": sum(wanted.values()),
        "cathe2_distinct_superfamilies": len(wanted),
        "covered_superfamilies": len(wanted) - len(missing),
        "missing_superfamilies": len(missing),
        "missing_test_domains": sum(missing.values()),
        "examples_missing": sorted(missing)[:10],
    }


def conflict_report(
    primary: Dict[str, ChainRecord],
    others: Iterable[ChainRecord],
) -> Dict[str, object]:
    """Compare other sources against CATH wherever they describe one sequence.

    Sources are joined on sequence MD5, which Pfam and TED both publish, so
    boundaries are compared only where the underlying residues are provably
    identical. This is the measurement risk 1 calls for: it quantifies how far
    apart the sources are before any of them is used as a training target.
    """
    by_source: Dict[str, Dict[str, object]] = {}
    for record in others:
        if not record.sequence_md5:
            continue
        reference = primary.get(record.sequence_md5)
        if reference is None:
            continue

        bucket = by_source.setdefault(
            record.source,
            {
                "compared_chains": 0,
                "domain_count_equal": 0,
                "domain_count_delta": collections.Counter(),
                "best_iou": [],
                "coverage_delta": [],
            },
        )
        bucket["compared_chains"] = int(bucket["compared_chains"]) + 1  # type: ignore[arg-type]
        delta = record.n_domains - reference.n_domains
        if delta == 0:
            bucket["domain_count_equal"] = int(bucket["domain_count_equal"]) + 1  # type: ignore[arg-type]
        bucket["domain_count_delta"][max(-3, min(3, delta))] += 1  # type: ignore[index]

        for domain in record.domains:
            spans = [s.as_tuple() for s in domain.segments]
            best = max(
                (
                    iou(spans, [s.as_tuple() for s in reference_domain.segments])
                    for reference_domain in reference.domains
                ),
                default=0.0,
            )
            bucket["best_iou"].append(best)  # type: ignore[union-attr]
        bucket["coverage_delta"].append(  # type: ignore[union-attr]
            round(record.coverage - reference.coverage, 3)
        )

    summary: Dict[str, object] = {}
    for source, bucket in by_source.items():
        ious = bucket["best_iou"]  # type: ignore[index]
        summary[source] = {
            "compared_chains": bucket["compared_chains"],
            "domain_count_equal": bucket["domain_count_equal"],
            "domain_count_agreement": (
                round(
                    int(bucket["domain_count_equal"]) / int(bucket["compared_chains"]), 4
                )
                if int(bucket["compared_chains"])  # type: ignore[arg-type]
                else 0.0
            ),
            "domain_count_delta": dict(
                sorted(bucket["domain_count_delta"].items())  # type: ignore[union-attr]
            ),
            "per_domain_best_iou": _percentiles(ious),  # type: ignore[arg-type]
            "domains_matched_at_iou_0.8": (
                round(sum(1 for v in ious if v >= 0.8) / len(ious), 4) if ious else 0.0
            ),
            "coverage_delta": _percentiles(bucket["coverage_delta"]),  # type: ignore[arg-type]
        }
    return summary or {"compared_chains": 0, "note": "no shared sequences found"}
