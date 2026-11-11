"""Benchmark hold-out definitions.

Domaino is compared against two independent published systems, and task T1.5
requires both their test examples *and the close homologues of those examples*
to stay out of training. The two benchmarks are unrelated to each other:

Chainsaw (segmentation)
    Publishes its CATH1363 train/validation/test chain lists directly, so the
    hold-out is exact. Reusing its test chains as Domaino's segmentation test
    set also makes the comparison like-for-like.

CATHe2 (classification)
    Publishes only the superfamily labels of its test set on GitHub; the domain
    identifiers live in its Zenodo archive. `load_cathe2_test_domains` reads
    whatever identifier list has been extracted from that archive, and returns
    an empty set when it is absent, so the pipeline still runs and the audit
    reports the exclusion as unverified rather than silently skipping it.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# A CATH domain identifier: 4-character PDB code, chain character, 2-digit
# domain number, e.g. 1oaiA00.
DOMAIN_ID = re.compile(r"\b[0-9][a-z0-9]{3}[A-Za-z0-9][0-9]{2}\b")

# Column names that hold CATH domain identifiers, in preference order.
DOMAIN_COLUMNS = ("Domain", "domain", "domain_id", "Domain_ID", "cath_id", "CATH_ID")


def is_domain_id(token: str) -> bool:
    """Whether a token really is a CATH domain identifier.

    The shape alone is not enough: a seven-digit row index such as ``1035679``
    matches it too, and CATHe2's archive is full of them, so a real identifier
    is additionally required to contain a letter. Accepting indices here would
    silently produce a hold-out of chains that do not exist.
    """
    return bool(DOMAIN_ID.fullmatch(token)) and any(c.isalpha() for c in token)


def find_domain_ids(text: str) -> Set[str]:
    """Every genuine CATH domain identifier in a line of text."""
    return {m.group(0) for m in DOMAIN_ID.finditer(text) if is_domain_id(m.group(0))}


def load_chainsaw_splits(path: str) -> Dict[str, List[str]]:
    """Load Chainsaw's CATH1363 split file: ``{"train"|"validation"|"test"}``."""
    with open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: list(value) for key, value in payload.items()}


def load_chainsaw_ground_truth(path: str) -> Dict[str, str]:
    """Load Chainsaw's published chopping per test chain.

    Kept for the later benchmark comparison, where Domaino must be scored with
    the same ground truth Chainsaw reported against.
    """
    truth: Dict[str, str] = {}
    with open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            chain = row.get("chain_id") or row.get("chain") or row.get("pdb_chain")
            if not chain:
                continue
            for field in ("ground_truth", "true_chopping", "target", "cath_chopping"):
                if row.get(field):
                    truth[chain.strip()] = row[field].strip()
                    break
    return truth


MAX_TEXT_MEMBER_BYTES = 50_000_000


def load_cathe2_test_domains(
    path: Optional[str], expected_count: Optional[int] = None
) -> Set[str]:
    """Load CATHe2 test-domain identifiers, if they can be obtained.

    Accepts a plain list, a CSV with a column of identifiers, or the Zenodo
    archive itself. Returns an empty set when the identifiers are unavailable,
    which leaves the CATHe2 exclusion inactive; `stage_splits` reports that.
    """
    if not path or not os.path.exists(path):
        return set()
    if path.endswith(".zip"):
        domains, _ = extract_cathe2_test_domains(path, expected_count)
        return domains
    return _domains_from_text(path)


def _domains_from_text(path: str) -> Set[str]:
    with open(path, "rt", encoding="utf-8", errors="replace") as handle:
        domains, _ = _domains_from_csv(handle)
    return domains


def _domains_from_csv(handle: Iterable[str]) -> Tuple[Set[str], Optional[str]]:
    """Read domain identifiers from a table, by column where possible.

    Taking a named column beats scanning the whole row: CATHe2's ``Test.csv``
    carries the identifier in ``Domain`` alongside a numeric index and a free
    text description, and scanning everything would mix them together. Falls
    back to scanning when no identifier column is present, as for a plain list.
    """
    rows = list(handle)
    if not rows:
        return set(), None

    reader = csv.DictReader(rows)
    fields = reader.fieldnames or []
    for column in DOMAIN_COLUMNS:
        if column in fields:
            values = {
                (row.get(column) or "").strip()
                for row in csv.DictReader(rows)
            }
            found = {value for value in values if is_domain_id(value)}
            if found:
                return found, column

    found = set()
    for row in rows:
        found |= find_domain_ids(row)
    return found, None


def zip_text_members(path: str) -> List[zipfile.ZipInfo]:
    """Small text members of an archive, the only ones worth reading.

    The CATHe2 archive is 36.8 GB of mostly multi-gigabyte ``.npz`` embeddings,
    so this keeps inspection cheap.
    """
    with zipfile.ZipFile(path) as archive:
        return [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and info.file_size < MAX_TEXT_MEMBER_BYTES
            and info.filename.lower().endswith((".csv", ".tsv", ".txt"))
        ]


def extract_cathe2_test_domains(
    path: str,
    expected_count: Optional[int] = None,
    tolerance: float = 0.2,
) -> Tuple[Set[str], Dict[str, object]]:
    """Find the archive member holding CATHe2's test-domain identifiers.

    Selecting by filename alone is not safe here: a member holding the
    *training* identifiers would silently exclude a large part of the training
    set, which is a far worse outcome than no exclusion at all. So a candidate
    is accepted only if its identifier count is close to `expected_count` --
    the number of test domains, which is known independently from the row count
    of ``Y_Test_SF.csv`` -- and the chosen member is reported either way.

    Returns the identifiers and a report describing what was considered.
    """
    candidates: List[Dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for info in zip_text_members(path):
            looks_like_test = bool(re.search(r"test", info.filename, re.IGNORECASE))
            if not looks_like_test:
                continue
            with archive.open(info) as member:
                text = io.TextIOWrapper(member, encoding="utf-8", errors="replace")
                found, column = _domains_from_csv(text)
            if found:
                candidates.append(
                    {
                        "member": info.filename,
                        "domains": found,
                        "count": len(found),
                        "column": column,
                    }
                )

    report: Dict[str, object] = {
        "archive": os.path.basename(path),
        "expected_count": expected_count,
        "candidates": [
            {"member": c["member"], "count": c["count"]} for c in candidates
        ],
    }

    if not candidates:
        report["chosen"] = None
        report["reason"] = "no text member matching 'test' contained CATH domain ids"
        return set(), report

    if expected_count:
        # Closest count wins, but only inside the tolerance band.
        best = min(candidates, key=lambda c: abs(int(c["count"]) - expected_count))
        drift = abs(int(best["count"]) - expected_count) / expected_count
        if drift > tolerance:
            report["chosen"] = None
            report["reason"] = (
                f"best candidate {best['member']} has {best['count']} ids, "
                f"{drift:.0%} away from the expected {expected_count}; refusing "
                "to risk excluding the wrong set"
            )
            return set(), report
    else:
        best = max(candidates, key=lambda c: int(c["count"]))
        report["reason"] = "no expected count given; took the largest candidate"

    report["chosen"] = best["member"]
    report["chosen_column"] = best.get("column")
    report["chosen_count"] = best["count"]
    return set(best["domains"]), report  # type: ignore[arg-type]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Inspect a benchmark archive: ``python -m domaino.data.benchmarks <zip>``."""
    import sys

    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    archive = argv[0]
    expected = int(argv[1]) if len(argv) > 1 else None

    print(f"text members under {MAX_TEXT_MEMBER_BYTES} bytes:")
    for info in zip_text_members(archive):
        print(f"  {info.file_size:>12}  {info.filename}")
    domains, report = extract_cathe2_test_domains(archive, expected)
    print("\nselection report:")
    print(json.dumps(report, indent=2))
    print(f"\nidentifiers returned: {len(domains)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def load_cathe2_test_superfamilies(path: Optional[str]) -> List[str]:
    """Load the superfamily labels of the CATHe2 test set (``Y_Test_SF.csv``).

    Used by the audit to report how much of CATHe2's label space Domaino's
    CATH label space covers, which is checkable without the full archive.
    """
    if not path or not os.path.exists(path):
        return []
    labels: List[str] = []
    with open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("SF") or "").strip()
            if value:
                labels.append(value)
    return labels


def chains_of_domains(domain_ids: Iterable[str]) -> Set[str]:
    """Map CATH domain identifiers to their 5-character chain names."""
    return {domain_id[:5] for domain_id in domain_ids if len(domain_id) >= 5}
