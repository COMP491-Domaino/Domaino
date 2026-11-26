"""Optional 3-D structure for the result page (task T4.5).

The structure is for display only: Domaino reads the sequence alone, and
nothing here feeds back into segmentation or classification. It runs beside
the prediction rather than inside it, so a slow or failed lookup never delays
or breaks the sequence-only result.

The first source is the AlphaFold Protein Structure Database. A submitted
sequence is matched to UniProt entries by its CRC64 checksum, and the
AlphaFold model of a matching entry is accepted only if the model's own
sequence has the same MD5 as the submission. Exact identity is what makes the
model's residue numbers the submission's residue numbers, so the domain
colours on the structure are the domain colours on the sequence map
(milestone M4.3).

Privacy: what leaves the server is the checksum and then a public UniProt
accession, never the sequence. The lookup keeps only a fingerprint, so the
job can still drop its sequence the moment the prediction finishes.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

PENDING = "pending"
AVAILABLE = "available"
UNAVAILABLE = "unavailable"

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/"
ALPHAFOLD_ENTRY = "https://alphafold.ebi.ac.uk/entry/"

# A sequence can sit in many UniProt entries (orthologues with identical
# sequences, redundant unreviewed entries). Trying a handful finds a modelled
# one without turning one lookup into dozens of requests.
MAX_CANDIDATES = 8

USER_AGENT = "domaino-structure-lookup/1.0"

Fetch = Callable[[str], bytes]


def _crc64_table() -> List[int]:
    polynomial = 0xD800000000000000
    table = []
    for byte in range(256):
        value = byte
        for _ in range(8):
            value = (value >> 1) ^ polynomial if value & 1 else value >> 1
        table.append(value)
    return table


_CRC64_TABLE = _crc64_table()


def crc64(sequence: str) -> str:
    """UniProt's sequence checksum (CRC-64-ISO), as 16 upper-case hex digits."""
    value = 0
    for byte in sequence.encode("ascii"):
        value = _CRC64_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    return f"{value:016X}"


@dataclass(frozen=True)
class Fingerprint:
    """What a structure lookup may know about a sequence: not the sequence."""

    crc64: str
    md5: str
    length: int

    @classmethod
    def of(cls, sequence: str) -> "Fingerprint":
        return cls(
            crc64=crc64(sequence),
            md5=hashlib.md5(sequence.encode("ascii")).hexdigest(),
            length=len(sequence),
        )


@dataclass
class StructureResult:
    """A lookup's outcome: the summary the API shows, plus the model file."""

    summary: Dict[str, Any]
    model: Optional[str] = None

    @classmethod
    def unavailable(cls, message: str, source: Optional[str] = None) -> "StructureResult":
        summary: Dict[str, Any] = {"status": UNAVAILABLE, "message": message}
        if source:
            summary["source"] = source
        return cls(summary)


def http_fetch(timeout: float = 10.0) -> Fetch:
    """A GET that returns the body or raises; injectable so tests stay offline."""

    def fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    return fetch


class AlphaFoldDB:
    """Find the AlphaFold DB model whose sequence is exactly the submission."""

    source = "alphafold_db"

    def __init__(self, fetch: Optional[Fetch] = None) -> None:
        self.fetch = fetch or http_fetch()

    def lookup(self, fingerprint: Fingerprint) -> StructureResult:
        try:
            accessions = self._uniprot_accessions(fingerprint)
        except (urllib.error.URLError, OSError, ValueError) as problem:
            return StructureResult.unavailable(
                f"UniProt could not be reached to look the sequence up ({_reason(problem)})",
                self.source,
            )
        if not accessions:
            return StructureResult.unavailable(
                "this exact sequence is not in UniProt, so AlphaFold DB has no "
                "model of it; local structure prediction is planned for such "
                "sequences",
                self.source,
            )

        for accession in accessions[:MAX_CANDIDATES]:
            try:
                entry = self._matching_model(accession, fingerprint)
                if entry is None:
                    continue
                model = self.fetch(entry["pdbUrl"]).decode("utf-8")
            except urllib.error.HTTPError as problem:
                if problem.code == 404:
                    continue
                return self._unreachable(problem)
            except (urllib.error.URLError, OSError, ValueError, KeyError) as problem:
                return self._unreachable(problem)
            return StructureResult(self._summary(entry, fingerprint), model)

        return StructureResult.unavailable(
            "the sequence is in UniProt, but AlphaFold DB has no model of the "
            "complete chain; local structure prediction is planned for such "
            "sequences",
            self.source,
        )

    # -- internals ---------------------------------------------------------

    def _uniprot_accessions(self, fingerprint: Fingerprint) -> List[str]:
        """UniProt entries with this checksum, reviewed (Swiss-Prot) first.

        Identical sequences occur in several species; UniProt's own ranking,
        with reviewed entries ahead, puts the canonical entry first, which is
        the one whose name and organism the result page then shows.
        """
        query = urllib.parse.urlencode(
            {
                "query": f"checksum:{fingerprint.crc64}",
                "format": "tsv",
                "fields": "accession,reviewed,length",
                "size": "25",
            }
        )
        text = self.fetch(f"{UNIPROT_SEARCH}?{query}").decode("utf-8")
        reviewed: List[str] = []
        unreviewed: List[str] = []
        for line in text.splitlines()[1:]:
            columns = line.split("\t")
            if len(columns) < 3 or not columns[2].strip().isdigit():
                continue
            # A CRC64 collision would match a different sequence; the length
            # check catches most, and the MD5 check on the model catches all.
            if int(columns[2]) != fingerprint.length:
                continue
            target = reviewed if columns[1].strip() == "reviewed" else unreviewed
            target.append(columns[0].strip())
        return list(dict.fromkeys(reviewed + unreviewed))

    def _matching_model(
        self, accession: str, fingerprint: Fingerprint
    ) -> Optional[Dict[str, Any]]:
        entries = json.loads(self.fetch(ALPHAFOLD_API + urllib.parse.quote(accession)))
        for entry in entries if isinstance(entries, list) else []:
            sequence = entry.get("sequence") or ""
            # Long proteins are split into overlapping fragments (F1, F2, ...);
            # only a model of the whole chain numbers residues like the input.
            if entry.get("sequenceStart", 1) != 1:
                continue
            if hashlib.md5(sequence.encode("ascii")).hexdigest() == fingerprint.md5:
                return entry
        return None

    def _summary(self, entry: Dict[str, Any], fingerprint: Fingerprint) -> Dict[str, Any]:
        accession = entry.get("uniprotAccession")
        summary: Dict[str, Any] = {
            "status": AVAILABLE,
            "source": self.source,
            "message": "AlphaFold DB model of this exact sequence; shown for "
            "visualisation only and not used by Domaino",
            "model_id": entry.get("entryId") or entry.get("modelEntityId"),
            "version": entry.get("latestVersion"),
            "accession": accession,
            "description": entry.get("uniprotDescription"),
            "gene": entry.get("gene"),
            "organism": entry.get("organismScientificName"),
            "mean_plddt": entry.get("globalMetricValue"),
            "residues": fingerprint.length,
            "format": "pdb",
        }
        if accession:
            summary["entry_url"] = ALPHAFOLD_ENTRY + accession
        return {key: value for key, value in summary.items() if value is not None}

    def _unreachable(self, problem: BaseException) -> StructureResult:
        return StructureResult.unavailable(
            f"AlphaFold DB could not be reached ({_reason(problem)})", self.source
        )


def _reason(problem: BaseException) -> str:
    if isinstance(problem, urllib.error.HTTPError):
        return f"HTTP {problem.code}"
    if isinstance(problem, urllib.error.URLError):
        return str(problem.reason)
    return type(problem).__name__


def lookup_structure(provider, fingerprint: Fingerprint) -> StructureResult:
    """Run one provider, turning any fault into an `unavailable` result.

    The structure is optional by design, so no failure here may propagate into
    the job that carries the sequence-only prediction.
    """
    if provider is None:
        return StructureResult.unavailable("structure lookup is disabled on this server")
    try:
        return provider.lookup(fingerprint)
    except Exception as problem:  # noqa: BLE001 - reported to the client
        return StructureResult.unavailable(
            f"the structure lookup failed ({type(problem).__name__})",
            getattr(provider, "source", None),
        )


def provider_sources(provider) -> Tuple[str, ...]:
    return (provider.source,) if provider is not None else ()
