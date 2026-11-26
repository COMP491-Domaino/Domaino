"""Input validation for the inference service (part of task T4.2).

Every rejection names what is wrong and where, because the frontend shows the
message verbatim: "invalid sequence" sends a user looking through 800 residues,
while "residue 412 is 'Z', which is not an amino-acid letter" does not.

The accepted input is deliberately forgiving about *form* and strict about
*content*. A user pastes from a paper, a terminal or a FASTA file, so headers,
line breaks, digits from a numbered listing and lowercase all get normalised
and reported. What cannot be guessed at -- a letter that is not an amino acid,
two records in one box, nucleotides pasted by mistake -- is an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..data.schema import AMBIGUOUS_AA, STANDARD_AA

# ESM-2's training crop. Longer chains still work, through the overlapping
# windows of T2.2, but the user is told.
ENCODER_WINDOW = 1022
# Above this the service refuses: 100,000 residues is longer than any known
# protein and is almost always a pasted genome or a whole proteome file.
MAX_SEQUENCE_LENGTH = 100000
# Below this a domain prediction is not meaningful; the shortest CATH domain is
# around 16 residues.
MIN_SEQUENCE_LENGTH = 16

ALLOWED = STANDARD_AA | AMBIGUOUS_AA
# Characters that carry no residue and are simply dropped.
IGNORED = set(" \t\r\n0123456789*-.")
NUCLEOTIDES = frozenset("ACGTUN")


class SequenceProblem(ValueError):
    """A submitted sequence the service will not run.

    Carries the error code and details the API contract defines, so the HTTP
    layer only has to choose a status code.
    """

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: Dict[str, Any] = details

    def as_error(self) -> Dict[str, Any]:
        error: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            error["details"] = self.details
        return error


@dataclass
class NormalisedSequence:
    """A clean sequence plus everything the user should be told about it."""

    sequence: str
    name: Optional[str] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)


def _warning(code: str, message: str, **details: Any) -> Dict[str, Any]:
    warning: Dict[str, Any] = {"code": code, "message": message}
    if details:
        warning["details"] = details
    return warning


def normalise_sequence(raw: str, name: Optional[str] = None) -> NormalisedSequence:
    """Parse and check one submitted sequence.

    Raises `SequenceProblem` with a contract error code, or returns the
    cleaned sequence with a warning for every change made to it.
    """
    if raw is None or not isinstance(raw, str):
        raise SequenceProblem("invalid_request", "`sequence` must be a string")

    text = raw.strip()
    if not text:
        raise SequenceProblem("invalid_sequence", "the sequence is empty")

    if len(text) > MAX_SEQUENCE_LENGTH * 2:
        # Checked on the raw text too, so a 50 MB paste is refused before it is
        # scanned character by character.
        raise SequenceProblem(
            "sequence_too_long",
            f"the input is {len(text)} characters; the limit is "
            f"{MAX_SEQUENCE_LENGTH} residues",
            characters=len(text),
            limit=MAX_SEQUENCE_LENGTH,
        )

    warnings: List[Dict[str, Any]] = []
    header, body = _split_fasta(text)
    if header is not None:
        name = name or header
        warnings.append(
            _warning(
                "sequence_normalised",
                "a FASTA header was found and used as the name",
                header=header,
            )
        )

    cleaned, dropped, bad = _clean(body)

    if bad:
        shown = bad[:5]
        raise SequenceProblem(
            "invalid_sequence",
            "the sequence contains characters that are not amino-acid "
            "letters: "
            + ", ".join(f"{char!r} at residue {position}" for position, char in shown)
            + (f" and {len(bad) - len(shown)} more" if len(bad) > len(shown) else ""),
            positions=[position for position, _ in shown],
            characters=sorted({char for _, char in bad}),
        )

    if not cleaned:
        raise SequenceProblem(
            "invalid_sequence", "the sequence has no residues in it"
        )

    if len(cleaned) > MAX_SEQUENCE_LENGTH:
        raise SequenceProblem(
            "sequence_too_long",
            f"the sequence is {len(cleaned)} residues; the limit is "
            f"{MAX_SEQUENCE_LENGTH}",
            length=len(cleaned),
            limit=MAX_SEQUENCE_LENGTH,
        )

    if len(cleaned) < MIN_SEQUENCE_LENGTH:
        raise SequenceProblem(
            "invalid_sequence",
            f"the sequence is {len(cleaned)} residues; domain prediction needs "
            f"at least {MIN_SEQUENCE_LENGTH}",
            length=len(cleaned),
            minimum=MIN_SEQUENCE_LENGTH,
        )

    if _looks_like_nucleotides(cleaned):
        raise SequenceProblem(
            "invalid_sequence",
            "this looks like a nucleotide sequence; Domaino predicts domains "
            "from amino-acid sequence, so translate it first",
            distinct_letters=sorted(set(cleaned)),
        )

    if dropped:
        warnings.append(
            _warning(
                "sequence_normalised",
                f"{dropped} non-residue characters (whitespace, digits or gap "
                "symbols) were removed",
                removed=dropped,
            )
        )

    ambiguous = sum(1 for residue in cleaned if residue in AMBIGUOUS_AA)
    if ambiguous:
        warnings.append(
            _warning(
                "non_standard_residues",
                f"{ambiguous} residues are ambiguous (X, B, Z, J, O or U); "
                "the encoder handles them, but boundaries near them are less "
                "reliable",
                count=ambiguous,
            )
        )

    if len(cleaned) > ENCODER_WINDOW:
        warnings.append(
            _warning(
                "exceeds_encoder_window",
                f"the chain is {len(cleaned)} residues, longer than the "
                f"encoder's {ENCODER_WINDOW}-residue window, so it is embedded "
                "in overlapping windows",
                length=len(cleaned),
                window=ENCODER_WINDOW,
            )
        )

    return NormalisedSequence(sequence=cleaned, name=name, warnings=warnings)


def _split_fasta(text: str) -> Tuple[Optional[str], str]:
    """Separate a single FASTA header from the body, refusing multi-record."""
    lines = text.splitlines()
    headers = [index for index, line in enumerate(lines) if line.startswith(">")]
    if not headers:
        return None, text
    if len(headers) > 1:
        raise SequenceProblem(
            "multiple_records",
            f"the input holds {len(headers)} FASTA records; submit one "
            "sequence per prediction so that results are not silently mixed",
            records=len(headers),
        )
    if headers[0] != 0:
        raise SequenceProblem(
            "invalid_sequence",
            f"a FASTA header appears at line {headers[0] + 1} rather than at "
            "the start of the input",
            line=headers[0] + 1,
        )
    return lines[0][1:].strip() or None, "\n".join(lines[1:])


def _clean(body: str) -> Tuple[str, int, List[Tuple[int, str]]]:
    """Return (sequence, dropped count, offending (position, char) pairs)."""
    kept: List[str] = []
    dropped = 0
    bad: List[Tuple[int, str]] = []
    for position, char in enumerate(body, start=1):
        if char in IGNORED:
            dropped += 1
            continue
        upper = char.upper()
        if upper in ALLOWED:
            kept.append(upper)
        else:
            bad.append((position, char))
    return "".join(kept), dropped, bad


def _looks_like_nucleotides(sequence: str) -> bool:
    """True for a long sequence built only from nucleotide letters.

    ACGT are all valid amino-acid letters too, so this can only ever be a
    heuristic; it is applied only above a length where an all-ACGTUN protein
    is implausible, and it refuses rather than guesses because running it
    would return confident nonsense.
    """
    return len(sequence) >= 50 and set(sequence) <= NUCLEOTIDES


def validate_request(payload: Any, contract=None) -> Dict[str, Any]:
    """Check a POST body against the contract before touching its content."""
    if not isinstance(payload, dict):
        raise SequenceProblem(
            "invalid_request", "the request body must be a JSON object"
        )
    if contract is not None:
        errors = contract.errors("PredictionRequest", payload)
        if errors:
            raise SequenceProblem(
                "invalid_request",
                "the request does not match the v1 API contract: "
                + "; ".join(errors[:3]),
                violations=errors,
            )
    return payload
