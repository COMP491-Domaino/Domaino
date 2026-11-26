"""The API contract, loaded from the schema T4.1 defines.

Validating against the published schema rather than against a copy in code is
the point: a response the frontend cannot parse fails a backend test, and the
document in `docs/api/` cannot quietly become out of date.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any, Dict, List, Optional

API_VERSION = "1.0"

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "docs",
    "api",
    "domaino-api-v1.json",
)


class ContractError(ValueError):
    """A message that does not match the published schema."""

    def __init__(self, definition: str, errors: List[str]) -> None:
        super().__init__(
            f"{definition} violates the v{API_VERSION} contract: "
            + "; ".join(errors)
        )
        self.definition = definition
        self.errors = errors


class Contract:
    """Validation against one loaded copy of the schema document."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or SCHEMA_PATH
        with open(self.path, "r", encoding="utf-8") as handle:
            self.document: Dict[str, Any] = json.load(handle)
        if self.document.get("x-api-version") != API_VERSION:
            raise ValueError(
                f"schema at {self.path} declares version "
                f"{self.document.get('x-api-version')}, code expects {API_VERSION}"
            )
        self._validators: Dict[str, Any] = {}

    @property
    def definitions(self) -> List[str]:
        return sorted(self.document["$defs"])

    def validator(self, definition: str):
        """A cached validator for one `$defs` entry."""
        if definition not in self._validators:
            import jsonschema

            if definition not in self.document["$defs"]:
                raise KeyError(f"no such definition: {definition}")
            # Resolve $ref against the whole document, so sibling definitions
            # referenced from this one are found.
            schema = dict(self.document["$defs"][definition])
            schema["$defs"] = self.document["$defs"]
            self._validators[definition] = jsonschema.Draft202012Validator(schema)
        return self._validators[definition]

    def errors(self, definition: str, payload: Any) -> List[str]:
        found = self.validator(definition).iter_errors(payload)
        return [
            f"{'/'.join(str(p) for p in error.path) or '<root>'}: {error.message}"
            for error in sorted(found, key=lambda e: list(e.path))
        ]

    def check(self, definition: str, payload: Any) -> Any:
        """Raise `ContractError` unless `payload` matches `definition`."""
        errors = self.errors(definition, payload)
        if errors:
            raise ContractError(definition, errors)
        return payload

    def matches(self, definition: str, payload: Any) -> bool:
        return not self.errors(definition, payload)


@functools.lru_cache(maxsize=4)
def contract(path: Optional[str] = None) -> Contract:
    """The shared contract instance; the schema is read once per process."""
    return Contract(path)
