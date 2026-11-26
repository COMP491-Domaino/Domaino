"""Backend inference service (task T4.2).

Four concerns, one per module, so that a change to one does not drag the
others:

`contract`
    Loads `docs/api/domaino-api-v1.json` and validates against it. The schema
    is the single definition T4.1 promised, and the service reads it at
    runtime rather than restating it in code, which is the only way the two
    cannot drift.

`validate`
    Turns whatever arrived into a sequence or into a specific, actionable
    error.

`jobs`
    The job store and its state machine. Prediction is asynchronous because
    ESM-2 inference on a 4,000-residue chain is not a request-response
    latency.

`predict`
    Invokes the model, or returns a clearly labelled mock when no checkpoint
    is available, which is what makes the T4.1 milestone reachable before
    WP3's training finishes.

`app`
    The HTTP layer, on the standard library. No web framework is installed in
    this cluster's environment and the compute nodes have no internet, so
    adding one would make the service unrunnable exactly where the model
    lives. `app.handler` is a plain WSGI-ish router; swapping in FastAPI later
    means re-routing four endpoints, not rewriting the service.
"""

from .contract import API_VERSION, Contract, contract
from .jobs import Job, JobStore, JobStatus
from .validate import SequenceProblem, normalise_sequence

__all__ = [
    "API_VERSION",
    "Contract",
    "contract",
    "Job",
    "JobStore",
    "JobStatus",
    "SequenceProblem",
    "normalise_sequence",
]
