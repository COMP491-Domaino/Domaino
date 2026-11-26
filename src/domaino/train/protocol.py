"""Controlled training and experiment protocol (task T3.1).

T3.1 asks for the random seeds, dataset versions, training rules, computing
budget and checkpoint-selection criteria to be fixed, and for every run to
record its configuration, code revision, runtime and result. This module is
that protocol in executable form, so a claim about a result can always be
traced back to the run that produced it.

Two choices here are deliberate and worth stating.

Selection never looks at the test data
    `CHECKPOINT_METRIC` is a validation metric. The proposal requires the final
    model to be chosen "using validation data without repeatedly inspecting the
    external test benchmarks", so the selection rule cannot read them, and the
    registry records which split a number came from.

A run is identified by what would change its result
    The run id hashes the resolved configuration, so two runs with the same
    settings collide on purpose and a changed setting produces a new id. The
    code revision is recorded separately, because the same configuration on
    different code is a different experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# -- fixed protocol ---------------------------------------------------------

#: Seeds. Every run uses the first one unless a protocol experiment explicitly
#: repeats a configuration across all of them to measure seed variance.
SEEDS: Sequence[int] = (20260404, 20260405, 20260406)

#: The dataset this work package trains on. A run against a different version
#: is a different experiment and the registry refuses to mix them silently.
DATASET_VERSION = "dataset_v1"

#: Checkpoint selection: the validation metric to maximise, and how long to
#: wait for it to improve. Mean IoU is the measure CHAINSAW reports, so the
#: model is selected on the same quantity it will be compared on.
CHECKPOINT_METRIC = "val_mean_iou"
CHECKPOINT_MODE = "max"
EARLY_STOPPING_PATIENCE = 5

#: Computing budget agreed for T3.2, in GPU-hours, across every run it covers.
BUDGET_GPU_HOURS = 30.0
#: Per-run ceilings, so one run cannot consume the whole budget unnoticed.
MAX_EPOCHS = 20
MAX_RUN_HOURS = 8.0

#: Categories every result is broken down by, because T3.2 requires results
#: "separated by domain count and domain type" and an average hides exactly
#: the cases domain parsing exists for.
REPORT_GROUPS = (
    "all",
    "single_domain",
    "multidomain",
    "continuous",
    "discontinuous",
    "long",
)
LONG_CHAIN_RESIDUES = 500


def code_revision(repo: str = ".") -> Dict[str, str]:
    """The git revision and whether the tree was dirty when the run started.

    A dirty tree means the recorded revision does not fully describe the code,
    so it is flagged rather than quietly recorded as clean.
    """
    def git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=30
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "revision": revision or "unknown",
        "dirty": "yes" if status else ("no" if status == "" else "unknown"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
    }


def environment() -> Dict[str, str]:
    """What the run executed on, enough to explain a performance difference."""
    info = {
        "python": platform.python_version(),
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID", "none"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "cpu"
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpus"] = str(torch.cuda.device_count())
        else:
            info["gpu"] = "none"
    except Exception:
        info["torch"] = "unavailable"
    return info


def run_id(config: Dict[str, Any], prefix: str = "run") -> str:
    """Stable short id from the settings that would change the result."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:12]}"


@dataclass
class RunRecord:
    """Everything T3.1 requires a run to record."""

    run_id: str
    name: str
    architecture: str
    dataset_version: str
    seed: int
    config: Dict[str, Any] = field(default_factory=dict)
    code: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    runtime_seconds: float = 0.0
    gpu_hours: float = 0.0
    epochs_run: int = 0
    status: str = "running"
    # Validation history, one entry per epoch, and the selected checkpoint.
    history: List[Dict[str, Any]] = field(default_factory=list)
    selected: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    """A directory of run records, one JSON file per run.

    Deliberately plain files rather than a service: the records have to outlive
    this project and be readable by whoever picks it up.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path_for(self, identifier: str) -> str:
        return os.path.join(self.root, f"{identifier}.json")

    def save(self, record: RunRecord) -> str:
        path = self.path_for(record.run_id)
        with open(path, "wt", encoding="utf-8") as handle:
            json.dump(record.as_dict(), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        return path

    def load(self, identifier: str) -> Optional[RunRecord]:
        path = self.path_for(identifier)
        if not os.path.exists(path):
            return None
        with open(path, "rt", encoding="utf-8") as handle:
            return RunRecord(**json.load(handle))

    def all_runs(self) -> List[RunRecord]:
        runs = []
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".json") or name == "summary.json":
                continue
            try:
                runs.append(self.load(name[:-5]))
            except (TypeError, ValueError, KeyError):
                continue
        return [r for r in runs if r is not None]

    def spent_gpu_hours(self) -> float:
        """Budget consumed so far, so a run can refuse to overspend it."""
        return round(sum(r.gpu_hours for r in self.all_runs()), 3)

    def budget_remaining(self) -> float:
        return round(BUDGET_GPU_HOURS - self.spent_gpu_hours(), 3)


def is_better(candidate: float, incumbent: Optional[float]) -> bool:
    """Whether `candidate` beats `incumbent` under the selection rule."""
    if incumbent is None:
        return True
    return candidate > incumbent if CHECKPOINT_MODE == "max" else candidate < incumbent


def chain_groups(n_domains: int, discontinuous: bool, length: int) -> List[str]:
    """Reporting groups a chain belongs to (T3.2's required breakdown)."""
    groups = ["all"]
    groups.append("multidomain" if n_domains > 1 else "single_domain")
    groups.append("discontinuous" if discontinuous else "continuous")
    if length >= LONG_CHAIN_RESIDUES:
        groups.append("long")
    return groups


def protocol_summary() -> Dict[str, Any]:
    """The fixed protocol, for the record and for the report."""
    return {
        "seeds": list(SEEDS),
        "dataset_version": DATASET_VERSION,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "checkpoint_mode": CHECKPOINT_MODE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "budget_gpu_hours": BUDGET_GPU_HOURS,
        "max_epochs": MAX_EPOCHS,
        "max_run_hours": MAX_RUN_HOURS,
        "report_groups": list(REPORT_GROUPS),
        "long_chain_residues": LONG_CHAIN_RESIDUES,
        "selection_uses": "validation split only; test and benchmarks are never read during selection",
    }


class Stopwatch:
    """Wall-clock and GPU-hour accounting for one run."""

    def __init__(self, gpus: int = 1) -> None:
        self.gpus = max(gpus, 0)
        self.started = time.time()

    @property
    def seconds(self) -> float:
        return time.time() - self.started

    @property
    def gpu_hours(self) -> float:
        return round(self.seconds / 3600.0 * self.gpus, 4)

    def exceeded(self, hours: float = MAX_RUN_HOURS) -> bool:
        return self.seconds / 3600.0 >= hours
