"""Prediction jobs and their state machine (part of task T4.2).

Why jobs at all: embedding a long chain with ESM-2 takes seconds to minutes,
which is longer than a browser will wait on one request. The frontend submits,
gets an id, and polls -- which also gives it something honest to show while
waiting, because each stage names itself.

Storage is in memory with an optional directory mirror. A submitted sequence is
private (one of WP4's objectives), so it lives only in the job that needs it,
is never written to the mirror, and is dropped the moment the job finishes.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .contract import API_VERSION

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

JobStatus = str

TERMINAL = frozenset({SUCCEEDED, FAILED})
STAGES = ("validating", "embedding", "segmenting", "classifying", "done")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Job:
    """One prediction request and whatever has become of it."""

    job_id: str
    status: JobStatus = QUEUED
    name: Optional[str] = None
    submitted_at: str = field(default_factory=_timestamp)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    stage: str = "validating"
    stage_message: str = "waiting to start"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    # Never serialised and dropped as soon as the job leaves RUNNING.
    sequence: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        """The `Job` message of the v1 contract."""
        payload: Dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "api_version": API_VERSION,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": {"stage": self.stage, "message": self.stage_message},
            "links": {"self": f"/api/v1/predictions/{self.job_id}"},
        }
        if self.name:
            payload["name"] = self.name
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        return payload


class JobStore:
    """Thread-safe job registry with a bounded, expiring history.

    Bounded on purpose: a long-running service that keeps every result grows
    without limit, and a result nobody fetched is not worth the memory. The
    retention window is also the privacy promise reported in `provenance`.
    """

    def __init__(
        self,
        capacity: int = 256,
        retention_seconds: float = 3600.0,
        directory: Optional[str] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.capacity = capacity
        self.retention_seconds = retention_seconds
        self.directory = directory
        self.clock = clock
        self._jobs: Dict[str, Job] = {}
        self._created: Dict[str, float] = {}
        self._lock = threading.RLock()
        if directory:
            os.makedirs(directory, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    def create(self, sequence: str, name: Optional[str] = None) -> Job:
        with self._lock:
            self._expire()
            job = Job(job_id=secrets.token_hex(8), name=name, sequence=sequence)
            self._jobs[job.job_id] = job
            self._created[job.job_id] = self.clock()
            self._trim()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            self._expire()
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            self._created.pop(job_id, None)
            job = self._jobs.pop(job_id, None)
            if job is not None:
                job.sequence = None
            self._forget_on_disk(job_id)
            return job is not None

    def start(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status in TERMINAL:
                raise ValueError(
                    f"job {job_id} already finished as {job.status}; a "
                    "terminal job cannot be restarted"
                )
            job.status = RUNNING
            job.started_at = _timestamp()
            self.advance(job_id, "embedding", "computing residue embeddings")

    def advance(self, job_id: str, stage: str, message: str = "") -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        with self._lock:
            job = self._jobs[job_id]
            job.stage = stage
            job.stage_message = message or stage

    def succeed(self, job_id: str, result: Dict[str, Any]) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            job.status = SUCCEEDED
            job.result = result
            job.error = None
            job.finished_at = _timestamp()
            job.stage = "done"
            job.stage_message = "complete"
            # The sequence is not needed again, so it stops existing here.
            job.sequence = None
            self._persist(job)
            return job

    def fail(self, job_id: str, error: Dict[str, Any]) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            job.status = FAILED
            job.error = error
            job.result = None
            job.finished_at = _timestamp()
            job.stage_message = error.get("message", "failed")
            job.sequence = None
            self._persist(job)
            return job

    # -- reporting ---------------------------------------------------------

    def counts(self) -> Dict[str, int]:
        with self._lock:
            self._expire()
            counts = {"queued": 0, "running": 0, "completed": 0}
            for job in self._jobs.values():
                if job.status == QUEUED:
                    counts["queued"] += 1
                elif job.status == RUNNING:
                    counts["running"] += 1
                else:
                    counts["completed"] += 1
            return counts

    def retention_note(self) -> str:
        minutes = int(self.retention_seconds // 60)
        return (
            f"the submitted sequence is dropped as soon as the prediction "
            f"finishes; the result is kept for {minutes} minutes or until it "
            f"is deleted"
        )

    # -- internals ---------------------------------------------------------

    def _expire(self) -> None:
        now = self.clock()
        stale = [
            job_id
            for job_id, created in self._created.items()
            if now - created > self.retention_seconds
        ]
        for job_id in stale:
            self._created.pop(job_id, None)
            self._jobs.pop(job_id, None)
            self._forget_on_disk(job_id)

    def _trim(self) -> None:
        while len(self._jobs) > self.capacity:
            oldest = min(self._created, key=self._created.get)
            self._created.pop(oldest, None)
            self._jobs.pop(oldest, None)
            self._forget_on_disk(oldest)

    def _persist(self, job: Job) -> None:
        if not self.directory:
            return
        payload = job.as_json()
        # The sequence is absent from `as_json` by construction; this asserts
        # it, because a privacy promise that depends on a field order is not
        # one.
        assert "sequence" not in payload
        with open(self._path(job.job_id), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _forget_on_disk(self, job_id: str) -> None:
        if not self.directory:
            return
        try:
            os.remove(self._path(job_id))
        except FileNotFoundError:
            pass

    def _path(self, job_id: str) -> str:
        return os.path.join(self.directory or ".", f"{job_id}.json")
