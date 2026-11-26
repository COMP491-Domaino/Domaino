"""HTTP layer of the inference service (task T4.2).

On the standard library, not a web framework: none is installed in this
cluster's environment and the compute nodes that hold the model and the ESM-2
weights have no outbound internet, so a framework dependency would make the
service unrunnable exactly where it has to run. The routing table below is four
entries; moving to FastAPI later means re-declaring those four, not rewriting
the service.

Every response this module emits is validated against the published schema
before it leaves, so a change that breaks the frontend's contract fails here
rather than in the browser.
"""

from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple

from .contract import API_VERSION, Contract, ContractError, contract as load_contract
from .jobs import JobStore
from .predict import ResolvedOptions, build_predictor
from .validate import SequenceProblem, validate_request

# Error code to HTTP status. One table, so a new error code cannot be invented
# at a call site without deciding what it means over HTTP.
STATUS_FOR_ERROR = {
    "invalid_request": 400,
    "invalid_sequence": 400,
    "multiple_records": 400,
    "sequence_too_long": 413,
    "payload_too_large": 413,
    "not_found": 404,
    "method_not_allowed": 405,
    "model_unavailable": 503,
    "internal_error": 500,
}

# Refuse a body larger than this before reading it: the sequence limit is
# 100,000 residues, and the rest is JSON overhead and the optional name.
MAX_BODY_BYTES = 256 * 1024


class Service:
    """The service's behaviour, independent of HTTP.

    Separated from the request handler so the whole API can be tested by
    calling methods -- no socket, no port, no race -- while the handler stays
    a thin translation of status codes and JSON.
    """

    def __init__(
        self,
        predictor=None,
        store: Optional[JobStore] = None,
        api: Optional[Contract] = None,
        synchronous: bool = False,
        validate_responses: bool = True,
        log: Callable[[str], None] = print,
    ) -> None:
        self.predictor = predictor or build_predictor(log=log)
        self.store = store or JobStore()
        self.api = api or load_contract()
        # Synchronous mode runs the prediction inside the POST, which is what
        # the tests want: an asynchronous assertion is a flaky assertion.
        self.synchronous = synchronous
        self.validate_responses = validate_responses
        self.log = log
        self._threads: Dict[str, threading.Thread] = {}

    # -- endpoints ---------------------------------------------------------

    def health(self) -> Tuple[int, Dict[str, Any]]:
        payload = {
            "status": "ok",
            "api_version": API_VERSION,
            "model": self.predictor.model_info(
                ResolvedOptions.from_request(None, self._default_architecture())
            ),
            "jobs": self.store.counts(),
        }
        return 200, self._checked("Health", payload)

    def create_prediction(self, body: Any) -> Tuple[int, Dict[str, Any]]:
        from .validate import normalise_sequence

        try:
            payload = validate_request(body, self.api)
            parsed = normalise_sequence(payload["sequence"], payload.get("name"))
        except SequenceProblem as problem:
            return self._error(problem.as_error())

        options = ResolvedOptions.from_request(
            payload.get("options"), self._default_architecture()
        )
        if options.architecture not in ("pairwise", "query"):
            return self._error(
                {
                    "code": "invalid_request",
                    "message": f"unknown architecture {options.architecture!r}",
                }
            )

        job = self.store.create(parsed.sequence, name=parsed.name)
        self._run(job.job_id, parsed, options)
        current = self.store.get(job.job_id)
        assert current is not None
        return 202, self._checked("Job", current.as_json())

    def get_prediction(self, job_id: str) -> Tuple[int, Dict[str, Any]]:
        job = self.store.get(job_id)
        if job is None:
            return self._error(
                {
                    "code": "not_found",
                    "message": f"no job {job_id}; it may have expired, since "
                    "results are not kept indefinitely",
                }
            )
        return 200, self._checked("Job", job.as_json())

    def delete_prediction(self, job_id: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        if self.store.delete(job_id):
            return 204, None
        return self._error(
            {"code": "not_found", "message": f"no job {job_id}"}
        )

    def schema(self) -> Tuple[int, Dict[str, Any]]:
        """The contract itself, so a client can fetch what it must satisfy."""
        return 200, self.api.document

    # -- internals ---------------------------------------------------------

    def _default_architecture(self) -> str:
        return getattr(self.predictor, "architecture", "pairwise")

    def _run(self, job_id: str, parsed, options: ResolvedOptions) -> None:
        if self.synchronous:
            self._execute(job_id, parsed, options)
            return
        thread = threading.Thread(
            target=self._execute, args=(job_id, parsed, options), daemon=True
        )
        self._threads[job_id] = thread
        thread.start()

    def _execute(self, job_id: str, parsed, options: ResolvedOptions) -> None:
        def progress(stage: str, message: str) -> None:
            self.store.advance(job_id, stage, message)

        try:
            self.store.start(job_id)
            result = self.predictor.predict(
                parsed.sequence,
                options,
                name=parsed.name,
                warnings=parsed.warnings,
                progress=progress,
                retention=self.store.retention_note(),
            )
            if self.validate_responses:
                # A malformed result is a bug on this side, so it fails the job
                # with an internal error rather than reaching the frontend.
                self.api.check("Prediction", result)
            self.store.succeed(job_id, result)
        except ContractError as problem:
            self.log(f"job {job_id} produced an off-contract result: {problem}")
            self.store.fail(
                job_id,
                {
                    "code": "internal_error",
                    "message": "the prediction could not be returned in the "
                    "documented format; this is a server-side fault",
                    "details": {"violations": problem.errors[:5]},
                },
            )
        except SequenceProblem as problem:
            self.store.fail(job_id, problem.as_error())
        except Exception as problem:  # noqa: BLE001 - reported to the client
            self.log(f"job {job_id} failed: {traceback.format_exc()}")
            self.store.fail(
                job_id,
                {
                    "code": "internal_error",
                    "message": f"the prediction failed: {type(problem).__name__}",
                },
            )

    def _error(self, error: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        payload = self._checked("ErrorResponse", {"error": error})
        return STATUS_FOR_ERROR.get(error["code"], 500), payload

    def _checked(self, definition: str, payload: Any) -> Any:
        if self.validate_responses and definition in self.api.document["$defs"]:
            self.api.check(definition, payload)
        return payload


def route(service: Service, method: str, path: str, body: Any):
    """Dispatch one request. The whole routing table of API v1."""
    path = path.rstrip("/") or "/"

    if path in ("/", "/api/v1"):
        return service.health()
    if path == "/api/v1/health":
        if method != "GET":
            return service._error(
                {"code": "method_not_allowed", "message": f"{method} not allowed here"}
            )
        return service.health()
    if path == "/api/v1/schema":
        return service.schema()
    if path == "/api/v1/predictions":
        if method != "POST":
            return service._error(
                {
                    "code": "method_not_allowed",
                    "message": f"{method} not allowed; submit a prediction with POST",
                }
            )
        return service.create_prediction(body)
    if path.startswith("/api/v1/predictions/"):
        job_id = path[len("/api/v1/predictions/") :]
        if method == "GET":
            return service.get_prediction(job_id)
        if method == "DELETE":
            return service.delete_prediction(job_id)
        return service._error(
            {"code": "method_not_allowed", "message": f"{method} not allowed here"}
        )
    return service._error(
        {"code": "not_found", "message": f"no endpoint {path} in API v{API_VERSION}"}
    )


def make_handler(service: Service, static_root: Optional[str] = None):
    """A `BaseHTTPRequestHandler` bound to one `Service`."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"domaino/{API_VERSION}"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            service.log(f"{self.address_string()} {format % args}")

        # -- verbs ---------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            if static_root and not self.path.startswith("/api/"):
                self._serve_static()
                return
            self._dispatch("GET", None)

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch("DELETE", None)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._send(
                    *service._error(
                        {
                            "code": "payload_too_large",
                            "message": f"the request body is {length} bytes; the "
                            f"limit is {MAX_BODY_BYTES}",
                        }
                    )
                )
                return
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as problem:
                self._send(
                    *service._error(
                        {
                            "code": "invalid_request",
                            "message": f"the request body is not valid JSON: {problem}",
                        }
                    )
                )
                return
            self._dispatch("POST", body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            # The frontend is served from a different port during development.
            self.send_response(204)
            self._cors()
            self.send_header("Allow", "GET, POST, DELETE, OPTIONS")
            self.end_headers()

        # -- helpers -------------------------------------------------------

        def _dispatch(self, method: str, body: Any) -> None:
            path = self.path.split("?", 1)[0]
            try:
                status, payload = route(service, method, path, body)
            except Exception:  # noqa: BLE001
                service.log(traceback.format_exc())
                status, payload = 500, {
                    "error": {
                        "code": "internal_error",
                        "message": "the request could not be handled",
                    }
                }
            self._send(status, payload)

        def _send(self, status: int, payload: Optional[Dict[str, Any]]) -> None:
            data = b"" if payload is None else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._cors()
            if data:
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if data and self.command != "HEAD":
                self.wfile.write(data)

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _serve_static(self) -> None:
            import mimetypes
            import os
            import posixpath

            requested = self.path.split("?", 1)[0]
            if requested in ("/", ""):
                requested = "/index.html"
            # Normalise first, then confirm the result is still inside the
            # static root: a path such as /../../etc/passwd must not escape.
            relative = posixpath.normpath(requested).lstrip("/")
            target = os.path.realpath(os.path.join(static_root, relative))
            root = os.path.realpath(static_root)
            if not target.startswith(root + os.sep) and target != root:
                self._send(404, {"error": {"code": "not_found", "message": "no such file"}})
                return
            if not os.path.isfile(target):
                self._send(404, {"error": {"code": "not_found", "message": "no such file"}})
                return
            with open(target, "rb") as handle:
                data = handle.read()
            self.send_response(200)
            self._cors()
            self.send_header(
                "Content-Type", mimetypes.guess_type(target)[0] or "application/octet-stream"
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    service: Optional[Service] = None,
    static_root: Optional[str] = None,
) -> None:
    """Run the service until interrupted."""
    service = service or Service()
    httpd = ThreadingHTTPServer((host, port), make_handler(service, static_root))
    service.log(f"domaino API v{API_VERSION} listening on http://{host}:{port}/api/v1")
    if static_root:
        service.log(f"serving the interface from {static_root}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        service.log("shutting down")
    finally:
        httpd.server_close()
