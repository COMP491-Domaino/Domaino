"""Tests for the backend inference service (task T4.2).

Written against `Service` rather than a live socket: the API's behaviour is
what matters, and an asynchronous HTTP assertion is a flaky assertion. One
test does start a real server, to confirm the handler, the routing table and
the status codes agree with the rest.

The emphasis is on invalid input and on failures, because WP4's objectives ask
for clear errors and for the sequence-only path to survive when the optional
parts do not.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.service import validate  # noqa: E402
from domaino.service.app import Service, route  # noqa: E402
from domaino.service.contract import API_VERSION, Contract, ContractError  # noqa: E402
from domaino.service.jobs import JobStore  # noqa: E402
from domaino.service.predict import (  # noqa: E402
    MockPredictor, ResolvedOptions, unassigned_intervals,
)

PROTEIN = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHS"
    "LAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGI"
    "KATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAP"
)


def service(**kwargs) -> Service:
    kwargs.setdefault("predictor", MockPredictor())
    kwargs.setdefault("synchronous", True)
    kwargs.setdefault("log", lambda message: None)
    # Tests never reach the network; structure lookups get a fake provider.
    kwargs.setdefault("structure_provider", None)
    return Service(**kwargs)


class ContractTests(unittest.TestCase):
    def test_the_published_schema_loads_and_declares_its_version(self):
        api = Contract()
        self.assertEqual(api.document["x-api-version"], API_VERSION)
        for definition in ("PredictionRequest", "Job", "Prediction", "Error"):
            self.assertIn(definition, api.definitions)

    def test_a_valid_request_passes_and_an_invalid_one_does_not(self):
        api = Contract()
        api.check("PredictionRequest", {"sequence": PROTEIN})
        with self.assertRaises(ContractError):
            api.check("PredictionRequest", {"sequence": 7})
        with self.assertRaises(ContractError):
            # An unknown option is a typo, and silently ignoring it is how a
            # frontend ends up believing it asked for something it did not.
            api.check("PredictionRequest", {"sequence": PROTEIN, "tracks": True})

    def test_an_unknown_definition_is_an_error_not_a_pass(self):
        with self.assertRaises(KeyError):
            Contract().validator("Prediciton")


class SequenceValidationTests(unittest.TestCase):
    def test_a_plain_sequence_passes_unchanged(self):
        parsed = validate.normalise_sequence(PROTEIN)
        self.assertEqual(parsed.sequence, PROTEIN)
        self.assertEqual(parsed.warnings, [])

    def test_a_fasta_header_becomes_the_name(self):
        parsed = validate.normalise_sequence(f">sp|P12345| test protein\n{PROTEIN}")
        self.assertEqual(parsed.sequence, PROTEIN)
        self.assertEqual(parsed.name, "sp|P12345| test protein")
        self.assertEqual(parsed.warnings[0]["code"], "sequence_normalised")

    def test_whitespace_case_and_numbering_are_normalised(self):
        # A numbered, wrapped, lowercase listing, as copied out of a paper.
        messy = (
            f"  1 {PROTEIN[:10].lower()} {PROTEIN[10:20].lower()}\n"
            f" 21 {PROTEIN[20:30].lower()}\n{PROTEIN[30:]}"
        )
        parsed = validate.normalise_sequence(messy)
        self.assertEqual(parsed.sequence, PROTEIN)
        self.assertTrue(
            any(w["code"] == "sequence_normalised" for w in parsed.warnings)
        )

    def test_a_bad_letter_is_reported_with_its_position(self):
        with self.assertRaises(validate.SequenceProblem) as caught:
            validate.normalise_sequence(PROTEIN[:40] + "@" + PROTEIN[40:])
        problem = caught.exception
        self.assertEqual(problem.code, "invalid_sequence")
        self.assertIn("41", problem.message)
        self.assertEqual(problem.details["characters"], ["@"])

    def test_two_fasta_records_are_refused_not_truncated(self):
        with self.assertRaises(validate.SequenceProblem) as caught:
            validate.normalise_sequence(f">a\n{PROTEIN}\n>b\n{PROTEIN}")
        self.assertEqual(caught.exception.code, "multiple_records")
        self.assertEqual(caught.exception.details["records"], 2)

    def test_a_header_in_the_middle_is_an_error(self):
        with self.assertRaises(validate.SequenceProblem) as caught:
            validate.normalise_sequence(f"{PROTEIN}\n>b")
        self.assertEqual(caught.exception.code, "invalid_sequence")

    def test_a_nucleotide_sequence_is_refused_with_an_explanation(self):
        with self.assertRaises(validate.SequenceProblem) as caught:
            validate.normalise_sequence("ACGT" * 40)
        self.assertEqual(caught.exception.code, "invalid_sequence")
        self.assertIn("translate", caught.exception.message)

    def test_a_short_protein_of_acgt_letters_is_still_accepted(self):
        # ACGT are amino-acid letters too, so the nucleotide check must only
        # fire where an all-ACGT protein is implausible.
        parsed = validate.normalise_sequence("ACGT" * 5)
        self.assertEqual(parsed.length, 20)

    def test_too_short_and_too_long_are_separate_errors(self):
        with self.assertRaises(validate.SequenceProblem) as short:
            validate.normalise_sequence("MKT")
        self.assertEqual(short.exception.code, "invalid_sequence")
        with self.assertRaises(validate.SequenceProblem) as long_one:
            validate.normalise_sequence("M" * (validate.MAX_SEQUENCE_LENGTH + 1))
        self.assertEqual(long_one.exception.code, "sequence_too_long")

    def test_a_long_chain_warns_about_the_encoder_window(self):
        parsed = validate.normalise_sequence("M" * (validate.ENCODER_WINDOW + 10))
        self.assertTrue(
            any(w["code"] == "exceeds_encoder_window" for w in parsed.warnings)
        )

    def test_ambiguous_residues_warn_but_do_not_fail(self):
        parsed = validate.normalise_sequence(PROTEIN[:50] + "XXX" + PROTEIN[53:])
        self.assertTrue(
            any(w["code"] == "non_standard_residues" for w in parsed.warnings)
        )
        self.assertEqual(parsed.length, len(PROTEIN))

    def test_an_empty_submission_is_an_error(self):
        for value in ("", "   ", "\n\n"):
            with self.assertRaises(validate.SequenceProblem):
                validate.normalise_sequence(value)


class JobStoreTests(unittest.TestCase):
    def test_the_state_machine_runs_forwards_only(self):
        store = JobStore()
        job = store.create(PROTEIN)
        self.assertEqual(job.status, "queued")
        store.start(job.job_id)
        self.assertEqual(store.get(job.job_id).status, "running")
        store.succeed(job.job_id, {"sequence_length": 10})
        with self.assertRaises(ValueError):
            store.start(job.job_id)

    def test_the_sequence_is_dropped_when_the_job_finishes(self):
        store = JobStore()
        job = store.create(PROTEIN)
        self.assertEqual(job.sequence, PROTEIN)
        store.start(job.job_id)
        store.succeed(job.job_id, {})
        self.assertIsNone(store.get(job.job_id).sequence)

    def test_a_failed_job_also_drops_its_sequence(self):
        store = JobStore()
        job = store.create(PROTEIN)
        store.start(job.job_id)
        store.fail(job.job_id, {"code": "internal_error", "message": "x"})
        self.assertIsNone(store.get(job.job_id).sequence)

    def test_the_serialised_job_never_carries_the_sequence(self):
        store = JobStore()
        job = store.create(PROTEIN)
        self.assertNotIn("sequence", json.dumps(job.as_json()))

    def test_results_expire(self):
        now = [1000.0]
        store = JobStore(retention_seconds=60.0, clock=lambda: now[0])
        job = store.create(PROTEIN)
        now[0] += 61.0
        self.assertIsNone(store.get(job.job_id))

    def test_the_store_is_bounded(self):
        now = [0.0]
        store = JobStore(capacity=3, clock=lambda: now[0])
        ids = []
        for _ in range(5):
            now[0] += 1.0
            ids.append(store.create(PROTEIN).job_id)
        self.assertIsNone(store.get(ids[0]))
        self.assertIsNotNone(store.get(ids[-1]))

    def test_persisted_jobs_hold_no_sequence_and_are_removed_on_delete(self):
        directory = tempfile.mkdtemp()
        store = JobStore(directory=directory)
        job = store.create(PROTEIN)
        store.start(job.job_id)
        store.succeed(job.job_id, {})
        path = os.path.join(directory, f"{job.job_id}.json")
        self.assertTrue(os.path.exists(path))
        self.assertNotIn(PROTEIN, open(path).read())
        store.delete(job.job_id)
        self.assertFalse(os.path.exists(path))

    def test_an_unknown_stage_is_rejected(self):
        store = JobStore()
        job = store.create(PROTEIN)
        with self.assertRaises(ValueError):
            store.advance(job.job_id, "folding")


class MockPredictorTests(unittest.TestCase):
    def test_the_same_sequence_gives_the_same_result(self):
        options = ResolvedOptions()
        first = MockPredictor().predict(PROTEIN, options)
        second = MockPredictor().predict(PROTEIN, options)
        self.assertEqual(first["domains"], second["domains"])

    def test_every_mock_result_is_labelled_as_one(self):
        result = MockPredictor().predict(PROTEIN, ResolvedOptions())
        self.assertEqual(result["model"]["mode"], "mock")
        self.assertTrue(
            any("simulated" in w["message"] for w in result["warnings"])
        )

    def test_results_match_the_published_prediction_schema(self):
        api = Contract()
        # Many lengths, because the shape it simulates depends on length.
        for length in (40, 120, 260, 500, 900):
            result = MockPredictor().predict(
                "M" * length, ResolvedOptions(include_tracks=True)
            )
            api.check("Prediction", result)
            self.assertEqual(len(result["tracks"]["in_domain"]), length)

    def test_it_produces_discontinuous_domains_for_some_input(self):
        # The frontend has to render them, so the mock must emit them.
        found = False
        for index in range(60):
            result = MockPredictor().predict(
                "M" * 300 + "K" * index, ResolvedOptions()
            )
            if any(d["discontinuous"] for d in result["domains"]):
                found = True
                break
        self.assertTrue(found, "the mock never produced a discontinuous domain")

    def test_unassigned_intervals_cover_the_gaps(self):
        self.assertEqual(
            unassigned_intervals(10, [[(2, 4)], [(7, 8)]]),
            [{"start": 1, "end": 1}, {"start": 5, "end": 6},
             {"start": 9, "end": 10}],
        )
        self.assertEqual(unassigned_intervals(5, [[(1, 5)]]), [])


class ApiTests(unittest.TestCase):
    def test_health_reports_the_loaded_model(self):
        status, payload = service().health()
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model"]["mode"], "mock")

    def test_a_submission_runs_and_can_be_fetched(self):
        api = service()
        status, job = api.create_prediction({"sequence": PROTEIN})
        self.assertEqual(status, 202)
        self.assertEqual(job["status"], "succeeded")
        status, fetched = api.get_prediction(job["job_id"])
        self.assertEqual(status, 200)
        self.assertEqual(fetched["result"]["sequence_length"], len(PROTEIN))
        self.assertIn("retention", fetched["result"]["provenance"])

    def test_the_name_from_a_fasta_header_reaches_the_job(self):
        status, job = service().create_prediction(
            {"sequence": f">my protein\n{PROTEIN}"}
        )
        self.assertEqual(job["name"], "my protein")

    def test_an_explicit_name_is_kept(self):
        _, job = service().create_prediction(
            {"sequence": PROTEIN, "name": "test case"}
        )
        self.assertEqual(job["name"], "test case")

    def test_an_invalid_sequence_is_rejected_before_a_job_is_made(self):
        api = service()
        status, payload = api.create_prediction({"sequence": "MKT@@@"})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_sequence")
        self.assertEqual(api.store.counts()["completed"], 0)
        self.assertEqual(api.store.counts()["queued"], 0)

    def test_a_body_that_is_not_an_object_is_rejected(self):
        status, payload = service().create_prediction(["MKT"])
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")

    def test_a_missing_sequence_field_is_rejected(self):
        status, payload = service().create_prediction({"name": "no sequence"})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")

    def test_an_unknown_architecture_is_rejected(self):
        status, payload = service().create_prediction(
            {"sequence": PROTEIN, "options": {"architecture": "transformer"}}
        )
        self.assertEqual(status, 400)

    def test_an_oversized_sequence_gets_413(self):
        status, payload = service().create_prediction(
            {"sequence": "M" * (validate.MAX_SEQUENCE_LENGTH + 1)}
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "sequence_too_long")

    def test_an_unknown_job_gets_404_with_an_explanation(self):
        status, payload = service().get_prediction("0" * 16)
        self.assertEqual(status, 404)
        self.assertIn("expired", payload["error"]["message"])

    def test_a_job_can_be_deleted_and_is_then_gone(self):
        api = service()
        _, job = api.create_prediction({"sequence": PROTEIN})
        status, body = api.delete_prediction(job["job_id"])
        self.assertEqual(status, 204)
        self.assertIsNone(body)
        self.assertEqual(api.get_prediction(job["job_id"])[0], 404)

    def test_a_structure_request_still_returns_the_sequence_result(self):
        # One of WP4's objectives: the core prediction stays available when an
        # optional service does not.
        class Broken:
            source = "alphafold_db"

            def lookup(self, fingerprint):
                raise ConnectionError("no route to the structure source")

        api = service(structure_provider=Broken())
        _, job = api.create_prediction(
            {"sequence": PROTEIN, "options": {"include_structure": True}}
        )
        self.assertEqual(job["status"], "succeeded")
        self.assertTrue(job["result"]["domains"])
        status, body = api.get_structure(job["job_id"])
        self.assertEqual(status, 200)
        self.assertEqual(body["structure"]["status"], "unavailable")
        self.assertNotIn("model", body)

    def test_tracks_are_returned_only_when_asked_for(self):
        api = service()
        _, plain = api.create_prediction({"sequence": PROTEIN})
        self.assertNotIn("tracks", plain["result"])
        _, with_tracks = api.create_prediction(
            {"sequence": PROTEIN, "options": {"include_tracks": True}}
        )
        self.assertEqual(
            len(with_tracks["result"]["tracks"]["in_domain"]), len(PROTEIN)
        )

    def test_a_predictor_that_raises_fails_the_job_without_killing_the_service(self):
        class Broken(MockPredictor):
            def predict(self, *args, **kwargs):
                raise RuntimeError("the GPU fell over")

        api = service(predictor=Broken())
        status, job = api.create_prediction({"sequence": PROTEIN})
        self.assertEqual(status, 202)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"]["code"], "internal_error")
        # And the service still answers.
        self.assertEqual(api.health()[0], 200)

    def test_an_off_contract_result_fails_the_job_rather_than_reaching_a_client(self):
        class Wrong(MockPredictor):
            def predict(self, *args, **kwargs):
                return {"domains": "all of them"}

        status, job = service(predictor=Wrong()).create_prediction(
            {"sequence": PROTEIN}
        )
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"]["code"], "internal_error")
        self.assertIn("violations", job["error"]["details"])

    def test_an_asynchronous_submission_starts_queued_and_finishes(self):
        api = service(synchronous=False)
        _, job = api.create_prediction({"sequence": PROTEIN})
        self.assertIn(job["status"], ("queued", "running", "succeeded"))
        for thread in api._threads.values():
            thread.join(timeout=30)
        self.assertEqual(api.get_prediction(job["job_id"])[1]["status"], "succeeded")


class RoutingTests(unittest.TestCase):
    def test_the_routing_table(self):
        api = service()
        self.assertEqual(route(api, "GET", "/api/v1/health", None)[0], 200)
        self.assertEqual(route(api, "GET", "/", None)[0], 200)
        self.assertEqual(route(api, "GET", "/api/v1/schema", None)[0], 200)
        self.assertEqual(route(api, "GET", "/api/v1/nope", None)[0], 404)
        self.assertEqual(
            route(api, "GET", "/api/v1/predictions", None)[0], 405
        )
        self.assertEqual(
            route(api, "POST", "/api/v1/predictions", {"sequence": PROTEIN})[0], 202
        )

    def test_a_trailing_slash_routes_the_same_way(self):
        api = service()
        self.assertEqual(route(api, "GET", "/api/v1/health/", None)[0], 200)


class LiveServerTests(unittest.TestCase):
    """One end-to-end pass over a real socket.

    `Service` tests cover behaviour; this covers the parts only the handler
    has: status codes on the wire, JSON encoding, and the body-size guard.
    """

    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer

        from domaino.service.app import make_handler

        cls.api = service()
        cls.httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(cls.api)
        )
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_health_over_http(self):
        with urllib.request.urlopen(self.url("/api/v1/health")) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.load(response)["api_version"], API_VERSION)

    def test_a_full_submit_and_poll_over_http(self):
        request = urllib.request.Request(
            self.url("/api/v1/predictions"),
            data=json.dumps({"sequence": PROTEIN}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 202)
            job = json.load(response)
        with urllib.request.urlopen(self.url(job["links"]["self"])) as response:
            fetched = json.load(response)
        self.assertEqual(fetched["result"]["sequence_length"], len(PROTEIN))

    def test_malformed_json_gets_a_readable_400(self):
        request = urllib.request.Request(
            self.url("/api/v1/predictions"), data=b"{not json",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 400)
        body = json.load(caught.exception)
        self.assertEqual(body["error"]["code"], "invalid_request")

    def test_an_unknown_path_gets_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.url("/api/v1/predictionz"))
        self.assertEqual(caught.exception.code, 404)

    def test_a_wrong_method_gets_405(self):
        request = urllib.request.Request(
            self.url("/api/v1/predictions"), method="DELETE"
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 405)


if __name__ == "__main__":
    unittest.main()
