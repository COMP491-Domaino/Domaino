"""Tests for the optional 3-D structure lookup (task T4.5).

All offline: the HTTP layer is replaced by a fake that answers from a table and
records every URL it was asked for, which is also how the privacy promise is
checked -- the sequence must never appear in a request.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.service.app import Service, route  # noqa: E402
from domaino.service.predict import MockPredictor  # noqa: E402
from domaino.service.structure import (  # noqa: E402
    AlphaFoldDB, Fingerprint, crc64, lookup_structure,
)

# Human haemoglobin alpha (UniProt P69905); its checksums were confirmed
# against UniProt and AlphaFold DB.
HBA = (
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAH"
    "VDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
)
PDB_TEXT = "ATOM      1  CA  MET A   1      0.000   0.000   0.000  1.00 98.00           C\nEND\n"


def uniprot_tsv(*rows):
    """UniProtKB search output: (accession, reviewed?, length) per row."""
    lines = ["Entry\tReviewed\tLength"]
    lines += [
        f"{accession}\t{'reviewed' if reviewed else 'unreviewed'}\t{length}"
        for accession, reviewed, length in rows
    ]
    return "\n".join(lines) + "\n"


def afdb_entry(accession, sequence=HBA, start=1):
    return {
        "entryId": f"AF-{accession}-F1",
        "latestVersion": 6,
        "uniprotAccession": accession,
        "uniprotDescription": "Hemoglobin subunit alpha",
        "gene": "HBA1",
        "organismScientificName": "Homo sapiens",
        "globalMetricValue": 98.06,
        "sequence": sequence,
        "sequenceStart": start,
        "pdbUrl": f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb",
    }


class FakeWeb:
    """Answers by URL substring; records what was requested."""

    def __init__(self, answers):
        self.answers = answers
        self.requested = []

    def __call__(self, url):
        self.requested.append(url)
        for fragment, answer in self.answers.items():
            if fragment in url:
                if isinstance(answer, BaseException):
                    raise answer
                if not isinstance(answer, (str, bytes)):
                    answer = json.dumps(answer)
                return answer.encode() if isinstance(answer, str) else answer
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)


def standard_web(**overrides):
    answers = {
        "uniprotkb/search": uniprot_tsv(("A0A341AFD2", False, 142), ("P69905", True, 142)),
        "api/prediction/P69905": [afdb_entry("P69905")],
        "AF-P69905-F1-model_v6.pdb": PDB_TEXT,
    }
    answers.update(overrides)
    return FakeWeb(answers)


class ChecksumTests(unittest.TestCase):
    def test_crc64_matches_uniprot(self):
        self.assertEqual(crc64(HBA), "15E13666573BBBAE")

    def test_the_fingerprint_holds_checksums_and_length_only(self):
        print_ = Fingerprint.of(HBA)
        self.assertEqual(print_.md5, "6077c452d1dc6151040b2b179e2294c7")
        self.assertEqual(print_.length, 142)
        self.assertNotIn(HBA, repr(print_))


class AlphaFoldLookupTests(unittest.TestCase):
    def test_an_exact_match_returns_the_model_and_its_provenance(self):
        web = standard_web()
        result = AlphaFoldDB(web).lookup(Fingerprint.of(HBA))
        self.assertEqual(result.summary["status"], "available")
        self.assertEqual(result.summary["model_id"], "AF-P69905-F1")
        self.assertEqual(result.summary["accession"], "P69905")
        self.assertEqual(result.summary["residues"], 142)
        self.assertTrue(result.summary["entry_url"].endswith("/P69905"))
        self.assertEqual(result.model, PDB_TEXT)

    def test_the_sequence_never_leaves_the_server(self):
        web = standard_web()
        AlphaFoldDB(web).lookup(Fingerprint.of(HBA))
        self.assertTrue(web.requested)
        for url in web.requested:
            self.assertNotIn(HBA[:20], url)
        self.assertIn("15E13666573BBBAE", web.requested[0])

    def test_reviewed_entries_are_tried_before_unreviewed_ones(self):
        web = standard_web()
        AlphaFoldDB(web).lookup(Fingerprint.of(HBA))
        predictions = [u for u in web.requested if "api/prediction/" in u]
        self.assertTrue(predictions[0].endswith("/P69905"))

    def test_a_model_of_a_different_sequence_is_refused(self):
        mutant = HBA[:50] + "W" + HBA[51:]
        web = standard_web(**{"api/prediction/P69905": [afdb_entry("P69905", mutant)]})
        result = AlphaFoldDB(web).lookup(Fingerprint.of(HBA))
        self.assertEqual(result.summary["status"], "unavailable")
        self.assertIsNone(result.model)

    def test_a_fragment_model_is_not_used(self):
        web = standard_web(**{"api/prediction/P69905": [afdb_entry("P69905", start=2)]})
        result = AlphaFoldDB(web).lookup(Fingerprint.of(HBA))
        self.assertEqual(result.summary["status"], "unavailable")

    def test_a_sequence_missing_from_uniprot_says_so(self):
        web = standard_web(**{"uniprotkb/search": uniprot_tsv()})
        result = AlphaFoldDB(web).lookup(Fingerprint.of(HBA))
        self.assertEqual(result.summary["status"], "unavailable")
        self.assertIn("not in UniProt", result.summary["message"])

    def test_a_checksum_hit_of_another_length_is_ignored(self):
        web = standard_web(**{"uniprotkb/search": uniprot_tsv(("P69905", True, 141))})
        result = AlphaFoldDB(web).lookup(Fingerprint.of(HBA))
        self.assertIn("not in UniProt", result.summary["message"])

    def test_an_unmodelled_entry_falls_through_to_the_next(self):
        web = standard_web(**{
            "uniprotkb/search": uniprot_tsv(("Q00000", True, 142), ("P69905", True, 142)),
        })
        result = AlphaFoldDB(web).lookup(Fingerprint.of(HBA))
        self.assertEqual(result.summary["accession"], "P69905")

    def test_an_unreachable_source_is_reported_not_raised(self):
        web = standard_web(**{"uniprotkb/search": urllib.error.URLError("timed out")})
        result = AlphaFoldDB(web).lookup(Fingerprint.of(HBA))
        self.assertEqual(result.summary["status"], "unavailable")
        self.assertIn("could not be reached", result.summary["message"])

    def test_any_provider_fault_becomes_unavailable(self):
        class Broken:
            source = "alphafold_db"

            def lookup(self, fingerprint):
                raise RuntimeError("bug")

        self.assertEqual(
            lookup_structure(Broken(), Fingerprint.of(HBA)).summary["status"], "unavailable"
        )
        self.assertIn(
            "disabled", lookup_structure(None, Fingerprint.of(HBA)).summary["message"]
        )


def structure_service(web=None, **kwargs):
    return Service(
        predictor=MockPredictor(),
        synchronous=kwargs.pop("synchronous", True),
        log=lambda message: None,
        structure_provider=AlphaFoldDB(web or standard_web()),
        **kwargs,
    )


class StructureEndpointTests(unittest.TestCase):
    def submit(self, api, include=True):
        status, job = api.create_prediction(
            {"sequence": HBA, "options": {"include_structure": include}}
        )
        self.assertEqual(status, 202)
        return job

    def test_the_job_carries_a_summary_and_the_endpoint_the_model(self):
        api = structure_service()
        job = self.submit(api)
        self.assertEqual(job["structure"]["status"], "available")
        self.assertEqual(job["links"]["structure"], f"/api/v1/predictions/{job['job_id']}/structure")
        status, body = api.get_structure(job["job_id"])
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], {"format": "pdb", "data": PDB_TEXT})

    def test_the_job_itself_never_carries_the_model_or_the_sequence(self):
        api = structure_service()
        job = self.submit(api)
        text = json.dumps(api.get_prediction(job["job_id"])[1])
        self.assertNotIn(PDB_TEXT.strip(), text)
        self.assertNotIn(HBA, text)

    def test_a_job_without_a_structure_request_gets_404(self):
        api = structure_service()
        job = self.submit(api, include=False)
        self.assertNotIn("structure", job)
        self.assertEqual(api.get_structure(job["job_id"])[0], 404)

    def test_a_deleted_job_takes_its_structure_with_it(self):
        api = structure_service()
        job = self.submit(api)
        api.delete_prediction(job["job_id"])
        self.assertEqual(api.get_structure(job["job_id"])[0], 404)

    def test_health_names_the_structure_source(self):
        self.assertEqual(structure_service().health()[1]["structure_sources"], ["alphafold_db"])
        off = Service(predictor=MockPredictor(), log=lambda m: None, structure_provider=None)
        self.assertEqual(off.health()[1]["structure_sources"], [])

    def test_the_route_exists_and_only_for_get(self):
        api = structure_service()
        job = self.submit(api)
        path = f"/api/v1/predictions/{job['job_id']}/structure"
        self.assertEqual(route(api, "GET", path, None)[0], 200)
        self.assertEqual(route(api, "DELETE", path, None)[0], 405)

    def test_the_lookup_runs_beside_the_prediction_not_inside_it(self):
        release = threading.Event()

        class Slow:
            source = "alphafold_db"

            def lookup(self, fingerprint):
                release.wait(5)
                return AlphaFoldDB(standard_web()).lookup(fingerprint)

        api = Service(predictor=MockPredictor(), log=lambda m: None, structure_provider=Slow())
        job = self.submit(api)
        deadline = time.time() + 5
        while api.get_prediction(job["job_id"])[1]["status"] != "succeeded":
            self.assertLess(time.time(), deadline)
            time.sleep(0.02)
        # The prediction finished while the structure was still being looked up.
        self.assertEqual(api.get_structure(job["job_id"])[1]["structure"]["status"], "pending")
        release.set()
        while api.get_structure(job["job_id"])[1]["structure"]["status"] == "pending":
            self.assertLess(time.time(), deadline)
            time.sleep(0.02)
        self.assertEqual(api.get_structure(job["job_id"])[1]["structure"]["status"], "available")


if __name__ == "__main__":
    unittest.main()
