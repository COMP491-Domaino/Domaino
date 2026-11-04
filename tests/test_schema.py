"""Schema and coordinate-convention tests (milestone M1.2)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.data.schema import (  # noqa: E402
    CathLabel,
    ChainRecord,
    Domain,
    Segment,
    Supervision,
    iou,
    read_jsonl,
    sequence_md5,
    write_jsonl,
)


def cath_supervision(**overrides):
    defaults = dict(
        residue_labels=True,
        pairwise=True,
        cath_labels=True,
        complete_partition=True,
        weight=1.0,
    )
    defaults.update(overrides)
    return Supervision(**defaults)


class SegmentTests(unittest.TestCase):
    def test_rejects_inverted_and_zero_based_intervals(self):
        with self.assertRaises(ValueError):
            Segment(10, 4)
        with self.assertRaises(ValueError):
            Segment(0, 10)

    def test_length_is_inclusive(self):
        self.assertEqual(Segment(1, 154).length, 154)
        self.assertEqual(Segment(7, 7).length, 1)

    def test_parses_cath_discontinuous_chopping(self):
        segments = Segment.parse_many("2-78,187-208")
        self.assertEqual([s.as_tuple() for s in segments], [(2, 78), (187, 208)])
        self.assertEqual(sum(s.length for s in segments), 99)

    def test_parses_ted_chopping_with_underscores(self):
        segments = Segment.parse_many("11-41_290-389", segment_sep="_")
        self.assertEqual([s.as_tuple() for s in segments], [(11, 41), (290, 389)])

    def test_overlap_detection(self):
        self.assertTrue(Segment(1, 10).overlaps(Segment(10, 20)))
        self.assertFalse(Segment(1, 10).overlaps(Segment(11, 20)))


class CathLabelTests(unittest.TestCase):
    def test_full_label_depth_and_code(self):
        label = CathLabel(3, 40, 640, 10)
        self.assertEqual(label.depth, "H")
        self.assertEqual(label.code(), "3.40.640.10")
        self.assertEqual(label.code("T"), "3.40.640")
        self.assertEqual(label.code("C"), "3")

    def test_truncated_label_reports_shallower_depth(self):
        label = CathLabel.parse("1.10.287")
        self.assertEqual(label.depth, "T")
        self.assertIsNone(label.h)
        # Asking for H on a T-level label must not invent a fourth level.
        self.assertEqual(label.code("H"), "1.10.287")

    def test_unlabelled(self):
        self.assertEqual(CathLabel().depth, "none")
        self.assertIsNone(CathLabel().code())


class SupervisionTests(unittest.TestCase):
    def test_pairwise_requires_a_complete_partition(self):
        # This is the Pfam guard: regions do not tile a sequence, so residues
        # in different regions cannot be treated as a negative pair.
        with self.assertRaises(ValueError):
            Supervision(
                residue_labels=True,
                pairwise=True,
                cath_labels=False,
                complete_partition=False,
                weight=0.3,
            )

    def test_pfam_style_mask_is_allowed(self):
        mask = Supervision(
            residue_labels=True,
            pairwise=False,
            cath_labels=False,
            complete_partition=False,
            weight=0.3,
        )
        self.assertFalse(mask.pairwise)


class ChainRecordTests(unittest.TestCase):
    def make_record(self, sequence="A" * 220):
        return ChainRecord(
            record_id="cath:10gsA",
            source="cath",
            source_version="4.4.0",
            chain_id="10gsA",
            length=len(sequence),
            domains=[
                Domain(
                    domain_id="10gsA01",
                    segments=[Segment(2, 78), Segment(187, 208)],
                    cath=CathLabel(3, 40, 30, 10),
                    clusters={"s35": 1, "s60": 1, "s95": 1, "s100": 1},
                ),
                Domain(
                    domain_id="10gsA02",
                    segments=[Segment(79, 186)],
                    cath=CathLabel(1, 20, 1210, 10),
                    clusters={"s35": 4, "s60": 1, "s95": 1, "s100": 1},
                ),
            ],
            supervision=cath_supervision(),
            sequence=sequence,
            sequence_md5=sequence_md5(sequence),
        )

    def test_clean_record_validates(self):
        self.assertEqual(self.make_record().validate(), [])

    def test_discontinuous_and_multidomain_detection(self):
        record = self.make_record()
        self.assertTrue(record.is_multidomain)
        self.assertTrue(record.has_discontinuous_domain)
        self.assertEqual(record.domains[0].n_segments, 2)
        self.assertEqual(record.covered_residues, 99 + 108)

    def test_segment_beyond_sequence_is_caught(self):
        record = self.make_record(sequence="A" * 100)
        self.assertIn("segment_out_of_bounds:10gsA01", record.validate())

    def test_length_sequence_disagreement_is_caught(self):
        record = self.make_record()
        record.length = 999
        self.assertIn("length_sequence_mismatch", record.validate())

    def test_overlapping_domains_are_caught(self):
        record = self.make_record()
        record.domains[1] = Domain(
            domain_id="10gsA02", segments=[Segment(50, 186)], cath=CathLabel(1, 2, 3, 4)
        )
        self.assertIn("overlapping_domains", record.validate())
        self.assertEqual(record.overlapping_domains(), [("10gsA01", "10gsA02")])

    def test_cath_supervision_without_labels_is_caught(self):
        record = self.make_record()
        record.domains = [
            Domain(domain_id="10gsA01", segments=[Segment(1, 10)], cath=None)
        ]
        self.assertIn("cath_supervision_without_labels", record.validate())

    def test_non_standard_residues_are_flagged_not_dropped(self):
        record = self.make_record(sequence="X" * 220)
        self.assertIn("ambiguous_residues", record.validate())

    def test_residue_labels_mark_linker_as_zero(self):
        sequence = "A" * 20
        record = ChainRecord(
            record_id="cath:testA",
            source="cath",
            source_version="4.4.0",
            chain_id="testA",
            length=20,
            domains=[
                Domain(domain_id="testA01", segments=[Segment(1, 5), Segment(16, 20)]),
                Domain(domain_id="testA02", segments=[Segment(8, 12)]),
            ],
            supervision=cath_supervision(cath_labels=False),
            sequence=sequence,
        )
        labels = record.residue_labels()
        self.assertEqual(labels[0:5], [1, 1, 1, 1, 1])
        self.assertEqual(labels[5:7], [0, 0])
        self.assertEqual(labels[7:12], [2, 2, 2, 2, 2])
        self.assertEqual(labels[15:20], [1, 1, 1, 1, 1])

    def test_domain_sequence_concatenates_segments_in_order(self):
        sequence = "".join(str(i % 10) for i in range(1, 21))
        record = ChainRecord(
            record_id="cath:testA",
            source="cath",
            source_version="4.4.0",
            chain_id="testA",
            length=20,
            domains=[
                Domain(domain_id="testA01", segments=[Segment(1, 3), Segment(18, 20)])
            ],
            supervision=cath_supervision(cath_labels=False),
            sequence=sequence,
        )
        self.assertEqual(record.domain_sequence(record.domains[0]), "123890")

    def test_s35_keys_combine_superfamily_and_cluster(self):
        record = self.make_record()
        self.assertEqual(record.s35_keys(), ["1.20.1210.10.4", "3.40.30.10.1"])

    def test_jsonl_round_trip_preserves_coordinates_and_masks(self):
        record = self.make_record()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "records.jsonl.gz")
            self.assertEqual(write_jsonl(path, [record]), 1)
            restored = list(read_jsonl(path))[0]
        self.assertEqual(restored.record_id, record.record_id)
        self.assertEqual(
            [s.as_tuple() for s in restored.domains[0].segments],
            [(2, 78), (187, 208)],
        )
        self.assertEqual(restored.domains[0].cath, CathLabel(3, 40, 30, 10))
        self.assertEqual(restored.supervision, record.supervision)
        self.assertEqual(restored.s35_keys(), record.s35_keys())

    def test_serialised_form_carries_readable_label_fields(self):
        payload = json.loads(json.dumps(self.make_record().to_dict()))
        self.assertEqual(payload["domains"][0]["cath_code"], "3.40.30.10")
        self.assertEqual(payload["domains"][0]["label_depth"], "H")
        self.assertEqual(payload["domains"][0]["segments"], [[2, 78], [187, 208]])


class IouTests(unittest.TestCase):
    def test_identical_intervals(self):
        self.assertEqual(iou([(1, 10)], [(1, 10)]), 1.0)

    def test_disjoint_intervals(self):
        self.assertEqual(iou([(1, 10)], [(11, 20)]), 0.0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(iou([(1, 10)], [(6, 15)]), 5 / 15)

    def test_discontinuous_against_continuous(self):
        self.assertAlmostEqual(iou([(1, 5), (11, 15)], [(1, 15)]), 10 / 15)


if __name__ == "__main__":
    unittest.main()
