"""Per-source parser tests, including the CATH coordinate convention."""

import gzip
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.data.schema import sequence_md5  # noqa: E402
from domaino.data.sources import cath, pdb_seqres, pfam, ted  # noqa: E402

CHOPPING = """\
101mA00\t1-154
10gsA01\t2-78,187-208
10gsA02\t79-186
9zzzA01\t1-50
"""

# Column order is CLF 2.0: id C A T H S35 S60 S95 S100 count length resolution.
DOMAIN_LIST = """\
# comment line that must be skipped
101mA00     1    10   490    10     1     1     1     1     1   154 2.070
10gsA01     3    40    30    10     1     1     1     1     1    99 2.200
10gsA02     1    20  1210    10     4     1     1     1     1   108 2.200
"""


def write(directory, name, text, binary=False):
    path = os.path.join(directory, name)
    if binary:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        with open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    return path


class CathParsingTests(unittest.TestCase):
    def test_chopping_parse_handles_continuous_and_discontinuous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "chop.txt", CHOPPING)
            parsed, malformed = cath.parse_chopping_file(path)
        self.assertEqual([s.as_tuple() for s in parsed["101mA00"]], [(1, 154)])
        self.assertEqual(
            [s.as_tuple() for s in parsed["10gsA01"]], [(2, 78), (187, 208)]
        )
        self.assertEqual(malformed, {})

    def test_inverted_interval_is_reported_not_fatal(self):
        # CATH 4.4.0 really contains ranges like 567-1 for 1ohfB01. The build
        # must survive them and say so.
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "chop.txt", "1ohfB01\t1-200,567-1\n1ohfB02\t15-1\n")
            segments, malformed = cath.parse_chopping_file(path)
        self.assertEqual([s.as_tuple() for s in segments["1ohfB01"]], [(1, 200)])
        self.assertEqual(malformed["1ohfB01"], ["567-1"])
        self.assertEqual(segments["1ohfB02"], [])
        self.assertEqual(malformed["1ohfB02"], ["15-1"])

    def test_malformed_chopping_blocks_the_record(self):
        from domaino.data.audit import is_usable

        with tempfile.TemporaryDirectory() as directory:
            chop = write(directory, "chop.txt", "1ohfB01\t1-50,567-1\n")
            lst = write(
                directory,
                "list.txt",
                "1ohfB01     1    10     8    10     1     1     1     1     1    50 2.000\n",
            )
            records = list(
                cath.load_records(
                    chopping_path=chop,
                    domain_list_path=lst,
                    chain_sequences={("1ohf", "B"): "A" * 600},
                )
            )
        self.assertIn("malformed_chopping:1ohfB01", records[0].flags)
        self.assertFalse(is_usable(records[0]))

    def test_domain_list_parse_extracts_labels_and_clusters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "list.txt", DOMAIN_LIST)
            parsed = cath.parse_domain_list(path)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed["10gsA01"]["cath"].code(), "3.40.30.10")
        self.assertEqual(parsed["10gsA01"]["clusters"]["s35"], 1)
        self.assertEqual(parsed["10gsA02"]["clusters"]["s35"], 4)
        self.assertEqual(parsed["101mA00"]["resolution"], 2.070)

    def test_domain_sequence_header_range_is_ignored(self):
        # The FASTA header carries PDB author numbering, which does not index
        # the sequence; only the domain id is taken from it.
        fasta = ">cath|4_4_0|10gsA01/2-78_187-208\nPPYTVVYF\n>cath|4_4_0|101mA00/0-153\nMVLSEG\n"
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "seqs.fa", fasta)
            parsed = cath.parse_domain_sequences(path)
        self.assertEqual(parsed["10gsA01"], "PPYTVVYF")
        self.assertEqual(parsed["101mA00"], "MVLSEG")

    def test_chain_key_splits_cath_chain_name(self):
        self.assertEqual(pdb_seqres.chain_key("1oaiA"), ("1oai", "A"))
        # A chain character of 0 is kept rather than guessed at.
        self.assertEqual(pdb_seqres.chain_key("1amg0"), ("1amg", "0"))


class CathRecordTests(unittest.TestCase):
    def build(self, sequences):
        with tempfile.TemporaryDirectory() as directory:
            chop = write(directory, "chop.txt", CHOPPING)
            lst = write(directory, "list.txt", DOMAIN_LIST)
            fasta = write(
                directory,
                "seqs.fa",
                ">cath|4_4_0|101mA00/0-153\n" + "M" * 154 + "\n"
                ">cath|4_4_0|10gsA01/2-78_187-208\n" + "G" * 99 + "\n"
                ">cath|4_4_0|10gsA02/79-186\n" + "G" * 108 + "\n",
            )
            return list(
                cath.load_records(
                    chopping_path=chop,
                    domain_list_path=lst,
                    chain_sequences=sequences,
                    domain_sequences=cath.parse_domain_sequences(fasta),
                )
            )

    def test_groups_domains_into_one_record_per_chain(self):
        records = self.build(
            {("101m", "A"): "M" * 154, ("10gs", "A"): "G" * 209}
        )
        by_id = {r.chain_id: r for r in records}
        self.assertEqual(sorted(by_id), ["101mA", "10gsA", "9zzzA"])
        self.assertEqual(by_id["10gsA"].n_domains, 2)
        self.assertTrue(by_id["10gsA"].has_discontinuous_domain)

    def test_domains_are_ordered_by_first_segment(self):
        records = self.build({("10gs", "A"): "G" * 209})
        chain = next(r for r in records if r.chain_id == "10gsA")
        self.assertEqual(
            [d.domain_id for d in chain.domains], ["10gsA01", "10gsA02"]
        )

    def test_cath_supervision_mask_declares_a_complete_partition(self):
        records = self.build({("101m", "A"): "M" * 154})
        record = next(r for r in records if r.chain_id == "101mA")
        self.assertTrue(record.supervision.complete_partition)
        self.assertTrue(record.supervision.pairwise)
        self.assertTrue(record.supervision.cath_labels)
        self.assertEqual(record.coverage, 1.0)

    def test_exact_agreement_is_recorded(self):
        records = self.build({("101m", "A"): "M" * 154})
        record = next(r for r in records if r.chain_id == "101mA")
        self.assertEqual(record.domains[0].quality["agreement"], "exact")
        self.assertEqual(record.sequence_md5, sequence_md5("M" * 154))

    def test_whole_chain_domain_published_as_full_seqres_is_not_a_mismatch(self):
        # A 00 domain whose published sequence is the entire SEQRES while the
        # chopping covers only the assigned residues: a known CATH quirk, not
        # corruption, so it must not block the record.
        with tempfile.TemporaryDirectory() as directory:
            chop = write(directory, "chop.txt", "102lA00\t1-163\n")
            lst = write(
                directory,
                "list.txt",
                "102lA00     1    10     8    10     1     1     1     1     1   163 1.740\n",
            )
            fasta = write(
                directory, "seqs.fa", ">cath|4_4_0|102lA00/1-162\n" + "K" * 165 + "\n"
            )
            records = list(
                cath.load_records(
                    chopping_path=chop,
                    domain_list_path=lst,
                    chain_sequences={("102l", "A"): "K" * 165},
                    domain_sequences=cath.parse_domain_sequences(fasta),
                )
            )
        self.assertEqual(
            records[0].domains[0].quality["agreement"], "published_is_full_chain"
        )
        self.assertNotIn("domain_sequence_mismatch", records[0].flags)

    def test_missing_chain_sequence_is_flagged_not_dropped(self):
        records = self.build({})
        record = next(r for r in records if r.chain_id == "101mA")
        self.assertIsNone(record.sequence)
        self.assertIn("chain_sequence_unavailable", record.flags)

    def test_unclassified_chain_keeps_boundaries_but_loses_label_supervision(self):
        records = self.build({("9zzz", "A"): "Q" * 60})
        record = next(r for r in records if r.chain_id == "9zzzA")
        self.assertFalse(record.supervision.cath_labels)
        self.assertTrue(record.supervision.residue_labels)
        self.assertIn("no_classified_domain", record.flags)
        self.assertEqual(record.s35_keys(), [])


class UsabilityTests(unittest.TestCase):
    """Training usability and evaluation usability are different questions."""

    def record(self, flags, sequence="A" * 50):
        from domaino.data.schema import ChainRecord, Domain, Segment, Supervision

        return ChainRecord(
            record_id="cath:testA",
            source="cath",
            source_version="4.4.0",
            chain_id="testA",
            length=50,
            domains=[Domain(domain_id="testA00", segments=[Segment(1, 50)])],
            supervision=Supervision(
                residue_labels=True,
                pairwise=True,
                cath_labels=False,
                complete_partition=True,
                weight=1.0,
            ),
            sequence=sequence,
            flags=list(flags),
        )

    def test_label_disagreement_blocks_training_but_not_evaluation(self):
        from domaino.data.audit import has_valid_sequence, is_usable

        record = self.record(["domain_sequence_mismatch"])
        self.assertFalse(is_usable(record))
        self.assertTrue(has_valid_sequence(record))

    def test_missing_sequence_blocks_both(self):
        from domaino.data.audit import has_valid_sequence, is_usable

        record = self.record(["chain_sequence_unavailable"], sequence=None)
        self.assertFalse(is_usable(record))
        self.assertFalse(has_valid_sequence(record))

    def test_clean_record_is_usable_both_ways(self):
        from domaino.data.audit import has_valid_sequence, is_usable

        record = self.record([])
        self.assertTrue(is_usable(record))
        self.assertTrue(has_valid_sequence(record))

    def test_flag_kind_strips_the_domain_suffix(self):
        from domaino.data.audit import flag_kind

        self.assertEqual(flag_kind("segment_out_of_bounds:1abcA01"), "segment_out_of_bounds")
        self.assertEqual(flag_kind("schema:overlapping_domains"), "schema:overlapping_domains")
        self.assertEqual(flag_kind("no_classified_domain"), "no_classified_domain")


class TedTests(unittest.TestCase):
    def rows(self, *levels):
        return [
            {
                "ted_id": f"AF-X-F1-model_v4_TED0{index}",
                "chain_id": "AF-X-F1-model_v4",
                "consensus_level": level,
                "chopping": chopping,
                "n_residues": "100",
                "plddt": "95.0",
                "cath_code": code,
                "cath_depth": "H" if code else "-",
                "assignment_method": "foldseek",
            }
            for index, (level, chopping, code) in enumerate(levels, start=1)
        ]

    def test_high_consensus_chain_supports_pairwise_supervision(self):
        records = list(
            ted.build_records(
                self.rows(("high", "11-41_290-389", "3.90.1150.10"), ("high", "54-288", None)),
                chain_lengths={"AF-X-F1-model_v4": 394},
            )
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertTrue(record.supervision.pairwise)
        self.assertEqual(record.supervision.weight, ted.CONSENSUS_WEIGHTS["high"])
        self.assertTrue(record.has_discontinuous_domain)
        self.assertTrue(record.supervision.cath_labels)

    def test_low_consensus_chain_loses_pairwise_supervision(self):
        records = list(
            ted.build_records(
                self.rows(("high", "1-50", None), ("low", "60-100", None)),
                chain_lengths={"AF-X-F1-model_v4": 120},
            )
        )
        record = records[0]
        self.assertFalse(record.supervision.pairwise)
        self.assertFalse(record.supervision.complete_partition)
        self.assertEqual(record.supervision.weight, ted.CONSENSUS_WEIGHTS["low"])

    def test_sequence_is_attached_by_md5_when_available(self):
        sequence = "W" * 394
        records = list(
            ted.build_records(
                self.rows(("high", "11-41_290-389", None)),
                chain_lengths={"AF-X-F1-model_v4": 394},
                chain_md5={"AF-X-F1-model_v4": sequence_md5(sequence)},
                sequences_by_md5={sequence_md5(sequence): sequence},
            )
        )
        self.assertEqual(records[0].sequence, sequence)
        self.assertNotIn("sequence_unavailable", records[0].flags)

    def test_missing_sequence_is_flagged(self):
        records = list(
            ted.build_records(
                self.rows(("high", "1-50", None)),
                chain_lengths={"AF-X-F1-model_v4": 60},
            )
        )
        self.assertIn("sequence_unavailable", records[0].flags)

    def test_truncated_cath_code_keeps_its_real_depth(self):
        records = list(
            ted.build_records(
                self.rows(("medium", "1-50", "1.10.287")),
                chain_lengths={"AF-X-F1-model_v4": 60},
            )
        )
        self.assertEqual(records[0].domains[0].label_depth, "T")


class PfamTests(unittest.TestCase):
    HEADER = "\t".join(pfam.EXPECTED_HEADER) + "\n"

    def regions_file(self, directory, rows):
        return write(directory, "regions.tsv.gz", self.HEADER + rows, binary=True)

    def test_rejects_an_unexpected_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "bad.tsv.gz", "a\tb\tc\n", binary=True)
            with self.assertRaises(ValueError):
                list(pfam.iter_regions(path))

    def test_md5_filter_keeps_only_requested_sequences(self):
        rows = (
            "A0\t1\tCRC\tmd5aaa\tPF00001\t10\t50\t12\t48\n"
            "B0\t1\tCRC\tmd5bbb\tPF00002\t5\t40\t6\t39\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self.regions_file(directory, rows)
            kept = list(pfam.iter_regions(path, keep_md5={"md5bbb"}))
        self.assertEqual([row["pfamseq_acc"] for row in kept], ["B0"])

    def test_records_carry_a_mask_that_forbids_negative_linker_labels(self):
        sequence = "P" * 100
        md5 = sequence_md5(sequence)
        grouped = {
            md5: [
                {
                    "pfamseq_acc": "A0",
                    "md5": md5,
                    "pfamA_acc": "PF00001",
                    "seq_start": "10",
                    "seq_end": "50",
                    "ali_start": "12",
                    "ali_end": "48",
                }
            ]
        }
        records = list(pfam.build_records(grouped, {md5: sequence}))
        self.assertEqual(len(records), 1)
        mask = records[0].supervision
        self.assertFalse(mask.complete_partition)
        self.assertFalse(mask.pairwise)
        self.assertFalse(mask.cath_labels)
        self.assertEqual(records[0].domains[0].quality["pfam_family"], "PF00001")

    def test_region_beyond_the_sequence_is_flagged(self):
        sequence = "P" * 30
        md5 = sequence_md5(sequence)
        grouped = {
            md5: [
                {
                    "pfamseq_acc": "A0",
                    "md5": md5,
                    "pfamA_acc": "PF00001",
                    "seq_start": "10",
                    "seq_end": "500",
                    "ali_start": "10",
                    "ali_end": "500",
                },
                {
                    "pfamseq_acc": "A0",
                    "md5": md5,
                    "pfamA_acc": "PF00002",
                    "seq_start": "1",
                    "seq_end": "20",
                    "ali_start": "1",
                    "ali_end": "20",
                },
            ]
        }
        records = list(pfam.build_records(grouped, {md5: sequence}))
        self.assertIn("region_out_of_bounds:PF00001", records[0].flags)
        self.assertEqual(records[0].n_domains, 1)


if __name__ == "__main__":
    unittest.main()
