"""Tests for the CHAINSAW benchmark scoring (task T3.7).

The chopping parser gets the most attention: it is the one place where a
silent off-by-one would shift every residue of every comparison, and the
benchmark's whole point is that the numbers can be trusted beside a published
figure.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.train import benchmark as bench  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
PUBLISHED = os.path.join(
    REPO, "data/raw/benchmarks/chainsaw/chainsaw_model_v3_on_cath1363_test.csv"
)


class ChoppingTests(unittest.TestCase):
    def test_a_continuous_two_domain_chopping(self):
        self.assertEqual(
            bench.parse_chopping("0-259|260-442"),
            [[(0, 259)], [(260, 442)]],
        )

    def test_the_offset_converts_to_our_1_based_convention(self):
        # Chainsaw indexes from 0; the dataset indexes from 1. Getting this
        # wrong shifts every residue by one and is invisible in the averages.
        self.assertEqual(
            bench.parse_chopping("0-259|260-442", offset=1),
            [[(1, 260)], [(261, 443)]],
        )

    def test_an_underscore_marks_a_discontinuous_domain(self):
        self.assertEqual(
            bench.parse_chopping("0-76_185-227|238-244"),
            [[(0, 76), (185, 227)], [(238, 244)]],
        )

    def test_segments_come_back_sorted(self):
        self.assertEqual(
            bench.parse_chopping("185-227_0-76"), [[(0, 76), (185, 227)]]
        )

    def test_an_empty_chopping_is_no_domains_not_an_error(self):
        # They record a declined chain as an empty string.
        for value in ("", "   ", None):
            self.assertEqual(bench.parse_chopping(value), [])


class PerChainProportionTests(unittest.TestCase):
    def test_all_domains_matched(self):
        truth = [[(1, 50)], [(51, 100)]]
        self.assertEqual(bench.chain_proportion_correct(truth, truth), 1.0)

    def test_half_the_domains_matched(self):
        truth = [[(1, 50)], [(51, 100)]]
        predicted = [[(1, 50)], [(51, 60)]]
        self.assertEqual(bench.chain_proportion_correct(predicted, truth), 0.5)

    def test_a_chain_with_no_true_domains_is_undefined_not_zero(self):
        self.assertIsNone(bench.chain_proportion_correct([[(1, 10)]], []))

    def test_per_chain_averaging_differs_from_pooling(self):
        # Chainsaw averages per chain, so a 4-domain chain counts once. Pooling
        # over domains weights it four times. The two are genuinely different
        # numbers and the report carries both.
        rows = [
            bench.BenchmarkRow("a", 100, [[(1, 100)]], [[(1, 100)]]),
            bench.BenchmarkRow(
                "b", 100,
                [[(1, 25)], [(26, 50)], [(51, 75)], [(76, 100)]],
                [[(1, 100)]],
            ),
        ]
        scores = bench.score_rows(rows)
        self.assertEqual(scores["pct_correct_at_080_per_chain"], 0.5)
        self.assertNotEqual(
            scores["pct_correct_at_080"], scores["pct_correct_at_080_per_chain"]
        )


class GroupingTests(unittest.TestCase):
    def rows(self):
        return [
            bench.BenchmarkRow("s", 100, [[(1, 100)]], [[(1, 100)]]),
            bench.BenchmarkRow("m", 100, [[(1, 50)], [(51, 100)]], [[(1, 100)]]),
            bench.BenchmarkRow(
                "d", 120, [[(1, 40), (81, 120)], [(41, 80)]], [[(1, 120)]]
            ),
        ]

    def test_every_chain_is_counted_once_per_group_it_belongs_to(self):
        groups = bench.grouped_scores(self.rows())
        self.assertEqual(groups["all"]["chains"], 3)
        self.assertEqual(groups["single_domain"]["chains"], 1)
        self.assertEqual(groups["multidomain"]["chains"], 2)
        self.assertEqual(groups["discontinuous"]["chains"], 1)
        self.assertEqual(groups["continuous"]["chains"], 2)

    def test_an_empty_group_is_left_out_rather_than_reported_as_zero(self):
        rows = [bench.BenchmarkRow("s", 100, [[(1, 100)]], [[(1, 100)]])]
        groups = bench.grouped_scores(rows)
        self.assertNotIn("discontinuous", groups)

    def test_a_chain_with_no_prediction_is_counted(self):
        rows = [bench.BenchmarkRow("x", 100, [[(1, 100)]], [])]
        self.assertEqual(bench.score_rows(rows)["chains_with_no_prediction"], 1)


class GroundTruthAgreementTests(unittest.TestCase):
    def test_identical_partitions_agree_completely(self):
        ours = {"a": [[(1, 50)], [(51, 100)]]}
        theirs = {"a": [[(0, 49)], [(50, 99)]]}
        result = bench.ground_truth_agreement(ours, theirs)
        self.assertEqual(result["same_domain_count_pct"], 1.0)
        self.assertEqual(result["mean_shape_distance"], 0.0)

    def test_a_different_domain_count_is_reported(self):
        ours = {"a": [[(1, 100)]]}
        theirs = {"a": [[(0, 49)], [(50, 99)]]}
        result = bench.ground_truth_agreement(ours, theirs)
        self.assertEqual(result["same_domain_count"], 0)
        self.assertEqual(result["mean_signed_count_delta"], -1.0)

    def test_the_shape_distance_is_coordinate_free(self):
        # Same proportions, different absolute coordinates and lengths: a
        # 50/50 split is a 50/50 split in either space.
        ours = {"a": [[(1, 100)], [(101, 200)]]}
        theirs = {"a": [[(0, 9)], [(10, 19)]]}
        self.assertEqual(
            bench.ground_truth_agreement(ours, theirs)["mean_shape_distance"], 0.0
        )

    def test_no_shared_chains(self):
        self.assertEqual(
            bench.ground_truth_agreement({"a": [[(1, 2)]]}, {"b": [[(1, 2)]]}),
            {"chains": 0},
        )


@unittest.skipUnless(os.path.exists(PUBLISHED), "Chainsaw results not downloaded")
class PublishedFileTests(unittest.TestCase):
    """The real file, since the parser exists only to read it."""

    @classmethod
    def setUpClass(cls):
        cls.rows = bench.load_published(PUBLISHED)

    def test_the_whole_test_split_is_present(self):
        self.assertEqual(len(self.rows), 1363)

    def test_every_chopping_parses(self):
        for chain, row in self.rows.items():
            for field in ("true_chopping", "pred_chopping"):
                try:
                    bench.parse_chopping(row[field])
                except Exception as problem:  # noqa: BLE001
                    self.fail(f"{chain}/{field}={row[field]!r}: {problem!r}")

    def test_the_published_means_match_the_paper(self):
        # The paper reports an average IoU of 0.88 for Chainsaw on this split.
        means = bench.published_means(self.rows.values())
        self.assertAlmostEqual(means["iou"], 0.88, places=2)

    def test_discontinuous_ground_truth_exists_in_the_file(self):
        # If it did not, the discontinuous group would be silently empty.
        self.assertTrue(
            any("_" in row["true_chopping"] for row in self.rows.values())
        )


if __name__ == "__main__":
    unittest.main()
