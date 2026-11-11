"""Tests for the proposal's segmentation and classification measures."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino import metrics  # noqa: E402


class IouTests(unittest.TestCase):
    def test_identical_and_disjoint(self):
        self.assertEqual(metrics.iou([(1, 10)], [(1, 10)]), 1.0)
        self.assertEqual(metrics.iou([(1, 10)], [(11, 20)]), 0.0)

    def test_discontinuous_domain(self):
        self.assertAlmostEqual(metrics.iou([(1, 5), (11, 15)], [(1, 15)]), 10 / 15)


class MatchingTests(unittest.TestCase):
    def test_matching_ignores_domain_numbering(self):
        # The model has no reason to order domains as CATH does, so matching
        # must pair them by overlap, not by index.
        predicted = [[(60, 100)], [(1, 50)]]
        truth = [[(1, 50)], [(60, 100)]]
        matches = metrics.match_domains(predicted, truth)
        self.assertEqual({(p, t) for p, t, _ in matches}, {(0, 1), (1, 0)})
        self.assertTrue(all(score == 1.0 for _, _, score in matches))

    def test_matching_is_one_to_one(self):
        predicted = [[(1, 100)]]
        truth = [[(1, 50)], [(51, 100)]]
        matches = metrics.match_domains(predicted, truth)
        self.assertEqual(len(matches), 1)

    def test_non_overlapping_domains_are_not_matched(self):
        self.assertEqual(metrics.match_domains([[(1, 10)]], [[(20, 30)]]), [])


class BoundaryErrorTests(unittest.TestCase):
    def test_exact_boundaries_score_zero(self):
        self.assertEqual(metrics.boundary_error([[(1, 50)], [(51, 100)]],
                                                [[(1, 50)], [(51, 100)]]), 0.0)

    def test_offset_boundaries_measure_distance(self):
        # True edges {1,50,51,100}, predicted {1,48,49,100}. Nearest-predicted
        # distances are 0, 1 (50->49), 2 (51->49) and 0, so the mean is 0.75.
        self.assertAlmostEqual(
            metrics.boundary_error([[(1, 48)], [(49, 100)]],
                                   [[(1, 50)], [(51, 100)]]),
            0.75,
        )

    def test_undefined_when_a_side_is_empty(self):
        self.assertIsNone(metrics.boundary_error([], [[(1, 10)]]))


class SegmentationScoreTests(unittest.TestCase):
    def test_perfect_prediction(self):
        truth = [[[(1, 50)], [(51, 100)]]]
        scores = metrics.score_segmentation(truth, truth)
        self.assertEqual(scores.mean_iou, 1.0)
        self.assertEqual(scores.pct_correct_at_080, 1.0)
        self.assertEqual(scores.domain_count_mae, 0.0)
        self.assertEqual(scores.domain_count_exact, 1.0)

    def test_missing_a_domain_is_penalised_not_ignored(self):
        # One domain predicted where there are two: the unmatched true domain
        # must drag the chain's IoU down.
        scores = metrics.score_segmentation(
            [[[(1, 50)]]], [[[(1, 50)], [(51, 100)]]]
        )
        self.assertAlmostEqual(scores.mean_iou, 0.5)
        self.assertEqual(scores.correctly_parsed, 1)
        self.assertEqual(scores.pct_correct_at_080, 0.5)
        self.assertEqual(scores.domain_count_mae, 1.0)

    def test_iou_threshold_is_strict_at_080(self):
        # IoU 0.75 must not count as correctly parsed.
        scores = metrics.score_segmentation([[[(1, 75)]]], [[[(1, 100)]]])
        self.assertEqual(scores.correctly_parsed, 0)

    def test_counts_are_totalled(self):
        scores = metrics.score_segmentation(
            [[[(1, 100)]], [[(1, 50)], [(51, 100)]]],
            [[[(1, 100)]], [[(1, 100)]]],
        )
        self.assertEqual(scores.chains, 2)
        self.assertEqual(scores.domains_true, 2)
        self.assertEqual(scores.domains_predicted, 3)


class MacroF1Tests(unittest.TestCase):
    def test_perfect_prediction(self):
        acc, macro, per = metrics.macro_f1(["a", "b"], ["a", "b"])
        self.assertEqual((acc, macro), (1.0, 1.0))
        self.assertEqual(per, {"a": 1.0, "b": 1.0})

    def test_majority_guessing_shows_high_accuracy_but_low_macro_f1(self):
        # This is the imbalance trap the proposal warns about: always
        # predicting the common class looks accurate and is useless.
        truth = ["a"] * 9 + ["b"]
        predicted = ["a"] * 10
        acc, macro, per = metrics.macro_f1(truth, predicted)
        self.assertEqual(acc, 0.9)
        self.assertLess(macro, 0.5)
        self.assertEqual(per["b"], 0.0)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.macro_f1(["a"], ["a", "b"])

    def test_empty_input(self):
        self.assertEqual(metrics.macro_f1([], []), (0.0, 0.0, {}))


class ClassificationScoreTests(unittest.TestCase):
    def test_levels_are_reported_separately(self):
        truth = {"C": ["1", "3"], "H": ["1.10.10.10", "3.40.50.300"]}
        pred = {"C": ["1", "3"], "H": ["1.10.10.10", "9.99.99.99"]}
        scores = metrics.score_classification(truth, pred)
        self.assertEqual(scores.levels["C"]["accuracy"], 1.0)
        self.assertEqual(scores.levels["H"]["accuracy"], 0.5)
        self.assertEqual(scores.domains, 2)

    def test_absent_level_is_skipped(self):
        scores = metrics.score_classification({"C": ["1"]}, {"C": ["1"]})
        self.assertNotIn("H", scores.levels)


class StratifyTests(unittest.TestCase):
    def test_groups_preserve_pairing(self):
        groups = metrics.stratify(
            ["single", "multi", "single"], ["p1", "p2", "p3"], ["t1", "t2", "t3"]
        )
        self.assertEqual(groups["single"], (["p1", "p3"], ["t1", "t3"]))
        self.assertEqual(groups["multi"], (["p2"], ["t2"]))


if __name__ == "__main__":
    unittest.main()
