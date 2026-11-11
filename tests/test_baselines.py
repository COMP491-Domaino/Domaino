"""Tests for the minimum reference baselines (T2.1)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.model import baselines as bl  # noqa: E402


class SingleDomainTests(unittest.TestCase):
    def test_predicts_the_whole_chain(self):
        model = bl.SingleDomainBaseline().fit()
        self.assertEqual(model.predict(154), [(1, 154)])

    def test_rejects_a_non_positive_length(self):
        with self.assertRaises(ValueError):
            bl.SingleDomainBaseline().predict(0)

    def test_predict_many_wraps_each_chain_as_one_domain(self):
        self.assertEqual(
            bl.SingleDomainBaseline().predict_many([10, 20]),
            [[[(1, 10)]], [[(1, 20)]]],
        )

    def test_it_is_a_strong_floor_on_single_domain_chains(self):
        # Most CATH chains are single-domain, which is exactly why this
        # baseline exists: a model must beat it to be worth anything.
        from domaino import metrics

        truth = [[[(1, 100)]], [[(1, 200)]]]
        predicted = bl.SingleDomainBaseline().predict_many([100, 200])
        self.assertEqual(metrics.score_segmentation(predicted, truth).mean_iou, 1.0)


class BoundaryTargetTests(unittest.TestCase):
    def test_internal_edges_are_labelled(self):
        labels = bl.boundary_targets(100, [[(1, 50)], [(51, 100)]], tolerance=0)
        self.assertEqual(labels[49], 1)  # residue 50, end of domain one
        self.assertEqual(labels[50], 1)  # residue 51, start of domain two
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[99], 0)

    def test_chain_termini_are_not_labelled(self):
        # Residue 1 and residue L are boundaries of every chain, so training
        # on them teaches only where the sequence stops.
        labels = bl.boundary_targets(100, [[(1, 100)]], tolerance=0)
        self.assertEqual(sum(labels), 0)

    def test_tolerance_widens_the_positive_band(self):
        labels = bl.boundary_targets(100, [[(1, 50)], [(51, 100)]], tolerance=2)
        self.assertEqual(labels[47:53], [1, 1, 1, 1, 1, 1])

    def test_discontinuous_domain_contributes_every_segment_edge(self):
        labels = bl.boundary_targets(
            220, [[(2, 78), (187, 208)], [(79, 186)]], tolerance=0
        )
        for residue in (2, 78, 79, 186, 187, 208):
            self.assertEqual(labels[residue - 1], 1, f"residue {residue}")


class DecodeTests(unittest.TestCase):
    def test_no_peak_gives_one_domain(self):
        self.assertEqual(bl.decode_boundaries([0.0] * 100), [(1, 100)])

    def test_a_clear_peak_splits_the_chain(self):
        scores = [0.0] * 200
        scores[99] = 0.9
        self.assertEqual(
            bl.decode_boundaries(scores, min_domain_length=40), [(1, 100), (101, 200)]
        )

    def test_intervals_tile_the_chain_without_gaps_or_overlap(self):
        scores = [0.0] * 300
        for position in (99, 199):
            scores[position] = 0.9
        intervals = bl.decode_boundaries(scores, min_domain_length=40)
        self.assertEqual(intervals[0][0], 1)
        self.assertEqual(intervals[-1][1], 300)
        for (_, end), (next_start, _) in zip(intervals, intervals[1:]):
            self.assertEqual(next_start, end + 1)

    def test_min_domain_length_suppresses_shredding(self):
        # Without the guard a noisy score produces many tiny fragments and the
        # IoU collapses.
        scores = [0.9 if i % 3 == 0 else 0.0 for i in range(200)]
        intervals = bl.decode_boundaries(scores, min_domain_length=40)
        for start, end in intervals:
            self.assertGreaterEqual(end - start + 1, 40)

    def test_scores_below_threshold_are_ignored(self):
        scores = [0.0] * 200
        scores[99] = 0.4
        self.assertEqual(bl.decode_boundaries(scores, threshold=0.5), [(1, 200)])

    def test_empty_input(self):
        self.assertEqual(bl.decode_boundaries([]), [])


class BoundaryBaselineTests(unittest.TestCase):
    def synthetic(self, chains=12, length=200, dim=6):
        """Chains whose boundary residues carry a distinctive feature."""
        import numpy as np

        rng = np.random.default_rng(0)
        features, labels, truths = [], [], []
        for _ in range(chains):
            cut = int(rng.integers(60, length - 60))
            domains = [[(1, cut)], [(cut + 1, length)]]
            y = bl.boundary_targets(length, domains, tolerance=2)
            x = rng.normal(0, 0.1, size=(length, dim)).astype(np.float32)
            # Feature 0 marks the boundary band, so the task is learnable.
            x[:, 0] += np.asarray(y, dtype=np.float32)
            features.append(x)
            labels.append(y)
            truths.append(domains)
        return features, labels, truths

    def test_training_reduces_loss(self):
        features, labels, _ = self.synthetic()
        model = bl.BoundaryBaseline(dim=6, epochs=5).fit(features, labels)
        self.assertLess(model.history[-1]["loss"], model.history[0]["loss"])

    def test_predicts_valid_intervals(self):
        features, labels, _ = self.synthetic()
        model = bl.BoundaryBaseline(dim=6, epochs=5).fit(features, labels)
        intervals = model.predict(features[0])
        self.assertEqual(intervals[0][0], 1)
        self.assertEqual(intervals[-1][1], 200)
        for start, end in intervals:
            self.assertLessEqual(start, end)

    def test_it_learns_a_learnable_boundary(self):
        from domaino import metrics

        features, labels, truths = self.synthetic(chains=24)
        model = bl.BoundaryBaseline(dim=6, epochs=25, learning_rate=0.5)
        model.fit(features, labels)
        # predict() returns contiguous domains, so each becomes a one-segment
        # domain for scoring.
        predicted = [
            [[interval] for interval in model.predict(f)] for f in features
        ]
        scores = metrics.score_segmentation(predicted, truths)
        self.assertGreater(scores.mean_iou, 0.6)

    def test_predicting_before_fitting_is_an_error(self):
        import numpy as np

        with self.assertRaises(RuntimeError):
            bl.BoundaryBaseline(dim=6).predict(np.zeros((10, 6), dtype=np.float32))

    def test_width_mismatch_is_rejected(self):
        import numpy as np

        model = bl.BoundaryBaseline(dim=4)
        with self.assertRaises(ValueError):
            model.fit([np.zeros((10, 6), dtype=np.float32)], [[0] * 10])


class PoolingTests(unittest.TestCase):
    def test_pools_a_contiguous_segment(self):
        import numpy as np

        embedding = np.arange(20, dtype=np.float32).reshape(10, 2)
        pooled = bl.pool_segments(embedding, [(1, 10)])
        np.testing.assert_allclose(pooled, embedding.mean(axis=0))

    def test_pools_across_a_discontinuous_domain(self):
        # One vector for a domain whose residues are split in sequence but
        # joined in space.
        import numpy as np

        embedding = np.zeros((10, 2), dtype=np.float32)
        embedding[0:2] = 1.0
        embedding[8:10] = 3.0
        pooled = bl.pool_segments(embedding, [(1, 2), (9, 10)])
        np.testing.assert_allclose(pooled, [2.0, 2.0])

    def test_empty_selection_is_rejected(self):
        import numpy as np

        with self.assertRaises(ValueError):
            bl.pool_segments(np.zeros((5, 2), dtype=np.float32), [])


class PooledClassifierTests(unittest.TestCase):
    def labelled(self, per_class=24, dim=5):
        """Two well-separated CATH classes."""
        import numpy as np

        rng = np.random.default_rng(1)
        rows, c, a, t, h = [], [], [], [], []
        specs = [
            (("1", "1.10", "1.10.10", "1.10.10.10"), +1.0),
            (("3", "3.40", "3.40.50", "3.40.50.300"), -1.0),
        ]
        for codes, offset in specs:
            block = rng.normal(offset, 0.2, size=(per_class, dim)).astype(np.float32)
            rows.append(block)
            c += [codes[0]] * per_class
            a += [codes[1]] * per_class
            t += [codes[2]] * per_class
            h += [codes[3]] * per_class
        return np.concatenate(rows, axis=0), {"C": c, "A": a, "T": t, "H": h}

    def test_learns_separable_classes(self):
        from domaino import metrics

        x, labels = self.labelled()
        model = bl.PooledDomainClassifier(dim=5, epochs=40).fit(x, labels)
        predicted = model.predict(x)
        scores = metrics.score_classification(labels, predicted)
        for level in ("C", "A", "T", "H"):
            self.assertGreater(scores.levels[level]["accuracy"], 0.9, level)

    def test_every_level_gets_a_head(self):
        x, labels = self.labelled()
        model = bl.PooledDomainClassifier(dim=5, epochs=5).fit(x, labels)
        self.assertEqual(set(model.classes), {"C", "A", "T", "H"})

    def test_hierarchical_output_cannot_contradict_itself(self):
        # A standalone H guess may disagree with the C guess above it, which
        # CATH cannot express; the forced version must stay a real path.
        x, labels = self.labelled()
        model = bl.PooledDomainClassifier(dim=5, epochs=40).fit(x, labels)
        fixed = model.predict_hierarchical(x)
        for row in range(len(x)):
            c, a, t, h = (fixed[l][row] for l in ("C", "A", "T", "H"))
            self.assertTrue(a == c or a.startswith(c + "."), f"{c} vs {a}")
            self.assertTrue(t == a or t.startswith(a + "."), f"{a} vs {t}")
            self.assertTrue(h == t or h.startswith(t + "."), f"{t} vs {h}")

    def test_label_count_mismatch_is_rejected(self):
        import numpy as np

        model = bl.PooledDomainClassifier(dim=5)
        with self.assertRaises(ValueError):
            model.fit(np.zeros((4, 5), dtype=np.float32), {"C": ["1", "1"]})

    def test_width_mismatch_is_rejected(self):
        import numpy as np

        model = bl.PooledDomainClassifier(dim=5)
        with self.assertRaises(ValueError):
            model.fit(np.zeros((4, 9), dtype=np.float32), {"C": ["1"] * 4})


if __name__ == "__main__":
    unittest.main()
