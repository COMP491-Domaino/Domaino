"""Tests for the hierarchical CATH classifier (T2.5)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.model import classifier as clf  # noqa: E402

# A small but real-shaped slice of CATH: two classes, a shared topology depth,
# one superfamily with plenty of support and one with almost none.
LABELS = {
    "C": ["1"] * 30 + ["3"] * 30,
    "A": ["1.10"] * 30 + ["3.40"] * 30,
    "T": ["1.10.10"] * 30 + ["3.40.50"] * 30,
    "H": ["1.10.10.10"] * 29 + ["1.10.10.20"] * 1 + ["3.40.50.300"] * 30,
}


class HierarchyTests(unittest.TestCase):
    def tree(self):
        return clf.CathHierarchy.from_labels(LABELS)

    def test_label_space_sizes(self):
        tree = self.tree()
        self.assertEqual(tree.sizes(), {"C": 2, "A": 2, "T": 2, "H": 3})

    def test_parent_code(self):
        self.assertEqual(clf.parent_code("3.40.50.300"), "3.40.50")
        self.assertEqual(clf.parent_code("3.40"), "3")
        self.assertIsNone(clf.parent_code("3"))

    def test_children_follow_the_real_tree(self):
        tree = self.tree()
        under_one = tree.children_of("A", "1")
        self.assertEqual([tree.codes["A"][i] for i in under_one], ["1.10"])
        under_topology = tree.children_of("H", "1.10.10")
        self.assertEqual(
            sorted(tree.codes["H"][i] for i in under_topology),
            ["1.10.10.10", "1.10.10.20"],
        )

    def test_support_counts_come_from_the_training_labels(self):
        tree = self.tree()
        self.assertEqual(tree.h_support("1.10.10.10"), 29)
        self.assertEqual(tree.h_support("1.10.10.20"), 1)
        self.assertEqual(tree.h_support("nonexistent"), 0)

    def test_well_supported_filter(self):
        tree = self.tree()
        self.assertEqual(
            sorted(tree.well_supported_h(10)), ["1.10.10.10", "3.40.50.300"]
        )

    def test_encoding_marks_missing_labels_as_ignored(self):
        # A domain CATH resolves only to T must train C, A and T and leave H
        # alone, rather than learn a wrong H.
        tree = self.tree()
        encoded = tree.encode(
            {"C": ["1", "3"], "A": ["1.10", "3.40"], "T": ["1.10.10", "3.40.50"],
             "H": ["1.10.10.10", ""]}
        )
        self.assertEqual(list(encoded["H"]), [tree.index["H"]["1.10.10.10"], -100])

    def test_unknown_code_is_ignored_not_guessed(self):
        tree = self.tree()
        encoded = tree.encode({"C": ["9"]})
        self.assertEqual(list(encoded["C"]), [-100])

    def test_round_trip_through_a_checkpoint(self):
        tree = self.tree()
        restored = clf.CathHierarchy.from_dict(tree.as_dict())
        self.assertEqual(restored.sizes(), tree.sizes())
        self.assertEqual(restored.h_support("1.10.10.10"), 29)
        self.assertEqual(
            restored.children_of("A", "1"), tree.children_of("A", "1")
        )


class PoolingTests(unittest.TestCase):
    def test_pools_contiguous_residues(self):
        import torch

        features = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        pooled = clf.pool_domain(features, [(1, 10)])
        torch.testing.assert_close(pooled, features.mean(dim=0))

    def test_pools_across_a_discontinuous_domain(self):
        import torch

        features = torch.zeros(10, 2)
        features[0:2] = 1.0
        features[8:10] = 3.0
        pooled = clf.pool_domain(features, [(1, 2), (9, 10)])
        torch.testing.assert_close(pooled, torch.tensor([2.0, 2.0]))

    def test_query_vector_is_concatenated(self):
        import torch

        features = torch.ones(10, 4)
        pooled = clf.pool_domain(features, [(1, 10)], query_vector=torch.zeros(3))
        self.assertEqual(tuple(pooled.shape), (7,))

    def test_empty_selection_is_rejected(self):
        import torch

        with self.assertRaises(ValueError):
            clf.pool_domain(torch.zeros(5, 2), [])


class WeightTests(unittest.TestCase):
    def test_rare_classes_weigh_more(self):
        weights = clf.class_balanced_weights([1, 1000])
        self.assertGreater(float(weights[0]), float(weights[1]))

    def test_weights_are_normalised(self):
        weights = clf.class_balanced_weights([1, 10, 1000])
        self.assertAlmostEqual(float(weights.sum()), 3.0, places=3)

    def test_focal_loss_down_weights_easy_examples(self):
        import torch

        easy = torch.tensor([[10.0, -10.0]])
        hard = torch.tensor([[0.1, 0.0]])
        target = torch.tensor([0])
        self.assertLess(
            float(clf.focal_cross_entropy(easy, target)),
            float(clf.focal_cross_entropy(hard, target)),
        )

    def test_focal_loss_ignores_masked_targets(self):
        import torch

        logits = torch.randn(3, 4)
        ignored = torch.tensor([-100, -100, -100])
        self.assertEqual(float(clf.focal_cross_entropy(logits, ignored)), 0.0)


class ClassifierTests(unittest.TestCase):
    def build(self, **overrides):
        defaults = dict(domain_dim=8, hidden_dim=16, dropout=0.0)
        defaults.update(overrides)
        tree = clf.CathHierarchy.from_labels(LABELS)
        return tree, clf.build_classifier(tree, clf.ClassifierConfig(**defaults))

    def test_head_sizes_match_the_label_space(self):
        tree, model = self.build()
        import torch

        logits = model(torch.randn(4, 8))
        for level, size in tree.sizes().items():
            self.assertEqual(tuple(logits[level].shape), (4, size), level)

    def test_loss_skips_levels_with_no_label(self):
        import torch

        tree, model = self.build()
        logits = model(torch.randn(2, 8))
        targets = {
            level: torch.full((2,), -100, dtype=torch.long) for level in tree.levels
        }
        targets["C"] = torch.tensor([0, 1])
        total, parts = model.loss(logits, targets)
        self.assertEqual(set(parts), {"C"})
        self.assertTrue(torch.isfinite(total))

    def test_loss_is_zero_safe_when_nothing_is_supervised(self):
        import torch

        tree, model = self.build()
        logits = model(torch.randn(2, 8))
        targets = {
            level: torch.full((2,), -100, dtype=torch.long) for level in tree.levels
        }
        total, parts = model.loss(logits, targets)
        self.assertEqual(parts, {})
        self.assertEqual(float(total), 0.0)

    def test_every_weighting_scheme_runs(self):
        import torch

        for scheme in ("none", "balanced", "focal"):
            tree, model = self.build(loss_weighting=scheme)
            logits = model(torch.randn(4, 8))
            targets = tree.encode(
                {k: v[:4] for k, v in LABELS.items()}
            )
            targets = {k: torch.as_tensor(v) for k, v in targets.items()}
            total, parts = model.loss(logits, targets)
            self.assertTrue(torch.isfinite(total), scheme)

    def test_unknown_weighting_is_rejected(self):
        with self.assertRaises(ValueError):
            clf.ClassifierConfig(loss_weighting="magic")

    def test_h_cannot_be_predicted_without_its_parent(self):
        with self.assertRaises(ValueError):
            clf.ClassifierConfig(levels=("C", "A", "H"))

    def test_prediction_is_always_a_real_path_in_the_tree(self):
        # The point of path decoding: four independent softmaxes could return
        # 1.10.10 under class 3, which CATH cannot express.
        import torch

        tree, model = self.build(min_h_support=0, h_confidence_threshold=0.0)
        model.eval()
        for _ in range(20):
            rows = model.predict(torch.randn(8, 8))
            for row in rows:
                labels = row["labels"]
                for child, parent in (("A", "C"), ("T", "A"), ("H", "T")):
                    if labels[child] == clf.UNRESOLVED:
                        continue
                    self.assertTrue(
                        labels[child].startswith(labels[parent] + "."),
                        f"{labels[child]} is not under {labels[parent]}",
                    )

    def test_h_abstains_when_support_is_too_low(self):
        import torch

        tree, model = self.build(min_h_support=1000, h_confidence_threshold=0.0)
        model.eval()
        rows = model.predict(torch.randn(6, 8))
        for row in rows:
            self.assertEqual(row["labels"]["H"], clf.UNRESOLVED)
            # C, A and T must survive the abstention.
            self.assertNotEqual(row["labels"]["C"], clf.UNRESOLVED)
            self.assertNotEqual(row["labels"]["T"], clf.UNRESOLVED)

    def test_h_abstains_when_confidence_is_too_low(self):
        import torch

        tree, model = self.build(min_h_support=0, h_confidence_threshold=0.999)
        model.eval()
        rows = model.predict(torch.randn(6, 8))
        self.assertTrue(
            all(r["labels"]["H"] == clf.UNRESOLVED for r in rows)
        )

    def test_coverage_counts_only_named_superfamilies(self):
        rows = [
            {"labels": {"H": "1.10.10.10"}},
            {"labels": {"H": clf.UNRESOLVED}},
            {"labels": {"H": "3.40.50.300"}},
            {"labels": {"H": clf.UNRESOLVED}},
        ]
        self.assertEqual(clf.h_coverage(rows), 0.5)
        self.assertEqual(clf.h_coverage([]), 0.0)

    def test_it_can_learn_a_separable_label_space(self):
        import torch

        tree, model = self.build(
            loss_weighting="balanced", min_h_support=0, h_confidence_threshold=0.0
        )
        # Two well-separated groups matching the two CATH paths.
        rows = 60
        x = torch.zeros(rows, 8)
        x[:30, 0] = 1.0
        x[30:, 1] = 1.0
        x += torch.randn(rows, 8) * 0.05
        targets = {
            k: torch.as_tensor(v) for k, v in tree.encode(LABELS).items()
        }
        optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
        model.train()
        for _ in range(300):
            optimiser.zero_grad()
            total, _ = model.loss(model(x), targets)
            total.backward()
            optimiser.step()
        model.eval()
        predicted = model.predict(x)
        correct = sum(
            1
            for row, code in zip(predicted, LABELS["T"])
            if row["labels"]["T"] == code
        )
        self.assertGreater(correct / rows, 0.9)


if __name__ == "__main__":
    unittest.main()
