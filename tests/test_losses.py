"""Tests for the source-aware targets and losses (task T2.6).

The point of these losses is what they *refuse* to learn from, so most of
these tests assert an absence: a residue Pfam never annotated contributes no
gradient, a Pfam chain contributes no pairs, a chain without CATH labels
contributes nothing to the classifier. Those are exactly the mistakes that a
loss curve would never reveal.
"""

from __future__ import annotations

import math
import unittest

from domaino.data.schema import Supervision
from domaino.model import losses
from domaino.model.decode import BOUNDARY, DOMAIN, LINKER

try:
    import torch
except ImportError:  # pragma: no cover - torch is installed in the run env
    torch = None

requires_torch = unittest.skipIf(torch is None, "torch is not installed")


class TargetTests(unittest.TestCase):
    def test_assignment_marks_domain_residues(self):
        target = losses.assignment_targets(10, [[(2, 4)], [(7, 9)]])
        self.assertEqual(list(target), [0, 1, 1, 1, 0, 0, 1, 1, 1, 0])

    def test_assignment_covers_discontinuous_domains(self):
        target = losses.assignment_targets(8, [[(1, 2), (6, 7)]])
        self.assertEqual(list(target), [1, 1, 0, 0, 0, 1, 1, 0])

    def test_assignment_clips_out_of_range_segments(self):
        # A malformed chopping must not raise or write past the array; CATH
        # 4.4.0 really does ship intervals that exceed the chain.
        target = losses.assignment_targets(5, [[(3, 99)]])
        self.assertEqual(list(target), [0, 0, 1, 1, 1])

    def test_boundary_band_straddles_each_edge(self):
        target = losses.boundary_targets(12, [[(5, 8)]], boundary_width=1)
        # Edges 5 and 8, each with one residue either side.
        self.assertEqual(list(target), [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])

    def test_boundary_band_is_clipped_at_the_termini(self):
        target = losses.boundary_targets(4, [[(1, 4)]], boundary_width=2)
        self.assertEqual(list(target), [1, 1, 1, 1])

    def test_a_chain_with_no_domains_has_no_positives(self):
        self.assertEqual(sum(losses.assignment_targets(6, [])), 0)
        self.assertEqual(sum(losses.boundary_targets(6, [])), 0)


class SourceMaskTests(unittest.TestCase):
    def test_cath_supervises_every_residue(self):
        mask = losses.source_masks(
            10, [[(2, 4)]], losses.supervision_for("cath")
        )
        self.assertEqual(list(mask.assignment), [1.0] * 10)
        self.assertEqual(list(mask.boundary), [1.0] * 10)
        self.assertTrue(mask.pairwise)
        self.assertTrue(mask.classification)
        self.assertEqual(mask.weight, 1.0)

    def test_pfam_supervises_only_its_annotated_regions(self):
        mask = losses.source_masks(
            10, [[(3, 6)]], losses.supervision_for("pfam")
        )
        self.assertEqual(list(mask.assignment), [0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        # The unannotated tail is unknown, not linker: supervising it as a
        # negative is the specific error this mask exists to prevent.
        self.assertEqual(mask.assignment[8], 0.0)
        self.assertFalse(mask.pairwise)
        self.assertFalse(mask.classification)

    def test_low_consensus_ted_is_not_a_complete_partition(self):
        high = losses.supervision_for("ted", consensus="high")
        low = losses.supervision_for("ted", weight=0.05, consensus="low")
        self.assertTrue(high.pairwise)
        self.assertFalse(low.pairwise)
        self.assertEqual(
            list(losses.source_masks(6, [[(2, 3)]], low).assignment),
            [0, 1, 1, 0, 0, 0],
        )

    def test_a_record_without_residue_labels_supervises_nothing(self):
        supervision = Supervision(
            residue_labels=False, pairwise=False, cath_labels=True,
            complete_partition=False, weight=0.5,
        )
        mask = losses.source_masks(5, [[(1, 3)]], supervision)
        self.assertEqual(sum(mask.assignment), 0.0)
        self.assertEqual(sum(mask.boundary), 0.0)
        self.assertTrue(mask.classification)

    def test_default_weights_match_the_source_modules(self):
        # If these drift from data/sources/, a record built by hand trains at
        # a different strength than the same record read from disk.
        self.assertEqual(losses.supervision_for("cath").weight, 1.0)
        self.assertEqual(losses.supervision_for("pfam").weight, 0.3)
        self.assertEqual(losses.supervision_for("ted").weight, 0.15)

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            losses.supervision_for("interpro")


@requires_torch
class TermTests(unittest.TestCase):
    def _logits(self, probabilities):
        """Residue logits whose softmax is exactly `probabilities`."""
        return torch.log(torch.tensor(probabilities, dtype=torch.float32))

    def test_assignment_loss_is_the_marginal_binary_cross_entropy(self):
        # One residue, p(linker)=0.2, p(domain)=0.5, p(boundary)=0.3, so
        # P(in domain) = 0.8. Target 1 gives -log 0.8.
        logits = self._logits([[[0.2, 0.5, 0.3]]])
        target = torch.tensor([[1.0]])
        weight = torch.tensor([[1.0]])
        value = losses.assignment_loss(logits, target, weight)
        self.assertAlmostEqual(float(value), -math.log(0.8), places=5)

    def test_boundary_loss_uses_only_the_boundary_class(self):
        logits = self._logits([[[0.2, 0.5, 0.3]]])
        target = torch.tensor([[1.0]])
        weight = torch.tensor([[1.0]])
        value = losses.boundary_loss(logits, target, weight)
        self.assertAlmostEqual(float(value), -math.log(0.3), places=5)

    def test_the_two_terms_ask_different_questions(self):
        # Same logits and the same target vector, different answers: the point
        # of splitting the residue loss in two.
        logits = self._logits([[[0.2, 0.5, 0.3]]])
        target = torch.tensor([[1.0]])
        weight = torch.tensor([[1.0]])
        self.assertNotAlmostEqual(
            float(losses.assignment_loss(logits, target, weight)),
            float(losses.boundary_loss(logits, target, weight)),
            places=3,
        )

    def test_masked_residues_do_not_change_the_loss(self):
        logits = self._logits([[[0.2, 0.5, 0.3], [0.9, 0.05, 0.05]]])
        target = torch.tensor([[1.0, 1.0]])
        only_first = losses.assignment_loss(
            logits, target, torch.tensor([[1.0, 0.0]])
        )
        self.assertAlmostEqual(float(only_first), -math.log(0.8), places=5)
        both = losses.assignment_loss(logits, target, torch.tensor([[1.0, 1.0]]))
        self.assertGreater(float(both), float(only_first))

    def test_a_fully_masked_term_is_zero_and_still_differentiable(self):
        logits = self._logits([[[0.2, 0.5, 0.3]]]).requires_grad_(True)
        value = losses.assignment_loss(
            logits, torch.tensor([[1.0]]), torch.tensor([[0.0]])
        )
        self.assertEqual(float(value), 0.0)
        # A batch where one source masks a term out entirely must not break
        # backward for the terms that remain.
        value.backward()
        self.assertIsNotNone(logits.grad)

    def test_record_weight_scales_a_chain_s_share(self):
        logits = self._logits(
            [[[0.2, 0.5, 0.3]], [[0.9, 0.05, 0.05]]]
        )
        target = torch.tensor([[1.0], [1.0]])
        equal = losses.assignment_loss(logits, target, torch.tensor([[1.0], [1.0]]))
        tilted = losses.assignment_loss(logits, target, torch.tensor([[1.0], [0.1]]))
        # The second chain is badly wrong, so down-weighting it must lower the
        # mean.
        self.assertLess(float(tilted), float(equal))

    def test_pairwise_loss_honours_per_pair_weights(self):
        logits = torch.zeros(4)
        target = torch.tensor([1.0, 1.0, 1.0, 1.0])
        weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
        value = losses.pairwise_loss(logits, target, weight)
        self.assertAlmostEqual(float(value), -math.log(0.5), places=5)
        self.assertEqual(
            float(losses.pairwise_loss(logits, target, torch.zeros(4))), 0.0
        )

    def test_pairwise_loss_is_none_without_pairs(self):
        self.assertIsNone(losses.pairwise_loss(None, None, None))
        self.assertIsNone(
            losses.pairwise_loss(torch.zeros(0), torch.zeros(0), torch.zeros(0))
        )


@requires_torch
class BatchTargetTests(unittest.TestCase):
    def test_padding_carries_no_weight(self):
        built = losses.build_residue_targets(
            [4, 8],
            [[[(1, 4)]], [[(2, 7)]]],
            [losses.supervision_for("cath"), losses.supervision_for("cath")],
            longest=8,
        )
        self.assertEqual(list(built.assignment_weight[0]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(built.assignment_weight[1]), [1] * 8)

    def test_mixed_sources_get_different_masks_in_one_batch(self):
        built = losses.build_residue_targets(
            [6, 6],
            [[[(2, 4)]], [[(2, 4)]]],
            [losses.supervision_for("cath"), losses.supervision_for("pfam")],
        )
        self.assertEqual(list(built.assignment_weight[0]), [1] * 6)
        self.assertEqual(
            [round(float(v), 3) for v in built.assignment_weight[1]],
            [0.0, 0.3, 0.3, 0.3, 0.0, 0.0],
        )
        # The *targets* are identical; only the masks and weights differ.
        self.assertEqual(list(built.assignment[0]), list(built.assignment[1]))

    def test_a_batch_scores_as_its_chains_do_alone(self):
        # The property that makes length-bucketed batching safe: padding a
        # short chain next to a long one must not change its loss.
        torch.manual_seed(0)
        chain = [[(2, 5)]]
        alone = losses.build_residue_targets(
            [8], [chain], [losses.supervision_for("cath")]
        )
        padded = losses.build_residue_targets(
            [8, 20],
            [chain, [[(3, 18)]]],
            [losses.supervision_for("cath")] * 2,
            longest=20,
        )
        logits = torch.randn(1, 8, 3)
        wide = torch.zeros(2, 20, 3)
        wide[0, :8] = logits[0]
        single = losses.assignment_loss(logits, alone.assignment, alone.assignment_weight)
        row = losses.assignment_loss(
            wide[:1],
            padded.assignment[:1, :20],
            padded.assignment_weight[:1, :20],
        )
        self.assertAlmostEqual(float(single), float(row), places=5)


@requires_torch
class CompositeTests(unittest.TestCase):
    def test_composite_sums_its_weighted_terms(self):
        logits = torch.log(torch.tensor([[[0.2, 0.5, 0.3]]], dtype=torch.float32))
        targets = losses.ResidueBatchTargets(
            assignment=torch.tensor([[1.0]]),
            assignment_weight=torch.tensor([[1.0]]),
            boundary=torch.tensor([[1.0]]),
            boundary_weight=torch.tensor([[1.0]]),
            pair_weight=None,
        )
        weights = losses.LossWeights(assignment=1.0, boundary=0.5, pairwise=2.0)
        result = losses.source_aware_loss(
            logits,
            targets,
            pair_logits=torch.zeros(2),
            pair_target=torch.ones(2),
            pair_weight=torch.ones(2),
            weights=weights,
        )
        expected = (
            -math.log(0.8) + 0.5 * -math.log(0.3) + 2.0 * -math.log(0.5)
        )
        self.assertAlmostEqual(float(result["loss"]), expected, places=5)
        self.assertEqual(
            sorted(result["parts"]), ["assignment", "boundary", "pairwise"]
        )

    def test_composite_without_pairs_omits_the_term(self):
        logits = torch.log(torch.tensor([[[0.2, 0.5, 0.3]]], dtype=torch.float32))
        targets = losses.ResidueBatchTargets(
            assignment=torch.tensor([[1.0]]),
            assignment_weight=torch.tensor([[1.0]]),
            boundary=torch.tensor([[0.0]]),
            boundary_weight=torch.tensor([[1.0]]),
            pair_weight=None,
        )
        result = losses.source_aware_loss(logits, targets)
        self.assertNotIn("pairwise", result["parts"])
        self.assertTrue(math.isfinite(float(result["loss"])))


if __name__ == "__main__":
    unittest.main()
