"""Tests for the domain-segmentation network and its targets (T2.3)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.model import segmentation as seg  # noqa: E402
from domaino.model.decode import BOUNDARY, DOMAIN, LINKER  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_even_kernel_is_rejected(self):
        # An even kernel cannot keep the sequence length unchanged with
        # symmetric padding, which would silently shift every position.
        with self.assertRaises(ValueError):
            seg.SegmentationConfig(kernel_size=4)

    def test_pair_rank_must_be_positive(self):
        with self.assertRaises(ValueError):
            seg.SegmentationConfig(pair_rank=0)

    def test_distance_bucketing_is_monotonic(self):
        previous = -1
        for separation in (0, 1, 2, 3, 5, 9, 17, 33, 65, 200, 600, 5000):
            bucket = seg.bucket_distance(separation)
            self.assertGreaterEqual(bucket, previous)
            previous = bucket
        self.assertEqual(
            seg.bucket_distance(99999), len(seg.DISTANCE_BUCKETS)
        )


class ResidueTargetTests(unittest.TestCase):
    def test_single_domain_covering_the_chain(self):
        labels = seg.residue_targets(100, [[(1, 100)]], boundary_width=2)
        self.assertEqual(labels[50], DOMAIN)
        # The chain's own ends are domain edges and are marked as such.
        self.assertEqual(labels[0], BOUNDARY)
        self.assertEqual(labels[99], BOUNDARY)

    def test_linker_residues_are_labelled_linker(self):
        labels = seg.residue_targets(100, [[(1, 40)], [(61, 100)]])
        self.assertEqual(labels[50], LINKER)

    def test_boundary_band_has_the_configured_width(self):
        labels = seg.residue_targets(200, [[(1, 100)], [(101, 200)]], boundary_width=2)
        for position in (99, 100, 101, 102, 103):
            self.assertEqual(labels[position - 1], BOUNDARY, position)
        self.assertEqual(labels[90], DOMAIN)

    def test_boundary_never_overwrites_a_linker(self):
        # A residue in no domain stays linker even if it sits next to an edge,
        # or the model would be taught that linkers are domain boundaries.
        labels = seg.residue_targets(100, [[(1, 40)], [(61, 100)]], boundary_width=3)
        for position in range(41, 61):
            self.assertEqual(labels[position - 1], LINKER, position)

    def test_discontinuous_domain_marks_every_segment_edge(self):
        labels = seg.residue_targets(
            240, [[(1, 60), (181, 240)], [(61, 180)]], boundary_width=0
        )
        for position in (1, 60, 61, 180, 181, 240):
            self.assertEqual(labels[position - 1], BOUNDARY, position)


class PairTargetTests(unittest.TestCase):
    def test_same_domain_pairs_are_positive(self):
        target, mask = seg.pair_targets(100, [[(1, 50)], [(51, 100)]])
        self.assertEqual(target[0, 49], 1.0)
        self.assertEqual(target[60, 70], 1.0)

    def test_different_domain_pairs_are_negative_but_supervised(self):
        target, mask = seg.pair_targets(100, [[(1, 50)], [(51, 100)]])
        self.assertEqual(target[0, 60], 0.0)
        self.assertEqual(mask[0, 60], 1.0)

    def test_linker_pairs_are_unsupervised(self):
        # Whether a linker residue is "in the same domain" as anything has no
        # answer; supervising it as negative would teach the model that
        # linkers repel every residue.
        target, mask = seg.pair_targets(100, [[(1, 40)], [(61, 100)]])
        self.assertEqual(mask[50, 10], 0.0)
        self.assertEqual(mask[50, 50], 0.0)
        self.assertEqual(mask[10, 10], 1.0)

    def test_discontinuous_segments_are_the_same_domain(self):
        # This is the relation boundary prediction alone cannot express.
        target, mask = seg.pair_targets(240, [[(1, 60), (181, 240)], [(61, 180)]])
        self.assertEqual(target[0, 200], 1.0)
        self.assertEqual(target[0, 100], 0.0)

    def test_target_is_symmetric(self):
        import numpy as np

        target, mask = seg.pair_targets(80, [[(1, 30), (61, 80)], [(31, 60)]])
        np.testing.assert_array_equal(target, target.T)
        np.testing.assert_array_equal(mask, mask.T)


class PairSamplingTests(unittest.TestCase):
    def test_small_masks_are_returned_whole(self):
        _, mask = seg.pair_targets(20, [[(1, 20)]])
        rows, cols = seg.sample_pairs(mask, limit=10000)
        self.assertEqual(len(rows), 20 * 19 // 2)

    def test_sampling_respects_the_limit(self):
        _, mask = seg.pair_targets(300, [[(1, 300)]])
        rows, cols = seg.sample_pairs(mask, limit=500)
        self.assertEqual(len(rows), 500)

    def test_only_supervised_pairs_are_sampled(self):
        _, mask = seg.pair_targets(100, [[(1, 40)], [(61, 100)]])
        rows, cols = seg.sample_pairs(mask, limit=100000)
        for r, c in zip(rows, cols):
            self.assertEqual(mask[r, c], 1.0)

    def test_sampling_is_reproducible(self):
        _, mask = seg.pair_targets(200, [[(1, 200)]])
        a = seg.sample_pairs(mask, limit=100, seed=7)
        b = seg.sample_pairs(mask, limit=100, seed=7)
        self.assertEqual(list(a[0]), list(b[0]))

    def test_upper_triangle_only_so_no_pair_is_counted_twice(self):
        _, mask = seg.pair_targets(50, [[(1, 50)]])
        rows, cols = seg.sample_pairs(mask, limit=100000)
        self.assertTrue(all(r < c for r, c in zip(rows, cols)))


class NetworkTests(unittest.TestCase):
    def net(self, **overrides):
        defaults = dict(input_dim=16, hidden_dim=24, dilations=(1, 2), pair_rank=8)
        defaults.update(overrides)
        return seg.build_network(seg.SegmentationConfig(**defaults))

    def test_output_shapes(self):
        import torch

        model = self.net()
        x = torch.randn(2, 40, 16)
        out = model(x)
        self.assertEqual(tuple(out["residue_logits"].shape), (2, 40, 3))
        self.assertEqual(tuple(out["pair_logits"].shape), (2, 40, 40))

    def test_pair_logits_are_symmetric(self):
        # The same-domain relation is symmetric, so the network should not be
        # able to disagree with itself about (i,j) versus (j,i).
        import torch

        model = self.net().eval()
        with torch.no_grad():
            pair = model(torch.randn(1, 30, 16))["pair_logits"]
        torch.testing.assert_close(pair, pair.transpose(1, 2))

    def test_sampled_pair_scores_match_the_full_matrix(self):
        # Training scores sampled pairs and inference builds the whole matrix;
        # if the two disagreed, the model would be trained on one function and
        # evaluated on another.
        import torch

        model = self.net().eval()
        x = torch.randn(1, 24, 16)
        rows = torch.tensor([0, 3, 11, 20])
        cols = torch.tensor([5, 9, 2, 23])
        with torch.no_grad():
            full = model(x)["pair_logits"][0]
            sampled = model(x, pairs=(rows, cols))["pair_logits"][0]
        torch.testing.assert_close(sampled, full[rows, cols], rtol=1e-4, atol=1e-5)

    def test_padding_does_not_change_a_real_residues_output(self):
        # A padded batch must give the same answer as the sequence alone, or
        # batching would silently alter predictions.
        import torch

        model = self.net().eval()
        short = torch.randn(1, 20, 16)
        padded = torch.cat([short, torch.randn(1, 15, 16)], dim=1)
        mask = torch.cat([torch.ones(1, 20), torch.zeros(1, 15)], dim=1)
        with torch.no_grad():
            alone = model(short)["residue_logits"]
            batched = model(padded, mask=mask)["residue_logits"][:, :20]
        torch.testing.assert_close(alone, batched, rtol=1e-4, atol=1e-5)

    def test_receptive_field_covers_a_domain(self):
        model = seg.build_network(
            seg.SegmentationConfig(
                input_dim=16, hidden_dim=24, dilations=(1, 2, 4, 8, 16, 32)
            )
        )
        # A domain is typically 100-200 residues, so the trunk must see that far.
        self.assertGreater(model.receptive_field, 200)

    def test_gradients_reach_both_heads(self):
        import torch

        model = self.net()
        x = torch.randn(1, 30, 16)
        out = model(x)
        loss = out["residue_logits"].sum() + out["pair_logits"].sum()
        loss.backward()
        self.assertIsNotNone(model.residue_head.weight.grad)
        self.assertIsNotNone(model.pair_head.left.weight.grad)
        self.assertTrue(torch.isfinite(model.residue_head.weight.grad).all())

    def test_encoder_is_not_part_of_this_network(self):
        # The representation stays frozen and external; the network must not
        # smuggle in trainable encoder parameters.
        model = self.net()
        names = [n for n, _ in model.named_parameters()]
        self.assertFalse(any("esm" in n.lower() for n in names))


class LossTests(unittest.TestCase):
    def test_loss_is_finite_and_differentiable(self):
        import torch

        residue_logits = torch.randn(1, 20, 3, requires_grad=True)
        residue_target = torch.randint(0, 3, (1, 20))
        pair_logits = torch.randn(30, requires_grad=True)
        pair_target = torch.randint(0, 2, (30,)).float()
        result = seg.segmentation_loss(
            residue_logits, residue_target, pair_logits, pair_target
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(residue_logits.grad)

    def test_mask_excludes_padded_residues(self):
        import torch

        logits = torch.zeros(1, 4, 3)
        # Make the padded positions wildly wrong; the mask must ignore them.
        logits[0, 2:, 0] = 50.0
        target = torch.tensor([[1, 1, 2, 2]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        pair_logits = torch.zeros(3)
        pair_target = torch.zeros(3)
        masked = seg.segmentation_loss(
            logits, target, pair_logits, pair_target, mask=mask
        )["residue_loss"]
        unmasked = seg.segmentation_loss(
            logits, target, pair_logits, pair_target
        )["residue_loss"]
        self.assertLess(float(masked), float(unmasked))


if __name__ == "__main__":
    unittest.main()
