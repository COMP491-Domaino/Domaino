"""Tests for the query-based segmentation candidate (T2.3, second design)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.model import query_segmentation as qs  # noqa: E402
from domaino.model import segmentation as seg  # noqa: E402
from domaino.model.trunk import TrunkConfig, build_trunk  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_default_capacity_covers_the_dataset(self):
        # The architecture's stated cost is a fixed maximum number of domains.
        # The most any chain in dataset v1 carries is 20 (3vkgA), so at the
        # default the limit never binds on this data.
        self.assertEqual(qs.QuerySegmentationConfig().num_queries, 20)
        self.assertEqual(qs.MAX_DOMAINS_IN_DATASET, 20)

    def test_heads_must_divide_the_hidden_size(self):
        with self.assertRaises(ValueError):
            qs.QuerySegmentationConfig(hidden_dim=100, num_heads=8)

    def test_query_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            qs.QuerySegmentationConfig(num_queries=0)

    def test_both_architectures_describe_the_same_trunk(self):
        pair = seg.SegmentationConfig(input_dim=640, hidden_dim=128)
        query = qs.QuerySegmentationConfig(input_dim=640, hidden_dim=128)
        self.assertEqual(pair.trunk_config(), query.trunk_config())


class TrunkTests(unittest.TestCase):
    def test_trunk_preserves_length_and_widens_to_hidden(self):
        import torch

        trunk = build_trunk(TrunkConfig(input_dim=16, hidden_dim=24, dilations=(1, 2)))
        out = trunk(torch.randn(2, 40, 16), torch.ones(2, 40))
        self.assertEqual(tuple(out.shape), (2, 40, 24))

    def test_trunk_zeroes_padding(self):
        import torch

        trunk = build_trunk(TrunkConfig(input_dim=8, hidden_dim=16, dilations=(1,)))
        mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 6)], dim=1)
        out = trunk(torch.randn(1, 16, 8), mask)
        self.assertTrue(torch.all(out[0, 10:] == 0))

    def test_even_kernel_is_rejected(self):
        with self.assertRaises(ValueError):
            TrunkConfig(kernel_size=4)


class PositionalEncodingTests(unittest.TestCase):
    """Without positions, cross-attention cannot tell one stretch of the chain
    from another, and the model can only separate domains by their content."""

    def test_positions_are_distinguishable(self):
        import torch

        encoding = qs.sinusoidal_positions(50, 32)
        self.assertEqual(tuple(encoding.shape), (1, 50, 32))
        self.assertFalse(torch.allclose(encoding[0, 0], encoding[0, 25]))

    def test_the_same_index_always_gets_the_same_code(self):
        # A padded batch must give a real residue the same position it had
        # alone, or batching would change predictions.
        import torch

        short = qs.sinusoidal_positions(20, 32)
        long = qs.sinusoidal_positions(80, 32)
        torch.testing.assert_close(short[0, :20], long[0, :20])

    def test_the_model_is_position_sensitive(self):
        # Identical content at two different places must be able to produce
        # different masks; that is what the encoding buys.
        import torch

        model = qs.build_query_network(
            qs.QuerySegmentationConfig(
                input_dim=8, hidden_dim=32, dilations=(1,), num_queries=4,
                decoder_layers=2, num_heads=4, dropout=0.0,
            )
        ).eval()
        blob = torch.randn(1, 20, 8)
        left = torch.zeros(1, 120, 8)
        right = torch.zeros(1, 120, 8)
        left[:, 10:30] = blob
        right[:, 80:100] = blob
        with torch.no_grad():
            a = model(left)["mask_logits"]
            b = model(right)["mask_logits"]
        self.assertFalse(torch.allclose(a, b, atol=1e-3))


class MaskTargetTests(unittest.TestCase):
    def test_one_mask_per_domain(self):
        masks = qs.mask_targets(100, [[(1, 40)], [(61, 100)]])
        self.assertEqual(masks.shape, (2, 100))
        self.assertEqual(masks[0].sum(), 40)
        self.assertEqual(masks[1].sum(), 40)

    def test_linker_residues_belong_to_no_mask(self):
        masks = qs.mask_targets(100, [[(1, 40)], [(61, 100)]])
        self.assertEqual(masks[:, 40:60].sum(), 0)

    def test_discontinuous_domain_is_one_mask_with_a_gap(self):
        # The architectural claim: a mask can select separated positions, so
        # a discontinuous domain needs no special representation.
        masks = qs.mask_targets(240, [[(1, 60), (181, 240)], [(61, 180)]])
        self.assertEqual(masks.shape, (2, 240))
        self.assertEqual(masks[0].sum(), 120)
        self.assertEqual(masks[0][0], 1.0)
        self.assertEqual(masks[0][100], 0.0)
        self.assertEqual(masks[0][200], 1.0)

    def test_no_domains_gives_an_empty_target(self):
        self.assertEqual(qs.mask_targets(50, []).shape, (0, 50))


class MatchingTests(unittest.TestCase):
    def logits_for(self, masks, confident=True):
        """Mask logits that reproduce `masks`, in a shuffled query order."""
        import torch

        n, length = masks.shape
        logits = torch.full((n, length), -6.0)
        for i in range(n):
            logits[i] = torch.where(
                torch.as_tensor(masks[i]) > 0, torch.tensor(6.0), torch.tensor(-6.0)
            )
        cls = torch.zeros(n, 2)
        cls[:, 1] = 6.0 if confident else -6.0
        return logits, cls

    def test_matching_ignores_query_order(self):
        # Set prediction means the model is scored on which domains it found,
        # never the order it listed them in.
        import torch

        targets = torch.as_tensor(qs.mask_targets(100, [[(1, 50)], [(51, 100)]]))
        logits, cls = self.logits_for(targets.numpy())
        flipped = logits.flip(0)
        rows, cols = qs.match_queries(
            flipped, cls, targets, qs.QuerySegmentationConfig()
        )
        pairs = dict(zip(rows, cols))
        self.assertEqual(pairs, {0: 1, 1: 0})

    def test_matching_is_one_to_one(self):
        import torch

        targets = torch.as_tensor(
            qs.mask_targets(120, [[(1, 40)], [(41, 80)], [(81, 120)]])
        )
        logits, cls = self.logits_for(targets.numpy())
        rows, cols = qs.match_queries(logits, cls, targets, qs.QuerySegmentationConfig())
        self.assertEqual(len(set(rows)), len(rows))
        self.assertEqual(sorted(cols), [0, 1, 2])

    def test_extra_queries_go_unmatched(self):
        import torch

        targets = torch.as_tensor(qs.mask_targets(100, [[(1, 100)]]))
        logits = torch.randn(8, 100)
        cls = torch.zeros(8, 2)
        rows, cols = qs.match_queries(logits, cls, targets, qs.QuerySegmentationConfig())
        self.assertEqual(len(rows), 1)

    def test_no_targets_means_no_matches(self):
        import torch

        empty = torch.zeros(0, 60)
        rows, cols = qs.match_queries(
            torch.randn(5, 60), torch.zeros(5, 2), empty, qs.QuerySegmentationConfig()
        )
        self.assertEqual((rows, cols), ([], []))

    def test_dice_rewards_overlap_not_emptiness(self):
        # A domain covers a small part of a long chain, so an all-zero mask
        # already scores well per residue. Dice must not reward it.
        import torch

        target = torch.zeros(1, 200)
        target[0, :20] = 1.0
        empty = torch.zeros(1, 200)
        good = target.clone()
        self.assertGreater(
            float(qs.dice_cost(empty, target)), float(qs.dice_cost(good, target))
        )


class NetworkTests(unittest.TestCase):
    def net(self, **overrides):
        defaults = dict(
            input_dim=16, hidden_dim=32, dilations=(1, 2), num_queries=6,
            decoder_layers=2, num_heads=4,
        )
        defaults.update(overrides)
        return qs.build_query_network(qs.QuerySegmentationConfig(**defaults))

    def test_output_shapes(self):
        import torch

        model = self.net()
        out = model(torch.randn(2, 50, 16))
        self.assertEqual(tuple(out["class_logits"].shape), (2, 6, 2))
        self.assertEqual(tuple(out["mask_logits"].shape), (2, 6, 50))

    def test_no_pair_matrix_is_ever_built(self):
        # The architecture's advantage: output grows with queries times
        # length, not with length squared.
        import torch

        model = self.net()
        out = model(torch.randn(1, 1200, 16))
        self.assertEqual(tuple(out["mask_logits"].shape), (1, 6, 1200))

    def test_padding_is_excluded_from_masks(self):
        import torch

        model = self.net().eval()
        mask = torch.cat([torch.ones(1, 20), torch.zeros(1, 10)], dim=1)
        with torch.no_grad():
            out = model(torch.randn(1, 30, 16), mask=mask)
        self.assertTrue(torch.isinf(out["mask_logits"][0, :, 20:]).all())
        self.assertTrue((out["mask_logits"][0, :, 20:] < 0).all())

    def test_padding_does_not_change_a_real_residues_output(self):
        import torch

        model = self.net().eval()
        short = torch.randn(1, 24, 16)
        padded = torch.cat([short, torch.randn(1, 12, 16)], dim=1)
        mask = torch.cat([torch.ones(1, 24), torch.zeros(1, 12)], dim=1)
        with torch.no_grad():
            alone = model(short)["mask_logits"]
            batched = model(padded, mask=mask)["mask_logits"][:, :, :24]
        torch.testing.assert_close(alone, batched, rtol=1e-4, atol=1e-4)

    def test_gradients_reach_both_heads_and_the_queries(self):
        import torch

        model = self.net()
        out = model(torch.randn(1, 40, 16))
        (out["class_logits"].sum() + out["mask_logits"].sum()).backward()
        self.assertIsNotNone(model.class_head.weight.grad)
        self.assertIsNotNone(model.queries.weight.grad)
        self.assertTrue(torch.isfinite(model.queries.weight.grad).all())


class LossTests(unittest.TestCase):
    def net(self):
        return qs.build_query_network(
            qs.QuerySegmentationConfig(
                input_dim=16, hidden_dim=32, dilations=(1, 2), num_queries=6,
                decoder_layers=2, num_heads=4,
            )
        )

    def test_loss_is_finite_and_differentiable(self):
        import torch

        model = self.net()
        out = model(torch.randn(1, 80, 16))
        targets = [qs.mask_targets(80, [[(1, 40)], [(41, 80)]])]
        result = qs.query_segmentation_loss(
            out["class_logits"], out["mask_logits"], targets, lengths=[80]
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(model.queries.weight.grad)

    def test_matched_count_follows_the_targets(self):
        import torch

        model = self.net().eval()
        with torch.no_grad():
            out = model(torch.randn(2, 80, 16))
        targets = [
            qs.mask_targets(80, [[(1, 80)]]),
            qs.mask_targets(80, [[(1, 30)], [(31, 60)], [(61, 80)]]),
        ]
        result = qs.query_segmentation_loss(
            out["class_logits"], out["mask_logits"], targets, lengths=[80, 80]
        )
        self.assertEqual(result["matched_queries"], 4)

    def test_a_padded_batch_is_finite(self):
        """The regression test for a NaN that only padding could produce.

        The network fills padded positions with -inf, deliberately, so any
        reduction over the whole `mask_logits` tensor is -inf, and -inf times
        zero is NaN. Every other loss test here batches chains of equal
        length, where there is no padding and no -inf, so none of them could
        catch it -- a real batch of 24 cached chains did.
        """
        import torch

        model = self.net()
        lengths = [30, 55, 80]
        features = torch.randn(3, 80, 16)
        mask = torch.zeros(3, 80)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1.0
        out = model(features, mask=mask)
        self.assertFalse(
            torch.isfinite(out["mask_logits"]).all(),
            "padding should be -inf; if it is not, this test no longer guards"
            " what it was written for",
        )
        targets = [
            qs.mask_targets(length, [[(1, length // 2)], [(length // 2 + 1, length)]])
            for length in lengths
        ]
        result = qs.query_segmentation_loss(
            out["class_logits"], out["mask_logits"], targets, lengths=lengths
        )
        self.assertTrue(torch.isfinite(result["loss"]), result["loss"])
        result["loss"].backward()
        self.assertTrue(torch.isfinite(model.queries.weight.grad).all())

    def test_a_fully_gated_padded_batch_is_zero_and_differentiable(self):
        # The same padded batch with every chain masked out by T2.6's source
        # gate: zero, finite, and still attached to the graph.
        import torch

        model = self.net()
        lengths = [30, 55, 80]
        mask = torch.zeros(3, 80)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1.0
        out = model(torch.randn(3, 80, 16), mask=mask)
        result = qs.query_segmentation_loss(
            out["class_logits"], out["mask_logits"],
            [qs.mask_targets(length, [[(1, length)]]) for length in lengths],
            lengths=lengths, chain_weights=[0.0, 0.0, 0.0],
        )
        self.assertEqual(float(result["loss"]), 0.0)
        self.assertEqual(result["matched_queries"], 0)
        result["loss"].backward()
        self.assertIsNotNone(model.queries.weight.grad)

    def test_chain_weights_scale_a_chains_contribution(self):
        import torch

        model = self.net().eval()
        with torch.no_grad():
            out = model(torch.randn(2, 60, 16))
        targets = [qs.mask_targets(60, [[(1, 60)]]),
                   qs.mask_targets(60, [[(1, 20)], [(21, 60)]])]
        both = qs.query_segmentation_loss(
            out["class_logits"], out["mask_logits"], targets, lengths=[60, 60]
        )
        first_only = qs.query_segmentation_loss(
            out["class_logits"], out["mask_logits"], targets, lengths=[60, 60],
            chain_weights=[1.0, 0.0],
        )
        # Dropping a chain changes the loss but keeps it a mean, not a sum, so
        # the remaining chain's own value is what is left.
        alone = qs.query_segmentation_loss(
            out["class_logits"][:1], out["mask_logits"][:1], targets[:1], lengths=[60]
        )
        self.assertNotAlmostEqual(float(both["loss"]), float(first_only["loss"]), places=4)
        self.assertAlmostEqual(float(first_only["loss"]), float(alone["loss"]), places=4)

    def test_chains_with_different_domain_counts_batch_together(self):
        import torch

        model = self.net()
        out = model(torch.randn(3, 60, 16))
        targets = [
            qs.mask_targets(60, [[(1, 60)]]),
            qs.mask_targets(60, [[(1, 20)], [(21, 60)]]),
            qs.mask_targets(60, [[(1, 15)], [(16, 30)], [(31, 45)], [(46, 60)]]),
        ]
        result = qs.query_segmentation_loss(
            out["class_logits"], out["mask_logits"], targets, lengths=[60, 60, 60]
        )
        self.assertTrue(torch.isfinite(result["loss"]))


class DecodeTests(unittest.TestCase):
    def outputs(self, length, domains, confidences=None):
        """Confident query outputs that encode `domains` exactly."""
        import torch

        masks = qs.mask_targets(length, domains)
        n = len(domains)
        mask_logits = torch.where(
            torch.as_tensor(masks) > 0, torch.tensor(8.0), torch.tensor(-8.0)
        )
        cls = torch.zeros(n, 2)
        for i in range(n):
            score = 8.0 if confidences is None else confidences[i]
            cls[i, 1] = score
            cls[i, 0] = -score
        return cls, mask_logits

    def test_single_domain(self):
        cls, masks = self.outputs(120, [[(1, 120)]])
        self.assertEqual(qs.decode_queries(cls, masks, 120), [[(1, 120)]])

    def test_two_domains(self):
        cls, masks = self.outputs(200, [[(1, 100)], [(101, 200)]])
        self.assertEqual(
            qs.decode_queries(cls, masks, 200), [[(1, 100)], [(101, 200)]]
        )

    def test_discontinuous_domain_needs_no_clustering(self):
        # The architectural claim: thresholding a mask already yields the
        # separated intervals, with no pair matrix and no clustering step.
        cls, masks = self.outputs(240, [[(1, 60), (181, 240)], [(61, 180)]])
        domains = qs.decode_queries(cls, masks, 240)
        self.assertEqual(len(domains), 2)
        self.assertIn([(1, 60), (181, 240)], domains)

    def test_low_confidence_queries_are_dropped(self):
        cls, masks = self.outputs(200, [[(1, 100)], [(101, 200)]], confidences=[8.0, -8.0])
        domains = qs.decode_queries(cls, masks, 200)
        self.assertEqual(domains, [[(1, 100)]])

    def test_short_domains_are_dropped(self):
        cls, masks = self.outputs(200, [[(1, 180)], [(181, 200)]])
        domains = qs.decode_queries(
            cls, masks, 200, qs.QueryDecoderConfig(min_domain_length=40)
        )
        self.assertEqual(domains, [[(1, 180)]])

    def test_overlapping_queries_do_not_double_claim_a_residue(self):
        # Overlapping domains cannot be written as a CATH chopping, so the
        # more confident query keeps a contested residue.
        import torch

        masks = torch.full((2, 100), -8.0)
        masks[0, 0:80] = 8.0
        masks[1, 40:100] = 8.0
        cls = torch.tensor([[-8.0, 8.0], [-8.0, 8.0]])
        domains = qs.decode_queries(
            cls, masks, 100, qs.QueryDecoderConfig(min_domain_length=5)
        )
        claimed = [p for d in domains for s, e in d for p in range(s, e + 1)]
        self.assertEqual(len(claimed), len(set(claimed)))

    def test_gap_merging_never_swallows_another_querys_residues(self):
        # Query A owns 1-40 and 61-100, query B owns 41-60. Absorbing A's gap
        # would hand B's residues to A and produce an overlap, so the gap is
        # only bridged across residues nobody claimed.
        import torch

        masks = torch.full((2, 100), -8.0)
        masks[0, 0:40] = 8.0
        masks[0, 60:100] = 8.0
        masks[1, 40:60] = 8.0
        cls = torch.tensor([[-8.0, 8.0], [-8.0, 8.0]])
        domains = qs.decode_queries(
            cls, masks, 100,
            qs.QueryDecoderConfig(min_domain_length=5, max_linker_gap=40),
        )
        claimed = [p for d in domains for s, e in d for p in range(s, e + 1)]
        self.assertEqual(len(claimed), len(set(claimed)))
        self.assertIn([(1, 40), (61, 100)], domains)
        self.assertIn([(41, 60)], domains)

    def test_a_true_linker_gap_is_still_absorbed(self):
        # The flip side: a gap no query claims is a linker and belongs inside
        # the domain, or an ordinary domain would look discontinuous.
        import torch

        masks = torch.full((1, 100), -8.0)
        masks[0, 0:45] = 8.0
        masks[0, 55:100] = 8.0
        cls = torch.tensor([[-8.0, 8.0]])
        domains = qs.decode_queries(
            cls, masks, 100,
            qs.QueryDecoderConfig(min_domain_length=5, max_linker_gap=20),
        )
        self.assertEqual(domains, [[(1, 100)]])

    def test_always_returns_at_least_one_domain(self):
        # Every CATH chain has one, so answering "none" is never right.
        import torch

        cls = torch.full((6, 2), 0.0)
        cls[:, 0] = 8.0
        cls[:, 1] = -8.0
        domains = qs.decode_queries(cls, torch.full((6, 100), -8.0), 100)
        self.assertEqual(domains, [[(1, 100)]])

    def test_output_is_always_a_valid_parse(self):
        import numpy as np
        import torch

        rng = np.random.default_rng(0)
        for _ in range(20):
            length = int(rng.integers(80, 300))
            cls = torch.as_tensor(rng.normal(size=(8, 2)), dtype=torch.float32)
            masks = torch.as_tensor(
                rng.normal(size=(8, length)), dtype=torch.float32
            )
            domains = qs.decode_queries(cls, masks, length)
            claimed = []
            for domain in domains:
                for start, end in domain:
                    self.assertGreaterEqual(start, 1)
                    self.assertLessEqual(end, length)
                    self.assertLessEqual(start, end)
                    claimed.extend(range(start, end + 1))
            self.assertEqual(len(claimed), len(set(claimed)))
            self.assertGreaterEqual(len(domains), 1)


if __name__ == "__main__":
    unittest.main()
