"""Tests for putting ESM-2 in the training graph.

The expensive tests use the cached 150M encoder and are skipped unless
DOMAINO_SLOW_TESTS is set, because loading a transformer is not something the
rest of the suite should pay for on every run. The cheap tests cover the part
that decides what gets gradients, which is the part a mistake would hide in:
a run that silently trains nothing looks exactly like a run that trains.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.model import finetune as ft  # noqa: E402
from domaino.train import data as tdata  # noqa: E402
from domaino.train.trainer import TrainConfig  # noqa: E402

SLOW = os.environ.get("DOMAINO_SLOW_TESTS")
CACHE = "/scratch/esevinc22/new_TED/torch_cache"
SMALL = "esm2_t30_150M_UR50D"
slow_test = unittest.skipUnless(SLOW, "set DOMAINO_SLOW_TESTS=1 to load an encoder")


class ConfigTests(unittest.TestCase):
    def test_dimension_comes_from_the_model_name(self):
        self.assertEqual(ft.FinetuneConfig().embed_dim, 1280)
        self.assertEqual(ft.FinetuneConfig(model_name=SMALL).embed_dim, 640)

    def test_an_unknown_encoder_is_rejected_at_construction(self):
        with self.assertRaises(KeyError):
            ft.FinetuneConfig(model_name="esm2_t99_900B")

    def test_negative_trainable_layers_is_rejected(self):
        with self.assertRaises(ValueError):
            ft.FinetuneConfig(trainable_layers=-1)

    def test_overlap_must_be_smaller_than_the_window(self):
        with self.assertRaises(ValueError):
            ft.FinetuneConfig(window_size=256, window_overlap=256)

    def test_the_language_model_head_is_never_trained(self):
        # It is not on the path to a residue representation, so training it
        # would spend optimiser memory on gradients that cannot help.
        self.assertIn("lm_head", ft.ALWAYS_FROZEN)
        self.assertIn("embed_tokens", ft.ALWAYS_FROZEN)


class TrainConfigTests(unittest.TestCase):
    def test_fine_tuning_is_off_by_default(self):
        # Every result in reports/ reads the frozen T2.2 cache. If this
        # default ever flips, those runs stop being reproducible from config.
        self.assertFalse(TrainConfig().finetune_encoder)

    def test_the_encoder_gets_a_smaller_learning_rate_than_the_heads(self):
        config = TrainConfig()
        self.assertLess(config.encoder_learning_rate, config.learning_rate)

    def test_fine_tuning_fields_are_part_of_the_run_identity(self):
        # The run id hashes the config, so two runs differing only in whether
        # the encoder was trained must not collide.
        frozen = TrainConfig().as_dict()
        unfrozen = TrainConfig(finetune_encoder=True).as_dict()
        self.assertNotEqual(frozen, unfrozen)
        self.assertIn("finetune_encoder", frozen)
        self.assertIn("finetune_layers", frozen)


class ExampleSequenceTests(unittest.TestCase):
    def test_sequences_are_absent_unless_asked_for(self):
        example = tdata.ChainExample("a", 10, [[(1, 10)]])
        self.assertIsNone(example.sequence)

    def test_an_example_can_carry_its_sequence(self):
        example = tdata.ChainExample("a", 4, [[(1, 4)]], sequence="MKTA")
        self.assertEqual(example.sequence, "MKTA")


@slow_test
class EncoderTests(unittest.TestCase):
    """The real encoder, small variant, on CPU."""

    @classmethod
    def setUpClass(cls):
        cls.config = ft.FinetuneConfig(
            model_name=SMALL, trainable_layers=2, gradient_checkpointing=False,
            torch_home=CACHE, max_tokens_per_forward=2048,
        )
        cls.encoder = ft.build_trainable_encoder(cls.config)

    def test_only_the_top_layers_get_gradients(self):
        trainable = {
            name for name, p in self.encoder.esm.named_parameters()
            if p.requires_grad
        }
        self.assertTrue(trainable, "nothing is trainable")
        indices = {
            int(n.split(".")[1]) for n in trainable if n.startswith("layers.")
        }
        self.assertEqual(indices, {28, 29})  # the 150M model has 30 layers
        self.assertFalse(any(n.startswith("lm_head") for n in trainable))
        self.assertFalse(any(n.startswith("embed_tokens") for n in trainable))

    def test_trainable_count_is_a_fraction_of_the_whole(self):
        total = sum(p.numel() for p in self.encoder.esm.parameters())
        self.assertLess(self.encoder.trainable_count(), total / 4)

    def test_features_match_the_frozen_path_shape_contract(self):
        import torch

        sequences = ["MKTAYIAKQR", "MKTAYIAKQRQISFVKSHF"]
        features, mask, lengths = self.encoder(sequences)
        self.assertEqual(tuple(features.shape), (2, 19, 640))
        self.assertEqual(lengths, [10, 19])
        # Padding carries no mask, exactly as EmbeddingBatcher produces.
        self.assertEqual(mask[0, :10].sum().item(), 10.0)
        self.assertEqual(mask[0, 10:].sum().item(), 0.0)
        self.assertTrue(torch.isfinite(features).all())

    def test_gradients_reach_the_trainable_layers(self):
        import torch

        features, _, _ = self.encoder(["MKTAYIAKQRQISFVKSHF"])
        features.sum().backward()
        top = self.encoder.esm.layers[29]
        grads = [p.grad for p in top.parameters() if p.grad is not None]
        self.assertTrue(grads, "the top layer received no gradient")
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)
        # ...and not the frozen ones.
        bottom = self.encoder.esm.layers[0]
        self.assertTrue(all(p.grad is None for p in bottom.parameters()))

    def test_a_chain_longer_than_the_window_is_stitched_differentiably(self):
        import torch

        narrow = ft.build_trainable_encoder(
            ft.FinetuneConfig(
                model_name=SMALL, trainable_layers=1,
                gradient_checkpointing=False, torch_home=CACHE,
                window_size=32, window_overlap=8, max_tokens_per_forward=512,
            )
        )
        sequence = "MKTAYIAKQR" * 9  # 90 residues over 32-residue windows
        features, mask, lengths = narrow(sequence and [sequence])
        self.assertEqual(tuple(features.shape), (1, 90, 640))
        self.assertEqual(mask.sum().item(), 90.0)
        self.assertTrue(torch.isfinite(features).all())
        features.sum().backward()
        top = narrow.esm.layers[29]
        self.assertTrue(any(p.grad is not None for p in top.parameters()))

    def test_checkpointing_does_not_change_the_representation(self):
        import torch

        plain = self.encoder(["MKTAYIAKQRQISFVKSHF"])[0]
        checkpointed = ft.build_trainable_encoder(
            ft.FinetuneConfig(
                model_name=SMALL, trainable_layers=2,
                gradient_checkpointing=True, torch_home=CACHE,
            )
        )(["MKTAYIAKQRQISFVKSHF"])[0]
        torch.testing.assert_close(plain, checkpointed, rtol=1e-4, atol=1e-5)


@slow_test
class ModelIntegrationTests(unittest.TestCase):
    def test_encode_refuses_examples_without_sequences(self):
        import torch

        from domaino.train.trainer import CompleteModel

        config = TrainConfig(
            architecture="pairwise", hidden_dim=16, dropout=0.0,
            finetune_encoder=True, finetune_layers=1,
            gradient_checkpointing=False, encoder_name=SMALL,
            torch_home=CACHE, train_classifier=False,
        )
        model = CompleteModel(config, 640, None, torch.device("cpu"))
        with self.assertRaises(ValueError) as caught:
            model.encode([tdata.ChainExample("a", 10, [[(1, 10)]])])
        self.assertIn("with_sequences=True", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
