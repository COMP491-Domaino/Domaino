"""Tests for the ESM-2 representation pipeline (T2.2).

The encoder itself needs the real weights and a GPU to be worth running, so
the tests here cover the logic around it -- windowing, batching, configuration
and the embedding cache -- which is where the mistakes that silently corrupt
training actually live. The encoder is exercised end to end by
`scripts/smoke_representation.py`.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.model import embedding_store as store_mod  # noqa: E402
from domaino.model import representation as rep  # noqa: E402


class WindowPlanTests(unittest.TestCase):
    def test_short_sequence_is_one_window(self):
        self.assertEqual(rep.plan_windows(100, 1022, 256), [(0, 100)])
        self.assertEqual(rep.plan_windows(1022, 1022, 256), [(0, 1022)])

    def test_long_sequence_is_covered_completely(self):
        length = 4128  # the longest chain in dataset_v1
        windows = rep.plan_windows(length, 1022, 256)
        covered = set()
        for start, end in windows:
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(length)))

    def test_windows_never_exceed_the_trained_crop(self):
        for start, end in rep.plan_windows(4128, 1022, 256):
            self.assertLessEqual(end - start, 1022)

    def test_last_window_ends_exactly_at_the_sequence_end(self):
        windows = rep.plan_windows(1500, 1022, 256)
        self.assertEqual(windows[-1][1], 1500)

    def test_windows_overlap_so_seams_get_averaged(self):
        windows = rep.plan_windows(2000, 1022, 256)
        self.assertGreater(len(windows), 1)
        for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
            self.assertLess(next_start, prev_end)


class BatchingTests(unittest.TestCase):
    def test_every_index_appears_exactly_once(self):
        lengths = [10, 500, 30, 1000, 70, 200]
        batches = rep.batch_by_tokens(lengths, max_tokens=2048)
        flat = sorted(i for b in batches for i in b)
        self.assertEqual(flat, list(range(len(lengths))))

    def test_padded_batch_stays_within_the_token_budget(self):
        lengths = [100, 110, 120, 900, 950]
        budget = 2048
        for batch in rep.batch_by_tokens(lengths, budget):
            padded = max(lengths[i] + 2 for i in batch) * len(batch)
            self.assertLessEqual(padded, budget)

    def test_similar_lengths_are_grouped_to_limit_padding_waste(self):
        # With a budget tight enough to force a split, short and long
        # sequences must not share a batch, or the short ones would be padded
        # out to the long one's width.
        lengths = [10, 1000, 12, 1002]
        batches = rep.batch_by_tokens(lengths, max_tokens=2048)
        self.assertEqual(len(batches), 2)
        for batch in batches:
            spread = max(lengths[i] for i in batch) - min(lengths[i] for i in batch)
            self.assertLess(spread, 100)

    def test_each_batch_is_contiguous_in_length_order(self):
        lengths = [300, 10, 1000, 50, 700, 20, 450]
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        rank = {index: position for position, index in enumerate(order)}
        for batch in rep.batch_by_tokens(lengths, max_tokens=1600):
            ranks = sorted(rank[i] for i in batch)
            self.assertEqual(ranks, list(range(ranks[0], ranks[0] + len(ranks))))

    def test_a_single_oversized_sequence_still_gets_a_batch(self):
        batches = rep.batch_by_tokens([5000], max_tokens=1024)
        self.assertEqual(batches, [[0]])


class ConfigTests(unittest.TestCase):
    def test_known_variants_report_their_width(self):
        self.assertEqual(
            rep.EncoderConfig(model_name="esm2_t33_650M_UR50D").embed_dim, 1280
        )
        self.assertEqual(
            rep.EncoderConfig(model_name="esm2_t30_150M_UR50D").embed_dim, 640
        )

    def test_unknown_variant_is_rejected_rather_than_guessed(self):
        with self.assertRaises(KeyError):
            rep.EncoderConfig(model_name="esm2_t99_9B_UR50D").embed_dim

    def test_overlap_must_be_smaller_than_the_window(self):
        with self.assertRaises(ValueError):
            rep.EncoderConfig(window_size=512, window_overlap=512)

    def test_batch_budget_must_fit_one_window(self):
        with self.assertRaises(ValueError):
            rep.EncoderConfig(window_size=1022, max_tokens_per_batch=512)


class EmbeddingStoreTests(unittest.TestCase):
    def meta(self, dim=4):
        return store_mod.StoreMeta(
            model_name="esm2_t30_150M_UR50D", layer=30, dim=dim
        )

    def test_round_trip_preserves_shape_and_values(self):
        import numpy as np

        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        b = np.ones((5, 4), dtype=np.float32) * 0.5
        with tempfile.TemporaryDirectory() as directory:
            with store_mod.EmbeddingWriter(directory, self.meta()) as writer:
                writer.add("chainA", a)
                writer.add("chainB", b)
            store = store_mod.EmbeddingStore(directory)
            self.assertEqual(len(store), 2)
            self.assertIn("chainA", store)
            np.testing.assert_allclose(store.get("chainA"), a)
            np.testing.assert_allclose(store.get("chainB"), b)

    def test_shards_rotate_without_losing_entries(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            with store_mod.EmbeddingWriter(
                directory, self.meta(), shard_rows=4
            ) as writer:
                for i in range(5):
                    writer.add(f"c{i}", np.full((3, 4), i, dtype=np.float32))
            store = store_mod.EmbeddingStore(directory)
            self.assertEqual(len(store), 5)
            for i in range(5):
                np.testing.assert_allclose(
                    store.get(f"c{i}"), np.full((3, 4), i, dtype=np.float32)
                )

    def test_wrong_width_is_rejected_on_write(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            with store_mod.EmbeddingWriter(directory, self.meta()) as writer:
                with self.assertRaises(ValueError):
                    writer.add("bad", np.zeros((3, 7), dtype=np.float32))

    def test_duplicate_key_is_rejected(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            with store_mod.EmbeddingWriter(directory, self.meta()) as writer:
                writer.add("dup", np.zeros((2, 4), dtype=np.float32))
                with self.assertRaises(KeyError):
                    writer.add("dup", np.zeros((2, 4), dtype=np.float32))

    def test_cache_from_a_different_encoder_is_refused(self):
        # Mixing 650M and 150M vectors, or two layers, would train a model on
        # incoherent inputs, so the mismatch must surface immediately.
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            with store_mod.EmbeddingWriter(directory, self.meta()) as writer:
                writer.add("c", np.zeros((2, 4), dtype=np.float32))
            store = store_mod.EmbeddingStore(directory)
            store.verify("esm2_t30_150M_UR50D", 30)
            with self.assertRaises(ValueError):
                store.verify("esm2_t33_650M_UR50D")
            with self.assertRaises(ValueError):
                store.verify("esm2_t30_150M_UR50D", 12)

    def test_missing_key_raises(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            with store_mod.EmbeddingWriter(directory, self.meta()) as writer:
                writer.add("present", np.zeros((2, 4), dtype=np.float32))
            with self.assertRaises(KeyError):
                store_mod.EmbeddingStore(directory).get("absent")

    def test_size_estimate(self):
        self.assertEqual(store_mod.estimate_bytes([100, 200], 1280), 300 * 1280 * 2)


if __name__ == "__main__":
    unittest.main()
