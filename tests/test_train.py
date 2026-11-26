"""Tests for the training protocol and data pipeline (T3.1, T3.2, T2.6, T2.7)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.train import data as tdata  # noqa: E402
from domaino.train import protocol  # noqa: E402
from domaino.train.trainer import ARCHITECTURES, TrainConfig  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_selection_metric_is_a_validation_metric(self):
        # The proposal requires the final model to be chosen on validation
        # data without inspecting the external benchmarks, so the selection
        # rule must not name a test metric.
        self.assertTrue(protocol.CHECKPOINT_METRIC.startswith("val_"))
        self.assertIn("validation split only", protocol.protocol_summary()["selection_uses"])

    def test_run_id_is_stable_and_sensitive(self):
        a = protocol.run_id({"lr": 0.001, "arch": "pairwise"})
        b = protocol.run_id({"arch": "pairwise", "lr": 0.001})
        c = protocol.run_id({"lr": 0.002, "arch": "pairwise"})
        self.assertEqual(a, b)  # key order must not matter
        self.assertNotEqual(a, c)  # a changed setting must be a new run

    def test_is_better_follows_the_mode(self):
        self.assertTrue(protocol.is_better(0.5, None))
        self.assertTrue(protocol.is_better(0.6, 0.5))
        self.assertFalse(protocol.is_better(0.4, 0.5))

    def test_chain_groups_cover_the_required_breakdown(self):
        # T3.2 requires results separated by domain count and domain type.
        self.assertEqual(
            sorted(protocol.chain_groups(1, False, 100)),
            ["all", "continuous", "single_domain"],
        )
        self.assertEqual(
            sorted(protocol.chain_groups(3, True, 900)),
            ["all", "discontinuous", "long", "multidomain"],
        )

    def test_every_group_name_is_declared(self):
        for n, disc, length in ((1, False, 10), (2, True, 900), (5, False, 600)):
            for group in protocol.chain_groups(n, disc, length):
                self.assertIn(group, protocol.REPORT_GROUPS)

    def test_registry_round_trip_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = protocol.ExperimentRegistry(directory)
            self.assertEqual(registry.spent_gpu_hours(), 0.0)
            record = protocol.RunRecord(
                run_id="t32-abc", name="x", architecture="pairwise",
                dataset_version=protocol.DATASET_VERSION, seed=1, gpu_hours=2.5,
            )
            registry.save(record)
            restored = registry.load("t32-abc")
            self.assertEqual(restored.architecture, "pairwise")
            self.assertEqual(registry.spent_gpu_hours(), 2.5)
            self.assertAlmostEqual(
                registry.budget_remaining(), protocol.BUDGET_GPU_HOURS - 2.5, places=3
            )

    def test_registry_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "summary.json"), "w") as handle:
                handle.write("{}")
            with open(os.path.join(directory, "notes.txt"), "w") as handle:
                handle.write("hello")
            self.assertEqual(protocol.ExperimentRegistry(directory).all_runs(), [])

    def test_code_revision_reports_dirtiness(self):
        info = protocol.code_revision(".")
        self.assertIn("revision", info)
        self.assertIn(info["dirty"], ("yes", "no", "unknown"))

    def test_stopwatch_tracks_gpu_hours(self):
        watch = protocol.Stopwatch(gpus=2)
        self.assertGreaterEqual(watch.gpu_hours, 0.0)
        self.assertFalse(watch.exceeded(hours=10.0))
        self.assertTrue(watch.exceeded(hours=0.0))


class TrainConfigTests(unittest.TestCase):
    def test_unknown_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainConfig(architecture="transformer")

    def test_epochs_cannot_exceed_the_protocol_cap(self):
        # The budget is part of the protocol, so a run cannot quietly ask for
        # more epochs than it fixed.
        with self.assertRaises(ValueError):
            TrainConfig(epochs=protocol.MAX_EPOCHS + 1)

    def test_accumulate_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            TrainConfig(accumulate_steps=0)

    def test_both_architectures_accept_the_same_config(self):
        base = dict(epochs=2, dropout=0.1, max_tokens=8000)
        for architecture in ARCHITECTURES:
            config = TrainConfig(architecture=architecture, **base)
            self.assertEqual(config.max_tokens, 8000)
            self.assertEqual(config.dropout, 0.1)


class BatchingTests(unittest.TestCase):
    def examples(self, lengths):
        return [
            tdata.ChainExample(f"c{i}", length, [[(1, length)]])
            for i, length in enumerate(lengths)
        ]

    def test_every_example_appears_exactly_once(self):
        examples = self.examples([10, 500, 30, 1000, 70, 200])
        batches = tdata.batch_by_tokens(examples, 2000, shuffle=False)
        flat = sorted(i for b in batches for i in b)
        self.assertEqual(flat, list(range(len(examples))))

    def test_padded_batch_respects_the_token_budget(self):
        lengths = [100, 110, 120, 900, 950]
        examples = self.examples(lengths)
        for batch in tdata.batch_by_tokens(examples, 2000, shuffle=False):
            padded = max(lengths[i] for i in batch) * len(batch)
            self.assertLessEqual(padded, 2000)

    def test_a_single_oversized_chain_still_gets_a_batch(self):
        batches = tdata.batch_by_tokens(self.examples([4128]), 1000, shuffle=False)
        self.assertEqual(batches, [[0]])

    def test_shuffling_is_seeded(self):
        examples = self.examples(list(range(50, 300, 10)))
        a = tdata.batch_by_tokens(examples, 600, shuffle=True, seed=3)
        b = tdata.batch_by_tokens(examples, 600, shuffle=True, seed=3)
        self.assertEqual(a, b)


class ChainExampleTests(unittest.TestCase):
    def test_structure_flags(self):
        single = tdata.ChainExample("a", 100, [[(1, 100)]])
        self.assertEqual(single.n_domains, 1)
        self.assertFalse(single.discontinuous)
        split = tdata.ChainExample("b", 240, [[(1, 60), (181, 240)], [(61, 180)]])
        self.assertEqual(split.n_domains, 2)
        self.assertTrue(split.discontinuous)


class EmbeddingBatcherTests(unittest.TestCase):
    def store(self, directory, chains):
        import numpy as np

        from domaino.model.embedding_store import EmbeddingWriter, StoreMeta

        meta = StoreMeta(model_name="esm2_t30_150M_UR50D", layer=30, dim=4)
        with EmbeddingWriter(directory, meta) as writer:
            for chain_id, length in chains:
                writer.add(chain_id, np.full((length, 4), float(length), dtype=np.float32))
        from domaino.model.embedding_store import EmbeddingStore

        return EmbeddingStore(directory)

    def test_batch_is_padded_and_masked(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(directory, [("a", 10), ("b", 4)])
            batcher = tdata.EmbeddingBatcher(store)
            examples = [
                tdata.ChainExample("a", 10, [[(1, 10)]]),
                tdata.ChainExample("b", 4, [[(1, 4)]]),
            ]
            features, mask, lengths = batcher.build(examples)
            self.assertEqual(tuple(features.shape), (2, 10, 4))
            self.assertEqual(lengths, [10, 4])
            # The short chain's tail is padding and must be masked off.
            self.assertEqual(float(mask[1, :4].sum()), 4.0)
            self.assertEqual(float(mask[1, 4:].sum()), 0.0)
            self.assertEqual(float(features[1, 4:].abs().sum()), 0.0)

    def test_values_land_in_the_right_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(directory, [("a", 6), ("b", 3)])
            batcher = tdata.EmbeddingBatcher(store)
            features, _, _ = batcher.build(
                [
                    tdata.ChainExample("a", 6, [[(1, 6)]]),
                    tdata.ChainExample("b", 3, [[(1, 3)]]),
                ]
            )
            self.assertEqual(float(features[0, 0, 0]), 6.0)
            self.assertEqual(float(features[1, 0, 0]), 3.0)

    def test_a_shorter_cached_row_does_not_shift_residues(self):
        # If the cache held fewer rows than the record declares, copying
        # blindly would shift every residue; the extra positions stay masked.
        with tempfile.TemporaryDirectory() as directory:
            store = self.store(directory, [("a", 5)])
            batcher = tdata.EmbeddingBatcher(store)
            features, mask, _ = batcher.build(
                [tdata.ChainExample("a", 8, [[(1, 8)]])]
            )
            self.assertEqual(tuple(features.shape), (1, 8, 4))
            self.assertEqual(float(mask[0].sum()), 5.0)


class EvaluateTests(unittest.TestCase):
    """End-to-end evaluation, with the classifier on.

    A unit test of the parts missed a variable in `evaluate` that shadowed the
    list of (example, prediction) pairs once classification ran, so the group
    breakdown read a classifier output as if it were a chain. Only exercising
    the whole function catches that.
    """

    def build(self, architecture):
        import numpy as np
        import torch

        from domaino.model import classifier as clf
        from domaino.model.embedding_store import (
            EmbeddingStore, EmbeddingWriter, StoreMeta,
        )
        from domaino.train.trainer import CompleteModel, TrainConfig, evaluate

        directory = tempfile.mkdtemp()
        dim = 8
        chains = [
            ("s1", 120, [[(1, 120)]], ["1.10.10.10"]),
            ("m1", 200, [[(1, 100)], [(101, 200)]], ["1.10.10.10", "3.40.50.300"]),
            ("d1", 240, [[(1, 60), (181, 240)], [(61, 180)]],
             ["1.10.10.10", "3.40.50.300"]),
            ("l1", 600, [[(1, 600)]], ["3.40.50.300"]),
        ]
        meta = StoreMeta(model_name="esm2_t30_150M_UR50D", layer=30, dim=dim)
        with EmbeddingWriter(directory, meta) as writer:
            for chain_id, length, _, _ in chains:
                writer.add(
                    chain_id, np.random.RandomState(0).randn(length, dim).astype("float32")
                )
        store = EmbeddingStore(directory)

        examples = []
        for chain_id, length, domains, h_codes in chains:
            labels = {
                "C": [c.split(".")[0] for c in h_codes],
                "A": [".".join(c.split(".")[:2]) for c in h_codes],
                "T": [".".join(c.split(".")[:3]) for c in h_codes],
                "H": list(h_codes),
            }
            examples.append(tdata.ChainExample(chain_id, length, domains, labels))

        all_labels = {level: [] for level in ("C", "A", "T", "H")}
        for example in examples:
            for level in all_labels:
                all_labels[level].extend(example.labels[level])
        hierarchy = clf.CathHierarchy.from_labels(all_labels)

        config = TrainConfig(
            architecture=architecture, epochs=1, max_tokens=4000,
            hidden_dim=32, dropout=0.0, mixed_precision=False,
            label_min_h_support=0,
        )
        model = CompleteModel(config, dim, hierarchy, torch.device("cpu"))
        batcher = tdata.EmbeddingBatcher(store, device=torch.device("cpu"))
        return evaluate(model, examples, batcher, config)

    def test_evaluate_reports_every_group_for_both_architectures(self):
        for architecture in ARCHITECTURES:
            scores = self.build(architecture)
            segmentation = scores["segmentation"]
            for group in ("all", "single_domain", "multidomain",
                          "discontinuous", "continuous", "long"):
                self.assertIn(group, segmentation, f"{architecture}/{group}")
            # Four chains, each counted once in "all".
            self.assertEqual(segmentation["all"]["chains"], 4, architecture)
            self.assertEqual(segmentation["multidomain"]["chains"], 2, architecture)
            self.assertEqual(segmentation["discontinuous"]["chains"], 1, architecture)
            self.assertEqual(segmentation["long"]["chains"], 1, architecture)

    def test_evaluate_reports_classification_with_coverage(self):
        scores = self.build("pairwise")
        classification = scores["classification_true_segments"]
        # 1 + 2 + 2 + 1 domains across the four chains.
        self.assertEqual(classification["domains"], 6)
        self.assertIn("h_coverage", classification)
        self.assertLessEqual(classification["h_coverage"], 1.0)
        for level in ("C", "A", "T", "H"):
            self.assertIn(level, classification["levels"])


class SourceAwareStepTests(unittest.TestCase):
    """The trainer's T2.6 path, on a batch that mixes sources.

    `test_losses.py` checks the terms in isolation; these check that the
    trainer actually hands each chain the mask its source implies, which is
    where a mixed batch would otherwise quietly train Pfam silence as linker.
    """

    def build(self, architecture, sources, source_aware=True):
        import torch

        from domaino.model import losses
        from domaino.model.trunk import TrunkConfig  # noqa: F401
        from domaino.train.trainer import (
            CompleteModel, TrainConfig, _classification_step, _segmentation_step,
        )

        torch.manual_seed(0)
        dim = 8
        length = 60
        examples = [
            tdata.ChainExample(
                f"c{index}", length, [[(5, 25)], [(30, 55)]],
                {level: ["1.10.10.10", "3.40.50.300"] for level in ("C", "A", "T", "H")},
                source=source,
            )
            for index, source in enumerate(sources)
        ]
        config = TrainConfig(
            architecture=architecture, epochs=1, hidden_dim=16, dropout=0.0,
            mixed_precision=False, pairs_per_chain=200, source_aware=source_aware,
            label_min_h_support=0,
        )
        model = CompleteModel(config, dim, None, torch.device("cpu"))
        features = torch.randn(len(examples), length, dim)
        mask = torch.ones(len(examples), length)
        loss, out, parts = _segmentation_step(
            model, features, mask, examples, config
        )
        return loss, parts, examples, config, model, out

    def test_pairwise_reports_the_three_terms_and_backpropagates(self):
        loss, parts, _, _, model, _ = self.build("pairwise", ["cath", "cath"])
        self.assertEqual(
            sorted(k for k in parts if k != "chains_without_pairs"),
            ["assignment", "boundary", "pairwise"],
        )
        loss.backward()
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0
                for p in model.segmentation.parameters())
        )

    def test_a_pfam_chain_contributes_no_pairs(self):
        _, parts, _, _, _, _ = self.build("pairwise", ["cath", "pfam", "pfam"])
        self.assertEqual(parts["chains_without_pairs"], 2)
        # ...but it still contributes to the residue terms.
        self.assertGreater(parts["assignment"], 0.0)

    def test_an_all_pfam_batch_still_trains_the_residue_terms(self):
        loss, parts, _, _, _, _ = self.build("pairwise", ["pfam", "pfam"])
        self.assertNotIn("pairwise", parts)
        self.assertGreater(float(loss), 0.0)
        loss.backward()

    def test_the_query_architecture_masks_whole_chains(self):
        # Set prediction needs the complete set of domains, so an incomplete
        # source is dropped rather than partly masked.
        _, parts, _, _, _, _ = self.build("query", ["cath", "pfam"])
        self.assertEqual(parts["chains_masked_out"], 1)

    def test_an_all_pfam_batch_is_a_no_op_for_the_query_architecture(self):
        loss, parts, _, _, _, _ = self.build("query", ["pfam", "pfam"])
        self.assertEqual(parts["chains_masked_out"], 2)
        self.assertEqual(float(loss), 0.0)
        # Still differentiable: a batch a source mask emptied must not break
        # the backward pass for the rest of the epoch.
        loss.backward()

    def test_the_legacy_loss_is_unchanged_by_the_new_fields(self):
        # The T3.2 runs in the registry used this path; it must stay reachable
        # and must not consult the source masks.
        plain, parts, _, _, _, _ = self.build(
            "pairwise", ["cath", "pfam"], source_aware=False
        )
        self.assertEqual(sorted(parts), ["pair", "residue"])

    def test_classification_skips_sources_without_cath_labels(self):
        import torch

        from domaino.model import classifier as clf
        from domaino.train.trainer import CompleteModel, TrainConfig, _classification_step

        labels = {level: ["1.10.10.10", "3.40.50.300"] for level in ("C", "A", "T", "H")}
        hierarchy = clf.CathHierarchy.from_labels(
            {level: list(values) for level, values in labels.items()}
        )
        config = TrainConfig(
            architecture="pairwise", epochs=1, hidden_dim=16, dropout=0.0,
            mixed_precision=False, source_aware=True, label_min_h_support=0,
        )
        model = CompleteModel(config, 8, hierarchy, torch.device("cpu"))
        examples = [
            tdata.ChainExample("a", 60, [[(5, 25)], [(30, 55)]], labels, source="pfam"),
        ]
        features = torch.randn(1, 60, config.hidden_dim)
        loss, parts = _classification_step(model, features, examples, config)
        self.assertIsNone(loss)
        # The same chain from CATH does train the classifier.
        cath = [
            tdata.ChainExample("a", 60, [[(5, 25)], [(30, 55)]], labels, source="cath"),
        ]
        loss, parts = _classification_step(model, features, cath, config)
        self.assertIsNotNone(loss)


class SupervisionCarryTests(unittest.TestCase):
    def test_an_example_falls_back_to_its_source_policy(self):
        example = tdata.ChainExample("a", 10, [[(1, 5)]], source="pfam")
        self.assertFalse(example.effective_supervision.pairwise)
        self.assertEqual(example.effective_supervision.weight, 0.3)

    def test_a_records_own_supervision_wins(self):
        from domaino.data.schema import Supervision

        supervision = Supervision(
            residue_labels=True, pairwise=False, cath_labels=False,
            complete_partition=False, weight=0.07,
        )
        example = tdata.ChainExample(
            "a", 10, [[(1, 5)]], source="ted", supervision=supervision
        )
        self.assertEqual(example.effective_supervision.weight, 0.07)

    def test_the_default_source_is_cath(self):
        # Every example built before T2.6 existed is a CATH chain, so the
        # default must not change what those examples supervise.
        example = tdata.ChainExample("a", 10, [[(1, 5)]])
        self.assertTrue(example.effective_supervision.complete_partition)
        self.assertTrue(example.effective_supervision.pairwise)


if __name__ == "__main__":
    unittest.main()
