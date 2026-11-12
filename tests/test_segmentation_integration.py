"""Segmentation network and decoder working together (milestone M2.2).

M2.2 asks that the segmentation and clustering components operate together on
controlled examples, and M2.3 that the architecture fits a small controlled
dataset. These tests build a tiny network, fit it to a handful of synthetic
chains whose domain structure is known, then decode its outputs and check the
domains come back.

The point is not accuracy on real proteins -- that is work package 3 -- but
that the pieces compose: targets feed the network, the network's two heads feed
the decoder, and the decoder returns the structure that went in, including a
discontinuous domain.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino import metrics  # noqa: E402
from domaino.model import decode, segmentation as seg  # noqa: E402

# Controlled chains: one single-domain, two two-domain, one discontinuous.
CASES = [
    (160, [[(1, 160)]]),
    (200, [[(1, 100)], [(101, 200)]]),
    (240, [[(1, 60), (181, 240)], [(61, 180)]]),
    (220, [[(1, 90)], [(121, 220)]]),
]

EMBED_DIM = 12


def synthetic_embedding(length, domains, rng):
    """Embeddings that carry each residue's domain identity, plus noise.

    The signal is deliberately learnable: this checks that the components
    compose, not that the task is hard.
    """
    import numpy as np

    x = rng.normal(0.0, 0.1, size=(length, EMBED_DIM)).astype("float32")
    for index, segments in enumerate(domains):
        channel = index % (EMBED_DIM - 2)
        for start, end in segments:
            x[start - 1 : end, channel] += 1.0
    labels = seg.residue_targets(length, domains, boundary_width=2)
    for position, label in enumerate(labels):
        if label == decode.LINKER:
            x[position, EMBED_DIM - 1] += 1.0
        elif label == decode.BOUNDARY:
            x[position, EMBED_DIM - 2] += 1.0
    return x


class ControlledFitTests(unittest.TestCase):
    def fit(self, steps=400):
        """Fit a small network to the controlled chains."""
        import numpy as np
        import torch

        rng = np.random.default_rng(0)
        torch.manual_seed(0)

        model = seg.build_network(
            seg.SegmentationConfig(
                input_dim=EMBED_DIM,
                hidden_dim=48,
                dilations=(1, 2, 4, 8),
                pair_rank=16,
                dropout=0.0,
            )
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)

        batch = []
        for length, domains in CASES:
            x = synthetic_embedding(length, domains, rng)
            residue = seg.residue_targets(length, domains, boundary_width=2)
            target, mask = seg.pair_targets(length, domains)
            rows, cols = seg.sample_pairs(mask, limit=8000, seed=0)
            batch.append(
                {
                    "x": torch.from_numpy(x).unsqueeze(0),
                    "residue": torch.tensor(residue).unsqueeze(0),
                    "rows": torch.from_numpy(rows),
                    "cols": torch.from_numpy(cols),
                    "pair": torch.from_numpy(target[rows, cols]),
                    "domains": domains,
                    "length": length,
                }
            )

        first_loss = None
        model.train()
        for step in range(steps):
            total = 0.0
            optimiser.zero_grad()
            for item in batch:
                out = model(item["x"], pairs=(item["rows"], item["cols"]))
                result = seg.segmentation_loss(
                    out["residue_logits"],
                    item["residue"],
                    out["pair_logits"].reshape(-1),
                    item["pair"],
                )
                result["loss"].backward()
                total += float(result["loss"])
            optimiser.step()
            if step == 0:
                first_loss = total
        return model, batch, first_loss, total

    def test_loss_decreases_on_controlled_examples(self):
        _, _, first, last = self.fit(steps=200)
        self.assertLess(last, first)

    def test_decoder_recovers_the_controlled_domains(self):
        import torch

        model, batch, _, _ = self.fit(steps=400)
        model.eval()
        decoder = decode.DomainDecoder(
            decode.DecoderConfig(min_domain_length=30, max_linker_gap=10)
        )

        predictions, targets = [], []
        for item in batch:
            with torch.no_grad():
                out = model(item["x"])
                residue = torch.softmax(out["residue_logits"][0], dim=-1).numpy()
                pair = torch.sigmoid(out["pair_logits"][0]).numpy()
            result = decoder.decode(residue, pair)
            predictions.append([list(d) for d in result.domains])
            targets.append([list(d) for d in item["domains"]])

        scores = metrics.score_segmentation(predictions, targets)
        # Fitting four chains it has seen 400 times, the model should parse
        # them well; a low score here means the pieces are not composing.
        self.assertGreater(scores.mean_iou, 0.8)
        self.assertEqual(scores.chains, len(CASES))

    def test_domain_count_is_about_right(self):
        import torch

        model, batch, _, _ = self.fit(steps=400)
        model.eval()
        decoder = decode.DomainDecoder(
            decode.DecoderConfig(min_domain_length=30, max_linker_gap=10)
        )
        for item in batch:
            with torch.no_grad():
                out = model(item["x"])
                residue = torch.softmax(out["residue_logits"][0], dim=-1).numpy()
                pair = torch.sigmoid(out["pair_logits"][0]).numpy()
            result = decoder.decode(residue, pair)
            expected = len(item["domains"])
            self.assertLessEqual(
                abs(result.n_domains - expected),
                1,
                f"expected about {expected} domains, got {result.n_domains}",
            )

    def test_pipeline_runs_on_a_long_chain_without_a_dense_matrix_in_training(self):
        # The pairwise design's cost risk: training must stay on sampled pairs.
        import torch

        model = seg.build_network(
            seg.SegmentationConfig(
                input_dim=EMBED_DIM, hidden_dim=32, dilations=(1, 2), pair_rank=16
            )
        )
        length = 3000
        x = torch.randn(1, length, EMBED_DIM)
        rows = torch.randint(0, length, (5000,))
        cols = torch.randint(0, length, (5000,))
        out = model(x, pairs=(rows, cols))
        self.assertEqual(tuple(out["pair_logits"].shape), (1, 5000))
        self.assertEqual(tuple(out["residue_logits"].shape), (1, length, 3))


if __name__ == "__main__":
    unittest.main()
