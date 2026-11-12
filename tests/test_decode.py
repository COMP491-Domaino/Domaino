"""Tests for converting network outputs into domain assignments (T2.4)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.model import decode  # noqa: E402
from domaino.model.decode import BOUNDARY, DOMAIN, LINKER  # noqa: E402


def residue_probs(length, linker=(), boundary=()):
    """Confident residue probabilities with the given linker/boundary sets."""
    import numpy as np

    probs = np.zeros((length, 3), dtype=np.float64)
    probs[:, DOMAIN] = 0.9
    probs[:, LINKER] = 0.05
    probs[:, BOUNDARY] = 0.05
    for position in linker:
        probs[position - 1] = [0.9, 0.05, 0.05]
    for position in boundary:
        probs[position - 1] = [0.05, 0.05, 0.9]
    return probs


def pair_probs_from_groups(length, groups, same=0.95, other=0.02):
    """A same-domain matrix consistent with a known grouping."""
    import numpy as np

    owner = np.full(length, -1, dtype=int)
    for index, intervals in enumerate(groups):
        for start, end in intervals:
            owner[start - 1 : end] = index
    matrix = np.full((length, length), other, dtype=np.float64)
    for index in range(len(groups)):
        rows = np.nonzero(owner == index)[0]
        if len(rows):
            matrix[np.ix_(rows, rows)] = same
    return matrix


class RunTests(unittest.TestCase):
    def test_runs_of_true(self):
        self.assertEqual(decode._runs([True, True, False, True]), [(1, 2), (4, 4)])

    def test_all_false_and_all_true(self):
        self.assertEqual(decode._runs([False, False]), [])
        self.assertEqual(decode._runs([True, True]), [(1, 2)])


class BoundaryCutTests(unittest.TestCase):
    def test_single_peak(self):
        probs = [0.0] * 10
        probs[4] = 0.9
        self.assertEqual(decode.boundary_cuts(probs, 0.5), [5])

    def test_a_band_yields_one_cut_at_its_peak(self):
        # The network marks a band around each edge; one edge should give one
        # cut, not one per residue above threshold.
        probs = [0.0] * 20
        probs[8], probs[9], probs[10] = 0.6, 0.9, 0.6
        self.assertEqual(decode.boundary_cuts(probs, 0.5), [10])

    def test_below_threshold_is_ignored(self):
        probs = [0.0] * 10
        probs[4] = 0.3
        self.assertEqual(decode.boundary_cuts(probs, 0.5), [])


class BlockSplitTests(unittest.TestCase):
    def test_linker_runs_separate_blocks(self):
        assigned = [True] * 50 + [False] * 10 + [True] * 50
        self.assertEqual(
            decode.split_into_blocks(assigned, [], 5), [(1, 50), (61, 110)]
        )

    def test_cuts_split_touching_domains(self):
        # Adjacent CATH domains often touch with no linker between them, so a
        # boundary cut is the only way to separate them.
        assigned = [True] * 100
        self.assertEqual(
            decode.split_into_blocks(assigned, [50], 5), [(1, 50), (51, 100)]
        )

    def test_tiny_blocks_are_dropped(self):
        assigned = [True] * 3 + [False] + [True] * 50
        self.assertEqual(decode.split_into_blocks(assigned, [], 5), [(5, 54)])

    def test_cut_at_the_end_does_not_create_an_empty_block(self):
        assigned = [True] * 50
        self.assertEqual(decode.split_into_blocks(assigned, [50], 5), [(1, 50)])

    def test_terminal_cuts_are_refused_rather_than_losing_residues(self):
        # A chain's own first and last residues are domain edges, so the
        # network always proposes a cut there. Honouring it would make a
        # one-residue block that is then dropped, trimming the chain's ends.
        assigned = [True] * 100
        self.assertEqual(
            decode.split_into_blocks(assigned, [1, 99], 5), [(1, 100)]
        )

    def test_a_cut_is_kept_when_both_sides_are_viable(self):
        assigned = [True] * 100
        self.assertEqual(
            decode.split_into_blocks(assigned, [1, 50, 99], 5),
            [(1, 50), (51, 100)],
        )

    def test_whole_single_domain_chain_is_returned_intact(self):
        # The end-to-end symptom of the bug above: 1-154 came back as 2-154.
        decoder = decode.DomainDecoder(decode.DecoderConfig())
        probs = residue_probs(154, boundary=[1, 2, 3, 152, 153, 154])
        result = decoder.decode(probs, pair_probs_from_groups(154, [[(1, 154)]]))
        self.assertEqual(result.domains, [[(1, 154)]])
        self.assertEqual(result.coverage, 1.0)


class ClusterTests(unittest.TestCase):
    def test_blocks_of_one_domain_are_grouped(self):
        blocks = [(1, 50), (51, 100)]
        pair = pair_probs_from_groups(100, [[(1, 100)]])
        self.assertEqual(decode.cluster_blocks(pair, blocks, 0.5), [[0, 1]])

    def test_blocks_of_two_domains_stay_apart(self):
        blocks = [(1, 50), (51, 100)]
        pair = pair_probs_from_groups(100, [[(1, 50)], [(51, 100)]])
        groups = decode.cluster_blocks(pair, blocks, 0.5)
        self.assertEqual(sorted(groups), [[0], [1]])

    def test_discontinuous_domain_groups_separated_blocks(self):
        # The reason the pairwise design exists: blocks 0 and 2 belong to one
        # domain even though block 1 sits between them in the sequence.
        blocks = [(1, 50), (51, 120), (121, 170)]
        pair = pair_probs_from_groups(
            170, [[(1, 50), (121, 170)], [(51, 120)]]
        )
        groups = decode.cluster_blocks(pair, blocks, 0.5)
        self.assertIn([0, 2], groups)
        self.assertIn([1], groups)

    def test_single_block_needs_no_clustering(self):
        pair = pair_probs_from_groups(50, [[(1, 50)]])
        self.assertEqual(decode.cluster_blocks(pair, [(1, 50)], 0.5), [[0]])

    def test_no_blocks(self):
        pair = pair_probs_from_groups(10, [[(1, 10)]])
        self.assertEqual(decode.cluster_blocks(pair, [], 0.5), [])


class MergeIntervalTests(unittest.TestCase):
    def test_close_intervals_merge(self):
        self.assertEqual(
            decode.merge_close_intervals([(1, 50), (56, 100)], max_gap=10),
            [(1, 100)],
        )

    def test_distant_intervals_stay_separate(self):
        self.assertEqual(
            decode.merge_close_intervals([(1, 50), (120, 170)], max_gap=10),
            [(1, 50), (120, 170)],
        )

    def test_zero_gap_joins_only_adjacent(self):
        self.assertEqual(
            decode.merge_close_intervals([(1, 50), (51, 90)], max_gap=0), [(1, 90)]
        )


class DecoderTests(unittest.TestCase):
    def decoder(self, **overrides):
        config = decode.DecoderConfig(**overrides)
        return decode.DomainDecoder(config)

    def test_single_domain_chain(self):
        result = self.decoder().decode(
            residue_probs(200), pair_probs_from_groups(200, [[(1, 200)]])
        )
        self.assertEqual(result.n_domains, 1)
        self.assertEqual(result.domains[0], [(1, 200)])

    def test_two_touching_domains_are_separated_by_a_boundary_cut(self):
        result = self.decoder().decode(
            residue_probs(200, boundary=[100]),
            pair_probs_from_groups(200, [[(1, 100)], [(101, 200)]]),
        )
        self.assertEqual(result.n_domains, 2)
        self.assertEqual(result.domains, [[(1, 100)], [(101, 200)]])

    def test_two_domains_separated_by_a_linker(self):
        result = self.decoder(max_linker_gap=5).decode(
            residue_probs(230, linker=list(range(101, 131))),
            pair_probs_from_groups(230, [[(1, 100)], [(131, 230)]]),
        )
        self.assertEqual(result.n_domains, 2)
        self.assertLess(result.coverage, 1.0)

    def test_discontinuous_domain_is_recovered(self):
        # Target: domain A = 1-60 and 181-240, domain B = 61-180.
        groups = [[(1, 60), (181, 240)], [(61, 180)]]
        result = self.decoder().decode(
            residue_probs(240, boundary=[60, 180]),
            pair_probs_from_groups(240, groups),
        )
        self.assertEqual(result.n_domains, 2)
        self.assertEqual(result.discontinuous, 1)
        recovered = sorted(result.domains, key=len, reverse=True)[0]
        self.assertEqual(recovered, [(1, 60), (181, 240)])

    def test_short_domain_is_absorbed_not_emitted(self):
        # A 20-residue fragment is below the minimum, so it must be folded into
        # a real domain rather than reported as one.
        result = self.decoder(min_domain_length=40).decode(
            residue_probs(220, boundary=[200]),
            pair_probs_from_groups(220, [[(1, 200)], [(201, 220)]]),
        )
        for domain in result.domains:
            self.assertGreaterEqual(decode.domain_length(domain), 40)
        self.assertTrue(any("absorbed" in n for n in result.notes))

    def test_short_linker_gap_is_absorbed_into_one_interval(self):
        result = self.decoder(max_linker_gap=20).decode(
            residue_probs(200, linker=list(range(96, 101))),
            pair_probs_from_groups(200, [[(1, 200)]]),
        )
        self.assertEqual(result.n_domains, 1)
        self.assertEqual(result.domains[0], [(1, 200)])

    def test_coverage_floor_extends_domains(self):
        # Most of the chain called linker: the parse is unusable, so leftovers
        # are given to their nearest domain.
        probs = residue_probs(200, linker=list(range(61, 181)))
        result = self.decoder(min_coverage=0.8, min_domain_length=10).decode(
            probs, pair_probs_from_groups(200, [[(1, 60), (181, 200)]])
        )
        self.assertGreaterEqual(result.coverage, 0.8)
        self.assertTrue(any("coverage" in n for n in result.notes))

    def test_all_linker_falls_back_to_the_whole_chain(self):
        # Every CATH chain has at least one domain, so refusing to answer is
        # never right.
        probs = residue_probs(100, linker=list(range(1, 101)))
        result = self.decoder().decode(probs, pair_probs_from_groups(100, []))
        self.assertEqual(result.domains, [[(1, 100)]])
        self.assertTrue(any("whole_chain" in n for n in result.notes))

    def test_output_is_always_a_valid_parse(self):
        import numpy as np

        rng = np.random.default_rng(0)
        for trial in range(25):
            length = int(rng.integers(60, 400))
            probs = rng.random((length, 3))
            probs /= probs.sum(axis=1, keepdims=True)
            pair = rng.random((length, length))
            pair = (pair + pair.T) / 2
            result = self.decoder().decode(probs, pair)
            claimed = []
            for domain in result.domains:
                for start, end in domain:
                    self.assertGreaterEqual(start, 1)
                    self.assertLessEqual(end, length)
                    self.assertLessEqual(start, end)
                    claimed.extend(range(start, end + 1))
            # No residue may belong to two domains.
            self.assertEqual(len(claimed), len(set(claimed)), f"overlap in trial {trial}")
            self.assertGreaterEqual(result.n_domains, 1)

    def test_rejects_malformed_inputs(self):
        import numpy as np

        decoder = self.decoder()
        with self.assertRaises(ValueError):
            decoder.decode(np.zeros((10, 2)), np.zeros((10, 10)))
        with self.assertRaises(ValueError):
            decoder.decode(np.zeros((10, 3)), np.zeros((9, 9)))


if __name__ == "__main__":
    unittest.main()
