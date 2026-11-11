"""Homology-aware partitioning and benchmark-exclusion tests (task T1.5)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino.data import benchmarks, splits  # noqa: E402


class ComponentTests(unittest.TestCase):
    def test_chains_sharing_a_cluster_join_one_component(self):
        clusters = {
            "aaaaA": ["1.10.10.10.1"],
            "bbbbA": ["1.10.10.10.1"],
            "ccccA": ["2.20.20.20.5"],
        }
        component_of, members = splits.build_components(clusters)
        self.assertEqual(component_of["aaaaA"], component_of["bbbbA"])
        self.assertNotEqual(component_of["aaaaA"], component_of["ccccA"])
        self.assertEqual(len(members), 2)

    def test_multidomain_chain_links_two_clusters_transitively(self):
        # This transitivity is why grouping at superfamily level collapses:
        # a chain spanning two groups merges them.
        clusters = {
            "aaaaA": ["1.10.10.10.1"],
            "bbbbA": ["1.10.10.10.1", "3.30.30.30.2"],
            "ccccA": ["3.30.30.30.2"],
        }
        component_of, members = splits.build_components(clusters)
        self.assertEqual(len(members), 1)
        self.assertEqual(component_of["aaaaA"], component_of["ccccA"])

    def test_singleton_chains_form_their_own_components(self):
        clusters = {f"chain{i}": [f"1.1.1.1.{i}"] for i in range(5)}
        _, members = splits.build_components(clusters)
        self.assertEqual(len(members), 5)


class AssignmentTests(unittest.TestCase):
    def make_clusters(self, n=300):
        return {f"c{i:04d}A": [f"1.10.10.10.{i}"] for i in range(n)}

    def test_every_chain_is_assigned_exactly_one_partition(self):
        clusters = self.make_clusters()
        plan = splits.assign(clusters)
        self.assertEqual(set(plan.assignment), set(clusters))
        self.assertEqual(sum(plan.counts.values()), len(clusters))

    def test_ratios_are_approximately_respected(self):
        clusters = self.make_clusters(1000)
        plan = splits.assign(clusters, ratios=(0.8, 0.1, 0.1))
        self.assertAlmostEqual(plan.counts[splits.TRAIN] / 1000, 0.8, delta=0.02)
        self.assertAlmostEqual(plan.counts[splits.VALIDATION] / 1000, 0.1, delta=0.02)
        self.assertAlmostEqual(plan.counts[splits.TEST] / 1000, 0.1, delta=0.02)

    def test_assignment_is_deterministic_for_a_given_seed(self):
        clusters = self.make_clusters()
        first = splits.assign(clusters, seed=7).assignment
        second = splits.assign(clusters, seed=7).assignment
        self.assertEqual(first, second)
        self.assertEqual(
            splits.assign(clusters, seed=7).compute_digests(),
            splits.assign(clusters, seed=7).compute_digests(),
        )

    def test_homologues_never_span_partitions(self):
        # Ten clusters, each shared by three chains; all three must land
        # together or a near-identical relative would leak across the split.
        clusters = {}
        for group in range(10):
            for member in range(3):
                clusters[f"g{group}m{member}A"] = [f"2.20.20.20.{group}"]
        plan = splits.assign(clusters)
        for group in range(10):
            assigned = {
                plan.assignment[f"g{group}m{member}A"] for member in range(3)
            }
            self.assertEqual(len(assigned), 1)
        self.assertTrue(plan.leakage["clean"])

    def test_benchmark_test_chain_is_reserved_and_its_homologues_excluded(self):
        clusters = {
            "benchA": ["1.10.10.10.1"],
            "homolA": ["1.10.10.10.1"],
            "otherA": ["5.50.50.50.9"],
        }
        plan = splits.assign(clusters, chainsaw_test={"benchA"})
        self.assertEqual(plan.assignment["benchA"], splits.BENCHMARK_CHAINSAW)
        self.assertEqual(plan.assignment["homolA"], splits.EXCLUDED_HOMOLOGOUS)
        self.assertNotIn(
            plan.assignment["homolA"], (splits.TRAIN, splits.VALIDATION, splits.TEST)
        )
        self.assertEqual(plan.excluded_by_benchmark[splits.BENCHMARK_CHAINSAW], 1)

    def transitive_fixture(self):
        # bench -- direct shares a cluster with bench; distant reaches bench
        # only through direct, which is the case the two rules disagree on.
        return {
            "benchA": ["1.10.10.10.1"],
            "direcA": ["1.10.10.10.1", "2.20.20.20.2"],
            "distaA": ["2.20.20.20.2"],
        }

    def test_one_hop_keeps_a_chain_linked_only_transitively(self):
        plan = splits.assign(
            self.transitive_fixture(),
            chainsaw_test={"benchA"},
            benchmark_exclusion="one_hop",
        )
        self.assertEqual(plan.assignment["benchA"], splits.BENCHMARK_CHAINSAW)
        self.assertEqual(plan.assignment["direcA"], splits.EXCLUDED_HOMOLOGOUS)
        self.assertIn(
            plan.assignment["distaA"],
            (splits.TRAIN, splits.VALIDATION, splits.TEST),
        )
        self.assertTrue(plan.leakage["clean"])

    def test_component_rule_also_drops_the_transitive_chain(self):
        plan = splits.assign(
            self.transitive_fixture(),
            chainsaw_test={"benchA"},
            benchmark_exclusion="component",
        )
        self.assertEqual(plan.assignment["direcA"], splits.EXCLUDED_HOMOLOGOUS)
        self.assertEqual(plan.assignment["distaA"], splits.EXCLUDED_HOMOLOGOUS)
        self.assertEqual(plan.excluded_by_benchmark[splits.BENCHMARK_CHAINSAW], 2)

    def test_one_hop_never_leaves_a_benchmark_cluster_in_training(self):
        clusters = {f"c{i:03d}A": [f"1.10.10.10.{i % 20}"] for i in range(200)}
        plan = splits.assign(
            clusters, chainsaw_test={"c000A", "c050A"}, benchmark_exclusion="one_hop"
        )
        benchmark_clusters = set(clusters["c000A"]) | set(clusters["c050A"])
        for chain, partition in plan.assignment.items():
            if partition == splits.TRAIN:
                self.assertFalse(
                    benchmark_clusters & set(clusters[chain]),
                    f"{chain} shares a benchmark cluster but is in training",
                )
        self.assertTrue(plan.leakage["clean"])

    def test_unknown_exclusion_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            splits.assign({"aA": ["1.1.1.1.1"]}, benchmark_exclusion="whatever")

    def test_cathe2_test_chains_get_their_own_partition(self):
        clusters = {"c2A": ["1.10.10.10.1"], "farA": ["9.90.90.90.9"]}
        plan = splits.assign(clusters, cathe2_test={"c2A"})
        self.assertEqual(plan.assignment["c2A"], splits.BENCHMARK_CATHE2)

    def test_benchmark_chain_with_untrustworthy_labels_still_gets_evaluated(self):
        # Chainsaw brings its own ground truth, so a sound sequence is enough
        # even when our CATH labels for that chain are not usable for training.
        clusters = {"benchA": ["1.10.10.10.1"]}
        plan = splits.assign(
            clusters,
            unusable={"benchA"},
            evaluable={"benchA"},
            chainsaw_test={"benchA"},
        )
        self.assertEqual(plan.assignment["benchA"], splits.BENCHMARK_CHAINSAW)

    def test_benchmark_chain_without_a_sound_sequence_is_excluded(self):
        clusters = {"benchA": ["1.10.10.10.1"]}
        plan = splits.assign(
            clusters, unusable={"benchA"}, evaluable=set(), chainsaw_test={"benchA"}
        )
        self.assertEqual(plan.assignment["benchA"], splits.EXCLUDED_UNUSABLE)

    def test_untrustworthy_chain_is_kept_out_of_our_own_partitions(self):
        clusters = {"badA": ["1.10.10.10.1"], "okA": ["2.20.20.20.2"]}
        plan = splits.assign(clusters, unusable={"badA"}, evaluable={"badA", "okA"})
        self.assertEqual(plan.assignment["badA"], splits.EXCLUDED_UNUSABLE)
        self.assertIn(
            plan.assignment["okA"], (splits.TRAIN, splits.VALIDATION, splits.TEST)
        )

    def test_chains_without_a_cluster_are_held_out_of_every_partition(self):
        clusters = {"goodA": ["1.10.10.10.1"]}
        plan = splits.assign(clusters, no_cluster={"nocA"})
        self.assertEqual(plan.assignment["nocA"], splits.EXCLUDED_NO_CLUSTER)
        self.assertNotIn("nocA", {c for c, p in plan.assignment.items() if p == splits.TRAIN})

    def test_leakage_verifier_detects_a_deliberately_broken_split(self):
        clusters = {"aA": ["1.10.10.10.1"], "bA": ["1.10.10.10.1"]}
        plan = splits.assign(clusters)
        # Force the two homologues apart; the audit must notice.
        plan.assignment["aA"] = splits.TRAIN
        plan.assignment["bA"] = splits.TEST
        result = splits.verify(plan, clusters)
        self.assertFalse(result["clean"])
        self.assertEqual(result["clusters_leaking_train_into_eval"], 1)

    def test_summary_round_trip_preserves_the_audit_numbers(self):
        clusters = self.make_clusters(120)
        plan = splits.assign(clusters, chainsaw_test={"c0000A"})
        restored = splits.SplitPlan.from_summary(plan.summary())
        self.assertEqual(restored.counts, plan.counts)
        self.assertEqual(restored.component_count, plan.component_count)
        self.assertEqual(restored.leakage, plan.leakage)
        self.assertEqual(restored.digests, plan.digests)
        self.assertEqual(restored.excluded_by_benchmark, plan.excluded_by_benchmark)

    def test_digest_changes_when_membership_changes(self):
        clusters = self.make_clusters(50)
        plan = splits.assign(clusters)
        before = plan.compute_digests()
        moved = next(c for c, p in plan.assignment.items() if p == splits.TRAIN)
        plan.assignment[moved] = splits.TEST
        self.assertNotEqual(before, plan.compute_digests())


class BenchmarkLoaderTests(unittest.TestCase):
    def test_domain_ids_map_to_chain_names(self):
        self.assertEqual(
            benchmarks.chains_of_domains(["1oaiA00", "10gsA01", "10gsA02"]),
            {"1oaiA", "10gsA"},
        )

    def test_domain_id_pattern_matches_cath_identifiers(self):
        self.assertEqual(
            benchmarks.find_domain_ids("results for 1oaiA00 and 3frhA01 today"),
            {"1oaiA00", "3frhA01"},
        )

    def test_seven_digit_row_index_is_not_a_domain_id(self):
        # CATHe2's tables index rows with numbers like 1035679, which have the
        # same shape as a domain id. Accepting them would build a hold-out of
        # chains that do not exist, so a real id must contain a letter.
        self.assertFalse(benchmarks.is_domain_id("1035679"))
        self.assertTrue(benchmarks.is_domain_id("2vsqA04"))
        self.assertEqual(benchmarks.find_domain_ids("1035679,1035680"), set())

    def test_row_with_index_and_domain_yields_only_the_domain(self):
        row = "1035679,1035679,testsequence,2vsqA04,3.40.50.980,testsequence_0_2vsqA04"
        self.assertEqual(benchmarks.find_domain_ids(row), {"2vsqA04"})

    def test_missing_cathe2_id_file_yields_an_empty_holdout(self):
        self.assertEqual(benchmarks.load_cathe2_test_domains(None), set())
        self.assertEqual(
            benchmarks.load_cathe2_test_domains("/nonexistent/path.txt"), set()
        )


class Cathe2ArchiveTests(unittest.TestCase):
    """Selecting the wrong member would exclude the wrong chains, so the
    identifier count is checked against the known test-set size."""

    def make_archive(self, members):
        import tempfile
        import zipfile

        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "data.zip")
        with zipfile.ZipFile(path, "w") as archive:
            for name, body in members.items():
                archive.writestr(name, body)
        return path

    def ids(self, start, count):
        # Distinct, well-formed CATH domain ids: 4-char pdb code + chain + 2.
        return "\n".join(
            f"1a{i // 26:1x}{chr(ord('a') + i % 26)}A{j:02d}"
            for i, j in ((k // 90, k % 90) for k in range(start, start + count))
        )

    def test_picks_the_member_whose_count_matches_the_expected_size(self):
        path = self.make_archive(
            {
                "data/Dataset/csv/Test.csv": "domain\n" + self.ids(0, 300),
                "data/Dataset/csv/Train_test_split.csv": "domain\n" + self.ids(500, 4000),
                "data/Dataset/embeddings/Test_ProstT5.npz": "binary-ish",
            }
        )
        domains, report = benchmarks.extract_cathe2_test_domains(
            path, expected_count=300
        )
        self.assertEqual(report["chosen"], "data/Dataset/csv/Test.csv")
        self.assertEqual(len(domains), 300)

    def test_refuses_when_no_candidate_is_near_the_expected_size(self):
        # A member holding the training identifiers must not be accepted:
        # excluding the wrong set is worse than excluding nothing.
        path = self.make_archive(
            {"data/Dataset/csv/Train_test_split.csv": "d\n" + self.ids(0, 4000)}
        )
        domains, report = benchmarks.extract_cathe2_test_domains(
            path, expected_count=300
        )
        self.assertEqual(domains, set())
        self.assertIsNone(report["chosen"])
        self.assertIn("refusing", str(report["reason"]))

    def test_reports_when_nothing_looks_like_a_test_table(self):
        path = self.make_archive({"data/Dataset/embeddings/Train.npz": "x"})
        domains, report = benchmarks.extract_cathe2_test_domains(path, 300)
        self.assertEqual(domains, set())
        self.assertEqual(report["candidates"], [])

    def test_named_domain_column_is_preferred_over_scanning_the_row(self):
        # Mirrors CATHe2's real Test.csv: a numeric index, the domain id, and a
        # description that repeats it. Only the Domain column should be read.
        path = self.make_archive(
            {
                "data/Dataset/csv/Test.csv": (
                    ",Unnamed: 0,Set,Domain,SF,Description,Sequence\n"
                    "1035679,1035679,testsequence,2vsqA04,3.40.50.980,x_2vsqA04,YREL\n"
                    "1035680,1035680,testsequence,4ix3A01,3.30.200.20,x_4ix3A01,LKAR\n"
                )
            }
        )
        domains, report = benchmarks.extract_cathe2_test_domains(path, 2)
        self.assertEqual(domains, {"2vsqA04", "4ix3A01"})
        self.assertEqual(report["chosen_column"], "Domain")

    def test_index_only_table_is_not_mistaken_for_a_domain_list(self):
        # The archive also holds Test_ids_*.csv files of bare row indices.
        path = self.make_archive(
            {
                "data/Dataset/csv/Test_ids_pLDDT_0.csv": (
                    "Domain_id\n1035679\n1035680\n1035681\n"
                ),
            }
        )
        domains, report = benchmarks.extract_cathe2_test_domains(path, 3)
        self.assertEqual(domains, set())
        self.assertEqual(report["candidates"], [])

    def test_real_table_wins_over_an_index_table(self):
        path = self.make_archive(
            {
                "data/Dataset/csv/Test_ids_pLDDT_0.csv": "Domain_id\n1035679\n1035680\n",
                "data/Dataset/csv/Test.csv": (
                    ",Set,Domain,SF\n1035679,testsequence,2vsqA04,3.40.50.980\n"
                    "1035680,testsequence,4ix3A01,3.30.200.20\n"
                ),
            }
        )
        domains, report = benchmarks.extract_cathe2_test_domains(path, 2)
        self.assertEqual(report["chosen"], "data/Dataset/csv/Test.csv")
        self.assertEqual(domains, {"2vsqA04", "4ix3A01"})

    def test_only_small_text_members_are_considered(self):
        path = self.make_archive(
            {
                "data/Test.csv": "d\n" + self.ids(0, 50),
                "data/Test_embeddings.npz": "d\n" + self.ids(0, 50),
            }
        )
        names = [i.filename for i in benchmarks.zip_text_members(path)]
        self.assertEqual(names, ["data/Test.csv"])


if __name__ == "__main__":
    unittest.main()
