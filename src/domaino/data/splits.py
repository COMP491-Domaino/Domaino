"""Leakage-safe train / validation / test partitions (task T1.5).

Why not a random split
----------------------
Protein databases are full of homologues, so splitting individual chains at
random lets a near-identical relative of a test chain sit in training, which
inflates scores and makes comparison against published benchmarks meaningless
(proposal risk 2).

How chains are grouped
----------------------
CATH already clusters domains at 35% sequence identity (the S35 level), so no
separate MMseqs2 run is needed. A chain is linked to another chain when they
share an S35 cluster, and whole connected components are assigned to one
partition together. Measured on CATH 4.4.0 this gives 23,021 components over
403,962 classified chains, the largest holding 5.6% of chains -- coarse enough
to block leakage, fine enough to hit split ratios.

Grouping by homologous superfamily instead was tested and rejected: because
multidomain chains link superfamilies together transitively, 73.6% of all
chains collapse into a single component, which cannot be split at all.

Benchmark exclusion
-------------------
Benchmark test chains form evaluation-only partitions, and their homologues are
marked excluded rather than quietly dropped, so the audit reports exactly what
each exclusion cost. Two rules are available:

``one_hop`` (default)
    Exclude chains that share an S35 cluster with a benchmark chain, i.e. that
    are homologous to it at 35% identity or better. On CATH 4.4.0 with the
    Chainsaw hold-out this removes 43,787 chains.

``component``
    Exclude every chain in a homology component containing a benchmark chain,
    which removes 73,407. The extra 29,620 are linked only through a chain of
    intermediates -- largely multidomain chains sharing one unrelated domain
    family -- and one 22,429-chain hub component dominates the difference.

Both rules leave the leakage audit clean, because both guarantee that no S35
cluster is shared between a training chain and an evaluation chain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Partition names. The two benchmark partitions are evaluation-only.
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
BENCHMARK_CHAINSAW = "benchmark_chainsaw"
BENCHMARK_CATHE2 = "benchmark_cathe2"
EXCLUDED_HOMOLOGOUS = "excluded_homologous"
EXCLUDED_UNUSABLE = "excluded_unusable"
EXCLUDED_NO_CLUSTER = "excluded_no_cluster"

EVALUATION_PARTITIONS = (TEST, BENCHMARK_CHAINSAW, BENCHMARK_CATHE2)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self._parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[left_root] = right_root


@dataclass
class SplitPlan:
    """The outcome of partitioning, plus everything needed to audit it."""

    assignment: Dict[str, str] = field(default_factory=dict)
    homology_group: Dict[str, str] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    component_count: int = 0
    largest_component: int = 0
    excluded_by_benchmark: Dict[str, int] = field(default_factory=dict)
    leakage: Dict[str, object] = field(default_factory=dict)
    # Filled in by `assign`, and carried in the saved summary so a report built
    # in a later run can still state which split it describes.
    digests: Dict[str, str] = field(default_factory=dict)

    def compute_digests(self) -> Dict[str, str]:
        """Stable SHA-256 per partition, so a split can be verified later."""
        members: Dict[str, List[str]] = {}
        for chain, partition in self.assignment.items():
            members.setdefault(partition, []).append(chain)
        return {
            partition: hashlib.sha256(
                "\n".join(sorted(ids)).encode("utf-8")
            ).hexdigest()
            for partition, ids in members.items()
        }

    def summary(self) -> Dict[str, object]:
        """Everything needed to describe this split without the member lists."""
        return {
            "counts": self.counts,
            "component_count": self.component_count,
            "largest_component": self.largest_component,
            "excluded_by_benchmark": self.excluded_by_benchmark,
            "leakage": self.leakage,
            "digests": self.digests or self.compute_digests(),
        }

    @classmethod
    def from_summary(cls, payload: Dict[str, object]) -> "SplitPlan":
        """Rebuild a plan from a saved summary; `assignment` stays empty."""
        return cls(
            counts=dict(payload.get("counts", {})),  # type: ignore[arg-type]
            component_count=int(payload.get("component_count", 0)),  # type: ignore[arg-type]
            largest_component=int(payload.get("largest_component", 0)),  # type: ignore[arg-type]
            excluded_by_benchmark=dict(payload.get("excluded_by_benchmark", {})),  # type: ignore[arg-type]
            leakage=dict(payload.get("leakage", {})),  # type: ignore[arg-type]
            digests=dict(payload.get("digests", {})),  # type: ignore[arg-type]
        )


def build_components(
    chain_clusters: Dict[str, Sequence[str]]
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Group chains into homology components via shared S35 clusters.

    `chain_clusters` maps a chain to the S35 cluster keys its domains belong
    to. Returns the chain-to-component map and the component membership.
    """
    union = _UnionFind()
    for chain in chain_clusters:
        union.add(chain)

    first_chain_in_cluster: Dict[str, str] = {}
    for chain, clusters in chain_clusters.items():
        for cluster in clusters:
            if cluster in first_chain_in_cluster:
                union.union(chain, first_chain_in_cluster[cluster])
            else:
                first_chain_in_cluster[cluster] = chain

    component_of: Dict[str, str] = {}
    members: Dict[str, List[str]] = {}
    for chain in chain_clusters:
        root = union.find(chain)
        component_of[chain] = root
        members.setdefault(root, []).append(chain)
    return component_of, members


def assign(
    chain_clusters: Dict[str, Sequence[str]],
    unusable: Optional[Set[str]] = None,
    no_cluster: Optional[Set[str]] = None,
    evaluable: Optional[Set[str]] = None,
    chainsaw_test: Optional[Set[str]] = None,
    chainsaw_validation: Optional[Set[str]] = None,
    cathe2_test: Optional[Set[str]] = None,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 20260404,
    benchmark_exclusion: str = "one_hop",
) -> SplitPlan:
    """Assign every chain to exactly one partition.

    A benchmark test chain is kept in its evaluation partition whenever its
    sequence is sound, which `evaluable` reports, even if its CATH labels are
    not trustworthy enough to train on: the published benchmarks supply their
    own ground truth. Every other chain must be fully usable to be assigned to
    train, validation or test, because those partitions are scored against
    CATH.

    When `evaluable` is omitted, every chain outside `unusable` is treated as
    evaluable.
    """
    if benchmark_exclusion not in ("one_hop", "component"):
        raise ValueError(f"unknown benchmark_exclusion: {benchmark_exclusion}")

    plan = SplitPlan()
    unusable = unusable or set()
    no_cluster = no_cluster or set()
    chainsaw_test = chainsaw_test or set()
    chainsaw_validation = chainsaw_validation or set()
    cathe2_test = cathe2_test or set()
    if evaluable is None:
        evaluable = set(chain_clusters) - unusable

    component_of, members = build_components(chain_clusters)
    plan.component_count = len(members)
    plan.largest_component = max((len(m) for m in members.values()), default=0)
    plan.homology_group = {
        chain: f"s35c:{component[:16]}" for chain, component in component_of.items()
    }

    # Benchmark chains themselves, and which partition each belongs to. A
    # benchmark chain whose sequence is unsound cannot be evaluated at all and
    # falls through to the exclusion handling below.
    benchmark: Dict[str, str] = {}
    for chains, partition in (
        (chainsaw_test, BENCHMARK_CHAINSAW),
        (chainsaw_validation, BENCHMARK_CHAINSAW),
        (cathe2_test, BENCHMARK_CATHE2),
    ):
        for chain in chains:
            if chain in chain_clusters and chain in evaluable:
                benchmark.setdefault(chain, partition)

    # Homologues of benchmark chains, which may be evaluated but never trained.
    # Chains already excluded for a stronger reason are left out, so the
    # reported cost of benchmark exclusion counts only chains it alone removes.
    already_excluded = unusable | no_cluster
    excluded: Dict[str, str] = {}
    if benchmark_exclusion == "one_hop":
        benchmark_clusters: Dict[str, str] = {}
        for chain, partition in benchmark.items():
            for cluster in chain_clusters[chain]:
                benchmark_clusters.setdefault(cluster, partition)
        for chain, clusters in chain_clusters.items():
            if chain in benchmark or chain in already_excluded:
                continue
            for cluster in clusters:
                if cluster in benchmark_clusters:
                    excluded[chain] = benchmark_clusters[cluster]
                    break
    else:
        reserved = {component_of[chain]: p for chain, p in benchmark.items()}
        for component, partition in reserved.items():
            for chain in members[component]:
                if chain not in benchmark and chain not in already_excluded:
                    excluded[chain] = partition

    for partition in excluded.values():
        plan.excluded_by_benchmark[partition] = (
            plan.excluded_by_benchmark.get(partition, 0) + 1
        )

    # Assign whole components over the chains that are still free, so that no
    # cluster can span two of the train/validation/test partitions. Chains that
    # cannot be trained or scored against CATH are fixed out of this step.
    fixed = set(benchmark) | set(excluded) | unusable | no_cluster
    targets = {TRAIN: ratios[0], VALIDATION: ratios[1], TEST: ratios[2]}
    free_members = {
        component: [chain for chain in chains if chain not in fixed]
        for component, chains in members.items()
    }
    assignable = sorted(
        (c for c, chains in free_members.items() if chains),
        key=lambda component: (-len(free_members[component]), component),
    )
    total = sum(len(free_members[c]) for c in assignable)
    quota = {name: share * total for name, share in targets.items()}
    filled = {name: 0 for name in targets}

    for component in assignable:
        # Take the partition furthest below its quota, breaking ties with a
        # seeded hash so the result never depends on dictionary ordering.
        chosen = min(
            targets,
            key=lambda name: (
                filled[name] - quota[name],
                hashlib.sha256(f"{seed}:{name}:{component}".encode()).hexdigest(),
            ),
        )
        filled[chosen] += len(free_members[component])
        for chain in free_members[component]:
            plan.assignment[chain] = chosen

    for chain in excluded:
        plan.assignment[chain] = EXCLUDED_HOMOLOGOUS

    # Records that cannot train regardless of homology.
    for chain in no_cluster:
        plan.assignment[chain] = EXCLUDED_NO_CLUSTER
    for chain in unusable:
        plan.assignment[chain] = EXCLUDED_UNUSABLE

    # Benchmark partitions win last: an evaluable benchmark chain belongs in
    # its benchmark even when its CATH labels are untrustworthy.
    for chain, partition in benchmark.items():
        plan.assignment[chain] = partition

    counts: Dict[str, int] = {}
    for partition in plan.assignment.values():
        counts[partition] = counts.get(partition, 0) + 1
    plan.counts = dict(sorted(counts.items()))
    plan.leakage = verify(plan, chain_clusters)
    plan.digests = plan.compute_digests()
    return plan


def verify(
    plan: SplitPlan, chain_clusters: Dict[str, Sequence[str]]
) -> Dict[str, object]:
    """Check that no S35 cluster spans a training and an evaluation partition.

    This is the homology-leakage audit required by deliverable D1.2. A clean
    result means every reported score is measured on genuinely unfamiliar
    sequences.
    """
    cluster_partitions: Dict[str, Set[str]] = {}
    for chain, clusters in chain_clusters.items():
        partition = plan.assignment.get(chain)
        if partition is None:
            continue
        for cluster in clusters:
            cluster_partitions.setdefault(cluster, set()).add(partition)

    shared = {
        cluster: sorted(partitions)
        for cluster, partitions in cluster_partitions.items()
        if len(partitions) > 1
    }
    train_vs_eval = {
        cluster: partitions
        for cluster, partitions in shared.items()
        if TRAIN in partitions
        and any(p in partitions for p in EVALUATION_PARTITIONS)
    }
    return {
        "clusters_total": len(cluster_partitions),
        "clusters_spanning_partitions": len(shared),
        "clusters_leaking_train_into_eval": len(train_vs_eval),
        "examples": dict(list(train_vs_eval.items())[:10]),
        "clean": not train_vs_eval,
    }
