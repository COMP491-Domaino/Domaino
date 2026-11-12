"""Turn segmentation-network outputs into domain assignments (task T2.4).

The network emits two things: a per-residue distribution over
{linker, domain, boundary} and a matrix of same-domain probabilities for
residue pairs. Neither is a domain assignment, and the gap between them is
where this module lives.

How it works
------------
1. Residues the network calls linker are unassigned.
2. The chain's assigned stretches are cut at predicted boundary peaks. This
   step is essential rather than cosmetic: adjacent CATH domains frequently
   touch with no linker between them -- ``1chmA01`` is 2-156 and ``1chmA02`` is
   157-402 -- so a decoder that split only on linker runs could never separate
   them.
3. The resulting blocks are grouped by average-linkage agglomerative
   clustering on the pairwise probabilities. Grouping blocks rather than
   cutting a sequence is what lets a domain be **discontinuous**: a cluster may
   hold blocks that are far apart, which is precisely the case the proposal
   says boundary prediction alone cannot represent.
4. The four constraints the proposal names are applied: minimum domain length,
   acceptable linker gaps, residue coverage, and consistency among intervals.

Clustering is average linkage with a similarity cut rather than a method
needing the number of domains up front, because the number of domains is one of
the things being predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Residue class order, shared with the network.
LINKER, DOMAIN, BOUNDARY = 0, 1, 2
RESIDUE_CLASSES = ("linker", "domain", "boundary")

Interval = Tuple[int, int]
Domain = List[Interval]


@dataclass
class DecoderConfig:
    """Decision thresholds and the proposal's four structural constraints.

    Every value here is a post-processing threshold, which the proposal tunes
    on validation data in T3.5. The defaults are deliberately plain starting
    points, not tuned values.
    """

    # A residue is unassigned when the linker class reaches this probability.
    linker_threshold: float = 0.5
    # A boundary peak at least this strong becomes a candidate cut.
    boundary_threshold: float = 0.5
    # Average same-domain probability required to keep two blocks together.
    same_domain_threshold: float = 0.5
    # Constraint: a domain shorter than this is not a domain.
    min_domain_length: int = 40
    # Constraint: a gap this short inside one domain is absorbed rather than
    # splitting it into two intervals.
    max_linker_gap: int = 20
    # Constraint: if less than this fraction of the chain ends up assigned,
    # leftover residues are given to their nearest domain.
    min_coverage: float = 0.5
    # Blocks below this length are noise and are returned to linker.
    min_block_length: int = 5


def _runs(flags: Sequence[bool]) -> List[Interval]:
    """Maximal 1-based inclusive runs where `flags` is True."""
    runs: List[Interval] = []
    start: Optional[int] = None
    for index, flag in enumerate(flags, start=1):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def boundary_cuts(
    boundary_probs: Sequence[float], threshold: float
) -> List[int]:
    """Positions after which the chain may be cut.

    A cut is a local maximum of the boundary probability above `threshold`.
    Local maxima are used rather than every above-threshold residue because the
    network marks a band around each edge, and one edge should give one cut.
    """
    cuts: List[int] = []
    length = len(boundary_probs)
    for index in range(length):
        value = boundary_probs[index]
        if value < threshold:
            continue
        left = boundary_probs[index - 1] if index > 0 else -1.0
        right = boundary_probs[index + 1] if index + 1 < length else -1.0
        if value >= left and value >= right:
            cuts.append(index + 1)
    return cuts


def split_into_blocks(
    assigned: Sequence[bool],
    cuts: Sequence[int],
    min_block_length: int,
) -> List[Interval]:
    """Assigned stretches, further split at `cuts`, as 1-based intervals.

    A cut is taken only when both sides of it would be a viable block. The
    network marks a boundary band at every domain edge, and a chain's own first
    and last residues are edges, so a cut is always proposed at the termini. If
    those were honoured a one-residue block would be created and then dropped
    as too short, silently losing residues off both ends of the chain.
    """
    cut_set = set(cuts)
    blocks: List[Interval] = []
    for start, end in _runs(assigned):
        piece_start = start
        for position in range(start, end + 1):
            if position not in cut_set or position >= end:
                continue
            before = position - piece_start + 1
            after = end - position
            if before >= min_block_length and after >= min_block_length:
                blocks.append((piece_start, position))
                piece_start = position + 1
        blocks.append((piece_start, end))
    return [b for b in blocks if b[1] - b[0] + 1 >= min_block_length]


def block_affinities(pair_probs, blocks: Sequence[Interval]):
    """Mean same-domain probability for every pair of blocks.

    Returns the summed probability and the pair count separately, so that
    merging two clusters later is exact average linkage rather than an average
    of averages.
    """
    import numpy as np

    count = len(blocks)
    sums = np.zeros((count, count), dtype=np.float64)
    pairs = np.zeros((count, count), dtype=np.float64)
    for i, (a_start, a_end) in enumerate(blocks):
        a = slice(a_start - 1, a_end)
        for j in range(i + 1, count):
            b_start, b_end = blocks[j]
            b = slice(b_start - 1, b_end)
            block = np.asarray(pair_probs[a, b], dtype=np.float64)
            sums[i, j] = sums[j, i] = block.sum()
            pairs[i, j] = pairs[j, i] = block.size
    return sums, pairs


def cluster_blocks(
    pair_probs, blocks: Sequence[Interval], threshold: float
) -> List[List[int]]:
    """Group blocks into domains by average-linkage agglomerative clustering.

    Merges the most similar pair of clusters while their average same-domain
    probability stays at or above `threshold`. The number of clusters follows
    from the data, which it must: how many domains a chain has is part of what
    is being predicted.
    """
    import numpy as np

    if not blocks:
        return []
    if len(blocks) == 1:
        return [[0]]

    sums, pairs = block_affinities(pair_probs, blocks)
    clusters: Dict[int, List[int]] = {i: [i] for i in range(len(blocks))}

    while len(clusters) > 1:
        best = (-1.0, None, None)
        keys = sorted(clusters)
        for position, left in enumerate(keys):
            for right in keys[position + 1 :]:
                total = pairs[left, right]
                if total <= 0:
                    continue
                score = sums[left, right] / total
                if score > best[0]:
                    best = (score, left, right)
        score, left, right = best
        if left is None or score < threshold:
            break
        # Average linkage: accumulate sums and counts, never averages.
        for other in clusters:
            if other in (left, right):
                continue
            sums[left, other] = sums[other, left] = (
                sums[left, other] + sums[right, other]
            )
            pairs[left, other] = pairs[other, left] = (
                pairs[left, other] + pairs[right, other]
            )
        clusters[left] = clusters[left] + clusters[right]
        del clusters[right]

    return [sorted(members) for _, members in sorted(clusters.items())]


def merge_close_intervals(
    intervals: Sequence[Interval], max_gap: int
) -> List[Interval]:
    """Join intervals of one domain separated by at most `max_gap` residues.

    This is the acceptable-linker-gap constraint: a handful of residues the
    network was unsure about should not turn one domain into two intervals and
    make it look discontinuous when it is not.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def domain_length(domain: Domain) -> int:
    return sum(end - start + 1 for start, end in domain)


@dataclass
class DecodedChain:
    """A decoded assignment plus what the decoder had to do to get there."""

    domains: List[Domain] = field(default_factory=list)
    coverage: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def n_domains(self) -> int:
        return len(self.domains)

    @property
    def discontinuous(self) -> int:
        return sum(1 for d in self.domains if len(d) > 1)


class DomainDecoder:
    """Convert network probabilities into domain intervals."""

    def __init__(self, config: Optional[DecoderConfig] = None) -> None:
        self.config = config or DecoderConfig()

    def decode(self, residue_probs, pair_probs) -> DecodedChain:
        """Decode one chain.

        `residue_probs` is ``(length, 3)`` over {linker, domain, boundary} and
        `pair_probs` is ``(length, length)`` of same-domain probabilities.
        """
        import numpy as np

        residue = np.asarray(residue_probs, dtype=np.float64)
        if residue.ndim != 2 or residue.shape[1] != 3:
            raise ValueError(f"expected (length, 3) residue probs, got {residue.shape}")
        length = residue.shape[0]
        pair = np.asarray(pair_probs, dtype=np.float64)
        if pair.shape != (length, length):
            raise ValueError(
                f"pair probs {pair.shape} do not match {length} residues"
            )

        config = self.config
        result = DecodedChain()

        assigned = residue[:, LINKER] < config.linker_threshold
        cuts = boundary_cuts(residue[:, BOUNDARY].tolist(), config.boundary_threshold)
        blocks = split_into_blocks(
            assigned.tolist(), cuts, config.min_block_length
        )

        if not blocks:
            # Every CATH chain has at least one domain, so refusing to answer
            # is never right; fall back to the whole chain and say so.
            result.domains = [[(1, length)]]
            result.coverage = 1.0
            result.notes.append("no_blocks_fell_back_to_whole_chain")
            return result

        groups = cluster_blocks(pair, blocks, config.same_domain_threshold)
        domains = [
            merge_close_intervals([blocks[i] for i in group], config.max_linker_gap)
            for group in groups
        ]

        domains, absorbed = self._enforce_min_length(domains, pair)
        if absorbed:
            result.notes.append(f"absorbed_{absorbed}_short_domains")

        if not domains:
            result.domains = [[(1, length)]]
            result.coverage = 1.0
            result.notes.append("all_domains_too_short_fell_back_to_whole_chain")
            return result

        covered = sum(domain_length(d) for d in domains)
        if covered / length < config.min_coverage:
            domains = self._extend_to_coverage(domains, length)
            result.notes.append("extended_domains_to_meet_coverage_floor")

        domains = self._enforce_consistency(domains, length)
        result.domains = sorted(domains, key=lambda d: d[0][0])
        result.coverage = round(
            sum(domain_length(d) for d in result.domains) / length, 4
        )
        return result

    # -- constraints ---------------------------------------------------------

    def _enforce_min_length(
        self, domains: List[Domain], pair_probs
    ) -> Tuple[List[Domain], int]:
        """Absorb domains below the minimum length into their best neighbour.

        A fragment is merged into whichever surviving domain it most resembles
        rather than deleted, because its residues do belong somewhere; only
        when nothing survives is it dropped.
        """
        import numpy as np

        keep = [d for d in domains if domain_length(d) >= self.config.min_domain_length]
        short = [d for d in domains if domain_length(d) < self.config.min_domain_length]
        if not short:
            return domains, 0
        if not keep:
            return [], len(short)

        for fragment in short:
            scores = []
            fragment_rows = np.concatenate(
                [np.arange(s - 1, e) for s, e in fragment]
            )
            for target in keep:
                target_rows = np.concatenate(
                    [np.arange(s - 1, e) for s, e in target]
                )
                block = np.asarray(pair_probs)[np.ix_(fragment_rows, target_rows)]
                scores.append(float(block.mean()) if block.size else 0.0)
            best = int(np.argmax(scores))
            keep[best] = merge_close_intervals(
                list(keep[best]) + list(fragment), self.config.max_linker_gap
            )
        return keep, len(short)

    def _extend_to_coverage(self, domains: List[Domain], length: int) -> List[Domain]:
        """Give unassigned residues to their nearest domain.

        The residue-coverage constraint: when the network leaves most of a
        chain unassigned the result is not a usable parse, so the leftovers go
        to whichever domain is closest in sequence.
        """
        owner: List[Optional[int]] = [None] * (length + 1)
        for index, domain in enumerate(domains):
            for start, end in domain:
                for position in range(start, end + 1):
                    owner[position] = index

        # Nearest assigned residue on either side wins.
        nearest_left: List[Optional[int]] = [None] * (length + 1)
        last: Optional[int] = None
        for position in range(1, length + 1):
            if owner[position] is not None:
                last = owner[position]
            nearest_left[position] = last
        nearest_right: List[Optional[int]] = [None] * (length + 2)
        following: Optional[int] = None
        for position in range(length, 0, -1):
            if owner[position] is not None:
                following = owner[position]
            nearest_right[position] = following

        extended: List[List[Interval]] = [list(d) for d in domains]
        for position in range(1, length + 1):
            if owner[position] is not None:
                continue
            choice = nearest_left[position]
            if choice is None:
                choice = nearest_right[position]
            if choice is None:
                continue
            extended[choice].append((position, position))
        return [
            merge_close_intervals(d, self.config.max_linker_gap) for d in extended
        ]

    def _enforce_consistency(
        self, domains: List[Domain], length: int
    ) -> List[Domain]:
        """Make the output a valid parse: in range, sorted, non-overlapping.

        A residue claimed by two domains is not expressible as a CATH chopping,
        so the earlier domain keeps it. Without this the metrics would score an
        assignment the format cannot represent.
        """
        claimed = set()
        cleaned: List[Domain] = []
        for domain in sorted(domains, key=lambda d: min(s for s, _ in d)):
            intervals: List[Interval] = []
            for start, end in sorted(domain):
                start = max(1, start)
                end = min(length, end)
                if end < start:
                    continue
                positions = [p for p in range(start, end + 1) if p not in claimed]
                if not positions:
                    continue
                claimed.update(positions)
                run_start = positions[0]
                previous = positions[0]
                for position in positions[1:]:
                    if position != previous + 1:
                        intervals.append((run_start, previous))
                        run_start = position
                    previous = position
                intervals.append((run_start, previous))
            if intervals:
                cleaned.append(merge_close_intervals(intervals, 0))
        return cleaned
