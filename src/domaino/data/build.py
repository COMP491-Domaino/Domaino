"""Build a versioned Domaino dataset from the raw sources (work package 1).

Run the whole pipeline::

    python -m domaino.data.build --config configs/dataset_v1.json

Or a subset of stages, which is useful because the Pfam and TED passes read
tens of gigabytes and should go through Slurm::

    python -m domaino.data.build --config configs/dataset_v1.json \
        --stages cath,splits,report

Stages are individually resumable: the expensive cross-source scans write
small cache files and are skipped when those already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import audit as audit_mod
from . import benchmarks as bench_mod
from . import report as report_mod
from . import splits as splits_mod
from .schema import SCHEMA_VERSION, ChainRecord, read_jsonl, write_jsonl
from .sources import cath as cath_mod
from .sources import pdb_seqres
from .sources import pfam as pfam_mod
from .sources import ted as ted_mod

ALL_STAGES = ("cath", "pfam", "ted", "splits", "report")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_config(path: str) -> Dict[str, object]:
    with open(path, "rt", encoding="utf-8") as handle:
        config = json.load(handle)
    root = config.get("repo_root") or os.getcwd()

    def resolve(value: object) -> object:
        if isinstance(value, str) and value.startswith("@/"):
            return os.path.join(str(root), value[2:])
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return {key: resolve(value) for key, value in config.items()}


def _paths(config: Dict[str, object]) -> Dict[str, str]:
    return {key: str(value) for key, value in dict(config["paths"]).items()}  # type: ignore[arg-type]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# -- stage: CATH ------------------------------------------------------------


def stage_cath(config: Dict[str, object], out_dir: str) -> str:
    """Convert CATH into the common schema, writing one record per chain."""
    paths = _paths(config)
    destination = os.path.join(out_dir, "chains.cath.jsonl.gz")
    if os.path.exists(destination) and not config.get("overwrite"):
        log(f"cath: reusing {destination}")
        return destination

    log("cath: collecting the chains CATH references")
    wanted = cath_mod.wanted_chain_keys(paths["cath_chopping"])
    log(f"cath: {len(wanted)} chains referenced; loading SEQRES for those")
    sequences = pdb_seqres.load_chain_sequences(paths["pdb_seqres"], wanted)
    log(f"cath: resolved {len(sequences)} chain sequences")

    log("cath: loading published domain sequences for the agreement check")
    domain_sequences = cath_mod.parse_domain_sequences(paths["cath_domain_seqs"])

    log("cath: converting")
    records = cath_mod.load_records(
        chopping_path=paths["cath_chopping"],
        domain_list_path=paths["cath_domain_list"],
        chain_sequences=sequences,
        domain_sequences=domain_sequences,
        source_version=str(config.get("cath_version", "4.4.0")),
        weight=float(dict(config.get("weights", {})).get("cath", 1.0)),  # type: ignore[arg-type]
    )
    written = write_jsonl(destination, records)
    log(f"cath: wrote {written} records to {destination}")
    return destination


def cath_md5_index(path: str) -> Dict[str, str]:
    """Map sequence MD5 to CATH record id, for joining other sources on."""
    index: Dict[str, str] = {}
    for record in read_jsonl(path):
        if record.sequence_md5:
            index.setdefault(record.sequence_md5, record.record_id)
    return index


# -- stage: Pfam ------------------------------------------------------------


def stage_pfam(
    config: Dict[str, object], out_dir: str, cath_md5: Dict[str, str]
) -> Tuple[Optional[str], Dict[str, object]]:
    """Attach Pfam regions to chains we already hold, and summarise the rest.

    Pfam regions live on UniProt sequences that the project does not download,
    so the only regions usable as supervision are those whose sequence MD5
    matches a PDB chain. The scan over the full regions file also yields the
    statistics for the data card.
    """
    paths = _paths(config)
    cache = os.path.join(out_dir, "pfam_regions_on_cath.tsv.gz")
    destination = os.path.join(out_dir, "chains.pfam.jsonl.gz")
    limit = config.get("pfam_scan_limit")
    limit = int(limit) if limit else None

    stats_path = os.path.join(out_dir, "pfam_scan_stats.json")
    if not os.path.exists(cache) or config.get("overwrite"):
        log(f"pfam: single pass over the regions file (limit={limit})")
        scan = pfam_mod.scan_and_select(
            paths["pfam_regions"], keep_md5=set(cath_md5), cache_path=cache, limit=limit
        )
        report_mod.write_json(stats_path, scan)
        log(
            f"pfam: read {scan['rows_scanned']} regions, kept "
            f"{scan['rows_kept_on_held_sequences']} on sequences we hold -> {cache}"
        )
    else:
        log(f"pfam: reusing {cache}")
        with open(stats_path, "rt", encoding="utf-8") as handle:
            scan = json.load(handle)

    families = pfam_mod.parse_clans(paths["pfam_clans"])
    grouped = pfam_mod.group_by_sequence(pfam_mod.iter_regions(cache))
    if not grouped:
        log("pfam: no regions matched a chain sequence; no records written")
        return None, scan

    sequences_by_md5 = {
        record.sequence_md5: record.sequence
        for record in read_jsonl(os.path.join(out_dir, "chains.cath.jsonl.gz"))
        if record.sequence_md5 and record.sequence_md5 in grouped and record.sequence
    }
    records = pfam_mod.build_records(
        grouped,
        sequences_by_md5,  # type: ignore[arg-type]
        families=families,
        source_version=str(config.get("pfam_version", "38.2")),
        weight=float(dict(config.get("weights", {})).get("pfam", 0.3)),  # type: ignore[arg-type]
    )
    written = write_jsonl(destination, records)
    log(f"pfam: wrote {written} records to {destination}")
    return destination, scan


# -- stage: TED -------------------------------------------------------------


def stage_ted(
    config: Dict[str, object], out_dir: str, cath_md5: Dict[str, str]
) -> Optional[str]:
    """Attach TED choppings to chains we already hold.

    TED ships no sequences either, so the same MD5 join applies. Both TED
    files are very large, so the matched subset is cached.
    """
    paths = _paths(config)
    chain_cache = os.path.join(out_dir, "ted_chains_on_cath.tsv")
    domain_cache = os.path.join(out_dir, "ted_domains_on_cath.tsv")
    destination = os.path.join(out_dir, "chains.ted.jsonl.gz")
    limit = config.get("ted_scan_limit")
    limit = int(limit) if limit else None

    if not os.path.exists(chain_cache) or config.get("overwrite"):
        log(f"ted: scanning per-chain file for sequences we hold (limit={limit})")
        with open(chain_cache, "wt", encoding="utf-8") as handle:
            handle.write("chain_id\tmd5\tlength\n")
            count = 0
            for chain_id, md5, length in ted_mod.iter_chain_index(
                paths["ted_chains"], keep_md5=set(cath_md5), limit=limit
            ):
                handle.write(f"{chain_id}\t{md5}\t{length}\n")
                count += 1
        log(f"ted: matched {count} chains -> {chain_cache}")
    else:
        log(f"ted: reusing {chain_cache}")

    chain_md5: Dict[str, str] = {}
    chain_lengths: Dict[str, int] = {}
    with open(chain_cache, "rt", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            chain_id, md5, length = line.rstrip("\n").split("\t")
            chain_md5[chain_id] = md5
            chain_lengths[chain_id] = int(length)

    if not chain_md5:
        log("ted: no chains matched a sequence we hold; no records written")
        return None

    if not os.path.exists(domain_cache) or config.get("overwrite"):
        log("ted: scanning domain summary for the matched chains")
        with open(domain_cache, "wt", encoding="utf-8") as handle:
            count = 0
            for row in ted_mod.iter_domain_rows(
                paths["ted_domains"], keep_chains=set(chain_md5)
            ):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1
        log(f"ted: matched {count} domains -> {domain_cache}")
    else:
        log(f"ted: reusing {domain_cache}")

    with open(domain_cache, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: str(row["ted_id"]))

    # Only the matched sequences are needed, so the whole CATH set is not held.
    matched_md5 = set(chain_md5.values())
    sequences_by_md5 = {
        record.sequence_md5: record.sequence
        for record in read_jsonl(os.path.join(out_dir, "chains.cath.jsonl.gz"))
        if record.sequence_md5 in matched_md5 and record.sequence
    }
    records = ted_mod.build_records(
        rows,
        chain_lengths=chain_lengths,
        chain_md5=chain_md5,
        sequences_by_md5=sequences_by_md5,  # type: ignore[arg-type]
        source_version=str(config.get("ted_version", "v5")),
    )
    written = write_jsonl(destination, records)
    log(f"ted: wrote {written} records to {destination}")
    return destination


# -- stage: splits ----------------------------------------------------------


def stage_splits(
    config: Dict[str, object], out_dir: str, cath_path: str
) -> splits_mod.SplitPlan:
    """Partition CATH chains, excluding benchmark chains and their homologues."""
    paths = _paths(config)

    chain_clusters: Dict[str, List[str]] = {}
    unusable: Set[str] = set()
    no_cluster: Set[str] = set()
    evaluable: Set[str] = set()
    for record in read_jsonl(cath_path):
        if audit_mod.has_valid_sequence(record):
            evaluable.add(record.chain_id)
        if not audit_mod.is_usable(record):
            unusable.add(record.chain_id)
        keys = record.s35_keys()
        if keys:
            # Clusters are kept even for unusable chains: their homology is
            # real, so they must still be visible to the benchmark exclusion.
            chain_clusters[record.chain_id] = keys
        elif record.chain_id not in unusable:
            # No classified domain means no S35 cluster, so homology to the rest
            # of the dataset cannot be established and the chain is held out of
            # every partition rather than risking leakage.
            no_cluster.add(record.chain_id)

    chainsaw = bench_mod.load_chainsaw_splits(paths["chainsaw_splits"])

    # The CATHe2 test set size is known from its published superfamily labels,
    # so the identifier table pulled out of their archive can be checked
    # against it instead of trusted on filename alone.
    expected_cathe2 = len(
        bench_mod.load_cathe2_test_superfamilies(paths.get("cathe2_test_labels"))
    )
    cathe2_domains = bench_mod.load_cathe2_test_domains(
        paths.get("cathe2_test_ids"), expected_count=expected_cathe2 or None
    )
    log(
        f"splits: chainsaw test={len(chainsaw.get('test', []))} "
        f"validation={len(chainsaw.get('validation', []))}; "
        f"cathe2 test domains={len(cathe2_domains)} "
        f"(expected about {expected_cathe2 or 'unknown'})"
    )
    if not cathe2_domains:
        log(
            "splits: WARNING the CATHe2 exclusion is INACTIVE -- no test-domain "
            "identifiers were available. Classification comparison against "
            "CATHe2 is exposed to homology leakage until this is resolved."
        )

    plan = splits_mod.assign(
        chain_clusters=chain_clusters,
        unusable=unusable,
        no_cluster=no_cluster,
        evaluable=evaluable,
        chainsaw_test=set(chainsaw.get("test", [])),
        chainsaw_validation=set(chainsaw.get("validation", [])),
        cathe2_test=bench_mod.chains_of_domains(cathe2_domains),
        ratios=tuple(config.get("split_ratios", (0.8, 0.1, 0.1))),  # type: ignore[arg-type]
        seed=int(config.get("seed", 20260404)),
        benchmark_exclusion=str(config.get("benchmark_exclusion", "one_hop")),
    )
    log(f"splits: {plan.counts}")
    log(
        "splits: leakage audit "
        f"{'clean' if plan.leakage.get('clean') else 'LEAKAGE DETECTED'}"
    )
    return plan


def write_split_lists(manifest_dir: str, plan: splits_mod.SplitPlan) -> None:
    """Write one plain-text id list per partition, for version control."""
    directory = _ensure_dir(os.path.join(manifest_dir, "splits"))
    members: Dict[str, List[str]] = {}
    for chain, partition in plan.assignment.items():
        members.setdefault(partition, []).append(chain)
    for partition, chains in members.items():
        with open(
            os.path.join(directory, f"{partition}.txt"), "wt", encoding="utf-8"
        ) as handle:
            handle.write("\n".join(sorted(chains)) + "\n")


# -- driver -----------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"comma-separated subset of {','.join(ALL_STAGES)}",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild cached stage outputs"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.overwrite:
        config["overwrite"] = True
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")

    version = str(config.get("dataset_version", "dataset_v1"))
    out_dir = _ensure_dir(os.path.join(str(config["processed_root"]), version))
    manifest_dir = _ensure_dir(os.path.join(str(config["manifest_root"]), version))
    log(f"building {version}")
    log(f"  records  -> {out_dir}")
    log(f"  manifests-> {manifest_dir}")

    cath_path = os.path.join(out_dir, "chains.cath.jsonl.gz")
    if "cath" in stages:
        cath_path = stage_cath(config, out_dir)
    if not os.path.exists(cath_path):
        log("cath records are required by every later stage; run the cath stage")
        return 1

    cath_md5 = cath_md5_index(cath_path)
    log(f"cath: {len(cath_md5)} distinct chain sequences available for joining")

    pfam_path: Optional[str] = None
    pfam_scan: Dict[str, object] = {}
    if "pfam" in stages:
        pfam_path, pfam_scan = stage_pfam(config, out_dir, cath_md5)

    ted_path: Optional[str] = None
    if "ted" in stages:
        ted_path = stage_ted(config, out_dir, cath_md5)

    # Records written by an earlier invocation still belong in the report, so
    # the stages are discovered from disk rather than from this run's arguments.
    def existing(name: str) -> Optional[str]:
        path = os.path.join(out_dir, name)
        return path if os.path.exists(path) else None

    pfam_path = pfam_path or existing("chains.pfam.jsonl.gz")
    ted_path = ted_path or existing("chains.ted.jsonl.gz")
    if not pfam_scan:
        stats_path = existing("pfam_scan_stats.json")
        if stats_path:
            with open(stats_path, "rt", encoding="utf-8") as handle:
                pfam_scan = json.load(handle)

    plan: Optional[splits_mod.SplitPlan] = None
    summary_path = os.path.join(manifest_dir, "split_summary.json")
    if "splits" in stages:
        plan = stage_splits(config, out_dir, cath_path)
        write_split_lists(manifest_dir, plan)
        report_mod.write_json(summary_path, plan.summary())
    elif os.path.exists(summary_path):
        # A report-only run still describes the split written earlier.
        with open(summary_path, "rt", encoding="utf-8") as handle:
            plan = splits_mod.SplitPlan.from_summary(json.load(handle))
        log(f"splits: reusing summary from {summary_path}")

    if "report" in stages:
        log("report: auditing records")
        record_paths = [p for p in (cath_path, pfam_path, ted_path) if p]

        def all_records() -> Iterable[ChainRecord]:
            for path in record_paths:
                yield from read_jsonl(path)

        audits, _ = audit_mod.audit_records(all_records())
        label_space = audit_mod.label_space(read_jsonl(cath_path))

        def other_records() -> Iterable[ChainRecord]:
            for path in (pfam_path, ted_path):
                if path:
                    yield from read_jsonl(path)

        # Load only the CATH records another source also describes, instead of
        # holding all of them with their sequences.
        shared_md5 = {r.sequence_md5 for r in other_records() if r.sequence_md5}
        primary = {
            record.sequence_md5: record
            for record in read_jsonl(cath_path)
            if record.sequence_md5 in shared_md5
        }
        log(f"report: {len(primary)} CATH chains are also described elsewhere")

        conflicts = audit_mod.conflict_report(primary, other_records())
        del primary

        paths = _paths(config)
        coverage = audit_mod.cathe2_label_coverage(
            read_jsonl(cath_path),
            bench_mod.load_cathe2_test_superfamilies(paths.get("cathe2_test_labels")),
        )

        report_mod.write_json(
            os.path.join(manifest_dir, "audit_report.json"),
            {
                "dataset_version": version,
                "sources": {k: v.as_dict() for k, v in audits.items()},
                "label_space": label_space,
                "conflicts": conflicts,
                "cathe2_label_coverage": coverage,
                "pfam_scan": pfam_scan,
                "splits": plan.counts if plan else {},
                "leakage": plan.leakage if plan else {},
            },
        )
        with open(
            os.path.join(manifest_dir, "audit_report.md"), "wt", encoding="utf-8"
        ) as handle:
            handle.write(
                report_mod.render_audit(
                    audits, label_space, conflicts, plan, coverage, pfam_scan
                )
            )

        source_rows = _read_source_versions(
            os.path.join(str(config["manifest_root"]), "source-versions.tsv")
        )
        with open(
            os.path.join(manifest_dir, "data_card.md"), "wt", encoding="utf-8"
        ) as handle:
            handle.write(
                report_mod.render_data_card(
                    version,
                    audits,
                    plan,
                    source_rows,
                    label_space,
                    benchmark_exclusion=str(
                        config.get("benchmark_exclusion", "one_hop")
                    ),
                )
            )

        report_mod.write_json(
            os.path.join(manifest_dir, "dataset.lock.json"),
            {
                "dataset_version": version,
                "schema_version": SCHEMA_VERSION,
                "config": config,
                "sources": source_rows,
                "record_files": [os.path.basename(p) for p in record_paths],
                "counts": {k: v.records for k, v in audits.items()},
                "split_counts": plan.counts if plan else {},
                "split_digests": (plan.digests or plan.compute_digests()) if plan else {},
            },
        )
        log(f"report: wrote audit, data card and lock file to {manifest_dir}")

    log("done")
    return 0


def _read_source_versions(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            values = line.rstrip("\n").split("\t")
            if len(values) == len(header):
                rows.append(dict(zip(header, values)))
    return rows


if __name__ == "__main__":
    sys.exit(main())
