#!/usr/bin/env python
"""Run the sequence-to-domains path on real chains, for either architecture.

    PYTHONPATH=src python scripts/smoke_segmentation.py --architecture pairwise
    PYTHONPATH=src python scripts/smoke_segmentation.py --architecture query

Takes a handful of real CATH chains, embeds them with frozen ESM-2, fits one of
the two candidate segmentation networks to them, decodes its outputs into
domain assignments and scores those against the true choppings.

This is a wiring check against real data, not a training run: with a couple of
dozen chains the only meaningful question is whether the model can fit what it
was shown and whether the decoder returns a sane parse. Real training,
thresholds and the architecture comparison belong to work package 3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from domaino import metrics
from domaino.data.schema import read_jsonl
from domaino.model import decode, query_segmentation as qs, segmentation as seg
from domaino.model.embed import load_config, split_members
from domaino.model.representation import EncoderConfig, Esm2Encoder

ARCHITECTURES = ("pairwise", "query")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def pick_chains(records_path, members, count, max_length=700):
    """Take a mix of single-domain, multidomain and discontinuous chains.

    A sample of whatever comes first would be almost entirely single-domain,
    which would hide exactly the behaviour both designs exist for.
    """
    keep = set(members)
    buckets = {"single": [], "multi": [], "discontinuous": []}
    for record in read_jsonl(records_path):
        if record.chain_id not in keep or not record.sequence:
            continue
        if record.length > max_length:
            continue
        if record.has_discontinuous_domain:
            bucket = "discontinuous"
        elif record.is_multidomain:
            bucket = "multi"
        else:
            bucket = "single"
        if len(buckets[bucket]) < count:
            buckets[bucket].append(record)
        if all(len(v) >= count for v in buckets.values()):
            break

    share = max(1, count // 3)
    chosen = (
        buckets["single"][:share]
        + buckets["multi"][:share]
        + buckets["discontinuous"][: count - 2 * share]
    )
    log(
        f"selected {len(chosen)} chains: "
        f"{len(buckets['single'][:share])} single-domain, "
        f"{len(buckets['multi'][:share])} multidomain, "
        f"{len(buckets['discontinuous'][: count - 2 * share])} discontinuous"
    )
    return chosen


def build_pairwise(config, encoder_config, device, records, embeddings, dropout=None):
    """Model, per-chain training items and a decode function, pairwise design."""
    import torch

    overrides = dict(config.get("network", {}))
    if dropout is not None:
        overrides["dropout"] = dropout
    net_config = seg.SegmentationConfig(
        input_dim=encoder_config.embed_dim, **overrides
    )
    model = seg.build_network(net_config).to(device)

    items = []
    for record, embedding in zip(records, embeddings):
        domains = [[s.as_tuple() for s in d.segments] for d in record.domains]
        target, mask = seg.pair_targets(record.length, domains)
        rows, cols = seg.sample_pairs(
            mask, limit=net_config.max_pairs_per_chain, seed=0
        )
        items.append(
            {
                "x": embedding.unsqueeze(0).to(device),
                "residue": torch.tensor(
                    seg.residue_targets(record.length, domains)
                ).unsqueeze(0).to(device),
                "rows": torch.from_numpy(rows).to(device),
                "cols": torch.from_numpy(cols).to(device),
                "pair": torch.from_numpy(target[rows, cols]).to(device),
                "domains": domains,
                "record": record,
            }
        )

    def step(item):
        out = model(item["x"], pairs=(item["rows"], item["cols"]))
        return seg.segmentation_loss(
            out["residue_logits"],
            item["residue"],
            out["pair_logits"].reshape(-1),
            item["pair"],
        )

    decoder = decode.DomainDecoder(
        decode.DecoderConfig(**dict(config.get("decoder", {})))
    )

    def predict(item):
        out = model(item["x"])
        residue = torch.softmax(out["residue_logits"][0].float(), -1).cpu().numpy()
        pair = torch.sigmoid(out["pair_logits"][0].float()).cpu().numpy()
        return [list(d) for d in decoder.decode(residue, pair).domains]

    return model, items, step, predict, net_config


def build_query(config, encoder_config, device, records, embeddings, dropout=None):
    """Model, per-chain training items and a decode function, query design."""
    import torch

    overrides = dict(config.get("query_network", {}))
    if dropout is not None:
        overrides["dropout"] = dropout
    net_config = qs.QuerySegmentationConfig(
        input_dim=encoder_config.embed_dim, **overrides
    )
    model = qs.build_query_network(net_config).to(device)

    items = []
    for record, embedding in zip(records, embeddings):
        domains = [[s.as_tuple() for s in d.segments] for d in record.domains]
        items.append(
            {
                "x": embedding.unsqueeze(0).to(device),
                "targets": [qs.mask_targets(record.length, domains)],
                "length": record.length,
                "domains": domains,
                "record": record,
            }
        )

    def step(item):
        out = model(item["x"])
        return qs.query_segmentation_loss(
            out["class_logits"],
            out["mask_logits"],
            item["targets"],
            config=net_config,
            lengths=[item["length"]],
        )

    decoder_config = qs.QueryDecoderConfig(**dict(config.get("query_decoder", {})))

    def predict(item):
        out = model(item["x"])
        return [
            list(d)
            for d in qs.decode_queries(
                out["class_logits"][0].float().cpu(),
                out["mask_logits"][0].float().cpu(),
                item["length"],
                decoder_config,
            )
        ]

    return model, items, step, predict, net_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_v1.json")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="pairwise")
    parser.add_argument("--chains", type=int, default=24)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="overrides the config. Defaults to 0 because this script measures "
             "whether a model CAN fit a handful of chains; dropout is a "
             "regulariser against exactly that, and the query architecture's "
             "bipartite matching is unstable under it on a set this small.",
    )
    args = parser.parse_args(argv)

    import torch

    config = load_config(args.config)
    encoder_config = EncoderConfig(**dict(config["encoder"]))
    records = pick_chains(
        str(config["records"]),
        split_members(str(config["manifest_dir"]), "train"),
        args.chains,
    )
    if not records:
        log("no chains selected")
        return 1

    encoder = Esm2Encoder(encoder_config)
    log(f"embedding on {encoder.device} with {encoder_config.model_name}")
    embeddings = encoder.embed([r.sequence for r in records])
    device = encoder.device

    builder = build_pairwise if args.architecture == "pairwise" else build_query
    model, items, step, predict, net_config = builder(
        config, encoder_config, device, records, embeddings, dropout=args.dropout
    )
    log(
        f"architecture: {args.architecture} | "
        f"{sum(p.numel() for p in model.parameters()):,} parameters | "
        f"receptive field {model.receptive_field} residues | dropout {args.dropout}"
        + (f" | {model.num_queries} queries" if args.architecture == "query" else "")
    )

    optimiser = torch.optim.Adam(
        model.parameters(), lr=float(config.get("learning_rate", 1e-3))
    )
    log(f"fitting for {args.steps} steps")
    model.train()
    for index in range(args.steps):
        optimiser.zero_grad()
        total = 0.0
        for item in items:
            result = step(item)
            # Mean, not sum: summing made the step size scale with the number
            # of chains in the set.
            (result["loss"] / len(items)).backward()
            total += float(result["loss"])
        optimiser.step()
        if index % max(1, args.steps // 6) == 0 or index == args.steps - 1:
            log(f"  step {index:4d}  loss {total / len(items):.4f}")

    model.eval()
    predictions, targets, keys = [], [], []
    with torch.no_grad():
        for item in items:
            predictions.append(predict(item))
            targets.append([list(d) for d in item["domains"]])
            record = item["record"]
            keys.append(
                "discontinuous"
                if record.has_discontinuous_domain
                else ("multidomain" if record.is_multidomain else "single_domain")
            )

    log("")
    log(f"decoded against true choppings ({args.architecture}):")
    overall = metrics.score_segmentation(predictions, targets)
    log(
        f"  all           n={overall.chains:3d}  IoU={overall.mean_iou:.3f}  "
        f"correct@0.8={overall.pct_correct_at_080:.3f}  "
        f"countMAE={overall.domain_count_mae:.3f}"
    )
    for group, (preds, truths) in metrics.stratify(
        keys, predictions, targets
    ).items():
        scores = metrics.score_segmentation(preds, truths)
        log(
            f"  {group:<14}n={scores.chains:3d}  IoU={scores.mean_iou:.3f}  "
            f"correct@0.8={scores.pct_correct_at_080:.3f}  "
            f"countMAE={scores.domain_count_mae:.3f}"
        )

    log("")
    log("examples (true -> predicted):")
    for item, prediction in list(zip(items, predictions))[:6]:
        log(f"  {item['record'].chain_id}: {item['domains']} -> {prediction}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
