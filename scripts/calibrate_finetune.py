"""Measure what an unfrozen-ESM-2 epoch actually costs, before committing a run.

Unfreezing moves the cost by two orders of magnitude, and the honest way to
report "how long will this take" is to time real steps on the real chains at
the real settings rather than to extrapolate from a frozen epoch.

    PYTHONPATH=src python scripts/calibrate_finetune.py \
        --config configs/train_v1.json --steps 12 --max-tokens 4096

Reports residues/second and peak GPU memory for each setting, and projects the
epoch and 20-epoch wall-clock from the split sizes the config points at.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")

from domaino.model.embed import load_config, split_members
from domaino.train.data import iter_epoch, load_examples
from domaino.train.trainer import CompleteModel, TrainConfig, _segmentation_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_v1.json")
    parser.add_argument("--chains", type=int, default=400)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--max-tokens", type=int, action="append", default=[],
        help="padded residues per training batch; repeat to sweep",
    )
    parser.add_argument(
        "--layers", type=int, action="append", default=[],
        help="trainable top layers; 0 means all 33. Repeat to sweep",
    )
    args = parser.parse_args()
    token_budgets = args.max_tokens or [2048, 4096]
    layer_counts = args.layers or [8, 0]

    import torch

    file_config = load_config(args.config)
    encoder_block = dict(file_config["encoder"])

    examples = load_examples(
        str(file_config["records"]),
        split_members(str(file_config["manifest_dir"]), "train"),
        None, args.chains, with_sequences=True,
    )
    residues = sum(e.length for e in examples)
    print(f"{len(examples)} chains, {residues} residues, "
          f"longest {max(e.length for e in examples)}")

    members = split_members(str(file_config["manifest_dir"]), "train")
    validation = split_members(str(file_config["manifest_dir"]), "validation")
    print(f"split sizes: train {len(members)}, validation {len(validation)}")

    # Mean residues per chain from the sample, used to project the full split.
    mean_length = residues / max(len(examples), 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for layers in layer_counts:
        for max_tokens in token_budgets:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            config = TrainConfig(
                architecture="pairwise", epochs=1, max_tokens=max_tokens,
                accumulate_steps=1, mixed_precision=True, dropout=0.1,
                train_classifier=False, finetune_encoder=True,
                finetune_layers=layers, gradient_checkpointing=True,
                encoder_max_tokens=max_tokens,
                encoder_name=str(encoder_block["model_name"]),
                torch_home=str(encoder_block.get("torch_home") or "") or None,
            )
            try:
                model = CompleteModel(config, 1280, None, device)
                trainable = model.encoder.trainable_count()
                optimiser = torch.optim.AdamW(
                    [
                        {"params": list(model.head_parameters()), "lr": 1e-3},
                        {"params": list(model.encoder_parameters()), "lr": 1e-5},
                    ]
                )
                scaler = torch.amp.GradScaler("cuda", enabled=True)
                model.train()

                done = seen = 0
                started = None
                for batch in iter_epoch(examples, max_tokens, seed=0):
                    with torch.amp.autocast("cuda", enabled=True):
                        features, mask, _ = model.encode(batch)
                        loss, _, _ = _segmentation_step(
                            model, features, mask, batch, config
                        )
                    optimiser.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                    done += 1
                    # The first two steps pay for cuDNN autotuning and
                    # allocator warm-up, so timing starts after them.
                    if done == 2:
                        torch.cuda.synchronize()
                        started, seen = time.time(), 0
                    elif started is not None:
                        seen += sum(e.length for e in batch)
                    if done >= args.steps:
                        break
                torch.cuda.synchronize()
                elapsed = time.time() - started
                rate = seen / elapsed if elapsed > 0 else 0.0
                peak = torch.cuda.max_memory_allocated() / 2**30
                train_hours = (len(members) * mean_length / rate) / 3600 if rate else 0
                # Validation runs forward-only, roughly three times faster.
                val_hours = (
                    (len(validation) * mean_length / (rate * 3)) / 3600 if rate else 0
                )
                epoch_hours = train_hours + val_hours
                results.append(
                    {
                        "trainable_layers": layers,
                        "trainable_params": trainable,
                        "max_tokens": max_tokens,
                        "residues_per_second": round(rate, 1),
                        "peak_gpu_gb": round(peak, 2),
                        "projected_epoch_hours": round(epoch_hours, 2),
                        "projected_20_epoch_hours": round(epoch_hours * 20, 1),
                    }
                )
                print(
                    f"  layers={layers or 'all'} tokens={max_tokens}: "
                    f"{rate:,.0f} res/s, peak {peak:.1f} GB, "
                    f"epoch ~{epoch_hours:.2f} h, 20 epochs ~{epoch_hours * 20:.0f} h"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  layers={layers or 'all'} tokens={max_tokens}: OOM")
                results.append(
                    {"trainable_layers": layers, "max_tokens": max_tokens,
                     "error": "out of memory"}
                )
            finally:
                for name in ("model", "optimiser"):
                    if name in dir():
                        pass
                torch.cuda.empty_cache()

    os.makedirs("reports/experiments", exist_ok=True)
    out = "reports/experiments/finetune_calibration.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": args.config,
                "sample_chains": len(examples),
                "mean_chain_length": round(mean_length, 1),
                "train_chains": len(members),
                "validation_chains": len(validation),
                "frozen_epoch_minutes": 18.5,
                "results": results,
            },
            handle,
            indent=2,
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
