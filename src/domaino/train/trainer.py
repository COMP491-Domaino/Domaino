"""Train a complete Domaino model on CATH (task T3.2).

A complete model is one of the two candidate segmentation architectures plus
the hierarchical CATH classifier, so this trains the whole
sequence-to-domains-to-CATH path rather than segmentation alone.

Both architectures train through this one file on purpose: T3.2 requires them
trained "under the same conditions", and the surest way to mean that is for the
optimiser, schedule, batching, precision, early stopping and selection rule to
be literally the same code.

The classifier is trained on **ground-truth** segments. Early in training the
predicted domains are noise, and classifying noise teaches the classifier
nothing about CATH; evaluation then reports both modes, on true segments and on
predicted ones, which is how the proposal separates a segmentation error from a
classification error.

T2.7's memory controls live here and in `data.py`: length-based batching, pair
sampling, mixed precision and gradient accumulation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import metrics
from ..model import classifier as clf
from ..model import decode as dec
from ..model import losses as loss_terms
from ..model import query_segmentation as qs
from ..model import segmentation as seg
from . import protocol
from .data import ChainExample, EmbeddingBatcher, iter_epoch

ARCHITECTURES = ("pairwise", "query")


@dataclass
class TrainConfig:
    """Everything that defines a run. Hashed into the run id."""

    architecture: str = "pairwise"
    seed: int = protocol.SEEDS[0]
    epochs: int = protocol.MAX_EPOCHS
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Padded residues per batch. The budget, not a batch size, because chain
    # lengths span 1 to 4,128.
    max_tokens: int = 24000
    # Optimiser steps are taken every this many batches, which buys a larger
    # effective batch without the memory of one (T2.7).
    accumulate_steps: int = 4
    # Supervised residue pairs sampled per chain (T2.7). Set explicitly rather
    # than derived from the batch, or a large batch silently starves the pair
    # head of signal.
    pairs_per_chain: int = 4000
    mixed_precision: bool = True
    grad_clip: float = 1.0
    dropout: float = 0.1
    hidden_dim: int = 256
    # Weight of the classification loss against the segmentation loss.
    classifier_weight: float = 1.0
    train_classifier: bool = True
    # T2.6: use the source-aware composite loss (separate residue-assignment,
    # boundary, pairwise-grouping and CATH terms, each masked by what the
    # record's source can actually supervise) instead of the single
    # combined loss of T2.3. Off by default so the T3.2 runs already in the
    # registry stay reproducible from their stored config; T3.3's Pfam and TED
    # arms require it, since those sources cannot be used safely without it.
    source_aware: bool = False
    # ESM-2 fine-tuning. Off by default: every result so far reads the frozen
    # T2.2 cache, and the proposal plans frozen first. Turning it on puts the
    # 651M-parameter encoder in the graph, which changes the cost of an epoch
    # by two orders of magnitude -- see scripts/calibrate_finetune.py.
    finetune_encoder: bool = False
    # Transformer layers to train, from the top. 0 means all 33.
    finetune_layers: int = 0
    # The encoder gets a smaller step than the freshly initialised heads: at a
    # shared 1e-3 the pretrained weights are destroyed in the first epoch.
    encoder_learning_rate: float = 1e-5
    gradient_checkpointing: bool = True
    encoder_max_tokens: int = 8192
    # Which encoder, and where its weights are cached. Only consulted when
    # fine-tuning; the frozen path takes both from the representation config
    # that built the cache.
    encoder_name: str = "esm2_t33_650M_UR50D"
    torch_home: Optional[str] = None
    # Relative weight of the two residue terms when source_aware is on.
    assignment_weight: float = 1.0
    boundary_weight: float = 0.5
    pairwise_weight: float = 1.0
    boundary_width: int = 2
    # Caps for a smoke run; None means the whole split.
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None
    label_min_h_support: int = 10
    loss_weighting: str = "balanced"

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture: {self.architecture}")
        if self.accumulate_steps < 1:
            raise ValueError("accumulate_steps must be at least 1")
        if self.epochs > protocol.MAX_EPOCHS:
            raise ValueError(
                f"epochs {self.epochs} exceeds the protocol cap "
                f"{protocol.MAX_EPOCHS}"
            )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def loss_weights(self) -> loss_terms.LossWeights:
        return loss_terms.LossWeights(
            assignment=self.assignment_weight,
            boundary=self.boundary_weight,
            pairwise=self.pairwise_weight,
        )


class CompleteModel:
    """Segmentation architecture plus the hierarchical CATH classifier.

    A thin holder rather than an `nn.Module` subclass, so each half keeps its
    own construction path and the two architectures stay interchangeable.
    """

    def __init__(
        self,
        config: TrainConfig,
        input_dim: int,
        hierarchy: Optional[clf.CathHierarchy],
        device,
    ) -> None:
        import torch

        self.config = config
        self.device = device
        self.architecture = config.architecture

        # The encoder is part of the model only when it is being trained; the
        # frozen path gets its features from the T2.2 cache instead.
        self.encoder = None
        if config.finetune_encoder:
            from ..model import finetune as ft

            self.encoder = ft.build_trainable_encoder(
                ft.FinetuneConfig(
                    model_name=config.encoder_name,
                    trainable_layers=config.finetune_layers,
                    gradient_checkpointing=config.gradient_checkpointing,
                    max_tokens_per_forward=config.encoder_max_tokens,
                    torch_home=config.torch_home,
                )
            ).to(device)

        if config.architecture == "pairwise":
            self.seg_config = seg.SegmentationConfig(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
            self.segmentation = seg.build_network(self.seg_config).to(device)
        else:
            self.seg_config = qs.QuerySegmentationConfig(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
            self.segmentation = qs.build_query_network(self.seg_config).to(device)

        self.classifier = None
        self.hierarchy = hierarchy
        if config.train_classifier and hierarchy is not None:
            self.clf_config = clf.ClassifierConfig(
                domain_dim=config.hidden_dim,
                dropout=config.dropout,
                min_h_support=config.label_min_h_support,
                loss_weighting=config.loss_weighting,
            )
            self.classifier = clf.build_classifier(hierarchy, self.clf_config).to(
                device
            )

    def parameters(self):
        yield from self.head_parameters()
        yield from self.encoder_parameters()

    def head_parameters(self):
        yield from self.segmentation.parameters()
        if self.classifier is not None:
            yield from self.classifier.parameters()

    def encoder_parameters(self):
        if self.encoder is not None:
            yield from self.encoder.trainable_parameters()

    def encode(self, batch):
        """Features for a batch, from the encoder in the graph.

        Returns the same ``(features, mask, lengths)`` triple as
        `EmbeddingBatcher.build`, so nothing downstream has to know which
        path produced it.
        """
        if self.encoder is None:
            raise RuntimeError("this model has no encoder in its graph")
        missing = [e.chain_id for e in batch if not e.sequence]
        if missing:
            raise ValueError(
                "fine-tuning needs residue sequences; load examples with "
                f"with_sequences=True (missing for {missing[:3]})"
            )
        return self.encoder([e.sequence for e in batch], device=self.device)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def train(self) -> None:
        self.segmentation.train()
        if self.classifier is not None:
            self.classifier.train()
        if self.encoder is not None:
            self.encoder.train()

    def eval(self) -> None:
        self.segmentation.eval()
        if self.classifier is not None:
            self.classifier.eval()
        if self.encoder is not None:
            self.encoder.eval()

    def state_dict(self) -> Dict[str, Any]:
        state = {"segmentation": self.segmentation.state_dict()}
        if self.classifier is not None:
            state["classifier"] = self.classifier.state_dict()
            state["hierarchy"] = self.hierarchy.as_dict()  # type: ignore[union-attr]
        state["seg_config"] = asdict(self.seg_config)
        state["architecture"] = self.architecture
        if self.encoder is not None:
            # The fine-tuned weights are the point of such a run, so the
            # checkpoint carries them even though it triples in size.
            state["encoder"] = self.encoder.esm.state_dict()
            state["encoder_name"] = self.config.encoder_name
        return state


def _sampled_pairs(model: CompleteModel, batch, config, source_aware: bool):
    """Supervised residue pairs for a batch: indices, targets and weights.

    One bounded sample per chain, which is T2.7's pair sampling. When
    `source_aware` is on, a chain whose source cannot answer "are these two
    residues in the same domain?" contributes no pairs at all rather than
    pairs with a zero weight, so it costs nothing to score either.
    """
    import torch

    pair_index: List[Any] = []
    pair_values = []
    pair_weights = []
    skipped = 0
    for row, example in enumerate(batch):
        supervision = example.effective_supervision if source_aware else None
        if supervision is not None and not supervision.pairwise:
            skipped += 1
            continue
        target, supervised = seg.pair_targets(example.length, example.domains)
        rows, cols = seg.sample_pairs(
            supervised, limit=config.pairs_per_chain, seed=config.seed + row
        )
        if not len(rows):
            continue
        pair_index.append(
            torch.stack(
                [
                    torch.from_numpy(rows.astype("int64")),
                    torch.from_numpy(cols.astype("int64")),
                ]
            )
        )
        pair_values.append(torch.from_numpy(target[rows, cols].astype("float32")))
        weight = supervision.weight if supervision is not None else 1.0
        pair_weights.append(torch.full((len(rows),), float(weight)))
    return pair_index, pair_values, pair_weights, skipped


def _segmentation_step(model: CompleteModel, features, mask, batch, config, out=None):
    """Segmentation loss for one batch, per architecture.

    Two losses live here. The default is T2.3's combined one: a 3-way residue
    cross-entropy plus binary cross-entropy over sampled pairs, with every
    record treated alike. `config.source_aware` selects T2.6's instead, which
    splits the residue term into separate assignment and boundary losses and
    masks each term by what the record's source is entitled to supervise.
    """
    import torch

    if out is None:
        out = model.segmentation(features, mask=mask)

    if config.source_aware:
        return _source_aware_step(model, features, mask, batch, config, out)

    if model.architecture == "pairwise":
        residue_targets = []
        longest = features.shape[1]
        for example in batch:
            labels = seg.residue_targets(example.length, example.domains)
            residue_targets.append(labels + [0] * (longest - example.length))
        residue_target = torch.tensor(residue_targets, device=features.device)
        pair_index, pair_values, _, _ = _sampled_pairs(
            model, batch, config, source_aware=False
        )

        total = seg.segmentation_loss(
            out["residue_logits"],
            residue_target,
            # Pair logits for the sampled pairs of every chain in the batch.
            _gather_pairs(model, out["features"], pair_index),
            torch.cat(pair_values).to(features.device)
            if pair_values
            else torch.zeros(1, device=features.device),
            mask=mask,
        )
        return total["loss"], out, {"residue": float(total["residue_loss"]),
                                    "pair": float(total["pair_loss"])}

    targets = [qs.mask_targets(e.length, e.domains) for e in batch]
    result = qs.query_segmentation_loss(
        out["class_logits"],
        out["mask_logits"],
        targets,
        config=model.seg_config,
        lengths=[e.length for e in batch],
    )
    return result["loss"], out, {
        "class": float(result["class_loss"]),
        "mask": float(result["mask_loss"]),
        "dice": float(result["dice_loss"]),
    }


def _source_aware_step(model: CompleteModel, features, mask, batch, config, out):
    """T2.6's composite loss for one batch."""
    import torch

    supervisions = [e.effective_supervision for e in batch]
    weights = config.loss_weights()

    if model.architecture == "query":
        # A per-chain gate, not a per-residue mask: see the reasoning in
        # `query_segmentation_loss`.
        chain_weights = [
            s.weight if s.complete_partition and s.residue_labels else 0.0
            for s in supervisions
        ]
        result = qs.query_segmentation_loss(
            out["class_logits"],
            out["mask_logits"],
            [qs.mask_targets(e.length, e.domains) for e in batch],
            config=model.seg_config,
            lengths=[e.length for e in batch],
            chain_weights=chain_weights,
        )
        parts = {
            "class": float(result["class_loss"]),
            "mask": float(result["mask_loss"]),
            "dice": float(result["dice_loss"]),
            "chains_masked_out": sum(1 for w in chain_weights if w <= 0.0),
        }
        return result["loss"], out, parts

    targets = loss_terms.build_residue_targets(
        [e.length for e in batch],
        [e.domains for e in batch],
        supervisions,
        boundary_width=config.boundary_width,
        longest=features.shape[1],
    ).to(features.device)

    pair_index, pair_values, pair_weights, skipped = _sampled_pairs(
        model, batch, config, source_aware=True
    )
    pair_logits = pair_target = pair_weight = None
    if pair_index:
        pair_logits = _gather_pairs(model, out["features"], pair_index)
        pair_target = torch.cat(pair_values).to(features.device)
        pair_weight = torch.cat(pair_weights).to(features.device)

    result = loss_terms.source_aware_loss(
        out["residue_logits"],
        targets,
        pair_logits=pair_logits,
        pair_target=pair_target,
        pair_weight=pair_weight,
        weights=weights,
    )
    parts = dict(result["parts"])
    parts["chains_without_pairs"] = skipped
    return result["loss"], out, parts


def _gather_pairs(model: CompleteModel, features, pair_index):
    """Score each chain's sampled pairs with the shared pair head."""
    import torch

    if not pair_index:
        return torch.zeros(1, device=features.device)
    scores = []
    for row, pairs in enumerate(pair_index):
        rows, cols = pairs[0], pairs[1]
        scores.append(
            model.segmentation.pair_head.score_pairs(
                features[row : row + 1], rows, cols
            ).reshape(-1)
        )
    return torch.cat(scores)


def _classification_step(model: CompleteModel, features, batch, config):
    """Classifier loss on ground-truth segments, pooled from trunk features."""
    import torch

    if model.classifier is None or model.hierarchy is None:
        return None, {}

    pooled, labels = [], {level: [] for level in ("C", "A", "T", "H")}
    for row, example in enumerate(batch):
        # T2.6's source mask for this term: a record whose source grants no
        # CATH labels contributes nothing here. Pfam records carry family
        # identifiers, not CATH nodes, so training on them would mean inventing
        # a mapping the dataset does not have.
        if config.source_aware and not example.effective_supervision.cath_labels:
            continue
        for index, segments in enumerate(example.domains):
            codes = {
                level: (example.labels.get(level) or [""] * example.n_domains)[index]
                for level in labels
            }
            if not codes.get("C"):
                continue
            pooled.append(clf.pool_domain(features[row], segments))
            for level in labels:
                labels[level].append(codes.get(level, ""))
    if not pooled:
        return None, {}

    stacked = torch.stack(pooled)
    encoded = {
        level: torch.as_tensor(values, device=features.device)
        for level, values in model.hierarchy.encode(labels).items()
    }
    logits = model.classifier(stacked)
    loss, parts = model.classifier.loss(logits, encoded)
    return loss, parts


def _decode_batch(model: CompleteModel, out, batch) -> List[List[List[Tuple[int, int]]]]:
    """Predicted domains for every chain in a batch."""
    import torch

    predictions = []
    if model.architecture == "pairwise":
        decoder = dec.DomainDecoder()
        residue = torch.softmax(out["residue_logits"].float(), dim=-1).cpu().numpy()
        for row, example in enumerate(batch):
            pair = torch.sigmoid(
                model.segmentation.pair_head(
                    out["features"][row : row + 1, : example.length]
                )[0].float()
            ).cpu().numpy()
            result = decoder.decode(residue[row, : example.length], pair)
            predictions.append([list(d) for d in result.domains])
    else:
        for row, example in enumerate(batch):
            predictions.append(
                [
                    list(d)
                    for d in qs.decode_queries(
                        out["class_logits"][row].float().cpu(),
                        out["mask_logits"][row].float().cpu(),
                        example.length,
                    )
                ]
            )
    return predictions


def evaluate(
    model: CompleteModel,
    examples: Sequence[ChainExample],
    batcher: EmbeddingBatcher,
    config: TrainConfig,
) -> Dict[str, Any]:
    """Score a split: segmentation per group, plus classification both ways."""
    import torch

    model.eval()
    # Batching sorts by length, so a prediction is kept next to its own
    # example rather than matched back by position later.
    rows: List[Tuple[ChainExample, Any]] = []
    true_labels = {level: [] for level in ("C", "A", "T", "H")}
    true_predicted = {level: [] for level in ("C", "A", "T", "H")}
    abstained = 0
    classified = 0

    with torch.no_grad():
        for batch in iter_epoch(
            examples, config.max_tokens, config.seed, shuffle=False
        ):
            features, mask, _ = (
                model.encode(batch) if model.encoder is not None
                else batcher.build(batch)
            )
            out = model.segmentation(features, mask=mask)
            for example, prediction in zip(batch, _decode_batch(model, out, batch)):
                rows.append((example, prediction))

            if model.classifier is not None and model.hierarchy is not None:
                pooled, wanted = [], {l: [] for l in true_labels}
                for row, example in enumerate(batch):
                    for index, segments in enumerate(example.domains):
                        code = (example.labels.get("H") or [""] * example.n_domains)[index]
                        if not code:
                            continue
                        pooled.append(clf.pool_domain(out["features"][row], segments))
                        for level in wanted:
                            wanted[level].append(
                                (example.labels.get(level) or [""] * example.n_domains)[index]
                            )
                if pooled:
                    # Deliberately not named `rows`: that holds the
                    # (example, prediction) pairs this loop is filling.
                    predicted = model.classifier.predict(torch.stack(pooled))
                    for position, row_result in enumerate(predicted):
                        labels = row_result["labels"]
                        classified += 1
                        if labels.get("H") == clf.UNRESOLVED:
                            abstained += 1
                        for level in true_labels:
                            true_labels[level].append(wanted[level][position])
                            true_predicted[level].append(
                                labels.get(level, clf.UNRESOLVED)
                            )

    # Segmentation, overall and per reporting group. A chain belongs to
    # several groups ("all", "multidomain", "discontinuous", ...), so it is
    # emitted once per group it belongs to.
    keys: List[str] = []
    expanded_predictions: List[Any] = []
    expanded_targets: List[Any] = []
    for example, prediction in rows:
        for group in protocol.chain_groups(
            example.n_domains, example.discontinuous, example.length
        ):
            keys.append(group)
            expanded_predictions.append(prediction)
            expanded_targets.append(example.domains)

    grouped: Dict[str, Any] = {}
    for group, (preds, truths) in metrics.stratify(
        keys, expanded_predictions, expanded_targets
    ).items():
        grouped[group] = metrics.score_segmentation(preds, truths).as_dict()
    if "all" not in grouped:
        grouped["all"] = metrics.score_segmentation([], []).as_dict()

    results: Dict[str, Any] = {"segmentation": grouped}
    if classified:
        scores = metrics.score_classification(true_labels, true_predicted)
        results["classification_true_segments"] = {
            "domains": classified,
            "levels": scores.levels,
            "h_coverage": round((classified - abstained) / classified, 4),
        }
    return results


def train(
    config: TrainConfig,
    train_examples: Sequence[ChainExample],
    val_examples: Sequence[ChainExample],
    batcher: EmbeddingBatcher,
    val_batcher: EmbeddingBatcher,
    hierarchy: Optional[clf.CathHierarchy],
    record: protocol.RunRecord,
    registry: protocol.ExperimentRegistry,
    checkpoint_dir: str,
    feature_dim: Optional[int] = None,
    log=print,
) -> protocol.RunRecord:
    """Train one model, selecting the checkpoint on validation mean IoU."""
    import torch

    torch.manual_seed(config.seed)

    # A fine-tuning run has no batcher: its features come from the encoder in
    # the graph, so the device and feature width are passed in instead.
    device = batcher.device if batcher is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if feature_dim is None:
        if batcher is None:
            raise ValueError("feature_dim is required without a batcher")
        feature_dim = batcher.dim
    model = CompleteModel(config, feature_dim, hierarchy, device)
    log(
        f"{config.architecture}: {model.parameter_count():,} parameters "
        f"(classifier {'on' if model.classifier is not None else 'off'})"
    )

    groups = [
        {
            "params": list(model.head_parameters()),
            "lr": config.learning_rate,
            "name": "heads",
        }
    ]
    encoder_params = list(model.encoder_parameters())
    if encoder_params:
        # A pretrained encoder at the heads' learning rate is destroyed in the
        # first epoch, so it gets its own group two orders of magnitude lower.
        groups.append(
            {
                "params": encoder_params,
                "lr": config.encoder_learning_rate,
                "name": "encoder",
            }
        )
        log(
            f"encoder: {model.encoder.trainable_count():,} trainable of "
            f"{sum(p.numel() for p in model.encoder.esm.parameters()):,} "
            f"parameters, lr {config.encoder_learning_rate}, "
            f"checkpointing {'on' if config.gradient_checkpointing else 'off'}"
        )
    optimiser = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    use_amp = config.mixed_precision and device is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    watch = protocol.Stopwatch(gpus=1)
    best: Optional[float] = None
    stale = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        optimiser.zero_grad(set_to_none=True)
        for step, batch in enumerate(
            iter_epoch(train_examples, config.max_tokens, config.seed + epoch)
        ):
            features, mask, _ = (
                model.encode(batch) if model.encoder is not None
                else batcher.build(batch)
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                seg_loss, out, _ = _segmentation_step(
                    model, features, mask, batch, config
                )
                loss = seg_loss
                clf_loss, _ = _classification_step(
                    model, out["features"], batch, config
                )
                if clf_loss is not None:
                    # The fourth of T2.6's terms. It keeps its own weight
                    # rather than going through `source_aware_loss`, because
                    # it is computed on pooled ground-truth segments and so
                    # has no residue-level mask to apply.
                    loss = loss + config.classifier_weight * clf_loss
            # Divide by the accumulation count so the effective batch is an
            # average, not a sum, and the step size does not depend on it.
            scaler.scale(loss / config.accumulate_steps).backward()

            if (step + 1) % config.accumulate_steps == 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()), config.grad_clip
                )
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

            total_loss += float(loss)
            batches += 1

        # Flush a partial accumulation window so its gradient is not discarded.
        if batches % config.accumulate_steps:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), config.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

        scores = evaluate(model, val_examples, val_batcher, config)
        value = float(scores["segmentation"]["all"]["mean_iou"])
        entry = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(batches, 1), 5),
            protocol.CHECKPOINT_METRIC: value,
            "elapsed_minutes": round(watch.seconds / 60, 2),
        }
        if "classification_true_segments" in scores:
            levels = scores["classification_true_segments"]["levels"]
            entry["val_h_accuracy"] = levels.get("H", {}).get("accuracy")
            entry["val_h_macro_f1"] = levels.get("H", {}).get("macro_f1")
            entry["val_h_coverage"] = scores["classification_true_segments"][
                "h_coverage"
            ]
        record.history.append(entry)
        log(f"  epoch {epoch:2d} {json.dumps(entry)}")

        if protocol.is_better(value, best):
            best = value
            stale = 0
            path = os.path.join(checkpoint_dir, f"{record.run_id}-best.pt")
            torch.save(
                {
                    "state": model.state_dict(),
                    "config": config.as_dict(),
                    "epoch": epoch,
                    protocol.CHECKPOINT_METRIC: value,
                    "run_id": record.run_id,
                },
                path,
            )
            record.selected = {
                "epoch": epoch,
                protocol.CHECKPOINT_METRIC: value,
                "checkpoint": path,
            }
            record.results = scores
        else:
            stale += 1

        record.epochs_run = epoch + 1
        record.runtime_seconds = round(watch.seconds, 1)
        record.gpu_hours = watch.gpu_hours
        registry.save(record)

        if stale >= protocol.EARLY_STOPPING_PATIENCE:
            record.notes.append(
                f"early stopped at epoch {epoch}: no improvement in "
                f"{protocol.CHECKPOINT_METRIC} for {stale} epochs"
            )
            break
        if watch.exceeded():
            record.notes.append(
                f"stopped at epoch {epoch}: run exceeded the "
                f"{protocol.MAX_RUN_HOURS}h per-run cap"
            )
            break

    record.status = "completed"
    record.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record.runtime_seconds = round(watch.seconds, 1)
    record.gpu_hours = watch.gpu_hours
    registry.save(record)
    return record
