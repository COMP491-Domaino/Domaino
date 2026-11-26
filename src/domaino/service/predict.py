"""Turning a sequence into the contract's `Prediction` object (task T4.2).

Two predictors implement the same call, and the service picks one at startup:

`ModelPredictor`
    The real path: frozen ESM-2 from T2.2, one of the two trained
    segmentation models from T3.2, the decoder from T2.4 and the hierarchical
    classifier from T2.5.

`MockPredictor`
    A deterministic simulation, so the interface can be built and demonstrated
    before training finishes -- which is what milestone M4.1 asks for. Every
    mock result carries ``model.mode == "mock"``, the interface shows that
    prominently, and a simulated result is never presented as a prediction.

Both go through `format_prediction`, so the frontend sees one shape and the
mock cannot drift away from what the real model returns.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .contract import API_VERSION

# CATH names for the codes the mock emits, and for labelling real results. A
# small table on purpose: the full CATH name list is a WP4 data task (T4.4),
# and a wrong name is worse than none.
CATH_NAMES = {
    "1": "Mainly Alpha",
    "2": "Mainly Beta",
    "3": "Alpha Beta",
    "4": "Few Secondary Structures",
    "6": "Special",
    "1.10": "Orthogonal Bundle",
    "1.20": "Up-down Bundle",
    "2.40": "Beta Barrel",
    "2.60": "Sandwich",
    "3.40": "3-Layer(aba) Sandwich",
    "3.30": "2-Layer Sandwich",
    "3.90": "Alpha-Beta Complex",
}

UNRESOLVED = "unresolved"


@dataclass
class ResolvedOptions:
    """Request options after defaults are applied."""

    architecture: str = "pairwise"
    include_tracks: bool = False
    include_structure: bool = False

    @classmethod
    def from_request(
        cls, options: Optional[Dict[str, Any]], default_architecture: str
    ) -> "ResolvedOptions":
        options = options or {}
        return cls(
            architecture=str(options.get("architecture") or default_architecture),
            include_tracks=bool(options.get("include_tracks", False)),
            include_structure=bool(options.get("include_structure", False)),
        )


def _level_payload(
    code: Optional[str], confidence: float, reason: Optional[str] = None
) -> Dict[str, Any]:
    if not code or code == UNRESOLVED:
        payload: Dict[str, Any] = {
            "code": None,
            "name": None,
            "confidence": round(float(confidence), 4),
            "status": "unresolved",
        }
        if reason:
            payload["reason"] = reason
        return payload
    return {
        "code": code,
        "name": CATH_NAMES.get(code),
        "confidence": round(float(confidence), 4),
        "status": "assigned",
    }


def _cath_payload(
    labels: Dict[str, str],
    confidence: Dict[str, float],
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for level in ("C", "A", "T", "H"):
        code = labels.get(level)
        payload[level] = _level_payload(
            code,
            confidence.get(level, 0.0),
            reason if (not code or code == UNRESOLVED) else None,
        )
    assigned = [
        payload[level]["code"] for level in ("C", "A", "T", "H")
        if payload[level]["status"] == "assigned"
    ]
    # The deepest assigned code already contains the shallower ones, so the
    # path is that code rather than a join of the four.
    payload["path"] = assigned[-1] if assigned else ""
    return payload


def unassigned_intervals(
    length: int, domains: Sequence[Sequence[Tuple[int, int]]]
) -> List[Dict[str, int]]:
    """Residues in no predicted domain, as 1-based inclusive intervals.

    The frontend colours every residue, so a gap it was never told about would
    render as a silent hole in the sequence map.
    """
    covered = [False] * (length + 1)
    for segments in domains:
        for start, end in segments:
            for position in range(max(1, start), min(length, end) + 1):
                covered[position] = True
    intervals: List[Dict[str, int]] = []
    position = 1
    while position <= length:
        if covered[position]:
            position += 1
            continue
        start = position
        while position <= length and not covered[position]:
            position += 1
        intervals.append({"start": start, "end": position - 1})
    return intervals


def format_prediction(
    sequence_length: int,
    domains: Sequence[Sequence[Tuple[int, int]]],
    cath: Sequence[Optional[Dict[str, Any]]],
    confidences: Sequence[float],
    model: Dict[str, Any],
    name: Optional[str] = None,
    warnings: Optional[List[Dict[str, Any]]] = None,
    tracks: Optional[Dict[str, List[float]]] = None,
    retention: Optional[str] = None,
    code_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the contract's `Prediction` from decoded domains."""
    payload_domains = []
    for index, segments in enumerate(domains):
        ordered = sorted((int(a), int(b)) for a, b in segments)
        payload_domains.append(
            {
                "index": index + 1,
                "segments": [{"start": a, "end": b} for a, b in ordered],
                "length": sum(b - a + 1 for a, b in ordered),
                "discontinuous": len(ordered) > 1,
                "confidence": round(float(confidences[index]), 4),
                **({"cath": cath[index]} if index < len(cath) and cath[index] else {}),
            }
        )

    prediction: Dict[str, Any] = {
        "sequence_length": int(sequence_length),
        "domains": payload_domains,
        "unassigned": unassigned_intervals(sequence_length, domains),
        "model": model,
        "warnings": list(warnings or []),
        "provenance": {
            "api_version": API_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_revision": code_revision,
            "retention": retention or "not stated",
        },
    }
    if name:
        prediction["name"] = name
    if tracks:
        prediction["tracks"] = tracks
    return prediction


class MockPredictor:
    """A deterministic simulated prediction, labelled as one.

    Deterministic so that the interface can be developed against a stable
    result and so a test can assert on it: the same sequence always gives the
    same domains. The shapes it produces are drawn from the real distribution
    -- roughly three quarters of CATH chains are single-domain, and
    discontinuous domains are a minority -- so the frontend is exercised on
    realistic cases rather than on the easy one.
    """

    mode = "mock"

    def __init__(self, architecture: str = "pairwise") -> None:
        self.architecture = architecture

    def model_info(self, options: "ResolvedOptions") -> Dict[str, Any]:
        return {
            "architecture": options.architecture,
            "mode": self.mode,
            "run_id": None,
            "encoder": None,
            "dataset_version": None,
            "selection": {"metric": "val_mean_iou", "value": None, "epoch": None},
        }

    def predict(
        self,
        sequence: str,
        options: "ResolvedOptions",
        name: Optional[str] = None,
        warnings: Optional[List[Dict[str, Any]]] = None,
        progress: Optional[Callable[[str, str], None]] = None,
        retention: Optional[str] = None,
    ) -> Dict[str, Any]:
        length = len(sequence)
        if progress:
            progress("embedding", "simulating residue embeddings")
        seed = int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16)

        if progress:
            progress("segmenting", "simulating domain boundaries")
        domains = self._shape(length, seed)

        if progress:
            progress("classifying", "simulating CATH labels")
        cath, confidences = [], []
        for index, _ in enumerate(domains):
            cath.append(self._label(seed + index))
            confidences.append(round(0.6 + ((seed >> (index + 3)) % 35) / 100.0, 4))

        warnings = list(warnings or [])
        warnings.append(
            {
                "code": "label_unresolved",
                "message": "this is a simulated result from the mocked backend, "
                "not a prediction; no trained model was loaded",
            }
        )
        tracks = self._tracks(length, domains) if options.include_tracks else None
        if options.include_structure:
            warnings.append(
                {
                    "code": "structure_unavailable",
                    "message": "structure prediction is not part of the mocked "
                    "backend; the sequence-only result is unaffected",
                }
            )
        return format_prediction(
            length, domains, cath, confidences,
            model=self.model_info(options), name=name, warnings=warnings,
            tracks=tracks, retention=retention,
        )

    def _shape(self, length: int, seed: int) -> List[List[Tuple[int, int]]]:
        """Plausible domain layout for a chain of this length."""
        if length < 90:
            return [[(1, length)]]
        roll = seed % 100
        if roll < 55:
            return [[(1, length)]]
        if roll < 70 and length >= 200:
            # A discontinuous domain wrapping a continuous one, the layout a
            # boundary-only model cannot express.
            third = length // 3
            return [
                [(1, third), (2 * third + 1, length)],
                [(third + 1, 2 * third)],
            ]
        if roll < 90:
            cut = max(40, min(length - 40, length // 2 + (seed % 21) - 10))
            return [[(1, cut)], [(cut + 1, length)]]
        step = length // 3
        return [[(1, step)], [(step + 1, 2 * step)], [(2 * step + 1, length)]]

    def _label(self, seed: int) -> Dict[str, Any]:
        classes = ["1", "2", "3"]
        architectures = {"1": ["1.10", "1.20"], "2": ["2.40", "2.60"],
                         "3": ["3.40", "3.30", "3.90"]}
        code = classes[seed % len(classes)]
        architecture = architectures[code][(seed >> 2) % len(architectures[code])]
        topology = f"{architecture}.{10 * (1 + (seed >> 4) % 9)}"
        resolved = (seed >> 6) % 10 < 6
        labels = {
            "C": code, "A": architecture, "T": topology,
            "H": f"{topology}.{10 * (1 + (seed >> 8) % 5)}" if resolved else UNRESOLVED,
        }
        confidence = {
            "C": 0.80 + ((seed >> 3) % 18) / 100.0,
            "A": 0.65 + ((seed >> 5) % 25) / 100.0,
            "T": 0.50 + ((seed >> 7) % 35) / 100.0,
            "H": 0.55 + ((seed >> 9) % 30) / 100.0 if resolved else 0.21,
        }
        return _cath_payload(
            labels, confidence,
            reason=None if resolved else "below the calibrated confidence threshold",
        )

    def _tracks(self, length, domains) -> Dict[str, List[float]]:
        in_domain = [0.05] * length
        boundary = [0.02] * length
        for segments in domains:
            for start, end in segments:
                for position in range(start, end + 1):
                    in_domain[position - 1] = 0.95
                for edge in (start, end):
                    for position in range(edge - 2, edge + 3):
                        if 1 <= position <= length:
                            boundary[position - 1] = 0.8
        return {"in_domain": in_domain, "boundary": boundary}


class ModelPredictor:
    """The trained path: encoder, segmentation model, decoder, classifier.

    Constructed from a T3.2 checkpoint, which already carries its own
    architecture, segmentation config and label space, so the service does not
    have to be told which model it is serving.
    """

    mode = "trained"

    def __init__(
        self,
        checkpoint_path: str,
        encoder_config: Optional[Any] = None,
        device: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        import torch

        from ..model import classifier as clf
        from ..model import query_segmentation as qs
        from ..model import representation as rep
        from ..model import segmentation as seg

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload["state"]
        self.architecture = state["architecture"]
        self.config = payload.get("config", {})
        self.run_id = run_id or payload.get("run_id")
        self.selection = {
            "metric": "val_mean_iou",
            "value": payload.get("val_mean_iou"),
            "epoch": payload.get("epoch"),
        }

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.architecture == "pairwise":
            seg_config = seg.SegmentationConfig(**state["seg_config"])
            self.segmentation = seg.build_network(seg_config)
        else:
            seg_config = qs.QuerySegmentationConfig(**state["seg_config"])
            self.segmentation = qs.build_query_network(seg_config)
        self.segmentation.load_state_dict(state["segmentation"])
        self.segmentation.eval().to(self.device)
        self.seg_config = seg_config

        self.hierarchy = None
        self.classifier = None
        if "classifier" in state and "hierarchy" in state:
            self.hierarchy = clf.CathHierarchy.from_dict(state["hierarchy"])
            clf_config = clf.ClassifierConfig(
                domain_dim=seg_config.hidden_dim,
                dropout=0.0,
                min_h_support=int(self.config.get("label_min_h_support", 10)),
                loss_weighting=str(self.config.get("loss_weighting", "balanced")),
            )
            self.classifier = clf.build_classifier(self.hierarchy, clf_config)
            self.classifier.load_state_dict(state["classifier"])
            self.classifier.eval().to(self.device)

        self.encoder = rep.Esm2Encoder(encoder_config)
        self.encoder_name = self.encoder.config.model_name

    def model_info(self, options: "ResolvedOptions") -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "mode": self.mode,
            "run_id": self.run_id,
            "encoder": self.encoder_name,
            "dataset_version": self.config.get("dataset_version", "dataset_v1"),
            "selection": self.selection,
        }

    def predict(
        self,
        sequence: str,
        options: "ResolvedOptions",
        name: Optional[str] = None,
        warnings: Optional[List[Dict[str, Any]]] = None,
        progress: Optional[Callable[[str, str], None]] = None,
        retention: Optional[str] = None,
    ) -> Dict[str, Any]:
        import torch

        from ..model import classifier as clf
        from ..model import decode as dec
        from ..model import query_segmentation as qs

        length = len(sequence)
        warnings = list(warnings or [])

        if progress:
            progress("embedding", f"embedding {length} residues with {self.encoder_name}")
        features = self.encoder.embed([sequence])[0].unsqueeze(0).to(self.device)
        mask = torch.ones(1, length, device=self.device)

        if progress:
            progress("segmenting", "predicting domain boundaries")
        with torch.no_grad():
            out = self.segmentation(features.float(), mask=mask)
            if self.architecture == "pairwise":
                residue = torch.softmax(out["residue_logits"][0].float(), dim=-1)
                pair = torch.sigmoid(
                    self.segmentation.pair_head(out["features"][:, :length])[0].float()
                )
                decoded = dec.DomainDecoder().decode(
                    residue.cpu().numpy(), pair.cpu().numpy()
                )
                domains = [list(d) for d in decoded.domains]
                confidences = self._pairwise_confidence(domains, pair.cpu())
                tracks = (
                    {
                        "in_domain": [
                            round(float(residue[i, dec.DOMAIN] + residue[i, dec.BOUNDARY]), 4)
                            for i in range(length)
                        ],
                        "boundary": [
                            round(float(residue[i, dec.BOUNDARY]), 4)
                            for i in range(length)
                        ],
                    }
                    if options.include_tracks
                    else None
                )
            else:
                class_logits = out["class_logits"][0].float().cpu()
                mask_logits = out["mask_logits"][0].float().cpu()
                domains = [
                    list(d) for d in qs.decode_queries(class_logits, mask_logits, length)
                ]
                probabilities = torch.softmax(class_logits, dim=-1)[:, 1]
                confidences = self._query_confidence(domains, probabilities)
                tracks = (
                    {
                        "in_domain": [
                            round(float(torch.sigmoid(mask_logits[:, i]).max()), 4)
                            for i in range(length)
                        ]
                    }
                    if options.include_tracks
                    else None
                )

            cath: List[Optional[Dict[str, Any]]] = [None] * len(domains)
            if self.classifier is not None and domains:
                if progress:
                    progress("classifying", f"classifying {len(domains)} domains")
                pooled = torch.stack(
                    [clf.pool_domain(out["features"][0], d) for d in domains]
                )
                for index, row in enumerate(self.classifier.predict(pooled)):
                    labels = {k: str(v) for k, v in dict(row["labels"]).items()}
                    cath[index] = _cath_payload(
                        labels,
                        {k: float(v) for k, v in dict(row["confidence"]).items()},
                        reason="the superfamily is below the support or "
                        "confidence threshold, so no H label is given",
                    )
                    if labels.get("H") == clf.UNRESOLVED:
                        warnings.append(
                            {
                                "code": "label_unresolved",
                                "message": f"domain {index + 1} has no "
                                "superfamily label: the model was not "
                                "confident enough to give one",
                                "details": {"domain": index + 1},
                            }
                        )

        if options.include_structure:
            warnings.append(
                {
                    "code": "structure_unavailable",
                    "message": "structure prediction is not wired up yet "
                    "(T4.5); the sequence-only result is unaffected",
                }
            )

        return format_prediction(
            length, domains, cath, confidences,
            model=self.model_info(options), name=name, warnings=warnings,
            tracks=tracks, retention=retention,
        )

    @staticmethod
    def _pairwise_confidence(domains, pair) -> List[float]:
        """Mean same-domain probability among a domain's own residue pairs.

        A domain whose residues the model is sure belong together scores high;
        one the clustering only just held together does not. This is a
        segmentation confidence and is deliberately separate from the CATH
        label's confidence.
        """
        values = []
        for segments in domains:
            positions = [
                index - 1
                for start, end in segments
                for index in range(start, end + 1)
            ]
            if len(positions) < 2:
                values.append(0.0)
                continue
            # Sampled rather than exhaustive: a 600-residue domain has 180,000
            # pairs and the mean converges long before that.
            step = max(1, len(positions) // 64)
            sampled = positions[::step]
            total = count = 0.0
            for i, a in enumerate(sampled):
                for b in sampled[i + 1 :]:
                    total += float(pair[a, b])
                    count += 1
            values.append(round(total / count, 4) if count else 0.0)
        return values

    @staticmethod
    def _query_confidence(domains, probabilities) -> List[float]:
        """The domain-vs-no-domain probability of each kept query."""
        ordered = sorted(
            range(probabilities.shape[0]),
            key=lambda i: float(probabilities[i]),
            reverse=True,
        )
        return [
            round(float(probabilities[ordered[index]]), 4)
            if index < len(ordered)
            else 0.0
            for index in range(len(domains))
        ]


def build_predictor(
    checkpoint_path: Optional[str] = None,
    encoder_config: Optional[Any] = None,
    device: Optional[str] = None,
    architecture: str = "pairwise",
    log: Callable[[str], None] = print,
):
    """The real predictor if a checkpoint loads, otherwise the mock.

    Falling back rather than failing is deliberate: the interface and its
    tests must run on a laptop with no checkpoint and no GPU, and a mock
    result is unmistakable in the response.
    """
    if not checkpoint_path:
        log("no checkpoint given: serving mocked predictions")
        return MockPredictor(architecture)
    try:
        predictor = ModelPredictor(checkpoint_path, encoder_config, device)
        log(
            f"loaded {predictor.architecture} checkpoint {checkpoint_path} "
            f"(run {predictor.run_id})"
        )
        return predictor
    except Exception as problem:  # noqa: BLE001 - reported, not swallowed
        log(f"could not load {checkpoint_path}: {problem!r}")
        log("serving mocked predictions instead")
        return MockPredictor(architecture)
