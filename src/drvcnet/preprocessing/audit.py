"""Stratified assignment-audit sampling, summaries, and threshold selection."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

from drvcnet.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from drvcnet.records import FrameAssignment, UtteranceRecord

AUDIT_LABELS = {"correct", "incorrect", "undecidable"}


def _assignment_score(value: dict[str, Any]) -> float:
    if "quality_score" in value:
        return float(value["quality_score"])
    visibility = value.get("visibility", {})
    visible = max((float(item) for item in visibility.values()), default=0.0)
    return (
        float(value.get("detection_confidence", 0.0))
        * float(value.get("speaker_confidence", 0.0))
        * visible
    )


def build_assignment_audit_sample(
    manifest_path: str | Path,
    assignments_path: str | Path,
    output_path: str | Path,
    *,
    dataset: str,
    seed: int = 2024,
) -> list[dict[str, Any]]:
    """Sample the highest-quality accepted assignment per utterance by protocol stratum."""

    dataset = dataset.upper()
    if dataset not in {"IEMOCAP", "MELD"}:
        raise ValueError("dataset must be IEMOCAP or MELD")
    manifest = {
        value.utterance_id: value
        for value in (
            UtteranceRecord.from_dict(record) for record in read_jsonl(manifest_path)
        )
    }
    best: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(assignments_path):
        assignment = FrameAssignment.from_dict(value)
        if not assignment.assignment_valid or assignment.utterance_id not in manifest:
            continue
        previous = best.get(assignment.utterance_id)
        if previous is None or _assignment_score(value) > _assignment_score(previous):
            best[assignment.utterance_id] = value

    strata: dict[str, list[str]] = defaultdict(list)
    for utterance_id in best:
        record = manifest[utterance_id]
        stratum = record.session if dataset == "IEMOCAP" else record.split
        if stratum is None:
            raise ValueError(f"Missing audit stratum for {utterance_id}")
        strata[str(stratum).lower()].append(utterance_id)
    if dataset == "IEMOCAP":
        requested = {name: 60 for name in strata}
    else:
        requested = {"train": 150, "dev": 60, "test": 90}

    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for stratum in sorted(requested):
        candidates = sorted(strata.get(stratum, []))
        sample_size = requested[stratum]
        if len(candidates) < sample_size:
            raise ValueError(
                f"Stratum {stratum} has {len(candidates)} accepted assignments; "
                f"{sample_size} are required"
            )
        sampled = rng.choice(candidates, size=sample_size, replace=False)
        for utterance_id in sorted(str(value) for value in sampled):
            utterance = manifest[utterance_id]
            assignment = best[utterance_id]
            output.append(
                {
                    "dataset": dataset,
                    "stratum": stratum,
                    "stratum_population": len(candidates),
                    "utterance_id": utterance_id,
                    "dialogue_id": utterance.dialogue_id,
                    "annotated_speaker_id": utterance.speaker_id,
                    "video_path": utterance.video_path,
                    "frame_index": int(assignment["frame_index"]),
                    "timestamp": float(assignment["timestamp"]),
                    "track_id": assignment.get("track_id"),
                    "image_path": assignment.get("image_path"),
                    "assignment_score": _assignment_score(assignment),
                }
            )
    atomic_write_jsonl(output_path, output)
    return output


def _validated_label(record: dict[str, Any], field: str) -> str:
    value = str(record.get(field, "")).lower()
    if value not in AUDIT_LABELS:
        raise ValueError(f"{record.get('utterance_id')}: invalid {field} value {value!r}")
    return value


def summarize_assignment_audit(
    audit_path: str | Path,
    output_path: str | Path,
    *,
    repeats: int = 10_000,
    seed: int = 2024,
) -> dict[str, Any]:
    """Compute pre-adjudication agreement and stratified accepted-assignment accuracy."""

    records = list(read_jsonl(audit_path))
    if not records:
        raise ValueError("Audit file is empty")
    paired_first: list[str] = []
    paired_second: list[str] = []
    by_stratum: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for record in records:
        first = _validated_label(record, "annotator_1")
        second = _validated_label(record, "annotator_2")
        final = _validated_label(record, "adjudicated")
        if first != "undecidable" and second != "undecidable":
            paired_first.append(first)
            paired_second.append(second)
        by_stratum[str(record["stratum"])].append((record, final))
    if not paired_first:
        raise ValueError("No independently decidable pairs are available for agreement")

    def estimate(sample: dict[str, list[tuple[dict[str, Any], str]]]) -> float:
        correct_weight = 0.0
        decidable_weight = 0.0
        for values in sample.values():
            population = int(values[0][0]["stratum_population"])
            weight = population / len(values)
            for _record, label in values:
                if label == "undecidable":
                    continue
                decidable_weight += weight
                correct_weight += weight * float(label == "correct")
        if decidable_weight == 0:
            raise ValueError("Audit has no adjudicated decidable assignments")
        return correct_weight / decidable_weight

    accuracy = estimate(by_stratum)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sample: dict[str, list[tuple[dict[str, Any], str]]] = {}
        for stratum, values in by_stratum.items():
            indices = rng.integers(0, len(values), size=len(values))
            sample[stratum] = [values[int(index)] for index in indices]
        bootstrap[repeat] = estimate(sample)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    decided = [label for values in by_stratum.values() for _record, label in values]
    payload = {
        "records": len(records),
        "independently_decidable_pairs": len(paired_first),
        "cohen_kappa": float(cohen_kappa_score(paired_first, paired_second)),
        "adjudicated_decidable": sum(label != "undecidable" for label in decided),
        "adjudicated_correct": sum(label == "correct" for label in decided),
        "design_adjusted_accuracy": accuracy,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
    }
    atomic_write_json(output_path, payload)
    return payload


def select_quality_threshold(
    records: Sequence[dict[str, Any]],
    candidates: Sequence[float],
    minimum_accuracy: float,
) -> dict[str, Any]:
    """Maximize retained coverage subject to the audited match-accuracy constraint."""

    evaluated: list[dict[str, Any]] = []
    for threshold in sorted(float(value) for value in candidates):
        accepted = [
            record
            for record in records
            if float(record.get("quality_score", record.get("quality", -1.0))) >= threshold
        ]
        accuracy = (
            float(np.mean([int(record["target_correct"]) for record in accepted]))
            if accepted
            else 0.0
        )
        evaluated.append(
            {
                "threshold": threshold,
                "retained": len(accepted),
                "coverage": len(accepted) / max(len(records), 1),
                "accuracy": accuracy,
                "eligible": bool(accepted) and accuracy >= minimum_accuracy,
            }
        )
    eligible = [value for value in evaluated if value["eligible"]]
    if not eligible:
        raise ValueError("No threshold satisfies the minimum audited match accuracy")
    selected = max(eligible, key=lambda value: (value["retained"], value["threshold"]))
    return {
        "selected_threshold": selected["threshold"],
        "minimum_accuracy": minimum_accuracy,
        "records": len(records),
        "candidates": evaluated,
    }
