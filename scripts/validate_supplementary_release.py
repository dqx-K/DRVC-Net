#!/usr/bin/env python3
"""Validate the public supplementary package without importing model code."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


FINAL_SEEDS = {42, 123, 2026}
STABILITY_SEEDS = FINAL_SEEDS | {314, 2025}
STABILITY_MODELS = {"B2", "C1", "C2", "C3", "B6"}
BIDIRECTIONAL_MODELS = {f"{model}_bidir" for model in ("B2", "C2", "C3", "B6")}

PREDICTION_FIELDS = {
    "run_id",
    "dataset",
    "dialogue_id",
    "fold",
    "model",
    "probabilities",
    "seed",
    "utterance_id",
    "y_pred",
    "y_true",
}
UTTERANCE_METADATA_FIELDS = {
    "dataset",
    "utterance_id",
    "dialogue_id",
    "speaker_id",
    "order",
    "shift_group",
    "reference_availability",
    "both",
    "history_bin",
    "reference_age",
    "reference_age_bin",
    "visual_missing",
    "current_mask",
    "recent_mask",
    "historical_mask",
    "reference_ids",
}
RUN_PROVENANCE_FIELDS = {
    "git",
    "software",
    "hardware",
    "data_and_config",
    "checkpoint",
    "prediction_file",
    "evaluation",
}
ADDITIONAL_FIELDS = {
    "intervention": {
        "intervention",
        "intervention_eligible",
        "requested_slots",
        "replaced_slots",
    },
    "stress": {
        "stress_condition",
        "stress_level",
        "corruption_seed",
    },
    "extension": set(),
}
EXPECTED_ADDITIONAL_TYPES = set(ADDITIONAL_FIELDS)
EXPECTED_PREDICTED_EXTENSION_MODELS = {"CSS_mix"}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    handle = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8")
    )
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _reported_run(record: dict[str, Any]) -> bool:
    dataset = str(record["dataset"])
    model = str(record["model"])
    seed = int(record["seed"])
    if model.endswith("_bidir"):
        return dataset == "MELD" and model in BIDIRECTIONAL_MODELS and seed in FINAL_SEEDS
    seeds = STABILITY_SEEDS if model in STABILITY_MODELS else FINAL_SEEDS
    return seed in seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    issues: list[str] = []

    manifest_path = root / "run_manifest.jsonl"
    run_records = list(_jsonl(manifest_path)) if manifest_path.is_file() else []
    if not manifest_path.is_file():
        issues.append("missing run_manifest.jsonl")
    run_ids = {str(record["run_id"]) for record in run_records}
    if len(run_records) != 774 or len(run_ids) != 774:
        issues.append(f"expected 774 unique run records; found {len(run_records)}/{len(run_ids)}")
    out_of_scope = [str(record["run_id"]) for record in run_records if not _reported_run(record)]
    if out_of_scope:
        issues.append(f"{len(out_of_scope)} run records fall outside the paper protocol")
    missing_provenance: set[str] = set()
    for record in run_records:
        missing_provenance.update(RUN_PROVENANCE_FIELDS - record.keys())
    if missing_provenance:
        issues.append(f"run records lack provenance fields: {sorted(missing_provenance)}")

    prediction_path = root / "predictions_main.jsonl.gz"
    prediction_count = 0
    prediction_runs: set[str] = set()
    prediction_keys: set[tuple[str, str]] = set()
    missing_prediction_fields: set[str] = set()
    if not prediction_path.is_file():
        issues.append("missing predictions_main.jsonl.gz")
    else:
        for record in _jsonl(prediction_path):
            prediction_count += 1
            prediction_runs.add(str(record.get("run_id")))
            prediction_keys.add((str(record.get("dataset")), str(record.get("utterance_id"))))
            missing_prediction_fields.update(PREDICTION_FIELDS - record.keys())
    if prediction_count != 1_300_050:
        issues.append(f"expected 1,300,050 main predictions; found {prediction_count}")
    if prediction_runs != run_ids:
        issues.append("main prediction run IDs do not match run_manifest.jsonl")
    if missing_prediction_fields:
        issues.append(f"main predictions lack fields: {sorted(missing_prediction_fields)}")

    metadata_path = root / "utterance_metadata.jsonl.gz"
    metadata_keys: set[tuple[str, str]] = set()
    missing_metadata_fields: set[str] = set()
    if not metadata_path.is_file():
        issues.append("missing utterance_metadata.jsonl.gz")
    else:
        for record in _jsonl(metadata_path):
            missing_metadata_fields.update(UTTERANCE_METADATA_FIELDS - record.keys())
            key = (str(record.get("dataset")), str(record.get("utterance_id")))
            if key in metadata_keys:
                issues.append(f"duplicate utterance metadata key: {key}")
                break
            metadata_keys.add(key)
        if missing_metadata_fields:
            issues.append(f"utterance metadata lacks fields: {sorted(missing_metadata_fields)}")
        if prediction_keys and metadata_keys != prediction_keys:
            issues.append(
                "utterance metadata keys do not exactly cover the main prediction utterances"
            )

    additional_path = root / "additional_predictions.jsonl.gz"
    additional_types: set[str] = set()
    predicted_extension_models: set[str] = set()
    missing_additional_fields: dict[str, set[str]] = {
        name: set() for name in EXPECTED_ADDITIONAL_TYPES
    }
    if not additional_path.is_file():
        issues.append("missing additional_predictions.jsonl.gz")
    else:
        for record in _jsonl(additional_path):
            experiment_type = str(record.get("experiment_type"))
            if experiment_type not in EXPECTED_ADDITIONAL_TYPES:
                issues.append(f"unknown additional experiment type: {experiment_type}")
                break
            additional_types.add(experiment_type)
            required = PREDICTION_FIELDS | {"experiment_type"} | ADDITIONAL_FIELDS[experiment_type]
            missing_additional_fields[experiment_type].update(required - record.keys())
            if experiment_type == "extension":
                predicted_extension_models.add(str(record.get("model")))
        missing_types = EXPECTED_ADDITIONAL_TYPES - additional_types
        if missing_types:
            issues.append(f"additional predictions lack experiment types: {sorted(missing_types)}")
        for experiment_type, missing in missing_additional_fields.items():
            if missing:
                issues.append(
                    f"{experiment_type} predictions lack fields: {sorted(missing)}"
                )
        missing_models = EXPECTED_PREDICTED_EXTENSION_MODELS - predicted_extension_models
        if missing_models:
            issues.append(
                f"additional predictions lack interval-bearing extensions: {sorted(missing_models)}"
            )

    assignment = next(iter(_jsonl(root / "assignment_audit.jsonl")))
    missing = {"adjudicated", "stratum", "stratum_population"} - assignment.keys()
    if missing:
        issues.append(f"assignment audit lacks fields: {sorted(missing)}")
    calibration = next(iter(_jsonl(root / "calibration_audit.jsonl")))
    missing = {"adjudicated", "raw_score", "target_correct"} - calibration.keys()
    if missing:
        issues.append(f"calibration audit lacks fields: {sorted(missing)}")

    parameters = json.loads((root / "calibration_parameters.json").read_text(encoding="utf-8"))
    for dataset, folds in (
        ("IEMOCAP", parameters["IEMOCAP"]["folds"]),
        ("MELD", parameters["MELD"]["folds"]),
    ):
        for fold, mapping in folds.items():
            missing = {"score_knots", "confidence_knots", "tau_q"} - mapping.keys()
            if missing:
                issues.append(f"{dataset}/{fold} calibration lacks fields: {sorted(missing)}")

    payload = {
        "complete": not issues,
        "runs": len(run_records),
        "main_predictions": prediction_count,
        "utterance_metadata": len(metadata_keys),
        "additional_experiment_types": sorted(additional_types),
        "issues": issues,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
