"""Completeness checks for the public prediction and run-record package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from drvcnet.io import atomic_write_json, read_jsonl


PREDICTION_FIELDS = {
    "run_id",
    "dataset",
    "fold",
    "seed",
    "model",
    "dialogue_id",
    "utterance_id",
    "speaker_id",
    "order",
    "y_true",
    "y_pred",
    "probabilities",
    "shift_group",
    "current_mask",
    "recent_mask",
    "historical_mask",
    "recent_age",
    "historical_age",
    "historical_count",
    "reference_ids",
    "reference_availability",
    "both",
    "history_bin",
    "reference_age",
    "reference_age_bin",
    "visual_missing",
}

RUN_RECORD_FILES = (
    "provenance.json",
    "history.json",
    "run_summary.json",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _prediction_issues(path: Path) -> list[str]:
    if not path.is_file():
        return [f"missing prediction file: {path}"]
    records = list(read_jsonl(path))
    if not records:
        return [f"empty prediction file: {path}"]
    issues: list[str] = []
    for line_number, record in enumerate(records, start=1):
        missing = PREDICTION_FIELDS - record.keys()
        if missing:
            issues.append(f"{path}:{line_number}: missing fields {sorted(missing)}")
    identifiers = [str(record.get("utterance_id")) for record in records]
    if len(identifiers) != len(set(identifiers)):
        issues.append(f"{path}: duplicate utterance IDs")
    return issues


def validate_release_package(
    config_root: str | Path,
    output_path: str | Path,
    *,
    intervention_files: Iterable[str | Path] = (),
    audit_files: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Check expected files and prediction schemas without executing any model."""

    root = Path(config_root)
    intervention_paths = [Path(value) for value in intervention_files]
    audit_paths = [Path(value) for value in audit_files]
    issues: list[str] = []
    final_configs = [path for path in root.rglob("*.yaml") if "search" not in path.parts]
    search_configs = [path for path in root.rglob("*.yaml") if "search" in path.parts]
    for config_path in final_configs:
        config = _read_yaml(config_path)
        output_dir = Path(config["output_dir"])
        prediction_path = Path(config["evaluation"]["prediction_file"])
        issues.extend(_prediction_issues(prediction_path))
        for name in RUN_RECORD_FILES:
            path = output_dir / name
            if not path.is_file():
                issues.append(f"missing run record: {path}")
        evaluation_record = prediction_path.with_suffix(".metrics.json")
        if not evaluation_record.is_file():
            issues.append(f"missing evaluation record: {evaluation_record}")
    for config_path in search_configs:
        config = _read_yaml(config_path)
        path = Path(config["output_dir"]) / "search" / "search_summary.json"
        if not path.is_file():
            issues.append(f"missing search record: {path}")
    for path in intervention_paths:
        issues.extend(_prediction_issues(path))
    for path in audit_paths:
        if not path.is_file():
            issues.append(f"missing audit material: {path}")

    payload = {
        "complete": not issues,
        "final_configurations": len(final_configs),
        "search_configurations": len(search_configs),
        "intervention_files": len(intervention_paths),
        "audit_files": len(audit_paths),
        "issues": issues,
    }
    atomic_write_json(output_path, payload)
    return payload
