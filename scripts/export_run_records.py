#!/usr/bin/env python3
"""Build the compact provenance-complete public run manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _evaluation_record(run_directory: Path) -> tuple[Path, dict[str, Any]]:
    matches = sorted(run_directory.glob("*.metrics.json"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one evaluation record in {run_directory}; found {len(matches)}"
        )
    return matches[0], _read_object(matches[0])


def _model_label(config: dict[str, Any]) -> str:
    if "model_label" in config:
        return str(config["model_label"])
    model = config["model"]
    architecture = str(model["architecture"])
    if architecture != "drvc":
        return architecture.upper()
    label = str(model["variant"])
    if model.get("context_direction") == "bidirectional":
        label += "_bidir"
    return label


def _merged_record(run_directory: Path) -> dict[str, Any]:
    provenance_path = run_directory / "provenance.json"
    summary_path = run_directory / "run_summary.json"
    history_path = run_directory / "history.json"
    for path in (provenance_path, summary_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = _read_object(provenance_path)
    summary = _read_object(summary_path)
    evaluation_path, evaluation = _evaluation_record(run_directory)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"Expected a non-empty history list: {history_path}")
    config = provenance["resolved_config"]
    run_id = str(provenance.get("experiment", summary.get("experiment")))
    if run_id in {"", "None"}:
        raise ValueError(f"Missing experiment identifier: {run_directory}")
    return {
        "run_id": run_id,
        "dataset": config["dataset"]["name"],
        "model": _model_label(config),
        "fold": config["dataset"].get("outer_fold"),
        "seed": config["train"]["seed"],
        "git": provenance["git"],
        "software": provenance["software"],
        "hardware": provenance["hardware"],
        "data_and_config": {
            "identifiers": provenance.get("paths", {}),
            "recorded_hashes": provenance.get("hashes", {}),
            "resolved_config": config,
        },
        "checkpoint": {
            "identifier": summary["checkpoint"],
            "recorded_sha256": summary.get("checkpoint_sha256"),
        },
        "prediction_file": {
            "identifier": evaluation["prediction_file"],
            "recorded_sha256": evaluation.get("prediction_sha256"),
        },
        "best_epoch": summary["best_epoch"],
        "trainable_parameters": summary["trainable_parameters"],
        "training_history": {
            "identifier": str(history_path),
            "epochs": len(history),
            "last_epoch": int(history[-1]["epoch"]),
        },
        "evaluation": evaluation,
        "source_records": {
            "provenance": str(provenance_path),
            "run_summary": str(summary_path),
            "history": str(history_path),
            "evaluation": str(evaluation_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(
        path.parent for path in args.run_root.rglob("provenance.json") if path.is_file()
    )
    if not directories:
        raise FileNotFoundError(f"No provenance.json files under {args.run_root}")
    records = [_merged_record(directory) for directory in directories]
    run_ids = [str(record["run_id"]) for record in records]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run_id values in collected records")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=args.output.parent, suffix=".jsonl")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary_name, args.output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(json.dumps({"runs": len(records), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
