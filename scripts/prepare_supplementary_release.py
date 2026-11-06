#!/usr/bin/env python3
"""Normalize and scope legacy aggregate records to the paper protocol.

This utility only reorganizes existing records. It does not infer missing
predictions, provenance, annotations, or model outputs. The final public bundle
still requires the provenance-complete run manifest, normalized main prediction
stream, fixed utterance metadata, and compact additional prediction stream
defined in ``Supplementary_m/README.md``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any


FINAL_SEEDS = {42, 123, 2026}
STABILITY_SEEDS = FINAL_SEEDS | {314, 2025}
STABILITY_MODELS = {"B2", "C1", "C2", "C3", "B6"}
BIDIRECTIONAL_MODELS = {f"{model}_bidir" for model in ("B2", "C2", "C3", "B6")}
PART_BYTES = 28 * 1024 * 1024


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".jsonl")
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                count += 1
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return count


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _run_is_reported(record: dict[str, Any]) -> bool:
    dataset = str(record["dataset"])
    model = str(record["model"])
    seed = int(record["seed"])
    if model.endswith("_bidir"):
        return dataset == "MELD" and model in BIDIRECTIONAL_MODELS and seed in FINAL_SEEDS
    expected = STABILITY_SEEDS if model in STABILITY_MODELS else FINAL_SEEDS
    return seed in expected


def _canonical_assignment_audit(record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value["adjudicated"] = value.pop("final")
    value["stratum"] = str(value.get("session", value.get("split"))).lower()
    value["stratum_population"] = int(value.pop("stratum_size"))
    return value


def _canonical_calibration_audit(record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value["raw_score"] = float(value.pop("raw_confidence"))
    value["target_correct"] = bool(value.pop("human_correct"))
    value["adjudicated"] = value.pop("final")
    value["stratum"] = str(value.get("fold", value.get("split", "train"))).lower()
    return value


def _canonical_mapping(value: dict[str, Any]) -> dict[str, Any]:
    mapping = value["mapping"]
    return {
        "tau_q": float(value["tau_q"]),
        "n_samples": int(value["n_samples"]),
        "score_knots": [float(item) for item in mapping["x_nodes"]],
        "confidence_knots": [float(item) for item in mapping["y_nodes"]],
    }


def _normalize_calibration_parameters(path: Path) -> None:
    source = json.loads(path.read_text(encoding="utf-8"))
    iemocap = {
        str(fold): _canonical_mapping(value)
        for fold, value in source["IEMOCAP"]["folds"].items()
    }
    meld = _canonical_mapping(source["MELD"])
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "method": "monotone_piecewise_linear",
            "IEMOCAP": {"folds": iemocap},
            "MELD": {"folds": {"official": meld}},
        },
    )


def _filter_jsonl(
    path: Path, predicate: Callable[[dict[str, Any]], bool]
) -> int:
    return _atomic_jsonl(path, (record for record in _read_jsonl(path) if predicate(record)))


def _filter_search(record: dict[str, Any]) -> bool:
    if str(record.get("dataset")) == "IEMOCAP" and str(record.get("model", "")).endswith(
        "_bidir"
    ):
        return False
    return not (record.get("search_type") == "structure" and record.get("fold") is None)


def _join_prediction_parts(root: Path, destination: Path) -> None:
    parts = sorted(root.glob("predictions_part_*"))
    if not parts:
        raise FileNotFoundError("No predictions_part_* files found")
    with destination.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output)


def _filter_predictions(root: Path, allowed_run_ids: set[str]) -> int:
    with tempfile.TemporaryDirectory(dir=root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        joined = temporary_root / "predictions_full.gz"
        filtered = temporary_root / "predictions_filtered.gz"
        _join_prediction_parts(root, joined)
        count = 0
        with gzip.open(joined, "rt", encoding="utf-8") as source:
            with filtered.open("wb") as raw_output:
                with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output:
                        for line_number, line in enumerate(source, start=1):
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            if str(record.get("run_id")) not in allowed_run_ids:
                                continue
                            output.write(
                                json.dumps(
                                    record,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                            )
                            output.write("\n")
                            count += 1

        for old_part in root.glob("predictions_part_*"):
            old_part.unlink()
        with filtered.open("rb") as source:
            part_index = 0
            while True:
                chunk = source.read(PART_BYTES)
                if not chunk:
                    break
                suffix = chr(ord("a") + part_index // 26) + chr(ord("a") + part_index % 26)
                (root / f"predictions_part_{suffix}").write_bytes(chunk)
                part_index += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Supplementary_m directory")
    args = parser.parse_args()
    root = args.root.resolve()

    run_path = root / "run_records.jsonl"
    reported_runs = [record for record in _read_jsonl(run_path) if _run_is_reported(record)]
    allowed_run_ids = {str(record["run_id"]) for record in reported_runs}
    if len(allowed_run_ids) != len(reported_runs):
        raise ValueError("Filtered run records contain duplicate run_id values")
    _atomic_jsonl(run_path, reported_runs)

    counts = {
        "runs": len(reported_runs),
        "resolved_configs": _filter_jsonl(
            root / "resolved_configs.jsonl",
            lambda record: str(record["run_id"]) in allowed_run_ids,
        ),
        "history_rows": _filter_jsonl(
            root / "training_history.jsonl",
            lambda record: str(record["run_id"]) in allowed_run_ids,
        ),
        "search_rows": _filter_jsonl(root / "development_search.jsonl", _filter_search),
    }
    counts["predictions"] = _filter_predictions(root, allowed_run_ids)
    counts["assignment_audit"] = _atomic_jsonl(
        root / "assignment_audit.jsonl",
        (_canonical_assignment_audit(record) for record in _read_jsonl(root / "assignment_audit.jsonl")),
    )
    counts["calibration_audit"] = _atomic_jsonl(
        root / "calibration_audit.jsonl",
        (
            _canonical_calibration_audit(record)
            for record in _read_jsonl(root / "calibration_audit.jsonl")
        ),
    )
    _normalize_calibration_parameters(root / "calibration_parameters.json")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
