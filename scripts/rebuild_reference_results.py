#!/usr/bin/env python3
"""Rebuild legacy aggregate metrics while migrating old prediction parts.

The final public recomputation entry point is
``Supplementary_m/reproduce_results.py`` with ``--metadata``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LABELS = {
    "IEMOCAP": ["angry", "happy", "excited", "sad", "frustrated", "neutral"],
    "MELD": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
}
FINAL_SEEDS = [42, 123, 2026]
STABILITY_SEEDS = [42, 123, 2026, 314, 2025]
STABILITY_MODELS = {"B2", "C1", "C2", "C3", "B6"}

COMPARISONS = [("B6", model) for model in ("B2", "C1", "C2", "C3", "CSS")]
REPEATS = 10_000
STATISTICAL_SEED = 20_260_926


def _joined_predictions(root: Path, destination: Path) -> None:
    parts = sorted(root.glob("predictions_part_*"))
    if not parts:
        raise FileNotFoundError("predictions_part_*")
    with destination.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output)


def _metrics(matrix: list[list[int]]) -> dict[str, float]:
    supports = [sum(row) for row in matrix]
    predicted = [sum(row[index] for row in matrix) for index in range(len(matrix))]
    total = sum(supports)
    f1_values: list[float] = []
    for index in range(len(matrix)):
        true_positive = matrix[index][index]
        denominator = supports[index] + predicted[index]
        f1_values.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    weighted_f1 = sum(value * support for value, support in zip(f1_values, supports)) / total
    return {
        "WF1": weighted_f1 * 100.0,
        "MF1": statistics.mean(f1_values) * 100.0,
        "Acc": sum(matrix[index][index] for index in range(len(matrix))) / total * 100.0,
    }


def _weighted_f1(matrices: np.ndarray) -> np.ndarray:
    supports = matrices.sum(axis=-1)
    predicted = matrices.sum(axis=-2)
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1)
    denominator = supports + predicted
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator != 0,
    )
    totals = supports.sum(axis=-1)
    return np.divide(
        (class_f1 * supports).sum(axis=-1),
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals != 0,
    ) * 100.0


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(values))
    adjusted = np.zeros(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[int(index)])
        adjusted[int(index)] = min(running, 1.0)
    return [float(value) for value in adjusted]


def _paired_test(
    dataset: str,
    first_model: str,
    second_model: str,
    run_index: dict[tuple[str, str, int, str], str],
    dialogue_matrices: dict[str, dict[str, list[list[int]]]],
) -> dict[str, Any]:
    folds = sorted(
        {
            fold
            for ds, model, seed, fold in run_index
            if ds == dataset and model == first_model and seed in FINAL_SEEDS
        }
    )
    fold_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        run_pairs = [
            (
                run_index[(dataset, first_model, seed, fold)],
                run_index[(dataset, second_model, seed, fold)],
            )
            for seed in FINAL_SEEDS
        ]
        common_dialogues = set(dialogue_matrices[run_pairs[0][0]])
        for first_run, second_run in run_pairs:
            common_dialogues &= dialogue_matrices[first_run].keys()
            common_dialogues &= dialogue_matrices[second_run].keys()
        dialogue_ids = sorted(common_dialogues)
        if not dialogue_ids:
            raise ValueError(f"No paired dialogues for {dataset}/{fold}/{second_model}")
        first = np.asarray(
            [
                [dialogue_matrices[first_run][dialogue] for dialogue in dialogue_ids]
                for first_run, _ in run_pairs
            ],
            dtype=np.int64,
        )
        second = np.asarray(
            [
                [dialogue_matrices[second_run][dialogue] for dialogue in dialogue_ids]
                for _, second_run in run_pairs
            ],
            dtype=np.int64,
        )
        fold_arrays.append((first, second))

    bootstrap_rng = np.random.default_rng(STATISTICAL_SEED)
    bootstrap_fold_differences: list[np.ndarray] = []
    observed_fold_differences: list[np.ndarray] = []
    for first, second in fold_arrays:
        dialogue_count = first.shape[1]
        counts = bootstrap_rng.multinomial(
            dialogue_count,
            np.full(dialogue_count, 1.0 / dialogue_count),
            size=REPEATS,
        )
        sampled_first = np.einsum("bd,sdij->bsij", counts, first, optimize=True)
        sampled_second = np.einsum("bd,sdij->bsij", counts, second, optimize=True)
        bootstrap_fold_differences.append(
            _weighted_f1(sampled_first) - _weighted_f1(sampled_second)
        )
        observed_fold_differences.append(
            _weighted_f1(first.sum(axis=1)) - _weighted_f1(second.sum(axis=1))
        )
    bootstrap_differences = np.mean(np.stack(bootstrap_fold_differences), axis=(0, 2))
    observed = float(np.mean(np.stack(observed_fold_differences)))
    lower, upper = np.percentile(bootstrap_differences, [2.5, 97.5])

    permutation_rng = np.random.default_rng(STATISTICAL_SEED)
    permutation_fold_differences: list[np.ndarray] = []
    for first, second in fold_arrays:
        dialogue_count = first.shape[1]
        swaps = permutation_rng.integers(0, 2, size=(REPEATS, dialogue_count))
        change = second - first
        swapped_change = np.einsum("bd,sdij->bsij", swaps, change, optimize=True)
        permuted_first = first.sum(axis=1)[None, ...] + swapped_change
        permuted_second = second.sum(axis=1)[None, ...] - swapped_change
        permutation_fold_differences.append(
            _weighted_f1(permuted_first) - _weighted_f1(permuted_second)
        )
    permutation_differences = np.mean(np.stack(permutation_fold_differences), axis=(0, 2))
    p_value = (int(np.sum(np.abs(permutation_differences) >= abs(observed))) + 1) / (
        REPEATS + 1
    )
    return {
        "dataset": dataset,
        "comparison": f"{first_model}-{second_model}",
        "delta_WF1": round(observed, 2),
        "CI_95": [round(float(lower), 2), round(float(upper), 2)],
        "raw_p": round(p_value, 4),
        "n_boot": REPEATS,
        "n_perm": REPEATS,
        "statistical_seed": STATISTICAL_SEED,
        "seeds_used": FINAL_SEEDS,
        "iemocap_equal_session_weight": dataset == "IEMOCAP",
        "holm_family": "supplementary" if second_model == "CSS" else "main",
    }


def _summary(
    per_run: dict[str, dict[str, float]],
    runs: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, float]:
    seed_scores: list[dict[str, float]] = []
    for seed in seeds:
        fold_scores = [
            per_run[str(record["run_id"])]
            for record in runs
            if int(record["seed"]) == seed
        ]
        if not fold_scores:
            continue
        seed_scores.append(
            {
                metric: statistics.mean(score[metric] for score in fold_scores)
                for metric in ("WF1", "MF1", "Acc")
            }
        )
    if len(seed_scores) != len(seeds):
        raise ValueError(f"Expected seeds {seeds}; found {len(seed_scores)}")
    result: dict[str, float] = {}
    for metric in ("WF1", "MF1", "Acc"):
        values = [score[metric] for score in seed_scores]
        result[metric] = round(statistics.mean(values), 2)
        result[f"SD_{metric}"] = round(statistics.stdev(values), 2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_path = args.output or root / "recomputed_results.json"
    run_records = [
        json.loads(line)
        for line in (root / "run_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_lookup = {str(record["run_id"]): record for record in run_records}
    matrices: dict[str, list[list[int]]] = {}
    dialogue_matrices: dict[str, dict[str, list[list[int]]]] = defaultdict(dict)
    with tempfile.TemporaryDirectory() as temporary_directory:
        joined = Path(temporary_directory) / "predictions.gz"
        _joined_predictions(root, joined)
        with gzip.open(joined, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                run_id = str(record["run_id"])
                run = run_lookup[run_id]
                labels = LABELS[str(run["dataset"])]
                label_index = {label: index for index, label in enumerate(labels)}
                matrix = matrices.setdefault(
                    run_id, [[0 for _ in labels] for _ in labels]
                )
                true_index = label_index[record["y_true"]]
                predicted_index = label_index[record["y_pred"]]
                matrix[true_index][predicted_index] += 1
                dialogue = str(record["dialogue_id"])
                dialogue_matrix = dialogue_matrices[run_id].setdefault(
                    dialogue, [[0 for _ in labels] for _ in labels]
                )
                dialogue_matrix[true_index][predicted_index] += 1
    if matrices.keys() != run_lookup.keys():
        raise ValueError("Prediction and run-record identifiers differ")
    per_run = {run_id: _metrics(matrix) for run_id, matrix in matrices.items()}

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        grouped[(str(record["dataset"]), str(record["model"]))].append(record)
    results: dict[str, Any] = {"IEMOCAP": {}, "MELD": {}}
    for (dataset, model), runs in sorted(grouped.items()):
        s3 = _summary(per_run, runs, FINAL_SEEDS)
        model_result = {
            "WF1_3seed": s3["WF1"],
            "SD_WF1_3seed": s3["SD_WF1"],
            "MF1_3seed": s3["MF1"],
            "SD_MF1_3seed": s3["SD_MF1"],
            "Acc_3seed": s3["Acc"],
            "SD_Acc_3seed": s3["SD_Acc"],
        }
        if model in STABILITY_MODELS:
            s5 = _summary(per_run, runs, STABILITY_SEEDS)
            model_result.update(
                {
                    "WF1_5seed": s5["WF1"],
                    "SD_WF1_5seed": s5["SD_WF1"],
                    "MF1_5seed": s5["MF1"],
                    "SD_MF1_5seed": s5["SD_MF1"],
                    "Acc_5seed": s5["Acc"],
                    "SD_Acc_5seed": s5["SD_Acc"],
                }
            )
        results[dataset][model] = model_result

    run_index = {
        (
            str(record["dataset"]),
            str(record["model"]),
            int(record["seed"]),
            str(record.get("fold")),
        ): str(record["run_id"])
        for record in run_records
    }
    tests = [
        _paired_test(dataset, first, second, run_index, dialogue_matrices)
        for dataset in ("IEMOCAP", "MELD")
        for first, second in COMPARISONS
    ]
    for family in ("main", "supplementary"):
        selected = [test for test in tests if test["holm_family"] == family]
        adjusted = _holm([float(test["raw_p"]) for test in selected])
        for test, value in zip(selected, adjusted):
            test["holm_p"] = round(value, 4)
    results["statistical_tests"] = tests
    results["protocol_notes"] = [
        "Per-fold metrics are computed first and IEMOCAP sessions receive equal weight.",
        "Three-seed results use seeds 42, 123, and 2026; five-seed results are limited to B2, C1, C2, C3, and B6.",
        "Bidirectional B2/C2/C3/B6 controls are reported only on MELD with the three common seeds.",
        "Bootstrap intervals and permutation tests use 10,000 shared dialogue draws or swaps with statistical seed 20260926.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=output_path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(json.dumps({"runs": len(run_records), "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
