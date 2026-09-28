"""Aggregate fold/seed predictions into the paper's fixed reporting protocol."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from drvcnet.evaluation.groups import select_group
from drvcnet.evaluation.metrics import metrics_from_prediction_records


def summarize_predictions(
    records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    iemocap_equal_fold_weight: bool,
) -> dict[str, Any]:
    """Average folds inside each seed, then report mean and sample SD across seeds."""

    by_seed_fold: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed_fold[(int(record["seed"]), str(record.get("fold", "all")))].append(record)
    seed_metrics: dict[int, dict[str, float]] = {}
    for seed in sorted({key[0] for key in by_seed_fold}):
        fold_values = [
            metrics_from_prediction_records(values, labels)
            for (candidate_seed, _fold), values in by_seed_fold.items()
            if candidate_seed == seed
        ]
        if not fold_values:
            continue
        if not iemocap_equal_fold_weight and len(fold_values) > 1:
            combined = [
                record
                for (candidate_seed, _fold), values in by_seed_fold.items()
                if candidate_seed == seed
                for record in values
            ]
            seed_metrics[seed] = metrics_from_prediction_records(combined, labels)
        else:
            seed_metrics[seed] = {
                key: float(np.mean([value[key] for value in fold_values]))
                for key in ("weighted_f1", "macro_f1", "accuracy")
            }
    result: dict[str, Any] = {"per_seed": seed_metrics}
    for metric in ("weighted_f1", "macro_f1", "accuracy"):
        values = np.asarray([value[metric] for value in seed_metrics.values()], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    result["seeds"] = sorted(seed_metrics)
    result["records"] = len(records)
    return result


def subgroup_report(
    records: Sequence[dict[str, Any]], labels: Sequence[str], groups: Sequence[str]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for group in groups:
        selected = select_group(records, group)
        report[group] = {
            "utterances": len({str(record["utterance_id"]) for record in selected}),
            "dialogues": len({str(record["dialogue_id"]) for record in selected}),
            "metrics": metrics_from_prediction_records(selected, labels) if selected else None,
        }
    return report


def factorial_reference_interaction(model_scores: dict[str, float]) -> float:
    required = {"B2", "B3", "B4", "B5"}
    missing = required - model_scores.keys()
    if missing:
        raise ValueError(f"Missing factorial models: {sorted(missing)}")
    return model_scores["B5"] - model_scores["B3"] - model_scores["B4"] + model_scores["B2"]
