"""Fixed-label-set emotion classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_count: int,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays with equal shape")
    labels = np.arange(class_count)
    return {
        "weighted_f1": float(
            f1_score(truth, prediction, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(truth, prediction)),
        "count": int(truth.size),
    }


def metrics_from_prediction_records(
    records: Sequence[dict[str, Any]], labels: Sequence[str]
) -> dict[str, float]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    return classification_metrics(
        [label_to_index[str(record["y_true"])] for record in records],
        [label_to_index[str(record["y_pred"])] for record in records],
        len(labels),
    )
