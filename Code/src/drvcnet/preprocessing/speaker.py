"""Training-audit calibration for TalkNet or metadata-based assignment scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from drvcnet.io import atomic_write_json, atomic_write_jsonl, read_jsonl


@dataclass(frozen=True)
class PlattCalibration:
    slope: float
    intercept: float

    def transform(self, value: float) -> float:
        logit = self.slope * float(value) + self.intercept
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))


def fit_platt_calibration(audit_records: list[dict[str, Any]]) -> PlattCalibration:
    scores = np.asarray([float(record["raw_score"]) for record in audit_records]).reshape(-1, 1)
    targets = np.asarray([int(record["target_correct"]) for record in audit_records])
    if scores.shape[0] < 2 or np.unique(targets).size != 2:
        raise ValueError("Calibration audit requires positive and negative assignment examples")
    model = LogisticRegression(random_state=0, solver="lbfgs")
    model.fit(scores, targets)
    return PlattCalibration(float(model.coef_[0, 0]), float(model.intercept_[0]))


def calibrate_speaker_scores(
    audit_path: str | Path,
    scores_path: str | Path,
    output_path: str | Path,
    calibration_output: str | Path,
) -> list[dict[str, Any]]:
    """Freeze a training-audit mapping and apply it to every assignment score."""

    audit = list(read_jsonl(audit_path))
    calibration = fit_platt_calibration(audit)
    output: list[dict[str, Any]] = []
    for record in read_jsonl(scores_path):
        value = dict(record)
        value["speaker_confidence"] = calibration.transform(float(record["raw_score"]))
        output.append(value)
    atomic_write_jsonl(output_path, output)
    atomic_write_json(
        calibration_output,
        {
            "method": "platt_logistic",
            **asdict(calibration),
            "audit_records": len(audit),
        },
    )
    return output
