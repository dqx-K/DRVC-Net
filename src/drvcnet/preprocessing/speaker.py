"""Training-only monotone calibration for target-speaker assignment scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from drvcnet.io import atomic_write_json, atomic_write_jsonl, read_jsonl


@dataclass(frozen=True)
class MonotonePiecewiseCalibration:
    score_knots: tuple[float, ...]
    confidence_knots: tuple[float, ...]

    def transform(self, value: float) -> float:
        return float(
            np.interp(
                float(value),
                np.asarray(self.score_knots),
                np.asarray(self.confidence_knots),
                left=self.confidence_knots[0],
                right=self.confidence_knots[-1],
            )
        )


def fit_monotone_calibration(
    audit_records: list[dict[str, Any]],
) -> MonotonePiecewiseCalibration:
    scores = np.asarray([float(record["raw_score"]) for record in audit_records])
    targets = np.asarray([int(record["target_correct"]) for record in audit_records])
    if scores.shape[0] < 2 or np.unique(targets).size != 2:
        raise ValueError("Calibration audit requires positive and negative assignment examples")
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(scores, targets)
    return MonotonePiecewiseCalibration(
        tuple(float(value) for value in model.X_thresholds_),
        tuple(float(value) for value in model.y_thresholds_),
    )


def calibrate_speaker_scores(
    audit_path: str | Path,
    scores_path: str | Path,
    output_path: str | Path,
    calibration_output: str | Path,
) -> list[dict[str, Any]]:
    """Freeze a training-audit mapping and apply it to every assignment score."""

    audit = list(read_jsonl(audit_path))
    calibration = fit_monotone_calibration(audit)
    output: list[dict[str, Any]] = []
    for record in read_jsonl(scores_path):
        value = dict(record)
        value["speaker_confidence"] = calibration.transform(float(record["raw_score"]))
        output.append(value)
    atomic_write_jsonl(output_path, output)
    atomic_write_json(
        calibration_output,
        {
            "method": "isotonic_piecewise_linear",
            **asdict(calibration),
            "audit_records": len(audit),
        },
    )
    return output
