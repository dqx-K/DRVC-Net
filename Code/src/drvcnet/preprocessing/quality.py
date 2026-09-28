"""Hard validity and reliability factors from the paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class SharpnessCalibration:
    """Frozen monotone mapping fitted on training audits only."""

    lower_quantile: float
    upper_quantile: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower_quantile) or not np.isfinite(self.upper_quantile):
            raise ValueError("Sharpness endpoints must be finite")
        if self.upper_quantile <= self.lower_quantile:
            raise ValueError("upper_quantile must be larger than lower_quantile")

    def map(self, sharpness: float) -> float:
        return _unit(
            (float(sharpness) - self.lower_quantile)
            / (self.upper_quantile - self.lower_quantile)
        )

    @classmethod
    def fit(
        cls,
        training_values: np.ndarray,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
    ) -> "SharpnessCalibration":
        values = np.asarray(training_values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 2:
            raise ValueError("At least two finite training sharpness values are required")
        lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
        return cls(float(lower), float(upper))


def pose_quality(yaw_degrees: float, pitch_degrees: float) -> float:
    yaw_term = (abs(float(yaw_degrees)) / 45.0) ** 2
    pitch_term = (abs(float(pitch_degrees)) / 30.0) ** 2
    return float(np.exp(-(yaw_term + pitch_term)))


def frame_quality(
    *,
    valid: bool,
    detection_confidence: float,
    speaker_confidence: float,
    visible_fraction: float,
    blur_quality: float,
    yaw_degrees: float,
    pitch_degrees: float,
) -> float:
    """Equation (2): a reliability score, not a calibrated probability."""

    if not valid:
        return 0.0
    product = (
        _unit(detection_confidence)
        * _unit(speaker_confidence)
        * _unit(visible_fraction)
        * _unit(blur_quality)
        * pose_quality(yaw_degrees, pitch_degrees)
    )
    return _unit(product)


def laplacian_sharpness(image_bgr: np.ndarray) -> float:
    import cv2

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
