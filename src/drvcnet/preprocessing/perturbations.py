"""Pixel-space blur and regional occlusion cache preparation for stress tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drvcnet.constants import REGIONS
from drvcnet.io import atomic_write_jsonl, read_jsonl, resolve_record_path, safe_identifier
from drvcnet.preprocessing.quality import SharpnessCalibration, laplacian_sharpness


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.copy()
    radius = max(1, math.ceil(3.0 * sigma))
    kernel = 2 * radius + 1
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


def _center_occlusion(image: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Occlusion fraction must be in [0, 1]")
    result = image.copy()
    if fraction == 0.0:
        return result
    height, width = result.shape[:2]
    scale = math.sqrt(fraction)
    occlusion_width = min(width, max(1, round(width * scale)))
    occlusion_height = min(height, max(1, round(height * scale)))
    left = (width - occlusion_width) // 2
    top = (height - occlusion_height) // 2
    result[top : top + occlusion_height, left : left + occlusion_width] = 0
    return result


def _recompute_quality(
    value: dict[str, Any], image: np.ndarray, calibration: SharpnessCalibration, visibility: float
) -> float:
    factors = value.get("quality_factors")
    if not isinstance(factors, dict):
        raise ValueError("Observation lacks quality_factors; rebuild clean observations first")
    blur = calibration.map(laplacian_sharpness(image))
    # frame_quality receives yaw/pitch in the clean path. Here the stored pose
    # factor is multiplied directly so the frozen pose estimate remains exact.
    if not bool(value["valid"]):
        return 0.0
    product = (
        float(factors["detection_confidence"])
        * float(factors["speaker_confidence"])
        * float(visibility)
        * float(blur)
        * float(factors["pose_quality"])
    )
    return float(np.clip(product, 0.0, 1.0))


def build_corrupted_observations(
    observations_path: str | Path,
    output_dir: str | Path,
    *,
    kind: str,
    level: float,
    calibration: SharpnessCalibration,
) -> list[dict[str, Any]]:
    """Build a paired crop manifest; feature extraction then creates a separate cache."""

    if kind not in {"blur", "brow_eye_occlusion", "mouth_occlusion"}:
        raise ValueError(f"Unsupported pixel perturbation: {kind}")
    output = Path(output_dir).expanduser().resolve()
    crop_root = output / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for record in read_jsonl(observations_path):
        transformed = {
            "utterance_id": record["utterance_id"],
            "distinct_frame_count": record["distinct_frame_count"],
            "frames": [],
        }
        utterance_dir = crop_root / safe_identifier(str(record["utterance_id"]))
        utterance_dir.mkdir(parents=True, exist_ok=True)
        for frame in record["frames"]:
            target_frame = {key: value for key, value in frame.items() if key != "regions"}
            target_frame["regions"] = {}
            for region in REGIONS:
                source_value = dict(frame["regions"][region])
                source_path = resolve_record_path(observations_path, source_value["crop_path"])
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(source_path)
                affected = kind == "blur" or (
                    kind == "brow_eye_occlusion" and region in {"brow", "eye"}
                ) or (kind == "mouth_occlusion" and region == "mouth")
                if kind == "blur" and affected:
                    changed = _gaussian_blur(image, level)
                    visibility = float(source_value["visibility"])
                elif affected:
                    changed = _center_occlusion(image, level)
                    visibility = float(source_value["visibility"]) * (1.0 - level)
                else:
                    changed = image
                    visibility = float(source_value["visibility"])
                target_path = utterance_dir / f"f{int(frame['frame_index']):08d}_{region}.jpg"
                if not cv2.imwrite(str(target_path), changed):
                    raise OSError(target_path)
                source_value["crop_path"] = str(target_path)
                source_value["visibility"] = visibility
                factors = dict(source_value["quality_factors"])
                factors["visibility"] = visibility
                factors["blur_quality"] = calibration.map(laplacian_sharpness(changed))
                source_value["quality_factors"] = factors
                source_value["quality"] = _recompute_quality(
                    source_value, changed, calibration, visibility
                )
                target_frame["regions"][region] = source_value
            transformed["frames"].append(target_frame)
        records.append(transformed)
    atomic_write_jsonl(output / "observations.jsonl", records)
    return records
