"""Convert frozen frame assignments into aligned regional crop observations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drvcnet.constants import REGIONS
from drvcnet.io import atomic_write_jsonl, read_jsonl, resolve_record_path, safe_identifier
from drvcnet.preprocessing.quality import (
    SharpnessCalibration,
    frame_quality,
    laplacian_sharpness,
    pose_quality,
)
from drvcnet.preprocessing.regions import align_and_crop_regions
from drvcnet.records import FrameAssignment


def calibrate_sharpness(
    assignments_path: str | Path,
    training_ids: set[str],
    *,
    crop_size: int = 224,
    minimum_visibility: float = 0.5,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> SharpnessCalibration:
    """Fit the frozen regional sharpness map using training observations."""

    values: list[float] = []
    for raw in read_jsonl(assignments_path):
        assignment = FrameAssignment.from_dict(raw)
        if assignment.utterance_id not in training_ids or not assignment.assignment_valid:
            continue
        image_path = resolve_record_path(assignments_path, assignment.image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        aligned = align_and_crop_regions(image, assignment.landmarks68, output_size=crop_size)
        for region in REGIONS:
            if aligned.valid[region] and assignment.visibility[region] >= minimum_visibility:
                values.append(laplacian_sharpness(aligned.crops[region]))
    return SharpnessCalibration.fit(
        np.asarray(values, dtype=np.float64),
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )


def prepare_visual_observations(
    manifest_path: str | Path,
    assignments_path: str | Path,
    output_dir: str | Path,
    calibration: SharpnessCalibration,
    *,
    crop_size: int = 224,
    max_frames: int = 8,
    minimum_visibility: float = 0.5,
) -> list[dict[str, Any]]:
    """Validate the frozen assignment boundary and write deterministic crops."""

    utterance_ids = {str(value["utterance_id"]) for value in read_jsonl(manifest_path)}
    grouped: dict[str, list[FrameAssignment]] = defaultdict(list)
    for raw in read_jsonl(assignments_path):
        assignment = FrameAssignment.from_dict(raw)
        if assignment.utterance_id not in utterance_ids:
            raise ValueError(f"Assignment has unknown utterance: {assignment.utterance_id}")
        grouped[assignment.utterance_id].append(assignment)

    output = Path(output_dir).expanduser().resolve()
    crop_root = output / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for utterance_id in sorted(utterance_ids):
        assignments = sorted(
            grouped.get(utterance_id, []), key=lambda item: (item.frame_index, item.timestamp)
        )
        frame_indices = [item.frame_index for item in assignments]
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError(f"Repeated sampled frame index for {utterance_id}")
        if len(assignments) > max_frames:
            raise ValueError(f"{utterance_id} has {len(assignments)} frames, maximum is {max_frames}")
        utterance_dir = crop_root / safe_identifier(utterance_id)
        utterance_dir.mkdir(parents=True, exist_ok=True)
        frames: list[dict[str, Any]] = []

        for assignment in assignments:
            image_path = resolve_record_path(assignments_path, assignment.image_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot read assignment frame {image_path}")
            aligned = align_and_crop_regions(image, assignment.landmarks68, output_size=crop_size)
            region_values: dict[str, Any] = {}
            for region in REGIONS:
                regional_sharpness = laplacian_sharpness(aligned.crops[region])
                blur_quality = calibration.map(regional_sharpness)
                hard_valid = (
                    assignment.assignment_valid
                    and aligned.valid[region]
                    and assignment.visibility[region] >= minimum_visibility
                )
                quality = frame_quality(
                    valid=hard_valid,
                    detection_confidence=assignment.detection_confidence,
                    speaker_confidence=assignment.speaker_confidence,
                    visible_fraction=assignment.visibility[region],
                    blur_quality=blur_quality,
                    yaw_degrees=assignment.yaw,
                    pitch_degrees=assignment.pitch,
                )
                crop_path = utterance_dir / f"f{assignment.frame_index:08d}_{region}.jpg"
                if not cv2.imwrite(str(crop_path), aligned.crops[region]):
                    raise OSError(f"Failed to write {crop_path}")
                region_values[region] = {
                    "crop_path": str(crop_path),
                    "valid": hard_valid,
                    "quality": quality,
                    "visibility": assignment.visibility[region],
                    "sharpness": regional_sharpness,
                    "quality_factors": {
                        "detection_confidence": assignment.detection_confidence,
                        "speaker_confidence": assignment.speaker_confidence,
                        "visibility": assignment.visibility[region],
                        "blur_quality": blur_quality,
                        "pose_quality": pose_quality(assignment.yaw, assignment.pitch),
                    },
                }
            frames.append(
                {
                    "frame_index": assignment.frame_index,
                    "timestamp": assignment.timestamp,
                    "track_id": assignment.track_id,
                    "shot_id": assignment.shot_id,
                    "regions": region_values,
                }
            )
        records.append(
            {
                "utterance_id": utterance_id,
                "distinct_frame_count": len(frames),
                "frames": frames,
            }
        )
    atomic_write_jsonl(output / "observations.jsonl", records)
    (output / "preprocessing.json").write_text(
        json.dumps(
            {
                "crop_size": crop_size,
                "max_frames": max_frames,
                "minimum_visibility": minimum_visibility,
                "sharpness_calibration": {
                    "lower_quantile": calibration.lower_quantile,
                    "upper_quantile": calibration.upper_quantile,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return records
