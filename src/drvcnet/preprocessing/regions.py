"""Eye-aligned global and FAN-indexed local face crops."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from drvcnet.constants import REGIONS

LANDMARK_GROUPS: dict[str, np.ndarray] = {
    "brow": np.arange(17, 27),
    "eye": np.arange(36, 48),
    "mouth": np.arange(48, 68),
}


@dataclass(frozen=True)
class AlignedRegions:
    crops: dict[str, np.ndarray]
    valid: dict[str, bool]
    aligned_landmarks: np.ndarray
    transform: np.ndarray


def _similarity_transform(landmarks: np.ndarray, output_size: int) -> np.ndarray:
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    eye_vector = right_eye - left_eye
    distance = float(np.linalg.norm(eye_vector))
    if not np.isfinite(distance) or distance < 1.0:
        raise ValueError("Degenerate eye landmarks")
    source_third = left_eye + np.array([-eye_vector[1], eye_vector[0]], dtype=np.float32)
    destination_left = np.array([0.35 * output_size, 0.38 * output_size], dtype=np.float32)
    destination_right = np.array([0.65 * output_size, 0.38 * output_size], dtype=np.float32)
    destination_vector = destination_right - destination_left
    destination_third = destination_left + np.array(
        [-destination_vector[1], destination_vector[0]], dtype=np.float32
    )
    source = np.stack([left_eye, right_eye, source_third]).astype(np.float32)
    destination = np.stack([destination_left, destination_right, destination_third])
    return cv2.getAffineTransform(source, destination)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
    )
    return homogeneous @ transform.T


def _local_crop(
    aligned: np.ndarray,
    landmarks: np.ndarray,
    indices: np.ndarray,
    inter_pupil: float,
    output_size: int,
) -> np.ndarray | None:
    points = landmarks[indices]
    if not np.isfinite(points).all():
        return None
    margin = 0.15 * inter_pupil
    left = int(np.floor(points[:, 0].min() - margin))
    top = int(np.floor(points[:, 1].min() - margin))
    right = int(np.ceil(points[:, 0].max() + margin))
    bottom = int(np.ceil(points[:, 1].max() + margin))
    height, width = aligned.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right - left < 2 or bottom - top < 2:
        return None
    crop = aligned[top:bottom, left:right]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def align_and_crop_regions(
    image_bgr: np.ndarray,
    landmarks68: np.ndarray,
    output_size: int = 224,
) -> AlignedRegions:
    """Apply a shared eye similarity transform, then create all four regions."""

    landmarks = np.asarray(landmarks68, dtype=np.float32)
    if landmarks.shape != (68, 2):
        raise ValueError(f"Expected landmarks shape (68, 2), got {landmarks.shape}")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")
    transform = _similarity_transform(landmarks, output_size)
    aligned = cv2.warpAffine(
        image_bgr,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    aligned_landmarks = _transform_points(landmarks, transform)
    left_eye = aligned_landmarks[36:42].mean(axis=0)
    right_eye = aligned_landmarks[42:48].mean(axis=0)
    inter_pupil = float(np.linalg.norm(right_eye - left_eye))

    crops: dict[str, np.ndarray] = {"global": aligned}
    valid: dict[str, bool] = {"global": bool(np.isfinite(aligned_landmarks).all())}
    for region, indices in LANDMARK_GROUPS.items():
        crop = _local_crop(aligned, aligned_landmarks, indices, inter_pupil, output_size)
        valid[region] = crop is not None
        crops[region] = np.zeros_like(aligned) if crop is None else crop
    if tuple(crops) != REGIONS:
        raise AssertionError("Region order changed unexpectedly")
    return AlignedRegions(crops, valid, aligned_landmarks, transform)


def uniform_distinct_indices(frame_count: int, maximum: int = 8) -> list[int]:
    """Choose at most `maximum` unique integer positions over a clip."""

    if frame_count <= 0 or maximum <= 0:
        return []
    count = min(frame_count, maximum)
    values = np.linspace(0, frame_count - 1, num=count)
    indices = np.unique(np.rint(values).astype(np.int64))
    if indices.size != count:
        # This is rare for integer endpoints, but fill deterministically if needed.
        chosen = set(int(item) for item in indices)
        for candidate in range(frame_count):
            if len(chosen) == count:
                break
            chosen.add(candidate)
        indices = np.asarray(sorted(chosen), dtype=np.int64)
    return [int(item) for item in indices]
