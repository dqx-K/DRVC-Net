"""Frozen RetinaFace/FAN/ByteTrack candidate extraction.

This stage produces auditable face-track candidates. Dataset-specific speaker
assignment scores are merged afterwards, preserving a clean boundary between
face tracking and target-speaker selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drvcnet.io import atomic_write_jsonl, read_jsonl, safe_identifier
from drvcnet.preprocessing.quality import laplacian_sharpness
from drvcnet.preprocessing.regions import uniform_distinct_indices
from drvcnet.records import UtteranceRecord


@dataclass(frozen=True)
class FaceCandidate:
    bbox: np.ndarray
    confidence: float
    landmarks68: np.ndarray


class RetinaFaceFan:
    """Adapter over the standard `retina-face` and `face-alignment` packages."""

    def __init__(self, device: str = "cuda") -> None:
        from face_alignment import FaceAlignment, LandmarksType

        self.device = device
        self.aligner = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=device)

    def detect(self, image_bgr: np.ndarray) -> list[FaceCandidate]:
        from retinaface import RetinaFace

        detections = RetinaFace.detect_faces(image_bgr)
        if not isinstance(detections, dict):
            return []
        boxes: list[np.ndarray] = []
        scores: list[float] = []
        for value in detections.values():
            boxes.append(np.asarray(value["facial_area"], dtype=np.float32))
            scores.append(float(value["score"]))
        if not boxes:
            return []
        landmarks = self.aligner.get_landmarks_from_image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), detected_faces=boxes
        )
        if landmarks is None:
            return []
        return [
            FaceCandidate(box, score, np.asarray(points, dtype=np.float32))
            for box, score, points in zip(boxes, scores, landmarks, strict=True)
            if np.asarray(points).shape == (68, 2)
        ]


class ByteTrackAdapter:
    """Associate sampled RetinaFace detections without attaching identities."""

    def __init__(self, frame_rate: float = 30.0) -> None:
        import supervision as sv

        self.sv = sv
        self.tracker = sv.ByteTrack(frame_rate=frame_rate)

    def reset(self, frame_rate: float = 30.0) -> None:
        self.tracker = self.sv.ByteTrack(frame_rate=frame_rate)

    def update(self, candidates: list[FaceCandidate]) -> list[int | None]:
        if not candidates:
            return []
        detections = self.sv.Detections(
            xyxy=np.stack([item.bbox for item in candidates]),
            confidence=np.asarray([item.confidence for item in candidates], dtype=np.float32),
            class_id=np.zeros(len(candidates), dtype=np.int32),
        )
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            return [None] * len(candidates)
        tracked_boxes = np.asarray(tracked.xyxy, dtype=np.float32)
        tracked_ids = np.asarray(tracked.tracker_id)
        result: list[int | None] = []
        for candidate in candidates:
            left_top = np.maximum(candidate.bbox[:2], tracked_boxes[:, :2])
            right_bottom = np.minimum(candidate.bbox[2:], tracked_boxes[:, 2:])
            intersection_size = np.clip(right_bottom - left_top, 0.0, None)
            intersection = intersection_size[:, 0] * intersection_size[:, 1]
            candidate_area = np.prod(np.clip(candidate.bbox[2:] - candidate.bbox[:2], 0.0, None))
            tracked_area = np.prod(
                np.clip(tracked_boxes[:, 2:] - tracked_boxes[:, :2], 0.0, None), axis=1
            )
            iou = intersection / np.clip(candidate_area + tracked_area - intersection, 1.0e-6, None)
            match = int(np.argmax(iou))
            result.append(int(tracked_ids[match]) if iou[match] >= 0.3 else None)
        return result


def _pose_from_landmarks(landmarks: np.ndarray, image_shape: tuple[int, ...]) -> tuple[float, float]:
    image_points = np.asarray(
        [landmarks[30], landmarks[8], landmarks[36], landmarks[45], landmarks[48], landmarks[54]],
        dtype=np.float64,
    )
    model_points = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )
    height, width = image_shape[:2]
    focal = float(width)
    camera = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    success, rotation, _translation = cv2.solvePnP(
        model_points,
        image_points,
        camera,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0
    matrix, _ = cv2.Rodrigues(rotation)
    angles, *_ = cv2.RQDecomp3x3(matrix)
    pitch, yaw, _roll = angles
    return float(yaw), float(pitch)


def _box_visibility(box: np.ndarray, image_shape: tuple[int, ...]) -> float:
    x1, y1, x2, y2 = (float(item) for item in box)
    support = max((x2 - x1) * (y2 - y1), 1.0)
    clipped_x1, clipped_y1 = max(0.0, x1), max(0.0, y1)
    clipped_x2 = min(float(image_shape[1]), x2)
    clipped_y2 = min(float(image_shape[0]), y2)
    visible = max(0.0, (clipped_x2 - clipped_x1) * (clipped_y2 - clipped_y1))
    return float(np.clip(visible / support, 0.0, 1.0))


def _regional_visibility(
    landmarks: np.ndarray, bbox: np.ndarray, image_shape: tuple[int, ...]
) -> dict[str, float]:
    inter_pupil = float(
        np.linalg.norm(landmarks[36:42].mean(axis=0) - landmarks[42:48].mean(axis=0))
    )
    margin = 0.15 * inter_pupil
    groups = {
        "brow": landmarks[17:27],
        "eye": landmarks[36:48],
        "mouth": landmarks[48:68],
    }
    values = {"global": _box_visibility(bbox, image_shape)}
    for region, points in groups.items():
        box = np.asarray(
            [
                points[:, 0].min() - margin,
                points[:, 1].min() - margin,
                points[:, 0].max() + margin,
                points[:, 1].max() + margin,
            ],
            dtype=np.float32,
        )
        values[region] = _box_visibility(box, image_shape)
    return values


def _sample_frames(record: UtteranceRecord, maximum: int) -> list[tuple[int, float, np.ndarray]]:
    if record.video_path is None:
        return []
    capture = cv2.VideoCapture(record.video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video {record.video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            return []
        start = 0 if record.start_time is None else max(0, round(record.start_time * fps))
        end = total - 1 if record.end_time is None else min(total - 1, round(record.end_time * fps))
        local_count = max(0, end - start + 1)
        indices = [start + index for index in uniform_distinct_indices(local_count, maximum)]
        result: list[tuple[int, float, np.ndarray]] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if ok:
                result.append((frame_index, frame_index / fps, image))
        return result
    finally:
        capture.release()


def extract_face_track_candidates(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    maximum_frames: int = 8,
    device: str = "cuda",
    shot_histogram_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """Extract candidates and reset tracking when sampled-frame histograms jump."""

    output = Path(output_dir).expanduser().resolve()
    frame_root = output / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    detector = RetinaFaceFan(device=device)
    tracker = ByteTrackAdapter()
    records: list[dict[str, Any]] = []
    for raw in read_jsonl(manifest_path):
        utterance = UtteranceRecord.from_dict(raw)
        tracker.reset()
        previous_histogram: np.ndarray | None = None
        shot_id = 0
        for frame_index, timestamp, image in _sample_frames(utterance, maximum_frames):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(histogram, histogram)
            if previous_histogram is not None:
                distance = cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA)
                if distance >= shot_histogram_threshold:
                    shot_id += 1
                    tracker.reset()
            previous_histogram = histogram
            candidates = detector.detect(image)
            track_ids = tracker.update(candidates)
            frame_dir = frame_root / safe_identifier(utterance.utterance_id)
            frame_dir.mkdir(parents=True, exist_ok=True)
            image_path = frame_dir / f"{frame_index:08d}.jpg"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(image_path)
            for candidate, track_id in zip(candidates, track_ids, strict=True):
                yaw, pitch = _pose_from_landmarks(candidate.landmarks68, image.shape)
                visibility = _regional_visibility(
                    candidate.landmarks68, candidate.bbox, image.shape
                )
                records.append(
                    {
                        "utterance_id": utterance.utterance_id,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "image_path": str(image_path),
                        "bbox_xyxy": candidate.bbox.tolist(),
                        "landmarks68": candidate.landmarks68.tolist(),
                        "detection_confidence": candidate.confidence,
                        "sharpness": laplacian_sharpness(image),
                        "yaw": yaw,
                        "pitch": pitch,
                        "visibility": visibility,
                        "shot_id": shot_id,
                        "track_id": track_id,
                    }
                )
    atomic_write_jsonl(output / "face_candidates.jsonl", records)
    return records


def merge_speaker_assignments(
    candidates_path: str | Path,
    scores_path: str | Path,
    output_path: str | Path,
    *,
    minimum_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Select the highest-scoring frozen target track for every sampled frame."""

    scores: dict[tuple[str, int, str], float] = {}
    for value in read_jsonl(scores_path):
        key = (str(value["utterance_id"]), int(value["frame_index"]), str(value["track_id"]))
        scores[key] = float(value["speaker_confidence"])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for value in read_jsonl(candidates_path):
        grouped.setdefault((str(value["utterance_id"]), int(value["frame_index"])), []).append(value)
    output: list[dict[str, Any]] = []
    for (utterance_id, frame_index), candidates in sorted(grouped.items()):
        ranked = sorted(
            candidates,
            key=lambda value: scores.get(
                (utterance_id, frame_index, str(value.get("track_id"))), float("-inf")
            ),
            reverse=True,
        )
        selected = ranked[0]
        score = scores.get(
            (utterance_id, frame_index, str(selected.get("track_id"))), float("-inf")
        )
        record = dict(selected)
        record["speaker_confidence"] = 0.0 if not np.isfinite(score) else score
        record["assignment_valid"] = bool(np.isfinite(score) and score >= minimum_score)
        output.append(record)
    atomic_write_jsonl(output_path, output)
    return output
