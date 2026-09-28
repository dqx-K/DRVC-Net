"""Validated in-memory records shared by preprocessing and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drvcnet.constants import REGIONS


@dataclass(frozen=True)
class UtteranceRecord:
    dataset: str
    split: str
    session: str | None
    dialogue_id: str
    utterance_id: str
    order: int
    speaker_id: str
    role_id: int
    start_time: float | None
    end_time: float | None
    label: str
    text: str
    audio_path: str | None
    video_path: str | None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UtteranceRecord":
        return cls(
            dataset=str(value["dataset"]).upper(),
            split=str(value["split"]),
            session=None if value.get("session") is None else str(value["session"]),
            dialogue_id=str(value["dialogue_id"]),
            utterance_id=str(value["utterance_id"]),
            order=int(value["order"]),
            speaker_id=str(value["speaker_id"]),
            role_id=int(value["role_id"]),
            start_time=None if value.get("start_time") is None else float(value["start_time"]),
            end_time=None if value.get("end_time") is None else float(value["end_time"]),
            label=str(value["label"]),
            text=str(value.get("text", "")),
            audio_path=None if value.get("audio_path") is None else str(value["audio_path"]),
            video_path=None if value.get("video_path") is None else str(value["video_path"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameAssignment:
    utterance_id: str
    frame_index: int
    timestamp: float
    image_path: str
    bbox_xyxy: tuple[float, float, float, float]
    landmarks68: np.ndarray
    detection_confidence: float
    speaker_confidence: float
    visibility: dict[str, float]
    sharpness: float
    yaw: float
    pitch: float
    assignment_valid: bool
    shot_id: str | int | None = None
    track_id: str | int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FrameAssignment":
        landmarks = np.asarray(value["landmarks68"], dtype=np.float32)
        if landmarks.shape != (68, 2):
            raise ValueError(
                f"{value.get('utterance_id')}: landmarks68 must have shape (68, 2), got {landmarks.shape}"
            )
        bbox = tuple(float(item) for item in value["bbox_xyxy"])
        if len(bbox) != 4:
            raise ValueError("bbox_xyxy must contain four values")
        visibility = {region: float(value["visibility"].get(region, 0.0)) for region in REGIONS}
        return cls(
            utterance_id=str(value["utterance_id"]),
            frame_index=int(value["frame_index"]),
            timestamp=float(value["timestamp"]),
            image_path=str(value["image_path"]),
            bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3]),
            landmarks68=landmarks,
            detection_confidence=float(value["detection_confidence"]),
            speaker_confidence=float(value["speaker_confidence"]),
            visibility=visibility,
            sharpness=float(value["sharpness"]),
            yaw=float(value["yaw"]),
            pitch=float(value["pitch"]),
            assignment_valid=bool(value["assignment_valid"]),
            shot_id=value.get("shot_id"),
            track_id=value.get("track_id"),
        )


@dataclass(frozen=True)
class ReferenceRegion:
    recent_id: str | None
    recent_distance: float
    historical_ids: tuple[str, ...]
    historical_distances: tuple[float, ...]
    eligible_count: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceRegion":
        ids = tuple(str(item) for item in value.get("historical_ids", []))
        distances = tuple(float(item) for item in value.get("historical_distances", []))
        if len(ids) != len(distances):
            raise ValueError("historical_ids and historical_distances must have equal length")
        return cls(
            recent_id=None if value.get("recent_id") is None else str(value["recent_id"]),
            recent_distance=float(value.get("recent_distance", 0.0)),
            historical_ids=ids,
            historical_distances=distances,
            eligible_count=int(value.get("eligible_count", len(ids) + (value.get("recent_id") is not None))),
        )


@dataclass(frozen=True)
class ReferenceRecord:
    utterance_id: str
    dialogue_id: str
    regions: dict[str, ReferenceRegion]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceRecord":
        regions = {name: ReferenceRegion.from_dict(value["regions"][name]) for name in REGIONS}
        return cls(
            utterance_id=str(value["utterance_id"]),
            dialogue_id=str(value["dialogue_id"]),
            regions=regions,
        )


@dataclass(frozen=True)
class FeatureIndexRecord:
    utterance_id: str
    path: Path
    sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any], base: Path) -> "FeatureIndexRecord":
        path = Path(value["path"])
        return cls(
            utterance_id=str(value["utterance_id"]),
            path=path if path.is_absolute() else (base / path).resolve(),
            sha256=str(value["sha256"]),
        )
