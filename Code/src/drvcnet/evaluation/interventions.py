"""Frozen-reference substitutions for N1/N2/N3 intervention experiments."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drvcnet.config import ExperimentConfig
from drvcnet.constants import REGIONS
from drvcnet.io import atomic_write_jsonl, read_jsonl, resolve_record_path
from drvcnet.records import ReferenceRecord, UtteranceRecord


@dataclass(frozen=True)
class MatchBins:
    quality: tuple[float, ...] = (0.0, 0.5, 0.7, 0.85, 1.01)
    age: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0, float("inf"))

    @staticmethod
    def _index(value: float, boundaries: tuple[float, ...]) -> int:
        return max(0, min(len(boundaries) - 2, int(np.searchsorted(boundaries, value, side="right") - 1)))

    def signature(self, quality: float, age: float) -> tuple[int, int]:
        return self._index(quality, self.quality), self._index(age, self.age)


def _stable_choice(values: list[str], key: str, seed: int) -> str:
    if not values:
        raise ValueError("Cannot choose from an empty replacement set")
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return sorted(values)[int.from_bytes(digest[:8], "big") % len(values)]


def _feature_metadata(config: ExperimentConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    index_path = config.dataset.cache_dir / "feature_index.jsonl"
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for value in read_jsonl(index_path):
        path = resolve_record_path(index_path, value["path"])
        with np.load(path, allow_pickle=False) as cache:
            output[str(value["utterance_id"])] = (
                np.asarray(cache["visual_quality"], dtype=np.float32),
                np.asarray(cache["visual_valid"], dtype=np.bool_),
            )
    return output


def _time_available(past: UtteranceRecord, current: UtteranceRecord, exclude_overlap: bool) -> bool:
    if not exclude_overlap:
        return True
    return (
        past.end_time is not None
        and current.start_time is not None
        and past.end_time <= current.start_time
    )


def build_reference_intervention(
    config: ExperimentConfig,
    intervention: str,
    output_path: str | Path,
    *,
    seed: int = 17,
    bins: MatchBins = MatchBins(),
) -> list[dict[str, Any]]:
    """Create legal past substitutions with fixed quality/age-bin matching."""

    if intervention not in {"N1_other_speaker", "N2_older_random", "N3_self_copy"}:
        raise ValueError(f"Unknown intervention: {intervention}")
    records = [UtteranceRecord.from_dict(value) for value in read_jsonl(config.dataset.manifest)]
    by_id = {record.utterance_id: record for record in records}
    references = {
        value["utterance_id"]: ReferenceRecord.from_dict(value)
        for value in read_jsonl(config.dataset.reference_file)
    }
    feature_metadata = _feature_metadata(config)
    by_dialogue: dict[str, list[UtteranceRecord]] = defaultdict(list)
    for record in records:
        by_dialogue[record.dialogue_id].append(record)
    for dialogue in by_dialogue.values():
        dialogue.sort(key=lambda item: (item.order, item.utterance_id))

    output: list[dict[str, Any]] = []
    for current in records:
        history = [
            item
            for item in by_dialogue[current.dialogue_id]
            if item.order < current.order
            and _time_available(item, current, config.dataset.exclude_overlapping_history)
        ]
        clean = references[current.utterance_id]
        changed_map: dict[str, Any] = {}
        regions: dict[str, dict[str, Any]] = {}
        requested_slots = 0
        replaced_slots = 0
        for region_index, region in enumerate(REGIONS):
            clean_region = clean.regions[region]
            recent_id = clean_region.recent_id
            historical_ids = list(clean_region.historical_ids)
            replacement_details: dict[str, Any] = {"recent": None, "historical": []}

            def candidates_for(source_id: str) -> list[str]:
                source_quality = float(feature_metadata[source_id][0][region_index])
                source = by_id[source_id]
                signature = bins.signature(source_quality, float(current.order - source.order))
                candidates: list[str] = []
                for candidate in history:
                    if not feature_metadata[candidate.utterance_id][1][region_index]:
                        continue
                    candidate_quality = float(
                        feature_metadata[candidate.utterance_id][0][region_index]
                    )
                    if candidate_quality < config.video.quality_threshold:
                        continue
                    candidate_age = float(current.order - candidate.order)
                    if bins.signature(candidate_quality, candidate_age) != signature:
                        continue
                    if intervention == "N1_other_speaker" and candidate.speaker_id == current.speaker_id:
                        continue
                    if intervention == "N2_older_random":
                        if candidate.speaker_id != current.speaker_id:
                            continue
                        if candidate.order >= source.order or candidate.utterance_id == source_id:
                            continue
                    candidates.append(candidate.utterance_id)
                return candidates

            if intervention == "N3_self_copy":
                if recent_id is not None:
                    requested_slots += 1
                    replaced_slots += 1
                    replacement_details["recent"] = {"from": recent_id, "to": current.utterance_id}
                    recent_id = current.utterance_id
                requested_slots += len(historical_ids)
                replaced_slots += len(historical_ids)
                replacement_details["historical"] = [
                    {"from": source_id, "to": current.utterance_id}
                    for source_id in historical_ids
                ]
                historical_ids = [current.utterance_id for _ in historical_ids]
            else:
                if recent_id is not None:
                    requested_slots += 1
                    candidates = candidates_for(recent_id)
                    if candidates:
                        replacement = _stable_choice(
                            candidates, f"{current.utterance_id}|{region}|recent", seed
                        )
                        replacement_details["recent"] = {"from": recent_id, "to": replacement}
                        recent_id = replacement
                        replaced_slots += 1
                replacements: list[str] = []
                for member_index, source_id in enumerate(clean_region.historical_ids):
                    requested_slots += 1
                    candidates = candidates_for(source_id)
                    if candidates:
                        replacement = _stable_choice(
                            candidates,
                            f"{current.utterance_id}|{region}|historical|{member_index}",
                            seed,
                        )
                        replacements.append(replacement)
                        replacement_details["historical"].append(
                            {"from": source_id, "to": replacement}
                        )
                        replaced_slots += 1
                    else:
                        replacements.append(source_id)
                historical_ids = replacements
            regions[region] = {
                "recent_id": recent_id,
                "recent_distance": clean_region.recent_distance,
                "historical_ids": historical_ids,
                "historical_distances": list(clean_region.historical_distances),
                "eligible_count": clean_region.eligible_count,
            }
            changed_map[region] = replacement_details
        output.append(
            {
                "utterance_id": current.utterance_id,
                "dialogue_id": current.dialogue_id,
                "regions": regions,
                "intervention": intervention,
                "intervention_eligible": requested_slots > 0 and replaced_slots == requested_slots,
                "requested_slots": requested_slots,
                "replaced_slots": replaced_slots,
                "replacement_map": changed_map,
            }
        )
    atomic_write_jsonl(output_path, output)
    return output
