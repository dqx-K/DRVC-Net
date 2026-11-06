#!/usr/bin/env python3
"""Convert official TalkNet track scores to the DRVC-Net score schema."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_MAP_FIELDS = {
    "talknet_track_index",
    "talknet_frame_index",
    "utterance_id",
    "frame_index",
    "track_id",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-pickle", type=Path, required=True)
    parser.add_argument("--track-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.scores_pickle.open("rb") as handle:
        scores = pickle.load(handle)
    if not isinstance(scores, (list, tuple)):
        raise ValueError("TalkNet scores.pckl must contain the official per-track score list")

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    with args.track_map.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            mapping = json.loads(line)
            missing = REQUIRED_MAP_FIELDS - mapping.keys()
            if missing:
                raise ValueError(
                    f"{args.track_map}:{line_number}: missing fields {sorted(missing)}"
                )
            track_index = int(mapping["talknet_track_index"])
            talknet_frame = int(mapping["talknet_frame_index"])
            raw_score = float(scores[track_index][talknet_frame])
            key = (
                str(mapping["utterance_id"]),
                int(mapping["frame_index"]),
                str(mapping["track_id"]),
            )
            if key in seen:
                raise ValueError(f"Duplicate DRVC-Net candidate mapping: {key}")
            seen.add(key)
            records.append(
                {
                    "utterance_id": key[0],
                    "frame_index": key[1],
                    "track_id": key[2],
                    "raw_score": raw_score,
                    "talknet_track_index": track_index,
                    "talknet_frame_index": talknet_frame,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=args.output.parent, suffix=".jsonl")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary_name, args.output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(json.dumps({"records": len(records), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
