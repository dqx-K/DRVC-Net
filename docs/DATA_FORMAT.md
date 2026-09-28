# Data and cache contracts

All JSONL files contain one UTF-8 JSON object per line. Paths may be absolute or
relative to the JSONL file. IDs must be strings and unique within a dataset.

## Utterance manifest

Required fields:

```json
{
  "dataset": "MELD",
  "split": "train",
  "session": null,
  "dialogue_id": "train:12",
  "utterance_id": "train:12:3",
  "order": 3,
  "speaker_id": "Chandler",
  "role_id": 1,
  "start_time": null,
  "end_time": null,
  "label": "neutral",
  "text": "Example utterance",
  "audio_path": ".../dia12_utt3.wav",
  "video_path": ".../dia12_utt3.mp4"
}
```

`role_id` is recomputed in first-appearance order and represents an anonymous
dialogue role. IEMOCAP `start_time` and `end_time` are seconds within the dialogue
clip. MELD uses annotated order as the causal boundary.

## Frame assignment input

One record describes one distinct sampled frame of an utterance:

```json
{
  "utterance_id": "train:12:3",
  "frame_index": 44,
  "timestamp": 1.47,
  "image_path": ".../frame_000044.jpg",
  "bbox_xyxy": [120.0, 40.0, 350.0, 310.0],
  "landmarks68": [[0.0, 0.0]],
  "detection_confidence": 0.99,
  "speaker_confidence": 0.91,
  "visibility": {"global": 1.0, "brow": 1.0, "eye": 0.95, "mouth": 1.0},
  "sharpness": 186.2,
  "yaw": 4.0,
  "pitch": -2.0,
  "assignment_valid": true,
  "shot_id": 2,
  "track_id": 7
}
```

`landmarks68` must contain 68 `[x,y]` image coordinates. The selected track and
speaker confidence must be generated without emotion labels or stored character
portraits. TalkNet logits must be calibrated or monotonically mapped and frozen
before test evaluation. `drvc-calibrate-speaker-scores` fits a Platt mapping on
training audit records with `raw_score` and `target_correct`, then writes
`speaker_confidence` for the candidate-score stream. `drvc-merge-assignments`
selects the highest-confidence target track for every sampled frame.

## Blur calibration

`drvc-calibrate-sharpness` creates a training-only calibration JSON consumed by
`drvc-prepare-visual`:

```json
{"lower_quantile": 42.0, "upper_quantile": 310.0}
```

Sharpness at/below the lower endpoint maps to 0, at/above the upper endpoint to
1, with linear interpolation. The command derives and freezes both endpoints
from the split file's training IDs, enforcing the development/test boundary.

## Feature cache

Each utterance is stored as `<cache_dir>/features/<safe_id>.npz` with:

- `text`: float32 `[768]`; `text_available`: uint8 scalar;
- `audio`: float32 `[768]`; `audio_available`: uint8 scalar;
- `visual`: float32 `[4,512]`, quality-pooled frozen features;
- `visual_quality`: float32 `[4]`;
- `visual_valid`: uint8 `[4]`;
- `frame_count`: int32 `[4]`.

An index JSONL maps utterance IDs to cache files and includes a SHA-256 digest.

## Reference index

Each utterance record contains per-region `recent_id`, `historical_ids`, and the
same-speaker turn distances. IDs point to feature-cache entries. Reference
construction is label-independent; `drvcnet.data.dataset` joins supervision after
the complete visual history index has been loaded.

## Predictions

Every prediction record contains dataset, fold, seed, model, dialogue and
utterance IDs, true/predicted label, probabilities, history/reference metadata,
and subgroup fields. Statistical tools preserve the paired design by resampling
complete dialogues.
