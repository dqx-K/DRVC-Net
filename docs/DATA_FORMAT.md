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
portraits. TalkNet logits must be monotonically calibrated and frozen before
test evaluation. `drvc-calibrate-speaker-scores` fits an isotonic piecewise-linear mapping on
training audit records with `raw_score` and `target_correct`, then writes
`speaker_confidence` for the candidate-score stream. `drvc-merge-assignments`
selects the highest-confidence target track for every sampled frame.

## Assignment audits

`drvc-build-audit-sample` writes the stratified sample list and selected-frame
metadata. Annotation records add `annotator_1`, `annotator_2`, and `adjudicated`,
each with one of `correct`, `incorrect`, or `undecidable`.
`drvc-summarize-audit` reports pre-adjudication Cohen's kappa on independently
decidable pairs and the stratified accepted-assignment accuracy. The complete
annotation procedure is specified in `docs/AUDIT_PROTOCOL.md`.

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

The public release normalizes prediction outputs into two files so that
utterance metadata is not repeated for every model, fold, and seed.

`predictions_main.jsonl.gz` contains one record per evaluated utterance and run:

```json
{
  "run_id": "iemocap_B6_fold1_seed42",
  "dataset": "IEMOCAP",
  "model": "B6",
  "fold": 1,
  "seed": 42,
  "dialogue_id": "Ses01F_impro01",
  "utterance_id": "Ses01F_impro01_F000",
  "y_true": "neutral",
  "y_pred": "neutral",
  "probabilities": [0.01, 0.02, 0.03, 0.04, 0.05, 0.85]
}
```

`utterance_metadata.jsonl.gz` contains one record per unique
`(dataset, utterance_id)` and provides the fixed fields used by the subgroup
analyses:

```json
{
  "dataset": "IEMOCAP",
  "utterance_id": "Ses01F_impro01_F000",
  "dialogue_id": "Ses01F_impro01",
  "speaker_id": "Ses01F",
  "order": 0,
  "shift_group": "no_previous",
  "reference_availability": "none",
  "both": false,
  "history_bin": "0",
  "reference_age": null,
  "reference_age_bin": null,
  "visual_missing": false,
  "current_mask": [true, true, true, true],
  "recent_mask": [false, false, false, false],
  "historical_mask": [false, false, false, false],
  "reference_ids": {"recent": [null, null, null, null], "historical": [[], [], [], []]}
}
```

Allowed `shift_group` values are `shift`, `stable`, and `no_previous`.
Allowed `history_bin` values are `0`, `1`, `2_4`, and `5_plus`. Reference-age
bins are `recent`, `medium`, and `old`, with `null` used when no legal recent
reference exists. Prediction and metadata records join on
`(dataset, utterance_id)`; their dialogue identifiers must agree.

`additional_predictions.jsonl.gz` stores only the per-utterance outputs needed
to reconstruct interval-bearing intervention, stress, and exploratory CSS-mix
results. Its `experiment_type` is one of `intervention`, `stress`, or
`extension`. Point-estimate-only operating-point studies are represented by
their resolved configurations and run-level metrics rather than duplicated
prediction streams.

Statistical tools preserve the paired design by resampling complete dialogues.
