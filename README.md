# DRVC-Net

DRVC-Net is a multimodal emotion recognition model for conversations. It combines
text, audio, and facial information with causal dialogue context, and represents
facial change relative to two same-speaker references: the latest reliable visual
state and a quality-weighted historical baseline.

The visual stream contains four regions—global face, brows, eyes, and mouth.
Region-specific change features and quality-conditioned gates determine how much
each reference contributes to the current utterance. The fused utterance sequence
is processed by a causal Transformer for emotion classification.

Experiments use six-class IEMOCAP and seven-class MELD. The repository includes
data preparation, feature extraction, model training, controlled variants,
evaluation, statistical analysis, and visual-quality stress tests.

## Repository structure

| Path | Contents |
|---|---|
| `src/drvcnet/data/` | IEMOCAP/MELD manifests, splits, causal windows, and dual-reference indices |
| `src/drvcnet/preprocessing/` | Face tracking, speaker-score calibration, regional crops, quality estimation, and frozen features |
| `src/drvcnet/models/` | Dual-reference visual module, multimodal fusion, and causal context encoder |
| `src/drvcnet/training/` | Losses, optimizer schedule, development search, checkpoints, and run provenance |
| `src/drvcnet/evaluation/` | Metrics, subgroups, bootstrap tests, interventions, and stress evaluation |
| `configs/` | IEMOCAP, MELD, and model-variant configurations |
| `scripts/` | Static checks and parameter-count reports |
| `docs/` | Method details and data formats |

Detailed tensor definitions are available in
[`docs/METHOD_MAPPING.md`](docs/METHOD_MAPPING.md), and file formats are described
in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).

## Installation

Python 3.10 or 3.11 is recommended:

```bash
python -m venv .venv
.venv/Scripts/activate                 # Windows
python -m pip install -e .
python -m pip install -e ".[vision]"   # RetinaFace/FAN/ByteTrack adapters
```

The default environment uses PyTorch 2.3.1 and CUDA 12.1.

## Data preparation

### IEMOCAP

```bash
drvc-build-manifest iemocap --root D:/datasets/IEMOCAP_full_release \
  --output data/iemocap/manifest.jsonl
drvc-make-splits --manifest data/iemocap/manifest.jsonl \
  --output-dir data/iemocap/splits --dataset IEMOCAP --seed 2026
```

The IEMOCAP configuration uses `ang`, `hap`, `exc`, `sad`, `fru`, and `neu`.
Historical references satisfy `past.end_time <= current.start_time`.

### MELD

```bash
drvc-build-manifest meld --root D:/datasets/MELD.Raw \
  --train-csv D:/datasets/MELD/train_sent_emo.csv \
  --dev-csv D:/datasets/MELD/dev_sent_emo.csv \
  --test-csv D:/datasets/MELD/test_sent_emo.csv \
  --output data/meld/manifest.jsonl
drvc-make-splits --manifest data/meld/manifest.jsonl \
  --output-dir data/meld/splits --dataset MELD --seed 2026
```

### Visual preprocessing

The visual pipeline separates candidate tracking from target-speaker assignment.
`drvc-detect-tracks` extracts RetinaFace/FAN/ByteTrack candidates, and
`drvc-merge-assignments` joins frozen TalkNet or metadata-based speaker scores.
The resulting `frame_assignments.jsonl` stores the selected track, confidence,
landmarks, pose, and regional visibility. The same preprocessing interface is
used for IEMOCAP metadata-based assignment and MELD TalkNet scores.

Sharpness endpoints are fitted on training observations, frozen, and then used
by the four-region quality equation:

```bash
drvc-detect-tracks --manifest data/meld/manifest.jsonl \
  --output-dir cache/meld/candidates --device cuda

drvc-calibrate-speaker-scores \
  --audit data/meld/train_assignment_audit.jsonl \
  --scores data/meld/talknet_raw_scores.jsonl \
  --output data/meld/talknet_target_scores.jsonl \
  --calibration-output cache/meld/speaker_calibration.json

drvc-merge-assignments \
  --candidates cache/meld/candidates/face_candidates.jsonl \
  --scores data/meld/talknet_target_scores.jsonl \
  --output data/meld/frame_assignments.jsonl

drvc-calibrate-sharpness \
  --assignments data/meld/frame_assignments.jsonl \
  --split-file data/meld/splits/official.json \
  --output cache/meld/sharpness_calibration.json

drvc-prepare-visual --manifest data/meld/manifest.jsonl \
  --assignments data/meld/frame_assignments.jsonl \
  --calibration cache/meld/sharpness_calibration.json \
  --output-dir cache/meld/visual_observations

drvc-extract-features --config configs/meld_b6.yaml \
  --visual-observations cache/meld/visual_observations/observations.jsonl

drvc-build-references --config configs/meld_b6.yaml
```

## Training and evaluation

Each configuration represents one dataset split or fold and one training seed.
The main experiments use seeds 42, 123, and 2026. IEMOCAP uses five
leave-one-session-out folds, and MELD uses the official split. Development search
uses seed 17.

```bash
drvc-train --config configs/iemocap_b6.yaml
drvc-evaluate --config configs/iemocap_b6.yaml \
  --checkpoint outputs/iemocap/fold1_seed42/best.pt
```

Training uses AdamW, two-step accumulation to effective batch 32, 5% warmup,
cosine decay, a 40-epoch budget, gradient norm 1.0, and dev-WF1 early stopping
with MF1/earlier-epoch tie breaking. Only the final utterance in each causal
window receives a loss. Every run records the resolved configuration, Git state,
data/cache hashes, software versions, hardware, checkpoint hash, epoch history,
and utterance-level prediction hash.

Set `model.variant` to select B0--B7, B3Q/B4Q, C1--C3, N4, G1, or G2. Each
structural variant is trained independently.

Aggregate all seeds/folds and compute paired comparisons directly from saved
utterance predictions:

```bash
drvc-summarize --config configs/iemocap_b6.yaml \
  --predictions outputs/iemocap/*/predictions.jsonl \
  --output outputs/iemocap/b6_summary.json

drvc-statistics --config configs/iemocap_b6.yaml \
  --first outputs/iemocap/b6/*/predictions.jsonl \
  --second outputs/iemocap/b2/*/predictions.jsonl \
  --output outputs/iemocap/b6_vs_b2.json
```

`drvc-stress-test` evaluates current/reference missingness, blur, occlusion, and
speaker-assignment perturbations. `drvc-build-intervention` creates the N1/N2/N3
reference substitutions.

## Static validation

Run the static repository check with:

```bash
python scripts/static_check.py
```

The checker parses Python files, validates YAML/JSON syntax, and resolves local
module paths without starting model or data execution.
