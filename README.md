# DRVC-Net

This repository provides a paper-aligned PyTorch implementation of
**DRVC-Net: Dual-Reference Speaker-Relative Visual Change Learning for Multimodal
Emotion Recognition in Conversations**.

DRVC-Net turns same-speaker visual history into two explicit coordinates: the
latest reliable state and a distinct baseline of earlier states. This codebase
implements that idea as a complete, auditable experiment pipeline:

1. build a causal, dialogue-ordered utterance manifest;
2. create IEMOCAP leave-one-session-out or MELD official splits;
3. align full-face, brow, eye, and mouth observations and compute reliability;
4. cache frozen RoBERTa-base, wav2vec2-base-960h, and ImageNet ResNet-18 features;
5. build region-specific same-speaker recent and historical reference indices;
6. train B6 or the paper's B/C/N/G controlled alternatives;
7. export utterance-level predictions, mechanism-focused subgroup metrics,
   paired dialogue bootstrap intervals, permutation tests, interventions, and
   quality stress tests.

The repository keeps licensed assets in their official distribution channels:
obtain IEMOCAP, MELD, and the upstream RetinaFace, FAN, ByteTrack, and TalkNet
checkpoints under their respective terms, then connect them through the paths
and manifests described below.

## Paper-to-code map

| Paper component | Implementation |
|---|---|
| Eqs. 1--4, regional observations and quality | `drvcnet.preprocessing.quality`, `regions`, `features` |
| Eqs. 5--8, causal dual references | `drvcnet.data.references` |
| Eqs. 9--12, change/gates/regional attention | `drvcnet.models.visual_reference` |
| Eqs. 13--15, multimodal causal context | `drvcnet.models.context`, `drvc_net` |
| Weighted objective and training protocol | `drvcnet.training` |
| WF1/MF1/accuracy and paired statistics | `drvcnet.evaluation` |
| B0--B7, B3Q/B4Q, C1--C3, N4, G1/G2 | `ModelVariant` and the visual module |

See [`docs/METHOD_MAPPING.md`](docs/METHOD_MAPPING.md) for exact tensor rules and
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for the manifest/cache contracts.

## Installation

Python 3.10 or 3.11 and the paper's PyTorch/CUDA pair are recommended:

```bash
python -m venv .venv
.venv/Scripts/activate                 # Windows
python -m pip install -e .
python -m pip install -e ".[vision]"   # RetinaFace/FAN/ByteTrack adapters
```

The pinned core versions reproduce the declared PyTorch 2.3.1 environment.
GPU wheel selection may require the matching PyTorch CUDA 12.1 index.

## Data preparation

### IEMOCAP

```bash
drvc-build-manifest iemocap --root D:/datasets/IEMOCAP_full_release \
  --output data/iemocap/manifest.jsonl
drvc-make-splits --manifest data/iemocap/manifest.jsonl \
  --output-dir data/iemocap/splits --dataset IEMOCAP --seed 2026
```

The parser retains only `ang`, `hap`, `exc`, `sad`, `fru`, and `neu`, and assigns
speaker IDs from the utterance identifier. Historical references additionally
require `past.end_time <= current.start_time`.

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

### Auditable face assignment

The visual pipeline separates candidate tracking from target-speaker assignment.
`drvc-detect-tracks` extracts RetinaFace/FAN/ByteTrack candidates, and
`drvc-merge-assignments` joins frozen TalkNet or metadata-based speaker scores.
The resulting `frame_assignments.jsonl` makes every selected track, confidence,
landmark set, pose estimate, and visibility factor traceable. This explicit
contract supports both IEMOCAP's role-to-track mapping and MELD's TalkNet-based
assignment while keeping the regional model unchanged.

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

Each configuration represents one dataset split/fold and one training seed.
For the paper protocol, run seeds 42, 123, and 2026; for IEMOCAP, repeat all five
outer folds. Hyperparameter search uses seed 17 only.

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

To switch a controlled model, copy a dataset config and change `model.variant`.
Structural alternatives must be trained from scratch.

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

The stress pipeline preserves the clean/current/reference cache distinction.
Build blur or occlusion crops, extract a paired feature cache, and select it as
the current or reference store with `drvc-stress-test`. N1/N2/N3 reference
substitutions are produced with `drvc-build-intervention`.

## Static validation

The fast validation path checks repository structure before any experiment:

```bash
python scripts/static_check.py
```

The checker parses every Python file with `ast`, validates YAML/JSON syntax, and
resolves local module paths. It performs a pure static check and leaves model and
data execution to the experiment commands.

## Reproducibility notes

- References are built before inserting the current observation into memory.
- Memory resets at each dialogue and is label-independent by construction.
- A historical pool excludes the most recent eligible observation and contains
  at most four earlier observations.
- Frozen visual features are cached before the trainable regional projection;
  current and historical observations therefore share the live projection.
- Every historical token retains the references available at its own prediction
  boundary, guaranteeing prefix-consistent causal windows.
- Missing inputs are zeroed and accompanied by explicit masks. An all-missing
  visual utterance returns an exact zero visual vector without a fully masked
  softmax.
- Every reported result is regenerated from saved utterance-level predictions,
  making folds, seeds, subgroup supports, and paired comparisons fully traceable.
