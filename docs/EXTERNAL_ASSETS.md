# External Model Assets

DRVC-Net keeps third-party model weights outside the source tree and records the selected asset in each run configuration and provenance record. The following procedures reproduce the two external preprocessing assets used by the visual pipeline.

## FER+ ResNet-18

The FER+ extension uses a torchvision ResNet-18 initialized on ImageNet and fine-tuned with FER+ labels. FER+ annotations and the official data-generation utility are available from the [Microsoft FERPlus repository](https://github.com/microsoft/FERPlus). The corresponding FER2013 image CSV is distributed through the [FER2013 Kaggle challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

Prepare the FER+ training layout with the official utility:

```bash
git clone https://github.com/microsoft/FERPlus.git
python FERPlus/src/generate_training_data.py \
  -d D:/datasets/FERPlus/data \
  -fer D:/datasets/FER2013/fer2013.csv \
  -ferplus FERPlus/fer2013new.csv
```

The DRVC-Net loader accepts a PyTorch mapping containing either a raw torchvision ResNet-18 state dictionary or a state dictionary under `state_dict`, `model`, or `backbone`. Prefixes `module.`, `model.`, and `backbone.` are removed, and the classification head is ignored. Store the resulting file as:

```text
checkpoints/ferplus_resnet18.pt
```

The convolutional stem and residual blocks must match `torchvision.models.resnet18` from torchvision 0.18.1. Feature extraction uses the 512-dimensional global-average-pooled representation and the ImageNet ResNet-18 normalization defined by torchvision. The checkpoint path is selected by `video.visual_checkpoint` in `configs/revision_protocol.yaml`.

## TalkNet active-speaker scores

MELD speaker assignment uses the official [TalkNet-ASD implementation](https://github.com/TaoRuijie/TalkNet-ASD) accompanying the ACM Multimedia 2021 paper. Its `demoTalkNet.py` entry point downloads `pretrain_TalkSet.model`, generates face tracks, and stores per-frame active-speaker scores in `scores.pckl`.

```bash
git clone https://github.com/TaoRuijie/TalkNet-ASD.git
cd TalkNet-ASD
pip install -r requirement.txt
python demoTalkNet.py --videoName <meld-video-id>
```

For each processed video, create `talknet_track_map.jsonl` by aligning the sorted TalkNet crop tracks with the candidate tracks produced by `drvc-detect-tracks`. Each line records:

```json
{"talknet_track_index":0,"talknet_frame_index":0,"utterance_id":"dia1_utt1","frame_index":0,"track_id":"3"}
```

Convert the official score file into the DRVC-Net raw-score stream:

```bash
python scripts/convert_talknet_scores.py \
  --scores-pickle TalkNet-ASD/demo/<meld-video-id>/pywork/scores.pckl \
  --track-map data/meld/talknet_track_map.jsonl \
  --output data/meld/talknet_raw_scores.jsonl
```

The output fields are `utterance_id`, `frame_index`, `track_id`, and `raw_score`, with the TalkNet indices retained for traceability. `drvc-calibrate-speaker-scores` fits the training-audit mapping and writes `speaker_confidence`; `drvc-merge-assignments` then selects the highest-scoring candidate for each sampled frame.
