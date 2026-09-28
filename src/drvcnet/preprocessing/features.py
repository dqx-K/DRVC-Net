"""Frozen RoBERTa, wav2vec 2.0, and ResNet-18 feature caching."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchaudio
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, Wav2Vec2Model

from drvcnet.config import ExperimentConfig
from drvcnet.constants import REGIONS
from drvcnet.io import atomic_write_jsonl, read_jsonl, resolve_record_path, safe_identifier, sha256_file
from drvcnet.records import UtteranceRecord


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class FrozenTextEncoder(nn.Module):
    def __init__(self, checkpoint: str = "roberta-base", max_tokens: int = 128) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.max_tokens = max_tokens
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def forward(self, texts: list[str], device: torch.device) -> Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = tokens.pop("special_tokens_mask").to(device)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        output = self.model(**tokens).last_hidden_state
        valid = tokens["attention_mask"].bool() & ~special.bool()
        return _masked_mean(output, valid)


class FrozenAudioEncoder(nn.Module):
    def __init__(self, checkpoint: str = "facebook/wav2vec2-base-960h") -> None:
        super().__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
        self.model = Wav2Vec2Model.from_pretrained(checkpoint)
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def forward(self, waveform: Tensor, sample_rate: int, device: torch.device) -> Tensor:
        mono = waveform.mean(dim=0)
        if sample_rate != 16_000:
            mono = torchaudio.functional.resample(mono, sample_rate, 16_000)
        inputs = self.extractor(
            mono.cpu().numpy(), sampling_rate=16_000, return_tensors="pt", padding=True
        )
        input_values = inputs["input_values"].to(device)
        attention = inputs.get("attention_mask")
        if attention is not None:
            attention = attention.to(device)
        hidden = self.model(input_values=input_values, attention_mask=attention).last_hidden_state
        if attention is None:
            return hidden.mean(dim=1)
        feature_mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], attention)
        return _masked_mean(hidden, feature_mask)


class FrozenVisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.transform = weights.transforms()
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def forward(self, images_bgr: list[np.ndarray], device: torch.device) -> Tensor:
        tensors: list[Tensor] = []
        for image in images_bgr:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(torch.from_numpy(rgb).permute(2, 0, 1)))
        batch = torch.stack(tensors).to(device)
        return self.model(batch).flatten(1)


def _load_audio(path: Path) -> tuple[Tensor, int]:
    try:
        return torchaudio.load(str(path))
    except Exception as error:
        raise RuntimeError(
            f"Unable to decode audio from {path}. For MELD MP4 files, install a "
            "torchaudio build with FFmpeg or extract 16 kHz mono WAV files first."
        ) from error


def _encode_visual_observation(
    observation: dict[str, Any] | None,
    observation_file: Path,
    encoder: FrozenVisualEncoder,
    device: torch.device,
    min_valid_frames: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    visual = np.zeros((len(REGIONS), 512), dtype=np.float32)
    quality = np.zeros(len(REGIONS), dtype=np.float32)
    valid = np.zeros(len(REGIONS), dtype=np.uint8)
    counts = np.zeros(len(REGIONS), dtype=np.int32)
    if observation is None:
        return visual, quality, valid, counts
    frames = observation.get("frames", [])
    distinct_count = int(observation.get("distinct_frame_count", len(frames)))
    for region_index, region in enumerate(REGIONS):
        images: list[np.ndarray] = []
        qualities: list[float] = []
        for frame in frames:
            region_value = frame["regions"][region]
            if not bool(region_value["valid"]):
                continue
            crop_path = resolve_record_path(observation_file, region_value["crop_path"])
            image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(crop_path)
            images.append(image)
            qualities.append(float(region_value["quality"]))
        counts[region_index] = len(images)
        if len(images) < min_valid_frames:
            continue
        frame_features = encoder(images, device).cpu().numpy().astype(np.float32)
        weights = np.asarray(qualities, dtype=np.float32)
        denominator = float(weights.sum()) + epsilon
        visual[region_index] = (frame_features * weights[:, None]).sum(axis=0) / denominator
        quality[region_index] = float(weights.sum()) / max(distinct_count, 1)
        valid[region_index] = 1
    return visual, quality, valid, counts


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(file_descriptor)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def extract_frozen_features(
    config: ExperimentConfig,
    visual_observations: str | Path,
    *,
    text_checkpoint: str = "roberta-base",
    audio_checkpoint: str = "facebook/wav2vec2-base-960h",
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Cache raw frozen features while leaving all trainable projections live."""

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    text_encoder = FrozenTextEncoder(text_checkpoint).to(torch_device).eval()
    audio_encoder = FrozenAudioEncoder(audio_checkpoint).to(torch_device).eval()
    visual_encoder = FrozenVisualEncoder().to(torch_device).eval()
    observation_path = Path(visual_observations).expanduser().resolve()
    observations = {str(value["utterance_id"]): value for value in read_jsonl(observation_path)}
    records = [UtteranceRecord.from_dict(value) for value in read_jsonl(config.dataset.manifest)]
    feature_dir = config.dataset.cache_dir / "features"
    index: list[dict[str, Any]] = []

    for record in records:
        text_available = bool(record.text.strip())
        if text_available:
            text = text_encoder([record.text], torch_device)[0].cpu().numpy().astype(np.float32)
        else:
            text = np.zeros(config.model.text_input_dim, dtype=np.float32)

        audio_path = None if record.audio_path is None else Path(record.audio_path)
        audio_available = audio_path is not None and audio_path.exists()
        if audio_available and audio_path is not None:
            waveform, sample_rate = _load_audio(audio_path)
            audio = (
                audio_encoder(waveform, sample_rate, torch_device)[0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            audio = np.zeros(config.model.audio_input_dim, dtype=np.float32)

        visual, visual_quality, visual_valid, frame_count = _encode_visual_observation(
            observations.get(record.utterance_id),
            observation_path,
            visual_encoder,
            torch_device,
            config.video.min_distinct_valid_frames,
            config.video.quality_epsilon,
        )
        output_path = feature_dir / f"{safe_identifier(record.utterance_id)}.npz"
        _atomic_save_npz(
            output_path,
            text=text,
            text_available=np.asarray(text_available, dtype=np.uint8),
            audio=audio,
            audio_available=np.asarray(audio_available, dtype=np.uint8),
            visual=visual,
            visual_quality=visual_quality,
            visual_valid=visual_valid,
            frame_count=frame_count,
        )
        index.append(
            {
                "utterance_id": record.utterance_id,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
            }
        )
    atomic_write_jsonl(config.dataset.cache_dir / "feature_index.jsonl", index)
    return index
