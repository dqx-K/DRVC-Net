"""Typed configuration loading with path resolution and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from drvcnet.constants import IEMOCAP_LABELS, MELD_LABELS, REGIONS, ModelVariant


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    manifest: Path
    split_file: Path
    cache_dir: Path
    reference_file: Path
    labels: tuple[str, ...]
    outer_fold: int | None = None
    split_seed: int = 2026
    exclude_overlapping_history: bool = False


@dataclass(frozen=True)
class VideoConfig:
    max_frames: int = 8
    regions: tuple[str, ...] = REGIONS
    crop_size: int = 224
    min_distinct_valid_frames: int = 2
    quality_threshold: float = 0.0
    long_history_k: int = 4
    quality_epsilon: float = 1.0e-6


@dataclass(frozen=True)
class ModelConfig:
    variant: ModelVariant = ModelVariant.B6
    text_input_dim: int = 768
    audio_input_dim: int = 768
    visual_input_dim: int = 512
    hidden_dim: int = 256
    role_dim: int = 32
    max_roles: int = 16
    context_layers: int = 2
    context_heads: int = 4
    context_ff_dim: int = 1024
    context_window: int = 32
    classifier_hidden_dim: int = 256
    dropout: float = 0.2
    enabled_modalities: tuple[str, ...] = ("text", "audio", "visual")
    norm_first: bool = True
    c1_hidden_dim: int = 1152
    c2_hidden_dim: int = 342
    c3_ff_dim: int = 640
    shift_lambda: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    physical_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_epochs: int = 40
    patience: int = 5
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 4

    @property
    def effective_batch_size(self) -> int:
        return self.physical_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_repeats: int = 10_000
    permutation_repeats: int = 10_000
    statistical_seed: int = 20_260_926
    prediction_file: Path = Path("outputs/predictions.jsonl")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    dataset: DatasetConfig
    video: VideoConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    output_dir: Path
    source_path: Path = field(compare=False, default=Path("."))


def _path(base: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field '{key}' must be a mapping")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML without executing arbitrary constructors and validate invariants."""

    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Top-level configuration must be a mapping")
    base = source.parent.parent if source.parent.name == "configs" else source.parent

    d = _require_mapping(raw, "dataset")
    v = _require_mapping(raw, "video")
    m = _require_mapping(raw, "model")
    t = _require_mapping(raw, "train")
    e = _require_mapping(raw, "evaluation")

    dataset = DatasetConfig(
        name=str(d["name"]).upper(),
        manifest=_path(base, d["manifest"]),
        split_file=_path(base, d["split_file"]),
        cache_dir=_path(base, d["cache_dir"]),
        reference_file=_path(base, d["reference_file"]),
        labels=tuple(str(item) for item in d["labels"]),
        outer_fold=d.get("outer_fold"),
        split_seed=int(d.get("split_seed", 2026)),
        exclude_overlapping_history=bool(d.get("exclude_overlapping_history", False)),
    )
    video = VideoConfig(
        max_frames=int(v.get("max_frames", 8)),
        regions=tuple(str(item) for item in v.get("regions", REGIONS)),
        crop_size=int(v.get("crop_size", 224)),
        min_distinct_valid_frames=int(v.get("min_distinct_valid_frames", 2)),
        quality_threshold=float(v["quality_threshold"]),
        long_history_k=int(v.get("long_history_k", 4)),
        quality_epsilon=float(v.get("quality_epsilon", 1.0e-6)),
    )
    model = ModelConfig(
        variant=ModelVariant(str(m.get("variant", "B6")).upper()),
        text_input_dim=int(m.get("text_input_dim", 768)),
        audio_input_dim=int(m.get("audio_input_dim", 768)),
        visual_input_dim=int(m.get("visual_input_dim", 512)),
        hidden_dim=int(m.get("hidden_dim", 256)),
        role_dim=int(m.get("role_dim", 32)),
        max_roles=int(m.get("max_roles", 16)),
        context_layers=int(m.get("context_layers", 2)),
        context_heads=int(m.get("context_heads", 4)),
        context_ff_dim=int(m.get("context_ff_dim", 1024)),
        context_window=int(m.get("context_window", 32)),
        classifier_hidden_dim=int(m.get("classifier_hidden_dim", 256)),
        dropout=float(m.get("dropout", 0.2)),
        enabled_modalities=tuple(m.get("enabled_modalities", ["text", "audio", "visual"])),
        norm_first=bool(m.get("norm_first", True)),
        c1_hidden_dim=int(m.get("c1_hidden_dim", 1152)),
        c2_hidden_dim=int(m.get("c2_hidden_dim", 342)),
        c3_ff_dim=int(m.get("c3_ff_dim", 640)),
        shift_lambda=float(m.get("shift_lambda", 0.0)),
    )
    train = TrainConfig(
        seed=int(t.get("seed", 42)),
        lr=float(t.get("lr", 3.0e-4)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        physical_batch_size=int(t.get("physical_batch_size", 16)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 2)),
        max_epochs=int(t.get("max_epochs", 40)),
        patience=int(t.get("patience", 5)),
        warmup_fraction=float(t.get("warmup_fraction", 0.05)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        mixed_precision=bool(t.get("mixed_precision", True)),
        num_workers=int(t.get("num_workers", 4)),
    )
    evaluation = EvaluationConfig(
        bootstrap_repeats=int(e.get("bootstrap_repeats", 10_000)),
        permutation_repeats=int(e.get("permutation_repeats", 10_000)),
        statistical_seed=int(e.get("statistical_seed", 20_260_926)),
        prediction_file=_path(base, e.get("prediction_file", "outputs/predictions.jsonl")),
    )
    config = ExperimentConfig(
        experiment=str(raw.get("experiment", "drvc")),
        dataset=dataset,
        video=video,
        model=model,
        train=train,
        evaluation=evaluation,
        output_dir=_path(base, raw.get("output_dir", "outputs/default")),
        source_path=source,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """Reject settings that violate the declared paper protocol."""

    if config.dataset.name not in {"IEMOCAP", "MELD"}:
        raise ValueError("dataset.name must be IEMOCAP or MELD")
    expected = IEMOCAP_LABELS if config.dataset.name == "IEMOCAP" else MELD_LABELS
    if config.dataset.labels != expected:
        raise ValueError(f"Labels must preserve the fixed {config.dataset.name} order: {expected}")
    if config.video.regions != REGIONS:
        raise ValueError(f"Main protocol requires region order {REGIONS}")
    if config.video.quality_threshold < 0 or config.video.quality_threshold > 1:
        raise ValueError("quality_threshold must be in [0, 1]")
    if config.video.long_history_k < 1:
        raise ValueError("long_history_k must be positive")
    if config.model.context_window < 1:
        raise ValueError("context_window must be positive")
    if config.model.hidden_dim % config.model.context_heads != 0:
        raise ValueError("hidden_dim must be divisible by context_heads")
    if not 0 <= config.model.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if config.model.shift_lambda < 0:
        raise ValueError("shift_lambda must be nonnegative")
    if config.model.variant is ModelVariant.B7 and config.model.shift_lambda <= 0:
        raise ValueError("B7 requires a positive shift_lambda")
    if config.train.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.train.effective_batch_size != 32:
        raise ValueError("The paper protocol requires effective batch size 32")
    unknown_modalities = set(config.model.enabled_modalities) - {"text", "audio", "visual"}
    if unknown_modalities:
        raise ValueError(f"Unknown modalities: {sorted(unknown_modalities)}")
