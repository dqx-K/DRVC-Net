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
    include_recent_in_history: bool = False
    visual_backbone: str = "imagenet_resnet18"
    visual_checkpoint: Path | None = None


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "drvc"
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
    active_regions: tuple[str, ...] = REGIONS
    change_operator: tuple[str, ...] = ("signed", "absolute", "product")
    context_encoder: str = "transformer"
    context_direction: str = "causal"
    bidirectional_past: int = 15
    bidirectional_future: int = 15
    norm_first: bool = True
    c1_hidden_dim: int = 1152
    c2_hidden_dim: int = 342
    c3_ff_dim: int = 640
    shift_lambda: float = 0.0
    distillation_weight: float = 0.5
    css_rank: int = 16
    css_polynomial_order: int = 3
    css_pgm_mix: float = 0.0
    mmgcn_layers: int = 64
    mmgcn_alpha: float = 0.1
    mmgcn_lambda: float = 0.5


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
    model_label: str
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


def _project_base(source: Path) -> Path:
    for candidate in source.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return source.parent.parent if source.parent.name == "configs" else source.parent


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML without executing arbitrary constructors and validate invariants."""

    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Top-level configuration must be a mapping")
    base = _project_base(source)

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
        include_recent_in_history=bool(v.get("include_recent_in_history", False)),
        visual_backbone=str(v.get("visual_backbone", "imagenet_resnet18")).lower(),
        visual_checkpoint=(
            None if v.get("visual_checkpoint") is None else _path(base, v["visual_checkpoint"])
        ),
    )
    model = ModelConfig(
        architecture=str(m.get("architecture", "drvc")).lower(),
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
        active_regions=tuple(str(item) for item in m.get("active_regions", REGIONS)),
        change_operator=tuple(
            str(item).lower()
            for item in m.get("change_operator", ["signed", "absolute", "product"])
        ),
        context_encoder=str(m.get("context_encoder", "transformer")).lower(),
        context_direction=str(m.get("context_direction", "causal")).lower(),
        bidirectional_past=int(m.get("bidirectional_past", 15)),
        bidirectional_future=int(m.get("bidirectional_future", 15)),
        norm_first=bool(m.get("norm_first", True)),
        c1_hidden_dim=int(m.get("c1_hidden_dim", 1152)),
        c2_hidden_dim=int(m.get("c2_hidden_dim", 342)),
        c3_ff_dim=int(m.get("c3_ff_dim", 640)),
        shift_lambda=float(m.get("shift_lambda", 0.0)),
        distillation_weight=float(m.get("distillation_weight", 0.5)),
        css_rank=int(m.get("css_rank", 16)),
        css_polynomial_order=int(m.get("css_polynomial_order", 3)),
        css_pgm_mix=float(m.get("css_pgm_mix", 0.0)),
        mmgcn_layers=int(m.get("mmgcn_layers", 64)),
        mmgcn_alpha=float(m.get("mmgcn_alpha", 0.1)),
        mmgcn_lambda=float(m.get("mmgcn_lambda", 0.5)),
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
        model_label=str(
            raw.get(
                "model_label",
                model.variant.value if model.architecture == "drvc" else model.architecture.upper(),
            )
        ),
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
    if config.video.visual_backbone not in {"imagenet_resnet18", "ferplus_resnet18"}:
        raise ValueError("visual_backbone must be imagenet_resnet18 or ferplus_resnet18")
    if config.video.visual_backbone == "ferplus_resnet18" and config.video.visual_checkpoint is None:
        raise ValueError("ferplus_resnet18 requires video.visual_checkpoint")
    if config.model.architecture not in {"drvc", "mmgcn", "sdt", "css"}:
        raise ValueError("model.architecture must be drvc, mmgcn, sdt, or css")
    if config.model.architecture != "drvc" and config.model.context_direction != "causal":
        raise ValueError("External architecture reproductions use causal dialogue access")
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
    unknown_regions = set(config.model.active_regions) - set(REGIONS)
    if unknown_regions:
        raise ValueError(f"Unknown active regions: {sorted(unknown_regions)}")
    if len(set(config.model.active_regions)) != len(config.model.active_regions):
        raise ValueError("active_regions must not contain duplicates")
    allowed_operators = {"signed", "absolute", "product"}
    unknown_operators = set(config.model.change_operator) - allowed_operators
    if unknown_operators or not config.model.change_operator:
        raise ValueError(f"Invalid change_operator values: {sorted(unknown_operators)}")
    if len(set(config.model.change_operator)) != len(config.model.change_operator):
        raise ValueError("change_operator must not contain duplicates")
    if config.model.context_encoder not in {"transformer", "gru"}:
        raise ValueError("context_encoder must be transformer or gru")
    if config.model.context_direction not in {"causal", "bidirectional"}:
        raise ValueError("context_direction must be causal or bidirectional")
    if config.model.context_direction == "bidirectional":
        required = config.model.bidirectional_past + config.model.bidirectional_future + 1
        if config.model.context_window < required:
            raise ValueError("context_window is too short for the bidirectional target window")
        if config.model.context_encoder != "transformer":
            raise ValueError("The bidirectional control uses the Transformer context encoder")
    if config.model.distillation_weight < 0:
        raise ValueError("distillation_weight must be nonnegative")
    if config.model.css_rank < 1 or config.model.css_polynomial_order < 1:
        raise ValueError("CSS rank and polynomial order must be positive")
    if not 0 <= config.model.css_pgm_mix <= 1:
        raise ValueError("css_pgm_mix must be in [0, 1]")
    if config.model.mmgcn_layers < 1:
        raise ValueError("mmgcn_layers must be positive")
    if not 0 < config.model.mmgcn_alpha < 1 or config.model.mmgcn_lambda <= 0:
        raise ValueError("MMGCN alpha must be in (0, 1) and lambda must be positive")
