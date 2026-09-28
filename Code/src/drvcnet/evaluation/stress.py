"""Paired stress evaluation over clean or explicitly corrupted feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from drvcnet.config import ExperimentConfig, load_config
from drvcnet.constants import REGIONS
from drvcnet.data.dataset import DialogueWindowDataset
from drvcnet.io import atomic_write_json, atomic_write_jsonl
from drvcnet.models import DRVCNet
from drvcnet.training.engine import _loader, predict, seed_everything


def _uniform(seed: int, utterance_id: str, region: str, kind: str) -> float:
    payload = f"{seed}|{utterance_id}|{region}|{kind}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


class MissingnessStressDataset(Dataset[dict[str, Any]]):
    """Deterministic paired masks applied only at the target prediction boundary."""

    def __init__(
        self,
        base: DialogueWindowDataset,
        kind: str,
        level: float,
        seed: int,
    ) -> None:
        if kind not in {"current_missingness", "history_missingness"}:
            raise ValueError(f"Unsupported mask stress kind: {kind}")
        if not 0.0 <= level <= 1.0:
            raise ValueError("Missingness level must be a fraction in [0, 1]")
        self.base = base
        self.kind = kind
        self.level = level
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        target = item["target"]
        last = int(item["length"]) - 1
        for region_index, region in enumerate(REGIONS):
            selected = _uniform(self.seed, target.utterance_id, region, self.kind) < self.level
            if not selected:
                continue
            if self.kind == "current_missingness":
                item["visual"][last, region_index] = 0.0
                item["visual_quality"][last, region_index] = 0.0
                item["visual_valid"][last, region_index] = False
            else:
                item["recent"][last, region_index] = 0.0
                item["recent_quality"][last, region_index] = 0.0
                item["recent_mask"][last, region_index] = False
                item["recent_age"][last, region_index] = 0.0
                item["historical"][last, region_index] = 0.0
                item["historical_quality"][last, region_index] = 0.0
                item["historical_mask"][last, region_index] = False
                item["historical_age"][last, region_index] = 0.0
        return item


def evaluate_stress_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    kind: str,
    level: float,
    perturbation_seed: int,
    current_feature_index: Path | None = None,
    reference_feature_index: Path | None = None,
    reference_file: Path | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate one paired condition while keeping clean/reference stores explicit."""

    if kind in {"current_blur", "brow_eye_occlusion", "mouth_occlusion"}:
        if current_feature_index is None:
            raise ValueError(f"{kind} requires --current-feature-index")
    if kind == "reference_blur" and reference_feature_index is None:
        raise ValueError("reference_blur requires --reference-feature-index")
    if kind == "wrong_speaker_detected" and (
        reference_feature_index is None or reference_file is None
    ):
        raise ValueError(
            "wrong_speaker_detected requires --reference-feature-index and --reference-file"
        )
    if kind in {"wrong_speaker_undetected", "reference_intervention"} and reference_file is None:
        raise ValueError(f"{kind} requires --reference-file")
    seed_everything(config.train.seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    base = DialogueWindowDataset(
        config,
        "test",
        current_feature_index=current_feature_index,
        reference_feature_index=reference_feature_index,
        reference_file=reference_file,
    )
    dataset: Dataset[dict[str, Any]]
    if kind in {"current_missingness", "history_missingness"}:
        dataset = MissingnessStressDataset(base, kind, level, perturbation_seed)
    elif kind in {
        "current_blur",
        "reference_blur",
        "brow_eye_occlusion",
        "mouth_occlusion",
        "wrong_speaker_detected",
        "wrong_speaker_undetected",
        "reference_intervention",
    }:
        dataset = base
    else:
        raise ValueError(f"Unknown stress condition: {kind}")
    loader = _loader(dataset, config, shuffle=False)  # type: ignore[arg-type]
    model = DRVCNet(config).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics, predictions = predict(model, loader, config, device)
    for record in predictions:
        record["condition"] = kind
        record["level"] = level
        record["perturbation_seed"] = perturbation_seed
    return {"metrics": metrics, "predictions": predictions}


def stress_test_cli() -> None:
    parser = argparse.ArgumentParser(description="Run a paired DRVC-Net stress condition")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--kind",
        required=True,
        choices=[
            "current_missingness",
            "history_missingness",
            "current_blur",
            "reference_blur",
            "brow_eye_occlusion",
            "mouth_occlusion",
            "wrong_speaker_detected",
            "wrong_speaker_undetected",
            "reference_intervention",
        ],
    )
    parser.add_argument("--level", required=True, type=float)
    parser.add_argument("--perturbation-seed", type=int, required=True)
    parser.add_argument("--current-feature-index", type=Path)
    parser.add_argument("--reference-feature-index", type=Path)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = evaluate_stress_checkpoint(
        load_config(args.config),
        args.checkpoint,
        kind=args.kind,
        level=args.level,
        perturbation_seed=args.perturbation_seed,
        current_feature_index=args.current_feature_index,
        reference_feature_index=args.reference_feature_index,
        reference_file=args.reference_file,
        device_name=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.kind}_{args.level:g}_seed{args.perturbation_seed}"
    atomic_write_jsonl(args.output_dir / f"{stem}.jsonl", result.pop("predictions"))
    atomic_write_json(args.output_dir / f"{stem}.metrics.json", result)
    print(json.dumps(result, indent=2))
