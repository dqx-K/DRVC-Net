"""Paper-aligned optimization, early stopping, and prediction export."""

from __future__ import annotations

import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from drvcnet.config import ExperimentConfig
from drvcnet.data.dataset import DialogueWindowDataset, collate_windows, move_batch_to_device
from drvcnet.evaluation.metrics import classification_metrics
from drvcnet.io import atomic_write_json, atomic_write_jsonl, sha256_file
from drvcnet.models import DRVCNet
from drvcnet.training.losses import class_balanced_weights, weighted_cross_entropy
from drvcnet.training.provenance import collect_provenance
from drvcnet.training.schedule import warmup_cosine_scheduler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_seed(worker_id: int) -> None:
    del worker_id
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


def _loader(
    dataset: DialogueWindowDataset,
    config: ExperimentConfig,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(config.train.seed)
    return DataLoader(
        dataset,
        batch_size=config.train.physical_batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_windows,
        worker_init_fn=_worker_seed,
        generator=generator,
        persistent_workers=config.train.num_workers > 0,
    )


def _class_weights(dataset: DialogueWindowDataset, device: torch.device) -> Tensor:
    counts = Counter(dataset.label_to_index[record.label] for record in dataset.targets)
    ordered = torch.tensor(
        [counts[index] for index in range(len(dataset.label_to_index))], dtype=torch.float32
    )
    return class_balanced_weights(ordered).to(device)


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _target_tensor(values: Tensor, lengths: Tensor) -> Tensor:
    index = lengths.long() - 1
    batch_index = torch.arange(values.shape[0], device=values.device)
    return values[batch_index, index]


@torch.inference_mode()
def predict(
    model: DRVCNet,
    loader: DataLoader[dict[str, Any]],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    truth_indices: list[int] = []
    predicted_indices: list[int] = []
    labels = config.dataset.labels
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        predicted = output.probabilities.argmax(dim=-1)
        truth_indices.extend(int(item) for item in batch["labels"].cpu())
        predicted_indices.extend(int(item) for item in predicted.cpu())
        current_mask = _target_tensor(batch["visual_valid"].bool(), batch["lengths"])
        current_quality = _target_tensor(batch["visual_quality"].float(), batch["lengths"])
        recent_mask = _target_tensor(output.visual.recent_mask, batch["lengths"])
        recent_quality = _target_tensor(batch["recent_quality"].float(), batch["lengths"])
        recent_age = _target_tensor(batch["recent_age"].float(), batch["lengths"])
        historical_mask = _target_tensor(output.visual.historical_mask, batch["lengths"])
        historical_member_mask = _target_tensor(
            batch["historical_mask"].bool(), batch["lengths"]
        )
        historical_member_quality = _target_tensor(
            batch["historical_quality"].float(), batch["lengths"]
        ) * historical_member_mask
        historical_member_age = _target_tensor(
            batch["historical_age"].float(), batch["lengths"]
        )
        historical_weights = historical_member_quality / historical_member_quality.sum(
            dim=-1, keepdim=True
        ).clamp_min(config.video.quality_epsilon)
        historical_quality = (historical_weights * historical_member_quality).sum(dim=-1)
        historical_age = (historical_weights * historical_member_age).sum(dim=-1)
        historical_count = historical_member_mask.sum(dim=-1)
        regional_attention = _target_tensor(output.visual.attention, batch["lengths"])
        recent_gate = _target_tensor(output.visual.recent_gate, batch["lengths"])
        historical_gate = _target_tensor(output.visual.historical_gate, batch["lengths"])
        for index, target in enumerate(batch["targets"]):
            probabilities = output.probabilities[index].detach().cpu().tolist()
            record = {
                    "dataset": config.dataset.name,
                    "fold": config.dataset.outer_fold,
                    "seed": config.train.seed,
                    "model": config.model.variant.value,
                    "dialogue_id": target.dialogue_id,
                    "utterance_id": target.utterance_id,
                    "speaker_id": target.speaker_id,
                    "order": target.order,
                    "y_true": target.label,
                    "y_pred": labels[int(predicted[index].item())],
                    "probabilities": probabilities,
                    "shift_group": batch["shift_groups"][index],
                    "current_mask": current_mask[index].cpu().tolist(),
                    "current_quality": current_quality[index].cpu().tolist(),
                    "recent_mask": recent_mask[index].cpu().tolist(),
                    "recent_quality": recent_quality[index].cpu().tolist(),
                    "recent_age": recent_age[index].cpu().tolist(),
                    "historical_mask": historical_mask[index].cpu().tolist(),
                    "historical_quality": historical_quality[index].cpu().tolist(),
                    "historical_count": historical_count[index].cpu().tolist(),
                    "historical_age": historical_age[index].cpu().tolist(),
                    "reference_ids": batch["reference_ids"][index],
                    "regional_attention": regional_attention[index].cpu().tolist(),
                    "recent_gate": recent_gate[index].cpu().tolist(),
                    "historical_gate": historical_gate[index].cpu().tolist(),
                }
            record.update(batch["reference_annotations"][index])
            predictions.append(record)
    metrics = classification_metrics(truth_indices, predicted_indices, len(labels))
    return metrics, predictions


def train_experiment(
    config: ExperimentConfig,
    device_name: str | None = None,
) -> dict[str, Any]:
    seed_everything(config.train.seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = DialogueWindowDataset(config, "train")
    dev_dataset = DialogueWindowDataset(config, "dev")
    train_loader = _loader(train_dataset, config, shuffle=True)
    dev_loader = _loader(dev_dataset, config, shuffle=False)
    model = DRVCNet(config).to(device)
    weights = _class_weights(train_dataset, device)
    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / config.train.gradient_accumulation_steps)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=steps_per_epoch * config.train.max_epochs,
        warmup_fraction=config.train.warmup_fraction,
    )
    use_amp = config.train.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.output_dir / "provenance.json", collect_provenance(config))
    checkpoint_path = config.output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_wf1 = float("-inf")
    best_mf1 = float("-inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, config.train.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(batch)
                loss = weighted_cross_entropy(output.logits, batch["labels"], weights)
                if output.shift_logits is not None:
                    valid_shift = batch["shift_labels"] >= 0
                    if valid_shift.any():
                        shift_loss = torch.nn.functional.cross_entropy(
                            output.shift_logits[valid_shift], batch["shift_labels"][valid_shift]
                        )
                        loss = loss + config.model.shift_lambda * shift_loss
                scaled_loss = loss / config.train.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            total_loss += float(loss.detach().cpu())
            should_step = (
                step % config.train.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        dev_metrics, _ = predict(model, dev_loader, config, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
        }
        history.append(epoch_record)
        wf1 = dev_metrics["weighted_f1"]
        mf1 = dev_metrics["macro_f1"]
        improved = wf1 > best_wf1 or (math.isclose(wf1, best_wf1) and mf1 > best_mf1)
        if improved:
            best_wf1, best_mf1, best_epoch = wf1, mf1, epoch
            stale_epochs = 0
            _atomic_checkpoint(
                checkpoint_path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "dev_metrics": dev_metrics,
                    "labels": list(config.dataset.labels),
                    "variant": config.model.variant.value,
                    "trainable_parameters": model.trainable_parameter_count(),
                },
            )
        else:
            stale_epochs += 1
        atomic_write_json(config.output_dir / "history.json", history)
        if stale_epochs >= config.train.patience:
            break

    summary = {
        "experiment": config.experiment,
        "best_epoch": best_epoch,
        "best_dev_weighted_f1": best_wf1,
        "best_dev_macro_f1": best_mf1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trainable_parameters": model.trainable_parameter_count(),
        "device": str(device),
        "effective_batch_size": config.train.effective_batch_size,
    }
    atomic_write_json(config.output_dir / "run_summary.json", summary)
    return summary


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    split: str = "test",
    device_name: str | None = None,
) -> dict[str, Any]:
    seed_everything(config.train.seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = DialogueWindowDataset(config, split)
    loader = _loader(dataset, config, shuffle=False)
    model = DRVCNet(config).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    if checkpoint.get("labels") != list(config.dataset.labels):
        raise ValueError("Checkpoint label order does not match configuration")
    if checkpoint.get("variant") != config.model.variant.value:
        raise ValueError("Checkpoint variant does not match configuration")
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics, predictions = predict(model, loader, config, device)
    prediction_path = config.evaluation.prediction_file
    atomic_write_jsonl(prediction_path, predictions)
    payload = {
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "prediction_file": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "metrics": metrics,
    }
    atomic_write_json(prediction_path.with_suffix(".metrics.json"), payload)
    return payload
