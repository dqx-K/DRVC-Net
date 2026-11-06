"""Paper objective and class-weight construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def class_balanced_weights(counts: Tensor) -> Tensor:
    """Square-root inverse-frequency weights, clipped and frequency-normalized."""

    values = counts.to(torch.float64)
    if values.ndim != 1 or torch.any(values <= 0):
        raise ValueError("All fixed-label classes must have positive training support")
    total = values.sum()
    classes = values.numel()
    raw = torch.sqrt(total / (classes * values)).clamp(0.5, 2.0)
    normalizer = (values * raw).sum() / total
    return (raw / normalizer).to(torch.float32)


def weighted_cross_entropy(logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
    return torch.nn.functional.cross_entropy(logits, labels, weight=weights)


def _distillation_kl(student: Tensor, teacher: Tensor) -> Tensor:
    return torch.nn.functional.kl_div(
        torch.log_softmax(student, dim=-1),
        torch.softmax(teacher, dim=-1),
        reduction="batchmean",
    )


def _pareto_weights(
    losses: list[Tensor], shared_parameters: list[torch.nn.Parameter], mix: float
) -> Tensor:
    gradients: list[Tensor] = []
    for loss in losses:
        values = torch.autograd.grad(
            loss, shared_parameters, retain_graph=True, allow_unused=True
        )
        flattened = [value.reshape(-1) for value in values if value is not None]
        if not flattened:
            raise ValueError("CSS PGM objective has no gradient on shared parameters")
        gradient = torch.cat(flattened)
        gradients.append(gradient / gradient.norm().clamp_min(1.0e-8))
    gram = (torch.stack(gradients) @ torch.stack(gradients).t()).detach().cpu().numpy()

    from scipy.optimize import minimize

    count = len(losses)
    initial = np.full(count, 1.0 / count, dtype=np.float64)
    result = minimize(
        lambda value: 0.5 * value.dot(gram.dot(value)),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=[{"type": "eq", "fun": lambda value: float(value.sum() - 1.0)}],
    )
    if not result.success:
        raise RuntimeError(f"CSS PGM solver failed: {result.message}")
    values = torch.as_tensor(result.x, dtype=losses[0].dtype, device=losses[0].device)
    if mix > 0:
        values = (1.0 - mix) * values + mix * torch.full_like(values, 1.0 / count)
    return values


def training_objective(
    model: torch.nn.Module,
    output: Any,
    labels: Tensor,
    class_weights: Tensor,
    architecture: str,
    *,
    distillation_weight: float,
    css_pgm_mix: float,
) -> tuple[Tensor, dict[str, float]]:
    """Return the configured emotion objective and compact component values."""

    primary = weighted_cross_entropy(output.logits, labels, class_weights)
    if architecture in {"drvc", "mmgcn"}:
        return primary, {"emotion_loss": float(primary.detach())}
    if len(output.auxiliary_logits) != 3:
        raise ValueError(f"{architecture.upper()} requires three unimodal prediction heads")
    auxiliary = [
        weighted_cross_entropy(logits, labels, class_weights)
        for logits in output.auxiliary_logits
    ]
    distillation = [_distillation_kl(logits, output.logits) for logits in output.auxiliary_logits]
    auxiliary_mean = torch.stack(auxiliary).mean()
    distillation_mean = torch.stack(distillation).mean()
    if architecture == "sdt":
        loss = primary + distillation_weight * (auxiliary_mean + distillation_mean)
        return loss, {
            "emotion_loss": float(primary.detach()),
            "unimodal_loss": float(auxiliary_mean.detach()),
            "distillation_loss": float(distillation_mean.detach()),
        }
    if architecture != "css" or not hasattr(model, "shared_parameters"):
        raise ValueError(f"Unsupported external architecture: {architecture}")
    tasks = [primary, torch.stack(auxiliary).sum(), torch.stack(distillation).sum()]
    shared = [parameter for parameter in model.shared_parameters() if parameter.requires_grad]
    pgm = _pareto_weights(tasks, shared, css_pgm_mix)
    loss = sum(weight * task for weight, task in zip(pgm, tasks, strict=True))
    return loss, {
        "emotion_loss": float(tasks[0].detach()),
        "unimodal_loss": float(tasks[1].detach()),
        "distillation_loss": float(tasks[2].detach()),
        "pgm_emotion": float(pgm[0].detach()),
        "pgm_unimodal": float(pgm[1].detach()),
        "pgm_distillation": float(pgm[2].detach()),
    }
