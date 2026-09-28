"""Paper objective and class-weight construction."""

from __future__ import annotations

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
