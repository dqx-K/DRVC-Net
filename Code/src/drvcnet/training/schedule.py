"""Five-percent warmup followed by cosine decay."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_fraction: float = 0.05,
) -> LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    warmup_steps = max(1, round(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, factor)
