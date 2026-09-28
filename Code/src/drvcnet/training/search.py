"""Equal-budget six-trial development search from the paper protocol."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from drvcnet.config import ExperimentConfig
from drvcnet.io import atomic_write_json
from drvcnet.training.engine import train_experiment


def development_search(
    config: ExperimentConfig,
    *,
    device_name: str | None = None,
    learning_rates: tuple[float, ...] = (3.0e-4, 1.0e-3),
    dropouts: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> dict[str, Any]:
    """Run the full LR-by-dropout grid with search seed 17."""

    root = config.output_dir / "search"
    trials: list[dict[str, Any]] = []
    for learning_rate in learning_rates:
        for dropout in dropouts:
            trial_name = f"lr{learning_rate:g}_dropout{dropout:g}"
            trial = replace(
                config,
                experiment=f"{config.experiment}_{trial_name}",
                model=replace(config.model, dropout=dropout),
                train=replace(config.train, seed=17, lr=learning_rate),
                output_dir=root / trial_name,
            )
            summary = train_experiment(trial, device_name=device_name)
            trials.append(
                {
                    "learning_rate": learning_rate,
                    "dropout": dropout,
                    **summary,
                }
            )
    ranked = sorted(
        trials,
        key=lambda value: (
            float(value["best_dev_weighted_f1"]),
            float(value["best_dev_macro_f1"]),
            -int(value["best_epoch"]),
        ),
        reverse=True,
    )
    result = {
        "selected": {
            "learning_rate": ranked[0]["learning_rate"],
            "dropout": ranked[0]["dropout"],
        },
        "trials": trials,
        "search_seed": 17,
    }
    atomic_write_json(root / "search_summary.json", result)
    return result
