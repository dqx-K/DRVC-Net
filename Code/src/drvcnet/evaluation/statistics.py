"""Dialogue-cluster bootstrap, paired permutation tests, and Holm correction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from drvcnet.evaluation.metrics import classification_metrics
from drvcnet.evaluation.groups import both_reference_group


def _align(
    first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    key = lambda record: (
        record.get("fold"),
        int(record.get("seed", 0)),
        str(record["utterance_id"]),
    )
    a = {key(record): record for record in first}
    b = {key(record): record for record in second}
    if a.keys() != b.keys():
        only_a = sorted(a.keys() - b.keys())[:5]
        only_b = sorted(b.keys() - a.keys())[:5]
        raise ValueError(f"Prediction supports differ; only first={only_a}, only second={only_b}")
    keys = sorted(a, key=str)
    return [a[item] for item in keys], [b[item] for item in keys]


def _score(records: Sequence[dict[str, Any]], labels: Sequence[str], metric: str) -> float:
    label_to_index = {label: index for index, label in enumerate(labels)}
    values = classification_metrics(
        [label_to_index[str(record["y_true"])] for record in records],
        [label_to_index[str(record["y_pred"])] for record in records],
        len(labels),
    )
    return float(values[metric])


def _resample_dialogues(
    records: Sequence[dict[str, Any]], rng: np.random.Generator, stratify_fold: bool
) -> list[dict[str, Any]]:
    strata: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        fold = str(record.get("fold", "all")) if stratify_fold else "all"
        strata[fold][str(record["dialogue_id"])].append(record)
    sampled: list[dict[str, Any]] = []
    for fold in sorted(strata):
        dialogues = sorted(strata[fold])
        draws = rng.integers(0, len(dialogues), size=len(dialogues))
        for draw_number, index in enumerate(draws):
            # Replicated cluster identities are intentionally retained as rows.
            _ = draw_number
            sampled.extend(strata[fold][dialogues[int(index)]])
    return sampled


def _mean_seed_fold_score(
    records: Sequence[dict[str, Any]], labels: Sequence[str], metric: str
) -> float:
    by_seed_fold: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed_fold[(int(record.get("seed", 0)), str(record.get("fold", "all")))].append(record)
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (seed, _fold), values in by_seed_fold.items():
        by_seed[seed].append(_score(values, labels, metric))
    return float(np.mean([np.mean(values) for values in by_seed.values()]))


def paired_dialogue_bootstrap(
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    metric: str = "weighted_f1",
    repeats: int = 10_000,
    seed: int = 20_260_926,
    stratify_fold: bool = False,
) -> dict[str, float]:
    """Return first-minus-second percentage-point interval with shared draws."""

    aligned_a, aligned_b = _align(first, second)
    paired = list(zip(aligned_a, aligned_b, strict=True))
    clusters: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (record, _) in enumerate(paired):
        fold = str(record.get("fold", "all")) if stratify_fold else "all"
        clusters[(fold, str(record["dialogue_id"]))].append(index)
    folds: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(clusters):
        folds[key[0]].append(key)
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        indices: list[int] = []
        for fold in sorted(folds):
            keys = folds[fold]
            draws = rng.integers(0, len(keys), size=len(keys))
            for draw in draws:
                indices.extend(clusters[keys[int(draw)]])
        sample_a = [aligned_a[index] for index in indices]
        sample_b = [aligned_b[index] for index in indices]
        differences[repeat] = (
            _mean_seed_fold_score(sample_a, labels, metric)
            - _mean_seed_fold_score(sample_b, labels, metric)
        ) * 100.0
    observed = (
        _mean_seed_fold_score(aligned_a, labels, metric)
        - _mean_seed_fold_score(aligned_b, labels, metric)
    ) * 100.0
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {
        "difference": float(observed),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "repeats": repeats,
        "seed": seed,
    }


def paired_permutation_test(
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    metric: str = "weighted_f1",
    repeats: int = 10_000,
    seed: int = 20_260_926,
) -> dict[str, float]:
    """Swap complete paired dialogue predictions and use the +1 correction."""

    aligned_a, aligned_b = _align(first, second)
    cluster_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(aligned_a):
        key = (
            str(record.get("fold", "all")),
            str(record["dialogue_id"]),
        )
        cluster_indices[key].append(index)
    observed = abs(
        _mean_seed_fold_score(aligned_a, labels, metric)
        - _mean_seed_fold_score(aligned_b, labels, metric)
    )
    rng = np.random.default_rng(seed)
    extreme = 0
    keys = sorted(cluster_indices)
    for _ in range(repeats):
        permuted_a = list(aligned_a)
        permuted_b = list(aligned_b)
        swaps = rng.integers(0, 2, size=len(keys)).astype(bool)
        for swap, key in zip(swaps, keys, strict=True):
            if not swap:
                continue
            for index in cluster_indices[key]:
                permuted_a[index], permuted_b[index] = permuted_b[index], permuted_a[index]
        value = abs(
            _mean_seed_fold_score(permuted_a, labels, metric)
            - _mean_seed_fold_score(permuted_b, labels, metric)
        )
        extreme += int(value >= observed)
    return {"difference": observed * 100.0, "p_value": (extreme + 1) / (repeats + 1)}


def holm_correction(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a one-dimensional sequence in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return [float(value) for value in adjusted]


def paired_shift_gain_contrast(
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    population: str = "both",
    metric: str = "weighted_f1",
    repeats: int = 10_000,
    seed: int = 20_260_926,
    stratify_fold: bool = False,
) -> dict[str, float]:
    """Bootstrap (first-second gain on Shift) minus the gain on Stable."""

    if population not in {"all", "both"}:
        raise ValueError("population must be all or both")
    aligned_a, aligned_b = _align(first, second)
    clusters: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(aligned_a):
        fold = str(record.get("fold", "all")) if stratify_fold else "all"
        clusters[(fold, str(record["dialogue_id"]))].append(index)
    folds: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(clusters):
        folds[key[0]].append(key)

    def contrast(indices: Sequence[int]) -> float:
        selected_a = [aligned_a[index] for index in indices]
        selected_b = [aligned_b[index] for index in indices]
        if population == "both":
            keep = [both_reference_group(record) for record in selected_a]
            selected_a = [record for record, flag in zip(selected_a, keep, strict=True) if flag]
            selected_b = [record for record, flag in zip(selected_b, keep, strict=True) if flag]
        shift_a = [record for record in selected_a if record.get("shift_group") == "shift"]
        shift_b = [record for record in selected_b if record.get("shift_group") == "shift"]
        stable_a = [record for record in selected_a if record.get("shift_group") == "stable"]
        stable_b = [record for record in selected_b if record.get("shift_group") == "stable"]
        if not shift_a or not stable_a:
            raise ValueError("Shift and Stable groups must both be nonempty")
        shift_gain = _mean_seed_fold_score(shift_a, labels, metric) - _mean_seed_fold_score(
            shift_b, labels, metric
        )
        stable_gain = _mean_seed_fold_score(stable_a, labels, metric) - _mean_seed_fold_score(
            stable_b, labels, metric
        )
        return (shift_gain - stable_gain) * 100.0

    all_indices = list(range(len(aligned_a)))
    observed = contrast(all_indices)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        indices: list[int] = []
        for fold in sorted(folds):
            keys = folds[fold]
            sampled = rng.integers(0, len(keys), size=len(keys))
            for index in sampled:
                indices.extend(clusters[keys[int(index)]])
        draws[repeat] = contrast(indices)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {
        "contrast": float(observed),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "population": population,
        "repeats": repeats,
        "seed": seed,
    }
