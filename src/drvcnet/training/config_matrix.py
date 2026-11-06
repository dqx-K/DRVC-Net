"""Expand the revision protocol into per-dataset, fold, seed, and model configs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _merge(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {source}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _model_label(experiment_name: str) -> str:
    if experiment_name.startswith("bidirectional_"):
        return f"{experiment_name.removeprefix('bidirectional_')}_bidir"
    return experiment_name.replace("FERPlus", "FER").replace("_relative", "_rel")


def build_revision_configs(
    base_configs: dict[str, str | Path],
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    selected_experiments: Iterable[str] | None = None,
) -> list[Path]:
    """Materialize every requested run while keeping search configs separate."""

    protocol = _load_mapping(protocol_path)
    seeds = protocol.get("seed_sets", {})
    datasets = protocol.get("datasets", {})
    experiments = protocol.get("experiments", {})
    if not isinstance(seeds, dict) or not isinstance(datasets, dict) or not isinstance(experiments, dict):
        raise ValueError("Protocol requires seed_sets, datasets, and experiments mappings")
    selected = None if selected_experiments is None else set(selected_experiments)
    unknown = set() if selected is None else selected - experiments.keys()
    if unknown:
        raise ValueError(f"Unknown revision experiments: {sorted(unknown)}")

    root = Path(output_dir)
    written: list[Path] = []
    for experiment_name, specification in experiments.items():
        if selected is not None and experiment_name not in selected:
            continue
        if not isinstance(specification, dict):
            raise ValueError(f"Experiment {experiment_name} must be a mapping")
        seed_key = str(specification.get("seed_set", "final"))
        run_seeds = [int(value) for value in seeds[seed_key]]
        experiment_datasets = [str(value).upper() for value in specification["datasets"]]
        overrides = specification.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Experiment {experiment_name} overrides must be a mapping")
        for dataset_name in experiment_datasets:
            base = _load_mapping(base_configs[dataset_name])
            dataset_spec = datasets[dataset_name]
            folds = dataset_spec.get("folds", [None])
            for fold in folds:
                fold_name = "official" if fold is None else f"fold{int(fold)}"
                for run_seed in run_seeds:
                    config = copy.deepcopy(base)
                    _merge(config, overrides)
                    config["experiment"] = (
                        f"{dataset_name.lower()}_{_model_label(experiment_name)}_"
                        f"{fold_name}_seed{run_seed}"
                    )
                    config["model_label"] = _model_label(experiment_name)
                    config["train"]["seed"] = run_seed
                    if bool(specification.get("feature_cache_tag", False)):
                        config["dataset"]["cache_dir"] = (
                            Path("cache") / dataset_name.lower() / experiment_name
                        ).as_posix()
                    if bool(specification.get("reference_tag", False)):
                        config["dataset"]["reference_file"] = (
                            Path("cache")
                            / dataset_name.lower()
                            / f"references_{experiment_name}.jsonl"
                        ).as_posix()
                    if dataset_name == "IEMOCAP":
                        config["dataset"]["outer_fold"] = int(fold)
                        config["dataset"]["split_file"] = str(
                            dataset_spec["split_template"]
                        ).format(fold=int(fold))
                    else:
                        config["dataset"]["outer_fold"] = None
                        config["dataset"]["split_file"] = str(dataset_spec["split_file"])
                    run_root = (
                        Path("outputs")
                        / "revision"
                        / dataset_name.lower()
                        / experiment_name
                        / f"{fold_name}_seed{run_seed}"
                    )
                    config["output_dir"] = run_root.as_posix()
                    config["evaluation"]["prediction_file"] = (
                        run_root / "predictions.jsonl"
                    ).as_posix()
                    output = (
                        root
                        / dataset_name.lower()
                        / experiment_name
                        / f"{fold_name}_seed{run_seed}.yaml"
                    )
                    _write_yaml(output, config)
                    written.append(output)

                search_config = copy.deepcopy(base)
                _merge(search_config, overrides)
                search_config["experiment"] = (
                    f"{dataset_name.lower()}_{_model_label(experiment_name)}_{fold_name}_search"
                )
                search_config["model_label"] = _model_label(experiment_name)
                search_config["train"]["seed"] = 17
                if bool(specification.get("feature_cache_tag", False)):
                    search_config["dataset"]["cache_dir"] = (
                        Path("cache") / dataset_name.lower() / experiment_name
                    ).as_posix()
                if bool(specification.get("reference_tag", False)):
                    search_config["dataset"]["reference_file"] = (
                        Path("cache")
                        / dataset_name.lower()
                        / f"references_{experiment_name}.jsonl"
                    ).as_posix()
                if dataset_name == "IEMOCAP":
                    search_config["dataset"]["outer_fold"] = int(fold)
                    search_config["dataset"]["split_file"] = str(
                        dataset_spec["split_template"]
                    ).format(fold=int(fold))
                search_root = (
                    Path("outputs")
                    / "revision_search"
                    / dataset_name.lower()
                    / experiment_name
                    / fold_name
                )
                search_config["output_dir"] = search_root.as_posix()
                search_config["evaluation"]["prediction_file"] = (
                    search_root / "predictions.jsonl"
                ).as_posix()
                search_output = (
                    root
                    / "search"
                    / dataset_name.lower()
                    / experiment_name
                    / f"{fold_name}.yaml"
                )
                _write_yaml(search_output, search_config)
                written.append(search_output)
    return written
