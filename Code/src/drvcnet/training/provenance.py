"""Run provenance for hashes, software, hardware, and Git state."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from drvcnet.config import ExperimentConfig
from drvcnet.io import sha256_file


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _git(args: list[str], working_directory: Path) -> str | None:
    process = subprocess.run(
        ["git", *args],
        cwd=working_directory,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def collect_provenance(config: ExperimentConfig) -> dict[str, Any]:
    paths = {
        "config": config.source_path,
        "manifest": config.dataset.manifest,
        "split": config.dataset.split_file,
        "references": config.dataset.reference_file,
        "feature_index": config.dataset.cache_dir / "feature_index.jsonl",
    }
    hashes = {
        name: sha256_file(path) if path.is_file() else None for name, path in paths.items()
    }
    working_directory = config.source_path.parent
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "count": torch.cuda.device_count(),
        }
    return {
        "experiment": config.experiment,
        "resolved_config": _jsonable(asdict(config)),
        "hashes": hashes,
        "git": {
            "commit": _git(["rev-parse", "HEAD"], working_directory),
            "status": _git(["status", "--short"], working_directory),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "hardware": {"gpu": gpu},
    }
