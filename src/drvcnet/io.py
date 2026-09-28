"""Deterministic JSONL, hashing, and path helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            yield value


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(target, payload)


def atomic_write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(record), ensure_ascii=False, sort_keys=True) for record in records]
    _atomic_write_text(target, "\n".join(lines) + ("\n" if lines else ""))


def _atomic_write_text(target: Path, payload: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False, newline="\n"
    ) as stream:
        stream.write(payload)
        temp_name = stream.name
    os.replace(temp_name, target)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_records(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def safe_identifier(value: str) -> str:
    """Stable filename component; the suffix prevents normalization collisions."""

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"
    suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:100]}_{suffix}"


def resolve_record_path(record_file: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(record_file).resolve().parent / path).resolve()
