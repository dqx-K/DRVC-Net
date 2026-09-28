"""Diagnostic groups derived only after prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def both_reference_group(record: dict[str, Any]) -> bool:
    """Current/recent/historical validity in at least two local regions."""

    current = [bool(value) for value in record["current_mask"]]
    recent = [bool(value) for value in record["recent_mask"]]
    historical = [bool(value) for value in record["historical_mask"]]
    valid_local = sum(
        current[index] and recent[index] and historical[index] for index in (1, 2, 3)
    )
    return valid_local >= 2


def select_group(records: Sequence[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "all":
        return list(records)
    if group == "both":
        return [record for record in records if both_reference_group(record)]
    if group in {"shift", "stable", "no_previous"}:
        return [record for record in records if record.get("shift_group") == group]
    if group.startswith("both_"):
        subgroup = group.removeprefix("both_")
        return [
            record
            for record in records
            if both_reference_group(record) and record.get("shift_group") == subgroup
        ]
    if group == "current_visual_missing":
        return [record for record in records if not any(record["current_mask"])]
    raise ValueError(f"Unknown subgroup: {group}")
