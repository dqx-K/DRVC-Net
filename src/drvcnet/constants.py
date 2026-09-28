"""Shared constants and controlled-model identifiers."""

from __future__ import annotations

from enum import Enum

REGIONS: tuple[str, ...] = ("global", "brow", "eye", "mouth")
REGION_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(REGIONS)}

IEMOCAP_LABELS: tuple[str, ...] = (
    "angry",
    "happy",
    "excited",
    "sad",
    "frustrated",
    "neutral",
)
MELD_LABELS: tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)


class ModelVariant(str, Enum):
    """Models and ablations described in the paper."""

    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"
    B3Q = "B3Q"
    B4Q = "B4Q"
    B6 = "B6"
    B7 = "B7"
    N4 = "N4"
    G1 = "G1"
    G2 = "G2"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"

    @property
    def uses_visual(self) -> bool:
        return self is not ModelVariant.B0

    @property
    def active_regions(self) -> tuple[int, ...]:
        if self is ModelVariant.B0:
            return ()
        if self is ModelVariant.B1:
            return (0,)
        return tuple(range(len(REGIONS)))

    @property
    def uses_recent(self) -> bool:
        return self in {
            ModelVariant.B3,
            ModelVariant.B5,
            ModelVariant.B3Q,
            ModelVariant.B6,
            ModelVariant.B7,
            ModelVariant.N4,
            ModelVariant.G1,
            ModelVariant.G2,
            ModelVariant.C2,
            ModelVariant.C3,
        }

    @property
    def uses_historical(self) -> bool:
        return self in {
            ModelVariant.B4,
            ModelVariant.B5,
            ModelVariant.B4Q,
            ModelVariant.B6,
            ModelVariant.B7,
            ModelVariant.N4,
            ModelVariant.G1,
            ModelVariant.G2,
            ModelVariant.C2,
            ModelVariant.C3,
        }

    @property
    def uses_learned_gate(self) -> bool:
        return self in {
            ModelVariant.B3Q,
            ModelVariant.B4Q,
            ModelVariant.B6,
            ModelVariant.B7,
            ModelVariant.N4,
            ModelVariant.G1,
            ModelVariant.C2,
        }
