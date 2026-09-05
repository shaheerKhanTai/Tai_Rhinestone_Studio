"""Size presets for common rhinestone applications.

Presets are stored as data (name + physical size + unit), never hard-coded
into UI widgets. The default set below is loaded when no user preset file
exists; :mod:`app.services.settings_service` persists user edits.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.geometry import GeometryError, Unit, to_mm


@dataclass(frozen=True, slots=True)
class SizePreset:
    """A named physical design size, e.g. *T-Shirt Front — 11 × 12 in*."""

    name: str
    width: float
    height: float
    unit: Unit

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise GeometryError(f"Preset {self.name!r} must have positive dimensions")

    @property
    def width_mm(self) -> float:
        return to_mm(self.width, self.unit)

    @property
    def height_mm(self) -> float:
        return to_mm(self.height, self.unit)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "unit": self.unit.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SizePreset":
        return cls(
            name=str(data["name"]),
            width=float(data["width"]),
            height=float(data["height"]),
            unit=Unit.from_str(str(data.get("unit", "in"))),
        )


#: The special preset name signalling "user enters custom dimensions".
CUSTOM_PRESET_NAME = "Custom"

#: Built-in presets. Dimensions are typical print areas and are fully editable.
DEFAULT_PRESETS: tuple[SizePreset, ...] = (
    SizePreset("T-Shirt Front", 11.0, 12.0, Unit.INCH),
    SizePreset("T-Shirt Pocket", 4.0, 4.0, Unit.INCH),
    SizePreset("T-Shirt Back", 12.0, 14.0, Unit.INCH),
    SizePreset("Sleeve", 3.5, 12.0, Unit.INCH),
    SizePreset("Cap", 5.0, 2.25, Unit.INCH),
    SizePreset("Dance Costume", 8.0, 8.0, Unit.INCH),
    SizePreset("Tumbler", 9.0, 4.0, Unit.INCH),
    SizePreset(CUSTOM_PRESET_NAME, 6.0, 6.0, Unit.INCH),
)


def default_presets() -> list[SizePreset]:
    """Return a fresh mutable list of the built-in presets."""
    return list(DEFAULT_PRESETS)
