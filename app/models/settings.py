"""Generation settings — the parameters that drive rhinestone placement.

These map the intuitive UI sliders (Detail, Density, Contrast, Smoothing, Edge
Preservation, each ``0.0``–``1.0``) plus the stone size, spacing, placement
mode and optional maximum-stone limit into a single validated value object.
The engine consumes this; the UI never passes raw algorithm parameters around.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

from app.core.geometry import GeometryError, center_to_center
from app.models.stone import resolve_diameter


class PlacementMode(enum.Enum):
    """Supported stone-layout strategies."""

    GRID = "grid"
    OFFSET = "offset"
    HEXAGONAL = "hexagonal"
    ORGANIC = "organic"

    @property
    def label(self) -> str:
        return _PLACEMENT_LABELS[self]

    @classmethod
    def from_str(cls, value: str) -> "PlacementMode":
        key = value.strip().lower()
        for mode in cls:
            if key == mode.value or key == mode.name.lower():
                return mode
        raise GeometryError(f"Unknown placement mode: {value!r}")


_PLACEMENT_LABELS: dict[PlacementMode, str] = {
    PlacementMode.GRID: "Grid",
    PlacementMode.OFFSET: "Offset / Brick",
    PlacementMode.HEXAGONAL: "Hexagonal",
    PlacementMode.ORGANIC: "Organic",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class GenerationSettings:
    """All parameters required to generate a design from a density map.

    The five ``0..1`` sliders are interpreted by the engine:

    * **detail** — how much fine image structure influences placement.
    * **density** — global stone coverage (higher → more stones).
    * **contrast** — steepness of the light/dark threshold in the density map.
    * **smoothing** — pre-blur applied to reduce noise/speckle.
    * **edge_preservation** — how strongly edges are reinforced with stones.
    """

    stone_size_id: str = "SS10"
    custom_diameter_mm: float | None = None
    spacing_mm: float = 0.5
    placement_mode: PlacementMode = PlacementMode.HEXAGONAL
    #: Push All — dragging a stone pushes neighbours away to keep the spacing.
    push_all: bool = False

    detail: float = 0.5
    density: float = 0.5
    contrast: float = 0.5
    smoothing: float = 0.3
    edge_preservation: float = 0.5

    max_stones: int | None = None

    def __post_init__(self) -> None:
        self.detail = _clamp01(self.detail)
        self.density = _clamp01(self.density)
        self.contrast = _clamp01(self.contrast)
        self.smoothing = _clamp01(self.smoothing)
        self.edge_preservation = _clamp01(self.edge_preservation)
        self.validate()

    # -- derived geometry --------------------------------------------------- #
    def effective_diameter(self) -> float:
        """Resolve the stone diameter in millimetres (standard or custom)."""
        return resolve_diameter(self.stone_size_id, self.custom_diameter_mm)

    def center_distance(self) -> float:
        """Centre-to-centre pitch (diameter + gap) in millimetres."""
        return center_to_center(self.effective_diameter(), self.spacing_mm)

    # -- validation --------------------------------------------------------- #
    def validate(self) -> None:
        """Raise :class:`GeometryError` if any value is out of range."""
        if self.spacing_mm < 0:
            raise GeometryError(f"Spacing must not be negative, got {self.spacing_mm}")
        if self.stone_size_id == "Custom" and (
            self.custom_diameter_mm is None or self.custom_diameter_mm <= 0
        ):
            raise GeometryError("Custom stone size requires a positive diameter")
        # Resolves and validates the diameter (raises for unknown size ids).
        diameter = self.effective_diameter()
        if diameter <= 0:
            raise GeometryError(f"Resolved stone diameter must be positive, got {diameter}")
        if self.max_stones is not None and self.max_stones <= 0:
            raise GeometryError(f"Maximum stone count must be positive, got {self.max_stones}")

    def copy(self) -> "GenerationSettings":
        return GenerationSettings(
            stone_size_id=self.stone_size_id,
            custom_diameter_mm=self.custom_diameter_mm,
            spacing_mm=self.spacing_mm,
            placement_mode=self.placement_mode,
            push_all=self.push_all,
            detail=self.detail,
            density=self.density,
            contrast=self.contrast,
            smoothing=self.smoothing,
            edge_preservation=self.edge_preservation,
            max_stones=self.max_stones,
        )

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "stone_size_id": self.stone_size_id,
            "custom_diameter_mm": self.custom_diameter_mm,
            "spacing_mm": self.spacing_mm,
            "placement_mode": self.placement_mode.value,
            "push_all": self.push_all,
            "detail": self.detail,
            "density": self.density,
            "contrast": self.contrast,
            "smoothing": self.smoothing,
            "edge_preservation": self.edge_preservation,
            "max_stones": self.max_stones,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationSettings":
        return cls(
            stone_size_id=str(data.get("stone_size_id", "SS10")),
            custom_diameter_mm=(
                None if data.get("custom_diameter_mm") is None
                else float(data["custom_diameter_mm"])
            ),
            spacing_mm=float(data.get("spacing_mm", 0.5)),
            placement_mode=PlacementMode.from_str(str(data.get("placement_mode", "hexagonal"))),
            push_all=bool(data.get("push_all", False)),
            detail=float(data.get("detail", 0.5)),
            density=float(data.get("density", 0.5)),
            contrast=float(data.get("contrast", 0.5)),
            smoothing=float(data.get("smoothing", 0.3)),
            edge_preservation=float(data.get("edge_preservation", 0.5)),
            max_stones=(None if data.get("max_stones") is None else int(data["max_stones"])),
        )
