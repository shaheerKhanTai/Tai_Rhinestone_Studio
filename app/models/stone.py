"""Rhinestone stone model and the standard SS (stone-size) table.

Sizes use industry-standard SS designations mapped to representative
diameters in millimetres. Real rhinestones are sold within a small size
*range* per SS; we use a single representative diameter per size, which is
what production workflows expect for spacing calculations.

All stone coordinates are in **physical design millimetres** — never pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.geometry import Point

# --------------------------------------------------------------------------- #
# Stone sizes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StoneSize:
    """A named rhinestone size with a representative diameter in millimetres."""

    name: str
    diameter_mm: float

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0


#: Label used for user-defined custom diameters.
CUSTOM_SIZE_NAME = "Custom"

#: Standard SS sizes with representative diameters (mm). These match the
#: commonly published flat-back rhinestone size chart.
STANDARD_STONE_SIZES: dict[str, StoneSize] = {
    "SS4": StoneSize("SS4", 1.6),
    "SS6": StoneSize("SS6", 2.0),
    "SS8": StoneSize("SS8", 2.4),
    "SS10": StoneSize("SS10", 2.8),
    "SS12": StoneSize("SS12", 3.2),
    "SS16": StoneSize("SS16", 4.0),
    "SS20": StoneSize("SS20", 4.8),
    "SS30": StoneSize("SS30", 6.4),
}

#: Display order for size selectors (smallest to largest).
STONE_SIZE_ORDER: tuple[str, ...] = (
    "SS4",
    "SS6",
    "SS8",
    "SS10",
    "SS12",
    "SS16",
    "SS20",
    "SS30",
)


def get_stone_size(name: str) -> StoneSize:
    """Return the :class:`StoneSize` for a standard SS name.

    Raises :class:`KeyError` for unknown names. Use :func:`make_custom_size`
    for user-defined diameters.
    """
    return STANDARD_STONE_SIZES[name]


def make_custom_size(diameter_mm: float) -> StoneSize:
    """Create a custom :class:`StoneSize` with the given diameter."""
    if diameter_mm <= 0:
        raise ValueError(f"Custom stone diameter must be positive, got {diameter_mm}")
    return StoneSize(CUSTOM_SIZE_NAME, float(diameter_mm))


def resolve_diameter(size_id: str, custom_diameter_mm: float | None = None) -> float:
    """Resolve a size identifier plus optional custom value to a diameter (mm)."""
    if size_id == CUSTOM_SIZE_NAME:
        if custom_diameter_mm is None:
            raise ValueError("Custom size requires an explicit diameter")
        return make_custom_size(custom_diameter_mm).diameter_mm
    return get_stone_size(size_id).diameter_mm


# --------------------------------------------------------------------------- #
# Stone
# --------------------------------------------------------------------------- #

#: Default stone colour (a deep near-black used for single-colour designs).
DEFAULT_STONE_COLOR = "#1F2933"


@dataclass(slots=True)
class Stone:
    """A single placed rhinestone.

    Coordinates ``x``/``y`` and ``diameter`` are physical millimetres. ``id``
    is unique within a :class:`~app.models.design.Design`. ``size_id`` records
    the SS label (or ``"Custom"``) purely for reporting; ``diameter`` is the
    authoritative physical size.
    """

    id: int
    x: float
    y: float
    diameter: float
    size_id: str = "SS10"
    color: str = DEFAULT_STONE_COLOR
    selected: bool = False

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def position(self) -> Point:
        return Point(self.x, self.y)

    def move_to(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def move_by(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def to_dict(self) -> dict:
        """Serialise to a plain JSON-compatible dict (selection is transient)."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "diameter": self.diameter,
            "size_id": self.size_id,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Stone":
        return cls(
            id=int(data["id"]),
            x=float(data["x"]),
            y=float(data["y"]),
            diameter=float(data["diameter"]),
            size_id=str(data.get("size_id", "SS10")),
            color=str(data.get("color", DEFAULT_STONE_COLOR)),
            selected=False,
        )
