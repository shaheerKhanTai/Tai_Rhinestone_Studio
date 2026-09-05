"""Geometry and unit-conversion primitives for Rhinestone Studio.

This module is the single source of truth for the distinction between

* **IMAGE SPACE** — integer pixel coordinates of the source raster, and
* **PHYSICAL DESIGN SPACE** — real-world millimetres of the finished design.

Every physical quantity inside the application is stored in **millimetres**.
Only the UI layer converts to and from the user's preferred display unit
(inches / centimetres / millimetres). Exporters consume millimetres and emit
document units appropriate to each format. DPI is *never* used as a stand-in
for physical geometry.

The module has no dependency on PySide6 and is fully unit-testable.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MM_PER_INCH: float = 25.4
MM_PER_CM: float = 10.0

#: Values whose absolute magnitude is below this are treated as zero. Used to
#: guard divisions and equality checks without exposing raw float noise.
EPSILON: float = 1e-9


class GeometryError(ValueError):
    """Raised when a geometric quantity is invalid (e.g. non-positive size)."""


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


class Unit(enum.Enum):
    """A physical length unit understood by the application."""

    MM = "mm"
    CM = "cm"
    INCH = "in"

    @property
    def mm_per_unit(self) -> float:
        """Number of millimetres in one of this unit."""
        return _MM_PER_UNIT[self]

    @property
    def symbol(self) -> str:
        """Short display symbol, e.g. ``"in"``."""
        return self.value

    @property
    def label(self) -> str:
        """Human-readable label for menus and combo boxes."""
        return _UNIT_LABELS[self]

    @classmethod
    def from_str(cls, value: str) -> "Unit":
        """Parse a unit from its symbol or name (case-insensitive)."""
        key = value.strip().lower()
        for unit in cls:
            if key in (unit.value, unit.name.lower()):
                return unit
        raise GeometryError(f"Unknown unit: {value!r}")


_MM_PER_UNIT: dict[Unit, float] = {
    Unit.MM: 1.0,
    Unit.CM: MM_PER_CM,
    Unit.INCH: MM_PER_INCH,
}

_UNIT_LABELS: dict[Unit, str] = {
    Unit.MM: "Millimetres",
    Unit.CM: "Centimetres",
    Unit.INCH: "Inches",
}


def to_mm(value: float, unit: Unit) -> float:
    """Convert ``value`` expressed in ``unit`` to millimetres."""
    return float(value) * unit.mm_per_unit


def from_mm(value_mm: float, unit: Unit) -> float:
    """Convert a millimetre ``value_mm`` to ``unit``."""
    return float(value_mm) / unit.mm_per_unit


def convert(value: float, src: Unit, dst: Unit) -> float:
    """Convert ``value`` from unit ``src`` to unit ``dst``."""
    if src is dst:
        return float(value)
    return from_mm(to_mm(value, src), dst)


# --------------------------------------------------------------------------- #
# Points and rectangles
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Point:
    """An immutable 2-D point. Units are contextual (px or mm) but consistent."""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        """Euclidean distance to ``other``."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def distance_sq_to(self, other: "Point") -> float:
        """Squared Euclidean distance (cheaper; avoids the square root)."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def translated(self, dx: float, dy: float) -> "Point":
        """Return a copy shifted by ``(dx, dy)``."""
        return Point(self.x + dx, self.y + dy)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle defined by its top-left corner and size.

    ``width`` and ``height`` may be negative on construction (useful for
    interactive drag rectangles); call :meth:`normalized` to obtain an
    equivalent rectangle with non-negative extents.
    """

    x: float
    y: float
    width: float
    height: float

    # -- derived corners ---------------------------------------------------- #
    @property
    def left(self) -> float:
        return min(self.x, self.x + self.width)

    @property
    def top(self) -> float:
        return min(self.y, self.y + self.height)

    @property
    def right(self) -> float:
        return max(self.x, self.x + self.width)

    @property
    def bottom(self) -> float:
        return max(self.y, self.y + self.height)

    @property
    def abs_width(self) -> float:
        return abs(self.width)

    @property
    def abs_height(self) -> float:
        return abs(self.height)

    @property
    def area(self) -> float:
        return self.abs_width * self.abs_height

    @property
    def center(self) -> Point:
        return Point((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    def normalized(self) -> "Rect":
        """Return an equivalent rectangle with non-negative width and height."""
        return Rect(self.left, self.top, self.abs_width, self.abs_height)

    def contains(self, point: Point) -> bool:
        """True if ``point`` lies within the rectangle (edges inclusive)."""
        return self.left <= point.x <= self.right and self.top <= point.y <= self.bottom

    def intersects(self, other: "Rect") -> bool:
        """True if this rectangle overlaps ``other`` (touching edges count)."""
        return not (
            self.right < other.left
            or other.right < self.left
            or self.bottom < other.top
            or other.bottom < self.top
        )

    def inflated(self, dx: float, dy: float | None = None) -> "Rect":
        """Return a rectangle grown by ``dx`` (and ``dy``) on every side."""
        if dy is None:
            dy = dx
        n = self.normalized()
        return Rect(n.x - dx, n.y - dy, n.width + 2 * dx, n.height + 2 * dy)

    @classmethod
    def from_points(cls, p1: Point, p2: Point) -> "Rect":
        """Build a normalised rectangle spanning two corner points."""
        return cls(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y).normalized()

    @classmethod
    def from_size(cls, width: float, height: float) -> "Rect":
        """Rectangle anchored at the origin with the given size."""
        return cls(0.0, 0.0, width, height)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)


# --------------------------------------------------------------------------- #
# Spacing / packing helpers
# --------------------------------------------------------------------------- #


def center_to_center(diameter_mm: float, gap_mm: float) -> float:
    """Return the centre-to-centre spacing (pitch) for stones.

    The pitch is the stone diameter plus the requested edge-to-edge gap.
    Two stones are non-overlapping exactly when their centres are at least
    this distance apart.
    """
    if diameter_mm <= 0:
        raise GeometryError(f"Stone diameter must be positive, got {diameter_mm}")
    if gap_mm < 0:
        raise GeometryError(f"Gap must not be negative, got {gap_mm}")
    return diameter_mm + gap_mm


def stones_overlap(
    x1: float, y1: float, d1: float, x2: float, y2: float, d2: float, gap_mm: float = 0.0
) -> bool:
    """True if two stones violate the minimum required separation.

    The minimum centre distance is the mean of the two diameters plus the gap,
    which reduces to ``diameter + gap`` for identically sized stones.
    """
    required = (d1 + d2) / 2.0 + gap_mm
    dx = x1 - x2
    dy = y1 - y2
    return (dx * dx + dy * dy) < (required * required - EPSILON)


# --------------------------------------------------------------------------- #
# Image ↔ design transform
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DesignTransform:
    """Maps source-image pixel coordinates to physical design millimetres.

    A pixel ``(px, py)`` with ``px`` in ``[0, image_width_px]`` and ``py`` in
    ``[0, image_height_px]`` maps linearly onto the physical design rectangle
    ``[0, design_width_mm] × [0, design_height_mm]``. The horizontal and
    vertical scales are independent, so a design whose aspect ratio differs
    from the image is handled correctly (the UI can lock the aspect ratio to
    keep them equal).

    The Y axis points *down* in both spaces, matching image conventions; the
    exporters flip it where a format expects a bottom-left origin.
    """

    image_width_px: int
    image_height_px: int
    design_width_mm: float
    design_height_mm: float

    def __post_init__(self) -> None:
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise GeometryError(
                f"Image dimensions must be positive, got "
                f"{self.image_width_px}×{self.image_height_px} px"
            )
        if self.design_width_mm <= 0 or self.design_height_mm <= 0:
            raise GeometryError(
                f"Design dimensions must be positive, got "
                f"{self.design_width_mm}×{self.design_height_mm} mm"
            )

    @property
    def scale_x(self) -> float:
        """Millimetres per pixel along X."""
        return self.design_width_mm / self.image_width_px

    @property
    def scale_y(self) -> float:
        """Millimetres per pixel along Y."""
        return self.design_height_mm / self.image_height_px

    def image_to_design(self, px: float, py: float) -> Point:
        """Map a single image pixel to a physical design point (mm)."""
        return Point(px * self.scale_x, py * self.scale_y)

    def design_to_image(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Map a physical design point (mm) back to image pixels."""
        return (x_mm / self.scale_x, y_mm / self.scale_y)

    def image_to_design_array(self, points_px: np.ndarray) -> np.ndarray:
        """Vectorised map of an ``(N, 2)`` pixel array to an ``(N, 2)`` mm array."""
        pts = np.asarray(points_px, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise GeometryError(f"Expected an (N, 2) array, got shape {pts.shape}")
        out = np.empty_like(pts)
        out[:, 0] = pts[:, 0] * self.scale_x
        out[:, 1] = pts[:, 1] * self.scale_y
        return out

    @property
    def design_rect_mm(self) -> Rect:
        """The physical design bounds as a millimetre rectangle."""
        return Rect(0.0, 0.0, self.design_width_mm, self.design_height_mm)


def fit_dimensions(
    image_width_px: int,
    image_height_px: int,
    target_width_mm: float,
    *,
    lock_aspect: bool = True,
) -> tuple[float, float]:
    """Compute physical design dimensions from a target width.

    When ``lock_aspect`` is true the returned height preserves the image's
    pixel aspect ratio; otherwise the caller supplies the height separately.
    Useful when the user types a width and expects a proportional height.
    """
    if image_width_px <= 0 or image_height_px <= 0:
        raise GeometryError("Image dimensions must be positive")
    if target_width_mm <= 0:
        raise GeometryError("Target width must be positive")
    if not lock_aspect:
        return (target_width_mm, target_width_mm)
    aspect = image_height_px / image_width_px
    return (target_width_mm, target_width_mm * aspect)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    if low > high:
        low, high = high, low
    return max(low, min(high, value))
