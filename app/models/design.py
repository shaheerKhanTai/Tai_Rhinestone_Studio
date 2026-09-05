
"""The :class:`Design` — the vector-like collection of placed stones.

A design is *not* a raster image. It is the physical geometry of the finished
piece: a set of :class:`~app.models.stone.Stone` objects positioned in
millimetres inside a physical canvas of ``width_mm × height_mm``.

The design owns unique-ID allocation and exposes NumPy projections of the
stone data for the rendering, collision and export layers, which must operate
on arrays rather than Python-level per-stone loops for performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from app.core.geometry import GeometryError, Rect
from app.models.stone import DEFAULT_STONE_COLOR, Stone


@dataclass(slots=True)
class Design:
    """Physical rhinestone layout: a stone list plus the canvas size in mm."""

    width_mm: float
    height_mm: float
    stones: list[Stone] = field(default_factory=list)
    _next_id: int = 1

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise GeometryError(
                f"Design dimensions must be positive, got {self.width_mm}×{self.height_mm} mm"
            )
        # Ensure the id counter is always ahead of any pre-supplied stones.
        if self.stones:
            self._next_id = max(self._next_id, max(s.id for s in self.stones) + 1)

    # -- size --------------------------------------------------------------- #
    @property
    def canvas_rect(self) -> Rect:
        """The physical canvas rectangle in millimetres."""
        return Rect(0.0, 0.0, self.width_mm, self.height_mm)

    def resize(self, width_mm: float, height_mm: float) -> None:
        """Set the physical canvas size (stone positions are left unchanged)."""
        if width_mm <= 0 or height_mm <= 0:
            raise GeometryError("Design dimensions must be positive")
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)

    # -- id allocation ------------------------------------------------------ #
    def allocate_id(self) -> int:
        """Return a fresh unique stone id."""
        new_id = self._next_id
        self._next_id += 1
        return new_id

    # -- mutation ----------------------------------------------------------- #
    def add_stone(
        self,
        x: float,
        y: float,
        diameter: float,
        size_id: str = "SS10",
        color: str = DEFAULT_STONE_COLOR,
    ) -> Stone:
        """Create, append and return a new stone with a freshly allocated id."""
        stone = Stone(self.allocate_id(), float(x), float(y), float(diameter), size_id, color)
        self.stones.append(stone)
        return stone

    def add(self, stone: Stone) -> Stone:
        """Append an existing stone, keeping the id counter consistent."""
        self.stones.append(stone)
        self._next_id = max(self._next_id, stone.id + 1)
        return stone

    def extend(self, stones: Iterable[Stone]) -> None:
        """Append many stones efficiently, keeping the id counter consistent."""
        max_id = self._next_id - 1
        for stone in stones:
            self.stones.append(stone)
            if stone.id > max_id:
                max_id = stone.id
        self._next_id = max_id + 1

    def remove(self, stone: Stone) -> None:
        """Remove a stone by identity (matched on id). Raises if absent."""
        self.remove_by_id(stone.id)

    def remove_by_id(self, stone_id: int) -> None:
        for i, s in enumerate(self.stones):
            if s.id == stone_id:
                del self.stones[i]
                return
        raise KeyError(f"No stone with id {stone_id}")

    def remove_many(self, stone_ids: Iterable[int]) -> list[Stone]:
        """Remove every stone whose id is in ``stone_ids``; return the removed."""
        wanted = set(stone_ids)
        if not wanted:
            return []
        removed = [s for s in self.stones if s.id in wanted]
        if removed:
            self.stones = [s for s in self.stones if s.id not in wanted]
        return removed

    def clear(self) -> None:
        """Remove all stones (the id counter is *not* reset)."""
        self.stones.clear()

    def replace_all(self, stones: list[Stone]) -> None:
        """Replace the entire stone list (used by regeneration)."""
        self.stones = list(stones)
        if self.stones:
            self._next_id = max(self._next_id, max(s.id for s in self.stones) + 1)

    # -- lookup ------------------------------------------------------------- #
    def get_by_id(self, stone_id: int) -> Stone | None:
        for s in self.stones:
            if s.id == stone_id:
                return s
        return None

    @property
    def stone_count(self) -> int:
        return len(self.stones)

    def __len__(self) -> int:
        return len(self.stones)

    def __iter__(self) -> Iterator[Stone]:
        return iter(self.stones)

    # -- selection ---------------------------------------------------------- #
    def selected_stones(self) -> list[Stone]:
        return [s for s in self.stones if s.selected]

    def clear_selection(self) -> None:
        for s in self.stones:
            s.selected = False

    def select_all(self) -> None:
        for s in self.stones:
            s.selected = True

    # -- array projections (for rendering / collision / export) ------------- #
    def positions(self) -> np.ndarray:
        """``(N, 2)`` float array of stone centres in millimetres."""
        if not self.stones:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([(s.x, s.y) for s in self.stones], dtype=np.float64)

    def diameters(self) -> np.ndarray:
        """``(N,)`` float array of stone diameters in millimetres."""
        if not self.stones:
            return np.empty((0,), dtype=np.float64)
        return np.array([s.diameter for s in self.stones], dtype=np.float64)

    def ids(self) -> np.ndarray:
        """``(N,)`` int array of stone ids, index-aligned with the stone list."""
        if not self.stones:
            return np.empty((0,), dtype=np.int64)
        return np.array([s.id for s in self.stones], dtype=np.int64)

    def bounds(self) -> Rect | None:
        """Tight bounding rectangle of stone *edges* (mm), or ``None`` if empty."""
        if not self.stones:
            return None
        pos = self.positions()
        rad = self.diameters() / 2.0
        min_x = float(np.min(pos[:, 0] - rad))
        min_y = float(np.min(pos[:, 1] - rad))
        max_x = float(np.max(pos[:, 0] + rad))
        max_y = float(np.max(pos[:, 1] + rad))
        return Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def color_counts(self) -> dict[str, int]:
        """Map of colour → stone count (for multi-colour reporting)."""
        counts: dict[str, int] = {}
        for s in self.stones:
            counts[s.color] = counts.get(s.color, 0) + 1
        return counts

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "next_id": self._next_id,
            "stones": [s.to_dict() for s in self.stones],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Design":
        stones = [Stone.from_dict(d) for d in data.get("stones", [])]
        design = cls(
            width_mm=float(data["width_mm"]),
            height_mm=float(data["height_mm"]),
            stones=stones,
        )
        # Honour a stored counter if it is ahead of the derived value.
        stored_next = int(data.get("next_id", design._next_id))
        design._next_id = max(design._next_id, stored_next)
        return design
