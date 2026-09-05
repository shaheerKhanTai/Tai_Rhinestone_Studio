"""Collision detection and resolution for placed stones.

Two stones are *invalid* when the distance between their centres is less than
the required minimum, ``(d_i + d_j) / 2 + gap`` (which reduces to the pitch
``diameter + gap`` for equal-sized stones). The engine's generated layouts are
collision-free by construction; collisions arise from **manual editing** — the
user dragging or adding a stone into an occupied spot.

This module provides:

* :class:`SpatialHash` — a uniform grid for ``O(1)`` neighbour queries, shared
  by detection, resolution and (later) canvas hit-testing.
* :func:`find_collisions` / :func:`stones_in_collision` — fast reporting and the
  index set the canvas highlights.
* :func:`detect_collisions` — a :class:`Design`-level report (stone *ids*).
* :func:`resolve_collisions` — automatic resolution that first **relaxes**
  overlapping stones apart (preserving every stone and its approximate
  position) and only removes stones as a last resort when the area is
  genuinely over-packed.

All functions are pure/deterministic and free of PySide6.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.models.design import Design

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Spatial hash
# --------------------------------------------------------------------------- #


class SpatialHash:
    """Uniform-grid spatial index over 2-D points for neighbour queries.

    ``cell_size`` must be at least the largest distance two points can be apart
    while still counting as neighbours, so that a 3×3 cell scan is exhaustive.
    """

    __slots__ = ("cell", "_grid")

    def __init__(self, positions: np.ndarray, cell_size: float) -> None:
        self.cell = max(float(cell_size), _EPS)
        self._grid: dict[tuple[int, int], list[int]] = {}
        inv = 1.0 / self.cell
        for i in range(positions.shape[0]):
            cx = int(math.floor(positions[i, 0] * inv))
            cy = int(math.floor(positions[i, 1] * inv))
            self._grid.setdefault((cx, cy), []).append(i)

    def candidates_near(self, x: float, y: float) -> list[int]:
        """Indices of points in the 3×3 block of cells around ``(x, y)``."""
        inv = 1.0 / self.cell
        cx = int(math.floor(x * inv))
        cy = int(math.floor(y * inv))
        out: list[int] = []
        grid = self._grid
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                bucket = grid.get((gx, gy))
                if bucket:
                    out.extend(bucket)
        return out


def _cell_size(diameters: np.ndarray, gap_mm: float) -> float:
    max_d = float(diameters.max()) if diameters.size else 1.0
    return max_d + max(0.0, gap_mm)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def find_collisions(
    positions: np.ndarray, diameters: np.ndarray, gap_mm: float
) -> list[tuple[int, int]]:
    """Return unique index pairs ``(i, j)`` with ``i < j`` that overlap.

    Uses a spatial hash, so cost is ``O(N)`` for typical (locally bounded)
    layouts rather than ``O(N²)``.
    """
    n = positions.shape[0]
    if n < 2:
        return []
    cell = _cell_size(diameters, gap_mm)
    sh = SpatialHash(positions, cell)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        xi, yi = positions[i, 0], positions[i, 1]
        di = diameters[i]
        for j in sh.candidates_near(xi, yi):
            if j <= i:
                continue
            required = (di + diameters[j]) / 2.0 + gap_mm
            dx = xi - positions[j, 0]
            dy = yi - positions[j, 1]
            if dx * dx + dy * dy < required * required - _EPS:
                pairs.append((i, j))
    return pairs


def stones_in_collision(
    positions: np.ndarray, diameters: np.ndarray, gap_mm: float
) -> set[int]:
    """Set of stone *indices* involved in at least one collision."""
    result: set[int] = set()
    for i, j in find_collisions(positions, diameters, gap_mm):
        result.add(i)
        result.add(j)
    return result


def check_position(
    positions: np.ndarray,
    diameters: np.ndarray,
    gap_mm: float,
    x: float,
    y: float,
    diameter: float,
    *,
    ignore_index: int | None = None,
) -> int | None:
    """Return the index of the first stone that would collide with a stone of
    ``diameter`` placed at ``(x, y)``, or ``None`` if the position is valid.

    Intended for interactive validity checks (add/move a single stone).
    """
    n = positions.shape[0]
    if n == 0:
        return None
    cell = _cell_size(diameters, gap_mm)
    sh = SpatialHash(positions, cell)
    for j in sh.candidates_near(x, y):
        if j == ignore_index:
            continue
        required = (diameter + diameters[j]) / 2.0 + gap_mm
        dx = x - positions[j, 0]
        dy = y - positions[j, 1]
        if dx * dx + dy * dy < required * required - _EPS:
            return j
    return None


@dataclass(slots=True)
class CollisionReport:
    """A :class:`Design`-level collision summary in terms of stone ids."""

    pairs: list[tuple[int, int]] = field(default_factory=list)  # id pairs
    colliding_ids: set[int] = field(default_factory=set)
    worst_penetration_mm: float = 0.0

    @property
    def has_collisions(self) -> bool:
        return bool(self.pairs)

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    @property
    def stone_count(self) -> int:
        return len(self.colliding_ids)


def detect_collisions(design: Design, gap_mm: float) -> CollisionReport:
    """Detect collisions in ``design`` and report them by stone id."""
    positions = design.positions()
    diameters = design.diameters()
    ids = design.ids()
    index_pairs = find_collisions(positions, diameters, gap_mm)
    id_pairs: list[tuple[int, int]] = []
    colliding: set[int] = set()
    worst = 0.0
    for i, j in index_pairs:
        id_pairs.append((int(ids[i]), int(ids[j])))
        colliding.add(int(ids[i]))
        colliding.add(int(ids[j]))
        required = (diameters[i] + diameters[j]) / 2.0 + gap_mm
        dist = math.hypot(positions[i, 0] - positions[j, 0], positions[i, 1] - positions[j, 1])
        worst = max(worst, required - dist)
    return CollisionReport(pairs=id_pairs, colliding_ids=colliding, worst_penetration_mm=worst)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def _separation_direction(index: int) -> tuple[float, float]:
    """A deterministic unit vector for separating coincident stones."""
    angle = (index * 2.399963229728653) % (2.0 * math.pi)  # golden angle
    return (math.cos(angle), math.sin(angle))


def relax_positions(
    positions: np.ndarray,
    diameters: np.ndarray,
    gap_mm: float,
    bounds_wh: tuple[float, float] | None = None,
    *,
    max_iterations: int = 60,
) -> tuple[np.ndarray, bool]:
    """Iteratively push overlapping stones apart (a deterministic relaxation).

    Each colliding pair contributes an equal-and-opposite push proportional to
    the penetration; pushes are accumulated per stone and applied together each
    iteration, so the result is order-independent. Positions are clamped to
    ``bounds_wh`` (``(width_mm, height_mm)``) so stones stay on the canvas.
    Returns ``(new_positions, converged)``.
    """
    pos = positions.astype(np.float64, copy=True)
    n = pos.shape[0]
    if n < 2:
        return pos, True

    radii = diameters / 2.0
    for _ in range(max_iterations):
        pairs = find_collisions(pos, diameters, gap_mm)
        if not pairs:
            # No overlaps: return as-is. Any push in a prior iteration was
            # already clamped, and an initially-valid layout is left untouched
            # (we never relocate non-colliding stones just to enforce bounds).
            return pos, True
        disp = np.zeros_like(pos)
        for i, j in pairs:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = math.hypot(dx, dy)
            required = (diameters[i] + diameters[j]) / 2.0 + gap_mm
            if dist < _EPS:
                ux, uy = _separation_direction(i)
                dist = _EPS
            else:
                ux, uy = dx / dist, dy / dist
            pen = required - dist
            if pen <= 0:
                continue
            push = 0.5 * pen
            disp[i, 0] += push * ux
            disp[i, 1] += push * uy
            disp[j, 0] -= push * ux
            disp[j, 1] -= push * uy
        pos += disp
        pos = _clamp(pos, radii, bounds_wh)

    converged = not find_collisions(pos, diameters, gap_mm)
    return pos, converged


def _clamp(
    pos: np.ndarray, radii: np.ndarray, bounds_wh: tuple[float, float] | None
) -> np.ndarray:
    if bounds_wh is None:
        return pos
    w, h = bounds_wh
    lo_x = np.minimum(radii, w / 2.0)
    hi_x = np.maximum(w - radii, w / 2.0)
    lo_y = np.minimum(radii, h / 2.0)
    hi_y = np.maximum(h - radii, h / 2.0)
    pos[:, 0] = np.clip(pos[:, 0], lo_x, hi_x)
    pos[:, 1] = np.clip(pos[:, 1], lo_y, hi_y)
    return pos


def _greedy_remove(
    positions: np.ndarray, diameters: np.ndarray, gap_mm: float
) -> list[int]:
    """Return indices to remove so that no collisions remain (greedy).

    Repeatedly removes the stone participating in the most collisions (ties
    broken by lowest index for determinism).
    """
    removed: list[int] = []
    alive = np.ones(positions.shape[0], dtype=bool)
    while True:
        idx = np.flatnonzero(alive)
        if idx.size < 2:
            break
        pairs = find_collisions(positions[idx], diameters[idx], gap_mm)
        if not pairs:
            break
        counts: Counter[int] = Counter()
        for a, b in pairs:
            counts[int(idx[a])] += 1
            counts[int(idx[b])] += 1
        victim = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        alive[victim] = False
        removed.append(victim)
    return removed


@dataclass(slots=True)
class ResolutionResult:
    """Summary of an automatic resolution pass (for user feedback / undo)."""

    moved_count: int = 0
    removed_ids: list[int] = field(default_factory=list)
    converged: bool = True
    remaining_collisions: int = 0

    @property
    def removed_count(self) -> int:
        return len(self.removed_ids)

    @property
    def changed(self) -> bool:
        return self.moved_count > 0 or bool(self.removed_ids)


def resolve_collisions(
    design: Design,
    gap_mm: float,
    *,
    max_iterations: int = 60,
    allow_removal: bool = True,
    move_epsilon: float = 1e-4,
) -> ResolutionResult:
    """Resolve all collisions in ``design`` in place.

    Strategy: relax overlapping stones apart first (keeps every stone). If the
    region is too crowded to separate within ``max_iterations`` and
    ``allow_removal`` is set, greedily remove the fewest stones needed to clear
    the residual overlaps. Returns a :class:`ResolutionResult`.
    """
    n = design.stone_count
    if n < 2:
        return ResolutionResult(converged=True)

    original = design.positions()
    diameters = design.diameters()
    ids = design.ids()
    bounds = (design.width_mm, design.height_mm)

    relaxed, converged = relax_positions(
        original, diameters, gap_mm, bounds, max_iterations=max_iterations
    )

    # Write relaxed positions back to the stones.
    moved = 0
    for k, stone in enumerate(design.stones):
        nx, ny = float(relaxed[k, 0]), float(relaxed[k, 1])
        if abs(nx - stone.x) > move_epsilon or abs(ny - stone.y) > move_epsilon:
            stone.x, stone.y = nx, ny
            moved += 1

    removed_ids: list[int] = []
    if not converged and allow_removal:
        remove_local = _greedy_remove(relaxed, diameters, gap_mm)
        removed_ids = [int(ids[i]) for i in remove_local]
        if removed_ids:
            design.remove_many(removed_ids)
        converged = not find_collisions(design.positions(), design.diameters(), gap_mm)

    remaining = len(find_collisions(design.positions(), design.diameters(), gap_mm))
    return ResolutionResult(
        moved_count=moved,
        removed_ids=removed_ids,
        converged=converged and remaining == 0,
        remaining_collisions=remaining,
    )
