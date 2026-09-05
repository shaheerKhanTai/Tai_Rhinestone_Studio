"""Push All — propagate a stone displacement to preserve the configured spacing.

When the user drags a rhinestone into an occupied spot, *Push All* keeps the
moved stone exactly where the user put it and pushes the minimum set of
neighboring stones out of the way — cascading only as far as necessary — so the
layout's minimum centre distance (diameter + gap, per
:func:`app.core.geometry.center_to_center`) is preserved everywhere.

Design constraints (see the Push All spec):

* **The dragged stones are pinned.** They never move off the user's cursor.
* **Only violating stones move.** A stone far from the disturbance is never
  touched — propagation is a breadth-first wave that stops as soon as the
  region is valid again.
* **Deterministic.** Frontier stones are processed in index order and neighbour
  queries are sorted, so the same input always yields the same output.
* **Bounded.** A maximum number of passes and total pushes stop runaway
  propagation; if the layout cannot be resolved within them (e.g. a stone is
  squeezed against the canvas boundary) the result is reported as
  ``converged=False`` and the caller restores the previous valid state.

The implementation reuses :class:`app.core.collision.SpatialHash` conventions
(an incremental uniform grid) and the single spacing definition from
:mod:`app.core.geometry`. Pure NumPy + dict — no PySide6 — so it is fully
unit-testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-9

#: Safety limits so pathological layouts can never spin forever. Passes are
#: cheap (only frontier stones are scanned), so the pass limit is set high
#: enough for a wave to cross a large fully-packed grid; ``max_pushes`` is the
#: real backstop against runaway work.
DEFAULT_MAX_PASSES = 512
DEFAULT_MAX_PUSHES = 10_000

#: Fraction of the design a single Push All wave may move before it is
#: declared unresolvable (Issue 3: "do not move the entire design"). In a
#: zero-slack, fully packed lattice an interior drag would cascade through the
#: whole cluster; that is exactly the case we refuse (restore + notify)
#: instead. The floor keeps tiny designs working: with 5 stones, pushing 4 of
#: them is normal local propagation, not a runaway wave.
DEFAULT_MAX_MOVED_FRACTION = 0.20
_MIN_WAVE_FLOOR = 30


@dataclass(slots=True)
class PushResult:
    """Outcome of a Push All propagation."""

    positions: np.ndarray                 # (N, 2) new positions (a copy)
    pushed_indices: list[int] = field(default_factory=list)  # stones moved by pushing
    converged: bool = True
    passes: int = 0
    total_pushes: int = 0

    @property
    def pushed_count(self) -> int:
        return len(self.pushed_indices)


def propagate_push(
    positions: np.ndarray,
    diameters: np.ndarray,
    gap_mm: float,
    pinned: np.ndarray,
    bounds_wh: tuple[float, float] | None = None,
    *,
    max_passes: int = DEFAULT_MAX_PASSES,
    max_pushes: int = DEFAULT_MAX_PUSHES,
    max_moved_fraction: float = DEFAULT_MAX_MOVED_FRACTION,
) -> PushResult:
    """Push violating neighbours away from the pinned (user-moved) stones.

    ``positions``/``diameters`` describe the current layout (the dragged stones
    already at their tentative new spots). ``pinned`` is a boolean mask that is
    True for the dragged stones; those positions are treated as immovable.
    Returns a :class:`PushResult` with the adjusted positions — the input array
    is never modified.

    ``converged`` is False when the wave could not resolve within the limits —
    because the displacement would cascade through more than
    ``max_moved_fraction`` of the design (a fully packed lattice has no slack,
    and Issue 3 forbids moving the entire design), because it exhausted its
    pass/push budget, or because stones are jammed against the canvas bounds.
    In every such case the caller should discard the result and restore the
    previous valid layout.
    """
    pos = np.array(positions, dtype=np.float64, copy=True)
    n = pos.shape[0]
    result = PushResult(positions=pos)
    if n == 0:
        return result

    wave_limit = max(_MIN_WAVE_FLOOR, int(max_moved_fraction * n))

    dia = np.asarray(diameters, dtype=np.float64)
    pin = np.asarray(pinned, dtype=bool)
    w, h = bounds_wh if bounds_wh is not None else (math.inf, math.inf)
    cell = float(dia.max()) + max(0.0, gap_mm) if dia.size else 1.0
    cell = max(cell, _EPS)
    inv = 1.0 / cell

    # Incremental uniform grid (same cell-size contract as collision.SpatialHash:
    # any two stones that can violate are within one cell of each other).
    grid: dict[tuple[int, int], list[int]] = {}
    stone_cell: list[tuple[int, int]] = [(-1, -1)] * n

    def _key(i: int) -> tuple[int, int]:
        return (int(math.floor(pos[i, 0] * inv)), int(math.floor(pos[i, 1] * inv)))

    for i in range(n):
        k = _key(i)
        stone_cell[i] = k
        grid.setdefault(k, []).append(i)

    def _relocate(i: int, old: tuple[int, int]) -> None:
        new = _key(i)
        if new == old:
            return
        bucket = grid.get(old)
        if bucket is not None:
            bucket.remove(i)
            if not bucket:
                del grid[old]
        grid.setdefault(new, []).append(i)
        stone_cell[i] = new

    def _required(i: int, j: int) -> float:
        return (dia[i] + dia[j]) / 2.0 + gap_mm

    def _violate(i: int, j: int) -> bool:
        req = _required(i, j)
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        return dx * dx + dy * dy < req * req - _EPS

    def _neighbours(i: int) -> list[int]:
        cx, cy = stone_cell[i]
        out: list[int] = []
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                bucket = grid.get((gx, gy))
                if bucket:
                    out.extend(bucket)
        out.sort()  # deterministic
        return out

    def _push_out(mover: int, away_from: int) -> None:
        """Move ``mover`` so its centre is exactly the required distance from
        ``away_from``, along the line joining them (deterministic fallback
        direction for coincident centres)."""
        req = _required(mover, away_from)
        dx = pos[mover, 0] - pos[away_from, 0]
        dy = pos[mover, 1] - pos[away_from, 1]
        dist = math.hypot(dx, dy)
        if dist < _EPS:
            angle = (mover * 2.399963229728653) % (2.0 * math.pi)  # golden angle
            ux, uy = math.cos(angle), math.sin(angle)
        else:
            ux, uy = dx / dist, dy / dist
        nx = pos[away_from, 0] + ux * req
        ny = pos[away_from, 1] + uy * req
        # Clamp the centre to the canvas, matching the canvas drag clamp.
        if bounds_wh is not None:
            nx = min(max(nx, 0.0), w)
            ny = min(max(ny, 0.0), h)
        old = stone_cell[mover]
        pos[mover, 0] = nx
        pos[mover, 1] = ny
        _relocate(mover, old)

    # The wave starts at the pinned stones and stops as soon as no stone in the
    # frontier violates anything.
    frontier: set[int] = {int(i) for i in np.flatnonzero(pin)} if pin.any() else set()
    # When nothing is pinned the caller has no moved stone; nothing to do.
    if not frontier:
        return result

    pushed: set[int] = set()
    passes = 0
    total = 0
    aborted = False

    while frontier and passes < max_passes and total < max_pushes:
        passes += 1
        next_frontier: set[int] = set()
        for i in sorted(frontier):
            for j in _neighbours(i):
                if j == i or not _violate(i, j):
                    continue
                if pin[j]:
                    if pin[i]:
                        continue  # two dragged stones overlapping: user's choice
                    # The neighbour is pinned; move the propagating stone off it.
                    _push_out(i, j)
                    total += 1
                    next_frontier.add(i)
                else:
                    _push_out(j, i)
                    total += 1
                    pushed.add(j)
                    next_frontier.add(j)
        # Wave-size guard: a wave that has grown to move a large fraction of
        # the whole design is a cluster shift, not local propagation. Refuse
        # it (Issue 3: never move the entire design) instead of churning.
        if len(pushed) > wave_limit:
            aborted = True
            break
        frontier = next_frontier

    result.passes = passes
    result.total_pushes = total
    result.pushed_indices = sorted(pushed)

    if aborted:
        result.converged = False
        return result

    # Converged when the wave died out (no stone in the last frontier — or the
    # loop never exhausted its limits). Any residual violation means the layout
    # could not be resolved.
    if frontier:
        result.converged = False
    else:
        # Verify the whole disturbed region is clean (defensive: a push earlier
        # in the final pass may have re-violated a stone already processed).
        check = set(pushed) | {int(i) for i in np.flatnonzero(pin)}
        for i in sorted(check):
            for j in _neighbours(i):
                if j != i and _violate(i, j):
                    if pin[i] and pin[j]:
                        continue  # dragged stones overlapping each other: allowed
                    result.converged = False
                    break
            if not result.converged:
                break
    return result
