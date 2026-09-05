"""Geometry validation for AI suggestions.

Every concrete operation is checked against the authoritative geometry rules
before it may be applied:

* positions must lie inside the physical design bounds (millimetres);
* ADD/MOVE must not violate the configured minimum centre distance
  (``(d_i + d_j)/2 + gap`` — the single spacing definition from
  :mod:`app.core.geometry`, enforced via :mod:`app.core.collision`);
* referenced stone ids must exist.

Invalid operations are *rejected with a reason* — never silently coerced, never
partially applied. The design is only ever mutated by ops that passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.ai.operations import AISuggestion, OpKind, StoneOp
from app.models.design import Design

_EPS = 1e-6


@dataclass(slots=True)
class ValidatedSuggestion:
    """A suggestion split into geometry-valid operations and rejections."""

    suggestion: AISuggestion
    valid_ops: list[StoneOp] = field(default_factory=list)
    rejected: list[tuple[StoneOp, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.valid_ops

    def summary(self) -> str:
        counts = {kind: 0 for kind in OpKind}
        for op in self.valid_ops:
            counts[op.kind] += 1
        parts = [f"{counts[kind]} {kind.value}" for kind in OpKind if counts[kind]]
        base = ", ".join(parts) if parts else "no changes"
        if self.rejected:
            base += f" ({len(self.rejected)} rejected as invalid geometry)"
        return base


def _in_bounds(design: Design, x: float, y: float) -> bool:
    return -_EPS <= x <= design.width_mm + _EPS and -_EPS <= y <= design.height_mm + _EPS


def _would_collide(
    design: Design, gap_mm: float, x: float, y: float, diameter: float,
    ignore_ids: set[int],
) -> bool:
    """True if a stone ``diameter`` at ``(x, y)`` violates spacing vs the design.

    Uses the same uniform-grid spatial index the rest of the app uses
    (:class:`app.core.collision.SpatialHash`), so validation of thousands of
    candidate ops stays O(N).
    """
    from app.core.collision import SpatialHash

    stones = [s for s in design.stones if s.id not in ignore_ids]
    if not stones:
        return False
    pos = np.array([(s.x, s.y) for s in stones], dtype=np.float64)
    dia = np.array([s.diameter for s in stones], dtype=np.float64)
    sh = SpatialHash(pos, float(dia.max()) + gap_mm)
    for j in sh.candidates_near(x, y):
        required = (diameter + dia[j]) / 2.0 + gap_mm
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        if dx * dx + dy * dy < required * required - _EPS:
            return True
    return False


def validate_suggestion(
    suggestion: AISuggestion, design: Design, gap_mm: float
) -> ValidatedSuggestion:
    """Validate every concrete operation; regional ops must be expanded first."""
    result = ValidatedSuggestion(suggestion=suggestion)
    id_set = {s.id for s in design.stones}

    for op in suggestion.operations:
        if op.kind is OpKind.ADD_STONE:
            if op.x is None or op.y is None:
                result.rejected.append((op, "ADD_STONE requires x/y"))
                continue
            diameter = op.diameter if op.diameter else 2.8
            if not _in_bounds(design, op.x, op.y):
                result.rejected.append((op, "position outside the design bounds"))
                continue
            if _would_collide(design, gap_mm, op.x, op.y, diameter, ignore_ids=set()):
                result.rejected.append((op, "would violate the minimum spacing"))
                continue
            result.valid_ops.append(op)

        elif op.kind is OpKind.MOVE_STONE:
            stone = design.get_by_id(op.stone_id) if op.stone_id is not None else None
            if stone is None:
                result.rejected.append((op, f"no stone with id {op.stone_id}"))
                continue
            if op.x is None or op.y is None:
                result.rejected.append((op, "MOVE_STONE requires x/y"))
                continue
            if not _in_bounds(design, op.x, op.y):
                result.rejected.append((op, "target outside the design bounds"))
                continue
            if _would_collide(design, gap_mm, op.x, op.y, stone.diameter,
                              ignore_ids={stone.id}):
                result.rejected.append((op, "target violates the minimum spacing"))
                continue
            result.valid_ops.append(op)

        elif op.kind is OpKind.DELETE_STONE:
            if op.stone_id in id_set:
                result.valid_ops.append(op)
            else:
                result.rejected.append((op, f"no stone with id {op.stone_id}"))

        elif op.kind is OpKind.MOVE_GROUP:
            missing = [sid for sid in op.stone_ids if sid not in id_set]
            if missing:
                result.rejected.append((op, f"unknown stone ids: {missing[:5]}"))
                continue
            ok = True
            for sid in op.stone_ids:
                s = design.get_by_id(sid)
                assert s is not None
                if not _in_bounds(design, s.x + op.dx, s.y + op.dy):
                    result.rejected.append((op, "translated stone leaves the design bounds"))
                    ok = False
                    break
            if not ok:
                continue
            # A rigid translation keeps the group internally valid; each moved
            # stone must clear every stone outside the group.
            group = set(op.stone_ids)
            for sid in op.stone_ids:
                s = design.get_by_id(sid)
                assert s is not None
                if _would_collide(design, gap_mm, s.x + op.dx, s.y + op.dy, s.diameter,
                                  ignore_ids=group):
                    result.rejected.append((op, "translated group violates the minimum spacing"))
                    ok = False
                    break
            if ok:
                result.valid_ops.append(op)

        else:  # ADJUST_DENSITY / OPTIMIZE_REGION — regional, expand first
            result.rejected.append((op, "regional operations must be expanded into "
                                        "concrete operations before validation"))

    return result
