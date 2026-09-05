"""The AI assistant — orchestrates provider → expansion → validation → preview → apply.

The assistant is the ONLY component that turns AI suggestions into design
changes, and it always goes through the same gate:

    provider.suggest(request)            # structured ops, never code
      → expand regional ops              # deterministic, engine-grade math
      → validate_suggestion(...)         # authoritative geometry rules
      → PREVIEW (user sees ops, applies nothing)
      → user clicks Apply
      → ONE CompositeCommand in history  # single Ctrl+Z reverts everything

Reject never touches the design. Apply never bypasses validation. The geometry
engine (:mod:`app.core.collision` / :mod:`app.core.geometry`) remains the sole
authority on validity — the assistant has no rule of its own.
"""
from __future__ import annotations

import numpy as np

from app.ai.local_provider import density_at
from app.ai.operations import AISuggestion, OpKind, StoneOp
from app.ai.provider import AIProvider, AIRequest
from app.ai.validator import ValidatedSuggestion, validate_suggestion
from app.core.geometry import GeometryError
from app.models.design import Design
from app.models.stone import Stone

#: Upper bound on concrete operations in one suggestion (safety valve).
MAX_OPERATIONS = 5000


class AIAssistant:
    """Facade over the AI layer. Qt-free and unit-testable."""

    def __init__(self, provider: AIProvider | None = None) -> None:
        if provider is None:
            from app.ai.provider import provider_registry

            available = provider_registry.available()
            provider = available[0] if available else None
        self._provider = provider

    @property
    def provider_label(self) -> str:
        return self._provider.label if self._provider else "No provider"

    @property
    def has_provider(self) -> bool:
        return self._provider is not None and self._provider.is_available()

    # ------------------------------------------------------------------ #
    # Suggestion pipeline
    # ------------------------------------------------------------------ #
    def build_request(
        self,
        design: Design,
        command: str,
        *,
        gap_mm: float,
        stone_diameter_mm: float,
        density: np.ndarray | None = None,
        region: tuple[float, float, float, float] | None = None,
    ) -> AIRequest:
        """Assemble an :class:`AIRequest` from the live design (no file paths)."""
        return AIRequest(
            command=command,
            design_width_mm=design.width_mm,
            design_height_mm=design.height_mm,
            stone_ids=design.ids(),
            stone_positions_mm=design.positions(),
            stone_diameters_mm=design.diameters(),
            gap_mm=gap_mm,
            stone_diameter_mm=stone_diameter_mm,
            density=density,
            region=region,
        )

    def suggest(self, request: AIRequest) -> AISuggestion | None:
        if not self.has_provider:
            return None
        return self._provider.suggest(request)

    def preview(
        self,
        design: Design,
        command: str,
        *,
        gap_mm: float,
        stone_diameter_mm: float,
        density: np.ndarray | None = None,
        region: tuple[float, float, float, float] | None = None,
    ) -> ValidatedSuggestion | None:
        """Provider suggestion → expansion → validation. Nothing is applied.

        Returns ``None`` when no provider is available.
        """
        request = self.build_request(
            design, command, gap_mm=gap_mm, stone_diameter_mm=stone_diameter_mm,
            density=density, region=region,
        )
        suggestion = self.suggest(request)
        if suggestion is None:
            return None
        expanded = self.expand_regional_ops(suggestion, request)
        return validate_suggestion(expanded, design, gap_mm)

    # ------------------------------------------------------------------ #
    # Regional-op expansion (deterministic, engine-grade)
    # ------------------------------------------------------------------ #
    def expand_regional_ops(self, suggestion: AISuggestion, request: AIRequest) -> AISuggestion:
        """Replace ADJUST_DENSITY / OPTIMIZE_REGION ops with concrete ones.

        Expansion reuses the generator's own placement rules (lattice at the
        configured pitch, density-weighted acceptance) so suggestions look like
        engine output rather than an invented second layout style.
        """
        out: list[StoneOp] = []
        for op in suggestion.operations:
            if op.kind in (OpKind.ADJUST_DENSITY, OpKind.OPTIMIZE_REGION):
                if op.kind is OpKind.ADJUST_DENSITY:
                    out.extend(self._expand_density(op, request))
                else:
                    out.extend(self._expand_optimize(op, request))
            else:
                out.append(op)
        # Hard cap: a suggestion may never overwhelm the design or the UI.
        if len(out) > MAX_OPERATIONS:
            out = out[:MAX_OPERATIONS]
        return AISuggestion(
            provider=suggestion.provider,
            description=suggestion.description,
            operations=out,
        )

    def _region_rect(self, request: AIRequest, region) -> tuple[float, float, float, float]:
        if region is not None:
            return region
        return (0.0, 0.0, request.design_width_mm, request.design_height_mm)

    def _expand_density(self, op: StoneOp, request: AIRequest) -> list[StoneOp]:
        """Densify (add) or thin (delete) inside the region using the density map."""
        rx, ry, rw, rh = self._region_rect(request, op.region)
        direction = 1.0 if op.direction >= 0 else -1.0
        diameter = op.diameter or request.stone_diameter_mm
        pitch = diameter + request.gap_mm
        if pitch <= 0:
            return []
        if direction > 0:
            # Candidate lattice points inside the region, filtered by density.
            xs = np.arange(rx + pitch / 2.0, rx + rw, pitch)
            ys = np.arange(ry + pitch / 2.0, ry + rh, pitch)
            if xs.size == 0 or ys.size == 0:
                return []
            gx, gy = np.meshgrid(xs, ys)
            cx = gx.ravel()
            cy = gy.ravel()
            dens = density_at(
                request.density, request.design_width_mm, request.design_height_mm, cx, cy
            )
            keep = dens > 0.45
            return [
                StoneOp(OpKind.ADD_STONE, x=float(cx[i]), y=float(cy[i]), diameter=diameter)
                for i in np.flatnonzero(keep)
            ]
        # Thin: delete the lowest-density stones in the region.
        inside = self._stones_in_region(request, (rx, ry, rw, rh))
        if not inside:
            return []
        ids = np.array([sid for sid, _, _ in inside], dtype=np.int64)
        xs = np.array([x for _, x, _ in inside])
        ys = np.array([y for _, _, y in inside])
        dens = density_at(
            request.density, request.design_width_mm, request.design_height_mm, xs, ys
        )
        # Deterministic: sort by (density, id) and drop the faintest third.
        order = np.lexsort((ids, dens))
        n_drop = max(1, len(ids) // 3) if len(ids) else 0
        drop = order[:n_drop]
        return [StoneOp(OpKind.DELETE_STONE, stone_id=int(ids[i])) for i in drop]

    def _expand_optimize(self, op: StoneOp, request: AIRequest) -> list[StoneOp]:
        """Relax collisions in the region (deterministic, same engine call)."""
        from app.core.collision import find_collisions, relax_positions

        rx, ry, rw, rh = self._region_rect(request, op.region)
        inside = self._stones_in_region(request, (rx, ry, rw, rh))
        if len(inside) < 2:
            return []
        # Operate on the full layout but only propose moves for in-region stones.
        positions = request.stone_positions_mm.astype(np.float64, copy=True)
        diameters = request.stone_diameters_mm.astype(np.float64, copy=True)
        pairs = find_collisions(positions, diameters, request.gap_mm)
        if not pairs:
            return []
        relaxed, _converged = relax_positions(
            positions, diameters, request.gap_mm,
            (request.design_width_mm, request.design_height_mm),
        )
        region_set = {sid for sid, _, _ in inside}
        ops: list[StoneOp] = []
        for i in range(request.stone_ids.shape[0]):
            sid = int(request.stone_ids[i])
            if sid not in region_set:
                continue
            ox, oy = positions[i, 0], positions[i, 1]
            nx, ny = relaxed[i, 0], relaxed[i, 1]
            if abs(nx - ox) > 1e-4 or abs(ny - oy) > 1e-4:
                ops.append(StoneOp(OpKind.MOVE_STONE, stone_id=sid, x=float(nx), y=float(ny)))
        return ops

    @staticmethod
    def _stones_in_region(request: AIRequest, region) -> list[tuple[int, float, float]]:
        rx, ry, rw, rh = region
        out = []
        for i in range(request.stone_ids.shape[0]):
            x = float(request.stone_positions_mm[i, 0])
            y = float(request.stone_positions_mm[i, 1])
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                out.append((int(request.stone_ids[i]), x, y))
        return out

    # ------------------------------------------------------------------ #
    # Apply (the ONLY path from suggestion → design)
    # ------------------------------------------------------------------ #
    def apply(self, validated: ValidatedSuggestion, design: Design) -> object | None:
        """Apply validated ops as ONE undoable CompositeCommand.

        Returns the command (for the caller to record via
        ``CommandStack.push_applied``), or ``None`` when there is nothing to do.
        Raises :class:`GeometryError` if the design changed since validation
        made an op impossible (defence in depth).
        """
        from app.core.history import (
            AddStonesCommand,
            CompositeCommand,
            DeleteStonesCommand,
            MoveStonesCommand,
        )

        if validated.is_empty:
            return None

        # Order matters for undo correctness: deletions first (removing stones
        # clears space the moves/adds want), then moves, then adds.
        deletes = [op for op in validated.valid_ops if op.kind is OpKind.DELETE_STONE]
        moves = [op for op in validated.valid_ops if op.kind is OpKind.MOVE_STONE]
        adds = [op for op in validated.valid_ops if op.kind is OpKind.ADD_STONE]
        group_moves = [op for op in validated.valid_ops if op.kind is OpKind.MOVE_GROUP]

        commands = []
        if deletes:
            commands.append(
                DeleteStonesCommand(design, [op.stone_id for op in deletes])
            )
        if moves or group_moves:
            move_map: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
            for op in moves:
                stone = design.get_by_id(op.stone_id) if op.stone_id is not None else None
                if stone is None or op.x is None or op.y is None:
                    raise GeometryError(f"AI move references missing stone {op.stone_id}")
                move_map[stone.id] = ((stone.x, stone.y), (op.x, op.y))
            for op in group_moves:
                for sid in op.stone_ids:
                    stone = design.get_by_id(sid)
                    if stone is None:
                        raise GeometryError(f"AI move references missing stone {sid}")
                    move_map[sid] = (
                        (stone.x, stone.y),
                        (stone.x + op.dx, stone.y + op.dy),
                    )
            if move_map:
                commands.append(MoveStonesCommand(design, move_map))
        if adds:
            stones = []
            for op in adds:
                diameter = op.diameter or 2.8
                stones.append(Stone(design.allocate_id(), op.x, op.y, diameter))
            commands.append(AddStonesCommand(design, stones))

        if not commands:
            return None
        return CompositeCommand(
            commands, label=f"AI: {validated.suggestion.description[:40]}"
        )
