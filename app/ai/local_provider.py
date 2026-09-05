"""The local, offline, rule-based assistant.

IMPORTANT — this is NOT a machine-learning model and never claims to be. It is
a deterministic keyword interpreter that maps a small set of plain-English
design commands onto structured operations, using the same density map the
generator uses. It exists so the AI *architecture* (structured suggestions →
geometry validation → preview → approval → undoable apply) is exercised by a
real, offline provider. Its UI label says exactly what it is.

Everything runs locally: no network, no telemetry, no image upload. Ever.
"""
from __future__ import annotations

import math

import numpy as np

from app.ai.operations import AISuggestion, OpKind, StoneOp
from app.ai.provider import AIProvider, AIRequest

#: Commands this interpreter recognises, with the ops they map to. Ordering
#: matters: the first matching intent wins (deterministic).
_INTENTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("fill", "empty"), "fill"),
    (("more detail", "more detailed", "sharper", "outline", "edges", "edge"), "edges"),
    (("more stones", "add stones", "denser", "add more", "more rhinestones", "increase density"), "densify"),
    (("reduce", "fewer", "less dense", "thin out", "thin", "decrease density", "remove some"), "thin"),
    (("clean", "tidy", "optimize", "optimise", "improve", "arrange", "fix spacing"), "optimize"),
    (("move away", "away from", "clear out", "push away"), "move_away"),
    (("delete", "remove all", "erase"), "delete_region"),
)


class LocalRuleProvider(AIProvider):
    """Deterministic offline command interpreter — rule-based, not learned."""

    label = "Local assistant (rule-based, offline)"

    def is_available(self) -> bool:
        return True  # pure local computation, always available

    def suggest(self, request: AIRequest) -> AISuggestion:
        text = request.command.lower()
        intent = "optimize"  # default: tidy the arrangement
        for keys, name in _INTENTS:
            if any(k in text for k in keys):
                intent = name
                break

        region = request.region
        if intent == "densify":
            return AISuggestion(
                provider=self.label,
                description="Add stones in the highest-density open spots of the region.",
                operations=[StoneOp(OpKind.ADJUST_DENSITY, region=region, direction=1.0)],
            )
        if intent == "thin":
            return AISuggestion(
                provider=self.label,
                description="Remove the lowest-density stones in the region.",
                operations=[StoneOp(OpKind.ADJUST_DENSITY, region=region, direction=-1.0)],
            )
        if intent == "fill":
            return AISuggestion(
                provider=self.label,
                description="Fill the empty region with a regular stone lattice.",
                operations=[StoneOp(OpKind.ADJUST_DENSITY, region=region, direction=1.0)],
            )
        if intent == "edges":
            return AISuggestion(
                provider=self.label,
                description="Reinforce outlines: add stones along the strongest edges.",
                operations=[StoneOp(OpKind.ADJUST_DENSITY, region=region, direction=1.0,
                                    diameter=request.stone_diameter_mm)],
            )
        if intent == "move_away":
            # Concrete per-stone pushes away from the region — the provider has
            # the design geometry, so it can compute valid targets itself.
            ops = self._move_away_ops(request)
            return AISuggestion(
                provider=self.label,
                description="Move stones near the region outward, away from its centre.",
                operations=ops,
            )
        if intent == "delete_region":
            ops = [
                StoneOp(OpKind.DELETE_STONE, stone_id=int(sid))
                for sid in request.stone_ids
                if self._inside(request, sid, region)
            ]
            return AISuggestion(
                provider=self.label,
                description="Delete the stones inside the region.",
                operations=ops,
            )
        return AISuggestion(
            provider=self.label,
            description="Relax the arrangement in the region to restore even spacing.",
            operations=[StoneOp(OpKind.OPTIMIZE_REGION, region=region)],
        )

    # -- deterministic helpers --------------------------------------------- #
    @staticmethod
    def _inside(request: AIRequest, stone_id, region) -> bool:
        matches = np.flatnonzero(request.stone_ids == stone_id)
        if matches.size == 0:
            return False
        x = float(request.stone_positions_mm[matches[0], 0])
        y = float(request.stone_positions_mm[matches[0], 1])
        if region is None:
            return True
        rx, ry, rw, rh = region
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def _move_away_ops(self, request: AIRequest) -> list[StoneOp]:
        """Push every stone within two pitches of the region one pitch outward."""
        if request.region is None:
            return []
        rx, ry, rw, rh = request.region
        cx, cy = rx + rw / 2.0, ry + rh / 2.0
        pitch = request.stone_diameter_mm + request.gap_mm
        ops: list[StoneOp] = []
        for i in range(request.stone_ids.shape[0]):
            x = float(request.stone_positions_mm[i, 0])
            y = float(request.stone_positions_mm[i, 1])
            # Distance to the region rectangle (0 when inside).
            dx = max(rx - x, 0.0, x - (rx + rw))
            dy = max(ry - y, 0.0, y - (ry + rh))
            dist = math.hypot(dx, dy)
            if dist > pitch * 2.0:
                continue
            if dist < 1e-9:
                # Inside the region: push along the vector from the centre.
                ox, oy = x - cx, y - cy
                norm = math.hypot(ox, oy)
                if norm < 1e-9:
                    ox, oy, norm = 1.0, 0.0, 1.0
                ux, uy = ox / norm, oy / norm
            else:
                ux, uy = dx / dist, dy / dist
            target_d = pitch * 2.0
            ops.append(
                StoneOp(
                    OpKind.MOVE_STONE,
                    stone_id=int(request.stone_ids[i]),
                    x=x + ux * target_d,
                    y=y + uy * target_d,
                )
            )
        return ops


def density_at(
    density: np.ndarray | None,
    design_w_mm: float,
    design_h_mm: float,
    xs_mm: np.ndarray,
    ys_mm: np.ndarray,
) -> np.ndarray:
    """Sample the density field (or uniform 1.0 when absent) at mm positions."""
    if density is None or density.size == 0:
        return np.ones(xs_mm.shape[0], dtype=np.float64)
    h, w = density.shape
    ix = np.clip(np.round(xs_mm / design_w_mm * (w - 1)).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(ys_mm / design_h_mm * (h - 1)).astype(np.int64), 0, h - 1)
    return density[iy, ix].astype(np.float64)
