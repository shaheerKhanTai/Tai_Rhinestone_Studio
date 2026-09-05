"""Structured AI operations — the only language an AI provider may speak.

A provider returns an :class:`AISuggestion`: a plain description plus a list of
:class:`StoneOp` records. Providers can emit *concrete* per-stone operations
(ADD/MOVE/DELETE/MOVE_GROUP) and *regional* generator commands
(ADJUST_DENSITY/OPTIMIZE_REGION) which the assistant expands into concrete
operations deterministically before validation. No arbitrary code, no file
paths, no direct design mutation — suggestions are inert data until the user
approves them.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class OpKind(enum.Enum):
    """The closed set of operations an AI suggestion may contain."""

    ADD_STONE = "add_stone"
    MOVE_STONE = "move_stone"
    DELETE_STONE = "delete_stone"
    MOVE_GROUP = "move_group"
    ADJUST_DENSITY = "adjust_density"
    OPTIMIZE_REGION = "optimize_region"


#: A design-space rectangle in millimetres: ``(x, y, w, h)``.
Region = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class StoneOp:
    """One structured operation. Only the fields relevant to ``kind`` are set.

    * ``ADD_STONE`` — ``x``/``y`` (mm, required), optional ``diameter``.
    * ``MOVE_STONE`` — ``stone_id`` + target ``x``/``y`` (absolute mm).
    * ``DELETE_STONE`` — ``stone_id``.
    * ``MOVE_GROUP`` — ``stone_ids`` + translation ``dx``/``dy`` (mm).
    * ``ADJUST_DENSITY`` — ``region`` + ``direction`` (+1 densify, −1 thin).
    * ``OPTIMIZE_REGION`` — ``region`` (relax collisions / tidy arrangement).
    """

    kind: OpKind
    stone_id: int | None = None
    stone_ids: tuple[int, ...] = ()
    x: float | None = None
    y: float | None = None
    dx: float = 0.0
    dy: float = 0.0
    diameter: float | None = None
    region: Region | None = None
    direction: float = 1.0


@dataclass(slots=True)
class AISuggestion:
    """A provider's complete, inert proposal."""

    provider: str
    description: str
    operations: list[StoneOp] = field(default_factory=list)
