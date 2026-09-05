"""High-level rhinestone generation: image + settings → placed stones.

:func:`generate` runs the whole pipeline end to end:

    RGBA image → density map → physical transform → stone placement → Stones

It is the single call the UI's background worker (Phase 8) will make. It stays
free of PySide6 and reports progress through a plain callback so it can drive
any progress UI, and honours a cancellation check between stages.

The engine allocates fresh stone ids ``1..N`` for a full regeneration; the
caller replaces the project's design stones wholesale
(:meth:`~app.models.design.Design.replace_all`), so id reuse across regenerations
is expected and safe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.geometry import DesignTransform
from app.core.image_processor import (
    DensityField,
    DensityParams,
    process_to_density,
)
from app.core.placement import (
    CancelCheck,
    PlacementResult,
    ProgressCallback,
    place_stones,
)
from app.models.settings import GenerationSettings
from app.models.stone import DEFAULT_STONE_COLOR, Stone

#: Longest side (px) the density analysis runs at. Larger is slightly more
#: faithful but slower; 1400 keeps regeneration snappy at high fidelity.
DEFAULT_ANALYSIS_MAX_DIM = 1400


@dataclass(slots=True)
class GenerationResult:
    """Everything a caller needs after a generation pass."""

    stones: list[Stone]
    density_field: DensityField
    transform: DesignTransform
    placement: PlacementResult

    @property
    def stone_count(self) -> int:
        return len(self.stones)

    @property
    def was_limited(self) -> bool:
        return self.placement.was_limited

    @property
    def total_before_limit(self) -> int:
        return self.placement.total_before_limit


def density_params_from_settings(settings: GenerationSettings, *, invert: bool = False) -> DensityParams:
    """Translate :class:`GenerationSettings` sliders into :class:`DensityParams`."""
    return DensityParams(
        detail=settings.detail,
        density=settings.density,
        contrast=settings.contrast,
        smoothing=settings.smoothing,
        edge_preservation=settings.edge_preservation,
        invert=invert,
    )


def generate(
    rgba: np.ndarray,
    settings: GenerationSettings,
    width_mm: float,
    height_mm: float,
    *,
    invert: bool = False,
    color: str = DEFAULT_STONE_COLOR,
    analysis_max_dim: int = DEFAULT_ANALYSIS_MAX_DIM,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
    on_batch=None,
) -> GenerationResult:
    """Generate a rhinestone design from an already-transformed RGBA image.

    ``rgba`` must already have the user's crop/rotate/flip applied (see
    :func:`app.core.image_processor.apply_image_transforms`). ``width_mm`` /
    ``height_mm`` are the physical design dimensions the image maps onto.

    ``on_batch`` (ISSUE 5) is called with a list of preview :class:`Stone`
    objects as placement streams them out. The streamed stones are for
    progressive display only — the authoritative result is ``GenerationResult``
    (its ids/order are canonical); a caller displaying batches should discard
    them when the final result commits.
    """
    if progress is not None:
        progress("Analysing image", 0.0)

    params = density_params_from_settings(settings, invert=invert)
    field = process_to_density(rgba, params, max_dim=analysis_max_dim)

    if progress is not None:
        progress("Generating density map", 0.3)
    if is_cancelled is not None and is_cancelled():
        from app.core.placement import PlacementCancelled

        raise PlacementCancelled()

    transform = DesignTransform(field.width, field.height, width_mm, height_mm)

    # Build preview stones as position batches stream out of placement.
    preview_seq = iter(range(1, 10**9))

    def _stone_batch(chunk: np.ndarray) -> None:
        diameter = settings.effective_diameter()
        size_id = settings.stone_size_id
        stones = [
            Stone(next(preview_seq), float(p[0]), float(p[1]), diameter, size_id, color)
            for p in chunk
        ]
        if stones:
            on_batch(stones)

    # Nest placement progress (0.3 → 1.0) beneath the overall pass.
    def _placement_progress(stage: str, frac: float) -> None:
        if progress is not None:
            progress(stage, 0.3 + 0.7 * frac)

    placement = place_stones(
        field.density,
        transform,
        settings,
        progress=_placement_progress,
        is_cancelled=is_cancelled,
        on_batch=_stone_batch if on_batch is not None else None,
    )

    diameter = settings.effective_diameter()
    size_id = settings.stone_size_id
    positions = placement.positions_mm
    stones = [
        Stone(i + 1, float(positions[i, 0]), float(positions[i, 1]), diameter, size_id, color)
        for i in range(positions.shape[0])
    ]

    return GenerationResult(
        stones=stones,
        density_field=field,
        transform=transform,
        placement=placement,
    )
