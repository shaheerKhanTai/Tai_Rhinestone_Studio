"""Rhinestone placement engine — density map in, stone positions out.

Given a density map (Phase 2), a :class:`~app.core.geometry.DesignTransform`
mapping density pixels to physical millimetres, and the
:class:`~app.models.settings.GenerationSettings`, this module produces stone
centre positions (mm) for four layouts:

* **grid** — regular rows and columns at the stone pitch;
* **offset / brick** — alternate rows shifted by half a pitch;
* **hexagonal** — densest valid packing (rows shifted, vertical pitch √3/2·p);
* **organic** — a density-weighted, jittered packing that still respects the
  minimum spacing.

Two hard guarantees:

* **Determinism.** The same density map + settings always yield byte-identical
  positions (no global RNG; jitter comes from a coordinate hash).
* **Validity.** No two placed stones are closer than the required pitch, so the
  output never contains overlaps (organic enforces this with a spatial hash;
  the lattices are overlap-free by construction).

Tonal reproduction uses an 8×8 ordered-dither (Bayer) matrix: a stone is placed
at a lattice cell when the local density exceeds the cell's dither threshold, so
coverage is approximately proportional to density (a clean halftone). Solid-dark
regions fill completely; midtones render as a graceful stipple.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.core.geometry import DesignTransform, GeometryError
from app.models.settings import GenerationSettings, PlacementMode

#: ``progress(stage_label, fraction_0_to_1)`` — optional UI progress hook.
ProgressCallback = Callable[[str, float], None]
#: ``is_cancelled() -> bool`` — polled between stages to abort cooperatively.
CancelCheck = Callable[[], bool]
#: ``on_batch(positions_chunk)`` — optional streaming hook called with an
#: ``(K, 2)`` mm array as stones become available during placement (ISSUE 5:
#: progressive display). Batches grow over the run so a huge design produces a
#: bounded number of callbacks, not one per stone.
BatchCallback = Callable[[np.ndarray], None]

#: Batch sizes for streamed placement results: 100 → 250 → 500 → 1000 → 2000.
_STREAM_BATCH_SIZES = (100, 250, 500, 1000, 2000)


class PlacementCancelled(Exception):
    """Raised when generation is cancelled via the :data:`CancelCheck` hook."""


# 8×8 Bayer ordered-dither matrix, normalised to thresholds in ``(0, 1)``.
_BAYER8 = np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ],
    dtype=np.float64,
)
_BAYER_THRESHOLD = (_BAYER8 + 0.5) / 64.0

_EPS = 1e-6


@dataclass(slots=True)
class PlacementResult:
    """Placement output: positions (mm), sampled densities and count metadata."""

    positions_mm: np.ndarray  # (N, 2) float64
    densities: np.ndarray     # (N,) float64 density sampled at each position
    total_before_limit: int   # candidates before the max-stone cap was applied
    pitch_mm: float           # centre-to-centre spacing used

    @property
    def count(self) -> int:
        return int(self.positions_mm.shape[0])

    @property
    def was_limited(self) -> bool:
        return self.total_before_limit > self.count


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _report(progress: ProgressCallback | None, stage: str, fraction: float) -> None:
    if progress is not None:
        progress(stage, max(0.0, min(1.0, fraction)))


def _check_cancel(is_cancelled: CancelCheck | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise PlacementCancelled()


def _stream_batches(
    x: np.ndarray,
    y: np.ndarray,
    on_batch: BatchCallback | None,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
    progress_band: tuple[float, float] = (0.10, 0.55),
    max_emit: int | None = None,
) -> int:
    """Emit ``on_batch`` with growing chunks of ``(x, y)`` positions.

    Returns the number of positions actually emitted. ``progress`` receives a
    fraction inside ``progress_band`` as chunks stream out (monotonic and
    always below the later 0.7 "Checking collisions" stage). Cancellation is
    polled between batches.
    """
    if on_batch is None:
        return 0
    n = x.shape[0]
    if max_emit is not None:
        n = min(n, max_emit)
    lo, hi = progress_band
    emitted = 0
    i = 0
    size_idx = 0
    while i < n:
        _check_cancel(is_cancelled)
        k = min(_STREAM_BATCH_SIZES[size_idx], n - i)
        if size_idx < len(_STREAM_BATCH_SIZES) - 1:
            size_idx += 1
        chunk = np.column_stack([x[i : i + k], y[i : i + k]])
        on_batch(chunk)
        emitted += k
        i += k
        if progress is not None:
            _report(progress, "Placing rhinestones", lo + (hi - lo) * (i / max(1, n)))
    return emitted


def _hash01(a: np.ndarray, b: np.ndarray, salt: int = 0) -> np.ndarray:
    """Deterministic pseudo-random values in ``[0, 1)`` from integer coords.

    A cheap integer hash — no global RNG state — so jittered layouts are
    perfectly reproducible.
    """
    with np.errstate(over="ignore"):
        ha = a.astype(np.uint64) * np.uint64(73856093)
        hb = b.astype(np.uint64) * np.uint64(19349663)
        h = ha ^ hb ^ np.uint64(2654435761 + salt)
        h ^= h >> np.uint64(13)
        h = h * np.uint64(1274126177)
        h ^= h >> np.uint64(16)
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def _sample_density(
    density: np.ndarray, transform: DesignTransform, xs_mm: np.ndarray, ys_mm: np.ndarray
) -> np.ndarray:
    """Nearest-neighbour sample of the density map at physical mm coordinates."""
    h, w = density.shape
    ix = np.clip(np.round(xs_mm / transform.scale_x).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(ys_mm / transform.scale_y).astype(np.int64), 0, h - 1)
    return density[iy, ix]


def _lattice(
    width_mm: float, height_mm: float, pitch: float, vpitch: float, *, stagger: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a (optionally staggered) lattice covering the canvas.

    Returns ``(x, y, col_index, row_index)`` flattened arrays, filtered so every
    centre lies within ``[0, width] × [0, height]``.
    """
    xs = np.arange(pitch / 2.0, width_mm, pitch)
    ys = np.arange(pitch / 2.0, height_mm, vpitch)
    if xs.size == 0 or ys.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty, empty.astype(np.int64), empty.astype(np.int64)

    x_grid, y_grid = np.meshgrid(xs, ys)          # (nr, nc)
    ci, ri = np.meshgrid(np.arange(xs.size), np.arange(ys.size))
    if stagger:
        x_grid = x_grid + (ri % 2) * (pitch / 2.0)

    x = x_grid.ravel()
    y = y_grid.ravel()
    ci = ci.ravel()
    ri = ri.ravel()
    mask = x <= width_mm  # staggered odd rows can push the last column past the edge
    return x[mask], y[mask], ci[mask], ri[mask]


def _bayer_at(col_index: np.ndarray, row_index: np.ndarray) -> np.ndarray:
    """Dither threshold for each lattice cell from its integer indices."""
    return _BAYER_THRESHOLD[row_index % 8, col_index % 8]


def _greedy_min_distance(
    x: np.ndarray,
    y: np.ndarray,
    order: np.ndarray,
    pitch: float,
    on_batch: BatchCallback | None = None,
    max_emit: int | None = None,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> np.ndarray:
    """Greedily accept points (in ``order``) keeping a minimum centre distance.

    Uses a uniform spatial hash of cell size ``pitch`` so each acceptance test
    only inspects the 3×3 neighbourhood — overall ``O(N)``.
    Returns the indices of accepted points, in acceptance order.

    When ``on_batch`` is given, accepted positions are streamed out in growing
    batches *as they are accepted* (genuine incremental work — ISSUE 5), up to
    ``max_emit`` positions.
    """
    cell = pitch
    min_sq = pitch * pitch - _EPS
    grid: dict[tuple[int, int], list[int]] = {}
    ax: list[float] = []
    ay: list[float] = []
    accepted: list[int] = []

    stream = on_batch is not None
    pending_x: list[float] = []
    pending_y: list[float] = []
    emitted = 0
    size_idx = 0
    total = order.size

    def _flush() -> None:
        nonlocal pending_x, pending_y, size_idx, emitted
        if not pending_x:
            return
        on_batch(np.column_stack([np.array(pending_x), np.array(pending_y)]))
        emitted += len(pending_x)
        pending_x, pending_y = [], []
        if size_idx < len(_STREAM_BATCH_SIZES) - 1:
            size_idx += 1

    for oi in range(total):
        if stream and oi % 256 == 0:
            _check_cancel(is_cancelled)
        idx = order[oi]
        px = x[idx]
        py = y[idx]
        cx = int(px // cell)
        cy = int(py // cell)
        ok = True
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                bucket = grid.get((gx, gy))
                if not bucket:
                    continue
                for j in bucket:
                    dx = px - ax[j]
                    dy = py - ay[j]
                    if dx * dx + dy * dy < min_sq:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            j = len(ax)
            ax.append(px)
            ay.append(py)
            grid.setdefault((cx, cy), []).append(j)
            accepted.append(int(idx))
            if stream and (max_emit is None or emitted + len(pending_x) < max_emit):
                pending_x.append(px)
                pending_y.append(py)
                if len(pending_x) >= _STREAM_BATCH_SIZES[size_idx]:
                    _flush()
                    if progress is not None:
                        _report(progress, "Placing rhinestones", 0.10 + 0.45 * (oi / max(1, total)))

    if stream:
        _flush()
    return np.array(accepted, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Placement modes
# --------------------------------------------------------------------------- #


def _place_lattice(
    density: np.ndarray,
    transform: DesignTransform,
    pitch: float,
    mode: PlacementMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid / offset / hexagonal placement via ordered-dither thresholding."""
    if mode is PlacementMode.GRID:
        x, y, ci, ri = _lattice(transform.design_width_mm, transform.design_height_mm,
                                 pitch, pitch, stagger=False)
    elif mode is PlacementMode.OFFSET:
        x, y, ci, ri = _lattice(transform.design_width_mm, transform.design_height_mm,
                                 pitch, pitch, stagger=True)
    elif mode is PlacementMode.HEXAGONAL:
        vpitch = pitch * math.sqrt(3.0) / 2.0
        x, y, ci, ri = _lattice(transform.design_width_mm, transform.design_height_mm,
                                 pitch, vpitch, stagger=True)
    else:  # pragma: no cover - guarded by caller
        raise GeometryError(f"{mode} is not a lattice mode")

    if x.size == 0:
        return np.empty((0, 2)), np.empty((0,))

    dens = _sample_density(density, transform, x, y)
    keep = dens > _bayer_at(ci, ri)
    positions = np.column_stack([x[keep], y[keep]])
    return positions, dens[keep]


def _place_organic(
    density: np.ndarray,
    transform: DesignTransform,
    pitch: float,
    on_batch: BatchCallback | None = None,
    max_emit: int | None = None,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Density-weighted jittered packing that respects the minimum spacing.

    Candidates come from a hex lattice (so solid areas can pack tightly),
    displaced by a coordinate-hash jitter, halftone-filtered by density, then
    accepted greedily in descending-density order under a min-distance rule.
    Fully deterministic.

    With ``on_batch``, accepted positions stream out during the greedy pass
    (up to ``max_emit`` — when a max-stone cap is set the first ``max_emit``
    accepted ARE the stones the cap keeps, since acceptance order is
    descending density).

    The packer runs to completion (no early cap); the maximum-stone limit is
    applied uniformly by :func:`place_stones` afterwards, keeping the count of
    "candidates before the limit" honest and preserving the densest stones.

    Candidates are drawn from a lattice finer than the final pitch (so the
    greedy dart-throwing can pack solid regions almost as tightly as the hex
    layout, rather than leaving large gaps), each displaced by a
    coordinate-hash jitter. This yields a Poisson-disk-like distribution:
    irregular, but a full fill where the image is solid.
    """
    # Finer candidate spacing → the greedy packer has enough options to fill
    # solid areas well while jitter keeps the result irregular.
    cand_pitch = pitch * 0.5
    x, y, ci, ri = _lattice(transform.design_width_mm, transform.design_height_mm,
                            cand_pitch, cand_pitch, stagger=False)
    if x.size == 0:
        return np.empty((0, 2)), np.empty((0,))

    jitter = 0.5 * cand_pitch
    jx = (_hash01(ci, ri, salt=1) - 0.5) * 2.0 * jitter
    jy = (_hash01(ci, ri, salt=2) - 0.5) * 2.0 * jitter
    xj = np.clip(x + jx, 0.0, transform.design_width_mm)
    yj = np.clip(y + jy, 0.0, transform.design_height_mm)

    dens = _sample_density(density, transform, xj, yj)
    keep = dens > _bayer_at(ci, ri)
    xj, yj, dens = xj[keep], yj[keep], dens[keep]
    if xj.size == 0:
        return np.empty((0, 2)), np.empty((0,))

    # Descending density, with index as a deterministic tie-breaker.
    order = np.lexsort((np.arange(dens.size), -dens))
    accepted = _greedy_min_distance(
        xj, yj, order, pitch,
        on_batch=on_batch, max_emit=max_emit,
        progress=progress, is_cancelled=is_cancelled,
    )
    positions = np.column_stack([xj[accepted], yj[accepted]])
    return positions, dens[accepted]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def place_stones(
    density: np.ndarray,
    transform: DesignTransform,
    settings: GenerationSettings,
    *,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
    on_batch: BatchCallback | None = None,
) -> PlacementResult:
    """Place stones over ``density`` and return their physical positions.

    ``density`` must be an ``(H, W)`` ``float``/``float32`` array in ``[0, 1]``
    whose shape matches ``transform``'s image dimensions.

    ``on_batch`` (ISSUE 5) receives position chunks as they become available:
    for organic placement the chunks stream out during the greedy pass; for
    the lattice modes (where placement itself is one vectorised step) the
    final, post-cap positions stream out at the end. Either way the streamed
    positions match the final result's positions.
    """
    if density.ndim != 2:
        raise GeometryError(f"Density map must be 2-D, got shape {density.shape}")
    h, w = density.shape
    if (w, h) != (transform.image_width_px, transform.image_height_px):
        raise GeometryError(
            f"Density shape {(h, w)} does not match transform image size "
            f"{(transform.image_height_px, transform.image_width_px)}"
        )
    settings.validate()
    pitch = settings.center_distance()
    if pitch <= 0:
        raise GeometryError(f"Stone pitch must be positive, got {pitch}")

    density = np.ascontiguousarray(density, dtype=np.float64)

    _report(progress, "Placing stones", 0.1)
    _check_cancel(is_cancelled)

    mode = settings.placement_mode
    if mode is PlacementMode.ORGANIC:
        # With a max-stone cap the first ``max_stones`` accepted (descending
        # density) are exactly the stones the cap keeps, so streaming can stop
        # there — the preview shows precisely the final set.
        positions, densities = _place_organic(
            density, transform, pitch,
            on_batch=on_batch,
            max_emit=settings.max_stones,
            progress=progress,
            is_cancelled=is_cancelled,
        )
        streamed = on_batch is not None
    else:
        positions, densities = _place_lattice(density, transform, pitch, mode)
        streamed = False

    _report(progress, "Checking collisions", 0.7)
    _check_cancel(is_cancelled)

    total_before_limit = int(positions.shape[0])

    # Enforce the maximum stone count, preserving the highest-density stones.
    if settings.max_stones is not None and positions.shape[0] > settings.max_stones:
        _report(progress, "Optimising", 0.85)
        keep = np.argsort(-densities, kind="stable")[: settings.max_stones]
        keep.sort()
        positions = positions[keep]
        densities = densities[keep]

    _check_cancel(is_cancelled)

    # Lattice modes stream the final (post-cap) positions here; organic already
    # streamed during its greedy pass.
    if on_batch is not None and not streamed and positions.shape[0] > 0:
        _stream_batches(
            positions[:, 0], positions[:, 1], on_batch,
            progress=progress, is_cancelled=is_cancelled,
            progress_band=(0.88, 0.98),
        )

    _report(progress, "Finalising", 1.0)

    return PlacementResult(
        positions_mm=positions,
        densities=densities,
        total_before_limit=total_before_limit,
        pitch_mm=pitch,
    )
