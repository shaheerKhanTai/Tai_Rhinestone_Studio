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
  the lattices are overlap-free by construction). Every centre also keeps half
  a pitch of margin from the artboard edge, so no stone is clipped by the
  canvas boundary in any mode.

Tonal reproduction uses an 8×8 ordered-dither (Bayer) matrix: a stone is placed
at a lattice cell when the local density exceeds the cell's dither threshold, so
coverage is approximately proportional to density (a clean halftone). Solid-dark
regions fill completely; midtones render as a graceful stipple.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cv2
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


#: Cap (px) on the detail-dilation radius — keeps the max-filter cheap on
#: small designs analysed at high resolution.
_DETAIL_DILATE_MAX_PX = 15

#: Density (of 255) above which a pixel counts as subject when a *uniform*
#: tonal field leaves Otsu nothing to split (see ``_shape_from_density``).
#: Mirrors the density pipeline's tone floor — anything that survived it is a
#: solid shape, not background haze.
_SOLID_TONE_MIN = 20

#: Share of a toned field below which an Otsu split is treated as an artefact
#: of that field rather than a subject against a background.
_SPLIT_MIN_FRACTION = 0.02

#: Peak density at which the artwork (the image's own silhouette) counts as
#: *pale*: white, pastel or flat-coloured artwork whose pixels carry no usable
#: tonal structure. Inside such artwork the density is lifted to a solid fill —
#: a white logo is still a shape, even though nothing in it is dark.
_SUBJECT_PALE_PEAK = 0.5

#: Peak density below which the artwork is not merely pale but toneless: the
#: whole shape takes the stones.
_SUBJECT_FLAT_PEAK = 0.02

#: A pale region *inside* the artwork becomes a hole only when it is both very
#: pale and flat — which is what a logo's white detail (the same white as its
#: background) looks like in the density map. A photo's or a gradient's pale
#: areas are neither, so they stay part of the filled shape.
_DETAIL_PALE_MAX = 0.15
_DETAIL_PALE_SPREAD = 0.08

#: Bounds on the share of the artwork that such a light detail may cover before
#: the split is treated as noise rather than structure.
_DETAIL_HOLE_MIN = 0.02
_DETAIL_HOLE_MAX = 0.60

#: Polygon-simplification tolerance (px) used to turn a traced contour into a
#: polyline. It also bounds how far a stone laid on that contour can sit from
#: the mask it came from — the trace gate is inset by that much before the
#: contour is cut along it.
_CONTOUR_EPS_PX = 2.0


def _detail_keep_map(
    density: np.ndarray, transform: DesignTransform, pitch: float
) -> np.ndarray:
    """Dilate the density map by ~half a stone diameter for the *keep* test.

    A lattice cell decides "stone or no stone" by sampling density at a single
    point — its centre. A stroke thinner than the pitch can slip between two
    rows of centres and vanish from the design entirely, even at full density.
    Testing against the dilated map asks instead "does the stone's disc touch
    any part of the feature?", so thin lines and small details always catch a
    contiguous row of stones. Solid regions are untouched (already at 1.0);
    shapes only gain at most half a stone of boldness at their silhouette.

    The returned map is used **only** for the dither threshold; ranking,
    density reporting and the organic ordering still sample the true map.
    """
    h, w = density.shape
    if w <= 0 or h <= 0 or transform.design_width_mm <= 0:
        return density
    px_per_mm = w / transform.design_width_mm
    radius_px = int(round(0.5 * pitch * px_per_mm))
    if radius_px < 1:
        return density
    radius_px = min(radius_px, _DETAIL_DILATE_MAX_PX)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
    )
    return cv2.dilate(density, kernel)


def _apply_subject(
    density: np.ndarray,
    subject: np.ndarray,
    gate: np.ndarray | None,
) -> np.ndarray:
    """Restrict ``density`` to the artwork silhouette and make it visible there.

    ``subject`` is the image's own outline (1 inside the artwork, 0 outside —
    see :func:`~app.core.image_processor.subject_mask_from_alpha`), and it does
    two jobs the density alone cannot:

    * **Containment** — outside the silhouette there is no artwork, so nothing
      may be placed there. Stray density (background haze, JPEG ringing, a
      pale-but-nonzero scan) stops sprinkling stones across the canvas.
    * **Visibility** — the silhouette *is* the artwork even where its pixels
      are barely darker than the background. A white, pastel or flat-coloured
      logo has almost no density anywhere, so a tone-driven halftone renders
      it as nothing at all; the user's mark and its stroke simply vanish. The
      artwork's own tones are therefore rescaled inside it, so the darkest
      part of the shape reads as solid: a flat shape fills completely, a
      shaded one keeps its relative shading.

    With a ``gate`` (the traced region) both jobs are confined to it, and the
    rule above is applied more strongly: tracing is the user saying "put
    rhinestones here", so artwork inside the trace that carries no usable tone
    is *filled* rather than halftoned at a trickle. Artwork that does carry
    tone keeps following the image's shading inside the trace, as before.
    """
    if subject.shape != density.shape:
        subject = cv2.resize(
            subject, (density.shape[1], density.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    artwork = subject > 0.5
    region = artwork & (gate > 0.5) if gate is not None else artwork
    if region.any():
        peak = float(np.percentile(density[region], 99.0))
        if gate is not None:
            if peak < _SUBJECT_PALE_PEAK:
                density = np.where(region, 1.0, density)
        elif peak < _SUBJECT_FLAT_PEAK:
            # Toneless artwork (flat white/pastel): the shape is all there is.
            density = np.where(region, 1.0, density)
        elif peak < _SUBJECT_PALE_PEAK:
            density = np.where(region, np.clip(density / peak, 0.0, 1.0), density)
    # Containment last, so the silhouette is the hard edge of the result.
    return density * subject


def _lattice(
    width_mm: float,
    height_mm: float,
    pitch: float,
    vpitch: float,
    *,
    stagger: bool,
    inset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a (optionally staggered) lattice covering the canvas.

    Returns ``(x, y, col_index, row_index)`` flattened arrays, filtered so every
    centre lies within ``[inset, width - inset] × [inset, height - inset]``.

    ``inset`` is the margin a centre must keep from the artboard edge. Callers
    pass half a pitch, which is the smallest value that guarantees **no stone
    is clipped by the canvas**: a centre at least ``pitch/2 = (diameter + gap)/2``
    from the edge leaves at least ``gap/2`` of clearance beyond the stone's own
    radius. Without it, the last column of a lattice whose pitch does not divide
    the canvas evenly could land within one stone radius of the edge.
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
    # Staggered odd rows can push the last column past the edge, and the final
    # column/row of any lattice can sit closer to the edge than one radius.
    lo_x, hi_x = inset, width_mm - inset
    lo_y, hi_y = inset, height_mm - inset
    mask = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
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
    keep_density: np.ndarray,
    transform: DesignTransform,
    pitch: float,
    mode: PlacementMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid / offset / hexagonal placement via ordered-dither thresholding."""
    # Half a pitch of margin keeps every stone fully inside the artboard.
    inset = pitch / 2.0
    if mode is PlacementMode.GRID:
        x, y, ci, ri = _lattice(transform.design_width_mm, transform.design_height_mm,
                                 pitch, pitch, stagger=False, inset=inset)
    elif mode is PlacementMode.OFFSET:
        x, y, ci, ri = _lattice(transform.design_width_mm, transform.design_height_mm,
                                 pitch, pitch, stagger=True, inset=inset)
    elif mode is PlacementMode.HEXAGONAL:
        vpitch = pitch * math.sqrt(3.0) / 2.0
        x, y, ci, ri = _lattice(transform.design_width_mm, transform.design_height_mm,
                                 pitch, vpitch, stagger=True, inset=inset)
    else:  # pragma: no cover - guarded by caller
        raise GeometryError(f"{mode} is not a lattice mode")

    if x.size == 0:
        return np.empty((0, 2)), np.empty((0,))

    dens = _sample_density(density, transform, x, y)
    keep = _sample_density(keep_density, transform, x, y) > _bayer_at(ci, ri)
    positions = np.column_stack([x[keep], y[keep]])
    return positions, dens[keep]


def _place_organic(
    density: np.ndarray,
    keep_density: np.ndarray,
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
    # Half-pitch margin so a jittered candidate can never end up clipped by the
    # canvas edge (the same containment every other mode guarantees).
    inset = pitch / 2.0
    x, y, ci, ri = _lattice(transform.design_width_mm, transform.design_height_mm,
                            cand_pitch, cand_pitch, stagger=False, inset=inset)
    if x.size == 0:
        return np.empty((0, 2)), np.empty((0,))

    jitter = 0.5 * cand_pitch
    jx = (_hash01(ci, ri, salt=1) - 0.5) * 2.0 * jitter
    jy = (_hash01(ci, ri, salt=2) - 0.5) * 2.0 * jitter
    xj = np.clip(x + jx, inset, transform.design_width_mm - inset)
    yj = np.clip(y + jy, inset, transform.design_height_mm - inset)

    dens = _sample_density(density, transform, xj, yj)
    keep = _sample_density(keep_density, transform, xj, yj) > _bayer_at(ci, ri)
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
# Outline + Fill placement (professional mode)
# --------------------------------------------------------------------------- #


def _resample_closed_path(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon onto ``n`` points at equal arc spacing."""
    if points.shape[0] < 2 or n < 1:
        return points[:1]
    closed = np.vstack([points, points[:1]])  # close the loop
    seg = np.diff(closed, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(points[:1], n, axis=0)
    targets = total * (np.arange(n) + 0.5) / n
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, seg_len.shape[0] - 1)
    frac = (targets - cum[idx]) / np.maximum(seg_len[idx], 1e-12)
    return closed[idx] + seg[idx] * frac[:, None]


def _otsu_threshold(values: np.ndarray) -> float | None:
    """Otsu's threshold for 0..255 ``values``, or ``None`` if it is degenerate.

    ``None`` means the field is effectively single-valued — there is no split
    to find, so the caller must not invent one.
    """
    if values.size == 0:
        return None
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    if np.count_nonzero(hist) < 2:
        return None
    total = hist.sum()
    levels = np.arange(256, dtype=np.float64)
    p = hist / total
    omega = np.cumsum(p)                       # weight of the dark class
    mu = np.cumsum(p * levels)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(denom > 1e-12, (mu_total * omega - mu) ** 2 / denom, 0.0)
    return float(np.argmax(between))


def _despeckle(mask: np.ndarray, pitch_px: float) -> np.ndarray:
    """Drop blobs too small to hold a stone; seal pinholes of the same size.

    This replaces the stone-scale morphological opening: an opening removes
    speckle by *eroding* the shape, which also shaves thin features (letter
    stems, a star's points) and pulls the whole boundary a fraction of a stone
    inwards — the silhouette then no longer matches the artwork's own outline.
    Filtering by connected-component *area* removes exactly the noise and
    leaves the artwork's edge where it is.
    """
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return np.zeros(mask.shape, np.uint8)
    min_area = max(4.0, 0.25 * math.pi * (0.5 * pitch_px) ** 2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    if not keep.any():
        return (keep * 255).astype(np.uint8)
    # Pinholes: background pockets of that size that do not reach the border
    # (a pocket touching the border is the world outside the shape, not a hole).
    holes = (1 - keep).astype(np.uint8)
    h_count, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    border = set(np.unique(np.concatenate([h_labels[0], h_labels[-1],
                                           h_labels[:, 0], h_labels[:, -1]])).tolist())
    for i in range(1, h_count):
        if i in border:
            continue
        if h_stats[i, cv2.CC_STAT_AREA] < min_area:
            keep[h_labels == i] = 1
    return (keep * 255).astype(np.uint8)


def _shape_from_density(density: np.ndarray, pitch_px: float) -> np.ndarray:
    """Solid shapes from the tone field — the path for opaque artwork.

    Otsu splits subject from background, a uniformly tonal field falls back to
    the tone floor, and stone-scale noise is filtered away.
    """
    mask8 = np.clip(density * 255.0, 0, 255).astype(np.uint8)
    threshold = _otsu_threshold(mask8)
    bw = np.where(mask8 > (0 if threshold is None else threshold), 255, 0).astype(np.uint8)

    # A split that keeps only a sliver of an otherwise *toned* field is not a
    # subject standing against a background: there is no background to
    # separate (a solid, full-bleed design — Otsu then latches onto some
    # one-level artefact of it). The whole tonal field is the shape.
    sliver = int(np.count_nonzero(bw)) < _SPLIT_MIN_FRACTION * mask8.size
    if sliver and int(mask8.min()) > _SOLID_TONE_MIN:
        bw = (mask8 > _SOLID_TONE_MIN).astype(np.uint8) * 255

    bw = _despeckle(bw, pitch_px)
    if not bw.any():
        # Otsu needs a bimodal field. A *uniformly* tonal one — a solid dark
        # image, a full-bleed subject — gives it nothing to split, so it puts
        # the threshold at the very top of the range and binarises to a
        # speckle that the stone-scale clean then erases: a solid design
        # generated no stones at all, and only at some analysis resolutions,
        # which made the failure look random. With no background to separate,
        # the whole tonal field is the shape — retry from the tone floor.
        bw = _despeckle((mask8 > _SOLID_TONE_MIN).astype(np.uint8) * 255, pitch_px)
    return bw


def _shape_from_silhouette(
    density: np.ndarray, subject: np.ndarray, pitch_px: float
) -> np.ndarray:
    """The artwork silhouette, exactly as the image declares it.

    The silhouette comes from the image's own transparency, so the traced row
    lands precisely on the outline the user sees — thin features and sharp
    points included (the old stone-scale opening rounded those off and pulled
    the whole boundary inside the artwork).

    Internal structure is still read from the image: where a *pale, flat*
    region sits inside the shape — a logo's white detail, the same white as its
    background — it is carved out as a hole and traced in turn. Artwork that is
    tonally flat (a solid or pale logo) has no such structure and fills whole.
    """
    silhouette = _despeckle((subject > 0.5).astype(np.uint8) * 255, pitch_px)
    inside = silhouette > 0
    if not inside.any():
        return silhouette

    mask8 = np.clip(density * 255.0, 0, 255).astype(np.uint8)
    detail_values = mask8[inside]
    threshold = _otsu_threshold(detail_values)
    if threshold is None:
        return silhouette

    # The pale class of that split (low density == light-coloured artwork).
    pale = detail_values[detail_values <= threshold]
    if pale.size == 0:
        return silhouette
    pale01 = pale.astype(np.float32) / 255.0
    # Only a pale *and flat* region reads as an intentional hole; a photo's or
    # a gradient's light areas are spread out and stay part of the shape.
    if float(pale01.mean()) > _DETAIL_PALE_MAX or float(pale01.std()) > _DETAIL_PALE_SPREAD:
        return silhouette

    holes = np.where((mask8 <= threshold) & inside, 255, 0).astype(np.uint8)
    fraction = float(np.count_nonzero(holes)) / float(np.count_nonzero(silhouette))
    if not (_DETAIL_HOLE_MIN <= fraction <= _DETAIL_HOLE_MAX):
        return silhouette
    return np.where(inside & ~(holes > 0), 255, 0).astype(np.uint8)


def _gate_reach(gate: np.ndarray) -> np.ndarray:
    """The traced region pulled in by the contour tolerance.

    Outline lays stones *on* the contours of the shape it is handed, and both
    deriving and tracing a contour costs a pixel or two of accuracy (Otsu's
    edge, the polygon simplification, the boundary smoothing). Along the
    artwork's own outline that slop is invisible; along the brush edge it
    would push stones outside the region the user painted. Pulling the clip in
    by exactly that tolerance keeps the contract that stones land inside the
    trace, without moving the artwork's outline by anything at all.
    """
    reach = (gate > 0.5).astype(np.uint8)
    for _ in range(1 + int(round(_CONTOUR_EPS_PX))):
        eroded = cv2.erode(reach, np.ones((3, 3), np.uint8))
        if not eroded.any():
            break
        reach = eroded
    return reach


def _place_outline(
    density: np.ndarray,
    transform: DesignTransform,
    pitch: float,
    shape: np.ndarray | None = None,
    clip: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Outline + Fill: even stone rows on shape contours, lattice interior.

    The professional rhinestone conversion (no halftone dithering):

    1. take the artwork's solid shapes — its exact silhouette when the image
       declares one (transparent PNG/SVG), otherwise an Otsu split of the tone
       field (opaque artwork);
    2. clean at stone scale — specks smaller than a stone (JPG noise, scan
       dust) are dropped *by area*, so the edge itself is not eroded, and
       pinholes are sealed;
    3. smooth the pixel boundary so outline rows glide around curves instead
       of staircasing;
    4. trace every contour (outlines AND holes) and lay stones at *equal*
       arc-length spacing ``L / floor(L / pitch)`` — perfectly even, never
       below the pitch, and the loop closes without a seam gap;
    5. fill the interior (everywhere further than one pitch from the
       boundary) with the hexagonal lattice.

    ``clip`` (optional) bounds the shape *after* it is derived — the traced
    region, inset by the contour tolerance so stones laid on a contour that
    runs along the brush edge stay inside the painted area.

    Deterministic throughout.
    """
    h, w = density.shape
    scale = min(transform.scale_x, transform.scale_y)  # mm per px, conservative
    pitch_px = pitch / max(scale, 1e-9)

    # -- 1-2. Solid shape mask ------------------------------------------------ #
    if shape is not None:
        bw = _shape_from_silhouette(density, shape, pitch_px)
    else:
        bw = _shape_from_density(density, pitch_px)
    if clip is not None:
        bw = np.where(clip > 0, bw, 0).astype(np.uint8)
    if not bw.any():
        return np.empty((0, 2)), np.empty((0,))

    # -- 3. Boundary smoothing (round the pixel staircase) --------------------- #
    blur = cv2.GaussianBlur(bw, (0, 0), sigmaX=1.0)
    bw = np.where(blur >= 128, 255, 0).astype(np.uint8)
    if not bw.any():
        return np.empty((0, 2)), np.empty((0,))

    # -- 4. Outline stones on every contour ------------------------------------- #
    # Chord distances between arc-spaced points run slightly short of the arc
    # (corner-cutting at polygon vertices, curvature); the 5% margin keeps
    # every consecutive pair at or above the true pitch so the greedy safety
    # filter below never has to drop a stone on a smooth contour.
    contours, _hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outline_px: list[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        poly = cv2.approxPolyDP(contour, _CONTOUR_EPS_PX, True).reshape(-1, 2).astype(np.float64)
        if poly.shape[0] < 3:
            continue
        length_px = float(cv2.arcLength(poly.astype(np.float32), True))
        if length_px < 0.5 * pitch_px:
            continue
        if length_px < 1.5 * pitch_px:
            # A tiny blob: one stone at its centre.
            m = cv2.moments(contour)
            if m["m00"] > 0:
                outline_px.append(np.array([[[m["m10"] / m["m00"], m["m01"] / m["m00"]]]]))
            continue
        n = max(1, int(math.floor(length_px / (pitch_px * 1.05))))
        outline_px.append(_resample_closed_path(poly, n))

    if outline_px:
        outline_pts_px = np.vstack(outline_px)
    else:
        outline_pts_px = np.empty((0, 2))

    # px → mm; clamp into the artboard with a half-pitch inset (a stone centred
    # exactly on the canvas edge would be half off the transfer). Collision
    # filtering happens below, together with the fill candidates, in exact
    # millimetres — pixel-quantised insets cannot guarantee the pitch on
    # low-resolution density maps.
    positions = transform.image_to_design_array(outline_pts_px)
    inset = min(pitch / 2.0, transform.design_width_mm / 2.0, transform.design_height_mm / 2.0)
    positions[:, 0] = np.clip(positions[:, 0], inset, transform.design_width_mm - inset)
    positions[:, 1] = np.clip(positions[:, 1], inset, transform.design_height_mm - inset)
    dens = np.full(positions.shape[0], 1.0)

    # -- 5. Hex fill of the deep interior --------------------------------------- #
    # Candidates: hex lattice cells at least ~one pitch inside the shape (the
    # exact inset is enforced below in mm). The mask is zero-padded first — a
    # full-bleed shape has no background pixels, and without the pad the
    # distance transform would not inset the fill from the artboard edge.
    bw_padded = cv2.copyMakeBorder(bw, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(bw_padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    fill_region = dist >= pitch_px
    if fill_region.any():
        vpitch = pitch * math.sqrt(3.0) / 2.0
        fx, fy, _ci, _ri = _lattice(
            transform.design_width_mm, transform.design_height_mm,
            pitch, vpitch, stagger=True, inset=pitch / 2.0,
        )
        if fx.size:
            ix = np.clip(np.round(fx / transform.scale_x).astype(np.int64), 0, w - 1)
            iy = np.clip(np.round(fy / transform.scale_y).astype(np.int64), 0, h - 1)
            keep = fill_region[iy, ix]
            positions = np.vstack([positions, np.column_stack([fx[keep], fy[keep]])])
            dens = np.concatenate([dens, np.full(int(keep.sum()), 0.85)])

    # -- 6. Collision filter: outline stones in path order first, then fill -- #
    # A neck narrower than the pitch between contour walls, or a fill row that
    # lands a hair under the pitch from an outline stone, drops only that
    # stone — the rest keep their perfectly even spacing.
    if positions.shape[0] > 1:
        order = np.arange(positions.shape[0])
        accepted = _greedy_min_distance(positions[:, 0], positions[:, 1], order, pitch)
        positions = positions[accepted]
        dens = dens[accepted]

    return positions, dens


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def place_stones(
    density: np.ndarray,
    transform: DesignTransform,
    settings: GenerationSettings,
    *,
    gate: np.ndarray | None = None,
    subject: np.ndarray | None = None,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
    on_batch: BatchCallback | None = None,
) -> PlacementResult:
    """Place stones over ``density`` and return their physical positions.

    ``density`` must be an ``(H, W)`` ``float``/``float32`` array in ``[0, 1]``
    whose shape matches ``transform``'s image dimensions.

    ``gate`` (optional, same shape as ``density`` after resampling, values in
    ``[0, 1]``) is the trace-tool mask: it multiplies both the density and the
    detail-keep map, so stones land **strictly** inside the traced region —
    not even the half-pitch detail dilation can bleed past the trace edge.

    ``subject`` (optional) is the image's artwork silhouette in ``[0, 1]`` at
    the density grid's resolution (see
    :func:`~app.core.image_processor.subject_mask_from_alpha`). When present,
    no stone is placed outside it and pale artwork inside it is lifted to a
    solid fill, so a white or pastel design still gets its stones. Outline
    mode traces the silhouette itself, which is what makes the stone rows line
    up with the artwork's own outline.

    Every returned position is fully inside the artboard: centres keep at least
    half a pitch from each edge, which leaves at least half the configured gap
    of clearance beyond the stone's own radius.

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

    if gate is not None:
        gate = np.asarray(gate, dtype=np.float64)
        if gate.shape != density.shape:
            from app.core.trace_mask import density_gate as _dg

            # ``density_gate`` wants a 0/255 raster. ``gate`` may arrive either
            # as that raster or already normalised to 0..1 — check before
            # scaling, because multiplying a 0/255 array by 255 wraps at
            # uint8 and turns the whole mask into a 1/255 near-zero gate.
            arr = (np.clip(gate * 255.0, 0, 255).astype(np.uint8)
                   if float(np.max(gate)) <= 1.0 else gate.astype(np.uint8))
            gate = _dg(arr, h, w).astype(np.float64)
        density = density * gate

    if subject is not None:
        shape = np.ascontiguousarray(np.asarray(subject, dtype=np.float64))
        if shape.shape != density.shape:
            shape = cv2.resize(
                (shape > 0.5).astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float64)
        # The trace, when there is one, bounds the shape too: a traced region
        # limits the artwork that may be stoned, and Outline must trace what is
        # left of it rather than the whole silhouette.
        if gate is not None:
            shape = shape * _gate_reach(gate)
        density = _apply_subject(density, shape, gate)
    else:
        shape = None

    _report(progress, "Placing stones", 0.1)
    _check_cancel(is_cancelled)

    mode = settings.placement_mode
    if mode is PlacementMode.OUTLINE:
        # Contour tracing + lattice fill: no dithering, so the detail-keep
        # map does not apply.
        clip = _gate_reach(gate) if gate is not None else None
        positions, densities = _place_outline(density, transform, pitch,
                                              shape=shape, clip=clip)
        streamed = False
    elif mode is PlacementMode.ORGANIC:
        # With a max-stone cap the first ``max_stones`` accepted (descending
        # density) are exactly the stones the cap keeps, so streaming can stop
        # there — the preview shows precisely the final set.
        keep_density = _detail_keep_map(density, transform, pitch)
        if gate is not None:
            keep_density = keep_density * gate
        positions, densities = _place_organic(
            density, keep_density, transform, pitch,
            on_batch=on_batch,
            max_emit=settings.max_stones,
            progress=progress,
            is_cancelled=is_cancelled,
        )
        streamed = on_batch is not None
    else:
        keep_density = _detail_keep_map(density, transform, pitch)
        if gate is not None:
            keep_density = keep_density * gate
        # The half-stone detail dilation must not push stones off the artwork:
        # the thin detail it exists to catch is inside the shape.
        if shape is not None:
            keep_density = keep_density * shape
        positions, densities = _place_lattice(density, keep_density, transform, pitch, mode)
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
