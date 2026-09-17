"""The trace mask — a painted region that gates where stones may be placed.

The Trace tool lets the user drag-paint the areas of the artboard that should
receive rhinestones. The mask lives in **design (mm) space**, not image space:

* the canvas paints in mm, so strokes are written in the coordinate system the
  user sees — no image-transform bookkeeping;
* the mask survives image rotate/flip (those transform the image *within* the
  artboard, which does not move) and is only resampled when the physical
  aspect ratio changes;
* the placement engine converts it to a per-pixel gate on the density grid via
  the same design→density mapping.

The mask is a small ``uint8`` raster (long side :data:`MASK_LONG_SIDE`, values
0/255) attached to :class:`~app.models.project.Project` and embedded in the
``.rhinestone`` archive by :mod:`app.services.project_service`.

Pure NumPy/OpenCV, no Qt — independently testable.
"""
from __future__ import annotations

import cv2
import numpy as np

#: Resolution of the mask raster: the longer side, in pixels. 768 keeps a
#: full mask under ~600 KB and gives ~3 px/mm on a 10-inch design — plenty for
#: brush-stroke regions, which are many stone-pitches wide.
MASK_LONG_SIDE = 768


def mask_shape_for(width_mm: float, height_mm: float) -> tuple[int, int]:
    """Return the mask raster shape ``(h, w)`` for a design of this size."""
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError(f"Design size must be positive, got {width_mm}x{height_mm}")
    aspect = height_mm / width_mm
    if aspect >= 1.0:
        h = MASK_LONG_SIDE
        w = max(1, int(round(MASK_LONG_SIDE / aspect)))
    else:
        w = MASK_LONG_SIDE
        h = max(1, int(round(MASK_LONG_SIDE * aspect)))
    return h, w


def blank_mask(width_mm: float, height_mm: float) -> np.ndarray:
    """A fresh all-zero (untraced) mask for a design of this size."""
    return np.zeros(mask_shape_for(width_mm, height_mm), dtype=np.uint8)


def resample_mask(mask: np.ndarray, width_mm: float, height_mm: float) -> np.ndarray:
    """Resample ``mask`` to the shape for the (possibly new) design size.

    Returns ``mask`` itself when the shape already matches. Resampling is
    bilinear followed by a 0/255 threshold, so the result stays binary and the
    painted region keeps its *relative* position on the artboard.
    """
    target = mask_shape_for(width_mm, height_mm)
    if mask.shape == target:
        return mask
    out = cv2.resize(mask, (target[1], target[0]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray((out >= 128).astype(np.uint8) * 255)


def _mm_to_px(mask: np.ndarray, width_mm: float, height_mm: float,
              x_mm: float, y_mm: float) -> tuple[float, float]:
    """Map a design-mm point to fractional mask-pixel coordinates."""
    h, w = mask.shape
    px = np.clip(x_mm / width_mm, 0.0, 1.0) * (w - 1)
    py = np.clip(y_mm / height_mm, 0.0, 1.0) * (h - 1)
    return float(px), float(py)


def paint_stroke(
    mask: np.ndarray,
    from_mm: tuple[float, float] | None,
    to_mm: tuple[float, float],
    radius_mm: float,
    width_mm: float,
    height_mm: float,
    *,
    erase: bool = False,
) -> bool:
    """Paint one brush segment into ``mask`` (in place).

    Draws a thick line from ``from_mm`` to ``to_mm`` (round caps via end
    circles); ``from_mm=None`` paints a dot. Returns True if any pixel
    changed. Coordinates are design millimetres, clamped to the artboard.
    """
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError(f"Expected a 2-D uint8 mask, got {mask.dtype} {mask.shape}")
    radius_mm = max(0.5, float(radius_mm))
    h, w = mask.shape
    scale = max(w / max(width_mm, 1e-9), h / max(height_mm, 1e-9))
    r_px = max(1, int(round(radius_mm * scale)))

    tx, ty = _mm_to_px(mask, width_mm, height_mm, *to_mm)
    pts = [tx, ty]
    if from_mm is not None:
        fx, fy = _mm_to_px(mask, width_mm, height_mm, *from_mm)
        pts = [fx, fy, tx, ty]

    color = 0 if erase else 255
    before_sum = int(mask.sum(dtype=np.int64))
    if len(pts) == 4:
        # LINE_8 (not LINE_AA): the mask must stay strictly binary (0/255).
        cv2.line(mask, (int(round(pts[0])), int(round(pts[1]))),
                 (int(round(pts[2])), int(round(pts[3]))), color, r_px * 2)
    cv2.circle(mask, (int(round(tx)), int(round(ty))), r_px, color, -1)
    return int(mask.sum(dtype=np.int64)) != before_sum


def stroke_bbox(mask: np.ndarray, before: np.ndarray | None) -> tuple[int, int, int, int] | None:
    """Bounding box ``(x, y, w, h)`` of pixels that differ from ``before``.

    ``before`` may be None (the mask did not exist before the stroke) — then
    every non-zero pixel is "changed". Returns None when nothing changed.
    """
    if before is None:
        diff = mask != 0
    else:
        if before.shape != mask.shape:
            return None
        diff = mask != before
    ys, xs = np.nonzero(diff)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def extract_patch(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Copy the mask region ``bbox = (x, y, w, h)`` (clamped to bounds)."""
    h, w = mask.shape
    x, y, bw, bh = bbox
    x0 = max(0, min(x, w))
    y0 = max(0, min(y, h))
    x1 = max(x0, min(x + bw, w))
    y1 = max(y0, min(y + bh, h))
    return mask[y0:y1, x0:x1].copy()


def apply_patch(mask: np.ndarray, bbox: tuple[int, int, int, int], patch: np.ndarray) -> None:
    """Write ``patch`` back into ``mask`` at ``bbox`` (in place, bounds-clamped).

    If the mask was resampled to a new aspect since the patch was taken, the
    patch is resized to fit — a best-effort undo rather than a crash.
    """
    h, w = mask.shape
    x, y, bw, bh = bbox
    x0 = max(0, min(x, w))
    y0 = max(0, min(y, h))
    x1 = max(x0, min(x + bw, w))
    y1 = max(y0, min(y + bh, h))
    if x1 <= x0 or y1 <= y0:
        return
    region = mask[y0:y1, x0:x1]
    if patch.shape != region.shape:
        patch = cv2.resize(patch, (region.shape[1], region.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    region[...] = patch


def mask_is_empty(mask: np.ndarray | None) -> bool:
    """True when there is no mask, or it contains no painted pixels."""
    return mask is None or not bool(np.any(mask))


def density_gate(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resample the mask to the density grid as a float gate in ``[0, 1]``.

    Nearest-neighbour resampling keeps the trace edge **hard**: a lattice
    cell is either inside the painted region or outside it — no fractional
    admission at the boundary (the user drew a boundary, not a gradient).
    """
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Gate shape must be positive, got {out_h}x{out_w}")
    if mask.shape == (out_h, out_w):
        return mask.astype(np.float32) / 255.0
    gate = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(gate.astype(np.float32) / 255.0)
