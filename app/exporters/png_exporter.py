"""PNG preview export and shared stone-rasterisation.

The PNG is a **visual preview**, never the production source (that is the SVG /
DXF vector output). :func:`render_design_png` rasterises the design to an RGBA
NumPy array at a requested pixel width; it is reused by the PDF production sheet
so the two previews look identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.exporters.base import ExportError, atomic_write_bytes
from app.models.design import Design

#: Paper-like background for previews (matches the canvas artboard).
_BG_RGBA = (244, 246, 248, 255)


@dataclass(slots=True)
class PngOptions:
    width_px: int = 2000
    background: tuple[int, int, int, int] | None = _BG_RGBA  # None → transparent
    padding_frac: float = 0.03  # margin as a fraction of the long side


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)  # OpenCV is BGR


def render_design_png(design: Design, options: PngOptions | None = None) -> np.ndarray:
    """Rasterise ``design`` to an ``(H, W, 4)`` RGBA array at physical aspect.

    The canvas maps the full physical design rectangle to the image, so the
    preview always shows the whole artboard at the correct aspect ratio.
    """
    options = options or PngOptions()
    if design.width_mm <= 0 or design.height_mm <= 0:
        raise ExportError("Design has invalid physical dimensions")

    width_px = max(64, int(options.width_px))
    pad = int(round(width_px * options.padding_frac))
    inner_w = width_px - 2 * pad
    scale = inner_w / design.width_mm  # px per mm
    inner_h = int(round(design.height_mm * scale))
    height_px = inner_h + 2 * pad

    # Build a BGRA canvas (OpenCV order) then convert to RGBA at the end.
    if options.background is None:
        bgra = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    else:
        r, g, b, a = options.background
        bgra = np.empty((height_px, width_px, 4), dtype=np.uint8)
        bgra[:, :] = (b, g, r, a)

    for stone in design.stones:
        cx = int(round(pad + stone.x * scale))
        cy = int(round(pad + stone.y * scale))
        radius = max(1, int(round(stone.radius * scale)))
        b, g, r = _hex_to_bgr(stone.color)
        cv2.circle(bgra, (cx, cy), radius, (b, g, r, 255), thickness=-1, lineType=cv2.LINE_AA)

    # BGRA → RGBA.
    rgba = bgra[:, :, [2, 1, 0, 3]].copy()
    return rgba


def encode_png(rgba: np.ndarray) -> bytes:
    """Encode an RGBA array to PNG bytes."""
    bgra = rgba[:, :, [2, 1, 0, 3]].copy()
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise ExportError("Failed to encode PNG")
    return buf.tobytes()


def export_png(design: Design, path: str | Path, options: PngOptions | None = None) -> Path:
    """Render ``design`` and write a PNG preview to ``path`` atomically."""
    rgba = render_design_png(design, options)
    return atomic_write_bytes(path, encode_png(rgba))
