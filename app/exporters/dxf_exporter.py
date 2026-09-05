"""DXF export via :mod:`ezdxf` — vector stone geometry for CAD/laser/cutter.

Produces a clean DXF R2010 document whose coordinates are in **millimetres**
(``$INSUNITS = 4``, the DXF code for mm), with stones as true ``CIRCLE``
entities. Three layers are available and exported selectively:

* ``RHINESTONES`` — one circle per stone (always exported).
* ``BOUNDARY`` — the physical design rectangle (optional).
* ``GUIDES`` — centre cross-hairs / registration marks (optional).

DXF uses a bottom-left origin with Y pointing up, the opposite of image/screen
space, so stone Y coordinates are flipped about the design height on export.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ezdxf

from app.exporters.base import ExportError
from app.models.design import Design

LAYER_RHINESTONES = "RHINESTONES"
LAYER_BOUNDARY = "BOUNDARY"
LAYER_GUIDES = "GUIDES"

# AutoCAD Color Index values for the layers.
_ACI_RHINESTONE = 5   # blue
_ACI_BOUNDARY = 1     # red
_ACI_GUIDE = 8        # grey


@dataclass(slots=True)
class DxfOptions:
    """Which layers to include and their styling."""

    include_boundary: bool = True
    include_guides: bool = False
    # If true, stones are drawn at (diameter + allowance) — e.g. template holes.
    hole_allowance_mm: float = 0.0


def _flip_y(y_mm: float, height_mm: float) -> float:
    """Convert image/screen Y (down) to DXF Y (up)."""
    return height_mm - y_mm


def build_dxf(design: Design, options: DxfOptions | None = None):
    """Build and return an ``ezdxf`` document for ``design``."""
    options = options or DxfOptions()
    if design.width_mm <= 0 or design.height_mm <= 0:
        raise ExportError("Design has invalid physical dimensions")

    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 4  # millimetres
    msp = doc.modelspace()

    doc.layers.add(LAYER_RHINESTONES, color=_ACI_RHINESTONE)
    if options.include_boundary:
        doc.layers.add(LAYER_BOUNDARY, color=_ACI_BOUNDARY)
    if options.include_guides:
        doc.layers.add(LAYER_GUIDES, color=_ACI_GUIDE)

    h = design.height_mm
    allowance = options.hole_allowance_mm / 2.0
    for stone in design.stones:
        radius = stone.radius + allowance
        if radius <= 0:
            continue
        msp.add_circle(
            center=(stone.x, _flip_y(stone.y, h)),
            radius=radius,
            dxfattribs={"layer": LAYER_RHINESTONES},
        )

    if options.include_boundary:
        w = design.width_mm
        corners = [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]
        msp.add_lwpolyline(corners, close=True, dxfattribs={"layer": LAYER_BOUNDARY})

    if options.include_guides:
        w = design.width_mm
        cx, cy = w / 2.0, h / 2.0
        span = min(w, h) * 0.05
        msp.add_line((cx - span, cy), (cx + span, cy), dxfattribs={"layer": LAYER_GUIDES})
        msp.add_line((cx, cy - span), (cx, cy + span), dxfattribs={"layer": LAYER_GUIDES})

    return doc


def export_dxf(design: Design, path: str | Path, options: DxfOptions | None = None) -> Path:
    """Write ``design`` to a DXF file at ``path``.

    Writes to a temporary sibling file first, then atomically replaces the
    destination, so a failure never leaves a half-written DXF.
    """
    import os

    path = Path(path)
    doc = build_dxf(design, options)
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.saveas(tmp)
        os.replace(tmp, path)
    except (OSError, IOError) as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExportError(f"Could not write DXF {path}: {exc}") from exc
    return path
