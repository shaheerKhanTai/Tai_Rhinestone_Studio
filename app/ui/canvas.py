"""The design canvas — a high-performance custom-painted stone editor view.

Rendering strategy (the reason this is a hand-painted ``QWidget`` and not a
``QGraphicsItem`` per stone):

* Stone geometry is held as NumPy arrays; every frame converts mm→screen
  vectorised and **culls** to the visible viewport.
* Tiny stones (a few pixels) are composited into a single ``QImage`` with NumPy
  and blitted once — this keeps zoomed-out views of tens of thousands of stones
  smooth.
* Larger stones are drawn from a cached anti-aliased disc *stamp* per colour.

The canvas owns view state (zoom + pan), draws the artboard, an optional
background-image overlay (never part of the exported geometry), an optional
grid and rulers, collision highlights and the current selection. Mutating tools
(move/add/delete/brush) are layered on in Phase 9; this phase provides the view,
hit-testing and rubber-band selection.
"""
from __future__ import annotations

import enum
import math

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QGridLayout, QScrollBar, QWidget

from app.core.geometry import Unit
from app.core.history import (
    AddStonesCommand,
    Command,
    DeleteStonesCommand,
    MoveStonesCommand,
    PaintTraceCommand,
    SetTraceMaskCommand,
)
from app.core.placement import REGION_OUTLINE, REGION_SOLID, REGION_TONAL
from app.core.trace_mask import (
    blank_mask,
    paint_stroke as paint_mask_stroke,
    resample_mask,
    stroke_bbox,
)
from app.models.design import Design
from app.models.stone import DEFAULT_STONE_COLOR, Stone
from app.ui.qt_image import numpy_rgba_to_qimage
from app.ui.theme import Theme

# Zoom limits in pixels-per-millimetre.
_MIN_ZOOM = 0.05
_MAX_ZOOM = 80.0
_RULER = 20  # ruler strip thickness (px)
#: Screen pixels scrolled per wheel notch (120 units of ``angleDelta``).
_WHEEL_STEP_PX = 60.0

#: The artboard ("paper") colour — a near-white used in both themes so stones
#: read with strong contrast against the document surface.
_ARTBOARD_COLOR = "#f4f6f8"
#: Grid line colour drawn over the (light) artboard.
_GRID_COLOR = "#c8d0da"


class CanvasTool(enum.Enum):
    """Active pointer tool. Mutating tools are wired in Phase 9."""

    SELECT = "select"
    PAN = "pan"
    ADD = "add"
    DELETE = "delete"
    BRUSH = "brush"
    ERASER = "eraser"
    CROP = "crop"
    TRACE = "trace"


def _as_point(mm: QPointF):
    from app.core.geometry import Point

    return Point(mm.x(), mm.y())


class CanvasView(QWidget):
    """Interactive, zoomable rhinestone design canvas."""

    zoom_changed = Signal(float)          # percent relative to actual size
    cursor_moved = Signal(float, float)   # design mm (x, y)
    selection_changed = Signal(int)       # number of selected stones
    stones_mutated = Signal()             # emitted after editing (Phase 9)
    command_produced = Signal(object)     # a history Command the app should record
    crop_committed = Signal(float, float, float, float)  # normalized x, y, w, h of the artboard
    crop_cancelled = Signal()
    push_failed = Signal()                # Push All could not resolve the move
    trace_changed = Signal()              # trace mask painted / erased / cleared
    trace_mask_replaced = Signal()        # mask array identity changed (undo/clear)

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._design: Design | None = None
        self._unit = Unit.INCH
        # Owning project (for the trace mask); set by the main window.
        self._project = None

        # View transform: screen = origin + mm * zoom.
        self._zoom = 1.0
        self._origin = QPointF(0.0, 0.0)

        # Scrollbars: attached by CanvasScrollHost, the canvas owns the model.
        self._hbar: QScrollBar | None = None
        self._vbar: QScrollBar | None = None
        self._syncing_bars = False
        self._wheel_zooms = False  # Preferences: plain wheel zooms (pre-scrollbars)

        # Cached stone geometry (rebuilt on set_design / refresh_geometry).
        self._pos = np.empty((0, 2), dtype=np.float64)
        self._dia = np.empty((0,), dtype=np.float64)
        self._color_idx = np.empty((0,), dtype=np.int64)
        self._palette: list[QColor] = []
        self._ids = np.empty((0,), dtype=np.int64)
        self._id_to_index: dict[int, int] = {}
        self._selected = np.empty((0,), dtype=bool)
        self._collision = np.empty((0,), dtype=bool)

        # Background image overlay.
        self._bg_pixmap: QPixmap | None = None
        self._show_image = True
        self._bg_opacity = 0.45
        self._show_grid = False
        self._show_rulers = True

        # Stamp cache: (rgb, diameter_px) -> QPixmap.
        self._stamp_cache: dict[tuple[int, int], QPixmap] = {}

        # Interaction state.
        self._tool = CanvasTool.SELECT
        self._panning = False
        self._space_down = False
        self._last_pos: QPointF | None = None
        self._rubber_start: QPointF | None = None
        self._rubber_now: QPointF | None = None

        # Crop tool state (rect is in design mm, clamped to the artboard).
        self._crop_mode = False
        self._crop_rect_mm: QRectF | None = None
        self._crop_dragging = False
        self._crop_anchor_mm: QPointF | None = None

        # Editing state.
        self._new_stone_diameter = 2.8
        self._new_stone_size_id = "SS10"
        self._new_stone_color = DEFAULT_STONE_COLOR
        self._min_gap_mm = 0.5
        self._brush_radius_mm = 8.0
        self._dragging_selection = False
        self._drag_anchor_mm: QPointF | None = None
        self._drag_start_positions: dict[int, tuple[float, float]] | None = None
        self._brush_touched_ids: set[int] = set()
        self._brush_added: list[Stone] = []
        self._brush_active = False

        # Push All (Issue 3): when enabled, a drag pushes neighbours aside to
        # preserve the configured spacing.
        self._push_all = False
        self._push_snapshot: np.ndarray | None = None  # pre-drag positions (N, 2)
        self._push_failed = False

        # Trace tool state: the mask belongs to the project; the canvas holds
        # a live reference for painting and renders it as an overlay.
        self._trace_mask: np.ndarray | None = None
        self._trace_pixmap: QPixmap | None = None   # overlay cache (mask → px)
        self._trace_stroke_before: np.ndarray | None = None  # snapshot at press
        self._trace_last_mm: QPointF | None = None
        self._trace_erasing = False
        self._trace_changed = False
        self._trace_hover_mm: QPointF | None = None

        # "Fill Areas" overlay: the region map the Automatic placement mode
        # judged (0 empty, 1 outline, 2 solid fill, 3 tonal/scattered). Read-only
        # — it visualises a past generation and never feeds back into placement.
        self._region_map: np.ndarray | None = None
        self._region_pixmap: QPixmap | None = None
        self._show_fill_areas = False

        # AI suggestion preview (Issue 6): ghost stones for proposed adds and
        # arrows for proposed moves, drawn above everything else.
        self._ai_preview_adds: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._ai_preview_diameter: float = 2.8
        self._ai_preview_moves: list[tuple[float, float, float, float]] = []

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.ArrowCursor)
        self.setMinimumSize(320, 240)

    # ------------------------------------------------------------------ #
    # Public configuration
    # ------------------------------------------------------------------ #
    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._stamp_cache.clear()
        # Overlay colours are palette-derived, so they must be rebuilt.
        self._trace_pixmap = None
        self._region_pixmap = None
        self.update()

    def set_unit(self, unit: Unit) -> None:
        self._unit = unit
        self.update()

    def set_design(self, design: Design | None, *, fit: bool = True) -> None:
        self._design = design
        self.refresh_geometry()
        if design is not None and fit:
            self.fit_to_view()
        self.update()
        self._sync_scrollbars()

    def set_background_image(self, rgba: np.ndarray | None) -> None:
        """Set the source-image overlay (or ``None`` to clear)."""
        if rgba is None:
            self._bg_pixmap = None
        else:
            self._bg_pixmap = QPixmap.fromImage(numpy_rgba_to_qimage(rgba))
        self.update()

    def set_show_image(self, show: bool) -> None:
        self._show_image = show
        self.update()

    def set_background_opacity(self, percent: int) -> None:
        self._bg_opacity = max(0.0, min(1.0, percent / 100.0))
        self.update()

    def set_show_grid(self, show: bool) -> None:
        self._show_grid = show
        self.update()

    def set_show_rulers(self, show: bool) -> None:
        self._show_rulers = show
        self.update()

    def set_tool(self, tool: CanvasTool) -> None:
        if self._crop_mode and tool is not CanvasTool.CROP:
            self.cancel_crop()
        self._tool = tool
        self._update_cursor()

    # ------------------------------------------------------------------ #
    # Crop tool
    # ------------------------------------------------------------------ #
    def begin_crop(self) -> None:
        """Enter crop mode with the selection initialised to the whole artboard."""
        w_mm, h_mm = self._design_size_mm()
        self._crop_mode = True
        self._crop_dragging = False
        self._crop_rect_mm = QRectF(0.0, 0.0, w_mm, h_mm)
        self._tool = CanvasTool.CROP
        self._update_cursor()
        self.update()

    def cancel_crop(self) -> None:
        if not self._crop_mode:
            return
        self._crop_mode = False
        self._crop_dragging = False
        self._crop_rect_mm = None
        self.crop_cancelled.emit()
        self.update()

    def commit_crop(self) -> bool:
        """Emit the current crop as normalized artboard fractions. Returns success."""
        if not self._crop_mode or self._crop_rect_mm is None:
            return False
        w_mm, h_mm = self._design_size_mm()
        r = self._crop_rect_mm.normalized()
        if w_mm <= 0 or h_mm <= 0 or r.width() < 1e-6 or r.height() < 1e-6:
            return False
        fx = max(0.0, min(1.0, r.left() / w_mm))
        fy = max(0.0, min(1.0, r.top() / h_mm))
        fw = max(0.0, min(1.0 - fx, r.width() / w_mm))
        fh = max(0.0, min(1.0 - fy, r.height() / h_mm))
        if fw < 1e-4 or fh < 1e-4:
            return False
        self._crop_mode = False
        self._crop_dragging = False
        self._crop_rect_mm = None
        self.crop_committed.emit(fx, fy, fw, fh)
        self.update()
        return True

    @property
    def crop_active(self) -> bool:
        return self._crop_mode

    def _clamp_crop_point(self, mm: QPointF) -> QPointF:
        w_mm, h_mm = self._design_size_mm()
        return QPointF(min(max(mm.x(), 0.0), w_mm), min(max(mm.y(), 0.0), h_mm))

    def set_new_stone_spec(self, diameter_mm: float, size_id: str, color: str, gap_mm: float) -> None:
        """Configure the stone that add/brush tools create (from settings)."""
        self._new_stone_diameter = float(diameter_mm)
        self._new_stone_size_id = size_id
        self._new_stone_color = color
        self._min_gap_mm = float(gap_mm)

    def set_push_all(self, enabled: bool) -> None:
        """Enable/disable Push All for stone drags (from generation settings)."""
        self._push_all = bool(enabled)

    # ------------------------------------------------------------------ #
    # Trace tool — paint the regions generation should fill
    # ------------------------------------------------------------------ #
    def set_project(self, project) -> None:
        """Remember the owning project (the trace mask lives on it)."""
        self._project = project

    def set_trace_mask(self, mask: np.ndarray | None) -> None:
        """Bind the project's trace mask (no copy) for painting/display."""
        self._trace_mask = mask
        self._trace_pixmap = None
        self.update()

    def trace_mask(self) -> np.ndarray | None:
        return self._trace_mask

    def has_trace(self) -> bool:
        return self._trace_mask is not None and bool(np.any(self._trace_mask))

    def clear_trace(self) -> None:
        """Remove the whole traced area (undoable)."""
        command = self.clear_trace_command()
        if command is not None:
            self.command_produced.emit(command)

    def clear_trace_command(self, *, label: str = "Clear trace") -> Command | None:
        """Clear the traced area and return the undoable command, unrecorded.

        The *caller* owns the returned command so an operation that consumes the
        trace can group the clear with its own step (generation does this, so a
        single Ctrl+Z restores both the previous design and the traced area).
        Returns ``None`` when nothing was traced, or when the canvas has no
        project to swap the mask on.

        The overlay and the project's mask are cleared immediately and
        ``trace_changed`` fires as for any other trace edit, so the UI stays
        consistent without the command being applied.
        """
        if self._project is None or not self.has_trace():
            return None
        before = self._trace_mask
        self._trace_mask = None
        self._project.trace_mask = None
        self._trace_pixmap = None
        command = SetTraceMaskCommand(
            self._project, before, None, on_change=self._on_trace_swapped, label=label
        )
        self.trace_mask_replaced.emit()
        self.trace_changed.emit()
        self.update()
        return command

    def resync_trace_to_project(self) -> None:
        """Re-bind the mask after undo/redo or a project-level swap."""
        mask = getattr(self._project, "trace_mask", None) if self._project is not None else None
        self.set_trace_mask(mask)

    def resample_trace_to(self, width_mm: float, height_mm: float) -> None:
        """Adopt the mask to a changed design aspect (keeps relative position)."""
        if self._trace_mask is None:
            return
        resampled = resample_mask(self._trace_mask, width_mm, height_mm)
        if resampled is not self._trace_mask and self._project is not None:
            self._project.trace_mask = resampled
        self.set_trace_mask(resampled)

    def _on_trace_swapped(self) -> None:
        """History callback: a command swapped the mask — refresh and notify."""
        self._trace_pixmap = None
        self.resync_trace_to_project()
        self.trace_mask_replaced.emit()
        self.trace_changed.emit()
        self.update()

    def _begin_trace(self, mm: QPointF, erase: bool) -> None:
        if self._design is None:
            return
        if self._trace_mask is None:
            self._trace_mask = blank_mask(self._design.width_mm, self._design.height_mm)
            if self._project is not None:
                self._project.trace_mask = self._trace_mask
            # No mask existed before this stroke: record it as the None→mask
            # transition so a single undo removes the mask entirely.
            self._trace_stroke_before = None
        else:
            self._trace_stroke_before = self._trace_mask.copy()
        self._trace_last_mm = QPointF(mm)
        self._trace_erasing = erase
        self._trace_changed = False
        self._apply_trace(mm)

    def _apply_trace(self, mm: QPointF) -> None:
        if self._design is None or self._trace_mask is None or self._trace_last_mm is None:
            return
        changed = paint_mask_stroke(
            self._trace_mask,
            (self._trace_last_mm.x(), self._trace_last_mm.y()),
            (mm.x(), mm.y()),
            self._brush_radius_mm,
            self._design.width_mm,
            self._design.height_mm,
            erase=self._trace_erasing,
        )
        self._trace_changed = self._trace_changed or changed
        self._trace_last_mm = QPointF(mm)
        self._trace_pixmap = None  # rebuild overlay from the mutated mask
        self.update()

    def _end_trace(self) -> None:
        before = self._trace_stroke_before
        self._trace_stroke_before = None
        self._trace_last_mm = None
        mask = self._trace_mask
        if not self._trace_changed or mask is None or self._design is None:
            # Nothing painted: drop a freshly-created empty mask.
            if mask is not None and not np.any(mask):
                self._trace_mask = None
                if self._project is not None:
                    self._project.trace_mask = None
                self._trace_pixmap = None
                self.update()
            return
        bbox = stroke_bbox(mask, before)
        label = "Erase trace" if self._trace_erasing else "Trace area"
        if before is None or bbox is None:
            # First-ever stroke (mask was just created) — swap None ↔ mask.
            cmd = SetTraceMaskCommand(
                self._project, None, mask,
                on_change=self._on_trace_swapped, label=label,
            )
        else:
            cmd = PaintTraceCommand(
                self._project, before, bbox,
                on_change=self._trace_overlay_invalidated, label=label,
            )
        self.command_produced.emit(cmd)
        self.trace_changed.emit()

    def _trace_overlay_invalidated(self) -> None:
        """History callback: patches were swapped in place — refresh overlay."""
        self._trace_pixmap = None
        self.update()

    def _paint_trace_overlay(self, painter: QPainter, art: QRectF) -> None:
        if self._trace_mask is None or not np.any(self._trace_mask):
            return
        if self._trace_pixmap is None:
            h, w = self._trace_mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            color = QColor(self._theme.palette.primary)
            rgba[:, :, 0] = color.red()
            rgba[:, :, 1] = color.green()
            rgba[:, :, 2] = color.blue()
            rgba[:, :, 3] = self._trace_mask  # painted pixels opaque-ish
            self._trace_pixmap = QPixmap.fromImage(numpy_rgba_to_qimage(rgba))
        painter.save()
        painter.setClipRect(art)
        painter.setOpacity(0.35)
        painter.drawPixmap(art.toRect(), self._trace_pixmap)
        painter.restore()

    def _paint_trace_cursor(self, painter: QPainter) -> None:
        """Brush-radius circle preview while the Trace tool is active."""
        if self._tool is not CanvasTool.TRACE or self._trace_hover_mm is None:
            return
        c = self.mm_to_screen(self._trace_hover_mm.x(), self._trace_hover_mm.y())
        r = self._brush_radius_mm * self._zoom
        pal = self._theme.palette
        pen = QPen(QColor(pal.primary))
        pen.setStyle(Qt.DashLine)
        pen.setWidthF(1.2)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(c, r, r)

    # ------------------------------------------------------------------ #
    # "Fill Areas" overlay — what the Automatic engine judged
    # ------------------------------------------------------------------ #
    def set_fill_areas(self, regions: np.ndarray | None) -> None:
        """Show the region map a generation judged, or ``None`` to clear it.

        ``regions`` is the ``(H, W)`` uint8 map from
        :attr:`~app.core.placement.PlacementResult.regions`. The canvas converts
        it to a coloured overlay itself so the colours come from the current
        theme and re-tint with it, exactly like the trace overlay.
        """
        self._region_map = None if regions is None else np.asarray(regions, dtype=np.uint8)
        self._region_pixmap = None
        self.update()

    def has_fill_areas(self) -> bool:
        """True when a judged region map is available to show."""
        return self._region_map is not None and bool(np.any(self._region_map))

    def set_show_fill_areas(self, show: bool) -> None:
        self._show_fill_areas = bool(show)
        self.update()

    def _paint_fill_areas(self, painter: QPainter, art: QRectF) -> None:
        """Tint the artboard with the engine's region decision.

        Green marks the traced outline row, blue the solid lattice fill and
        amber the tonal scatter — the three treatments the Automatic mode
        chooses between. Drawn over the stones (see the paint order in
        :meth:`paintEvent`) so the outline band is not hidden by the very
        stones that trace it, at a low enough opacity that the stones stay
        readable underneath.
        """
        if not self._show_fill_areas or self._region_map is None:
            return
        if not bool(np.any(self._region_map)):
            return
        if self._region_pixmap is None:
            region = self._region_map
            rgba = np.zeros(region.shape + (4,), dtype=np.uint8)
            pal = self._theme.palette
            # (label, colour) — the colour identifies the treatment at a glance.
            for label, hex_color in (
                (REGION_OUTLINE, pal.success),
                (REGION_SOLID, pal.primary),
                (REGION_TONAL, pal.warning),
            ):
                sel = region == label
                if not sel.any():
                    continue
                c = QColor(hex_color)
                rgba[sel, 0] = c.red()
                rgba[sel, 1] = c.green()
                rgba[sel, 2] = c.blue()
                rgba[sel, 3] = 255
            self._region_pixmap = QPixmap.fromImage(numpy_rgba_to_qimage(rgba))
        painter.save()
        painter.setClipRect(art)
        painter.setOpacity(0.45)
        painter.drawPixmap(art.toRect(), self._region_pixmap)
        painter.restore()

    # ------------------------------------------------------------------ #
    # Progressive generation streaming (ISSUE 5)
    # ------------------------------------------------------------------ #
    def begin_stone_stream(self) -> None:
        """Start a progressive preview stream (called when generation starts).

        The render cache is reset from the design so preview stones can be
        appended on top of it; the design itself is NOT modified until the
        generation result commits.
        """
        self.refresh_geometry()

    def append_stone_batch(self, stones) -> None:
        """Append a batch of preview stones to the render cache (vectorised).

        Preview stones live only in the cache: :meth:`refresh_geometry`
        rebuilds the cache from the design and discards them, so a cancelled
        generation leaves the design untouched.
        """
        if not stones:
            return
        n = len(stones)
        pos = np.empty((n, 2), dtype=np.float64)
        dia = np.empty((n,), dtype=np.float64)
        ids = np.empty((n,), dtype=np.int64)
        cidx = np.empty((n,), dtype=np.int64)
        color_map = {c.rgb(): i for i, c in enumerate(self._palette)}
        for i, s in enumerate(stones):
            pos[i, 0] = s.x
            pos[i, 1] = s.y
            dia[i] = s.diameter
            ids[i] = s.id
            rgb = QColor(s.color).rgb()
            ci = color_map.get(rgb)
            if ci is None:
                ci = len(self._palette)
                color_map[rgb] = ci
                self._palette.append(QColor(s.color))
            cidx[i] = ci
        base = self._pos.shape[0]
        self._pos = np.vstack([self._pos, pos])
        self._dia = np.concatenate([self._dia, dia])
        self._ids = np.concatenate([self._ids, ids])
        self._selected = np.concatenate([self._selected, np.zeros(n, dtype=bool)])
        self._collision = np.concatenate([self._collision, np.zeros(n, dtype=bool)])
        self._color_idx = np.concatenate([self._color_idx, cidx])
        for i in range(n):
            self._id_to_index[int(ids[i])] = base + i
        self.update()

    def end_stone_stream(self) -> None:
        """Discard the preview stream; the cache returns to the design state."""
        self.refresh_geometry()
        self.update()

    # ------------------------------------------------------------------ #
    # AI suggestion preview (Issue 6)
    # ------------------------------------------------------------------ #
    def set_ai_preview(
        self,
        adds: np.ndarray | None,
        moves: list[tuple[float, float, float, float]] | None,
        *,
        diameter: float = 2.8,
    ) -> None:
        """Show a proposed AI change as ghost stones (adds) and arrows (moves).

        ``adds`` is an ``(K, 2)`` mm array; ``moves`` is a list of
        ``(from_x, from_y, to_x, to_y)`` mm tuples. ``None`` clears the preview.
        """
        if adds is None and moves is None:
            self._ai_preview_adds = np.empty((0, 2), dtype=np.float64)
            self._ai_preview_moves = []
        else:
            self._ai_preview_adds = (
                np.asarray(adds, dtype=np.float64) if adds is not None
                else np.empty((0, 2), dtype=np.float64)
            )
            self._ai_preview_moves = list(moves or [])
            self._ai_preview_diameter = float(diameter)
        self.update()

    def _paint_ai_preview(self, painter: QPainter) -> None:
        """Ghost render of a pending AI suggestion (semi-transparent)."""
        pal = self._theme.palette
        ghost = QColor(pal.primary)
        ghost.setAlpha(150)
        r_px = self._ai_preview_diameter * 0.5 * self._zoom
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self._ai_preview_adds.shape[0]:
            painter.setPen(QPen(ghost, 1.0))
            painter.setBrush(Qt.NoBrush)
            for x, y in self._ai_preview_adds:
                c = self.mm_to_screen(x, y)
                painter.drawEllipse(c, max(2.0, r_px), max(2.0, r_px))
        if self._ai_preview_moves:
            pen = QPen(ghost)
            pen.setWidthF(1.5)
            painter.setPen(pen)
            import math as _math

            for fx, fy, tx, ty in self._ai_preview_moves:
                a = self.mm_to_screen(fx, fy)
                b = self.mm_to_screen(tx, ty)
                painter.drawLine(a, b)
                # Small arrowhead.
                ang = _math.atan2(b.y() - a.y(), b.x() - a.x())
                for da in (2.6, -2.6):
                    painter.drawLine(
                        b,
                        QPointF(
                            b.x() - 8 * _math.cos(ang + da),
                            b.y() - 8 * _math.sin(ang + da),
                        ),
                    )

    def set_collisions(self, colliding_ids: set[int]) -> None:
        """Mark stones (by id) that are currently in collision, for highlight."""
        if self._ids.size == 0 or not colliding_ids:
            self._collision = np.zeros(self._ids.size, dtype=bool)
        else:
            self._collision = np.array(
                [sid in colliding_ids for sid in self._ids], dtype=bool
            )
        self.update()

    # ------------------------------------------------------------------ #
    # Geometry cache
    # ------------------------------------------------------------------ #
    def refresh_geometry(self) -> None:
        """Rebuild cached NumPy arrays from the current design's stones."""
        design = self._design
        if design is None or design.stone_count == 0:
            self._pos = np.empty((0, 2), dtype=np.float64)
            self._dia = np.empty((0,), dtype=np.float64)
            self._color_idx = np.empty((0,), dtype=np.int64)
            self._palette = []
            self._ids = np.empty((0,), dtype=np.int64)
            self._id_to_index = {}
            self._selected = np.empty((0,), dtype=bool)
            self._collision = np.empty((0,), dtype=bool)
            return

        stones = design.stones
        n = len(stones)
        self._pos = np.empty((n, 2), dtype=np.float64)
        self._dia = np.empty((n,), dtype=np.float64)
        self._ids = np.empty((n,), dtype=np.int64)
        self._selected = np.empty((n,), dtype=bool)
        color_map: dict[str, int] = {}
        self._palette = []
        self._color_idx = np.empty((n,), dtype=np.int64)
        self._id_to_index = {}
        for i, s in enumerate(stones):
            self._pos[i, 0] = s.x
            self._pos[i, 1] = s.y
            self._dia[i] = s.diameter
            self._ids[i] = s.id
            self._selected[i] = s.selected
            self._id_to_index[s.id] = i
            ci = color_map.get(s.color)
            if ci is None:
                ci = len(self._palette)
                color_map[s.color] = ci
                self._palette.append(QColor(s.color))
            self._color_idx[i] = ci
        self._collision = np.zeros(n, dtype=bool)

    def refresh_selection(self) -> None:
        """Re-read the selected flag from stones into the cache."""
        if self._design is None:
            return
        for i, s in enumerate(self._design.stones):
            if i < self._selected.size:
                self._selected[i] = s.selected
        self.update()

    # ------------------------------------------------------------------ #
    # Coordinate transforms
    # ------------------------------------------------------------------ #
    def _actual_zoom(self) -> float:
        """Pixels-per-mm that render the design at physical size on screen."""
        dpi = self.logicalDpiX() or 96
        return dpi / 25.4

    def mm_to_screen(self, x_mm: float, y_mm: float) -> QPointF:
        return QPointF(self._origin.x() + x_mm * self._zoom,
                       self._origin.y() + y_mm * self._zoom)

    def screen_to_mm(self, x: float, y: float) -> QPointF:
        return QPointF((x - self._origin.x()) / self._zoom,
                       (y - self._origin.y()) / self._zoom)

    def _screen_coords(self) -> tuple[np.ndarray, np.ndarray]:
        sx = self._origin.x() + self._pos[:, 0] * self._zoom
        sy = self._origin.y() + self._pos[:, 1] * self._zoom
        return sx, sy

    # ------------------------------------------------------------------ #
    # View operations
    # ------------------------------------------------------------------ #
    def _design_size_mm(self) -> tuple[float, float]:
        if self._design is not None:
            return self._design.width_mm, self._design.height_mm
        return 254.0, 203.2

    def _fitted_origin(self) -> QPointF:
        """Origin that centres the artboard — the position ``fit_to_view`` uses."""
        w_mm, h_mm = self._design_size_mm()
        return QPointF((self.width() + _RULER) / 2 - (w_mm * self._zoom) / 2,
                       (self.height() + _RULER) / 2 - (h_mm * self._zoom) / 2)

    def fit_to_view(self) -> None:
        w_mm, h_mm = self._design_size_mm()
        avail_w = max(50, self.width() - 2 * _RULER - 48)
        avail_h = max(50, self.height() - 2 * _RULER - 48)
        zoom = min(avail_w / w_mm, avail_h / h_mm)
        self._set_zoom(zoom)
        # Centre the artboard.
        self._origin = self._fitted_origin()
        self._emit_zoom()
        self.update()
        self._sync_scrollbars()

    def actual_size(self) -> None:
        center = self.screen_to_mm(self.width() / 2, self.height() / 2)
        self._set_zoom(self._actual_zoom())
        self._center_on_mm(center)
        self._emit_zoom()
        self.update()

    def zoom_in(self) -> None:
        self._zoom_about(QPointF(self.width() / 2, self.height() / 2), 1.25)

    def zoom_out(self) -> None:
        self._zoom_about(QPointF(self.width() / 2, self.height() / 2), 1 / 1.25)

    def _set_zoom(self, zoom: float) -> None:
        self._zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, zoom))

    def _zoom_about(self, screen_pt: QPointF, factor: float) -> None:
        before = self.screen_to_mm(screen_pt.x(), screen_pt.y())
        self._set_zoom(self._zoom * factor)
        after = self.mm_to_screen(before.x(), before.y())
        self._origin += screen_pt - after
        # Zooming out near an edge slides the artboard back into reach instead
        # of leaving the user staring at empty canvas (as Photoshop does).
        self._clamp_origin()
        self._emit_zoom()
        self.update()
        self._sync_scrollbars()

    def _center_on_mm(self, mm: QPointF) -> None:
        target = QPointF(self.width() / 2, self.height() / 2)
        cur = self.mm_to_screen(mm.x(), mm.y())
        self._origin += target - cur
        self._clamp_origin()
        self._sync_scrollbars()

    def _emit_zoom(self) -> None:
        self.zoom_changed.emit(self._zoom / self._actual_zoom() * 100.0)

    # ------------------------------------------------------------------ #
    # Scrolling (Photoshop-style rails + touchpad panning)
    # ------------------------------------------------------------------ #
    def attach_scrollbars(self, hbar: QScrollBar, vbar: QScrollBar) -> None:
        """Bind the rails supplied by :class:`CanvasScrollHost`."""
        self._hbar, self._vbar = hbar, vbar
        hbar.valueChanged.connect(lambda _value: self._on_scrollbar())
        vbar.valueChanged.connect(lambda _value: self._on_scrollbar())
        self._sync_scrollbars()

    def set_wheel_zooms(self, on: bool) -> None:
        """Preferences: plain wheel zooms instead of scrolling the view."""
        self._wheel_zooms = bool(on)

    def _pan_bounds(self) -> tuple[float, float, float, float]:
        """Allowed origin box ``(min_x, max_x, min_y, max_y)`` in screen px.

        Photoshop's canvas slack: the artboard may be panned until either of its
        edges reaches the viewport centre — never entirely out of sight. The
        fitted (centred) position is always inside the box, so ``fit_to_view``
        and the clamped view agree.
        """
        w_mm, h_mm = self._design_size_mm()
        cw, ch = w_mm * self._zoom, h_mm * self._zoom
        fit = self._fitted_origin()
        vw, vh = float(self.width()), float(self.height())
        return (min(vw / 2 - cw, fit.x()), max(vw / 2, fit.x()),
                min(vh / 2 - ch, fit.y()), max(vh / 2, fit.y()))

    def _clamp_origin(self) -> None:
        """Keep the artboard within panning reach (no-op before a design exists)."""
        if self._design is None or self.width() <= 0 or self.height() <= 0:
            return
        lo_x, hi_x, lo_y, hi_y = self._pan_bounds()
        self._origin = QPointF(min(max(self._origin.x(), lo_x), hi_x),
                               min(max(self._origin.y(), lo_y), hi_y))

    def _scrollable(self) -> bool:
        return self._design is not None and self.width() > 0 and self.height() > 0

    def _sync_scrollbars(self) -> None:
        """Push the current zoom/pan into the rails (range, page step, position).

        The rail spans one document plus one viewport, the standard mapping: the
        value counts pixels scrolled from "artboard centred left" (0) to
        "centred right" (content px), with ``pageStep = viewport``. When the
        whole design fits, the range collapses to 0 so the handle fills the rail
        — Photoshop's "nothing to scroll" look.
        """
        if self._hbar is None or self._vbar is None or self._syncing_bars:
            return
        if self.width() <= 0 or self.height() <= 0:
            return
        w_mm, h_mm = self._design_size_mm()
        self._syncing_bars = True
        try:
            for bar, content_mm, view, origin in (
                (self._hbar, w_mm, float(self.width()), self._origin.x()),
                (self._vbar, h_mm, float(self.height()), self._origin.y()),
            ):
                bar.setPageStep(max(1, int(view)))
                bar.setSingleStep(max(16, int(view / 10)))
                content = content_mm * self._zoom
                if self._scrollable() and content > view:
                    bar.setRange(0, int(round(content)))
                    bar.setValue(int(round(view / 2.0 - origin)))
                else:
                    bar.setRange(0, 0)
        finally:
            self._syncing_bars = False

    def _on_scrollbar(self) -> None:
        """A rail moved: adopt it as the new pan position."""
        if self._hbar is None or self._vbar is None or self._syncing_bars:
            return
        _lo_x, hi_x, _lo_y, hi_y = self._pan_bounds()
        self._origin = QPointF(hi_x - self._hbar.value(), hi_y - self._vbar.value())
        self._clamp_origin()
        self.update()
        self._sync_scrollbars()

    def scroll_by_pixels(self, dx: float, dy: float) -> None:
        """Pan by a scroll delta in screen px — the content follows the gesture."""
        if not dx and not dy:
            return
        self._origin = QPointF(self._origin.x() + dx, self._origin.y() + dy)
        self._clamp_origin()
        self.update()
        self._sync_scrollbars()

    @staticmethod
    def _wheel_delta_px(event: QWheelEvent) -> tuple[float, float]:
        """Wheel/touchpad delta in screen px, content-follows-gesture sign.

        Touchpads report ``pixelDelta`` (smooth, two axes); a mouse notch only
        reports ``angleDelta``, which we scale to :data:`_WHEEL_STEP_PX`.
        """
        pixel = event.pixelDelta()
        if not pixel.isNull():
            dx, dy = float(pixel.x()), float(pixel.y())
        else:
            angle = event.angleDelta()
            dx = angle.x() / 120.0 * _WHEEL_STEP_PX
            dy = angle.y() / 120.0 * _WHEEL_STEP_PX
        if event.modifiers() & Qt.ShiftModifier and not dx:
            dx, dy = dy, 0.0  # Shift+wheel scrolls sideways, as in Photoshop.
        return dx, dy

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def select_all(self) -> None:
        if self._design is None:
            return
        self._design.select_all()
        self._selected[:] = True
        self._emit_selection()
        self.update()

    def clear_selection(self) -> None:
        if self._design is None:
            return
        self._design.clear_selection()
        if self._selected.size:
            self._selected[:] = False
        self._emit_selection()
        self.update()

    def selected_count(self) -> int:
        return int(self._selected.sum()) if self._selected.size else 0

    def _emit_selection(self) -> None:
        self.selection_changed.emit(self.selected_count())

    def _apply_selection(self, mask: np.ndarray, additive: bool) -> None:
        if not additive and self._selected.size:
            self._selected[:] = False
        self._selected |= mask
        if self._design is not None:
            for i, s in enumerate(self._design.stones):
                s.selected = bool(self._selected[i])
        self._emit_selection()
        self.update()

    def _hit_test(self, mm: QPointF, tolerance_px: float = 3.0) -> int | None:
        """Return the index of the stone under ``mm`` (nearest within radius)."""
        if self._pos.shape[0] == 0:
            return None
        dx = self._pos[:, 0] - mm.x()
        dy = self._pos[:, 1] - mm.y()
        dist_sq = dx * dx + dy * dy
        idx = int(np.argmin(dist_sq))
        tol_mm = tolerance_px / self._zoom
        reach = self._dia[idx] / 2.0 + tol_mm
        if dist_sq[idx] <= reach * reach:
            return idx
        return None

    # ------------------------------------------------------------------ #
    # Editing operations (each emits a history Command via command_produced)
    # ------------------------------------------------------------------ #
    def _would_collide(self, x_mm: float, y_mm: float, diameter: float) -> bool:
        """True if a stone at ``(x, y)`` would overlap an existing one."""
        if self._pos.shape[0] == 0:
            return False
        required = (self._dia + diameter) / 2.0 + self._min_gap_mm
        dx = self._pos[:, 0] - x_mm
        dy = self._pos[:, 1] - y_mm
        return bool(np.any(dx * dx + dy * dy < required * required - 1e-6))

    def add_stone_at(self, mm: QPointF, *, respect_collision: bool = True) -> bool:
        """Add a stone at ``mm``. Returns False if it would collide (and is
        rejected). Emits an :class:`AddStonesCommand`."""
        if self._design is None:
            return False
        if not self._design.canvas_rect.contains(_as_point(mm)):
            return False
        if respect_collision and self._would_collide(mm.x(), mm.y(), self._new_stone_diameter):
            return False
        stone = Stone(
            self._design.allocate_id(), mm.x(), mm.y(),
            self._new_stone_diameter, self._new_stone_size_id, self._new_stone_color,
        )
        cmd = AddStonesCommand(self._design, [stone])
        cmd.redo()
        self.command_produced.emit(cmd)
        self.refresh_geometry()
        self.stones_mutated.emit()
        self.update()
        return True

    def delete_selected(self) -> int:
        """Delete selected stones. Returns the count deleted."""
        if self._design is None:
            return 0
        ids = [s.id for s in self._design.stones if s.selected]
        if not ids:
            return 0
        cmd = DeleteStonesCommand(self._design, ids)
        cmd.redo()
        self.command_produced.emit(cmd)
        self.refresh_geometry()
        self._emit_selection()
        self.stones_mutated.emit()
        self.update()
        return len(ids)

    def _delete_at(self, mm: QPointF) -> bool:
        idx = self._hit_test(mm)
        if idx is None or self._design is None:
            return False
        sid = int(self._ids[idx])
        cmd = DeleteStonesCommand(self._design, [sid])
        cmd.redo()
        self.command_produced.emit(cmd)
        self.refresh_geometry()
        self.stones_mutated.emit()
        self.update()
        return True

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        p = self._theme.palette
        painter.fillRect(self.rect(), QColor(p.canvas_bg))

        if self._design is None:
            painter.end()
            return

        w_mm, h_mm = self._design_size_mm()
        tl = self.mm_to_screen(0, 0)
        art = QRectF(tl.x(), tl.y(), w_mm * self._zoom, h_mm * self._zoom)

        self._paint_artboard(painter, art)
        if self._show_image and self._bg_pixmap is not None:
            painter.save()
            painter.setClipRect(art)
            painter.setOpacity(self._bg_opacity)
            painter.drawPixmap(art.toRect(), self._bg_pixmap)
            painter.restore()
        if self._show_grid:
            self._paint_grid(painter, art, w_mm, h_mm)

        self._paint_trace_overlay(painter, art)
        self._paint_stones(painter, art)
        # Above the stones, deliberately: the outline band runs exactly where
        # the outline row of stones sits, so drawn underneath it would be the
        # one region this overlay exists to show. The overlay is an explicit,
        # off-by-default analysis view, and it never affects the design.
        self._paint_fill_areas(painter, art)
        self._paint_rubber_band(painter)
        self._paint_trace_cursor(painter)
        self._paint_ai_preview(painter)
        if self._crop_mode:
            self._paint_crop_overlay(painter, art)
        if self._show_rulers:
            self._paint_rulers(painter, w_mm, h_mm)

        if self._pos.shape[0] == 0:
            self._paint_empty_hint(painter)
        painter.end()

    def _paint_artboard(self, painter: QPainter, art: QRectF) -> None:
        p = self._theme.palette
        shadow = QColor(0, 0, 0, 60)
        painter.setPen(Qt.NoPen)
        painter.setBrush(shadow)
        painter.drawRoundedRect(art.translated(0, 3), 2, 2)
        # The artboard is a light "paper" surface in both themes so dark and
        # coloured stones (and the image overlay) read with strong contrast —
        # the standard document metaphor in professional design tools.
        painter.setBrush(QColor(_ARTBOARD_COLOR))
        painter.setPen(QPen(QColor(p.border), 1))
        painter.drawRect(art)

    def _paint_grid(self, painter: QPainter, art: QRectF, w_mm: float, h_mm: float) -> None:
        step_mm = self._nice_step(60 / self._zoom)
        pen = QPen(QColor(_GRID_COLOR))
        pen.setWidthF(0.6)
        painter.setPen(pen)
        painter.setClipRect(art)
        x = 0.0
        while x <= w_mm + 1e-6:
            sx = art.left() + x * self._zoom
            painter.drawLine(QPointF(sx, art.top()), QPointF(sx, art.bottom()))
            x += step_mm
        y = 0.0
        while y <= h_mm + 1e-6:
            sy = art.top() + y * self._zoom
            painter.drawLine(QPointF(art.left(), sy), QPointF(art.right(), sy))
            y += step_mm
        painter.setClipping(False)

    def _paint_stones(self, painter: QPainter, art: QRectF) -> None:
        if self._pos.shape[0] == 0:
            return
        sx, sy = self._screen_coords()
        r_px = self._dia * 0.5 * self._zoom

        # Cull to the viewport (with a margin of one radius).
        margin = float(r_px.max()) + 2 if r_px.size else 2
        vis = (
            (sx >= -margin) & (sx <= self.width() + margin)
            & (sy >= -margin) & (sy <= self.height() + margin)
        )
        if not vis.any():
            return

        diameter_px = max(1.0, float(np.median(r_px[vis])) * 2.0)
        if diameter_px < 3.0:
            self._paint_stones_fast(painter, sx, sy, vis)
        else:
            self._paint_stones_stamped(painter, sx, sy, r_px, vis)

        self._paint_stone_overlays(painter, sx, sy, r_px, vis)

    def _paint_stones_fast(
        self, painter: QPainter, sx: np.ndarray, sy: np.ndarray, vis: np.ndarray
    ) -> None:
        """Composite tiny stones into one QImage with NumPy and blit once.

        Draws through the caller's active painter — a widget may only have one
        active :class:`QPainter` at a time.
        """
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        for ci, color in enumerate(self._palette):
            mask = vis & (self._color_idx == ci)
            if not mask.any():
                continue
            xs = np.round(sx[mask]).astype(np.int64)
            ys = np.round(sy[mask]).astype(np.int64)
            inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs, ys = xs[inb], ys[inb]
            if xs.size == 0:
                continue
            rgba = (color.red(), color.green(), color.blue(), 255)
            overlay[ys, xs] = rgba
            # Thicken to a 2px block so single pixels remain visible.
            xr = np.clip(xs + 1, 0, w - 1)
            yr = np.clip(ys + 1, 0, h - 1)
            overlay[ys, xr] = rgba
            overlay[yr, xs] = rgba
            overlay[yr, xr] = rgba
        img = numpy_rgba_to_qimage(overlay)
        painter.drawImage(0, 0, img)

    def _paint_stones_stamped(
        self, painter: QPainter, sx: np.ndarray, sy: np.ndarray, r_px: np.ndarray, vis: np.ndarray
    ) -> None:
        aa = self._zoom > 2.0
        painter.setRenderHint(QPainter.Antialiasing, aa)
        for ci, color in enumerate(self._palette):
            mask = vis & (self._color_idx == ci)
            if not mask.any():
                continue
            d = max(2, int(round(float(np.median(r_px[mask])) * 2.0)))
            stamp = self._disc_stamp(color, d)
            half = stamp.width() / 2.0
            xs = sx[mask]
            ys = sy[mask]
            for k in range(xs.size):
                painter.drawPixmap(QPointF(xs[k] - half, ys[k] - half), stamp)

    def _paint_stone_overlays(
        self, painter: QPainter, sx: np.ndarray, sy: np.ndarray, r_px: np.ndarray, vis: np.ndarray
    ) -> None:
        """Draw selection rings and collision highlights over the stones."""
        pal = self._theme.palette
        painter.setRenderHint(QPainter.Antialiasing, True)

        coll = vis & self._collision if self._collision.size else np.zeros_like(vis)
        if coll.any():
            painter.setBrush(Qt.NoBrush)
            pen = QPen(QColor(pal.error))
            pen.setWidthF(max(1.5, 2.0))
            painter.setPen(pen)
            for k in np.flatnonzero(coll):
                r = max(2.5, r_px[k] + 1.5)
                painter.drawEllipse(QPointF(sx[k], sy[k]), r, r)

        sel = vis & self._selected if self._selected.size else np.zeros_like(vis)
        if sel.any():
            painter.setBrush(Qt.NoBrush)
            pen = QPen(QColor(pal.primary))
            pen.setWidthF(2.0)
            painter.setPen(pen)
            for k in np.flatnonzero(sel):
                r = max(3.0, r_px[k] + 2.0)
                painter.drawEllipse(QPointF(sx[k], sy[k]), r, r)

    def _disc_stamp(self, color: QColor, diameter_px: int) -> QPixmap:
        key = (color.rgb(), diameter_px)
        stamp = self._stamp_cache.get(key)
        if stamp is not None:
            return stamp
        pad = 2
        size = diameter_px + pad * 2
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        pr = QPainter(pm)
        pr.setRenderHint(QPainter.Antialiasing, True)
        pr.setPen(Qt.NoPen)
        pr.setBrush(color)
        pr.drawEllipse(QRectF(pad, pad, diameter_px, diameter_px))
        pr.end()
        if len(self._stamp_cache) > 64:
            self._stamp_cache.clear()
        self._stamp_cache[key] = pm
        return pm

    def _paint_crop_overlay(self, painter: QPainter, art: QRectF) -> None:
        """Dim the area outside the crop rectangle and draw its frame + guides."""
        if self._crop_rect_mm is None:
            return
        tl = self.mm_to_screen(self._crop_rect_mm.left(), self._crop_rect_mm.top())
        br = self.mm_to_screen(self._crop_rect_mm.right(), self._crop_rect_mm.bottom())
        crop = QRectF(tl, br).normalized()

        # Dim everything on the artboard outside the crop rectangle.
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 110))
        top = QRectF(art.left(), art.top(), art.width(), crop.top() - art.top())
        bottom = QRectF(art.left(), crop.bottom(), art.width(), art.bottom() - crop.bottom())
        left = QRectF(art.left(), crop.top(), crop.left() - art.left(), crop.height())
        right = QRectF(crop.right(), crop.top(), art.right() - crop.right(), crop.height())
        for band in (top, bottom, left, right):
            if band.width() > 0 and band.height() > 0:
                painter.drawRect(band)
        painter.restore()

        # Crop frame + rule-of-thirds guides.
        pen = QPen(QColor("#ffffff"))
        pen.setWidthF(1.5)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(pen)
        painter.drawRect(crop)
        guide = QPen(QColor(255, 255, 255, 90))
        guide.setWidthF(0.8)
        painter.setPen(guide)
        for i in (1, 2):
            gx = crop.left() + crop.width() * i / 3.0
            gy = crop.top() + crop.height() * i / 3.0
            painter.drawLine(QPointF(gx, crop.top()), QPointF(gx, crop.bottom()))
            painter.drawLine(QPointF(crop.left(), gy), QPointF(crop.right(), gy))

    def _paint_rubber_band(self, painter: QPainter) -> None:
        if self._rubber_start is None or self._rubber_now is None:
            return
        pal = self._theme.palette
        rect = QRectF(self._rubber_start, self._rubber_now).normalized()
        fill = QColor(pal.primary)
        fill.setAlpha(40)
        painter.setBrush(fill)
        pen = QPen(QColor(pal.primary))
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)

    def _paint_rulers(self, painter: QPainter, w_mm: float, h_mm: float) -> None:
        pal = self._theme.palette
        painter.setRenderHint(QPainter.Antialiasing, False)
        bg = QColor(pal.surface)
        painter.fillRect(QRect(0, 0, self.width(), _RULER), bg)
        painter.fillRect(QRect(0, 0, _RULER, self.height()), bg)
        painter.setPen(QPen(QColor(pal.border), 1))
        painter.drawLine(0, _RULER, self.width(), _RULER)
        painter.drawLine(_RULER, 0, _RULER, self.height())

        font = QFont(self.font())
        font.setPixelSize(9)
        painter.setFont(font)
        painter.setPen(QColor(pal.muted_text))

        step_units = self._nice_step(70 / (self._zoom * self._unit.mm_per_unit))
        step_mm = step_units * self._unit.mm_per_unit

        # Horizontal ruler.
        x_mm = 0.0
        while x_mm <= w_mm + 1e-6:
            sx = self.mm_to_screen(x_mm, 0).x()
            if sx >= _RULER:
                painter.drawLine(QPointF(sx, _RULER - 5), QPointF(sx, _RULER))
                label = f"{x_mm / self._unit.mm_per_unit:g}"
                painter.drawText(QPointF(sx + 2, _RULER - 7), label)
            x_mm += step_mm
        # Vertical ruler.
        y_mm = 0.0
        while y_mm <= h_mm + 1e-6:
            sy = self.mm_to_screen(0, y_mm).y()
            if sy >= _RULER:
                painter.drawLine(QPointF(_RULER - 5, sy), QPointF(_RULER, sy))
                painter.save()
                painter.translate(_RULER - 8, sy - 2)
                painter.rotate(-90)
                painter.drawText(QPointF(0, 0), f"{y_mm / self._unit.mm_per_unit:g}")
                painter.restore()
            y_mm += step_mm

    def _paint_empty_hint(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QColor(self._theme.palette.muted_text))
        font = QFont(self.font())
        font.setPixelSize(15)
        painter.setFont(font)
        painter.drawText(
            self.rect().adjusted(_RULER, _RULER, 0, 0),
            Qt.AlignCenter,
            "Press Generate to create your rhinestone design",
        )

    @staticmethod
    def _nice_step(raw: float) -> float:
        """Round ``raw`` up to a 1/2/5×10ⁿ 'nice' number for grid/ruler steps."""
        if raw <= 0:
            return 1.0
        exp = math.floor(math.log10(raw))
        base = 10 ** exp
        for m in (1, 2, 5, 10):
            if raw <= m * base:
                return m * base
        return 10 * base

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def _update_cursor(self) -> None:
        if self._space_down or self._tool is CanvasTool.PAN:
            self.setCursor(Qt.OpenHandCursor)
        elif self._tool in (CanvasTool.ADD, CanvasTool.BRUSH, CanvasTool.CROP, CanvasTool.TRACE):
            self.setCursor(Qt.CrossCursor)
        elif self._tool in (CanvasTool.DELETE, CanvasTool.ERASER):
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        # Photoshop convention: scrolling pans the canvas, Ctrl/Alt+scroll zooms
        # about the cursor. A laptop touchpad's two-finger scroll reaches us as
        # the same event as a wheel, so this is the only way both gestures can
        # coexist — the preference flips it back for wheel-zoom muscle memory.
        if self._wheel_zooms or event.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            delta = event.angleDelta().y()
            if delta == 0:
                return
            self._zoom_about(event.position(), 1.0015 ** delta)
            return
        dx, dy = self._wheel_delta_px(event)
        if dx or dy:
            self.scroll_by_pixels(dx, dy)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        mm = self.screen_to_mm(pos.x(), pos.y())
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and (self._space_down or self._tool is CanvasTool.PAN)
        ):
            self._panning = True
            self._last_pos = pos
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            # The Trace tool's erase stroke rides on the right button.
            if event.button() == Qt.RightButton and self._tool is CanvasTool.TRACE:
                self._begin_trace(mm, erase=True)
            return

        tool = self._tool
        if tool is CanvasTool.CROP:
            # Record the anchor; the rect is (re)built once the mouse moves, so a
            # bare click (e.g. the first press of a double-click-to-commit) keeps
            # the box the user already dragged.
            self._crop_anchor_mm = self._clamp_crop_point(mm)
            self._crop_dragging = True
            return
        if tool is CanvasTool.ADD:
            self.add_stone_at(mm)
            return
        if tool is CanvasTool.DELETE:
            self._delete_at(mm)
            return
        if tool in (CanvasTool.BRUSH, CanvasTool.ERASER):
            self._begin_brush(tool)
            self._apply_brush(mm, tool)
            return
        if tool is CanvasTool.TRACE:
            self._begin_trace(mm, erase=False)
            return
        # SELECT: either start dragging a hit stone, or begin a rubber band.
        idx = self._hit_test(mm)
        if idx is not None and self._selected.size and self._selected[idx]:
            self._begin_move(mm)
        elif idx is not None:
            additive = bool(event.modifiers() & Qt.ShiftModifier)
            self._select_at_point(pos, additive)
            self._begin_move(mm)
        else:
            self._rubber_start = pos
            self._rubber_now = pos

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        mm = self.screen_to_mm(pos.x(), pos.y())
        self.cursor_moved.emit(mm.x(), mm.y())
        if self._tool is CanvasTool.TRACE:
            self._trace_hover_mm = QPointF(mm)
        if self._panning and self._last_pos is not None:
            self._origin += pos - self._last_pos
            self._last_pos = pos
            self._clamp_origin()
            self.update()
            self._sync_scrollbars()
            return
        if self._crop_dragging and self._crop_anchor_mm is not None:
            p = self._clamp_crop_point(mm)
            self._crop_rect_mm = QRectF(self._crop_anchor_mm, p).normalized()
            self.update()
            return
        if self._trace_stroke_before is not None:
            self._apply_trace(mm)
            return
        if self._brush_active:
            self._apply_brush(mm, self._tool)
            return
        if self._dragging_selection and self._drag_anchor_mm is not None:
            self._update_move(mm)
            return
        if self._rubber_start is not None:
            self._rubber_now = pos
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            self._panning = False
            self._last_pos = None
            self._update_cursor()
            return
        if self._crop_dragging:
            self._crop_dragging = False
            self._crop_anchor_mm = None
            self.update()
            return
        if self._trace_stroke_before is not None and event.button() in (
            Qt.LeftButton, Qt.RightButton
        ):
            self._end_trace()
            return
        if self._brush_active:
            self._end_brush()
            return
        if self._dragging_selection:
            self._end_move()
            return
        if self._rubber_start is not None and self._tool is CanvasTool.SELECT:
            start = self._rubber_start
            end = event.position()
            additive = bool(event.modifiers() & Qt.ShiftModifier)
            moved = (abs(end.x() - start.x()) + abs(end.y() - start.y())) > 4
            if moved:
                self._select_in_rect(start, end, additive)
            else:
                self._select_at_point(end, additive)
            self._rubber_start = None
            self._rubber_now = None
            self.update()

    # -- move-by-drag ------------------------------------------------------- #
    def _begin_move(self, mm: QPointF) -> None:
        if self._design is None:
            return
        self._dragging_selection = True
        self._drag_anchor_mm = mm
        self._drag_start_positions = {
            s.id: (s.x, s.y) for s in self._design.stones if s.selected
        }
        if self._push_all:
            # Snapshot every stone so each mouse move recomputes the push
            # deterministically from the last valid layout.
            self._push_snapshot = self._pos.copy()
            self._push_failed = False
        self.setCursor(Qt.ClosedHandCursor)

    def _update_move(self, mm: QPointF) -> None:
        if self._design is None or self._drag_start_positions is None or self._drag_anchor_mm is None:
            return
        dx = mm.x() - self._drag_anchor_mm.x()
        dy = mm.y() - self._drag_anchor_mm.y()
        w_mm, h_mm = self._design_size_mm()
        if self._push_all and self._push_snapshot is not None:
            self._restore_push_snapshot()
        for i, s in enumerate(self._design.stones):
            start = self._drag_start_positions.get(s.id)
            if start is None:
                continue
            nx = min(max(start[0] + dx, 0.0), w_mm)
            ny = min(max(start[1] + dy, 0.0), h_mm)
            s.x, s.y = nx, ny
            self._pos[i, 0] = nx
            self._pos[i, 1] = ny
        if self._push_all:
            self._apply_push_propagation()
        self.update()

    def _restore_push_snapshot(self) -> None:
        """Put every stone back at its pre-drag position (only changed ones touched)."""
        if self._push_snapshot is None or self._design is None:
            return
        snap = self._push_snapshot
        if snap.shape != self._pos.shape:
            return
        changed = np.any(self._pos != snap, axis=1)
        for i in np.flatnonzero(changed):
            s = self._design.stones[i]
            s.x, s.y = float(snap[i, 0]), float(snap[i, 1])
            self._pos[i, 0] = snap[i, 0]
            self._pos[i, 1] = snap[i, 1]

    def _apply_push_propagation(self) -> None:
        """Push neighbours away from the dragged stones to keep the spacing."""
        from app.core.push import propagate_push

        if self._design is None or self._pos.shape[0] == 0:
            return
        n = self._pos.shape[0]
        pinned = np.zeros(n, dtype=bool)
        dragged_ids = self._drag_start_positions or {}
        for i, s in enumerate(self._design.stones):
            if s.id in dragged_ids:
                pinned[i] = True
        if not pinned.any():
            return
        result = propagate_push(
            self._pos,
            self._dia,
            self._min_gap_mm,
            pinned,
            (self._design.width_mm, self._design.height_mm),
        )
        if not result.converged:
            # The movement cannot satisfy the spacing constraint: restore the
            # previous valid layout and tell the user.
            self._restore_push_snapshot()
            self._push_failed = True
            self.push_failed.emit()
            return
        self._push_failed = False
        for k in result.pushed_indices:
            s = self._design.stones[k]
            s.x, s.y = float(result.positions[k, 0]), float(result.positions[k, 1])
            self._pos[k, 0] = s.x
            self._pos[k, 1] = s.y

    def _end_move(self) -> None:
        self._dragging_selection = False
        self._update_cursor()
        if self._design is None or self._drag_start_positions is None:
            return
        if self._push_failed:
            # Impossible movement: the layout was already restored live, so
            # there is nothing to record in history.
            self._drag_start_positions = None
            self._drag_anchor_mm = None
            self._push_snapshot = None
            self._push_failed = False
            return
        moves: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
        if self._push_all and self._push_snapshot is not None and self._push_snapshot.shape == self._pos.shape:
            # One command covering the drag AND every pushed neighbour, so a
            # single Ctrl+Z reverts the whole Push All operation.
            snap = self._push_snapshot
            for i, s in enumerate(self._design.stones):
                ox, oy = float(snap[i, 0]), float(snap[i, 1])
                if abs(ox - s.x) > 1e-6 or abs(oy - s.y) > 1e-6:
                    moves[s.id] = ((ox, oy), (s.x, s.y))
        else:
            for s in self._design.stones:
                start = self._drag_start_positions.get(s.id)
                if start is not None and (abs(start[0] - s.x) > 1e-6 or abs(start[1] - s.y) > 1e-6):
                    moves[s.id] = (start, (s.x, s.y))
        self._drag_start_positions = None
        self._drag_anchor_mm = None
        self._push_snapshot = None
        if moves:
            # The move is already applied; record it so it can be undone.
            self.command_produced.emit(MoveStonesCommand(self._design, moves))
            self.stones_mutated.emit()

    # -- brush / eraser ----------------------------------------------------- #
    def _begin_brush(self, tool: CanvasTool) -> None:
        self._brush_active = True
        self._brush_touched_ids = set()
        self._brush_added = []

    def _apply_brush(self, mm: QPointF, tool: CanvasTool) -> None:
        if self._design is None:
            return
        if tool is CanvasTool.ERASER:
            self._erase_near(mm)
        else:
            self._brush_add(mm)

    def _brush_add(self, mm: QPointF) -> None:
        """Add stones on a hex lattice within the brush radius, no overlaps."""
        if self._design is None:
            return
        pitch = self._new_stone_diameter + self._min_gap_mm
        r = self._brush_radius_mm
        added: list[Stone] = []
        # Sample a small hex grid around the cursor.
        vy = pitch * math.sqrt(3.0) / 2.0
        rows = int(r / vy) + 1
        for ry in range(-rows, rows + 1):
            cy = mm.y() + ry * vy
            offset = (pitch / 2.0) if (ry % 2) else 0.0
            cols = int(r / pitch) + 1
            for rx in range(-cols, cols + 1):
                cx = mm.x() + rx * pitch + offset
                if (cx - mm.x()) ** 2 + (cy - mm.y()) ** 2 > r * r:
                    continue
                if not (0 <= cx <= self._design.width_mm and 0 <= cy <= self._design.height_mm):
                    continue
                if self._would_collide(cx, cy, self._new_stone_diameter):
                    continue
                stone = Stone(self._design.allocate_id(), cx, cy,
                              self._new_stone_diameter, self._new_stone_size_id, self._new_stone_color)
                self._design.add(stone)
                added.append(stone)
                self._brush_added.append(stone)
                # Update cache incrementally so subsequent collision checks see it.
                self._append_to_cache(stone)
        if added:
            self.update()

    def _erase_near(self, mm: QPointF) -> None:
        if self._design is None or self._pos.shape[0] == 0:
            return
        r = self._brush_radius_mm
        dx = self._pos[:, 0] - mm.x()
        dy = self._pos[:, 1] - mm.y()
        hit = np.flatnonzero(dx * dx + dy * dy <= r * r)
        if hit.size == 0:
            return
        ids = [int(self._ids[i]) for i in hit]
        removed = self._design.remove_many(ids)
        for s in removed:
            self._brush_touched_ids.add(s.id)
            if s.id not in getattr(self, "_erased_store", {}):
                self._erased_store = getattr(self, "_erased_store", {})
                self._erased_store[s.id] = s
        self.refresh_geometry()
        self.update()

    def _end_brush(self) -> None:
        self._brush_active = False
        if self._design is None:
            return
        if self._tool is CanvasTool.BRUSH and self._brush_added:
            self.refresh_geometry()
            self.command_produced.emit(AddStonesCommand(self._design, list(self._brush_added)))
            self.stones_mutated.emit()
        elif self._tool is CanvasTool.ERASER and self._brush_touched_ids:
            store = getattr(self, "_erased_store", {})
            removed = [store[i] for i in self._brush_touched_ids if i in store]
            if removed:
                self.command_produced.emit(
                    DeleteStonesCommand.from_removed(
                        self._design, removed, label=f"Erase {len(removed)} stones"
                    )
                )
            self._erased_store = {}
            self.stones_mutated.emit()
        self._brush_added = []
        self._brush_touched_ids = set()

    def _append_to_cache(self, stone: Stone) -> None:
        """Append a single stone to the render cache without a full rebuild."""
        self._pos = np.vstack([self._pos, [stone.x, stone.y]])
        self._dia = np.append(self._dia, stone.diameter)
        self._ids = np.append(self._ids, stone.id)
        self._selected = np.append(self._selected, False)
        self._collision = np.append(self._collision, False)
        ci = None
        for i, c in enumerate(self._palette):
            if c.rgb() == QColor(stone.color).rgb():
                ci = i
                break
        if ci is None:
            ci = len(self._palette)
            self._palette.append(QColor(stone.color))
        self._color_idx = np.append(self._color_idx, ci)
        self._id_to_index[stone.id] = self._pos.shape[0] - 1

    def _select_in_rect(self, a: QPointF, b: QPointF, additive: bool) -> None:
        mm_a = self.screen_to_mm(min(a.x(), b.x()), min(a.y(), b.y()))
        mm_b = self.screen_to_mm(max(a.x(), b.x()), max(a.y(), b.y()))
        if self._pos.shape[0] == 0:
            return
        mask = (
            (self._pos[:, 0] >= mm_a.x()) & (self._pos[:, 0] <= mm_b.x())
            & (self._pos[:, 1] >= mm_a.y()) & (self._pos[:, 1] <= mm_b.y())
        )
        self._apply_selection(mask, additive)

    def _select_at_point(self, pt: QPointF, additive: bool) -> None:
        mm = self.screen_to_mm(pt.x(), pt.y())
        idx = self._hit_test(mm)
        mask = np.zeros(self._pos.shape[0], dtype=bool)
        if idx is not None:
            mask[idx] = True
        self._apply_selection(mask, additive)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._crop_mode and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.commit_crop()
        elif self._crop_mode and event.key() == Qt.Key_Escape:
            self.cancel_crop()
        elif event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self._update_cursor()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._crop_mode:
            self.commit_crop()
            return
        super().mouseDoubleClickEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self._update_cursor()
        else:
            super().keyReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._clamp_origin()
        self._sync_scrollbars()


class CanvasScrollHost(QWidget):
    """The canvas plus its Photoshop-style scrollbar rails.

    The rails live *outside* the canvas widget so the canvas keeps its own
    coordinate space — every ``self.width()``/``self.height()`` use inside
    :class:`CanvasView` still describes the full drawing area. The canvas owns
    the scroll model; the host only supplies the chrome.
    """

    def __init__(self, canvas: CanvasView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = canvas
        self.hbar = QScrollBar(Qt.Horizontal, self)
        self.vbar = QScrollBar(Qt.Vertical, self)
        self.hbar.setToolTip("Scroll horizontally — or two-finger scroll on the canvas")
        self.vbar.setToolTip("Scroll vertically — or two-finger scroll on the canvas")
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(canvas, 0, 0)
        grid.addWidget(self.vbar, 0, 1)
        grid.addWidget(self.hbar, 1, 0)
        canvas.attach_scrollbars(self.hbar, self.vbar)
