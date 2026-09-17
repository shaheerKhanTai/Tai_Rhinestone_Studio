"""Vector icon set drawn with QPainter — no external image assets.

Icons are simple 2px line glyphs on a 24×24 grid, rendered to a crisp pixmap at
request time and tinted with a theme colour. Drawing them in code keeps the app
dependency-free, sharp at any DPI, and recolourable per theme. The app logo is a
faceted rhinestone gem.
"""
from __future__ import annotations

import math
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)

_GRID = 24.0


def _new_pixmap(size: int) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    return pm


def _painter(pm: QPixmap, color: QColor, width: float = 2.0) -> QPainter:
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    scale = pm.width() / _GRID
    p.scale(scale, scale)
    pen = QPen(color, width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    return p


# --------------------------------------------------------------------------- #
# Glyph drawing functions (24×24 coordinate space)
# --------------------------------------------------------------------------- #


def _new(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(5, 3, 14, 18), 2, 2)
    p.drawLine(QPointF(12, 8), QPointF(12, 16))
    p.drawLine(QPointF(8, 12), QPointF(16, 12))


def _open(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(3, 6)
    path.lineTo(9, 6)
    path.lineTo(11, 9)
    path.lineTo(21, 9)
    path.lineTo(21, 19)
    path.lineTo(3, 19)
    path.closeSubpath()
    p.drawPath(path)


def _save(p: QPainter) -> None:
    p.drawLine(QPointF(12, 4), QPointF(12, 15))
    path = QPainterPath()
    path.moveTo(8, 11)
    path.lineTo(12, 15)
    path.lineTo(16, 11)
    p.drawPath(path)
    p.drawLine(QPointF(5, 19), QPointF(19, 19))


def _import_image(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3, 5, 18, 14), 2, 2)
    p.drawEllipse(QRectF(6.5, 8, 3, 3))
    path = QPainterPath()
    path.moveTo(5, 17)
    path.lineTo(10, 12)
    path.lineTo(13, 15)
    path.lineTo(16, 11)
    path.lineTo(20, 16)
    p.drawPath(path)


def _export(p: QPainter) -> None:
    p.drawLine(QPointF(12, 4), QPointF(12, 15))
    head = QPainterPath()
    head.moveTo(8, 8)
    head.lineTo(12, 4)
    head.lineTo(16, 8)
    p.drawPath(head)
    path = QPainterPath()
    path.moveTo(5, 13)
    path.lineTo(5, 20)
    path.lineTo(19, 20)
    path.lineTo(19, 13)
    p.drawPath(path)


def _settings(p: QPainter) -> None:
    p.drawEllipse(QRectF(9, 9, 6, 6))
    for i in range(8):
        a = i * math.pi / 4.0
        x, y = 12 + math.cos(a), 12 + math.sin(a)
        p.drawLine(QPointF(12 + 7 * (x - 12), 12 + 7 * (y - 12)),
                   QPointF(12 + 9.5 * (x - 12), 12 + 9.5 * (y - 12)))


def _zoom(p: QPainter, plus: bool | None) -> None:
    p.drawEllipse(QRectF(4, 4, 12, 12))
    p.drawLine(QPointF(14.5, 14.5), QPointF(20, 20))
    if plus is not None:
        p.drawLine(QPointF(7, 10), QPointF(13, 10))
        if plus:
            p.drawLine(QPointF(10, 7), QPointF(10, 13))


def _fit(p: QPainter) -> None:
    for (cx, cy, dx, dy) in [(4, 4, 1, 1), (20, 4, -1, 1), (4, 20, 1, -1), (20, 20, -1, -1)]:
        p.drawLine(QPointF(cx, cy), QPointF(cx + 5 * dx, cy))
        p.drawLine(QPointF(cx, cy), QPointF(cx, cy + 5 * dy))


def _actual_size(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4, 4, 16, 16), 2, 2)
    p.drawRoundedRect(QRectF(9, 9, 6, 6), 1, 1)


def _select(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(5, 4)
    path.lineTo(5, 18)
    path.lineTo(9, 14)
    path.lineTo(12, 20)
    path.lineTo(14, 19)
    path.lineTo(11, 13)
    path.lineTo(16, 13)
    path.closeSubpath()
    p.drawPath(path)


def _move(p: QPainter) -> None:
    p.drawLine(QPointF(12, 4), QPointF(12, 20))
    p.drawLine(QPointF(4, 12), QPointF(20, 12))
    for (ax, ay, bx, by, cx, cy) in [
        (12, 4, 9, 7, 15, 7), (12, 20, 9, 17, 15, 17),
        (4, 12, 7, 9, 7, 15), (20, 12, 17, 9, 17, 15),
    ]:
        p.drawLine(QPointF(ax, ay), QPointF(bx, by))
        p.drawLine(QPointF(ax, ay), QPointF(cx, cy))


def _add_stone(p: QPainter) -> None:
    p.drawEllipse(QRectF(4, 4, 12, 12))
    p.drawLine(QPointF(17, 17), QPointF(21, 21))
    p.drawLine(QPointF(7.5, 10), QPointF(12.5, 10))
    p.drawLine(QPointF(10, 7.5), QPointF(10, 12.5))


def _delete(p: QPainter) -> None:
    p.drawLine(QPointF(5, 6), QPointF(19, 6))
    p.drawLine(QPointF(9, 6), QPointF(9, 4))
    p.drawLine(QPointF(15, 6), QPointF(15, 4))
    path = QPainterPath()
    path.moveTo(6, 6)
    path.lineTo(7, 20)
    path.lineTo(17, 20)
    path.lineTo(18, 6)
    p.drawPath(path)
    p.drawLine(QPointF(10, 10), QPointF(10, 17))
    p.drawLine(QPointF(14, 10), QPointF(14, 17))


def _brush(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(19, 5)
    path.lineTo(11, 13)
    path.lineTo(13, 15)
    path.lineTo(21, 7)
    path.closeSubpath()
    p.drawPath(path)
    tip = QPainterPath()
    tip.moveTo(11, 13)
    tip.lineTo(6, 16)
    tip.lineTo(5, 20)
    tip.lineTo(9, 19)
    tip.lineTo(13, 15)
    p.drawPath(tip)


def _eraser(p: QPainter) -> None:
    p.save()
    p.translate(12, 12)
    p.rotate(-45)
    p.drawRoundedRect(QRectF(-8, -4, 16, 8), 1.5, 1.5)
    p.drawLine(QPointF(0, -4), QPointF(0, 4))
    p.restore()


def _undo(p: QPainter) -> None:
    rect = QRectF(5, 6, 14, 14)
    path = QPainterPath()
    path.arcMoveTo(rect, 150)
    path.arcTo(rect, 150, 160)
    p.drawPath(path)
    p.drawLine(QPointF(5, 9), QPointF(5.5, 4.5))
    p.drawLine(QPointF(5, 9), QPointF(9.5, 8.5))


def _redo(p: QPainter) -> None:
    rect = QRectF(5, 6, 14, 14)
    path = QPainterPath()
    path.arcMoveTo(rect, 30)
    path.arcTo(rect, 30, -160)
    p.drawPath(path)
    p.drawLine(QPointF(19, 9), QPointF(18.5, 4.5))
    p.drawLine(QPointF(19, 9), QPointF(14.5, 8.5))


def _resolve(p: QPainter) -> None:
    p.drawEllipse(QRectF(4, 8, 8, 8))
    p.drawEllipse(QRectF(12, 8, 8, 8))
    p.drawLine(QPointF(9, 5), QPointF(6, 7))
    p.drawLine(QPointF(15, 5), QPointF(18, 7))


def _grid(p: QPainter) -> None:
    for i in range(1, 3):
        v = 4 + i * 16 / 3
        p.drawLine(QPointF(v, 4), QPointF(v, 20))
        p.drawLine(QPointF(4, v), QPointF(20, v))
    p.drawRect(QRectF(4, 4, 16, 16))


def _select_rect(p: QPainter) -> None:
    pen = p.pen()
    pen.setStyle(Qt.DashLine)
    p.setPen(pen)
    p.drawRect(QRectF(4, 6, 16, 12))


def _generate(p: QPainter) -> None:
    # Wand with a sparkle.
    p.drawLine(QPointF(5, 19), QPointF(15, 9))
    for (cx, cy, r) in [(17, 6, 2.6), (19, 12, 1.6), (12, 4, 1.6)]:
        star = QPainterPath()
        for k in range(4):
            a = k * math.pi / 2
            star.moveTo(cx, cy)
            star.lineTo(cx + r * math.cos(a), cy + r * math.sin(a))
        p.drawPath(star)


def _image_toggle(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3, 5, 18, 14), 2, 2)
    p.drawEllipse(QRectF(6, 8, 3, 3))
    path = QPainterPath()
    path.moveTo(5, 17)
    path.lineTo(10, 12)
    path.lineTo(14, 16)
    path.lineTo(17, 13)
    path.lineTo(20, 16)
    p.drawPath(path)


def _rotate(p: QPainter, clockwise: bool) -> None:
    """A circular arrow; the arrowhead points along the direction of travel."""
    cx, cy, r = 12.0, 12.0, 7.6
    a0, a1 = 22.0, 322.0  # degrees clockwise from the top; gap across the top
    if not clockwise:
        a0, a1 = 360.0 - a0, 360.0 - a1  # mirror across the vertical
    steps = 48
    path = QPainterPath()
    for i in range(steps + 1):
        t = math.radians(a0 + (a1 - a0) * i / steps)
        x, y = cx + r * math.sin(t), cy - r * math.cos(t)
        (path.moveTo if i == 0 else path.lineTo)(x, y)
    p.drawPath(path)
    # Arrowhead at the end, barbs swept back from the travel direction.
    tend = math.radians(a1)
    ex, ey = cx + r * math.sin(tend), cy - r * math.cos(tend)
    dx, dy = (1.0 if clockwise else -1.0) * math.cos(tend), (1.0 if clockwise else -1.0) * math.sin(tend)

    def _barb(deg: float) -> QPointF:
        a = math.radians(deg)
        bx, by = -dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) - dy * math.cos(a)
        return QPointF(ex + 4.4 * bx, ey + 4.4 * by)

    p.drawLine(QPointF(ex, ey), _barb(28))
    p.drawLine(QPointF(ex, ey), _barb(-28))


def _flip_h(p: QPainter) -> None:
    p.save()
    pen = p.pen()
    pen.setStyle(Qt.DashLine)
    p.setPen(pen)
    p.drawLine(QPointF(12, 3), QPointF(12, 21))
    p.restore()
    left = QPolygonF([QPointF(9, 7), QPointF(9, 17), QPointF(4, 12)])
    right = QPolygonF([QPointF(15, 7), QPointF(15, 17), QPointF(20, 12)])
    p.drawPolygon(left)
    p.drawPolygon(right)


def _flip_v(p: QPainter) -> None:
    p.save()
    pen = p.pen()
    pen.setStyle(Qt.DashLine)
    p.setPen(pen)
    p.drawLine(QPointF(3, 12), QPointF(21, 12))
    p.restore()
    top = QPolygonF([QPointF(7, 9), QPointF(17, 9), QPointF(12, 4)])
    bottom = QPolygonF([QPointF(7, 15), QPointF(17, 15), QPointF(12, 20)])
    p.drawPolygon(top)
    p.drawPolygon(bottom)


def _crop(p: QPainter) -> None:
    p.drawLine(QPointF(7, 3), QPointF(7, 17))
    p.drawLine(QPointF(7, 17), QPointF(21, 17))
    p.drawLine(QPointF(3, 7), QPointF(17, 7))
    p.drawLine(QPointF(17, 7), QPointF(17, 21))


def _reset(p: QPainter) -> None:
    """A revert arrow: a counter-clockwise loop returning to a start bar."""
    rect = QRectF(6, 5, 13, 13)
    path = QPainterPath()
    path.arcMoveTo(rect, 60)
    path.arcTo(rect, 60, 250)
    p.drawPath(path)
    p.drawLine(QPointF(8.5, 6.5), QPointF(8.0, 11.0))
    p.drawLine(QPointF(8.5, 6.5), QPointF(12.5, 7.0))


def _trace(p: QPainter) -> None:
    """A freehand brush path dotted with stones it will place."""
    path = QPainterPath()
    path.moveTo(4, 18)
    path.cubicTo(QPointF(8, 8), QPointF(14, 22), QPointF(20, 6))
    pen = p.pen()
    pen.setStyle(Qt.DashLine)
    p.setPen(pen)
    p.drawPath(path)
    pen.setStyle(Qt.SolidLine)
    p.setPen(pen)
    for cx, cy, r in [(7, 15, 1.8), (12, 13, 1.8), (16, 9, 1.8)]:
        p.drawEllipse(QPointF(cx, cy), r, r)


_GLYPHS: dict[str, Callable[[QPainter], None]] = {
    "new": _new,
    "open": _open,
    "save": _save,
    "import": _import_image,
    "export": _export,
    "settings": _settings,
    "zoom_in": lambda p: _zoom(p, True),
    "zoom_out": lambda p: _zoom(p, False),
    "fit": _fit,
    "actual_size": _actual_size,
    "select": _select,
    "move": _move,
    "add": _add_stone,
    "delete": _delete,
    "brush": _brush,
    "eraser": _eraser,
    "undo": _undo,
    "redo": _redo,
    "resolve": _resolve,
    "grid": _grid,
    "select_rect": _select_rect,
    "generate": _generate,
    "image": _image_toggle,
    "rotate_left": lambda p: _rotate(p, False),
    "rotate_right": lambda p: _rotate(p, True),
    "flip_h": _flip_h,
    "flip_v": _flip_v,
    "crop": _crop,
    "reset": _reset,
    "trace": _trace,
}


def make_icon(name: str, color: QColor | str, size: int = 48) -> QIcon:
    """Return a tinted :class:`QIcon` for a named glyph.

    Unknown names produce an empty (but valid) icon so the UI never crashes on a
    typo. Render size is generous so Qt downscales crisply on toolbars/menus.
    """
    if isinstance(color, str):
        color = QColor(color)
    pm = _new_pixmap(size)
    glyph = _GLYPHS.get(name)
    if glyph is not None:
        p = _painter(pm, color)
        glyph(p)
        p.end()
    return QIcon(pm)


def make_app_logo(size: int = 256, primary: str = "#4f86d6", accent: str = "#8fc0ff") -> QPixmap:
    """Draw the Rhinestone Studio gem logo as a pixmap."""
    pm = _new_pixmap(size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    s = size / _GRID
    p.scale(s, s)

    top = QColor(primary)
    hi = QColor(accent)

    # Gem outline (rounded hexagon-ish diamond).
    crown = 8.0
    body = QPolygonF([
        QPointF(6, crown), QPointF(18, crown), QPointF(21, crown),
        QPointF(12, 21), QPointF(3, crown),
    ])
    # Simplify to a clean faceted gem.
    outline = QPolygonF([
        QPointF(4, crown), QPointF(8, 4), QPointF(16, 4), QPointF(20, crown),
        QPointF(12, 21),
    ])
    grad = QLinearGradient(0, 4, 0, 21)
    grad.setColorAt(0.0, hi)
    grad.setColorAt(1.0, top)
    p.setPen(QPen(QColor(top).darker(130), 0.6))
    p.setBrush(QBrush(grad))
    p.drawPolygon(outline)

    # Facet lines.
    p.setPen(QPen(QColor(255, 255, 255, 150), 0.5))
    p.drawLine(QPointF(4, crown), QPointF(20, crown))
    p.drawLine(QPointF(8, 4), QPointF(10.5, crown))
    p.drawLine(QPointF(16, 4), QPointF(13.5, crown))
    p.drawLine(QPointF(10.5, crown), QPointF(12, 21))
    p.drawLine(QPointF(13.5, crown), QPointF(12, 21))
    p.drawLine(QPointF(10.5, crown), QPointF(13.5, crown))
    p.end()
    return pm


def make_app_icon(primary: str = "#4f86d6", accent: str = "#8fc0ff") -> QIcon:
    """Return the application :class:`QIcon` at several sizes."""
    icon = QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(make_app_logo(sz, primary, accent))
    return icon
