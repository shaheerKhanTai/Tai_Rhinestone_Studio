"""Photoshop-style canvas scrolling: scrollbar rails, trackpad panning, clamping.

The canvas owns the scroll model (``origin``, zoom) and the rails are chrome
around it, so these tests drive the model directly through the bars and through
synthetic wheel events — no window, no real gestures.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QScrollBar

from app.models.design import Design
from app.ui.canvas import CanvasScrollHost, CanvasView
from app.ui.theme import DARK_THEME


def design_100mm() -> Design:
    d = Design(100.0, 100.0)
    for i in range(10):
        for j in range(10):
            d.add_stone(5.0 + i * 5.0, 5.0 + j * 5.0, 2.8)
    return d


def wheel_event(
    angle_y: int = 0,
    angle_x: int = 0,
    modifiers=Qt.NoModifier,
    pixel: tuple[int, int] | None = None,
) -> QWheelEvent:
    pos = QPointF(400.0, 300.0)
    pixel_delta = QPoint(*pixel) if pixel else QPoint(0, 0)
    return QWheelEvent(
        pos, pos, pixel_delta, QPoint(angle_x, angle_y),
        Qt.NoButton, modifiers, Qt.ScrollUpdate, False,
    )


def mouse_event(kind, x: float, y: float, button=Qt.NoButton, buttons=Qt.NoButton) -> QMouseEvent:
    pos = QPointF(x, y)
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


@pytest.fixture()
def canvas(qapp):
    """A laid-out canvas with a 100x100mm design attached to real rails."""
    c = CanvasView(DARK_THEME)
    c.resize(800, 600)
    c.attach_scrollbars(QScrollBar(Qt.Horizontal), QScrollBar(Qt.Vertical))
    c.set_design(design_100mm())
    yield c
    c.deleteLater()


def zoomed(canvas: CanvasView, times: int = 3) -> CanvasView:
    """Zoom in until the 100mm artboard overflows the 800x600 viewport."""
    for _ in range(times):
        canvas.zoom_in()
    return canvas


def artboard_rect(canvas: CanvasView):
    """The artboard's rect in screen px."""
    top_left = canvas.mm_to_screen(0.0, 0.0)
    return top_left.x(), top_left.y(), canvas._zoom * 100.0


class TestRailState:
    def test_rails_are_idle_when_the_whole_design_fits(self, canvas):
        assert canvas._hbar.maximum() == 0
        assert canvas._vbar.maximum() == 0

    def test_zooming_in_activates_both_rails(self, canvas):
        zoomed(canvas)
        assert canvas._hbar.maximum() > 0
        assert canvas._vbar.maximum() > 0
        # The rail spans one artboard's worth of scroll (standard page model).
        assert canvas._hbar.maximum() == pytest.approx(canvas._zoom * 100.0, abs=1.0)
        assert canvas._hbar.pageStep() == 800

    def test_zooming_out_again_empties_the_rails(self, canvas):
        zoomed(canvas)
        canvas.fit_to_view()
        assert canvas._hbar.maximum() == 0
        assert canvas._vbar.maximum() == 0

    def test_rails_follow_wheel_panning(self, canvas):
        zoomed(canvas)
        before = canvas._hbar.value()
        canvas.scroll_by_pixels(-100.0, 0.0)  # content moves left → scroll right
        assert canvas._hbar.value() == before + 100


class TestRailDragging:
    def test_dragging_a_rail_pans_the_view(self, canvas):
        zoomed(canvas)
        origin_before = QPointF(canvas._origin)
        canvas._hbar.setValue(canvas._hbar.maximum())
        assert canvas._origin.x() < origin_before.x()
        # The far edge of the design has come into view.
        assert canvas.screen_to_mm(0.0, 0.0).x() > 0.0

    def test_rail_movement_is_one_to_one_with_the_view(self, canvas):
        zoomed(canvas)
        canvas._hbar.setValue(200)
        first = canvas._origin.x()
        canvas._hbar.setValue(400)
        assert canvas._origin.x() == pytest.approx(first - 200.0)

    def test_vertical_rail_pans_the_view(self, canvas):
        zoomed(canvas)
        canvas._vbar.setValue(canvas._vbar.maximum())
        assert canvas.screen_to_mm(0.0, 0.0).y() > 0.0


class TestWheelPanning:
    def test_plain_wheel_pans_without_zooming(self, canvas):
        zoomed(canvas)
        zoom_before = canvas._zoom
        origin_before = QPointF(canvas._origin)
        QApplication.sendEvent(canvas, wheel_event(angle_y=-120))  # scroll down
        assert canvas._zoom == zoom_before
        # Scrolling down reveals the lower part: the content moves up.
        assert canvas._origin.y() == pytest.approx(origin_before.y() - 60.0)

    def test_touchpad_pixel_delta_pans_smoothly(self, canvas):
        zoomed(canvas)
        origin_before = QPointF(canvas._origin)
        QApplication.sendEvent(canvas, wheel_event(pixel=(7, 3)))
        assert canvas._origin.x() == pytest.approx(origin_before.x() + 7.0)
        assert canvas._origin.y() == pytest.approx(origin_before.y() + 3.0)

    def test_horizontal_trackpad_scroll_pans_sideways(self, canvas):
        zoomed(canvas)
        origin_before = QPointF(canvas._origin)
        QApplication.sendEvent(canvas, wheel_event(angle_x=-120))
        assert canvas._origin.x() == pytest.approx(origin_before.x() - 60.0)

    def test_shift_wheel_scrolls_sideways(self, canvas):
        zoomed(canvas)
        origin_before = QPointF(canvas._origin)
        QApplication.sendEvent(canvas, wheel_event(angle_y=-120, modifiers=Qt.ShiftModifier))
        assert canvas._origin.x() == pytest.approx(origin_before.x() - 60.0)
        assert canvas._origin.y() == pytest.approx(origin_before.y())

    @pytest.mark.parametrize("modifier", [Qt.ControlModifier, Qt.AltModifier])
    def test_modifier_wheel_zooms_about_the_cursor(self, canvas, modifier):
        zoomed(canvas)
        zoom_before = canvas._zoom
        QApplication.sendEvent(canvas, wheel_event(angle_y=120, modifiers=modifier))
        assert canvas._zoom > zoom_before

    def test_wheel_zooms_preference_restores_wheel_zoom(self, canvas):
        zoomed(canvas)
        canvas.set_wheel_zooms(True)
        zoom_before = canvas._zoom
        QApplication.sendEvent(canvas, wheel_event(angle_y=120))
        assert canvas._zoom > zoom_before


class TestClamping:
    def test_scrolling_cannot_lose_the_design(self, canvas):
        zoomed(canvas)
        canvas.scroll_by_pixels(1e6, 1e6)  # far past the design's far corner
        left, top, size = artboard_rect(canvas)
        # The viewport centre is still covered by the artboard.
        assert left <= 400 <= left + size
        assert top <= 300 <= top + size

    def test_scrolling_back_stops_at_the_near_edge(self, canvas):
        zoomed(canvas)
        canvas.scroll_by_pixels(-1e6, -1e6)
        left, top, size = artboard_rect(canvas)
        assert left <= 400 <= left + size
        assert top <= 300 <= top + size

    def test_zoom_out_pulls_the_view_back_into_reach(self, canvas):
        zoomed(canvas)
        canvas.scroll_by_pixels(1e6, 1e6)  # sit at the far clamp edge
        for _ in range(6):
            canvas.zoom_out()
        left, top, size = artboard_rect(canvas)
        # Zoomed back out the artboard is visible again rather than stranded.
        assert left < 800 and top < 600
        assert left + size > 0 and top + size > 0

    def test_middle_drag_pan_follows_the_pointer(self, canvas):
        zoomed(canvas)
        before = QPointF(canvas._origin)
        QApplication.sendEvent(canvas, mouse_event(
            QEvent.Type.MouseButtonPress, 400, 300, Qt.MiddleButton, Qt.MiddleButton))
        QApplication.sendEvent(canvas, mouse_event(
            QEvent.Type.MouseMove, 413, 306, Qt.NoButton, Qt.MiddleButton))
        QApplication.sendEvent(canvas, mouse_event(
            QEvent.Type.MouseButtonRelease, 413, 306, Qt.MiddleButton, Qt.NoButton))
        assert canvas._origin.x() == pytest.approx(before.x() + 13.0)
        assert canvas._origin.y() == pytest.approx(before.y() + 6.0)

    def test_long_drag_pan_is_clamped(self, canvas):
        zoomed(canvas)
        QApplication.sendEvent(canvas, mouse_event(
            QEvent.Type.MouseButtonPress, 400, 300, Qt.MiddleButton, Qt.MiddleButton))
        QApplication.sendEvent(canvas, mouse_event(
            QEvent.Type.MouseMove, 9000, 9000, Qt.NoButton, Qt.MiddleButton))
        QApplication.sendEvent(canvas, mouse_event(
            QEvent.Type.MouseButtonRelease, 9000, 9000, Qt.MiddleButton, Qt.NoButton))
        left, top, size = artboard_rect(canvas)
        assert left <= 400 <= left + size
        assert top <= 300 <= top + size


class TestHostLayout:
    def test_host_places_rails_around_the_canvas(self, qapp):
        canvas = CanvasView(DARK_THEME)
        host = CanvasScrollHost(canvas)
        host.resize(800, 600)
        host.layout().activate()
        try:
            assert host.canvas is canvas
            assert canvas._hbar is host.hbar and canvas._vbar is host.vbar
            # The rails take their 12px off the canvas's own rect.
            assert canvas.width() < 800
            assert canvas.height() < 600
            assert host.vbar.x() >= canvas.width()
            assert host.hbar.y() >= canvas.height()
        finally:
            host.deleteLater()

    def test_main_window_installs_the_host(self, qapp):
        from app.models.app_settings import AppSettings
        from app.ui.main_window import MainWindow

        window = MainWindow(AppSettings())
        try:
            assert window._canvas_widget is window._canvas_scroller
            assert window._canvas_scroller.canvas is window._canvas
            assert window._canvas._hbar is window._canvas_scroller.hbar
        finally:
            window.state.dirty = False
            window.close()
            window.deleteLater()

    def test_preference_reaches_the_canvas(self, qapp):
        from app.models.app_settings import AppSettings
        from app.ui.main_window import MainWindow

        settings = AppSettings()
        settings.wheel_zooms = True
        window = MainWindow(settings)
        try:
            assert window._canvas._wheel_zooms is True
            window._apply_app_settings(AppSettings(wheel_zooms=False))
            assert window._canvas._wheel_zooms is False
        finally:
            window.state.dirty = False
            window.close()
            window.deleteLater()
