"""Offscreen tests for the design canvas: transforms, zoom, selection, paint."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from app.core.geometry import Unit
from app.models.design import Design
from app.ui.canvas import CanvasView, CanvasTool
from app.ui.theme import DARK_THEME


def grid_design(n_side=10, pitch=5.0, w=100.0, h=100.0) -> Design:
    d = Design(w, h)
    for i in range(n_side):
        for j in range(n_side):
            d.add_stone(5.0 + i * pitch, 5.0 + j * pitch, 2.8)
    return d


@pytest.fixture()
def canvas(qapp):
    c = CanvasView(DARK_THEME)
    c.resize(800, 600)
    yield c
    c.deleteLater()


class TestTransforms:
    def test_roundtrip(self, canvas):
        canvas._set_zoom(3.0)
        canvas._origin = QPointF(40, 30)
        for (x, y) in [(0, 0), (25.5, 10.0), (100, 80)]:
            s = canvas.mm_to_screen(x, y)
            back = canvas.screen_to_mm(s.x(), s.y())
            assert back.x() == pytest.approx(x)
            assert back.y() == pytest.approx(y)

    def test_zoom_clamped(self, canvas):
        canvas._set_zoom(1e6)
        assert canvas._zoom <= 80.0
        canvas._set_zoom(1e-6)
        assert canvas._zoom >= 0.05


class TestView:
    def test_fit_centers_design(self, canvas):
        canvas.set_design(grid_design(), fit=True)
        # After fit the whole design should be visible (positive zoom).
        assert canvas._zoom > 0

    def test_zoom_in_out(self, canvas):
        canvas.set_design(grid_design())
        z0 = canvas._zoom
        canvas.zoom_in()
        assert canvas._zoom > z0
        canvas.zoom_out()
        assert canvas._zoom < canvas._zoom * 1.26

    def test_zoom_about_keeps_point(self, canvas):
        canvas.set_design(grid_design())
        pt = QPointF(400, 300)
        before = canvas.screen_to_mm(pt.x(), pt.y())
        canvas._zoom_about(pt, 2.0)
        after = canvas.screen_to_mm(pt.x(), pt.y())
        assert after.x() == pytest.approx(before.x(), abs=1e-6)
        assert after.y() == pytest.approx(before.y(), abs=1e-6)


class TestGeometryCache:
    def test_refresh_builds_arrays(self, canvas):
        d = grid_design(4)
        canvas.set_design(d)
        assert canvas._pos.shape == (16, 2)
        assert canvas._dia.shape == (16,)
        assert len(canvas._palette) == 1  # single colour

    def test_multiple_colors(self, canvas):
        d = Design(50, 50)
        d.add_stone(10, 10, 3.0, color="#FF0000")
        d.add_stone(20, 20, 3.0, color="#00FF00")
        canvas.set_design(d)
        assert len(canvas._palette) == 2

    def test_empty_design(self, canvas):
        canvas.set_design(Design(50, 50))
        assert canvas._pos.shape == (0, 2)


class TestSelection:
    def test_select_all_clear(self, canvas):
        d = grid_design(5)
        canvas.set_design(d)
        canvas.select_all()
        assert canvas.selected_count() == 25
        assert len(d.selected_stones()) == 25
        canvas.clear_selection()
        assert canvas.selected_count() == 0

    def test_rect_selection(self, canvas):
        d = grid_design(10, pitch=5.0)  # stones at 5,10,...,50
        canvas.set_design(d, fit=False)
        canvas._set_zoom(4.0)
        canvas._origin = QPointF(0, 0)
        # Rect covering mm 0..27 in both axes → stones at 5..25 → 5x5 = 25.
        a = canvas.mm_to_screen(0, 0)
        b = canvas.mm_to_screen(27, 27)
        canvas._select_in_rect(a, b, additive=False)
        assert canvas.selected_count() == 25

    def test_hit_test(self, canvas):
        d = Design(50, 50)
        d.add_stone(25, 25, 4.0)
        canvas.set_design(d, fit=False)
        assert canvas._hit_test(QPointF(25, 25)) == 0
        assert canvas._hit_test(QPointF(45, 45)) is None

    def test_point_selection_additive(self, canvas):
        d = grid_design(3, pitch=10.0)
        canvas.set_design(d, fit=False)
        canvas._set_zoom(4.0)
        canvas._origin = QPointF(0, 0)
        p1 = canvas.mm_to_screen(5, 5)
        canvas._select_at_point(p1, additive=False)
        assert canvas.selected_count() == 1
        p2 = canvas.mm_to_screen(15, 5)
        canvas._select_at_point(p2, additive=True)
        assert canvas.selected_count() == 2


class TestPaint:
    @pytest.mark.parametrize("zoom", [0.2, 1.0, 3.0, 10.0])
    def test_paint_at_various_zooms(self, canvas, zoom):
        canvas.set_design(grid_design(20, pitch=4.0), fit=False)
        canvas._set_zoom(zoom)
        pm = canvas.grab()  # exercises fast + stamped paths without error
        assert pm.width() == 800

    def test_paint_with_background(self, canvas):
        rgba = np.zeros((80, 80, 4), dtype=np.uint8)
        rgba[:, :] = (100, 120, 140, 255)
        canvas.set_background_image(rgba)
        canvas.set_design(grid_design(5))
        assert canvas.grab().width() == 800

    def test_paint_empty(self, canvas):
        canvas.set_design(Design(100, 80))
        assert canvas.grab().width() == 800

    def test_collision_highlight_paints(self, canvas):
        d = grid_design(4)
        canvas.set_design(d)
        ids = {d.stones[0].id, d.stones[1].id}
        canvas.set_collisions(ids)
        assert int(canvas._collision.sum()) == 2
        assert canvas.grab().width() == 800


class TestConfig:
    def test_tool_and_unit(self, canvas):
        canvas.set_tool(CanvasTool.PAN)
        canvas.set_unit(Unit.MM)
        assert canvas._unit is Unit.MM

    def test_toggle_options(self, canvas):
        canvas.set_show_grid(True)
        canvas.set_show_rulers(False)
        canvas.set_show_image(False)
        canvas.set_background_opacity(80)
        assert canvas._show_grid and not canvas._show_rulers
        assert canvas._bg_opacity == pytest.approx(0.8)
