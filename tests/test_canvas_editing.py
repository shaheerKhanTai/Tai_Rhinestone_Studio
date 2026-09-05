"""Tests for canvas manual-editing operations and history integration."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF

from app.core.history import CommandStack
from app.models.design import Design
from app.ui.canvas import CanvasView, CanvasTool
from app.ui.theme import DARK_THEME


@pytest.fixture()
def edit_canvas(qapp):
    c = CanvasView(DARK_THEME)
    c.resize(800, 600)
    d = Design(100, 100)
    c.set_design(d, fit=False)
    c._set_zoom(6.0)
    c._origin = QPointF(20, 20)
    c.set_new_stone_spec(2.8, "SS10", "#111111", 0.5)
    stack = CommandStack()
    c.command_produced.connect(stack.push_applied)
    return c, d, stack


class TestAdd:
    def test_add_stone(self, edit_canvas):
        c, d, stack = edit_canvas
        assert c.add_stone_at(QPointF(20, 20))
        assert d.stone_count == 1
        assert stack.can_undo

    def test_add_rejects_collision(self, edit_canvas):
        c, d, stack = edit_canvas
        c.add_stone_at(QPointF(20, 20))
        assert not c.add_stone_at(QPointF(20.3, 20))  # overlaps
        assert d.stone_count == 1

    def test_add_rejects_outside_canvas(self, edit_canvas):
        c, d, stack = edit_canvas
        assert not c.add_stone_at(QPointF(200, 200))
        assert d.stone_count == 0

    def test_undo_add(self, edit_canvas):
        c, d, stack = edit_canvas
        c.add_stone_at(QPointF(20, 20))
        stack.undo()
        c.refresh_geometry()
        assert d.stone_count == 0


class TestDelete:
    def test_delete_selected(self, edit_canvas):
        c, d, stack = edit_canvas
        for i in range(4):
            c.add_stone_at(QPointF(10 + i * 10, 10 + i * 10))
        c.select_all()
        n = c.delete_selected()
        assert n == 4
        assert d.stone_count == 0

    def test_delete_undo_restores(self, edit_canvas):
        c, d, stack = edit_canvas
        c.add_stone_at(QPointF(20, 20))
        c.add_stone_at(QPointF(40, 40))
        c.select_all()
        c.delete_selected()
        assert d.stone_count == 0
        # Undo delete, then undo the two adds.
        stack.undo()
        c.refresh_geometry()
        assert d.stone_count == 2

    def test_delete_nothing_selected(self, edit_canvas):
        c, d, stack = edit_canvas
        c.add_stone_at(QPointF(20, 20))
        assert c.delete_selected() == 0


class TestMove:
    def test_move_drag_updates_and_records(self, edit_canvas):
        c, d, stack = edit_canvas
        c.add_stone_at(QPointF(30, 30))
        stone = d.stones[0]
        stone.selected = True
        c.refresh_selection()
        c._begin_move(QPointF(30, 30))
        c._update_move(QPointF(50, 50))
        c._end_move()
        assert stone.x == pytest.approx(50, abs=0.5)
        assert stone.y == pytest.approx(50, abs=0.5)
        # Move recorded → undo restores original.
        stack.undo()
        assert stone.x == pytest.approx(30, abs=0.5)

    def test_move_clamped_to_canvas(self, edit_canvas):
        c, d, stack = edit_canvas
        c.add_stone_at(QPointF(50, 50))
        stone = d.stones[0]
        stone.selected = True
        c.refresh_selection()
        c._begin_move(QPointF(50, 50))
        c._update_move(QPointF(500, 500))  # far outside
        c._end_move()
        assert 0 <= stone.x <= 100
        assert 0 <= stone.y <= 100


class TestBrushEraser:
    def test_brush_adds_multiple(self, edit_canvas):
        c, d, stack = edit_canvas
        c.set_tool(CanvasTool.BRUSH)
        c._begin_brush(CanvasTool.BRUSH)
        c._apply_brush(QPointF(50, 50), CanvasTool.BRUSH)
        c._end_brush()
        assert d.stone_count > 1  # a cluster
        assert stack.can_undo

    def test_brush_respects_spacing(self, edit_canvas):
        from app.core.collision import find_collisions

        c, d, stack = edit_canvas
        c.set_tool(CanvasTool.BRUSH)
        c._begin_brush(CanvasTool.BRUSH)
        c._apply_brush(QPointF(50, 50), CanvasTool.BRUSH)
        c._end_brush()
        pairs = find_collisions(d.positions(), d.diameters(), 0.5)
        assert pairs == []  # no overlaps created

    def test_brush_undo(self, edit_canvas):
        c, d, stack = edit_canvas
        c.set_tool(CanvasTool.BRUSH)
        c._begin_brush(CanvasTool.BRUSH)
        c._apply_brush(QPointF(50, 50), CanvasTool.BRUSH)
        c._end_brush()
        count = d.stone_count
        assert count > 0
        stack.undo()
        c.refresh_geometry()
        assert d.stone_count == 0

    def test_eraser_removes(self, edit_canvas):
        c, d, stack = edit_canvas
        for i in range(5):
            c.add_stone_at(QPointF(40 + i * 6, 40))
        before = d.stone_count
        c.set_tool(CanvasTool.ERASER)
        c._begin_brush(CanvasTool.ERASER)
        c._apply_brush(QPointF(40, 40), CanvasTool.ERASER)
        c._end_brush()
        assert d.stone_count < before

    def test_eraser_undo_restores(self, edit_canvas):
        c, d, stack = edit_canvas
        for i in range(5):
            c.add_stone_at(QPointF(40 + i * 6, 40))
        before = d.stone_count
        c.set_tool(CanvasTool.ERASER)
        c._begin_brush(CanvasTool.ERASER)
        c._apply_brush(QPointF(40, 40), CanvasTool.ERASER)
        c._end_brush()
        # Undo the erase (last command).
        stack.undo()
        c.refresh_geometry()
        assert d.stone_count == before
