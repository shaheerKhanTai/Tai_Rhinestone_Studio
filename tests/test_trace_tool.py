"""Tests for the Trace canvas tool: painting, undo/redo, clear, persistence of
project state. Drives the canvas's private stroke API the same way
test_canvas_editing drives add/move (headless, no synthetic mouse events)."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from app.core.history import CommandStack
from app.models.project import Project
from app.ui.canvas import CanvasTool, CanvasView
from app.ui.theme import DARK_THEME


@pytest.fixture()
def trace_canvas(qapp):
    project = Project.new("Trace")
    project.set_physical_size(100.0, 100.0)
    c = CanvasView(DARK_THEME)
    c.set_project(project)
    c.set_design(project.design, fit=False)
    c.set_new_stone_spec(2.8, "SS10", "#111111", 0.5)
    stack = CommandStack()
    c.command_produced.connect(stack.push_applied)
    return c, project, stack


def paint(c, x0, y0, x1, y1, erase=False):
    c._begin_trace(QPointF(x0, y0), erase=erase)
    c._apply_trace(QPointF(x1, y1))
    c._end_trace()


class TestTracePainting:
    def test_stroke_creates_mask_on_project(self, trace_canvas):
        c, p, _ = trace_canvas
        assert p.trace_mask is None
        paint(c, 20, 20, 80, 20)
        assert p.trace_mask is not None
        assert c.has_trace()

    def test_mask_is_binary_and_inside_canvas(self, trace_canvas):
        c, p, _ = trace_canvas
        paint(c, 0, 50, 100, 50)
        m = p.trace_mask
        assert set(np.unique(m)).issubset({0, 255})
        assert m.any()

    def test_first_stroke_undo_removes_mask(self, trace_canvas):
        c, p, stack = trace_canvas
        paint(c, 20, 20, 80, 20)
        assert stack.undo()
        assert p.trace_mask is None
        assert not c.has_trace()
        assert stack.redo()
        assert c.has_trace()

    def test_later_stroke_undo_restores_previous_mask(self, trace_canvas):
        c, p, stack = trace_canvas
        paint(c, 20, 20, 80, 20)
        after_first = p.trace_mask.copy()
        paint(c, 20, 60, 80, 60)
        assert int(p.trace_mask.sum()) > int(after_first.sum())
        assert stack.undo()
        np.testing.assert_array_equal(p.trace_mask, after_first)
        assert stack.redo()
        assert int(p.trace_mask.sum()) > int(after_first.sum())

    def test_erase_stroke(self, trace_canvas):
        c, p, stack = trace_canvas
        paint(c, 50, 50, 50, 50)  # dot
        assert c.has_trace()
        paint(c, 50, 50, 50, 50, erase=True)
        assert not c.has_trace()
        assert stack.undo()  # undo the erase
        assert c.has_trace()

    def test_clear_trace_and_undo(self, trace_canvas):
        c, p, stack = trace_canvas
        paint(c, 20, 20, 80, 20)
        c.clear_trace()
        assert p.trace_mask is None
        assert not c.has_trace()
        assert stack.undo()
        assert c.has_trace()

    def test_noop_stroke_emits_nothing(self, trace_canvas):
        c, p, stack = trace_canvas
        # An erase stroke on a non-existent mask paints nothing: no command,
        # and no empty mask left behind on the project.
        paint(c, 50, 50, 60, 50, erase=True)
        assert p.trace_mask is None
        assert not stack.can_undo

    def test_trace_signals_fire(self, trace_canvas):
        c, p, _ = trace_canvas
        events = []
        c.trace_changed.connect(lambda: events.append("changed"))
        paint(c, 20, 20, 40, 20)
        assert events.count("changed") == 1

    def test_overlay_render_smoke(self, trace_canvas):
        c, _, _ = trace_canvas
        paint(c, 20, 20, 80, 80)
        c.set_tool(CanvasTool.TRACE)
        c._trace_hover_mm = QPointF(50, 50)
        c.resize(400, 300)
        pix = c.grab()
        assert not pix.isNull()


class TestTraceProjectIntegration:
    def test_set_project_resyncs_mask(self, trace_canvas):
        c, p, _ = trace_canvas
        paint(c, 20, 20, 80, 20)
        c.set_trace_mask(None)  # simulate a stale reference
        c.resync_trace_to_project()
        assert c.trace_mask() is p.trace_mask

    def test_resample_on_size_change(self, trace_canvas):
        from app.core.trace_mask import mask_shape_for

        c, p, _ = trace_canvas
        paint(c, 50, 20, 50, 20)
        p.set_physical_size(100.0, 50.0)
        c.resample_trace_to(p.width_mm, p.height_mm)
        assert p.trace_mask.shape == mask_shape_for(100.0, 50.0)
        assert c.has_trace()
