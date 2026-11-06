"""FEATURE 1 — a successful generation consumes the traced region.

After Generate completes, the traced overlay and the trace data must both be
gone (so a later Generate cannot silently re-use a stale gate), while the
stones just placed and the source image are untouched. The clear is grouped
with the stone replacement, so one Ctrl+Z restores both the previous design
and the traced area.

A generation that places *nothing* keeps the trace — the user will want to
adjust density and Generate again over the same painting. Cancelled and failed
runs keep it too.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QPointF, QTimer

from app.models.app_settings import AppSettings


@pytest.fixture()
def window(qapp):
    from app.ui.main_window import MainWindow

    w = MainWindow(AppSettings())
    yield w
    w.state.dirty = False
    w.close()
    w.deleteLater()


def import_image(window, tmp_path, *, dark_fraction: float = 0.8):
    """Import a wide dark subject (so a trace over part of it has something to
    fill) on a white background."""
    p = tmp_path / "logo.png"
    rgba = np.full((300, 300, 4), 255, np.uint8)
    x0 = int(300 * (1.0 - dark_fraction) / 2)
    rgba[:, x0:300 - x0, :3] = 25
    cv2.imwrite(str(p), rgba)
    window._import_image_path(p)
    return p


def import_blank(window, tmp_path):
    """Import a pure-white image: generation legitimately places no stones."""
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.full((200, 200, 4), 255, np.uint8))
    window._import_image_path(p)
    return p


def paint_band(canvas, project, x_mm: float, y_mm: float, radius_mm: float) -> None:
    """Paint a trace blob through the canvas's real stroke API."""
    canvas._begin_trace(QPointF(x_mm, y_mm), erase=False)
    canvas._apply_trace(QPointF(x_mm, y_mm))
    canvas._end_trace()
    assert project.trace_mask is not None
    assert canvas.has_trace()


def generate_sync(window, *, use_trace: bool = True, analysis_max_dim: int = 600):
    """Run generation and commit it, without the worker thread."""
    from app.core.image_processor import apply_image_transforms
    from app.core.rhinestone_engine import generate

    proj = window.state.project
    settings = window._design_panel.build_settings()
    rgba = apply_image_transforms(window.state.source_image.rgba)
    result = generate(
        rgba, settings, proj.width_mm, proj.height_mm,
        analysis_max_dim=analysis_max_dim,
        trace_mask=proj.trace_mask if use_trace else None,
    )
    window._on_generation_finished(result)
    return result


def pump_until(cond, timeout_ms: int = 60000) -> bool:
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(20)
    deadline = QTimer()
    deadline.setSingleShot(True)
    timer.timeout.connect(lambda: loop.quit() if cond() else None)
    deadline.timeout.connect(loop.quit)
    timer.start()
    deadline.start(timeout_ms)
    loop.exec()
    timer.stop()
    return cond()


class TestTraceConsumed:
    def test_successful_generation_clears_the_trace(self, window, tmp_path):
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)

        result = generate_sync(window)

        assert result.stone_count > 0
        assert proj.trace_mask is None
        assert canvas.trace_mask() is None
        assert not canvas.has_trace()
        assert canvas._trace_pixmap is None       # no cached overlay left over

    def test_stones_and_image_survive_the_clear(self, window, tmp_path):
        path = import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)

        result = generate_sync(window)
        placed = proj.design.positions().copy()

        assert proj.design.stone_count == result.stone_count > 0
        np.testing.assert_array_equal(proj.design.positions(), placed)
        # The source image reference is untouched too.
        assert proj.image.source_path == path
        assert window.state.source_image is not None

    def test_clear_trace_action_is_disabled_afterwards(self, window, tmp_path):
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)
        assert window.act_clear_trace.isEnabled()

        generate_sync(window)
        assert not window.act_clear_trace.isEnabled()

    def test_a_new_trace_can_be_painted_afterwards(self, window, tmp_path):
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)
        generate_sync(window)

        paint_band(canvas, proj, 60.0, proj.height_mm / 2, 8.0)
        assert canvas.has_trace()
        assert proj.trace_mask is not None


class TestNoStaleTrace:
    def test_second_generation_uses_the_whole_image(self, window, tmp_path):
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        # A narrow band on the left of a wide dark subject.
        paint_band(canvas, proj, 25.0, proj.height_mm / 2, 10.0)

        first = generate_sync(window)
        first_x = first.placement.positions_mm[:, 0]
        assert float(first_x.max()) < 45.0          # confined to the band

        # Generate again without painting: the whole subject must be used.
        second = generate_sync(window)
        second_x = second.placement.positions_mm[:, 0]
        assert float(second_x.max()) > 100.0
        assert second.stone_count > first.stone_count

    def test_undo_restores_stones_and_trace_in_one_step(self, window, tmp_path):
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas

        # A pre-existing design so the undo has something to restore.
        proj.design.add_stone(5.0, 5.0, 2.8)
        before_ids = [s.id for s in proj.design.stones]
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)
        traced = proj.trace_mask.copy()

        generate_sync(window)
        assert proj.design.stone_count > 1
        assert proj.trace_mask is None

        window.action_undo()
        assert [s.id for s in proj.design.stones] == before_ids   # stones back
        assert canvas.has_trace() and proj.trace_mask is not None # trace back
        np.testing.assert_array_equal(proj.trace_mask, traced)

        window.action_redo()
        assert proj.design.stone_count > 1
        assert proj.trace_mask is None
        assert not canvas.has_trace()

    def test_generation_without_a_trace_is_a_single_undo_step(self, window, tmp_path):
        import_image(window, tmp_path)
        proj = window.state.project
        proj.design.add_stone(5.0, 5.0, 2.8)

        generate_sync(window, use_trace=False)
        assert proj.design.stone_count > 1

        window.action_undo()
        assert proj.design.stone_count == 1     # one step back, no trace entry


class TestTraceKeptWhenNothingIsPlaced:
    def test_zero_stone_result_keeps_the_trace(self, window, tmp_path):
        import_blank(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)
        traced = proj.trace_mask.copy()

        result = generate_sync(window)

        assert result.stone_count == 0
        assert proj.trace_mask is not None
        np.testing.assert_array_equal(proj.trace_mask, traced)
        assert canvas.has_trace()
        assert "0 stones" in window.statusBar().currentMessage()

    def test_cancelled_generation_keeps_the_trace(self, window, tmp_path):
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)
        traced = proj.trace_mask.copy()

        window._on_generation_cancelled()

        assert proj.trace_mask is not None
        np.testing.assert_array_equal(proj.trace_mask, traced)
        assert canvas.has_trace()

    def test_failed_generation_keeps_the_trace(self, window, tmp_path, monkeypatch):
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)
        traced = proj.trace_mask.copy()

        # A failure is reported through the error dialog (which also carries the
        # cause and the suggested fixes); this test is about the trace only.
        monkeypatch.setattr(window, "show_error", lambda *a, **k: None)
        window._on_generation_failed("boom")

        assert proj.trace_mask is not None
        np.testing.assert_array_equal(proj.trace_mask, traced)

    def test_trace_survives_while_a_generation_is_in_flight(self, window, tmp_path):
        # begin_stone_stream() must not disturb the trace the worker is reading.
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)

        canvas.begin_stone_stream()
        assert proj.trace_mask is not None and canvas.has_trace()
        canvas.end_stone_stream()
        assert proj.trace_mask is not None and canvas.has_trace()


class TestClearTraceCommand:
    def test_returns_none_without_a_trace(self, window, tmp_path):
        import_image(window, tmp_path)
        assert window._canvas.clear_trace_command() is None
        assert window._canvas.clear_trace_command() is None

    def test_does_not_self_record(self, window, tmp_path):
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)
        window._command_stack.clear()

        command = canvas.clear_trace_command()
        assert command is not None
        assert not window._command_stack.can_undo   # caller owns it
        assert proj.trace_mask is None

        command.undo()
        assert proj.trace_mask is not None

    def test_clear_trace_action_still_records_its_own_step(self, window, tmp_path):
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 8.0)
        window._command_stack.clear()

        window.action_clear_trace()
        assert proj.trace_mask is None
        assert window._command_stack.can_undo
        window.action_undo()
        assert proj.trace_mask is not None


class TestEndToEndThroughTheWorker:
    def test_real_generate_consumes_the_trace(self, window, tmp_path):
        """The genuine path: Generate button → worker thread → commit."""
        import_image(window, tmp_path)
        proj, canvas = window.state.project, window._canvas
        paint_band(canvas, proj, 30.0, proj.height_mm / 2, 10.0)

        window.action_generate()
        assert window._active_worker is not None
        assert proj.trace_mask is not None        # not cleared prematurely
        assert pump_until(lambda: window._active_worker is None)

        assert proj.design.stone_count > 0
        assert proj.trace_mask is None
        assert not canvas.has_trace()

    def test_real_generate_without_trace_still_works(self, window, tmp_path):
        import_image(window, tmp_path)
        proj = window.state.project
        window.action_generate()
        assert pump_until(lambda: window._active_worker is None)
        assert proj.design.stone_count > 0
        assert proj.trace_mask is None
