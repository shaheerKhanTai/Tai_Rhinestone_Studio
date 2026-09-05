"""Tests for the interactive crop tool (Section 8, canvas + wiring).

Crop coordinates are emitted as normalized artboard fractions by the canvas and
composed into oriented-image pixels by the window, so a subsequent
``apply_image_transforms`` yields the expected sub-image.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.image_processor import apply_image_transforms
from app.models.app_settings import AppSettings
from app.models.design import Design


@pytest.fixture()
def window(qapp):
    from app.ui.main_window import MainWindow

    w = MainWindow(AppSettings())
    yield w
    w.state.dirty = False
    w.close()
    w.deleteLater()


def make_image(tmp_path, w=200, h=100):
    p = tmp_path / "logo.png"
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    cv2.rectangle(rgba, (0, 0), (w // 2, h), (255, 0, 0, 255), -1)
    cv2.imwrite(str(p), rgba)
    return p


class TestCanvasCrop:
    def test_begin_and_cancel(self, qapp):
        from app.ui.canvas import CanvasTool, CanvasView
        from app.ui.theme import get_theme, ThemeMode

        c = CanvasView(get_theme(ThemeMode.DARK))
        c.set_design(Design(100.0, 80.0))
        c.set_tool(CanvasTool.CROP)
        c.begin_crop()
        assert c.crop_active
        seen = []
        c.crop_cancelled.connect(lambda: seen.append(True))
        c.cancel_crop()
        assert not c.crop_active
        assert seen == [True]
        c.deleteLater()

    def test_switching_tool_cancels_crop(self, qapp):
        from app.ui.canvas import CanvasTool, CanvasView
        from app.ui.theme import get_theme, ThemeMode

        c = CanvasView(get_theme(ThemeMode.DARK))
        c.set_design(Design(100.0, 80.0))
        c.begin_crop()
        assert c.crop_active
        c.set_tool(CanvasTool.SELECT)  # leaving crop tool cancels
        assert not c.crop_active
        c.deleteLater()

    def test_commit_emits_normalized_rect(self, qapp):
        from app.ui.canvas import CanvasView
        from app.ui.theme import get_theme, ThemeMode
        from PySide6.QtCore import QRectF

        c = CanvasView(get_theme(ThemeMode.DARK))
        c.set_design(Design(100.0, 80.0))
        c.begin_crop()
        # Simulate a drag selecting the left half, top half.
        c._crop_rect_mm = QRectF(0.0, 0.0, 50.0, 40.0)
        got = []
        c.crop_committed.connect(lambda *a: got.append(a))
        assert c.commit_crop() is True
        assert not c.crop_active
        fx, fy, fw, fh = got[0]
        assert (fx, fy) == pytest.approx((0.0, 0.0))
        assert (fw, fh) == pytest.approx((0.5, 0.5))
        c.deleteLater()

    def test_commit_without_rect_returns_false(self, qapp):
        from app.ui.canvas import CanvasView
        from app.ui.theme import get_theme, ThemeMode

        c = CanvasView(get_theme(ThemeMode.DARK))
        c.set_design(Design(100.0, 80.0))
        # Not in crop mode → nothing to commit.
        assert c.commit_crop() is False
        c.deleteLater()


class TestCropWiring:
    def test_crop_tool_disabled_without_image(self, window):
        window.action_new_project()
        assert not window.act_tool_crop.isEnabled()

    def test_crop_tool_enabled_with_image(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path))
        assert window.act_tool_crop.isEnabled()

    def test_commit_sets_crop_in_oriented_pixels(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path, 200, 100))
        proj = window.state.project
        # Left 40%, top 60% of the image.
        window._on_crop_committed(0.0, 0.0, 0.4, 0.6)
        assert proj.image.crop == (0, 0, 80, 60)
        # Applying that crop yields an 80×60 sub-image.
        out = apply_image_transforms(
            window.state.source_image.rgba,
            rotation=proj.image.rotation, flip_h=proj.image.flip_h,
            flip_v=proj.image.flip_v, crop=proj.image.crop,
        )
        assert out.shape[1] == 80 and out.shape[0] == 60

    def test_second_crop_composes_with_first(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path, 200, 100))
        proj = window.state.project
        window._on_crop_committed(0.0, 0.0, 0.5, 1.0)   # → (0,0,100,100)
        assert proj.image.crop == (0, 0, 100, 100)
        # Crop the right half of the *already-cropped* 100×100 image.
        window._on_crop_committed(0.5, 0.0, 0.5, 1.0)   # → origin x = 0 + 50
        assert proj.image.crop == (50, 0, 50, 100)

    def test_crop_marks_dirty_and_returns_to_select(self, window, tmp_path):
        from app.ui.canvas import CanvasTool

        window._import_image_path(make_image(tmp_path))
        window.state.dirty = False
        window._on_crop_committed(0.1, 0.1, 0.5, 0.5)
        assert window.state.dirty is True
        assert window.act_tool_select.isChecked()
        assert window._canvas._tool is CanvasTool.SELECT


class TestCropUndoRedo:
    """ISSUE 2 — crop must be a normal undoable command (STATE A/B round-trips)."""

    def _add_stones(self, window, n=3):
        proj = window.state.project
        for i in range(n):
            proj.design.add_stone(20.0 + i * 10.0, 30.0, 3.0)

    def test_undo_restores_pre_crop_state(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path, 200, 100))
        proj = window.state.project
        self._add_stones(window)
        original_size = (proj.width_mm, proj.height_mm)
        original_positions = [(s.x, s.y) for s in proj.design.stones]

        # STATE A -> STATE B (crop).
        window._on_crop_committed(0.0, 0.0, 0.5, 1.0)
        assert proj.image.crop == (0, 0, 100, 100)
        assert proj.design.stone_count == 0

        # Ctrl+Z -> STATE A.
        window.action_undo()
        assert proj.image.crop is None
        assert proj.design.stone_count == 3
        assert [(s.x, s.y) for s in proj.design.stones] == original_positions
        assert (proj.width_mm, proj.height_mm) == original_size

        # Ctrl+Y -> STATE B.
        window.action_redo()
        assert proj.image.crop == (0, 0, 100, 100)
        assert proj.design.stone_count == 0

    def test_repeated_crop_undo_redo_sequence(self, window, tmp_path):
        # Original -> Crop 1 -> Crop 2 -> Undo (Crop 1) -> Undo (Original)
        # -> Redo (Crop 1) -> Redo (Crop 2).
        window._import_image_path(make_image(tmp_path, 200, 100))
        proj = window.state.project

        window._on_crop_committed(0.0, 0.0, 0.5, 1.0)   # Crop 1: (0,0,100,100)
        crop1 = proj.image.crop
        window._on_crop_committed(0.5, 0.0, 0.5, 1.0)   # Crop 2: (50,0,50,100)
        crop2 = proj.image.crop
        assert crop1 == (0, 0, 100, 100)
        assert crop2 == (50, 0, 50, 100)

        assert window._command_stack.undo()      # -> Crop 1
        assert proj.image.crop == crop1
        assert window._command_stack.undo()      # -> Original
        assert proj.image.crop is None
        assert not window._command_stack.can_undo
        assert window._command_stack.redo()      # -> Crop 1
        assert proj.image.crop == crop1
        assert window._command_stack.redo()      # -> Crop 2
        assert proj.image.crop == crop2
        assert not window._command_stack.can_redo

    def test_undo_restores_canvas_background(self, window, tmp_path):
        # The displayed (background) image must follow the restored crop.
        window._import_image_path(make_image(tmp_path, 200, 100))
        proj = window.state.project
        window._on_crop_committed(0.0, 0.0, 0.5, 1.0)
        cropped = window._current_oriented_rgba()
        assert cropped.shape[1] == 100
        window._command_stack.undo()
        restored = window._current_oriented_rgba()
        assert restored.shape[1] == 200  # full image back
        window._command_stack.redo()
        assert window._current_oriented_rgba().shape[1] == 100

    def test_crop_then_generate_then_double_undo(self, window, tmp_path):
        # Crop, generate, then undo twice: first the generation, then the crop.
        from app.core.history import ReplaceDesignCommand
        from app.models.stone import Stone

        window._import_image_path(make_image(tmp_path, 200, 100))
        proj = window.state.project
        window._on_crop_committed(0.0, 0.0, 0.5, 1.0)

        # Simulate a completed generation the way _on_generation_finished does.
        generated = [Stone(i + 1, 5.0 * i, 5.0, 2.8) for i in range(10)]
        old = list(proj.design.stones)
        proj.design.replace_all(generated)
        window._command_stack.push_applied(
            ReplaceDesignCommand(proj.design, old, list(generated))
        )
        assert proj.design.stone_count == 10

        assert window._command_stack.undo()      # undo generation
        assert proj.design.stone_count == 0
        assert proj.image.crop == (0, 0, 100, 100)
        assert window._command_stack.undo()      # undo crop
        assert proj.image.crop is None

    def test_crop_undo_then_stone_editing_still_works(self, window, tmp_path):
        # After a crop-undo round trip, normal stone edits must remain undoable.
        window._import_image_path(make_image(tmp_path, 200, 100))
        proj = window.state.project
        self._add_stones(window, n=1)
        window._on_crop_committed(0.0, 0.0, 0.5, 1.0)
        window._command_stack.undo()

        stone = proj.design.stones[0]
        moves = {stone.id: ((stone.x, stone.y), (stone.x + 5.0, stone.y + 5.0))}
        stone.x, stone.y = stone.x + 5.0, stone.y + 5.0
        from app.core.history import MoveStonesCommand

        window._command_stack.push_applied(MoveStonesCommand(proj.design, moves))
        assert window._command_stack.undo()
        assert (stone.x, stone.y) == (20.0, 30.0)
