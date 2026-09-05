"""Tests for the image-transform UI wiring: rotate / flip / reset (Section 8).

The transform *engine* is already covered in ``test_image_processor.py``; these
tests verify the main-window handlers mutate ``ImageState`` correctly, re-fit the
physical size, clear an invalidated crop, and that transforms persist.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models.app_settings import AppSettings
from app.models.project import Project


@pytest.fixture()
def window(qapp):
    from app.ui.main_window import MainWindow

    w = MainWindow(AppSettings())
    yield w
    w.state.dirty = False
    w.close()
    w.deleteLater()


def make_image(tmp_path, w=160, h=120):
    p = tmp_path / "logo.png"
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.circle(rgba, (w // 2, h // 2), min(w, h) // 3, (20, 20, 30, 255), -1)
    cv2.imwrite(str(p), rgba)
    return p


class TestTransformActions:
    def test_disabled_without_image(self, window):
        window.action_new_project()
        assert not window.act_rotate_right.isEnabled()
        assert not window.act_flip_h.isEnabled()
        assert not window.act_reset_transforms.isEnabled()

    def test_enabled_with_image(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path))
        assert window.act_rotate_right.isEnabled()
        assert window.act_flip_v.isEnabled()

    def test_rotate_right_sets_rotation_and_refits(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path, 160, 120))
        proj = window.state.project
        # 160x120 at 10in width → 10 x 7.5 in.
        assert proj.height_in_units == pytest.approx(7.5, abs=0.01)
        window._transform_image("rotate_right")
        assert proj.image.rotation == 90
        # Oriented dims are now 120x160 → width fixed at 10in, height 13.33in.
        assert proj.width_in_units == pytest.approx(10.0, abs=0.01)
        assert proj.height_in_units == pytest.approx(10.0 * 160 / 120, abs=0.02)

    def test_rotate_left_wraps(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path))
        window._transform_image("rotate_left")
        assert window.state.project.image.rotation == 270

    def test_flips_toggle(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path))
        img = window.state.project.image
        window._transform_image("flip_h")
        assert img.flip_h is True
        window._transform_image("flip_h")
        assert img.flip_h is False
        window._transform_image("flip_v")
        assert img.flip_v is True

    def test_reset_clears_all(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path))
        img = window.state.project.image
        window._transform_image("rotate_right")
        window._transform_image("flip_h")
        window._transform_image("reset")
        assert img.rotation == 0
        assert img.flip_h is False
        assert img.flip_v is False
        assert img.crop is None

    def test_rotate_clears_existing_crop(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path))
        img = window.state.project.image
        img.crop = (10, 10, 50, 50)
        window._transform_image("rotate_right")
        assert img.crop is None

    def test_transform_marks_dirty(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path))
        window.state.dirty = False
        window._transform_image("flip_h")
        assert window.state.dirty is True


class TestTransformPersistence:
    def test_project_round_trip_preserves_transforms(self):
        proj = Project.new("t")
        proj.image.source_path = None
        proj.image.original_width_px = 160
        proj.image.original_height_px = 120
        proj.image.rotation = 90
        proj.image.flip_h = True
        proj.image.crop = (5, 6, 40, 30)
        restored = Project.from_dict(proj.to_dict())
        assert restored.image.rotation == 90
        assert restored.image.flip_h is True
        assert restored.image.flip_v is False
        assert restored.image.crop == (5, 6, 40, 30)
