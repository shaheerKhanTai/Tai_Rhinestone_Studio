"""Offscreen smoke tests for the main window shell.

These verify construction, view switching, image import wiring and theme
changes without a display (Qt runs on the ``offscreen`` platform via conftest).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models.app_settings import AppSettings
from app.ui.theme import ThemeMode


@pytest.fixture()
def window(qapp):
    from app.ui.main_window import MainWindow

    w = MainWindow(AppSettings())
    yield w
    # Avoid the (correct) unsaved-changes modal blocking headless teardown.
    w.state.dirty = False
    w.close()
    w.deleteLater()


def make_image(tmp_path):
    p = tmp_path / "logo.png"
    rgba = np.zeros((120, 160, 4), dtype=np.uint8)
    cv2.circle(rgba, (80, 60), 40, (20, 20, 30, 255), -1)
    cv2.imwrite(str(p), rgba)
    return p


class TestMainWindow:
    def test_constructs(self, window):
        assert window.windowTitle().startswith("Rhinestone Studio")

    def test_starts_on_welcome(self, window):
        # No project → docks/toolbar hidden.
        assert not window._left_dock.isVisible()
        assert not window._toolbar.isVisible()

    def test_new_project_shows_editor(self, window):
        window.action_new_project()
        assert window.state.has_project
        assert window._stack.currentWidget() is window._canvas_host

    def test_import_image_sets_physical_size(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path))
        assert window.state.has_image
        proj = window.state.project
        # 160x120 image at 10in width → 10 x 7.5 in.
        assert proj.width_in_units == pytest.approx(10.0, abs=0.01)
        assert proj.height_in_units == pytest.approx(7.5, abs=0.01)

    def test_import_invalid_image_shows_error_not_crash(self, window, tmp_path, monkeypatch):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        errors = []
        monkeypatch.setattr(window, "show_error", lambda t, m: errors.append((t, m)))
        window._import_image_path(bad)
        assert errors  # a friendly error was shown
        assert not window.state.has_image

    def test_theme_switch(self, window):
        window.set_theme_mode(ThemeMode.LIGHT)
        assert window.theme.mode is ThemeMode.LIGHT
        assert window.app_settings.theme == "light"

    def test_action_states_without_design(self, window):
        window.action_new_project()
        # No stones yet → export disabled, generate disabled (no image).
        assert not window.act_export_svg.isEnabled()
        assert not window.act_generate.isEnabled()

    def test_generate_enabled_after_image(self, window, tmp_path):
        window._import_image_path(make_image(tmp_path))
        assert window.act_generate.isEnabled()

    def test_recent_menu_populates(self, window, tmp_path):
        p = make_image(tmp_path)
        window.app_settings.add_recent_project(str(p))
        window.refresh_recent()
        # Menu has at least the one recent entry plus separator/clear.
        assert len(window._recent_menu.actions()) >= 1

    def test_status_helpers(self, window):
        window.set_status_count(1234)
        window.set_status_zoom(150)
        window.set_status_dimensions("10 x 8 in")
        assert "1,234" in window._status_count.text()
        assert "150" in window._status_zoom.text()
