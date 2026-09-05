"""Main-window tests for project save/load and export wiring (Phase 13)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models.app_settings import AppSettings


@pytest.fixture()
def window(qapp):
    from app.ui.main_window import MainWindow

    w = MainWindow(AppSettings())
    yield w
    w.state.dirty = False
    w.close()
    w.deleteLater()


def load_image_into(window, tmp_path):
    p = tmp_path / "logo.png"
    rgba = np.zeros((120, 160, 4), dtype=np.uint8)
    cv2.circle(rgba, (80, 60), 40, (20, 20, 30, 255), -1)
    cv2.imwrite(str(p), rgba)
    window._import_image_path(p)


def generate_sync(window):
    """Run generation synchronously by driving the worker directly."""
    from app.core.image_processor import apply_image_transforms
    from app.core.rhinestone_engine import generate

    proj = window.state.project
    settings = window._design_panel.build_settings()
    rgba = apply_image_transforms(window.state.source_image.rgba)
    result = generate(rgba, settings, proj.width_mm, proj.height_mm)
    window._on_generation_finished(result)


class TestSaveLoad:
    def test_save_and_reopen(self, window, tmp_path):
        load_image_into(window, tmp_path)
        generate_sync(window)
        assert window.state.project.design.stone_count > 0
        out = tmp_path / "proj.rhinestone"
        assert window._write_project(out)
        assert not window.state.dirty

        # Reopen in a second window.
        from app.ui.main_window import MainWindow

        w2 = MainWindow(AppSettings())
        w2.open_project_path(str(out))
        assert w2.state.project.design.stone_count == window.state.project.design.stone_count
        assert w2.state.has_image  # embedded image restored
        w2.state.dirty = False
        w2.close()

    def test_recent_updated_on_save(self, window, tmp_path):
        window.action_new_project()
        window.state.project.design.add_stone(10, 10, 2.8)
        out = tmp_path / "recent.rhinestone"
        window._write_project(out)
        assert str(out) in window.app_settings.recent_projects

    def test_open_missing_shows_error(self, window, tmp_path, monkeypatch):
        errors = []
        monkeypatch.setattr(window, "show_error", lambda t, m: errors.append((t, m)))
        window.open_project_path(str(tmp_path / "ghost.rhinestone"))
        assert errors


class TestExportWiring:
    def test_validate_blocks_empty(self, window, monkeypatch):
        window.action_new_project()  # no stones
        errors = []
        monkeypatch.setattr(window, "show_error", lambda t, m: errors.append((t, m)))
        assert not window._validate_before_export()
        assert errors

    def test_export_single_writes(self, window, tmp_path, monkeypatch):
        load_image_into(window, tmp_path)
        generate_sync(window)
        out = tmp_path / "out.svg"
        # Bypass the file dialog.
        monkeypatch.setattr(
            "app.ui.main_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(out), ""),
        )
        window.action_export("svg")
        assert out.exists()

    def test_export_package(self, window, tmp_path, monkeypatch):
        load_image_into(window, tmp_path)
        generate_sync(window)
        monkeypatch.setattr(
            "app.ui.main_window.QFileDialog.getExistingDirectory",
            lambda *a, **k: str(tmp_path),
        )
        window.action_export_package()
        # A subfolder named after the project should now contain files.
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert subdirs
        assert any(f.suffix == ".svg" for f in subdirs[0].iterdir())


class TestAutosaveWiring:
    def test_autosave_provider_registered(self, window):
        project, image, dirty = window._autosave._provider()
        # No project yet on the welcome screen.
        assert project is None


class TestUnsavedGuard:
    def test_new_when_dirty_prompts(self, window, tmp_path, monkeypatch):
        load_image_into(window, tmp_path)
        window.state.mark_dirty(True)
        # Simulate the user choosing Discard.
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(
            "app.ui.main_window.QMessageBox.question",
            lambda *a, **k: QMessageBox.Discard,
        )
        window.action_new_project()
        assert window.state.project.name == "Untitled"
        assert not window.state.has_image
