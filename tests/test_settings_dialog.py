"""Tests for the Settings dialog and its main-window wiring (Phase 17).

Runs offscreen via conftest. The dialog is a pure view over ``AppSettings``:
it reads one in and returns an edited copy; the window applies + persists it.
"""
from __future__ import annotations

import pytest

from app.core.geometry import Unit
from app.models.app_settings import AppSettings
from app.ui.theme import ThemeMode


@pytest.fixture()
def dialog(qapp):
    from app.ui.dialogs.settings_dialog import SettingsDialog

    s = AppSettings()
    dlg = SettingsDialog(s)
    yield dlg
    dlg.deleteLater()


class TestSettingsDialog:
    def test_loads_source_values(self, qapp):
        from app.ui.dialogs.settings_dialog import SettingsDialog

        src = AppSettings(
            theme="light",
            default_units=Unit.MM,
            default_stone_size_id="SS16",
            autosave_enabled=False,
            autosave_interval_sec=300,
            show_grid=True,
            background_opacity=70,
            analysis_max_dim=2000,
            confirm_delete_threshold=40,
        )
        dlg = SettingsDialog(src)
        assert dlg._theme_combo.currentData() == "light"
        assert dlg._units_combo.currentData() is Unit.MM
        assert dlg._stone_combo.currentData() == "SS16"
        assert dlg._autosave_enabled.isChecked() is False
        # Interval spin is disabled when autosave is off.
        assert not dlg._autosave_interval.isEnabled()
        assert dlg._show_grid.isChecked() is True
        assert dlg._analysis_dim.value() == 2000
        assert dlg._delete_threshold.value() == 40
        assert dlg._bg_opacity.value() == pytest.approx(0.70, abs=0.01)
        dlg.deleteLater()

    def test_result_reflects_edits(self, dialog):
        dialog._theme_combo.setCurrentIndex(dialog._theme_combo.findData("light"))
        dialog._units_combo.setCurrentIndex(dialog._units_combo.findData(Unit.CM))
        dialog._show_rulers.setChecked(False)
        dialog._bg_opacity.set_value(0.25)
        dialog._analysis_dim.setValue(1600)

        out = dialog.result_settings()
        assert out.theme == "light"
        assert out.default_units is Unit.CM
        assert out.show_rulers is False
        assert out.background_opacity == 25
        assert out.analysis_max_dim == 1600

    def test_result_preserves_unexposed_fields(self, qapp):
        from app.ui.dialogs.settings_dialog import SettingsDialog

        src = AppSettings()
        src.add_recent_project("C:/designs/eagle.rhinestone")
        dlg = SettingsDialog(src)
        out = dlg.result_settings()
        # Recent projects aren't editable here, but must survive a round-trip.
        assert out.recent_projects == ["C:/designs/eagle.rhinestone"]
        dlg.deleteLater()

    def test_result_is_clamped(self, dialog):
        # Values are held within widget ranges, and clamp() runs on output.
        dialog._autosave_interval.setValue(15)
        out = dialog.result_settings()
        assert 15 <= out.autosave_interval_sec <= 3600

    def test_restore_defaults(self, qapp):
        from app.ui.dialogs.settings_dialog import SettingsDialog

        src = AppSettings(theme="light", analysis_max_dim=3000)
        dlg = SettingsDialog(src)
        dlg._restore_defaults()
        out = dlg.result_settings()
        d = AppSettings()
        assert out.theme == d.theme
        assert out.analysis_max_dim == d.analysis_max_dim
        dlg.deleteLater()

    def test_empty_export_dir_is_none(self, dialog):
        dialog._export_dir.setText("   ")
        assert dialog.result_settings().default_export_dir is None


class TestDialogSizing:
    """The dialog must be fully reachable on any display.

    The reported problem was the window running off the bottom of the screen
    with no way to scroll to the lower groups, so these pin both halves of the
    fix: the dialog sizes itself to the screen it opens on, and the groups
    scroll inside it independently of the button box.
    """

    def test_groups_live_in_a_scroll_area(self, dialog):
        from PySide6.QtWidgets import QScrollArea

        assert isinstance(dialog._scroll, QScrollArea)
        assert dialog._scroll.widget() is not None
        assert dialog._scroll.widgetResizable() is True

    def test_button_box_is_outside_the_scroll_area(self, dialog):
        """OK / Cancel must never be scrolled off — they are not in the scroller."""
        from PySide6.QtWidgets import QDialogButtonBox

        box = dialog.findChild(QDialogButtonBox)
        assert box is not None
        # The scroll area must not be an ancestor of the button box.
        ancestor = box.parentWidget()
        while ancestor is not None:
            assert ancestor is not dialog._scroll
            ancestor = ancestor.parentWidget()

    def test_dialog_fits_the_available_screen(self, dialog):
        from PySide6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        assert screen is not None
        available = screen.availableGeometry()
        assert dialog.width() <= available.width()
        assert dialog.height() <= available.height()

    def test_dialog_is_at_least_the_minimum_size(self, dialog):
        from app.ui.dialogs.settings_dialog import _MIN_HEIGHT, _MIN_WIDTH

        assert dialog.width() >= _MIN_WIDTH
        assert dialog.height() >= _MIN_HEIGHT

    def test_scaling_up_keeps_every_group_present(self, dialog):
        """Sizing must not drop controls — they all still exist and are reachable."""
        for name in ("_theme_combo", "_units_combo", "_stone_combo", "_export_dir",
                     "_autosave_enabled", "_autosave_interval", "_show_grid",
                     "_show_rulers", "_wheel_zooms", "_bg_opacity", "_analysis_dim",
                     "_confirm_overwrite", "_delete_threshold"):
            assert getattr(dialog, name) is not None, name

    def test_restore_defaults_refits(self, dialog):
        from PySide6.QtGui import QGuiApplication

        dialog.resize(2000, 2000)
        dialog._restore_defaults()
        available = QGuiApplication.primaryScreen().availableGeometry()
        assert dialog.height() <= available.height()


@pytest.fixture()
def window(qapp):
    from app.ui.main_window import MainWindow

    w = MainWindow(AppSettings())
    yield w
    w.state.dirty = False
    w.close()
    w.deleteLater()


class TestSettingsWiring:
    def test_apply_switches_theme_live(self, window):
        new = AppSettings.from_dict(window.app_settings.to_dict())
        new.theme = "light"
        window._apply_app_settings(new)
        assert window.theme.mode is ThemeMode.LIGHT
        assert window.act_theme_light.isChecked()
        assert not window.act_theme_dark.isChecked()

    def test_apply_pushes_canvas_prefs(self, window):
        new = AppSettings.from_dict(window.app_settings.to_dict())
        new.show_grid = True
        new.background_opacity = 80
        new.wheel_zooms = True
        window._apply_app_settings(new)
        assert window.act_show_grid.isChecked()
        assert window._canvas._show_grid is True
        assert window._canvas._wheel_zooms is True

    def test_apply_reconfigures_autosave(self, window):
        new = AppSettings.from_dict(window.app_settings.to_dict())
        new.autosave_interval_sec = 240
        window._apply_app_settings(new)
        assert window.app_settings.autosave_interval_sec == 240

    def test_action_settings_accept_persists(self, window, monkeypatch):
        from app.ui.dialogs import settings_dialog as sd
        from app.services import settings_service

        saved = {}
        monkeypatch.setattr(settings_service, "save_settings", lambda s: saved.update(theme=s.theme))

        # Auto-accept the dialog with an edit applied.
        def fake_exec(self):
            self._theme_combo.setCurrentIndex(self._theme_combo.findData("light"))
            return sd.QDialog.Accepted

        monkeypatch.setattr(sd.SettingsDialog, "exec", fake_exec)
        window.action_settings()
        assert window.theme.mode is ThemeMode.LIGHT
        assert saved.get("theme") == "light"
