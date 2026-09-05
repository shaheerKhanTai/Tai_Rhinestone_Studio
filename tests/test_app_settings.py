"""Tests for AppSettings, path helpers, and settings persistence."""
from __future__ import annotations

import json

from app.core.geometry import Unit
from app.models.app_settings import MAX_RECENT, AppSettings
from app.services import paths, settings_service


class TestAppSettings:
    def test_defaults(self):
        s = AppSettings()
        assert s.theme == "dark"
        assert s.default_units is Unit.INCH
        assert s.autosave_enabled

    def test_recent_projects_dedup_and_order(self):
        s = AppSettings()
        s.add_recent_project("a.rhinestone")
        s.add_recent_project("b.rhinestone")
        s.add_recent_project("a.rhinestone")  # re-add moves to front
        assert s.recent_projects[0] == "a.rhinestone"
        assert s.recent_projects.count("a.rhinestone") == 1

    def test_recent_projects_capped(self):
        s = AppSettings()
        for i in range(MAX_RECENT + 5):
            s.add_recent_project(f"p{i}.rhinestone")
        assert len(s.recent_projects) == MAX_RECENT

    def test_clamp_ranges(self):
        s = AppSettings(autosave_interval_sec=1, background_opacity=999, analysis_max_dim=10)
        s.clamp()
        assert s.autosave_interval_sec >= 15
        assert s.background_opacity == 100
        assert s.analysis_max_dim >= 400

    def test_roundtrip(self):
        s = AppSettings(theme="light", default_units=Unit.MM, background_opacity=70)
        s.add_recent_project("x.rhinestone")
        restored = AppSettings.from_dict(s.to_dict())
        assert restored.theme == "light"
        assert restored.default_units is Unit.MM
        assert restored.background_opacity == 70
        assert restored.recent_projects == ["x.rhinestone"]

    def test_from_dict_tolerates_bad_unit(self):
        restored = AppSettings.from_dict({"default_units": "furlongs"})
        assert restored.default_units is Unit.INCH  # falls back


class TestPaths:
    def test_dirs_created(self):
        assert paths.app_data_dir().exists()
        assert paths.logs_dir().exists()
        assert paths.autosave_dir().exists()

    def test_uses_env_override(self):
        # conftest sets RHINESTONE_DATA_DIR to a temp dir.
        import os

        assert str(paths.app_data_dir()).startswith(os.environ["RHINESTONE_DATA_DIR"])


class TestSettingsService:
    def test_save_and_load(self):
        s = AppSettings(theme="light")
        s.add_recent_project("proj.rhinestone")
        settings_service.save_settings(s)
        loaded = settings_service.load_settings()
        assert loaded.theme == "light"
        assert "proj.rhinestone" in loaded.recent_projects

    def test_load_missing_returns_defaults(self, tmp_path, monkeypatch):
        missing = tmp_path / "nope.json"
        monkeypatch.setattr(settings_service, "settings_file", lambda: missing)
        assert settings_service.load_settings().theme == "dark"

    def test_load_corrupt_returns_defaults(self, tmp_path, monkeypatch):
        bad = tmp_path / "settings.json"
        bad.write_text("{ this is not json", encoding="utf-8")
        monkeypatch.setattr(settings_service, "settings_file", lambda: bad)
        assert settings_service.load_settings().theme == "dark"

    def test_presets_roundtrip(self):
        from app.models.presets import default_presets

        presets = default_presets()
        settings_service.save_presets(presets)
        loaded = settings_service.load_presets()
        assert len(loaded) == len(presets)
        assert loaded[0].name == presets[0].name
