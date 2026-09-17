"""Tests for project I/O, export orchestration, and autosave."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.image_processor import loaded_from_array
from app.models.project import Project
from app.services import export_service, project_service
from app.services.project_service import PROJECT_SUFFIX, ProjectIOError


def sample_project(n=8) -> Project:
    p = Project.new("Test Eagle")
    p.set_physical_size(254.0, 203.2)
    for i in range(n):
        p.design.add_stone(10 + i * 10, 20, 2.8)
    return p


def sample_image():
    rgba = np.zeros((40, 60, 4), dtype=np.uint8)
    rgba[:, :] = (10, 20, 30, 255)
    return loaded_from_array(rgba)


class TestProjectIO:
    def test_save_adds_suffix(self, tmp_path):
        out = project_service.save_project(sample_project(), tmp_path / "proj")
        assert out.suffix == PROJECT_SUFFIX

    def test_roundtrip_without_image(self, tmp_path):
        p = sample_project(5)
        out = project_service.save_project(p, tmp_path / "p.rhinestone")
        loaded, image = project_service.load_project(out)
        assert loaded.name == "Test Eagle"
        assert loaded.design.stone_count == 5
        assert image is None

    def test_roundtrip_with_image(self, tmp_path):
        p = sample_project(3)
        img = sample_image()
        out = project_service.save_project(p, tmp_path / "p.rhinestone", img)
        loaded, image = project_service.load_project(out)
        assert image is not None
        assert image.width == 60 and image.height == 40

    def test_preserves_settings_and_size(self, tmp_path):
        p = sample_project(1)
        p.settings.stone_size_id = "SS16"
        p.settings.spacing_mm = 1.1
        out = project_service.save_project(p, tmp_path / "p.rhinestone")
        loaded, _ = project_service.load_project(out)
        assert loaded.settings.stone_size_id == "SS16"
        assert loaded.settings.spacing_mm == pytest.approx(1.1)
        assert loaded.width_mm == pytest.approx(254.0)

    def test_stone_positions_preserved(self, tmp_path):
        p = sample_project(4)
        original = [(s.x, s.y) for s in p.design.stones]
        out = project_service.save_project(p, tmp_path / "p.rhinestone")
        loaded, _ = project_service.load_project(out)
        assert [(s.x, s.y) for s in loaded.design.stones] == original

    def test_atomic_no_tmp_left(self, tmp_path):
        out = project_service.save_project(sample_project(), tmp_path / "p.rhinestone")
        assert not out.with_name(out.name + ".tmp").exists()

    def test_load_missing(self, tmp_path):
        with pytest.raises(ProjectIOError):
            project_service.load_project(tmp_path / "nope.rhinestone")

    def test_load_corrupt(self, tmp_path):
        bad = tmp_path / "bad.rhinestone"
        bad.write_bytes(b"not a zip file")
        with pytest.raises(ProjectIOError):
            project_service.load_project(bad)

    def test_load_zip_without_project(self, tmp_path):
        import zipfile

        bad = tmp_path / "empty.rhinestone"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("readme.txt", "hello")
        with pytest.raises(ProjectIOError):
            project_service.load_project(bad)


class TestValidation:
    def test_empty_design_fails(self):
        result = export_service.validate_for_export(Project.new())
        assert not result.ok
        assert any("no stones" in e.lower() for e in result.errors)

    def test_valid_design_passes(self):
        result = export_service.validate_for_export(sample_project())
        assert result.ok

    def test_outside_boundary_warns(self):
        p = sample_project(0)
        p.design.add_stone(-50, -50, 2.8)  # outside
        result = export_service.validate_for_export(p)
        assert result.warnings


class TestExportService:
    @pytest.mark.parametrize("kind,ext", [("svg", ".svg"), ("cricut", ".svg"),
                                          ("dxf", ".dxf"), ("png", ".png"), ("pdf", ".pdf")])
    def test_export_single(self, tmp_path, kind, ext):
        out = tmp_path / f"out{ext}"
        export_service.export_single(sample_project(), kind, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_unknown_kind(self, tmp_path):
        from app.exporters.base import ExportError

        with pytest.raises(ExportError):
            export_service.export_single(sample_project(), "eps", tmp_path / "x.eps")

    def test_production_package(self, tmp_path):
        result = export_service.export_production_package(sample_project(), tmp_path / "pkg")
        names = {f.name for f in result.files}
        assert "Test_Eagle.svg" in names
        assert "Test_Eagle.dxf" in names
        assert "Test_Eagle.png" in names
        assert "Test_Eagle.pdf" in names
        assert "Test_Eagle-cricut.svg" in names
        for f in result.files:
            assert f.exists()

    def test_package_without_cricut(self, tmp_path):
        result = export_service.export_production_package(
            sample_project(), tmp_path / "pkg", include_cricut=False
        )
        assert not any("cricut" in f.name for f in result.files)

    def test_safe_stem(self):
        assert export_service._safe_stem("My Design!") == "My_Design_"
        assert export_service._safe_stem("") == "rhinestone_design"
        assert export_service._safe_stem("  a  b  ") == "a_b"


class TestAutosave:
    def test_save_and_recover(self, qapp):
        from app.services.autosave import AutosaveService

        svc = AutosaveService()
        AutosaveService.clear_recovery()
        assert not AutosaveService.has_recovery()
        path = svc.save_now(sample_project(4), sample_image())
        assert path is not None
        assert AutosaveService.has_recovery()
        # Recover.
        project, image = project_service.load_project(AutosaveService.recovery_file())
        assert project.design.stone_count == 4
        assert image is not None
        AutosaveService.clear_recovery()

    def test_provider_tick_skips_clean(self, qapp):
        from app.services.autosave import AutosaveService

        AutosaveService.clear_recovery()
        svc = AutosaveService()
        svc.set_provider(lambda: (sample_project(3), None, False))  # dirty=False
        svc._tick()
        assert not AutosaveService.has_recovery()

    def test_provider_tick_saves_dirty(self, qapp):
        from app.services.autosave import AutosaveService

        AutosaveService.clear_recovery()
        svc = AutosaveService()
        svc.set_provider(lambda: (sample_project(3), None, True))  # dirty
        svc._tick()
        assert AutosaveService.has_recovery()
        AutosaveService.clear_recovery()


class TestTraceMaskIO:
    def test_roundtrip_preserves_mask(self, tmp_path):
        from app.core.trace_mask import blank_mask, paint_stroke

        p = sample_project(2)
        p.set_physical_size(100.0, 100.0)
        p.trace_mask = blank_mask(100.0, 100.0)
        paint_stroke(p.trace_mask, (20.0, 50.0), (80.0, 50.0), 6.0, 100.0, 100.0)
        out = project_service.save_project(p, tmp_path / "t.rhinestone")
        loaded, _ = project_service.load_project(out)
        assert loaded.trace_mask is not None
        assert loaded.trace_mask.shape == p.trace_mask.shape
        np.testing.assert_array_equal(loaded.trace_mask, p.trace_mask)

    def test_maskless_project_loads_none(self, tmp_path):
        p = sample_project(1)
        out = project_service.save_project(p, tmp_path / "t.rhinestone")
        loaded, _ = project_service.load_project(out)
        assert loaded.trace_mask is None

    def test_has_trace_flag_in_project_json(self, tmp_path):
        import json
        import zipfile

        p = sample_project(1)
        p.trace_mask = None
        out = project_service.save_project(p, tmp_path / "t.rhinestone")
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            data = json.loads(zf.read("project.json"))
        assert "trace.png" not in names
        assert data["has_trace"] is False
