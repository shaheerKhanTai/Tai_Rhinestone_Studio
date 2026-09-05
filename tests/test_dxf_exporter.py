"""Tests for the DXF exporter — validates layers, geometry and coordinates."""
from __future__ import annotations

import ezdxf
import pytest

from app.exporters.base import ExportError
from app.exporters.dxf_exporter import (
    LAYER_BOUNDARY,
    LAYER_GUIDES,
    LAYER_RHINESTONES,
    DxfOptions,
    build_dxf,
    export_dxf,
)
from app.models.design import Design


def make_design(n=4, w=100.0, h=80.0) -> Design:
    d = Design(w, h)
    for i in range(n):
        d.add_stone(10 + i * 15, 20, 2.8)
    return d


def circles(doc):
    return list(doc.modelspace().query("CIRCLE"))


class TestBuildDxf:
    def test_circle_per_stone(self):
        doc = build_dxf(make_design(5))
        assert len(circles(doc)) == 5

    def test_units_are_mm(self):
        doc = build_dxf(make_design(1))
        assert doc.header["$INSUNITS"] == 4  # mm

    def test_circle_radius_matches_stone(self):
        d = Design(100, 100)
        d.add_stone(50, 50, 2.8)
        doc = build_dxf(d)
        c = circles(doc)[0]
        assert c.dxf.radius == pytest.approx(1.4)

    def test_y_is_flipped(self):
        d = Design(100, 80)
        d.add_stone(30, 10, 2.8)  # near top in image space
        doc = build_dxf(d)
        c = circles(doc)[0]
        assert c.dxf.center.x == pytest.approx(30)
        assert c.dxf.center.y == pytest.approx(70)  # 80 - 10

    def test_rhinestone_layer(self):
        doc = build_dxf(make_design(2))
        for c in circles(doc):
            assert c.dxf.layer == LAYER_RHINESTONES

    def test_boundary_included_by_default(self):
        doc = build_dxf(make_design(1))
        assert LAYER_BOUNDARY in doc.layers
        polylines = list(doc.modelspace().query("LWPOLYLINE"))
        assert len(polylines) == 1

    def test_boundary_excluded(self):
        doc = build_dxf(make_design(1), DxfOptions(include_boundary=False))
        polylines = list(doc.modelspace().query("LWPOLYLINE"))
        assert polylines == []

    def test_guides_optional(self):
        doc = build_dxf(make_design(1), DxfOptions(include_guides=True))
        assert LAYER_GUIDES in doc.layers
        lines = list(doc.modelspace().query("LINE"))
        assert len(lines) == 2  # crosshair

    def test_no_guides_by_default(self):
        doc = build_dxf(make_design(1))
        assert LAYER_GUIDES not in doc.layers

    def test_hole_allowance(self):
        d = Design(100, 100)
        d.add_stone(50, 50, 2.8)
        doc = build_dxf(d, DxfOptions(hole_allowance_mm=1.0))
        c = circles(doc)[0]
        assert c.dxf.radius == pytest.approx(1.4 + 0.5)

    def test_empty_design(self):
        doc = build_dxf(Design(100, 100))
        assert circles(doc) == []

    def test_invalid_dimensions(self):
        d = make_design(1)
        d.width_mm = 0
        with pytest.raises(ExportError):
            build_dxf(d)


class TestExportDxf:
    def test_writes_readable_file(self, tmp_path):
        out = tmp_path / "design.dxf"
        export_dxf(make_design(3), out)
        assert out.exists()
        # Re-open with ezdxf to confirm validity.
        doc = ezdxf.readfile(str(out))
        assert len(list(doc.modelspace().query("CIRCLE"))) == 3

    def test_atomic_no_tmp(self, tmp_path):
        out = tmp_path / "design.dxf"
        export_dxf(make_design(2), out)
        assert not (tmp_path / "design.dxf.tmp").exists()

    def test_creates_parent_dir(self, tmp_path):
        out = tmp_path / "nested" / "d.dxf"
        export_dxf(make_design(1), out)
        assert out.exists()

    def test_coordinates_survive_roundtrip(self, tmp_path):
        d = Design(100, 80)
        d.add_stone(25, 15, 3.0)
        out = tmp_path / "d.dxf"
        export_dxf(d, out)
        doc = ezdxf.readfile(str(out))
        c = list(doc.modelspace().query("CIRCLE"))[0]
        assert c.dxf.center.x == pytest.approx(25)
        assert c.dxf.center.y == pytest.approx(65)  # flipped
        assert c.dxf.radius == pytest.approx(1.5)
