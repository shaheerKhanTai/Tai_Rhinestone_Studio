"""Tests for the SVG exporter — validates real vector output and dimensions."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from app.core.geometry import Unit
from app.exporters.base import ExportError
from app.exporters.svg_exporter import (
    SvgOptions,
    export_cricut_svg,
    export_svg,
    render_svg,
)
from app.models.design import Design

SVG_NS = "{http://www.w3.org/2000/svg}"


def make_design(n=3, w=254.0, h=203.2) -> Design:
    d = Design(w, h)
    for i in range(n):
        x = 10.0 + (i % 5) * 20.0
        y = 10.0 + (i // 5) * 20.0
        d.add_stone(x, y, 2.8, color="#1F2933")
    return d


def find_all_circles(root) -> list:
    return root.findall(f".//{SVG_NS}circle")


class TestRenderStructure:
    def test_valid_xml(self):
        svg = render_svg(make_design(), SvgOptions())
        root = ET.fromstring(svg)
        assert root.tag == f"{SVG_NS}svg"

    def test_circle_per_stone(self):
        d = make_design(5)
        root = ET.fromstring(render_svg(d, SvgOptions()))
        assert len(find_all_circles(root)) == 5

    def test_empty_design_valid(self):
        d = Design(100, 80)
        root = ET.fromstring(render_svg(d, SvgOptions()))
        assert len(find_all_circles(root)) == 0
        assert root.get("width") == "100mm"

    def test_declares_svg_namespace(self):
        svg = render_svg(make_design(), SvgOptions())
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg


class TestPhysicalDimensions:
    def test_width_height_in_mm(self):
        d = make_design(1, w=254.0, h=203.2)
        root = ET.fromstring(render_svg(d, SvgOptions(doc_unit=Unit.MM)))
        assert root.get("width") == "254mm"
        assert root.get("height") == "203.2mm"

    def test_viewbox_matches(self):
        d = make_design(1, w=254.0, h=203.2)
        root = ET.fromstring(render_svg(d, SvgOptions()))
        assert root.get("viewBox") == "0 0 254 203.2"

    def test_circle_coords_are_physical_mm(self):
        d = Design(100, 100)
        d.add_stone(25.0, 40.0, 2.8)
        root = ET.fromstring(render_svg(d, SvgOptions(doc_unit=Unit.MM)))
        c = find_all_circles(root)[0]
        assert float(c.get("cx")) == pytest.approx(25.0)
        assert float(c.get("cy")) == pytest.approx(40.0)
        assert float(c.get("r")) == pytest.approx(1.4)  # radius of 2.8mm stone

    def test_inch_document_scales_coords(self):
        d = Design(254.0, 254.0)  # 10 x 10 inches
        d.add_stone(25.4, 50.8, 2.8)  # 1in, 2in
        root = ET.fromstring(render_svg(d, SvgOptions(doc_unit=Unit.INCH)))
        assert root.get("width") == "10in"
        c = find_all_circles(root)[0]
        assert float(c.get("cx")) == pytest.approx(1.0, abs=1e-4)
        assert float(c.get("cy")) == pytest.approx(2.0, abs=1e-4)


class TestColorGrouping:
    def test_groups_by_color(self):
        d = Design(100, 100)
        d.add_stone(10, 10, 2.8, color="#FF0000")
        d.add_stone(20, 20, 2.8, color="#FF0000")
        d.add_stone(30, 30, 2.8, color="#00FF00")
        root = ET.fromstring(render_svg(d, SvgOptions(per_stone_color=True)))
        groups = root.findall(f".//{SVG_NS}g/{SVG_NS}g")
        fills = {g.get("fill") for g in groups}
        assert fills == {"#FF0000", "#00FF00"}

    def test_boundary_rect(self):
        root = ET.fromstring(render_svg(make_design(), SvgOptions(include_boundary=True)))
        rects = root.findall(f".//{SVG_NS}rect")
        assert len(rects) == 1
        assert rects[0].get("fill") == "none"


class TestFileExport:
    def test_export_creates_file(self, tmp_path):
        out = tmp_path / "design.svg"
        export_svg(make_design(3), out)
        assert out.exists()
        root = ET.parse(out).getroot()
        assert len(find_all_circles(root)) == 3

    def test_export_atomic_no_tmp_left(self, tmp_path):
        out = tmp_path / "design.svg"
        export_svg(make_design(), out)
        assert not (tmp_path / "design.svg.tmp").exists()

    def test_export_creates_parent_dir(self, tmp_path):
        out = tmp_path / "nested" / "sub" / "design.svg"
        export_svg(make_design(), out)
        assert out.exists()

    def test_invalid_design_raises(self, tmp_path):
        d = make_design(1)
        d.width_mm = 0  # corrupt after construction
        with pytest.raises(ExportError):
            export_svg(d, tmp_path / "bad.svg")


class TestCricutExport:
    def test_uses_inches(self, tmp_path):
        out = tmp_path / "cut.svg"
        export_cricut_svg(make_design(1, w=254.0, h=203.2), out)
        root = ET.parse(out).getroot()
        assert root.get("width") == "10in"
        assert root.get("height") == "8in"

    def test_single_cut_layer(self, tmp_path):
        out = tmp_path / "cut.svg"
        export_cricut_svg(make_design(4), out)
        root = ET.parse(out).getroot()
        # One top-level group with a single fill; no per-colour subgroups.
        top_groups = root.findall(f"{SVG_NS}g")
        assert len(top_groups) == 1
        assert top_groups[0].get("fill") == "#000000"
        assert top_groups[0].findall(f"{SVG_NS}g") == []

    def test_hole_allowance_enlarges_circles(self, tmp_path):
        d = Design(254.0, 254.0)
        d.add_stone(25.4, 25.4, 2.8)  # radius 1.4mm
        out = tmp_path / "cut.svg"
        export_cricut_svg(d, out, hole_allowance_mm=1.0)  # +0.5mm radius
        root = ET.parse(out).getroot()
        c = root.findall(f".//{SVG_NS}circle")[0]
        # (1.4 + 0.5) mm in inches
        assert float(c.get("r")) == pytest.approx((1.4 + 0.5) / 25.4, abs=1e-5)

    def test_negative_allowance_rejected(self, tmp_path):
        with pytest.raises(ExportError):
            export_cricut_svg(make_design(), tmp_path / "cut.svg", hole_allowance_mm=-1.0)

    def test_stone_count_in_desc(self, tmp_path):
        out = tmp_path / "cut.svg"
        export_cricut_svg(make_design(7), out)
        text = out.read_text(encoding="utf-8")
        assert "7 stones" in text
