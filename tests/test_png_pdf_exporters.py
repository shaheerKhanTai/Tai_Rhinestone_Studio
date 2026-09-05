"""Tests for the PNG preview and PDF production-sheet exporters."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.geometry import Unit
from app.exporters.base import ExportError
from app.exporters.pdf_exporter import ProductionSheetInfo, build_pdf_bytes, export_pdf
from app.exporters.png_exporter import (
    PngOptions,
    encode_png,
    export_png,
    render_design_png,
)
from app.models.design import Design


def make_design(n=6, w=100.0, h=80.0) -> Design:
    d = Design(w, h)
    for i in range(n):
        d.add_stone(10 + (i % 3) * 20, 10 + (i // 3) * 20, 2.8)
    return d


class TestPng:
    def test_render_shape_matches_aspect(self):
        d = make_design(w=100.0, h=50.0)  # 2:1
        rgba = render_design_png(d, PngOptions(width_px=1000))
        assert rgba.shape[2] == 4
        assert rgba.shape[1] == 1000
        # Height ≈ width/2 plus padding.
        assert rgba.shape[0] < rgba.shape[1]

    def test_transparent_background(self):
        rgba = render_design_png(make_design(1), PngOptions(width_px=200, background=None))
        # Corner pixel is transparent.
        assert rgba[0, 0, 3] == 0

    def test_paper_background_opaque(self):
        rgba = render_design_png(make_design(1), PngOptions(width_px=200))
        assert rgba[0, 0, 3] == 255

    def test_stones_drawn(self):
        # A design with stones should have some dark pixels over the paper.
        rgba = render_design_png(make_design(6), PngOptions(width_px=400))
        dark = np.sum(rgba[:, :, :3].sum(axis=2) < 300)
        assert dark > 0

    def test_export_creates_valid_png(self, tmp_path):
        out = tmp_path / "preview.png"
        export_png(make_design(4), out, PngOptions(width_px=300))
        assert out.exists()
        header = out.read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n"

    def test_encode_roundtrip(self):
        data = encode_png(render_design_png(make_design(2), PngOptions(width_px=200)))
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_invalid_dimensions(self):
        d = make_design(1)
        d.width_mm = 0
        with pytest.raises(ExportError):
            render_design_png(d)

    def test_empty_design_still_renders(self):
        rgba = render_design_png(Design(100, 100), PngOptions(width_px=200))
        assert rgba.shape[1] == 200


class TestPdf:
    def _info(self, design):
        return ProductionSheetInfo(
            design_name="Eagle Logo",
            unit=Unit.INCH,
            stone_size_id="SS10",
            stone_diameter_mm=2.8,
            spacing_mm=0.5,
            color_counts=design.color_counts(),
        )

    def test_build_pdf_bytes(self):
        d = make_design(10)
        data = build_pdf_bytes(d, self._info(d))
        assert data[:5] == b"%PDF-"
        assert len(data) > 1000

    def test_export_pdf_file(self, tmp_path):
        d = make_design(10)
        out = tmp_path / "sheet.pdf"
        export_pdf(d, out, self._info(d))
        assert out.exists()
        assert out.read_bytes()[:5] == b"%PDF-"

    def test_empty_design_pdf(self):
        d = Design(100, 80)
        data = build_pdf_bytes(d, self._info(d))
        assert data[:5] == b"%PDF-"

    def test_multicolor_breakdown(self):
        d = Design(100, 80)
        d.add_stone(10, 10, 2.8, color="#FF0000")
        d.add_stone(20, 20, 2.8, color="#00FF00")
        data = build_pdf_bytes(d, self._info(d))
        assert data[:5] == b"%PDF-"

    def test_invalid_dimensions(self):
        d = make_design(1)
        d.width_mm = -5
        with pytest.raises(ExportError):
            build_pdf_bytes(d, self._info(d))

    def test_atomic_no_tmp(self, tmp_path):
        d = make_design(3)
        out = tmp_path / "sheet.pdf"
        export_pdf(d, out, self._info(d))
        assert not (tmp_path / "sheet.pdf.tmp").exists()
