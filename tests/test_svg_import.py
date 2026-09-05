"""Tests for SVG import (ISSUE 1) — rasterisation, dimensions, viewBox, errors.

These exercise :func:`app.core.image_processor.load_image` with ``.svg`` inputs,
verifying the SVG enters the same pipeline as raster formats and that its
geometry/aspect/transparency is handled correctly. QtSvg rasterises via
:func:`_load_svg`; tests set ``QT_QPA_PLATFORM=offscreen`` via conftest so they
run headless.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.image_processor import (
    SUPPORTED_EXTENSIONS,
    ImageLoadError,
    load_image,
    process_to_density,
    DensityParams,
)


def _write_svg(path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


class TestSvgSupported:
    def test_svg_in_supported_extensions(self):
        assert ".svg" in SUPPORTED_EXTENSIONS


class TestSvgLoad:
    def test_load_svg_dimensions_from_width_height(self, tmp_path):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80">'
            '<rect width="120" height="80" fill="#ff0000"/></svg>'
        )
        p = tmp_path / "rect.svg"
        _write_svg(p, svg)
        img = load_image(p)
        assert img.format == "SVG"
        assert img.size == (120, 80)
        assert img.rgba.shape == (80, 120, 4)

    def test_load_svg_pixel_is_correct_colour(self, tmp_path):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
            '<rect width="40" height="40" fill="#00ff00"/></svg>'
        )
        p = tmp_path / "green.svg"
        _write_svg(p, svg)
        img = load_image(p)
        # Green fill, fully opaque.
        px = img.rgba[20, 20]
        assert px[0] == 0 and px[1] == 255 and px[2] == 0 and px[3] == 255

    def test_load_svg_with_viewbox_only(self, tmp_path):
        # viewBox but no width/height — dimensions derived from viewBox.
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
            '<rect width="200" height="100" fill="#0000ff"/></svg>'
        )
        p = tmp_path / "vb.svg"
        _write_svg(p, svg)
        img = load_image(p)
        assert img.size == (200, 100)

    def test_load_svg_aspect_preserved_no_xy_stretch(self, tmp_path):
        # width given, no height — height derived from viewBox aspect (no stretch).
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="300" viewBox="0 0 150 50">'
            '<rect width="150" height="50" fill="#888888"/></svg>'
        )
        p = tmp_path / "aspect.svg"
        _write_svg(p, svg)
        img = load_image(p)
        # 300 wide, aspect 3:1 → 100 tall. Never stretched independently.
        assert img.width == 300
        assert img.height == 100

    def test_load_svg_transparency_flagged(self, tmp_path):
        # Transparent background, opaque circle in the middle.
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">'
            '<circle cx="30" cy="30" r="10" fill="#ff0000"/></svg>'
        )
        p = tmp_path / "transparent.svg"
        _write_svg(p, svg)
        img = load_image(p)
        assert img.has_alpha is True
        # Corner (outside the circle) must be transparent.
        assert img.rgba[0, 0, 3] == 0
        # Centre of the circle must be opaque red.
        assert img.rgba[30, 30, 3] == 255
        assert img.rgba[30, 30, 0] == 255


class TestSvgErrors:
    def test_invalid_svg_raises(self, tmp_path):
        p = tmp_path / "bad.svg"
        _write_svg(p, "<not an svg></not an svg>")
        with pytest.raises(ImageLoadError):
            load_image(p)

    def test_corrupt_svg_raises(self, tmp_path):
        p = tmp_path / "broken.svg"
        _write_svg(p, "<svg xmlns= broken")
        with pytest.raises(ImageLoadError):
            load_image(p)

    def test_empty_svg_raises(self, tmp_path):
        p = tmp_path / "empty.svg"
        _write_svg(p, "   ")
        with pytest.raises(ImageLoadError):
            load_image(p)

    def test_svg_does_not_execute_scripts(self, tmp_path):
        # QSvgRenderer is a static renderer: a script tag must not run and must
        # not crash the loader. The SVG must still load (rendered without the script).
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
            '<rect width="20" height="20" fill="#ff0000"/>'
            '<script type="text/ecmascript">evil()</script>'
            '</svg>'
        )
        p = tmp_path / "script.svg"
        _write_svg(p, svg)
        img = load_image(p)
        assert img.size == (20, 20)


class TestSvgEdgeCases:
    def test_svg_physical_units_converted(self, tmp_path):
        # width/height in mm must convert to px at 96 DPI without distorting
        # the aspect ratio (50mm x 25mm -> ~189x94px, 2:1 aspect; the small
        # deviation is integer-pixel rounding, not anisotropic stretching).
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="50mm" height="25mm" '
            'viewBox="0 0 100 50">'
            '<rect width="100" height="50" fill="#333333"/></svg>'
        )
        p = tmp_path / "mm.svg"
        _write_svg(p, svg)
        img = load_image(p)
        assert img.width == 189 and img.height == 94  # round(50*96/25.4), round(25*96/25.4)
        assert abs(img.width / img.height - 2.0) < 0.02

    def test_svg_downscaled_to_default_max(self, tmp_path):
        # An SVG claiming a huge intrinsic size is rasterised capped at the
        # default longest side, preserving aspect.
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="5000" height="2500" '
            'viewBox="0 0 5000 2500">'
            '<rect width="5000" height="2500" fill="#000000"/></svg>'
        )
        p = tmp_path / "huge.svg"
        _write_svg(p, svg)
        img = load_image(p)
        assert max(img.size) <= 1400
        assert abs(img.width / img.height - 2.0) < 0.01

    def test_svg_raster_import_matches_raster_pipeline(self, tmp_path):
        # The same square artwork as PNG and SVG must both load into the
        # identical internal representation (H, W, 4) uint8 RGBA.
        import numpy as np
        from PIL import Image as PILImage

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50">'
            '<rect width="50" height="50" fill="#000000"/></svg>'
        )
        p_svg = tmp_path / "sq.svg"
        _write_svg(p_svg, svg)
        p_png = tmp_path / "sq.png"
        PILImage.new("RGBA", (50, 50), (0, 0, 0, 255)).save(p_png)
        a = load_image(p_svg)
        b = load_image(p_png)
        assert a.rgba.shape == b.rgba.shape == (50, 50, 4)


class TestSvgToRhinestones:
    def test_svg_runs_through_generation_pipeline(self, tmp_path):
        # A solid dark SVG must produce stones — proving the SVG output flows
        # through the same density→placement pipeline as raster images.
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">'
            '<rect width="80" height="80" fill="#000000"/></svg>'
        )
        p = tmp_path / "solid.svg"
        _write_svg(p, svg)
        img = load_image(p)
        field = process_to_density(img.rgba, DensityParams(density=0.8), max_dim=80)
        assert field.density.shape == (80, 80)
        # Solid black → high density across the field.
        assert field.density.mean() > 0.5

    def test_svg_transparent_background_zero_density(self, tmp_path):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60">'
            '<circle cx="30" cy="30" r="8" fill="#000000"/></svg>'
        )
        p = tmp_path / "circle.svg"
        _write_svg(p, svg)
        img = load_image(p)
        field = process_to_density(img.rgba, DensityParams(density=0.5), max_dim=60)
        # Transparent corners must have zero density.
        assert field.density[0, 0] == pytest.approx(0.0)

    def test_svg_full_generation_produces_spaced_stones(self, tmp_path):
        # End-to-end: SVG -> load -> generate() -> stones in physical mm with
        # valid spacing (the acceptance-workflow core path).
        from app.core.rhinestone_engine import generate
        from app.models.settings import GenerationSettings

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
            '<rect width="100" height="100" fill="#000000"/></svg>'
        )
        p = tmp_path / "solid.svg"
        _write_svg(p, svg)
        img = load_image(p)
        settings = GenerationSettings(stone_size_id="SS10", spacing_mm=0.5)
        result = generate(img.rgba, settings, 100.0, 100.0)
        assert result.stone_count > 50
        # Every pair of stones must respect the pitch.
        from app.core.collision import find_collisions

        pairs = find_collisions(
            result.placement.positions_mm,
            settings.effective_diameter() + np.zeros(result.stone_count),
            settings.spacing_mm,
        )
        assert pairs == []


class TestSvgDialogFilter:
    def test_file_dialog_filter_includes_svg(self):
        # The import dialog's filter must accept .svg alongside raster formats.
        import app.ui.main_window as mw

        assert "*.svg" in mw._IMAGE_FILTER
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
            assert ext in mw._IMAGE_FILTER
