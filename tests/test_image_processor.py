"""Tests for the image-processing pipeline."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.core.image_processor import (
    SUPPORTED_EXTENSIONS,
    DensityParams,
    ImageLoadError,
    ImageProcessingError,
    apply_contrast,
    apply_crop,
    apply_image_transforms,
    apply_orientation,
    build_density_map,
    edge_magnitude,
    extract_alpha01,
    load_image,
    loaded_from_array,
    local_texture,
    process_to_density,
    resize_max,
    smooth,
    to_luminance01,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def make_rgba(h=20, w=30, color=(0, 0, 0, 255)) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :] = color
    return arr


def save_png(path, arr) -> None:
    Image.fromarray(arr, mode="RGBA" if arr.shape[2] == 4 else "RGB").save(path)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class TestLoadImage:
    def test_load_png(self, tmp_path):
        p = tmp_path / "solid.png"
        save_png(p, make_rgba(color=(255, 0, 0, 255)))
        img = load_image(p)
        assert img.size == (30, 20)
        assert img.format == "PNG"
        assert img.rgba.shape == (20, 30, 4)
        assert img.file_size_bytes > 0

    def test_load_jpeg(self, tmp_path):
        p = tmp_path / "photo.jpg"
        Image.fromarray(make_rgba()[:, :, :3], "RGB").save(p)
        img = load_image(p)
        assert img.has_alpha is False
        assert img.rgba.shape[2] == 4  # normalised to RGBA

    def test_transparent_png_flagged(self, tmp_path):
        arr = make_rgba(color=(0, 0, 0, 0))
        arr[5:15, 5:15, 3] = 255  # opaque square on transparent field
        p = tmp_path / "logo.png"
        save_png(p, arr)
        img = load_image(p)
        assert img.has_alpha is True

    def test_fully_opaque_alpha_not_flagged(self, tmp_path):
        p = tmp_path / "opaque.png"
        save_png(p, make_rgba(color=(10, 20, 30, 255)))
        img = load_image(p)
        assert img.has_alpha is False

    def test_grayscale_image(self, tmp_path):
        gray = np.linspace(0, 255, 20 * 30, dtype=np.uint8).reshape(20, 30)
        p = tmp_path / "gray.png"
        Image.fromarray(gray, mode="L").save(p)
        img = load_image(p)
        assert img.rgba.shape == (20, 30, 4)
        # R, G, B equal for a grayscale source.
        np.testing.assert_array_equal(img.rgba[:, :, 0], img.rgba[:, :, 1])

    def test_missing_file(self, tmp_path):
        with pytest.raises(ImageLoadError):
            load_image(tmp_path / "nope.png")

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "file.gif"
        p.write_bytes(b"not really")
        with pytest.raises(ImageLoadError):
            load_image(p)

    def test_corrupt_file(self, tmp_path):
        p = tmp_path / "broken.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n garbage not a real png")
        with pytest.raises(ImageLoadError):
            load_image(p)

    def test_supported_extensions_present(self):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            assert ext in SUPPORTED_EXTENSIONS


class TestLoadedFromArray:
    def test_rgb_gets_alpha(self):
        img = loaded_from_array(np.zeros((10, 10, 3), dtype=np.uint8))
        assert img.rgba.shape[2] == 4
        assert img.has_alpha is False

    def test_bad_shape(self):
        with pytest.raises(ImageProcessingError):
            loaded_from_array(np.zeros((10, 10), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #


class TestTransforms:
    def test_rotate_90_swaps_dims(self):
        arr = make_rgba(h=10, w=20)
        out = apply_orientation(arr, rotation=90)
        assert out.shape[:2] == (20, 10)

    def test_rotate_180_preserves_dims(self):
        arr = make_rgba(h=10, w=20)
        out = apply_orientation(arr, rotation=180)
        assert out.shape[:2] == (10, 20)

    def test_flip_horizontal(self):
        arr = make_rgba(h=4, w=4)
        arr[:, 0] = (255, 255, 255, 255)  # white left column
        out = apply_orientation(arr, flip_h=True)
        np.testing.assert_array_equal(out[:, -1], arr[:, 0])

    def test_invalid_rotation(self):
        with pytest.raises(ImageProcessingError):
            apply_orientation(make_rgba(), rotation=45)

    def test_crop(self):
        arr = make_rgba(h=100, w=100)
        out = apply_crop(arr, (10, 20, 30, 40))
        assert out.shape[:2] == (40, 30)

    def test_crop_clamped(self):
        arr = make_rgba(h=50, w=50)
        out = apply_crop(arr, (40, 40, 100, 100))  # extends past edge
        assert out.shape[:2] == (10, 10)

    def test_crop_empty_raises(self):
        with pytest.raises(ImageProcessingError):
            apply_crop(make_rgba(h=50, w=50), (60, 60, 10, 10))

    def test_combined_transforms(self):
        arr = make_rgba(h=40, w=60)
        out = apply_image_transforms(arr, rotation=90, crop=(0, 0, 20, 20))
        assert out.shape[:2] == (20, 20)


class TestResize:
    def test_downscale(self):
        arr = make_rgba(h=2000, w=1000)
        out = resize_max(arr, 500)
        assert max(out.shape[:2]) == 500
        assert out.shape[:2] == (500, 250)

    def test_no_upscale(self):
        arr = make_rgba(h=100, w=100)
        out = resize_max(arr, 500)
        assert out.shape[:2] == (100, 100)


# --------------------------------------------------------------------------- #
# Channel analysis
# --------------------------------------------------------------------------- #


class TestChannels:
    def test_luminance_range(self):
        arr = make_rgba(color=(128, 128, 128, 255))
        lum = to_luminance01(arr)
        assert lum.dtype == np.float32
        assert lum.min() >= 0 and lum.max() <= 1
        assert lum.mean() == pytest.approx(128 / 255, abs=0.01)

    def test_white_is_bright(self):
        assert to_luminance01(make_rgba(color=(255, 255, 255, 255))).mean() == pytest.approx(1.0)

    def test_black_is_dark(self):
        assert to_luminance01(make_rgba(color=(0, 0, 0, 255))).mean() == pytest.approx(0.0)

    def test_alpha_extraction(self):
        arr = make_rgba(color=(0, 0, 0, 128))
        a = extract_alpha01(arr)
        assert a.mean() == pytest.approx(128 / 255, abs=0.01)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


class TestContrast:
    def test_identity_at_zero(self):
        x = np.linspace(0, 1, 50, dtype=np.float32)
        np.testing.assert_allclose(apply_contrast(x, 0.0), x, atol=1e-6)

    def test_endpoints_preserved(self):
        x = np.array([0.0, 1.0], dtype=np.float32)
        out = apply_contrast(x, 1.0)
        assert out[0] == pytest.approx(0.0, abs=1e-5)
        assert out[1] == pytest.approx(1.0, abs=1e-5)

    def test_monotonic(self):
        x = np.linspace(0, 1, 100, dtype=np.float32)
        out = apply_contrast(x, 0.8)
        assert np.all(np.diff(out) >= -1e-6)

    def test_increases_separation(self):
        # A value below the midpoint should be pushed lower.
        assert apply_contrast(np.array([0.3], dtype=np.float32), 1.0)[0] < 0.3


class TestSmooth:
    def test_zero_is_identity(self):
        g = np.random.default_rng(0).random((20, 20)).astype(np.float32)
        np.testing.assert_allclose(smooth(g, 0.0), g, atol=1e-6)

    def test_reduces_variance(self):
        g = np.random.default_rng(1).random((60, 60)).astype(np.float32)
        assert smooth(g, 1.0).var() < g.var()


class TestEdgesAndTexture:
    def test_edge_on_flat_is_zero(self):
        flat = np.full((30, 30), 0.5, dtype=np.float32)
        assert edge_magnitude(flat).max() == pytest.approx(0.0)

    def test_edge_detects_boundary(self):
        img = np.zeros((30, 30), dtype=np.float32)
        img[:, 15:] = 1.0  # vertical step edge
        e = edge_magnitude(img)
        assert e.max() == pytest.approx(1.0)
        assert e[:, 14:16].max() > 0.5

    def test_texture_on_flat_is_zero(self):
        flat = np.full((30, 30), 0.3, dtype=np.float32)
        assert local_texture(flat).max() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Density map
# --------------------------------------------------------------------------- #


class TestDensityMap:
    def test_transparent_yields_zero_density(self):
        lum = np.zeros((20, 20), dtype=np.float32)  # black -> would be dense
        alpha = np.zeros((20, 20), dtype=np.float32)  # but fully transparent
        field = build_density_map(lum, alpha, DensityParams())
        assert field.density.max() == pytest.approx(0.0)

    def test_dark_opaque_is_dense_by_default(self):
        lum = np.zeros((20, 20), dtype=np.float32)  # black subject
        alpha = np.ones((20, 20), dtype=np.float32)
        field = build_density_map(lum, alpha, DensityParams(edge_preservation=0, detail=0))
        assert field.density.mean() > 0.5

    def test_light_is_sparse_by_default(self):
        lum = np.ones((20, 20), dtype=np.float32)  # white background
        alpha = np.ones((20, 20), dtype=np.float32)
        field = build_density_map(lum, alpha, DensityParams(edge_preservation=0, detail=0))
        assert field.density.mean() < 0.05

    def test_invert_flips_polarity(self):
        lum = np.ones((20, 20), dtype=np.float32)  # white
        alpha = np.ones((20, 20), dtype=np.float32)
        field = build_density_map(
            lum, alpha, DensityParams(invert=True, edge_preservation=0, detail=0)
        )
        assert field.density.mean() > 0.5

    def test_higher_density_slider_increases_coverage(self):
        rng = np.random.default_rng(3)
        lum = rng.random((40, 40)).astype(np.float32)
        alpha = np.ones((40, 40), dtype=np.float32)
        low = build_density_map(lum, alpha, DensityParams(density=0.1)).density.mean()
        high = build_density_map(lum, alpha, DensityParams(density=0.9)).density.mean()
        assert high > low

    def test_output_range(self):
        rng = np.random.default_rng(4)
        lum = rng.random((30, 30)).astype(np.float32)
        alpha = rng.random((30, 30)).astype(np.float32)
        d = build_density_map(lum, alpha, DensityParams(detail=1, edge_preservation=1)).density
        assert d.min() >= 0.0 and d.max() <= 1.0
        assert d.dtype == np.float32

    def test_shape_mismatch_raises(self):
        with pytest.raises(ImageProcessingError):
            build_density_map(
                np.zeros((10, 10), np.float32), np.zeros((5, 5), np.float32), DensityParams()
            )


class TestProcessToDensity:
    def test_full_pipeline(self):
        arr = make_rgba(h=200, w=300, color=(0, 0, 0, 255))
        field = process_to_density(arr, DensityParams(), max_dim=100)
        assert max(field.shape) == 100
        assert field.density.dtype == np.float32
        assert 0.0 <= field.coverage() <= 1.0

    def test_high_resolution_downscaled(self):
        arr = make_rgba(h=3000, w=2400)
        field = process_to_density(arr, DensityParams(), max_dim=1400)
        assert max(field.shape) == 1400

    def test_bad_array(self):
        with pytest.raises(ImageProcessingError):
            process_to_density(np.zeros((10, 10), np.uint8), DensityParams())
