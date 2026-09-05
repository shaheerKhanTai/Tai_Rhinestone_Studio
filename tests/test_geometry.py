"""Tests for the geometry and unit-conversion layer."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.geometry import (
    EPSILON,
    MM_PER_INCH,
    DesignTransform,
    GeometryError,
    Point,
    Rect,
    Unit,
    center_to_center,
    clamp,
    convert,
    fit_dimensions,
    from_mm,
    stones_overlap,
    to_mm,
)


class TestUnits:
    def test_mm_per_unit(self):
        assert Unit.MM.mm_per_unit == 1.0
        assert Unit.CM.mm_per_unit == 10.0
        assert Unit.INCH.mm_per_unit == 25.4

    def test_to_and_from_mm_roundtrip(self):
        for unit in Unit:
            for value in (0.0, 1.0, 3.14159, 1000.0):
                mm = to_mm(value, unit)
                assert from_mm(mm, unit) == pytest.approx(value)

    def test_inch_to_mm(self):
        assert to_mm(10.0, Unit.INCH) == pytest.approx(254.0)
        assert to_mm(8.0, Unit.INCH) == pytest.approx(203.2)

    def test_convert_between_units(self):
        assert convert(1.0, Unit.INCH, Unit.MM) == pytest.approx(25.4)
        assert convert(25.4, Unit.MM, Unit.INCH) == pytest.approx(1.0)
        assert convert(1.0, Unit.CM, Unit.MM) == pytest.approx(10.0)
        assert convert(5.0, Unit.MM, Unit.MM) == 5.0

    def test_from_str(self):
        assert Unit.from_str("in") is Unit.INCH
        assert Unit.from_str("MM") is Unit.MM
        assert Unit.from_str("inch") is Unit.INCH
        with pytest.raises(GeometryError):
            Unit.from_str("furlong")


class TestPoint:
    def test_distance(self):
        a = Point(0, 0)
        b = Point(3, 4)
        assert a.distance_to(b) == pytest.approx(5.0)
        assert a.distance_sq_to(b) == pytest.approx(25.0)

    def test_translate(self):
        assert Point(1, 2).translated(3, -1) == Point(4, 1)

    def test_immutable(self):
        p = Point(1, 2)
        with pytest.raises(Exception):
            p.x = 5  # type: ignore[misc]


class TestRect:
    def test_derived_properties(self):
        r = Rect(10, 20, 30, 40)
        assert r.right == 40
        assert r.bottom == 60
        assert r.area == 1200
        assert r.center == Point(25, 40)

    def test_normalized_handles_negative(self):
        r = Rect(50, 50, -30, -20).normalized()
        assert (r.x, r.y, r.width, r.height) == (20, 30, 30, 20)

    def test_contains(self):
        r = Rect(0, 0, 10, 10)
        assert r.contains(Point(5, 5))
        assert r.contains(Point(0, 0))
        assert r.contains(Point(10, 10))
        assert not r.contains(Point(11, 5))

    def test_intersects(self):
        a = Rect(0, 0, 10, 10)
        assert a.intersects(Rect(5, 5, 10, 10))
        assert a.intersects(Rect(10, 10, 5, 5))  # touching edge
        assert not a.intersects(Rect(11, 0, 5, 5))

    def test_from_points(self):
        r = Rect.from_points(Point(30, 40), Point(10, 20))
        assert (r.x, r.y, r.width, r.height) == (10, 20, 20, 20)

    def test_inflated(self):
        r = Rect(10, 10, 20, 20).inflated(5)
        assert (r.x, r.y, r.width, r.height) == (5, 5, 30, 30)


class TestSpacing:
    def test_center_to_center(self):
        assert center_to_center(2.8, 0.5) == pytest.approx(3.3)
        assert center_to_center(3.0, 0.0) == pytest.approx(3.0)

    def test_center_to_center_invalid(self):
        with pytest.raises(GeometryError):
            center_to_center(0.0, 0.5)
        with pytest.raises(GeometryError):
            center_to_center(3.0, -0.1)

    def test_stones_overlap(self):
        # Same size 3mm stones, gap 0: overlap when centres < 3mm apart.
        assert stones_overlap(0, 0, 3.0, 2.0, 0, 3.0, gap_mm=0.0)
        assert not stones_overlap(0, 0, 3.0, 3.0, 0, 3.0, gap_mm=0.0)
        # With a gap the required distance grows.
        assert stones_overlap(0, 0, 3.0, 3.1, 0, 3.0, gap_mm=0.5)

    def test_stones_touching_not_overlapping(self):
        # Exactly touching (distance == required) must not count as overlap.
        assert not stones_overlap(0, 0, 3.0, 3.0, 0, 3.0, gap_mm=0.0)


class TestDesignTransform:
    def test_scale_2400x3000_to_10x8in(self):
        t = DesignTransform(2400, 3000, to_mm(10, Unit.INCH), to_mm(8, Unit.INCH))
        assert t.scale_x == pytest.approx(254.0 / 2400)
        assert t.scale_y == pytest.approx(203.2 / 3000)

    def test_corners_map_correctly(self):
        t = DesignTransform(2400, 3000, 254.0, 203.2)
        assert t.image_to_design(0, 0) == Point(0.0, 0.0)
        far = t.image_to_design(2400, 3000)
        assert far.x == pytest.approx(254.0)
        assert far.y == pytest.approx(203.2)

    def test_roundtrip_design_to_image(self):
        t = DesignTransform(1000, 500, 100.0, 50.0)
        px, py = t.design_to_image(50.0, 25.0)
        assert px == pytest.approx(500.0)
        assert py == pytest.approx(250.0)

    def test_array_transform(self):
        t = DesignTransform(100, 100, 50.0, 25.0)
        pts = np.array([[0, 0], [100, 100], [50, 50]], dtype=float)
        out = t.image_to_design_array(pts)
        assert out.shape == (3, 2)
        np.testing.assert_allclose(out[1], [50.0, 25.0])
        np.testing.assert_allclose(out[2], [25.0, 12.5])

    def test_array_transform_bad_shape(self):
        t = DesignTransform(100, 100, 50.0, 25.0)
        with pytest.raises(GeometryError):
            t.image_to_design_array(np.zeros((3, 3)))

    def test_zero_image_dimension_rejected(self):
        with pytest.raises(GeometryError):
            DesignTransform(0, 100, 50.0, 50.0)

    def test_negative_design_dimension_rejected(self):
        with pytest.raises(GeometryError):
            DesignTransform(100, 100, -5.0, 50.0)

    def test_design_rect(self):
        t = DesignTransform(100, 100, 50.0, 40.0)
        r = t.design_rect_mm
        assert (r.width, r.height) == (50.0, 40.0)


class TestFitDimensions:
    def test_lock_aspect_preserves_ratio(self):
        # Portrait image 1000x2000 → width 100mm → height 200mm.
        w, h = fit_dimensions(1000, 2000, 100.0, lock_aspect=True)
        assert w == pytest.approx(100.0)
        assert h == pytest.approx(200.0)

    def test_unlocked_returns_square_default(self):
        w, h = fit_dimensions(1000, 2000, 100.0, lock_aspect=False)
        assert (w, h) == (100.0, 100.0)

    def test_invalid_inputs(self):
        with pytest.raises(GeometryError):
            fit_dimensions(0, 100, 10.0)
        with pytest.raises(GeometryError):
            fit_dimensions(100, 100, 0.0)


class TestClamp:
    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(11, 0, 10) == 10

    def test_clamp_swapped_bounds(self):
        assert clamp(5, 10, 0) == 5


def test_constants():
    assert MM_PER_INCH == 25.4
    assert EPSILON < 1e-6
    assert math.isfinite(EPSILON)
