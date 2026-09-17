"""Tests for the trace-mask core: shapes, strokes, resampling, gating."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.trace_mask import (
    blank_mask,
    density_gate,
    extract_patch,
    mask_is_empty,
    mask_shape_for,
    paint_stroke,
    resample_mask,
    stroke_bbox,
)


class TestShapes:
    def test_long_side_capped(self):
        # Shape is (h, w); the longer physical side maps to 768 px.
        assert mask_shape_for(254.0, 203.2) == (614, 768)
        assert mask_shape_for(100.0, 300.0) == (768, 256)

    def test_blank_is_zeros(self):
        m = blank_mask(100.0, 100.0)
        assert m.dtype == np.uint8
        assert m.shape == mask_shape_for(100.0, 100.0)
        assert not m.any()

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            mask_shape_for(0, 100)


class TestPaintStroke:
    def test_dot_paints_disc(self):
        m = blank_mask(100.0, 100.0)
        assert paint_stroke(m, None, (50.0, 50.0), 10.0, 100.0, 100.0)
        h, w = m.shape
        assert m[h // 2, w // 2] == 255
        # Outside the radius stays clean.
        assert m[0, 0] == 0

    def test_stroke_paints_line(self):
        m = blank_mask(100.0, 100.0)
        assert paint_stroke(m, (10.0, 50.0), (90.0, 50.0), 3.0, 100.0, 100.0)
        h, w = m.shape
        # Samples along the painted horizontal line are set.
        assert m[h // 2, w // 10] == 255
        assert m[h // 2, w // 2] == 255
        assert m[h // 2, w * 9 // 10] == 255
        # Above the stroke stays clean (well past the radius).
        assert m[0, w // 2] == 0

    def test_mask_stays_binary(self):
        m = blank_mask(100.0, 100.0)
        paint_stroke(m, (5.0, 5.0), (95.0, 95.0), 6.0, 100.0, 100.0)
        assert set(np.unique(m)).issubset({0, 255})

    def test_erase_removes_paint(self):
        m = blank_mask(100.0, 100.0)
        paint_stroke(m, (50.0, 50.0), (50.0, 50.0), 10.0, 100.0, 100.0)
        assert paint_stroke(m, (50.0, 50.0), (50.0, 50.0), 10.0, 100.0, 100.0, erase=True)
        assert mask_is_empty(m)

    def test_clamped_outside_canvas(self):
        m = blank_mask(100.0, 100.0)
        paint_stroke(m, (-50.0, -50.0), (200.0, 200.0), 5.0, 100.0, 100.0)
        assert m.any()  # clamped part still paints

    def test_no_change_returns_false(self):
        m = blank_mask(100.0, 100.0)
        assert not paint_stroke(m, (50.0, 50.0), (50.0, 50.0), 10.0, 100.0, 100.0, erase=True)


class TestResample:
    def test_same_shape_identity(self):
        m = blank_mask(100.0, 100.0)
        assert resample_mask(m, 100.0, 100.0) is m

    def test_aspect_change_keeps_relative_position(self):
        m = blank_mask(100.0, 100.0)
        paint_stroke(m, (50.0, 25.0), (50.0, 25.0), 3.0, 100.0, 100.0)  # upper-middle
        out = resample_mask(m, 100.0, 50.0)
        assert out.shape == mask_shape_for(100.0, 50.0)
        h, w = out.shape
        assert out[h // 4, w // 2] == 255  # still upper-middle
        assert out[h * 3 // 4, w // 2] == 0
        assert set(np.unique(out)).issubset({0, 255})


class TestBboxPatches:
    def test_bbox_covers_exactly_the_change(self):
        m = blank_mask(100.0, 100.0)
        before = m.copy()
        paint_stroke(m, (50.0, 50.0), (50.0, 50.0), 5.0, 100.0, 100.0)
        bbox = stroke_bbox(m, before)
        assert bbox is not None
        x, y, bw, bh = bbox
        region = m[y : y + bh, x : x + bw]
        # Every changed pixel lies inside the bbox; the bbox is non-empty.
        diff = m != before
        assert diff.sum() == np.count_nonzero(diff[y : y + bh, x : x + bw])
        assert region.any()

    def test_no_change_no_bbox(self):
        m = blank_mask(100.0, 100.0)
        assert stroke_bbox(m, m.copy()) is None

    def test_patch_roundtrip_mirrors_undo(self):
        from app.core.trace_mask import apply_patch

        m = blank_mask(100.0, 100.0)
        paint_stroke(m, (20.0, 20.0), (80.0, 40.0), 6.0, 100.0, 100.0)  # earlier content
        before = m.copy()
        paint_stroke(m, (50.0, 60.0), (60.0, 80.0), 6.0, 100.0, 100.0)  # the stroke to undo
        bbox = stroke_bbox(m, before)
        assert bbox is not None
        before_patch = extract_patch(before, bbox)
        after_patch = extract_patch(m, bbox)
        assert (after_patch != before_patch).any()
        # Undo: restore the pre-stroke patch over the changed region.
        apply_patch(m, bbox, before_patch)
        np.testing.assert_array_equal(m, before)
        # Redo: put the post-stroke patch back.
        apply_patch(m, bbox, after_patch)
        assert (m != before).any()


class TestGate:
    def test_gate_shape_and_range(self):
        m = blank_mask(100.0, 100.0)
        paint_stroke(m, (50.0, 50.0), (50.0, 50.0), 10.0, 100.0, 100.0)
        g = density_gate(m, 200, 300)
        assert g.shape == (200, 300)
        assert g.dtype == np.float32
        assert 0.0 <= g.min() and g.max() <= 1.0

    def test_empty_checks(self):
        assert mask_is_empty(None)
        assert mask_is_empty(blank_mask(100.0, 100.0))
