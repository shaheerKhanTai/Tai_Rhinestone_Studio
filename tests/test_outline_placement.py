"""Tests for Outline + Fill placement — the professional (non-dithered) mode.

Shapes must render as: an even row of stones along every contour (outlines
AND holes), a hexagonal lattice filling the deep interior, and nothing
outside the shapes — regardless of PNG/JPG noise.
"""
from __future__ import annotations

import numpy as np
import pytest
import cv2

from app.core.collision import find_collisions
from app.core.geometry import DesignTransform
from app.core.image_processor import DensityParams, process_to_density
from app.core.placement import place_stones
from app.models.settings import GenerationSettings, PlacementMode

D = 2.8    # SS10
GAP = 0.5
PITCH = D + GAP


def white_canvas(h=300, w=300):
    return np.full((h, w, 4), 255, np.uint8)


def outline_settings(**kw) -> GenerationSettings:
    return GenerationSettings(placement_mode=PlacementMode.OUTLINE, **kw)


def run(rgba, w_mm=100.0, h_mm=100.0, **kw):
    field = process_to_density(rgba, DensityParams())
    s = outline_settings(**kw)
    t = DesignTransform(field.width, field.height, w_mm, h_mm)
    return place_stones(field.density, t, s), s


def disc_rgba(radius=100, size=300):
    rgba = white_canvas(size, size)
    cv2.circle(rgba, (size // 2, size // 2), radius, (25, 25, 35, 255), -1)
    return rgba


def solid_rgba(size=300, tone=25):
    """A uniformly dark, fully opaque image (a full-bleed subject)."""
    rgba = np.full((size, size, 4), tone, np.uint8)
    rgba[:, :, 3] = 255
    return rgba


def assert_no_overlaps(result, settings):
    dia = np.full(result.count, settings.effective_diameter())
    assert find_collisions(result.positions_mm, dia, settings.spacing_mm) == []


class TestOutlineRows:
    def test_disc_gets_even_ring_plus_fill(self):
        res, s = run(disc_rgba())
        pos = res.positions_mm
        assert res.count > 150
        r = np.hypot(pos[:, 0] - 50.0, pos[:, 1] - 50.0)
        # Ring stones sit on the contour (disc radius 33.3 mm, allow blur).
        ring = pos[(r > 31.5) & (r < 35.0)]
        assert ring.shape[0] >= 50  # a full row around the circle
        # Order by angle; consecutive spacing must be even and >= pitch.
        ang = np.arctan2(ring[:, 1] - 50.0, ring[:, 0] - 50.0)
        ordered = ring[np.argsort(ang)]
        gaps = np.hypot(*(np.roll(ordered, -1, axis=0) - ordered).T)
        assert gaps.min() >= PITCH - 1e-3          # never overlapping
        assert float(np.ptp(gaps)) < 0.5 * PITCH   # even, not scattered
        assert float(np.mean(gaps)) < 2.0 * PITCH
        # Interior is filled with lattice, not dithered speckle.
        assert int((r < 25.0).sum()) > 60

    def test_nothing_outside_shape(self):
        res, _ = run(disc_rgba())
        r = np.hypot(res.positions_mm[:, 0] - 50.0, res.positions_mm[:, 1] - 50.0)
        assert float(r.max()) < 33.3 + D / 2 + 0.5  # stone radius + slack only

    def test_thin_annulus_outline_only(self):
        # A ring 12 px (4 mm) wide — narrower than 2 pitches: outline rows,
        # no interior fill.
        rgba = disc_rgba(radius=100)
        cv2.circle(rgba, (150, 150), 88, (255, 255, 255, 255), -1)
        res, s = run(rgba)
        assert res.count > 40
        assert_no_overlaps(res, s)
        r = np.hypot(res.positions_mm[:, 0] - 50.0, res.positions_mm[:, 1] - 50.0)
        # No stones in the deep middle (the hole).
        assert int((r < 20.0).sum()) == 0

    def test_hole_contour_receives_stones(self):
        # An "O": both the outer boundary and the hole boundary get rows.
        rgba = white_canvas()
        cv2.circle(rgba, (150, 130), 80, (25, 25, 35, 255), -1)
        cv2.circle(rgba, (150, 130), 50, (255, 255, 255, 255), -1)
        res, s = run(rgba)
        cx, cy = 50.0, 130 / 3.0
        r = np.hypot(res.positions_mm[:, 0] - cx, res.positions_mm[:, 1] - cy)
        outer = int((r > 24.0).sum())   # outer boundary at 26.7 mm
        hole = int(((r > 13.5) & (r < 18.5)).sum())  # hole boundary at 16.7 mm
        assert outer >= 15
        assert hole >= 8
        assert_no_overlaps(res, s)

    def test_pinched_shape_still_collision_free(self):
        # Two discs joined by a neck narrower than 2 pitches: the greedy
        # safety filter must drop colliding stones, keeping validity.
        rgba = white_canvas()
        cv2.circle(rgba, (150, 90), 55, (25, 25, 35, 255), -1)
        cv2.circle(rgba, (150, 210), 55, (25, 25, 35, 255), -1)
        cv2.rectangle(rgba, (140, 90), (160, 210), (25, 25, 35, 255), -1)
        res, s = run(rgba)
        assert res.count > 50
        assert_no_overlaps(res, s)

    def test_empty_density_no_stones(self):
        res, _ = run(white_canvas())
        assert res.count == 0


class TestNoiseImmunity:
    def test_jpeg_artifacts_do_not_change_result(self):
        # The user's core complaint: PNG vs JPG precision. A speckled,
        # JPEG-degraded copy must produce the same clean conversion.
        clean = disc_rgba(radius=90)
        rng = np.random.default_rng(3)
        gray = np.clip(240 + rng.normal(0, 15, (300, 300)), 0, 255).astype(np.uint8)
        noisy = np.ascontiguousarray(
            np.stack([gray] * 3 + [np.full((300, 300), 255, np.uint8)], -1)
        )
        cv2.circle(noisy, (150, 150), 90, (25, 25, 35, 255), -1)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(noisy[:, :, :3], cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 60])
        dec = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        jpg = np.ascontiguousarray(
            np.concatenate([dec, np.full((300, 300, 1), 255, np.uint8)], -1)
        )
        res_clean, _ = run(clean)
        res_jpg, _ = run(jpg)
        # No background stones from either, and the counts match closely
        # (the JPEG copy's noise must not add or remove shape stones).
        r_c = np.hypot(res_clean.positions_mm[:, 0] - 50, res_clean.positions_mm[:, 1] - 50)
        r_j = np.hypot(res_jpg.positions_mm[:, 0] - 50, res_jpg.positions_mm[:, 1] - 50)
        assert float(r_c.max()) < 30.0 + D / 2 + 0.5
        assert float(r_j.max()) < 30.0 + D / 2 + 0.5
        assert abs(res_clean.count - res_jpg.count) <= max(6, 0.05 * res_clean.count)

    def test_speckles_smaller_than_a_stone_are_dropped(self):
        # Dark dots far smaller than a stone: the stone-scale open removes them.
        rgba = white_canvas(200, 200)
        rng = np.random.default_rng(11)
        for _ in range(40):
            x, y = rng.integers(10, 190, 2)
            cv2.circle(rgba, (int(x), int(y)), 2, (25, 25, 35, 255), -1)
        res, _ = run(rgba, w_mm=100.0, h_mm=100.0)
        assert res.count == 0


class TestUniformField:
    """A uniformly tonal field has no background for Otsu to split against.

    Otsu then puts its threshold at the very top of the range and binarises to
    a speckle, which the stone-scale open erases — so a solid dark design used
    to generate *zero* stones, and only at some analysis resolutions, which
    made the failure look random. A full-bleed subject must fill the design.
    """

    @pytest.mark.parametrize(
        "size,max_dim",
        [(150, 400), (200, 600), (300, 300), (300, 600), (600, 1400)],
    )
    def test_solid_dark_image_fills_the_design(self, size, max_dim):
        rgba = np.full((size, size, 4), 25, np.uint8)
        rgba[:, :, 3] = 255
        field = process_to_density(rgba, DensityParams(), max_dim=max_dim)
        s = outline_settings()
        t = DesignTransform(field.width, field.height, 100.0, 100.0)
        res = place_stones(field.density, t, s)
        assert res.count > 300, f"{size}px @ max_dim {max_dim} gave {res.count} stones"
        assert_no_overlaps(res, s)

    def test_result_is_resolution_independent(self):
        # The same artwork at different analysis resolutions must give the
        # same design — that inconsistency was the visible symptom.
        counts = set()
        for size, max_dim in ((150, 400), (300, 600), (600, 1400)):
            rgba = np.full((size, size, 4), 25, np.uint8)
            rgba[:, :, 3] = 255
            field = process_to_density(rgba, DensityParams(), max_dim=max_dim)
            s = outline_settings()
            t = DesignTransform(field.width, field.height, 100.0, 100.0)
            counts.add(place_stones(field.density, t, s).count)
        assert len(counts) == 1, f"counts differed by resolution: {sorted(counts)}"

    def test_solid_image_reaches_the_edges(self):
        # Not merely "some stones": a solid design is a solid fill, up to the
        # half-pitch containment margin.
        res, _ = run(solid_rgba(300))
        pos = res.positions_mm
        assert float(pos[:, 0].min()) < 3.0 and float(pos[:, 0].max()) > 97.0
        assert float(pos[:, 1].min()) < 3.0 and float(pos[:, 1].max()) > 97.0

    def test_uniform_white_field_stays_empty(self):
        # The fallback must not turn a blank canvas into a full-bleed shape.
        res, _ = run(white_canvas(300, 300))
        assert res.count == 0

    @pytest.mark.parametrize("rgb", [225, 235, 250])
    def test_uniform_near_white_field_stays_empty(self, rgb):
        # Sub-tone-floor fields are background haze, not a subject.
        rgba = np.full((300, 300, 4), rgb, np.uint8)
        rgba[:, :, 3] = 255
        res, _ = run(rgba)
        assert res.count == 0


class TestProperties:
    def test_deterministic(self):
        rgba = disc_rgba(radius=80)
        a, _ = run(rgba)
        b, _ = run(rgba)
        np.testing.assert_array_equal(a.positions_mm, b.positions_mm)

    def test_max_stones_prefers_outline(self):
        # Cap below the total: the kept stones must include the outline row
        # (density 1.0) rather than an arbitrary subset.
        res, s = run(disc_rgba(radius=100), max_stones=80)
        assert res.count <= 80
        assert res.was_limited
        r = np.hypot(res.positions_mm[:, 0] - 50.0, res.positions_mm[:, 1] - 50.0)
        assert int((r > 31.5).sum()) >= 50  # the ring survives the cap

    def test_trace_gate_respected(self):
        from app.core.trace_mask import blank_mask, paint_stroke

        mask = blank_mask(100.0, 100.0)
        paint_stroke(mask, (20.0, 50.0), (20.0, 50.0), 20.0, 100.0, 100.0)  # left band
        rgba = disc_rgba()
        field = process_to_density(rgba, DensityParams())
        s = outline_settings()
        t = DesignTransform(field.width, field.height, 100.0, 100.0)
        res = place_stones(field.density, t, s, gate=mask.astype(np.float32) / 255.0)
        pos = res.positions_mm
        assert res.count > 10
        # Every stone centre maps onto a painted mask pixel (strict).
        mh, mw = mask.shape
        ix = np.clip(np.round(pos[:, 0] / 100.0 * (mw - 1)).astype(int), 0, mw - 1)
        iy = np.clip(np.round(pos[:, 1] / 100.0 * (mh - 1)).astype(int), 0, mh - 1)
        assert bool(np.all(mask[iy, ix] == 255))
        assert_no_overlaps(res, s)

    def test_automatic_is_the_default_mode(self):
        # The engine judges the artwork itself, so the user is not required to
        # know whether their image is a logo or a photograph.
        assert GenerationSettings().placement_mode is PlacementMode.AUTOMATIC
        assert PlacementMode.AUTOMATIC.label == "Automatic (Smart)"

    def test_outline_is_still_selectable(self):
        assert PlacementMode.from_str("outline") is PlacementMode.OUTLINE
        assert PlacementMode.OUTLINE.label == "Outline + Fill"
