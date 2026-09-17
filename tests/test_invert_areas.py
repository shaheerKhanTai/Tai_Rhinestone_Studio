"""Tests for the "Invert Rhinestone Areas" option (FEATURE 2).

The option must swap the *occupied* and *empty* regions of the pattern: the
parts of the design that carried stones become empty and the empty parts take
them. It is a complement of the region the image fills — not a translation of
the stones and not a geometric flip of the image.

Two families of property are checked:

* **The inversion is a real complement** — the pattern moves to the other side
  of the shape's outline, and it cannot be reproduced by shifting the original
  pattern.
* **Nothing physical changes** — canvas dimensions, margins, stone size and the
  minimum spacing are exactly those of the un-inverted run, and the inverted
  pattern still respects transparency, the trace gate and the artboard edge.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.collision import find_collisions
from app.core.rhinestone_engine import generate
from app.models.settings import GenerationSettings, PlacementMode

D = 2.8    # SS10
GAP = 0.5
PITCH = D + GAP
RADIUS = D / 2.0


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def white_canvas(h: int = 300, w: int = 300) -> np.ndarray:
    """A fully opaque white RGBA image (tone 0 → no stones by default)."""
    return np.full((h, w, 4), 255, np.uint8)


def dark_canvas(h: int = 300, w: int = 300) -> np.ndarray:
    """A fully opaque near-black RGBA image (tone 1 → fully stoned)."""
    rgba = np.full((h, w, 4), 25, np.uint8)
    rgba[:, :, 3] = 255
    return rgba


def disc_rgba(radius: int = 100, size: int = 300, invert_bg: bool = False) -> np.ndarray:
    """A dark disc on a white canvas (the canonical 'subject' artwork)."""
    rgba = white_canvas(size, size) if not invert_bg else dark_canvas(size, size)
    cv2.circle(rgba, (size // 2, size // 2), radius, (25, 25, 35, 255), -1)
    return rgba


def bar_rgba(x0: int = 130, x1: int = 170, size: int = 300) -> np.ndarray:
    """An off-centre-free vertical dark bar (asymmetric structure)."""
    rgba = white_canvas(size, size)
    rgba[:, x0:x1, :3] = 25
    return rgba


def run(rgba, w_mm: float = 100.0, h_mm: float = 100.0, **kw):
    """Generate with SS10 / 0.5 mm gap plus the caller's overrides."""
    kw.setdefault("placement_mode", PlacementMode.HEXAGONAL)
    settings = GenerationSettings(stone_size_id="SS10", spacing_mm=GAP, **kw)
    return generate(rgba, settings, w_mm, h_mm, analysis_max_dim=600), settings


def disc_fraction(result, cx: float = 50.0, cy: float = 50.0, r_mm: float = 33.3) -> float:
    """Fraction of stones whose centre lies inside the disc of radius ``r_mm``."""
    pos = result.placement.positions_mm
    if pos.shape[0] == 0:
        return 0.0
    return float(np.mean(np.hypot(pos[:, 0] - cx, pos[:, 1] - cy) <= r_mm))


def assert_inside_canvas(result, settings, w_mm: float, h_mm: float) -> None:
    """No stone may be clipped by the artboard edge (centre ≥ one radius in)."""
    pos = result.placement.positions_mm
    if pos.shape[0] == 0:
        return
    assert float(pos[:, 0].min()) >= RADIUS - 1e-6
    assert float(pos[:, 1].min()) >= RADIUS - 1e-6
    assert float(pos[:, 0].max()) <= w_mm - RADIUS + 1e-6
    assert float(pos[:, 1].max()) <= h_mm - RADIUS + 1e-6


def assert_no_overlaps(result, settings) -> None:
    dia = np.full(result.stone_count, settings.effective_diameter())
    assert find_collisions(result.placement.positions_mm, dia, settings.spacing_mm) == []


# --------------------------------------------------------------------------- #
# The complement itself
# --------------------------------------------------------------------------- #


class TestComplement:
    def test_swaps_occupied_and_empty(self):
        rgba = disc_rgba(radius=100)
        normal, _ = run(rgba)
        inverted, _ = run(rgba, invert_areas=True)

        # Normal: the disc (33.3 mm radius on a 100 mm design) carries the stones.
        assert disc_fraction(normal) > 0.8
        # Inverted: the disc is empty and the surround carries them.
        assert inverted.stone_count > 0
        assert disc_fraction(inverted) < 0.15

    def test_cannot_be_reproduced_by_shifting_the_pattern(self):
        # A dark bar down the middle: the normal pattern is a single column.
        # A translation of that column can only ever move it — the inverted
        # pattern must appear on *both* sides of the bar, which no rigid shift
        # of a single column can do.
        rgba = bar_rgba(130, 170)   # 43.3 mm .. 56.7 mm on a 100 mm design
        normal, _ = run(rgba)
        inverted, _ = run(rgba, invert_areas=True)

        left_edge, right_edge = 43.3, 56.7
        n = normal.placement.positions_mm
        # The bar carries essentially the whole normal design (the silhouette
        # detail-keep lets the boundary columns gain by up to half a stone).
        assert float(np.mean((n[:, 0] > left_edge) & (n[:, 0] < right_edge))) > 0.7

        p = inverted.placement.positions_mm
        left = int(np.sum(p[:, 0] < left_edge - PITCH))
        right = int(np.sum(p[:, 0] > right_edge + PITCH))
        in_bar = int(np.sum((p[:, 0] > left_edge + PITCH) & (p[:, 0] < right_edge - PITCH)))
        assert left > 20 and right > 20      # stones on both sides of the bar
        assert in_bar == 0                   # ... and none in its deep interior

    def test_default_is_off(self):
        assert GenerationSettings().invert_areas is False

    def test_dark_subject_inverts_to_empty(self):
        # A near-black image is almost fully stoned; inverted, it is genuinely
        # empty — the dead-zone absorbs the residual tone of a subject that is
        # dark but not pure black (artwork rarely is).
        rgba = dark_canvas(200, 200)
        normal, _ = run(rgba)
        inverted, _ = run(rgba, invert_areas=True)
        assert normal.stone_count > 400
        assert inverted.stone_count == 0

    def test_solid_subject_interior_is_empty_not_stippled(self):
        # The regression the dead-zone exists for: without it, the complement
        # of a solid dark disc (RGB 25 — density 0.94) stipples specks across
        # the disc's own interior instead of clearing it.
        rgba = disc_rgba(radius=100)
        inverted, _ = run(rgba, invert_areas=True)
        pos = inverted.placement.positions_mm
        dist = np.hypot(pos[:, 0] - 50.0, pos[:, 1] - 50.0)
        # Deep inside the disc (radius 33.3 mm): nothing at all.
        assert int(np.sum(dist < 30.0)) == 0

    def test_white_background_inverts_to_full_coverage(self):
        # The empty background carries no stones by default; inverted, it is
        # the whole design and fills up.
        rgba = white_canvas(200, 200)
        normal, _ = run(rgba)
        inverted, _ = run(rgba, invert_areas=True)
        assert normal.stone_count == 0
        assert inverted.stone_count > 400

    def test_gradient_inverts_monotonically(self):
        # A left-dark → right-light gradient: coverage falls left→right, and
        # must rise left→right once inverted (mirrored smooth transitions).
        x = np.linspace(0.0, 1.0, 240)[None, :]
        gray = np.repeat(x, 240, axis=0)
        rgba = np.concatenate(
            [np.stack([(gray * 255)] * 3, axis=-1), np.full((240, 240, 1), 255.0)], axis=-1
        ).astype(np.uint8)

        def fifth_counts(result):
            pos = result.placement.positions_mm
            return [int(np.sum((pos[:, 0] >= 20 * k) & (pos[:, 0] < 20 * (k + 1))))
                    for k in range(5)]

        normal, _ = run(rgba)
        inverted, _ = run(rgba, invert_areas=True)
        n, i = fifth_counts(normal), fifth_counts(inverted)
        # Original: density decays towards the light right-hand side.
        assert n[0] > n[-1]
        # Inverted: the ordering is reversed.
        assert i[-1] > i[0]
        assert sum(i) > 0


# --------------------------------------------------------------------------- #
# Physical invariants — everything except the region must be untouched
# --------------------------------------------------------------------------- #


class TestPhysicalInvariants:
    def test_canvas_dimensions_preserved(self):
        normal, _ = run(disc_rgba())
        inverted, _ = run(disc_rgba(), invert_areas=True)
        assert inverted.transform.design_width_mm == pytest.approx(
            normal.transform.design_width_mm
        )
        assert inverted.transform.design_height_mm == pytest.approx(
            normal.transform.design_height_mm
        )
        # ... and the density grid keeps the image's aspect (no rescaling).
        assert inverted.density_field.shape == normal.density_field.shape

    def test_stones_stay_inside_the_canvas(self):
        res, settings = run(disc_rgba(), invert_areas=True)
        assert_inside_canvas(res, settings, 100.0, 100.0)

    def test_stone_size_preserved(self):
        res, settings = run(disc_rgba(), invert_areas=True)
        assert res.stone_count > 0
        assert np.allclose(
            [s.diameter for s in res.stones], settings.effective_diameter()
        )
        assert {s.size_id for s in res.stones} == {"SS10"}

    def test_minimum_spacing_preserved(self):
        res, settings = run(disc_rgba(), invert_areas=True)
        assert_no_overlaps(res, settings)

    @pytest.mark.parametrize(
        "mode",
        [PlacementMode.OUTLINE, PlacementMode.GRID, PlacementMode.OFFSET,
         PlacementMode.HEXAGONAL, PlacementMode.ORGANIC],
    )
    def test_every_mode_inverts_without_overlaps_or_clipping(self, mode):
        res, settings = run(disc_rgba(), placement_mode=mode, invert_areas=True)
        assert res.stone_count > 0
        assert_no_overlaps(res, settings)
        assert_inside_canvas(res, settings, 100.0, 100.0)

    def test_non_square_design_keeps_its_aspect(self):
        # A 2:1 artboard: the inverted design must stay 2:1 with stones inside.
        rgba = disc_rgba(size=300)
        res, settings = run(rgba, w_mm=160.0, h_mm=80.0, invert_areas=True)
        assert res.stone_count > 0
        assert_inside_canvas(res, settings, 160.0, 80.0)
        assert_no_overlaps(res, settings)


# --------------------------------------------------------------------------- #
# Transparency, tracing, determinism
# --------------------------------------------------------------------------- #


class TestHardMasks:
    def test_fully_transparent_image_stays_empty(self):
        rgba = disc_rgba()
        rgba[:, :, 3] = 0
        res, _ = run(rgba, invert_areas=True)
        assert res.stone_count == 0

    def test_transparent_region_is_not_filled_by_inversion(self):
        # Left half transparent, right half opaque white: the inversion fills
        # only the valid (opaque) side — transparency is not a design area.
        rgba = white_canvas(200, 200)
        rgba[:, :100, 3] = 0
        res, settings = run(rgba, invert_areas=True)
        assert res.stone_count > 0
        pos = res.placement.positions_mm
        assert float(pos[:, 0].min()) > 50.0 - PITCH    # nothing over the void
        assert float(np.mean(pos[:, 0] > 50.0)) > 0.9

    def test_inversion_is_confined_to_the_traced_region(self):
        from app.core.trace_mask import blank_mask, paint_stroke

        # A 12 mm-radius painted blob centred at (16.6, 50) mm: the band spans
        # x 4.6..28.6, y 38..62. The artwork's disc starts at x = 16.7 mm.
        mask = blank_mask(100.0, 100.0)
        paint_stroke(mask, (16.6, 50.0), (16.6, 50.0), 12.0, 100.0, 100.0)

        rgba = disc_rgba()
        settings = GenerationSettings(
            stone_size_id="SS10", spacing_mm=GAP,
            placement_mode=PlacementMode.HEXAGONAL, invert_areas=True,
        )
        res = generate(rgba, settings, 100.0, 100.0, analysis_max_dim=600, trace_mask=mask)
        assert res.stone_count > 0
        assert_no_overlaps(res, settings)

        pos = res.placement.positions_mm
        # Confined to the traced blob (y within the band, x inside its left
        # edge) — and inverted, so the stones stop at the disc's boundary at
        # x = 16.7 mm instead of running on to the band's right edge at 28.6.
        assert float(pos[:, 1].min()) >= 37.0
        assert float(pos[:, 1].max()) <= 63.0
        assert float(pos[:, 0].min()) >= 4.0
        assert float(pos[:, 0].max()) < 20.0


class TestDeterminism:
    def test_same_input_same_output(self):
        rgba = disc_rgba(radius=80)
        a, _ = run(rgba, invert_areas=True)
        b, _ = run(rgba, invert_areas=True)
        assert a.stone_count == b.stone_count
        np.testing.assert_array_equal(a.placement.positions_mm, b.placement.positions_mm)

    def test_inverted_is_stable_across_resolutions(self):
        # The same artwork at two resolutions must invert to a similar count
        # (the complement is resolution-independent geometry).
        small, _ = run(disc_rgba(radius=60, size=200), invert_areas=True)
        big, _ = run(disc_rgba(radius=120, size=400), invert_areas=True)
        assert abs(small.stone_count - big.stone_count) <= max(15, 0.05 * small.stone_count)


class TestSettingsPersistence:
    def test_round_trips_through_dict(self):
        s = GenerationSettings(invert_tone=True, invert_areas=True)
        again = GenerationSettings.from_dict(s.to_dict())
        assert again.invert_areas is True and again.invert_tone is True
        assert GenerationSettings.from_dict({}).invert_areas is False

    def test_copy_preserves_both_flags(self):
        s = GenerationSettings(invert_tone=True, invert_areas=True)
        c = s.copy()
        assert c.invert_tone is True and c.invert_areas is True

    def test_tone_polarity_still_works_independently(self):
        # invert_tone and invert_areas are independent axes: light-on-dark
        # *and* complement compose without interfering.
        rgba = white_canvas(200, 200)
        rgba[:100, :, :3] = 25  # dark top band on white
        res, _ = run(rgba, invert_tone=True, invert_areas=True)
        pos = res.placement.positions_mm
        assert res.stone_count > 0
        # invert_tone puts the subject on the light band, invert_areas then
        # complements it: the stones end up in the dark band.
        assert float(np.mean(pos[:, 1] < 50.0)) > 0.8
