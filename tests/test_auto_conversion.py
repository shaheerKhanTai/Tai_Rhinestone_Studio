"""Tests for automatic image → rhinestone conversion (ISSUE 4).

The conversion must behave like a halftone, not a random scatter: shading
controls stone *density*, but the geometry always respects the configured
minimum centre distance (diameter + gap) in physical millimetres. These tests
drive the full pipeline (image array → density → placement) with synthetic
images whose tonal structure is known exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.collision import find_collisions
from app.core.rhinestone_engine import generate
from app.models.settings import GenerationSettings

D = 2.8    # SS10
GAP = 0.5
PITCH = D + GAP


def rgba_from_gray(gray01: np.ndarray, opaque: bool = True) -> np.ndarray:
    """Build an (H, W, 4) uint8 RGBA image from a float grayscale in [0, 1]."""
    g = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    alpha = np.full(g.shape[:2] + (1,), 255 if opaque else 0, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def default_settings(**kw) -> GenerationSettings:
    from app.models.settings import PlacementMode

    base = dict(stone_size_id="SS10", spacing_mm=GAP, placement_mode="hexagonal")
    base.update(kw)
    if isinstance(base.get("placement_mode"), str):
        base["placement_mode"] = PlacementMode.from_str(base["placement_mode"])
    return GenerationSettings(**base)


def generate_from_gray(gray01, settings=None, w_mm=100.0, h_mm=100.0, analysis_max_dim=600,
                       invert=False, **kw):
    settings = settings or default_settings(**kw)
    return generate(rgba_from_gray(gray01), settings, w_mm, h_mm,
                    invert=invert, analysis_max_dim=analysis_max_dim)


class TestTonalResponse:
    def test_dark_image_dense(self):
        # Near-black image → most lattice cells filled.
        res = generate_from_gray(np.full((200, 200), 0.02))
        assert res.stone_count > 400

    def test_light_image_sparse(self):
        # Near-white image → almost no stones (dark-subject polarity).
        res = generate_from_gray(np.full((200, 200), 0.98))
        assert res.stone_count < 20

    def test_dark_denser_than_light(self):
        dark = generate_from_gray(np.full((200, 200), 0.02)).stone_count
        light = generate_from_gray(np.full((200, 200), 0.98)).stone_count
        assert dark > light * 5

    def test_invert_flips_polarity(self):
        # A light subject on a dark background: with invert the light region
        # receives the stones instead of the dark one.
        gray = np.full((200, 200), 0.1)
        gray[:100, :] = 0.95  # light band on top
        normal = generate_from_gray(gray)
        inverted = generate_from_gray(gray, invert=True)
        pos_n = normal.placement.positions_mm
        pos_i = inverted.placement.positions_mm
        top_n = float(np.mean(pos_n[:, 1] < 50.0)) if pos_n.size else 0.0
        top_i = float(np.mean(pos_i[:, 1] < 50.0)) if pos_i.size else 0.0
        # Normal: stones concentrate in the dark (bottom) band.
        assert top_n < 0.2
        # Inverted: stones concentrate in the light (top) band.
        assert top_i > 0.8

    def test_gradient_monotonic_density(self):
        # A horizontal black→white gradient: stone coverage must decrease
        # monotonically (within tolerance) from the dark left to the light right.
        x = np.linspace(0.0, 1.0, 240)[None, :]
        gray = np.repeat(x, 240, axis=0)
        res = generate_from_gray(gray)
        pos = res.placement.positions_mm
        assert pos.shape[0] > 100
        # Coverage per design-fifth: counts must be non-increasing.
        bins = [int(np.sum((pos[:, 0] >= 20 * k) & (pos[:, 0] < 20 * (k + 1))))
                for k in range(5)]
        assert bins[0] > bins[-1]
        for k in range(4):
            assert bins[k] >= bins[k + 1] - 2  # allow tiny dither noise

    def test_shaded_regions_three_tones(self):
        # Dark / medium / light vertical bands: stone counts must order
        # dark > medium > light — shading controls density continuously.
        gray = np.full((300, 300), 0.8)
        gray[:, :100] = 0.1    # dark band
        gray[:, 100:200] = 0.45  # medium band
        res = generate_from_gray(gray)
        pos = res.placement.positions_mm
        dark = int(np.sum((pos[:, 0] < 33.3)))
        medium = int(np.sum((pos[:, 0] >= 33.3) & (pos[:, 0] < 66.6)))
        light = int(np.sum((pos[:, 0] >= 66.6)))
        assert dark > medium > light
        # The light band is not empty-thresholded to zero (continuous density):
        # a 0.2-darkness band still receives some stones at default density.
        assert light > 0


class TestStructurePreservation:
    def test_edge_heavy_image_stones_on_edges(self):
        # A white canvas with a black square outline: with edge preservation
        # the stones must concentrate around the outline, not fill the middle.
        gray = np.full((300, 300), 0.95)
        gray[50:250, 50:60] = 0.05     # left edge of the square
        gray[50:250, 240:250] = 0.05   # right edge
        gray[50:60, 50:250] = 0.05     # top edge
        gray[240:250, 50:250] = 0.05   # bottom edge
        res = generate_from_gray(gray, edge_preservation=0.8)
        pos = res.placement.positions_mm
        assert pos.shape[0] > 50
        # Distance of each stone to the nearest square edge line (design mm:
        # the square spans x,y in [16.7, 83.3] with ~3.3mm-thick strokes).
        def edge_dist(px, py):
            # distance to the nearest of the 4 stroke bands' centre lines
            xs = [20.0, 80.0]
            ys = [20.0, 80.0]
            dx = min(abs(px - x) for x in xs)
            dy = min(abs(py - y) for y in ys)
            return min(dx, dy)
        dists = np.array([edge_dist(p[0], p[1]) for p in pos])
        # Most stones lie within one pitch of an edge line.
        assert float(np.mean(dists < PITCH * 1.5)) > 0.6

    def test_face_features_survive(self):
        # A crude "face": two dark eyes, a mouth line and an outline on a
        # light background — the features must receive stones.
        gray = np.full((400, 400), 0.9)
        gray[120:170, 120:180] = 0.05      # left eye
        gray[120:170, 220:280] = 0.05      # right eye
        gray[280:300, 150:250] = 0.05      # mouth
        gray[60:340, 60:340] = np.minimum(gray[60:340, 60:340], 0.4)  # outline-ish
        res = generate_from_gray(gray, edge_preservation=0.6)
        pos = res.placement.positions_mm
        # Scale pixel coords to design mm (400px -> 100mm).
        mm = pos * (400.0 / 100.0)
        in_eye_l = int(np.sum((mm[:, 0] >= 120) & (mm[:, 0] <= 180) &
                              (mm[:, 1] >= 120) & (mm[:, 1] <= 170)))
        in_eye_r = int(np.sum((mm[:, 0] >= 220) & (mm[:, 0] <= 280) &
                              (mm[:, 1] >= 120) & (mm[:, 1] <= 170)))
        in_mouth = int(np.sum((mm[:, 0] >= 150) & (mm[:, 0] <= 250) &
                              (mm[:, 1] >= 280) & (mm[:, 1] <= 300)))
        assert in_eye_l > 0 and in_eye_r > 0 and in_mouth > 0


class TestBackgroundSuppression:
    """The design, not the whole canvas, must receive the stones."""

    def test_noisy_offwhite_background_receives_no_stones(self):
        # A dark subject on a noisy off-white background (scan haze / JPEG
        # noise). The edge/texture layers are peak-normalised, so an additive
        # density combination once covered the background in speckle stones;
        # tone gating plus the tone floor must keep it essentially empty.
        rng = np.random.default_rng(7)
        bg = np.clip(235 + rng.normal(0, 12, (300, 300)), 0, 255).astype(np.uint8)
        gray = np.stack([bg] * 3, axis=-1)
        rgba = np.concatenate(
            [gray, np.full((300, 300, 1), 255, np.uint8)], axis=-1
        )
        rgba[80:220, 80:220, :3] = 25  # dark square subject
        res = generate(rgba, default_settings(), 100.0, 100.0, analysis_max_dim=600)
        pos = res.placement.positions_mm
        assert res.stone_count > 100  # the subject itself is well covered
        # A one-pitch margin around the subject box (26.7..73.3 mm).
        margin = PITCH
        outside = int(np.sum((pos[:, 0] < 80 / 3 - margin) | (pos[:, 0] > 220 / 3 + margin) |
                             (pos[:, 1] < 80 / 3 - margin) | (pos[:, 1] > 220 / 3 + margin)))
        assert outside <= 3  # background stays empty (allow a straggler)

    def test_edges_and_texture_cannot_create_background_stones(self):
        # With maximal edge preservation and detail, a pure background strip
        # (zero tone) must still receive zero stones: the reinforcement
        # multiplies tone, it never adds to it.
        gray = np.full((200, 200), 1.0)  # pure white everywhere
        rgba = rgba_from_gray(gray)
        res = generate(
            rgba, default_settings(edge_preservation=1.0, detail=1.0),
            100.0, 100.0, analysis_max_dim=400,
        )
        assert res.stone_count == 0


class TestThinDetailCapture:
    def test_thin_line_survives_placement(self):
        # A stroke thinner than the stone pitch must still render as a
        # contiguous run of stones, not vanish between lattice rows.
        gray = np.full((300, 300), 0.98)
        gray[148:152, :] = 0.02  # ~1.3 mm stroke on a 100 mm design
        res = generate_from_gray(gray)
        pos = res.placement.positions_mm
        assert res.stone_count > 20
        # The line spans the full width: stones must too.
        assert pos[:, 0].min() < 10.0
        assert pos[:, 0].max() > 90.0
        # ... and they hug the line's y position (within a stone radius).
        assert float(np.mean(np.abs(pos[:, 1] - 50.0) < PITCH)) > 0.95


class TestSpacing:
    def test_no_overlaps_dark_image(self):
        # Densest case: a near-black image must still respect the pitch.
        res = generate_from_gray(np.full((240, 240), 0.02))
        dia = np.full(res.stone_count, D)
        pairs = find_collisions(res.placement.positions_mm, dia, GAP)
        assert pairs == []

    def test_no_overlaps_gradient(self):
        x = np.linspace(0.0, 1.0, 240)[None, :]
        gray = np.repeat(x, 240, axis=0)
        res = generate_from_gray(gray)
        dia = np.full(res.stone_count, D)
        assert find_collisions(res.placement.positions_mm, dia, GAP) == []

    @pytest.mark.parametrize("mode", ["grid", "offset", "hexagonal", "organic"])
    def test_no_overlaps_all_modes_shaded(self, mode):
        gray = np.full((200, 200), 0.7)
        gray[:100, :] = 0.15
        res = generate_from_gray(gray, placement_mode=mode)
        dia = np.full(res.stone_count, D)
        assert find_collisions(res.placement.positions_mm, dia, GAP) == []

    def test_uniform_area_regular_spacing(self):
        # A solid dark area must produce a *regular* arrangement (lattice),
        # not a random scatter: neighbouring distances cluster at the pitch.
        res = generate_from_gray(np.full((200, 200), 0.02), placement_mode="hexagonal")
        pos = res.placement.positions_mm
        # For each sampled stone, the distance to its nearest neighbour.
        def nn_dist(i):
            d = pos[i] - pos
            dist = np.hypot(d[:, 0], d[:, 1])
            dist[i] = np.inf
            return float(dist.min())

        n = pos.shape[0]
        sample = np.random.default_rng(42).choice(n, size=min(200, n), replace=False)
        nn = np.array([nn_dist(int(i)) for i in sample])
        # Nearly all sampled stones have a neighbour within 1.2x the pitch
        # (a full hex fill does), and none closer than the pitch.
        assert float(np.mean(nn <= PITCH * 1.2)) > 0.9
        assert nn.min() >= PITCH - 1e-4


class TestPhysicalCoordinates:
    def test_high_res_image_maps_to_physical_design(self):
        # 3000x4000 px image on a 10x12 inch (254 x 304.8 mm) design: stones
        # must live in design millimetres, not pixels.
        x = np.linspace(0.0, 1.0, 3000)[None, :]
        gray = np.repeat(x, 4000, axis=0)  # 4000 x 3000
        w_mm, h_mm = 254.0, 304.8
        res = generate_from_gray(gray, w_mm=w_mm, h_mm=h_mm, analysis_max_dim=600)
        pos = res.placement.positions_mm
        assert pos.shape[0] > 100
        assert pos[:, 0].max() <= w_mm + 1e-6
        assert pos[:, 1].max() <= h_mm + 1e-6
        # A pitch-scale structure in mm (not a pixel-scale scatter).
        spread = float(np.median(np.diff(np.unique(np.round(pos[:, 0], 2)))))
        assert 0.5 < spread < 50.0

    def test_different_resolution_same_design(self):
        # The same artwork at 2x resolution must produce a similar stone
        # count on the same physical design (resolution independence).
        small = rgba_from_gray(np.full((150, 150), 0.1))
        big = rgba_from_gray(np.full((300, 300), 0.1))
        s = default_settings()
        a = generate(small, s, 100.0, 100.0, analysis_max_dim=600)
        b = generate(big, s, 100.0, 100.0, analysis_max_dim=600)
        # Same physical size and same settings → nearly identical counts
        # (small deviation from rasterisation differences is fine).
        assert abs(a.stone_count - b.stone_count) <= max(10, 0.05 * a.stone_count)


class TestStoneCountAndDeterminism:
    def test_max_stones_respected(self):
        gray = np.full((300, 300), 0.02)  # dense → thousands of candidates
        res = generate_from_gray(gray, max_stones=500)
        assert res.stone_count <= 500
        assert res.was_limited
        assert res.total_before_limit > 500

    def test_deterministic_same_input_same_output(self):
        gray = np.full((200, 200), 0.4)
        gray[:100, :] = 0.1
        s = default_settings()
        rgba = rgba_from_gray(gray)
        a = generate(rgba, s, 100.0, 100.0, analysis_max_dim=400)
        b = generate(rgba, s, 100.0, 100.0, analysis_max_dim=400)
        assert a.stone_count == b.stone_count
        np.testing.assert_array_equal(a.placement.positions_mm, b.placement.positions_mm)

    def test_organic_mode_deterministic(self):
        gray = np.full((200, 200), 0.3)
        s = default_settings(placement_mode="organic")
        rgba = rgba_from_gray(gray)
        a = generate(rgba, s, 100.0, 100.0, analysis_max_dim=400)
        b = generate(rgba, s, 100.0, 100.0, analysis_max_dim=400)
        np.testing.assert_array_equal(a.placement.positions_mm, b.placement.positions_mm)

    def test_transparent_regions_no_stones(self):
        # Transparency is a hard mask: a transparent image yields nothing.
        gray = np.full((200, 200), 0.02)
        rgba = rgba_from_gray(gray, opaque=False)
        res = generate(rgba, default_settings(), 100.0, 100.0, analysis_max_dim=400)
        assert res.stone_count == 0
