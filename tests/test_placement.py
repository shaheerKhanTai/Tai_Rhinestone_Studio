"""Tests for the placement engine: modes, determinism, validity, limits."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.geometry import DesignTransform, GeometryError
from app.core.placement import (
    PlacementCancelled,
    place_stones,
)
from app.models.settings import GenerationSettings, PlacementMode


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def solid_density(h=80, w=80) -> np.ndarray:
    """A fully-dense field — every valid lattice cell should be filled."""
    return np.ones((h, w), dtype=np.float32)


def transform_for(density: np.ndarray, width_mm: float, height_mm: float) -> DesignTransform:
    return DesignTransform(density.shape[1], density.shape[0], width_mm, height_mm)


def min_pairwise_distance(pos: np.ndarray) -> float:
    if pos.shape[0] < 2:
        return float("inf")
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def settings_for(mode: PlacementMode, **kw) -> GenerationSettings:
    base = dict(stone_size_id="SS10", spacing_mm=0.5, placement_mode=mode)
    base.update(kw)
    return GenerationSettings(**base)


ALL_MODES = [PlacementMode.GRID, PlacementMode.OFFSET, PlacementMode.HEXAGONAL, PlacementMode.ORGANIC]


# --------------------------------------------------------------------------- #
# Basic behaviour per mode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ALL_MODES)
class TestModes:
    def test_places_stones_on_solid(self, mode):
        d = solid_density()
        t = transform_for(d, 60.0, 60.0)
        result = place_stones(d, t, settings_for(mode))
        assert result.count > 0

    def test_positions_within_canvas(self, mode):
        d = solid_density()
        t = transform_for(d, 60.0, 60.0)
        pos = place_stones(d, t, settings_for(mode)).positions_mm
        assert np.all(pos[:, 0] >= 0) and np.all(pos[:, 0] <= 60.0 + 1e-6)
        assert np.all(pos[:, 1] >= 0) and np.all(pos[:, 1] <= 60.0 + 1e-6)

    def test_no_overlaps(self, mode):
        d = solid_density()
        t = transform_for(d, 50.0, 50.0)
        s = settings_for(mode)
        pos = place_stones(d, t, s).positions_mm
        # Every centre must be at least the pitch apart (touching allowed).
        assert min_pairwise_distance(pos) >= s.center_distance() - 1e-4

    def test_deterministic(self, mode):
        d = solid_density()
        t = transform_for(d, 55.0, 45.0)
        s = settings_for(mode)
        a = place_stones(d, t, s).positions_mm
        b = place_stones(d, t, s).positions_mm
        np.testing.assert_array_equal(a, b)

    def test_empty_density_no_stones(self, mode):
        d = np.zeros((60, 60), dtype=np.float32)
        t = transform_for(d, 60.0, 60.0)
        assert place_stones(d, t, settings_for(mode)).count == 0


# --------------------------------------------------------------------------- #
# Cross-mode relationships
# --------------------------------------------------------------------------- #


class TestPackingDensity:
    def test_hex_denser_than_grid(self):
        d = solid_density(100, 100)
        t = transform_for(d, 80.0, 80.0)
        grid = place_stones(d, t, settings_for(PlacementMode.GRID)).count
        hexn = place_stones(d, t, settings_for(PlacementMode.HEXAGONAL)).count
        assert hexn > grid

    def test_larger_stone_fewer_placements(self):
        d = solid_density(100, 100)
        t = transform_for(d, 80.0, 80.0)
        small = place_stones(d, t, settings_for(PlacementMode.GRID, stone_size_id="SS6")).count
        large = place_stones(d, t, settings_for(PlacementMode.GRID, stone_size_id="SS30")).count
        assert small > large

    def test_more_spacing_fewer_stones(self):
        d = solid_density(100, 100)
        t = transform_for(d, 80.0, 80.0)
        tight = place_stones(d, t, settings_for(PlacementMode.GRID, spacing_mm=0.2)).count
        loose = place_stones(d, t, settings_for(PlacementMode.GRID, spacing_mm=3.0)).count
        assert tight > loose


# --------------------------------------------------------------------------- #
# Maximum stone count
# --------------------------------------------------------------------------- #


class TestMaxStones:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_limit_respected(self, mode):
        d = solid_density(120, 120)
        t = transform_for(d, 100.0, 100.0)
        result = place_stones(d, t, settings_for(mode, max_stones=50))
        assert result.count <= 50
        assert result.was_limited
        assert result.total_before_limit > 50

    def test_no_limit_when_under(self):
        d = solid_density(40, 40)
        t = transform_for(d, 30.0, 30.0)
        result = place_stones(d, t, settings_for(PlacementMode.GRID, max_stones=100000))
        assert not result.was_limited
        assert result.count == result.total_before_limit

    def test_limit_keeps_highest_density(self):
        # Left half dense, right half faint: the cap should keep the dense side.
        d = np.zeros((80, 80), dtype=np.float32)
        d[:, :40] = 1.0
        d[:, 40:] = 0.30
        t = transform_for(d, 80.0, 80.0)
        result = place_stones(d, t, settings_for(PlacementMode.GRID, max_stones=30))
        assert result.count <= 30
        # Kept stones should skew to the dense (left) half.
        left = int(np.sum(result.positions_mm[:, 0] < 40.0))
        assert left >= result.count // 2


# --------------------------------------------------------------------------- #
# Progress and cancellation
# --------------------------------------------------------------------------- #


class TestProgressCancel:
    def test_progress_called(self):
        d = solid_density()
        t = transform_for(d, 50.0, 50.0)
        seen: list[tuple[str, float]] = []
        place_stones(d, t, settings_for(PlacementMode.HEXAGONAL),
                     progress=lambda s, f: seen.append((s, f)))
        assert seen
        assert seen[-1][1] == pytest.approx(1.0)
        assert all(0.0 <= f <= 1.0 for _, f in seen)

    def test_cancellation_raises(self):
        d = solid_density()
        t = transform_for(d, 50.0, 50.0)
        with pytest.raises(PlacementCancelled):
            place_stones(d, t, settings_for(PlacementMode.GRID), is_cancelled=lambda: True)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    def test_zero_spacing(self):
        d = solid_density(60, 60)
        t = transform_for(d, 50.0, 50.0)
        s = settings_for(PlacementMode.GRID, spacing_mm=0.0)
        result = place_stones(d, t, s)
        assert result.count > 0
        assert min_pairwise_distance(result.positions_mm) >= s.effective_diameter() - 1e-4

    def test_tiny_design_no_crash(self):
        d = solid_density(10, 10)
        t = transform_for(d, 1.0, 1.0)  # 1mm canvas, pitch 3.3mm → no room
        result = place_stones(d, t, settings_for(PlacementMode.GRID))
        assert result.count == 0

    def test_large_design_many_stones(self):
        d = solid_density(200, 200)
        t = transform_for(d, 300.0, 300.0)  # ~90k cells at SS10 pitch
        result = place_stones(d, t, settings_for(PlacementMode.HEXAGONAL))
        assert result.count > 5000

    def test_density_shape_mismatch_raises(self):
        d = solid_density(60, 60)
        t = DesignTransform(80, 80, 50.0, 50.0)  # wrong image size
        with pytest.raises(GeometryError):
            place_stones(d, t, settings_for(PlacementMode.GRID))

    def test_non_2d_density_raises(self):
        t = DesignTransform(10, 10, 50.0, 50.0)
        with pytest.raises(GeometryError):
            place_stones(np.ones((10, 10, 3), dtype=np.float32), t, settings_for(PlacementMode.GRID))

    def test_pitch_reported(self):
        d = solid_density(40, 40)
        t = transform_for(d, 30.0, 30.0)
        s = settings_for(PlacementMode.GRID)
        result = place_stones(d, t, s)
        assert result.pitch_mm == pytest.approx(s.center_distance())
