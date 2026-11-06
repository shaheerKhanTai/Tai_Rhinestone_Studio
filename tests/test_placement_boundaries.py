"""Boundary containment for every placement mode (FEATURE 3B, requirement 7).

A stone whose centre sits closer to the artboard edge than its own radius is
cut off by the output boundary — half a rhinestone on the finished transfer.
The lattice modes used to filter candidates with ``x <= width`` only, so a
canvas whose size is not a whole number of pitches could place its last column
within a radius of the edge (e.g. a 6 mm design at a 3.3 mm pitch: the columns
land on 1.65 and 4.95 mm, and 4.95 mm is 1.05 mm from the edge for a stone with
a 1.4 mm radius).

Every mode must now keep centres at least half a pitch — and therefore more
than one radius — from each edge, without emptying the design.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.collision import find_collisions
from app.core.geometry import DesignTransform
from app.core.image_processor import DensityParams, process_to_density
from app.core.placement import place_stones
from app.models.settings import GenerationSettings, PlacementMode

D = 2.8        # SS10
GAP = 0.5
PITCH = D + GAP
RADIUS = D / 2.0

ALL_MODES = [
    PlacementMode.AUTOMATIC,
    PlacementMode.OUTLINE,
    PlacementMode.GRID,
    PlacementMode.OFFSET,
    PlacementMode.HEXAGONAL,
    PlacementMode.ORGANIC,
]


def dark_image(h: int = 200, w: int = 200) -> np.ndarray:
    """A fully opaque near-black image: maximum density, every cell a candidate."""
    rgba = np.full((h, w, 4), 25, np.uint8)
    rgba[:, :, 3] = 255
    return rgba


def run(rgba, w_mm: float, h_mm: float, mode: PlacementMode = PlacementMode.GRID,
        **kw):
    settings = GenerationSettings(
        stone_size_id="SS10", spacing_mm=GAP, placement_mode=mode, **kw
    )
    field = process_to_density(rgba, DensityParams(), max_dim=600)
    transform = DesignTransform(field.width, field.height, w_mm, h_mm)
    return place_stones(field.density, transform, settings), settings


def assert_no_clipping(result, w_mm: float, h_mm: float) -> None:
    """No stone is cut off: every centre keeps a full radius from each edge."""
    pos = result.positions_mm
    assert pos.shape[0] >= 0
    if pos.shape[0] == 0:
        return
    assert float(pos[:, 0].min()) >= RADIUS - 1e-6, "stone clipped at the left edge"
    assert float(pos[:, 1].min()) >= RADIUS - 1e-6, "stone clipped at the top edge"
    assert float(pos[:, 0].max()) <= w_mm - RADIUS + 1e-6, "stone clipped at the right edge"
    assert float(pos[:, 1].max()) <= h_mm - RADIUS + 1e-6, "stone clipped at the bottom edge"


@pytest.mark.parametrize("mode", ALL_MODES)
class TestNoClipping:
    def test_solid_image_stays_inside(self, mode):
        res, _ = run(dark_image(), 100.0, 100.0, mode)
        assert res.count > 0
        assert_no_clipping(res, 100.0, 100.0)

    def test_non_square_design(self, mode):
        res, _ = run(dark_image(150, 300), 160.0, 80.0, mode)
        assert_no_clipping(res, 160.0, 80.0)


@pytest.mark.parametrize("mode", [PlacementMode.GRID, PlacementMode.OFFSET,
                                  PlacementMode.HEXAGONAL, PlacementMode.ORGANIC])
class TestAxisSweep:
    """Design widths chosen so the last lattice column lands right on the edge.

    At a 3.3 mm pitch the columns fall on 1.65 + 3.3k mm, so a 5 mm or 6 mm
    canvas puts one within a stone radius of the edge — the exact geometry the
    ``x <= width`` filter used to admit.
    """

    @pytest.mark.parametrize("w_mm", [4.6, 5.0, 5.8, 6.0, 7.7, 9.4, 12.35, 19.9])
    def test_width_sweep(self, mode, w_mm):
        res, settings = run(dark_image(), w_mm, w_mm, mode)
        assert_no_clipping(res, w_mm, w_mm)
        assert_no_overlaps(res, settings)

    @pytest.mark.parametrize("h_mm", [5.0, 6.0, 11.3, 17.8])
    def test_height_sweep(self, mode, h_mm):
        res, settings = run(dark_image(), 100.0, h_mm, mode)
        assert_no_clipping(res, 100.0, h_mm)
        assert_no_overlaps(res, settings)


def assert_no_overlaps(result, settings) -> None:
    if result.count == 0:
        return
    dia = np.full(result.count, settings.effective_diameter())
    assert find_collisions(result.positions_mm, dia, settings.spacing_mm) == []


class TestDesignIsStillFilled:
    """The containment margin must not empty the design out."""

    def test_solid_image_still_dense(self):
        res, _ = run(dark_image(), 100.0, 100.0, PlacementMode.GRID)
        # A 100 mm canvas at a 3.3 mm pitch holds a 30x30 grid minus the
        # contained margin; anything close to that means nothing was lost.
        assert res.count >= 28 * 28

    def test_edge_column_survives_when_it_fits(self):
        # 10 mm canvas: columns at 1.65 / 4.95 / 8.25 mm all clear the margin,
        # so the design keeps its full three columns.
        res, _ = run(dark_image(), 10.0, 10.0, PlacementMode.GRID)
        xs = np.unique(np.round(res.positions_mm[:, 0], 3))
        assert list(xs) == [1.65, 4.95, 8.25]

    def test_outline_mode_keeps_the_design(self):
        res, _ = run(dark_image(), 100.0, 100.0, PlacementMode.OUTLINE)
        assert res.count > 300


class TestSpacingStillGuaranteed:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_spacing_on_a_solid_field(self, mode):
        res, settings = run(dark_image(), 100.0, 100.0, mode)
        assert_no_overlaps(res, settings)

    def test_every_mode_is_deterministic(self):
        for mode in ALL_MODES:
            a, _ = run(dark_image(), 100.0, 100.0, mode)
            b, _ = run(dark_image(), 100.0, 100.0, mode)
            assert a.count == b.count
            np.testing.assert_array_equal(a.positions_mm, b.positions_mm)
