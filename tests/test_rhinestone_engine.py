"""Tests for the high-level rhinestone generation engine."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.rhinestone_engine import density_params_from_settings, generate
from app.models.settings import GenerationSettings, PlacementMode
from app.models.stone import Stone


def disc_image(h=300, w=300, radius=100) -> np.ndarray:
    """A solid dark disc on a transparent background."""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.circle(rgba, (w // 2, h // 2), radius, (20, 20, 20, 255), thickness=-1)
    return rgba


class TestGenerate:
    def test_produces_stones(self):
        rgba = disc_image()
        result = generate(rgba, GenerationSettings(), 100.0, 100.0)
        assert result.stone_count > 0
        assert all(isinstance(s, Stone) for s in result.stones)

    def test_stone_ids_unique_and_sequential(self):
        result = generate(disc_image(), GenerationSettings(), 100.0, 100.0)
        ids = [s.id for s in result.stones]
        assert ids == list(range(1, len(ids) + 1))
        assert len(set(ids)) == len(ids)

    def test_stones_carry_diameter_and_size(self):
        s = GenerationSettings(stone_size_id="SS16")
        result = generate(disc_image(), s, 100.0, 100.0)
        assert result.stones[0].diameter == pytest.approx(4.0)
        assert result.stones[0].size_id == "SS16"

    def test_positions_in_physical_mm(self):
        # 100mm canvas → all stones within [0,100].
        result = generate(disc_image(), GenerationSettings(), 100.0, 100.0)
        xs = np.array([s.x for s in result.stones])
        ys = np.array([s.y for s in result.stones])
        assert xs.max() <= 100.0 + 1e-6
        assert ys.max() <= 100.0 + 1e-6

    def test_stones_concentrate_on_subject(self):
        # Disc is centred; stones should cluster near the centre, not corners.
        result = generate(disc_image(radius=80), GenerationSettings(), 100.0, 100.0)
        pos = np.array([(s.x, s.y) for s in result.stones])
        # No stone in the extreme corner (transparent region).
        corner = np.sum((pos[:, 0] < 10) & (pos[:, 1] < 10))
        assert corner == 0

    def test_transparent_background_excluded(self):
        result = generate(disc_image(radius=60), GenerationSettings(), 100.0, 100.0)
        # All stones lie within ~ the disc radius (60/150*100 = 40mm) of centre.
        pos = np.array([(s.x, s.y) for s in result.stones])
        dist = np.sqrt((pos[:, 0] - 50.0) ** 2 + (pos[:, 1] - 50.0) ** 2)
        assert dist.max() < 48.0  # a little slack for the stone radius + edges

    def test_max_stones_enforced(self):
        s = GenerationSettings(max_stones=100)
        result = generate(disc_image(), s, 150.0, 150.0)
        assert result.stone_count <= 100
        assert result.was_limited

    def test_deterministic(self):
        rgba = disc_image()
        s = GenerationSettings(placement_mode=PlacementMode.ORGANIC)
        a = generate(rgba, s, 100.0, 100.0)
        b = generate(rgba, s, 100.0, 100.0)
        assert [(x.x, x.y) for x in a.stones] == [(x.x, x.y) for x in b.stones]

    def test_progress_reported(self):
        seen: list[float] = []
        generate(disc_image(), GenerationSettings(), 100.0, 100.0,
                 progress=lambda stage, frac: seen.append(frac))
        assert seen
        assert min(seen) >= 0.0 and max(seen) == pytest.approx(1.0)

    def test_custom_color_applied(self):
        result = generate(disc_image(), GenerationSettings(), 100.0, 100.0, color="#FF0000")
        assert all(s.color == "#FF0000" for s in result.stones)

    def test_transform_matches_density(self):
        result = generate(disc_image(), GenerationSettings(), 120.0, 90.0)
        assert result.transform.design_width_mm == 120.0
        assert result.transform.image_width_px == result.density_field.width


class TestDensityParamsFromSettings:
    def test_maps_sliders(self):
        s = GenerationSettings(detail=0.7, density=0.2, contrast=0.9)
        p = density_params_from_settings(s)
        assert p.detail == 0.7
        assert p.density == 0.2
        assert p.contrast == 0.9
        assert p.invert is False

    def test_invert_flag(self):
        p = density_params_from_settings(GenerationSettings(), invert=True)
        assert p.invert is True
