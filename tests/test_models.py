"""Tests for the domain models: stones, design, settings, presets, project."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.geometry import GeometryError, Unit
from app.models import (
    CUSTOM_SIZE_NAME,
    DEFAULT_PRESETS,
    Design,
    GenerationSettings,
    ImageState,
    PlacementMode,
    Project,
    SizePreset,
    Stone,
    get_stone_size,
    make_custom_size,
    resolve_diameter,
)


class TestStoneSize:
    def test_standard_lookup(self):
        assert get_stone_size("SS10").diameter_mm == pytest.approx(2.8)
        assert get_stone_size("SS30").diameter_mm == pytest.approx(6.4)

    def test_radius(self):
        assert get_stone_size("SS16").radius_mm == pytest.approx(2.0)

    def test_unknown_size_raises(self):
        with pytest.raises(KeyError):
            get_stone_size("SS999")

    def test_custom_size(self):
        cs = make_custom_size(3.5)
        assert cs.name == CUSTOM_SIZE_NAME
        assert cs.diameter_mm == 3.5

    def test_custom_size_invalid(self):
        with pytest.raises(ValueError):
            make_custom_size(0)

    def test_resolve_diameter(self):
        assert resolve_diameter("SS12") == pytest.approx(3.2)
        assert resolve_diameter(CUSTOM_SIZE_NAME, 5.0) == 5.0
        with pytest.raises(ValueError):
            resolve_diameter(CUSTOM_SIZE_NAME, None)


class TestStone:
    def test_properties(self):
        s = Stone(1, 10.0, 20.0, 3.0, "SS12")
        assert s.radius == 1.5
        assert s.position.x == 10.0

    def test_move(self):
        s = Stone(1, 0, 0, 3.0)
        s.move_by(5, -2)
        assert (s.x, s.y) == (5, -2)
        s.move_to(1, 1)
        assert (s.x, s.y) == (1, 1)

    def test_roundtrip(self):
        s = Stone(7, 1.5, 2.5, 4.0, "SS20", "#FF0000", selected=True)
        d = s.to_dict()
        assert "selected" not in d  # selection is transient
        restored = Stone.from_dict(d)
        assert restored.id == 7
        assert restored.color == "#FF0000"
        assert restored.selected is False


class TestDesign:
    def test_add_allocates_ids(self):
        d = Design(100, 80)
        a = d.add_stone(10, 10, 3.0)
        b = d.add_stone(20, 20, 3.0)
        assert a.id != b.id
        assert d.stone_count == 2

    def test_remove(self):
        d = Design(100, 80)
        a = d.add_stone(10, 10, 3.0)
        d.add_stone(20, 20, 3.0)
        d.remove(a)
        assert d.stone_count == 1
        assert d.get_by_id(a.id) is None

    def test_remove_missing_raises(self):
        d = Design(100, 80)
        with pytest.raises(KeyError):
            d.remove_by_id(999)

    def test_remove_many(self):
        d = Design(100, 80)
        ids = [d.add_stone(i, i, 3.0).id for i in range(5)]
        removed = d.remove_many(ids[:3])
        assert len(removed) == 3
        assert d.stone_count == 2

    def test_positions_and_diameters(self):
        d = Design(100, 80)
        d.add_stone(1, 2, 3.0)
        d.add_stone(4, 5, 6.0)
        pos = d.positions()
        assert pos.shape == (2, 2)
        np.testing.assert_allclose(pos[1], [4, 5])
        np.testing.assert_allclose(d.diameters(), [3.0, 6.0])

    def test_empty_positions(self):
        d = Design(100, 80)
        assert d.positions().shape == (0, 2)
        assert d.diameters().shape == (0,)
        assert d.bounds() is None

    def test_bounds_includes_radius(self):
        d = Design(100, 80)
        d.add_stone(50, 40, 4.0)  # radius 2 → bounds 48..52, 38..42
        b = d.bounds()
        assert b is not None
        assert b.left == pytest.approx(48.0)
        assert b.right == pytest.approx(52.0)

    def test_selection(self):
        d = Design(100, 80)
        d.add_stone(1, 1, 3.0)
        d.add_stone(2, 2, 3.0)
        d.select_all()
        assert len(d.selected_stones()) == 2
        d.clear_selection()
        assert len(d.selected_stones()) == 0

    def test_zero_dimension_rejected(self):
        with pytest.raises(GeometryError):
            Design(0, 80)

    def test_roundtrip_preserves_ids(self):
        d = Design(100, 80)
        d.add_stone(1, 1, 3.0)
        d.add_stone(2, 2, 3.0)
        restored = Design.from_dict(d.to_dict())
        assert restored.stone_count == 2
        # A newly added stone must not collide with restored ids.
        new = restored.add_stone(3, 3, 3.0)
        assert new.id not in {s.id for s in d.stones}

    def test_color_counts(self):
        d = Design(100, 80)
        d.add_stone(1, 1, 3.0, color="#000")
        d.add_stone(2, 2, 3.0, color="#000")
        d.add_stone(3, 3, 3.0, color="#FFF")
        assert d.color_counts() == {"#000": 2, "#FFF": 1}


class TestGenerationSettings:
    def test_defaults_valid(self):
        s = GenerationSettings()
        assert s.effective_diameter() == pytest.approx(2.8)  # SS10
        assert s.center_distance() == pytest.approx(3.3)

    def test_sliders_clamped(self):
        s = GenerationSettings(density=5.0, contrast=-1.0)
        assert s.density == 1.0
        assert s.contrast == 0.0

    def test_custom_requires_diameter(self):
        with pytest.raises(GeometryError):
            GenerationSettings(stone_size_id="Custom", custom_diameter_mm=None)

    def test_custom_diameter_used(self):
        s = GenerationSettings(stone_size_id="Custom", custom_diameter_mm=5.0)
        assert s.effective_diameter() == 5.0

    def test_negative_spacing_rejected(self):
        with pytest.raises(GeometryError):
            GenerationSettings(spacing_mm=-1.0)

    def test_max_stones_validation(self):
        with pytest.raises(GeometryError):
            GenerationSettings(max_stones=0)

    def test_roundtrip(self):
        s = GenerationSettings(
            stone_size_id="SS16",
            spacing_mm=0.8,
            placement_mode=PlacementMode.ORGANIC,
            detail=0.9,
            max_stones=5000,
        )
        restored = GenerationSettings.from_dict(s.to_dict())
        assert restored.stone_size_id == "SS16"
        assert restored.placement_mode is PlacementMode.ORGANIC
        assert restored.max_stones == 5000
        assert restored.detail == pytest.approx(0.9)

    def test_copy_independent(self):
        s = GenerationSettings()
        c = s.copy()
        c.density = 0.1
        assert s.density != 0.1


class TestPlacementMode:
    def test_from_str(self):
        assert PlacementMode.from_str("grid") is PlacementMode.GRID
        assert PlacementMode.from_str("HEXAGONAL") is PlacementMode.HEXAGONAL
        with pytest.raises(GeometryError):
            PlacementMode.from_str("spiral")

    def test_labels(self):
        assert PlacementMode.OFFSET.label == "Offset / Brick"


class TestPresets:
    def test_default_presets_valid(self):
        assert len(DEFAULT_PRESETS) >= 7
        for p in DEFAULT_PRESETS:
            assert p.width_mm > 0
            assert p.height_mm > 0

    def test_preset_mm_conversion(self):
        p = SizePreset("Test", 10.0, 8.0, Unit.INCH)
        assert p.width_mm == pytest.approx(254.0)
        assert p.height_mm == pytest.approx(203.2)

    def test_preset_roundtrip(self):
        p = SizePreset("Cap", 5.0, 2.25, Unit.INCH)
        restored = SizePreset.from_dict(p.to_dict())
        assert restored == p

    def test_invalid_preset(self):
        with pytest.raises(GeometryError):
            SizePreset("Bad", 0, 5, Unit.INCH)


class TestImageState:
    def test_invalid_rotation(self):
        with pytest.raises(GeometryError):
            ImageState(rotation=45)

    def test_reset(self):
        s = ImageState(rotation=90, flip_h=True, crop=(0, 0, 10, 10))
        s.reset_transforms()
        assert s.rotation == 0 and not s.flip_h and s.crop is None

    def test_roundtrip(self):
        s = ImageState(rotation=180, flip_v=True, crop=(1, 2, 3, 4))
        restored = ImageState.from_dict(s.to_dict())
        assert restored.rotation == 180
        assert restored.flip_v is True
        assert restored.crop == (1, 2, 3, 4)


class TestProject:
    def test_new_project_defaults(self):
        p = Project.new("My Design")
        assert p.name == "My Design"
        assert p.width_mm == pytest.approx(254.0)
        assert p.design.width_mm == pytest.approx(254.0)

    def test_set_physical_size_syncs_design(self):
        p = Project.new()
        p.set_physical_size(100.0, 50.0)
        assert p.design.width_mm == 100.0
        assert p.design.height_mm == 50.0

    def test_dimensions_in_units(self):
        p = Project.new()  # 254mm × 203.2mm, inches
        assert p.width_in_units == pytest.approx(10.0)
        assert p.height_in_units == pytest.approx(8.0)

    def test_full_roundtrip(self):
        p = Project.new("Eagle")
        p.set_physical_size(200.0, 150.0)
        p.design.add_stone(10, 10, 3.0)
        p.design.add_stone(20, 20, 3.0)
        p.settings.stone_size_id = "SS12"

        restored = Project.from_dict(p.to_dict())
        assert restored.name == "Eagle"
        assert restored.width_mm == 200.0
        assert restored.design.stone_count == 2
        assert restored.settings.stone_size_id == "SS12"
        assert restored.design.width_mm == 200.0

    def test_invalid_dimensions(self):
        with pytest.raises(GeometryError):
            Project(width_mm=0, height_mm=10)
