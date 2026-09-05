"""Tests for the collision detection and resolution engine."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.collision import (
    SpatialHash,
    check_position,
    detect_collisions,
    find_collisions,
    relax_positions,
    resolve_collisions,
    stones_in_collision,
)
from app.core.geometry import stones_overlap
from app.models.design import Design


def make_design(points, diameter=3.0, w=100.0, h=100.0) -> Design:
    d = Design(w, h)
    for (x, y) in points:
        d.add_stone(x, y, diameter)
    return d


def has_any_overlap(design: Design, gap: float) -> bool:
    return bool(find_collisions(design.positions(), design.diameters(), gap))


# --------------------------------------------------------------------------- #
# Spatial hash
# --------------------------------------------------------------------------- #


class TestSpatialHash:
    def test_finds_near_points(self):
        pos = np.array([[0, 0], [1, 1], [50, 50]], dtype=float)
        sh = SpatialHash(pos, cell_size=5.0)
        near = sh.candidates_near(0.5, 0.5)
        assert 0 in near and 1 in near
        assert 2 not in near

    def test_empty(self):
        sh = SpatialHash(np.empty((0, 2)), cell_size=5.0)
        assert sh.candidates_near(0, 0) == []


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


class TestDetection:
    def test_no_collision_when_apart(self):
        pos = np.array([[0, 0], [10, 0]], dtype=float)
        dia = np.array([3.0, 3.0])
        assert find_collisions(pos, dia, gap_mm=0.5) == []

    def test_detects_overlap(self):
        pos = np.array([[0, 0], [2.0, 0]], dtype=float)  # centres 2mm apart
        dia = np.array([3.0, 3.0])  # need >= 3mm
        pairs = find_collisions(pos, dia, gap_mm=0.0)
        assert pairs == [(0, 1)]

    def test_touching_not_collision(self):
        pos = np.array([[0, 0], [3.0, 0]], dtype=float)  # exactly touching
        dia = np.array([3.0, 3.0])
        assert find_collisions(pos, dia, gap_mm=0.0) == []

    def test_gap_enforced(self):
        pos = np.array([[0, 0], [3.2, 0]], dtype=float)
        dia = np.array([3.0, 3.0])
        assert find_collisions(pos, dia, gap_mm=0.0) == []      # ok without gap
        assert find_collisions(pos, dia, gap_mm=0.5) == [(0, 1)]  # violates gap

    def test_different_sizes(self):
        # required = (2 + 6)/2 = 4mm
        pos = np.array([[0, 0], [3.5, 0]], dtype=float)
        dia = np.array([2.0, 6.0])
        assert find_collisions(pos, dia, gap_mm=0.0) == [(0, 1)]

    def test_stones_in_collision_set(self):
        pos = np.array([[0, 0], [1.0, 0], [50, 50]], dtype=float)
        dia = np.array([3.0, 3.0, 3.0])
        assert stones_in_collision(pos, dia, 0.0) == {0, 1}

    def test_matches_geometry_helper(self):
        # Cross-check the vectorised detector against the scalar predicate.
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 20, size=(30, 2))
        dia = np.full(30, 3.0)
        pairs = set(find_collisions(pos, dia, 0.0))
        brute = set()
        for i in range(30):
            for j in range(i + 1, 30):
                if stones_overlap(pos[i, 0], pos[i, 1], 3.0, pos[j, 0], pos[j, 1], 3.0, 0.0):
                    brute.add((i, j))
        assert pairs == brute

    def test_single_stone_no_collision(self):
        assert find_collisions(np.array([[0, 0]], dtype=float), np.array([3.0]), 0.0) == []


class TestCheckPosition:
    def test_valid_position(self):
        pos = np.array([[0, 0], [10, 10]], dtype=float)
        dia = np.array([3.0, 3.0])
        assert check_position(pos, dia, 0.0, 5.0, 5.0, 3.0) is None

    def test_invalid_position(self):
        pos = np.array([[0, 0], [10, 10]], dtype=float)
        dia = np.array([3.0, 3.0])
        assert check_position(pos, dia, 0.0, 1.0, 0.0, 3.0) == 0

    def test_ignore_self(self):
        pos = np.array([[0, 0], [10, 10]], dtype=float)
        dia = np.array([3.0, 3.0])
        # Moving stone 0 "onto itself" must ignore index 0.
        assert check_position(pos, dia, 0.0, 0.0, 0.0, 3.0, ignore_index=0) is None

    def test_empty_design(self):
        assert check_position(np.empty((0, 2)), np.empty((0,)), 0.0, 5, 5, 3.0) is None


# --------------------------------------------------------------------------- #
# Design-level report
# --------------------------------------------------------------------------- #


class TestDetectCollisions:
    def test_reports_ids(self):
        d = make_design([(0, 0), (1.5, 0), (50, 50)])
        report = detect_collisions(d, gap_mm=0.0)
        assert report.has_collisions
        assert report.pair_count == 1
        ids = {s.id for s in d.stones[:2]}
        assert report.colliding_ids == ids
        assert report.worst_penetration_mm > 0

    def test_no_collisions_clean(self):
        d = make_design([(0, 0), (10, 0), (20, 0)])
        report = detect_collisions(d, gap_mm=0.0)
        assert not report.has_collisions
        assert report.stone_count == 0


# --------------------------------------------------------------------------- #
# Relaxation
# --------------------------------------------------------------------------- #


class TestRelax:
    def test_separates_overlap(self):
        pos = np.array([[0, 0], [1.0, 0]], dtype=float)  # 1mm apart, need 3mm
        dia = np.array([3.0, 3.0])
        out, converged = relax_positions(pos, dia, gap_mm=0.0)
        assert converged
        assert not find_collisions(out, dia, 0.0)

    def test_coincident_stones_separated(self):
        pos = np.array([[5, 5], [5, 5], [5, 5]], dtype=float)  # all identical
        dia = np.array([3.0, 3.0, 3.0])
        out, converged = relax_positions(pos, dia, gap_mm=0.0, max_iterations=200)
        assert converged
        assert not find_collisions(out, dia, 0.0)

    def test_stays_within_bounds(self):
        pos = np.array([[0.5, 0.5], [1.0, 0.5]], dtype=float)
        dia = np.array([3.0, 3.0])
        out, _ = relax_positions(pos, dia, 0.0, bounds_wh=(20.0, 20.0))
        radii = dia / 2.0
        assert np.all(out[:, 0] >= radii - 1e-6)
        assert np.all(out[:, 1] >= radii - 1e-6)
        assert np.all(out[:, 0] <= 20.0 - radii + 1e-6)

    def test_deterministic(self):
        pos = np.array([[5, 5], [5, 5], [6, 5], [5, 6]], dtype=float)
        dia = np.full(4, 3.0)
        a, _ = relax_positions(pos, dia, 0.0, max_iterations=100)
        b, _ = relax_positions(pos, dia, 0.0, max_iterations=100)
        np.testing.assert_array_equal(a, b)

    def test_already_valid_unchanged(self):
        pos = np.array([[0, 0], [10, 0]], dtype=float)
        dia = np.array([3.0, 3.0])
        out, converged = relax_positions(pos, dia, 0.0)
        assert converged
        np.testing.assert_allclose(out, pos)


# --------------------------------------------------------------------------- #
# Full resolution
# --------------------------------------------------------------------------- #


class TestResolveCollisions:
    def test_resolves_by_moving(self):
        d = make_design([(10, 10), (11, 10)])  # slight overlap
        result = resolve_collisions(d, gap_mm=0.0)
        assert result.converged
        assert result.moved_count >= 1
        assert result.removed_count == 0
        assert not has_any_overlap(d, 0.0)
        assert d.stone_count == 2  # nothing removed

    def test_preserves_stone_count_when_possible(self):
        # A ring of stones packed a bit tight but relaxable.
        pts = [(50 + 2.0 * np.cos(t), 50 + 2.0 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        d = make_design(pts, diameter=3.0)
        result = resolve_collisions(d, gap_mm=0.0)
        assert not has_any_overlap(d, 0.0)
        assert result.removed_count == 0

    def test_removes_when_overpacked(self):
        # 12 stones forced into a tiny 6x6mm box: cannot all fit → some removed.
        rng = np.random.default_rng(2)
        pts = [(50 + rng.uniform(-3, 3), 50 + rng.uniform(-3, 3)) for _ in range(12)]
        d = make_design(pts, diameter=3.0)
        before = d.stone_count
        result = resolve_collisions(d, gap_mm=0.0, max_iterations=40)
        assert not has_any_overlap(d, 0.0)
        assert result.converged
        assert result.removed_count > 0
        assert d.stone_count == before - result.removed_count

    def test_no_removal_when_disabled(self):
        rng = np.random.default_rng(3)
        pts = [(50 + rng.uniform(-2, 2), 50 + rng.uniform(-2, 2)) for _ in range(15)]
        d = make_design(pts, diameter=3.0)
        before = d.stone_count
        result = resolve_collisions(d, gap_mm=0.0, max_iterations=10, allow_removal=False)
        assert d.stone_count == before  # nothing removed
        assert result.removed_count == 0

    def test_clean_design_no_change(self):
        d = make_design([(0, 0), (10, 0), (20, 0)])
        result = resolve_collisions(d, gap_mm=0.0)
        assert not result.changed
        assert result.converged

    def test_single_stone(self):
        d = make_design([(5, 5)])
        result = resolve_collisions(d, gap_mm=0.0)
        assert result.converged
        assert not result.changed

    def test_result_changed_flag(self):
        d = make_design([(10, 10), (11, 10)])
        result = resolve_collisions(d, gap_mm=0.0)
        assert result.changed
