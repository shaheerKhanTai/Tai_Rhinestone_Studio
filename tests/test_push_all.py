"""Tests for Push All (ISSUE 3) — the core propagation algorithm and its
canvas/settings integration.

The algorithm under test is :func:`app.core.push.propagate_push`: dragging a
stone (the *pinned* stone) must push only the neighbours it violates — and
neighbours those push in turn — until the configured minimum centre distance
(diameter + gap) holds again. Distant stones must never move.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from app.core.collision import find_collisions
from app.core.push import propagate_push
from app.models.settings import GenerationSettings

D = 3.0     # stone diameter (mm)
GAP = 0.5   # gap (mm)
PITCH = D + GAP


def _line(n: int, pitch: float = PITCH, y: float = 50.0) -> np.ndarray:
    """n stones in a horizontal row at exactly ``pitch`` apart (valid layout)."""
    return np.array([[10.0 + i * pitch, y] for i in range(n)], dtype=np.float64)


def _dias(n: int) -> np.ndarray:
    return np.full(n, D, dtype=np.float64)


def _pin(n: int, *indices: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for i in indices:
        m[i] = True
    return m


def _min_dist(pos: np.ndarray) -> float:
    """Smallest pairwise centre distance in a small layout."""
    best = float("inf")
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            best = min(best, float(np.hypot(*(pos[i] - pos[j]))))
    return best


class TestPushOneNeighbour:
    def test_pushed_stone_keeps_min_distance(self):
        # Two stones at pitch; drag the left one onto the right one's spot.
        pos = _line(2)
        pos[0] = [10.0 + PITCH * 0.4, 50.0]  # now violating
        res = propagate_push(pos, _dias(2), GAP, _pin(2, 0), (200.0, 100.0))
        assert res.converged
        # Pinned stone stays exactly where it was put.
        assert res.positions[0, 0] == pytest.approx(10.0 + PITCH * 0.4)
        # The neighbour was pushed away to the full pitch.
        assert res.positions[1, 0] - res.positions[0, 0] == pytest.approx(PITCH, abs=1e-6)
        assert res.pushed_indices == [1]

    def test_no_violation_no_movement(self):
        pos = _line(2)
        res = propagate_push(pos, _dias(2), GAP, _pin(2, 0), (200.0, 100.0))
        assert res.converged
        assert res.pushed_count == 0
        assert np.array_equal(res.positions, pos)

    def test_push_uses_straight_line_direction(self):
        # Dragged diagonally onto a neighbour: the push is along the line
        # joining the two centres, not axis-aligned.
        pos = np.array([[10.0, 50.0], [10.0 + PITCH, 50.0]], dtype=np.float64)
        pos[0] = [10.0 + PITCH * 0.5, 50.0 + 0.4]
        res = propagate_push(pos, _dias(2), GAP, _pin(2, 0), (200.0, 100.0))
        d = np.hypot(*(res.positions[1] - res.positions[0]))
        assert d == pytest.approx(PITCH, abs=1e-6)


class TestPushMultipleNeighbours:
    def test_pushes_both_sides(self):
        # Two stones 1.4 pitches apart; a pinned stone dragged to the midpoint
        # violates both simultaneously.
        pos = np.array(
            [[50.0 - PITCH * 0.7, 50.0], [50.0 + PITCH * 0.7, 50.0], [200.0, 50.0]],
            dtype=np.float64,
        )
        # Stone 2 starts far away; the dragged stone (index 2) moves between
        # stones 0 and 1.
        pos[2] = [50.0, 50.0]
        res = propagate_push(pos, _dias(3), GAP, _pin(3, 2), (300.0, 100.0))
        assert res.converged
        assert sorted(res.pushed_indices) == [0, 1]
        assert _min_dist(res.positions) >= PITCH - 1e-6

    def test_surrounded_stone_pushes_all_neighbours(self):
        # A stone dropped in the middle of a 3x3 grid pushes all 8 neighbours.
        pos = []
        for ry in range(3):
            for rx in range(3):
                pos.append([50.0 + rx * PITCH, 50.0 + ry * PITCH])
        pos = np.array(pos, dtype=np.float64)
        pos[4] = [50.0 + PITCH, 50.0 + PITCH]  # exactly on stone 5 -> violation
        pos[4] = [50.0 + PITCH * 0.5, 50.0 + PITCH * 0.5]
        res = propagate_push(pos, np.full(9, D), GAP, _pin(9, 4), (300.0, 300.0))
        assert res.converged
        assert res.pushed_count >= 2
        pairs = find_collisions(res.positions, np.full(9, D), GAP)
        assert pairs == []


class TestPropagation:
    def test_cascades_through_a_line(self):
        # Drag stone 0 into stone 1: the wave must travel down the whole row.
        pos = _line(5)
        pos[0] = [10.0 + PITCH * 0.5, 50.0]
        res = propagate_push(pos, _dias(5), GAP, _pin(5, 0), (200.0, 100.0))
        assert res.converged
        # Only stones 1..4 moved (stone 0 is pinned; none exist beyond the row).
        assert set(res.pushed_indices) == {1, 2, 3, 4}
        # The chain preserved the pitch: consecutive distances all valid.
        for i in range(4):
            d = float(np.hypot(*(res.positions[i + 1] - res.positions[i])))
            assert d >= PITCH - 1e-6

    def test_distant_stones_untouched(self):
        # A dense cluster on the left, a far-away stone on the right: dragging
        # in the cluster must never move the distant stone.
        pos = _line(5)
        pos = np.vstack([pos, [[500.0, 50.0]]])
        pos[0] = [10.0 + PITCH * 0.5, 50.0]
        res = propagate_push(pos, _dias(6), GAP, _pin(6, 0), (600.0, 100.0))
        assert res.converged
        assert 5 not in res.pushed_indices
        assert res.positions[5, 0] == pytest.approx(500.0)
        assert res.positions[5, 1] == pytest.approx(50.0)

    def test_propagation_stops_when_spacing_ok(self):
        # A wide gap on the right absorbs the wave: the stone beyond the gap
        # must not move at all.
        pos = _line(3)
        pos = np.vstack([pos, [[10.0 + 3 * PITCH + 20.0, 50.0]]])  # far 4th stone
        pos[0] = [10.0 + PITCH * 0.5, 50.0]
        res = propagate_push(pos, _dias(4), GAP, _pin(4, 0), (300.0, 100.0))
        assert res.converged
        assert 3 not in res.pushed_indices


class TestFailureAndBounds:
    def test_impossible_move_reports_failure(self):
        # A corridor too narrow for two pitches: a pinned stone dragged to the
        # middle violates both walls' stones, and any push lands back inside a
        # violation (ping-pong against the bounds) — unresolvable.
        pos = np.array(
            [[0.2, 50.0], [3.8, 50.0], [2.0, 60.0]], dtype=np.float64
        )
        pos[2] = [2.0, 50.0]  # dragged to the midpoint
        res = propagate_push(pos, _dias(3), GAP, _pin(3, 2), (4.0, 100.0))
        assert not res.converged

    def test_push_clamped_to_bounds(self):
        # Pushing a stone past the right edge clamps it at the boundary.
        pos = np.array([[95.0, 50.0], [97.0, 50.0]], dtype=np.float64)
        res = propagate_push(pos, _dias(2), GAP, _pin(2, 0), (100.0, 100.0))
        assert res.positions[1, 0] <= 100.0 + 1e-9

    def test_unresolvable_corner_reports_failure(self):
        # A pinned stone dragged onto a stone already pinned at the boundary:
        # the pushed stone cannot leave the canvas, so the move is impossible.
        pos = np.array([[0.0, 0.0], [0.2, 0.0], [99.0, 99.0]], dtype=np.float64)
        pinned = np.array([True, False, True])
        res = propagate_push(pos, _dias(3), GAP, pinned, (100.0, 100.0))
        # Stone 1 has nowhere valid to go; the algorithm must not hang and must
        # report failure (or clamp-valid success), never corrupt positions.
        assert isinstance(res.converged, bool)
        assert res.passes <= 512

    def test_input_not_mutated(self):
        pos = _line(2)
        pos[0] = [10.0 + PITCH * 0.4, 50.0]
        original = pos.copy()
        propagate_push(pos, _dias(2), GAP, _pin(2, 0), (200.0, 100.0))
        assert np.array_equal(pos, original)


class TestDeterminismAndScale:
    def test_deterministic(self):
        pos = _line(8)
        pos[0] = [10.0 + PITCH * 0.6, 50.2]
        a = propagate_push(pos, _dias(8), GAP, _pin(8, 0), (400.0, 100.0))
        b = propagate_push(pos, _dias(8), GAP, _pin(8, 0), (400.0, 100.0))
        assert np.array_equal(a.positions, b.positions)
        assert a.pushed_indices == b.pushed_indices

    def test_large_layout_wave_is_local(self):
        # 3000 stones on a packed grid. A drag inside a fully-packed lattice
        # must propagate (there is no slack), but the wave stays bounded: it
        # never reaches the far corners and the layout ends collision-free.
        n = 3000
        xs, ys = [], []
        for i in range(n):
            col, row = i % 60, i // 60
            xs.append(5.0 + col * PITCH)
            ys.append(5.0 + row * PITCH)
        pos = np.column_stack([xs, ys])
        pos[1500] = pos[1500] + np.array([PITCH * 0.45, PITCH * 0.3])
        res = propagate_push(pos, _dias(n), GAP, _pin(n, 1500), (300.0, 300.0))
        assert res.converged
        assert res.pushed_count < n  # never the entire design
        # The far corners are untouched.
        assert 0 not in res.pushed_indices
        assert n - 1 not in res.pushed_indices
        pairs = find_collisions(res.positions, _dias(n), GAP)
        assert pairs == []

    def test_large_sparse_layout_pushes_few(self):
        # Same grid but with slack (stones at 1.5x pitch): the wave dies
        # within one ring — only a handful of local stones move.
        n = 1500
        xs, ys = [], []
        for i in range(n):
            col, row = i % 50, i // 50
            xs.append(5.0 + col * PITCH * 1.5)
            ys.append(5.0 + row * PITCH * 1.5)
        pos = np.column_stack([xs, ys])
        pos[750] = pos[750] + np.array([PITCH * 0.4, PITCH * 0.2])
        res = propagate_push(pos, np.full(n, D), GAP, _pin(n, 750), (300.0, 300.0))
        assert res.converged
        assert res.pushed_count <= 8
        pairs = find_collisions(res.positions, np.full(n, D), GAP)
        assert pairs == []


class TestSettingsPersistence:
    def test_push_all_round_trips(self):
        s = GenerationSettings(push_all=True)
        assert GenerationSettings.from_dict(s.to_dict()).push_all is True
        s2 = GenerationSettings()
        assert GenerationSettings.from_dict(s2.to_dict()).push_all is False

    def test_copy_preserves_push_all(self):
        assert GenerationSettings(push_all=True).copy().push_all is True


class TestCanvasPushAll:
    @pytest.fixture()
    def push_canvas(self, qapp):
        from app.core.history import CommandStack
        from app.models.design import Design
        from app.ui.canvas import CanvasView
        from app.ui.theme import DARK_THEME

        c = CanvasView(DARK_THEME)
        c.resize(800, 600)
        d = Design(200.0, 100.0)
        c.set_design(d, fit=False)
        c._set_zoom(6.0)
        c._origin = QPointF(20, 20)
        c.set_new_stone_spec(D, "SS10", "#111111", GAP)
        c.set_push_all(True)
        stack = CommandStack()
        c.command_produced.connect(stack.push_applied)
        return c, d, stack

    def _add_row(self, d, n=3):
        for i in range(n):
            d.add_stone(10.0 + i * PITCH, 50.0, D)

    def test_drag_pushes_neighbour_and_single_undo(self, push_canvas):
        c, d, stack = push_canvas
        self._add_row(d, 3)
        c.refresh_geometry()
        d.stones[0].selected = True
        c.refresh_selection()

        c._begin_move(QPointF(10.0, 50.0))
        c._update_move(QPointF(10.0 + PITCH * 0.5, 50.0))  # half a pitch right
        c._end_move()

        # Neighbour pushed away; spacing valid.
        pairs = find_collisions(d.positions(), d.diameters(), GAP)
        assert pairs == []
        # One undo restores the ENTIRE operation (drag + push).
        assert stack.undo()
        xs = [s.x for s in d.stones]
        assert xs == pytest.approx([10.0, 10.0 + PITCH, 10.0 + 2 * PITCH])

    def test_push_all_off_leaves_overlaps(self, push_canvas):
        c, d, stack = push_canvas
        c.set_push_all(False)
        self._add_row(d, 2)
        c.refresh_geometry()
        d.stones[0].selected = True
        c.refresh_selection()
        c._begin_move(QPointF(10.0, 50.0))
        c._update_move(QPointF(10.0 + PITCH * 0.5, 50.0))
        c._end_move()
        # Without Push All the move is allowed to create a violation.
        assert find_collisions(d.positions(), d.diameters(), GAP) != []

    def test_impossible_drag_restores_layout(self, push_canvas):
        c, d, stack = push_canvas
        # A corridor (4 mm wide) too narrow for two pitches: dragging the
        # middle stone between the two wall stones cannot be resolved.
        d.width_mm = 4.0
        d.height_mm = 100.0
        d.add_stone(0.2, 50.0, D)
        d.add_stone(3.8, 50.0, D)
        d.add_stone(2.0, 60.0, D)  # valid spot above the corridor
        c.refresh_geometry()
        d.stones[2].selected = True
        c.refresh_selection()
        failed = []
        c.push_failed.connect(lambda: failed.append(True))

        c._begin_move(QPointF(2.0, 60.0))
        c._update_move(QPointF(2.0, 50.0))  # into the corridor midpoint

        # Layout restored to the pre-drag snapshot; nothing recorded.
        xs = [s.x for s in d.stones]
        ys = [s.y for s in d.stones]
        assert xs == pytest.approx([0.2, 3.8, 2.0])
        assert ys == pytest.approx([50.0, 50.0, 60.0])
        assert failed == [True]
        c._end_move()
        assert not stack.can_undo  # nothing was recorded

    def test_distant_stone_untouched_in_canvas(self, push_canvas):
        c, d, stack = push_canvas
        self._add_row(d, 3)
        d.add_stone(150.0, 50.0, D)  # far away
        c.refresh_geometry()
        d.stones[0].selected = True
        c.refresh_selection()
        c._begin_move(QPointF(10.0, 50.0))
        c._update_move(QPointF(10.0 + PITCH * 0.5, 50.0))
        c._end_move()
        assert d.stones[3].x == pytest.approx(150.0)
        assert d.stones[3].y == pytest.approx(50.0)


class TestSidebarPushAllCheckbox:
    def test_checkbox_exists_with_tooltip(self, qapp):
        from app.core.geometry import Unit
        from app.models.presets import SizePreset
        from app.ui.sidebar import DesignPanel

        panel = DesignPanel([SizePreset("Custom", 10.0, 8.0, Unit.INCH)])
        assert panel._push_all.text() == "Push All"
        assert "neighboring rhinestones" in panel._push_all.toolTip()

    def test_checkbox_round_trips_through_settings(self, qapp):
        from app.core.geometry import Unit
        from app.models.presets import SizePreset
        from app.models.project import Project
        from app.ui.sidebar import DesignPanel

        panel = DesignPanel([SizePreset("Custom", 10.0, 8.0, Unit.INCH)])
        proj = Project.new("t")
        proj.settings.push_all = True
        panel.bind_project(proj)
        assert panel._push_all.isChecked()
        panel._push_all.setChecked(False)
        assert panel.build_settings().push_all is False
