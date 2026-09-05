"""Tests for the command/history system (Phase 10)."""
from __future__ import annotations

import pytest

from app.core.history import (
    AddStonesCommand,
    CommandStack,
    DeleteStonesCommand,
    MoveStonesCommand,
    ReplaceDesignCommand,
)
from app.models.design import Design
from app.models.stone import Stone


def design_with(n: int) -> Design:
    d = Design(100, 100)
    for i in range(n):
        d.add_stone(float(i), float(i), 3.0)
    return d


class TestCommandStack:
    def test_do_records(self):
        d = design_with(0)
        stack = CommandStack()
        stack.do(AddStonesCommand(d, [Stone(1, 5, 5, 3.0)]))
        assert d.stone_count == 1
        assert stack.can_undo
        assert not stack.can_redo

    def test_undo_redo(self):
        d = design_with(0)
        stack = CommandStack()
        stack.do(AddStonesCommand(d, [Stone(1, 5, 5, 3.0)]))
        stack.undo()
        assert d.stone_count == 0
        assert stack.can_redo
        stack.redo()
        assert d.stone_count == 1

    def test_new_command_clears_redo(self):
        d = design_with(0)
        stack = CommandStack()
        stack.do(AddStonesCommand(d, [Stone(1, 1, 1, 3.0)]))
        stack.undo()
        assert stack.can_redo
        stack.do(AddStonesCommand(d, [Stone(2, 2, 2, 3.0)]))
        assert not stack.can_redo

    def test_capacity_bounded(self):
        d = design_with(0)
        stack = CommandStack(capacity=3)
        for i in range(10):
            stack.do(AddStonesCommand(d, [Stone(i + 1, i, i, 3.0)]))
        # Only the last 3 are undoable.
        count = 0
        while stack.undo():
            count += 1
        assert count == 3

    def test_empty_undo_redo(self):
        stack = CommandStack()
        assert not stack.undo()
        assert not stack.redo()

    def test_on_change_fires(self):
        d = design_with(0)
        stack = CommandStack()
        calls = []
        stack.on_change = lambda: calls.append(1)
        stack.do(AddStonesCommand(d, [Stone(1, 1, 1, 3.0)]))
        stack.undo()
        assert len(calls) >= 2

    def test_labels(self):
        d = design_with(0)
        stack = CommandStack()
        stack.do(AddStonesCommand(d, [Stone(1, 1, 1, 3.0)]))
        assert "Add" in stack.undo_label

    def test_push_applied_not_reexecuted(self):
        # Simulate an interactive move already applied to the design.
        d = design_with(1)
        stone = d.stones[0]
        stone.x, stone.y = 50, 50  # user already dragged it
        stack = CommandStack()
        cmd = MoveStonesCommand(d, {stone.id: ((0.0, 0.0), (50.0, 50.0))})
        stack.push_applied(cmd)  # should NOT re-apply
        assert (stone.x, stone.y) == (50, 50)
        stack.undo()
        assert (stone.x, stone.y) == (0, 0)


class TestAddDelete:
    def test_delete_undo_restores(self):
        d = design_with(5)
        ids = [s.id for s in d.stones[:2]]
        stack = CommandStack()
        stack.do(DeleteStonesCommand(d, ids))
        assert d.stone_count == 3
        stack.undo()
        assert d.stone_count == 5
        # Restored stones keep their ids.
        assert set(ids).issubset({s.id for s in d.stones})

    def test_delete_redo(self):
        d = design_with(4)
        ids = [d.stones[0].id]
        stack = CommandStack()
        stack.do(DeleteStonesCommand(d, ids))
        stack.undo()
        stack.redo()
        assert d.stone_count == 3


class TestMove:
    def test_move_undo_redo(self):
        d = design_with(2)
        s0 = d.stones[0]
        stack = CommandStack()
        moves = {s0.id: ((s0.x, s0.y), (80.0, 90.0))}
        stack.do(MoveStonesCommand(d, moves))
        assert (s0.x, s0.y) == (80.0, 90.0)
        stack.undo()
        assert (s0.x, s0.y) == (0.0, 0.0)
        stack.redo()
        assert (s0.x, s0.y) == (80.0, 90.0)


class TestReplace:
    def test_replace_undo_restores_old(self):
        d = design_with(3)
        old = list(d.stones)
        new = [Stone(100, 10, 10, 3.0), Stone(101, 20, 20, 3.0)]
        stack = CommandStack()
        stack.do(ReplaceDesignCommand(d, old, new))
        assert d.stone_count == 2
        stack.undo()
        assert d.stone_count == 3
        stack.redo()
        assert d.stone_count == 2
