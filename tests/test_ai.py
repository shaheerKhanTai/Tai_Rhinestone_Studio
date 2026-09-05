"""Tests for the AI assistant layer (ISSUE 6).

The contract under test:

* providers return STRUCTURED operations (never code);
* every operation is geometry-validated — spacing, bounds, existence — and
  invalid ones are rejected with a reason, never applied;
* preview changes nothing in the design;
* Apply becomes exactly ONE undoable command (a single Ctrl+Z reverts the
  whole suggestion);
* Reject changes nothing;
* the shipped provider is honestly labelled and runs offline.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.ai.assistant import AIAssistant
from app.ai.local_provider import LocalRuleProvider
from app.ai.operations import AISuggestion, OpKind, StoneOp
from app.ai.validator import validate_suggestion
from app.core.collision import find_collisions
from app.core.history import CommandStack
from app.models.design import Design

D = 3.0
GAP = 0.5
PITCH = D + GAP


def design_with_row(n=5, start=20.0, y=50.0) -> Design:
    d = Design(200.0, 100.0)
    for i in range(n):
        d.add_stone(start + i * PITCH, y, D)
    return d


class TestStructuredOperations:
    def test_provider_returns_structured_ops(self):
        p = LocalRuleProvider()
        d = design_with_row()
        from app.ai.provider import AIRequest

        req = AIRequest(
            command="clean up this region",
            design_width_mm=d.width_mm, design_height_mm=d.height_mm,
            stone_ids=d.ids(), stone_positions_mm=d.positions(),
            stone_diameters_mm=d.diameters(), gap_mm=GAP, stone_diameter_mm=D,
        )
        s = p.suggest(req)
        assert isinstance(s.operations, list)
        for op in s.operations:
            assert isinstance(op, StoneOp)
            assert isinstance(op.kind, OpKind)

    def test_provider_label_is_honest(self):
        # The shipped provider must not claim to be an AI/ML model.
        label = LocalRuleProvider().label.lower()
        assert "rule-based" in label
        assert "offline" in label
        assert "ai" not in label.split()

    def test_registry_has_local_only_by_default(self):
        from app.ai.provider import provider_registry

        labels = [p.label for p in provider_registry.available()]
        assert any("Local" in l for l in labels)
        # No cloud provider ships or registers itself.
        assert not any("cloud" in l.lower() or "gpt" in l.lower() for l in labels)


class TestValidation:
    def _validate(self, design, ops):
        return validate_suggestion(
            AISuggestion(provider="test", description="t", operations=list(ops)),
            design, GAP,
        )

    def test_valid_add_accepted(self):
        d = design_with_row()
        res = self._validate(d, [StoneOp(OpKind.ADD_STONE, x=100.0, y=50.0, diameter=D)])
        assert len(res.valid_ops) == 1
        assert not res.rejected

    def test_add_on_occupied_spot_rejected(self):
        d = design_with_row()
        res = self._validate(d, [StoneOp(OpKind.ADD_STONE, x=20.0, y=50.0, diameter=D)])
        assert res.is_empty
        assert "spacing" in res.rejected[0][1]

    def test_add_outside_bounds_rejected(self):
        d = design_with_row()
        res = self._validate(d, [StoneOp(OpKind.ADD_STONE, x=500.0, y=50.0, diameter=D)])
        assert res.is_empty
        assert "bounds" in res.rejected[0][1]

    def test_move_into_collision_rejected(self):
        d = design_with_row(2)
        target_x = d.stones[0].x + PITCH * 0.5  # halfway to stone 1 → collision
        res = self._validate(
            d, [StoneOp(OpKind.MOVE_STONE, stone_id=d.stones[1].id, x=target_x, y=50.0)]
        )
        assert res.is_empty
        assert "spacing" in res.rejected[0][1]

    def test_move_to_valid_spot_accepted(self):
        d = design_with_row(2)
        res = self._validate(
            d, [StoneOp(OpKind.MOVE_STONE, stone_id=d.stones[1].id, x=100.0, y=80.0)]
        )
        assert len(res.valid_ops) == 1

    def test_move_outside_bounds_rejected(self):
        d = design_with_row(1)
        res = self._validate(
            d, [StoneOp(OpKind.MOVE_STONE, stone_id=d.stones[0].id, x=-10.0, y=50.0)]
        )
        assert res.is_empty
        assert "bounds" in res.rejected[0][1]

    def test_delete_unknown_id_rejected(self):
        d = design_with_row(1)
        res = self._validate(d, [StoneOp(OpKind.DELETE_STONE, stone_id=999)])
        assert res.is_empty

    def test_delete_existing_accepted(self):
        d = design_with_row(1)
        res = self._validate(d, [StoneOp(OpKind.DELETE_STONE, stone_id=d.stones[0].id)])
        assert len(res.valid_ops) == 1

    def test_move_group_valid(self):
        d = design_with_row(3)
        ids = tuple(s.id for s in d.stones)
        res = self._validate(d, [StoneOp(OpKind.MOVE_GROUP, stone_ids=ids, dx=50.0, dy=0.0)])
        assert len(res.valid_ops) == 1

    def test_move_group_outside_bounds_rejected(self):
        d = design_with_row(3)
        ids = tuple(s.id for s in d.stones)
        res = self._validate(d, [StoneOp(OpKind.MOVE_GROUP, stone_ids=ids, dx=500.0, dy=0.0)])
        assert res.is_empty

    def test_regional_op_requires_expansion(self):
        d = design_with_row()
        res = self._validate(d, [StoneOp(OpKind.ADJUST_DENSITY, region=(0, 0, 50, 50))])
        assert res.is_empty
        assert "expanded" in res.rejected[0][1]

    def test_mixed_ops_split_correctly(self):
        d = design_with_row(2)
        ops = [
            StoneOp(OpKind.ADD_STONE, x=100.0, y=50.0, diameter=D),  # valid
            StoneOp(OpKind.ADD_STONE, x=20.0, y=50.0, diameter=D),   # collision
            StoneOp(OpKind.DELETE_STONE, stone_id=d.stones[0].id),   # valid
        ]
        res = self._validate(d, ops)
        assert len(res.valid_ops) == 2
        assert len(res.rejected) == 1


class TestAssistantPipeline:
    def test_preview_changes_nothing(self):
        d = design_with_row(5)
        before = d.positions().copy()
        assistant = AIAssistant()
        v = assistant.preview(d, "clean up this region", gap_mm=GAP, stone_diameter_mm=D)
        assert v is not None
        np.testing.assert_array_equal(d.positions(), before)
        assert d.stone_count == 5

    def test_densify_adds_valid_stones(self):
        # A design with a big empty area + high density → the suggestion adds
        # stones only in valid spots.
        d = design_with_row(2)
        density = np.zeros((100, 200), dtype=np.float32)
        density[:, 100:] = 1.0  # dense right half (empty of stones)
        assistant = AIAssistant()
        v = assistant.preview(d, "add more stones", gap_mm=GAP,
                              stone_diameter_mm=D, density=density)
        adds = [op for op in v.valid_ops if op.kind is OpKind.ADD_STONE]
        assert len(adds) > 5
        # Preview did not mutate the design.
        assert d.stone_count == 2
        # All proposed adds sit in the dense right half.
        assert all(op.x >= 100.0 for op in adds)

    def test_apply_densify_respects_spacing_and_single_undo(self):
        d = design_with_row(2)
        density = np.zeros((100, 200), dtype=np.float32)
        density[:, 100:] = 1.0
        assistant = AIAssistant()
        v = assistant.preview(d, "add more stones", gap_mm=GAP,
                              stone_diameter_mm=D, density=density)
        stack = CommandStack()
        cmd = assistant.apply(v, d)
        assert cmd is not None
        stack.do(cmd)  # apply
        added = d.stone_count - 2
        assert added > 5
        # No spacing violations after apply.
        pairs = find_collisions(d.positions(), d.diameters(), GAP)
        assert pairs == []
        # ONE undo reverts the whole suggestion.
        assert stack.undo()
        assert d.stone_count == 2
        assert stack.can_redo
        assert stack.redo()
        assert d.stone_count == 2 + added

    def test_thin_removes_stones(self):
        d = design_with_row(6)
        assistant = AIAssistant()
        v = assistant.preview(d, "reduce the number of stones",
                              gap_mm=GAP, stone_diameter_mm=D)
        deletes = [op for op in v.valid_ops if op.kind is OpKind.DELETE_STONE]
        assert len(deletes) >= 1
        assert len(deletes) < 6  # a third, not everything
        # Preview changed nothing.
        assert d.stone_count == 6

    def test_move_away_proposes_concrete_moves(self):
        d = Design(200.0, 100.0)
        for i in range(4):
            d.add_stone(50.0 + i * PITCH, 50.0, D)
        assistant = AIAssistant()
        region = (45.0, 45.0, 10.0 + 3 * PITCH, 10.0)  # covers the row
        v = assistant.preview(d, "move stones away from this region",
                              gap_mm=GAP, stone_diameter_mm=D, region=region)
        moves = [op for op in v.valid_ops if op.kind is OpKind.MOVE_STONE]
        assert len(moves) == 4
        # Every proposed target is a valid, spaced position.
        for op in moves:
            assert op.x is not None and op.y is not None

    def test_delete_region_deletes_only_inside(self):
        d = design_with_row(4)  # stones at 20, 23.5, 27, 30.5
        d.add_stone(150.0, 50.0, D)  # far outside the region
        assistant = AIAssistant()
        region = (0.0, 0.0, 60.0, 100.0)
        v = assistant.preview(d, "delete stones in this region",
                              gap_mm=GAP, stone_diameter_mm=D, region=region)
        deletes = [op for op in v.valid_ops if op.kind is OpKind.DELETE_STONE]
        assert len(deletes) == 4  # the row, not the far stone
        far_id = d.stones[-1].id
        assert all(op.stone_id != far_id for op in deletes)

    def test_apply_reject_leaves_design_untouched(self):
        d = design_with_row(3)
        before = d.positions().copy()
        assistant = AIAssistant()
        # "Reject" = preview then discard: nothing was ever applied.
        assistant.preview(d, "reduce stones", gap_mm=GAP, stone_diameter_mm=D)
        np.testing.assert_array_equal(d.positions(), before)

    def test_empty_suggestion_apply_returns_none(self):
        d = design_with_row(1)
        assistant = AIAssistant()
        v = assistant.preview(d, "delete stones", gap_mm=GAP, stone_diameter_mm=D,
                              region=(150.0, 0.0, 10.0, 100.0))  # empty region
        # Either nothing inside (no ops) or valid deletes — apply handles both.
        cmd = assistant.apply(v, d)
        if v.is_empty:
            assert cmd is None

    def test_operations_capped(self):
        from app.ai.assistant import MAX_OPERATIONS

        d = Design(500.0, 500.0)
        # A huge region would generate a huge lattice; the cap bounds it.
        from app.ai.provider import AIRequest

        req = AIRequest(
            command="fill", design_width_mm=500.0, design_height_mm=500.0,
            stone_ids=np.empty(0, dtype=np.int64),
            stone_positions_mm=np.empty((0, 2)), stone_diameters_mm=np.empty(0),
            gap_mm=0.0, stone_diameter_mm=1.0,  # 1mm pitch → ~250k candidates
        )
        assistant = AIAssistant()
        suggestion = AISuggestion(
            provider="t", description="t",
            operations=[StoneOp(OpKind.ADJUST_DENSITY, direction=1.0)],
        )
        expanded = assistant.expand_regional_ops(suggestion, req)
        assert len(expanded.operations) <= MAX_OPERATIONS


class TestWindowIntegration:
    def test_apply_via_window_and_undo(self, qapp, tmp_path):
        import cv2

        from app.models.app_settings import AppSettings
        from app.ui.main_window import MainWindow

        p = tmp_path / "img.png"
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        cv2.circle(rgba, (100, 100), 70, (20, 20, 25, 255), -1)
        cv2.imwrite(str(p), rgba)

        w = MainWindow(AppSettings())
        try:
            w._import_image_path(p)
            proj = w.state.project
            # A few stones to work on.
            for i in range(4):
                proj.design.add_stone(30.0 + i * PITCH, 50.0, D)
            before = proj.design.stone_count

            assistant = AIAssistant()
            validated = assistant.preview(
                proj.design, "reduce the number of stones",
                gap_mm=GAP, stone_diameter_mm=D,
            )
            assert validated is not None and not validated.is_empty
            w._on_ai_applied(validated)
            assert proj.design.stone_count != before

            # ONE undo reverts the entire AI operation.
            assert w._command_stack.undo()
            assert proj.design.stone_count == before
            # Redo re-applies it.
            assert w._command_stack.redo()
            assert proj.design.stone_count != before
        finally:
            w.state.dirty = False
            w.close()
            w.deleteLater()

    def test_ai_action_requires_design(self, qapp):
        from app.models.app_settings import AppSettings
        from app.ui.main_window import MainWindow

        w = MainWindow(AppSettings())
        try:
            w.action_new_project()
            assert not w.act_ai.isEnabled()
        finally:
            w.state.dirty = False
            w.close()
            w.deleteLater()


class TestDialog:
    def test_dialog_flow(self, qapp):
        from PySide6.QtWidgets import QDialog

        from app.ui.canvas import CanvasView
        from app.ui.dialogs.ai_dialog import AIAssistantDialog
        from app.ui.theme import DARK_THEME

        c = CanvasView(DARK_THEME)
        c.resize(800, 600)
        d = design_with_row(5)
        c.set_design(d, fit=False)
        assistant = AIAssistant()

        dlg = AIAssistantDialog(
            assistant, d, c, gap_mm=GAP, stone_diameter_mm=D, parent=None,
        )
        applied = []
        dlg.applied.connect(lambda v: applied.append(v))
        rejected = []
        dlg.discard_requested.connect(lambda: rejected.append(True))

        # Suggest → preview painted, design untouched.
        dlg._command.setText("reduce the number of stones")
        dlg._suggest()
        assert d.stone_count == 5
        assert dlg._apply_btn.isEnabled()

        # Reject → nothing applied, preview cleared.
        dlg._reject()
        assert applied == []
        assert rejected == [True]
        assert dlg.result() == QDialog.Rejected
        assert c._ai_preview_adds.shape[0] == 0
        assert c._ai_preview_moves == []
        c.deleteLater()

    def test_dialog_apply_emits_validated(self, qapp):
        from app.ui.canvas import CanvasView
        from app.ui.dialogs.ai_dialog import AIAssistantDialog
        from app.ui.theme import DARK_THEME

        c = CanvasView(DARK_THEME)
        c.resize(800, 600)
        d = design_with_row(5)
        c.set_design(d, fit=False)
        assistant = AIAssistant()
        dlg = AIAssistantDialog(assistant, d, c, gap_mm=GAP, stone_diameter_mm=D)
        applied = []
        dlg.applied.connect(lambda v: applied.append(v))
        dlg._command.setText("reduce the number of stones")
        dlg._suggest()
        dlg._apply()
        assert len(applied) == 1
        assert applied[0].valid_ops  # a non-empty validated suggestion
        c.deleteLater()

    def test_canvas_ai_preview_paints(self, qapp):
        from app.ui.canvas import CanvasView
        from app.ui.theme import DARK_THEME

        c = CanvasView(DARK_THEME)
        c.resize(800, 600)
        d = Design(100.0, 100.0)
        c.set_design(d, fit=False)
        c.set_ai_preview(np.array([[10.0, 10.0], [20.0, 20.0]]),
                         [(30.0, 30.0, 40.0, 40.0)], diameter=3.0)
        pm = c.grab()
        assert not pm.isNull()
        c.set_ai_preview(None, None)
        assert c._ai_preview_adds.shape[0] == 0
        c.deleteLater()
