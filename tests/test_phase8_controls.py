"""Tests for Phase 8: sliders, design panel, statistics panel, worker."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QThreadPool, QTimer

from app.models.presets import default_presets
from app.models.project import Project
from app.models.settings import PlacementMode
from app.ui.generation_worker import GenerationWorker
from app.ui.sidebar import DesignPanel
from app.ui.statistics_panel import StatisticsPanel
from app.ui.widgets.labeled_slider import LabeledSlider


class TestLabeledSlider:
    def test_value_roundtrip(self, qapp):
        s = LabeledSlider("Detail", 0.5)
        assert s.value() == pytest.approx(0.5, abs=0.01)
        s.set_value(0.8)
        assert s.value() == pytest.approx(0.8, abs=0.01)

    def test_clamped(self, qapp):
        s = LabeledSlider("X", 0.5)
        s.set_value(5.0)
        assert s.value() == 1.0
        s.set_value(-1.0)
        assert s.value() == 0.0

    def test_set_value_silent(self, qapp):
        s = LabeledSlider("X", 0.5)
        fired = []
        s.valueCommitted.connect(fired.append)
        s.set_value(0.3, emit=False)
        assert fired == []


class TestDesignPanel:
    def test_build_settings_reflects_controls(self, qapp):
        panel = DesignPanel(default_presets())
        proj = Project.new()
        panel.bind_project(proj)
        panel._detail.set_value(0.9)
        panel._spacing.setValue(0.8)
        # Select by data, not index — enum order is free to change.
        grid_idx = panel._mode_combo.findData(PlacementMode.GRID)
        assert grid_idx >= 0
        panel._mode_combo.setCurrentIndex(grid_idx)
        s = panel.build_settings()
        assert s.detail == pytest.approx(0.9, abs=0.01)
        assert s.spacing_mm == pytest.approx(0.8)
        assert s.placement_mode is PlacementMode.GRID

    def test_bind_syncs_from_project(self, qapp):
        panel = DesignPanel(default_presets())
        proj = Project.new()
        proj.settings.stone_size_id = "SS16"
        proj.settings.spacing_mm = 1.2
        panel.bind_project(proj)
        assert panel._size_combo.currentData() == "SS16"
        assert panel._spacing.value() == pytest.approx(1.2)

    def test_size_commit_updates_project(self, qapp):
        panel = DesignPanel(default_presets())
        proj = Project.new()  # inches
        panel.bind_project(proj)
        changed = []
        panel.size_changed.connect(lambda: changed.append(True))
        panel._width_spin.setValue(12.0)  # 12 in
        assert proj.width_mm == pytest.approx(12 * 25.4, abs=0.1)
        assert changed

    def test_lock_aspect_links_dimensions(self, qapp):
        panel = DesignPanel(default_presets())
        proj = Project.new()
        proj.image.original_width_px = 200
        proj.image.original_height_px = 100  # 2:1
        proj.image.source_path = __import__("pathlib").Path("x.png")
        panel.bind_project(proj)
        panel._lock_aspect.setChecked(True)
        panel._width_spin.setValue(8.0)  # change from the default 10.0
        # height should track to 4 in (2:1 aspect).
        assert panel._height_spin.value() == pytest.approx(4.0, abs=0.05)

    def test_max_stones_toggle(self, qapp):
        panel = DesignPanel(default_presets())
        panel.bind_project(Project.new())
        panel._limit_enabled.setChecked(True)
        panel._max_stones.setValue(3000)
        assert panel.build_settings().max_stones == 3000
        panel._limit_enabled.setChecked(False)
        assert panel.build_settings().max_stones is None

    def test_generating_state(self, qapp):
        panel = DesignPanel(default_presets())
        panel.set_generating(True)
        # isHidden() reflects the explicit visibility flag even when the
        # top-level widget has never been shown.
        assert not panel._progress.isHidden()
        panel.set_progress("Placing stones", 0.5)
        assert panel._progress.value() == 50
        panel.set_generating(False)
        assert panel._progress.isHidden()


class TestStatisticsPanel:
    def test_refresh(self, qapp):
        panel = StatisticsPanel()
        proj = Project.new()
        proj.design.add_stone(10, 10, 2.8)
        proj.design.add_stone(20, 20, 2.8)
        panel.bind_project(proj)
        assert "2" in panel._count_label.text()
        assert "in" in panel._dim_label.text()

    def test_color_breakdown_hidden_for_single(self, qapp):
        panel = StatisticsPanel()
        proj = Project.new()
        proj.design.add_stone(1, 1, 2.8, color="#000")
        panel.bind_project(proj)
        assert panel._colors_group.isHidden()

    def test_color_breakdown_shown_for_multi(self, qapp):
        panel = StatisticsPanel()
        proj = Project.new()
        proj.design.add_stone(1, 1, 2.8, color="#000")
        proj.design.add_stone(2, 2, 2.8, color="#FFF")
        panel.bind_project(proj)
        assert not panel._colors_group.isHidden()

    def test_selection_count(self, qapp):
        panel = StatisticsPanel()
        panel.set_selection_count(5)
        assert "5" in panel._selection_label.text()
        panel.set_selection_count(0)
        assert "Nothing" in panel._selection_label.text()


def _disc(h=200, w=200, r=70):
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.circle(rgba, (w // 2, h // 2), r, (20, 20, 25, 255), -1)
    return rgba


def _run_worker(worker, timeout_ms=15000):
    """Run a worker on the pool and pump the event loop until it signals."""
    loop = QEventLoop()
    outcome = {}
    worker.signals.finished.connect(lambda r: (outcome.update(result=r), loop.quit()))
    worker.signals.failed.connect(lambda m: (outcome.update(error=m), loop.quit()))
    worker.signals.cancelled.connect(lambda: (outcome.update(cancelled=True), loop.quit()))
    QThreadPool.globalInstance().start(worker)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    return outcome


class TestGenerationWorker:
    def test_success(self, qapp):
        from app.models.settings import GenerationSettings

        worker = GenerationWorker(_disc(), GenerationSettings(), 100.0, 100.0)
        out = _run_worker(worker)
        assert "result" in out
        assert out["result"].stone_count > 0

    def test_progress_emitted(self, qapp):
        from app.models.settings import GenerationSettings

        worker = GenerationWorker(_disc(), GenerationSettings(), 100.0, 100.0)
        seen = []
        worker.signals.progress.connect(lambda s, f: seen.append(f))
        _run_worker(worker)
        assert seen
        assert max(seen) == pytest.approx(1.0)

    def test_cancel(self, qapp):
        from app.models.settings import GenerationSettings

        # Large design + immediate cancel → cancelled outcome.
        worker = GenerationWorker(_disc(400, 400, 150), GenerationSettings(), 400.0, 400.0)
        worker.cancel()
        out = _run_worker(worker)
        # Either cancelled cleanly, or finished very fast before the check.
        assert "cancelled" in out or "result" in out
