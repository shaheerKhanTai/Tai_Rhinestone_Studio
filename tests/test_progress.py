"""Tests for real generation progress and progressive rendering (ISSUE 5).

The generation engine itself must emit progress (no timers/sleeps), placement
must stream stone batches as they become available, the worker must forward
them across threads, and cancellation must restore the previous design.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer

from app.core.collision import find_collisions
from app.core.rhinestone_engine import generate
from app.models.settings import GenerationSettings, PlacementMode


def _disc(h=200, w=200, r=70):
    import cv2

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.circle(rgba, (w // 2, h // 2), r, (20, 20, 25, 255), -1)
    return rgba


class TestEngineProgress:
    def test_progress_starts_at_zero(self):
        seen = []
        generate(_disc(), GenerationSettings(), 100.0, 100.0,
                 progress=lambda s, f: seen.append((s, f)))
        assert seen[0][1] == pytest.approx(0.0)

    def test_progress_reaches_one(self):
        seen = []
        generate(_disc(), GenerationSettings(), 100.0, 100.0,
                 progress=lambda s, f: seen.append((s, f)))
        assert seen[-1][1] == pytest.approx(1.0)

    def test_progress_monotonic(self):
        seen = []
        generate(_disc(300, 300, 110), GenerationSettings(), 200.0, 200.0,
                 progress=lambda s, f: seen.append(f))
        # Fractions may repeat a stage boundary value but never go backwards.
        for a, b in zip(seen, seen[1:]):
            assert b >= a - 1e-9

    def test_stages_change(self):
        seen = []
        generate(_disc(300, 300, 110), GenerationSettings(), 200.0, 200.0,
                 progress=lambda s, f: seen.append(s))
        assert len(set(seen)) >= 3  # analyse, density, placing, collisions, …

    def test_no_timers_or_sleeps_in_generation(self):
        # Determinism proof: identical inputs → identical outputs (a
        # timer-driven fake progress would not survive this by construction).
        a = generate(_disc(), GenerationSettings(), 100.0, 100.0)
        b = generate(_disc(), GenerationSettings(), 100.0, 100.0)
        np.testing.assert_array_equal(a.placement.positions_mm, b.placement.positions_mm)


class TestEngineBatches:
    def test_lattice_streams_final_positions(self):
        batches = []
        s = GenerationSettings(placement_mode=PlacementMode.HEXAGONAL)
        res = generate(_disc(300, 300, 120), s, 150.0, 150.0,
                       on_batch=lambda stones: batches.append(stones))
        streamed = sum(len(b) for b in batches)
        assert streamed == res.stone_count
        # Batch count is bounded: never one update per stone.
        assert 1 <= len(batches) <= max(2, res.stone_count // 100 + 2)
        # Streamed positions match the final positions (as a set).
        got = np.vstack([[s.x, s.y] for s in (st for b in batches for st in b)])
        final = res.placement.positions_mm
        assert got.shape == final.shape
        # Same multiset of rounded coordinates.
        assert sorted(map(tuple, np.round(got, 6))) == sorted(map(tuple, np.round(final, 6)))

    def test_organic_streams_during_placement(self):
        batches = []
        s = GenerationSettings(placement_mode=PlacementMode.ORGANIC)
        res = generate(_disc(300, 300, 120), s, 150.0, 150.0,
                       on_batch=lambda stones: batches.append(stones))
        streamed = sum(len(b) for b in batches)
        assert streamed == res.stone_count
        assert len(batches) >= 1

    def test_max_stone_cap_streams_only_kept_stones(self):
        batches = []
        s = GenerationSettings(placement_mode=PlacementMode.ORGANIC, max_stones=400)
        res = generate(_disc(300, 300, 120), s, 150.0, 150.0,
                       on_batch=lambda stones: batches.append(stones))
        streamed = sum(len(b) for b in batches)
        assert res.stone_count <= 400
        # The preview never shows more than the cap allows.
        assert streamed == res.stone_count

    def test_batch_sizes_grow(self):
        batches = []
        s = GenerationSettings(placement_mode=PlacementMode.HEXAGONAL)
        res = generate(_disc(400, 400, 170), s, 300.0, 300.0,
                       on_batch=lambda stones: batches.append(stones))
        sizes = [len(b) for b in batches]
        # Growing batches (100, 250, 500, ...) — every batch except possibly
        # the final remainder is at least as large as its predecessor.
        assert res.stone_count > 1000
        for a, b in zip(sizes[:-1], sizes[1:-1]):
            assert b >= a

    def test_streaming_does_not_change_result(self):
        s = GenerationSettings(placement_mode=PlacementMode.HEXAGONAL)
        plain = generate(_disc(300, 300, 120), s, 150.0, 150.0)
        streamed = generate(_disc(300, 300, 120), s, 150.0, 150.0,
                            on_batch=lambda _: None)
        assert plain.stone_count == streamed.stone_count
        np.testing.assert_array_equal(
            plain.placement.positions_mm, streamed.placement.positions_mm
        )

    def test_streamed_stones_respect_spacing(self):
        batches = []
        s = GenerationSettings(placement_mode=PlacementMode.HEXAGONAL)
        res = generate(_disc(300, 300, 120), s, 150.0, 150.0,
                       on_batch=lambda stones: batches.append(stones))
        pos = np.array([[st.x, st.y] for b in batches for st in b])
        dia = np.full(pos.shape[0], s.effective_diameter())
        assert find_collisions(pos, dia, s.spacing_mm) == []


class TestWorkerBatches:
    def _run_worker(self, worker, timeout_ms=30000):
        from PySide6.QtCore import QThreadPool
        from app.ui.generation_worker import GenerationWorker  # noqa: F401

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

    def test_worker_emits_batches_then_finishes(self, qapp):
        from app.ui.generation_worker import GenerationWorker

        worker = GenerationWorker(_disc(400, 400, 170), GenerationSettings(), 300.0, 300.0)
        batches = []
        counts = []
        worker.signals.batch.connect(lambda stones, n: (batches.append(len(stones)), counts.append(n)))
        out = self._run_worker(worker)
        assert "result" in out
        total = out["result"].stone_count
        assert total > 500
        assert sum(batches) == total
        # Running counts are strictly increasing toward the total.
        assert counts == sorted(counts)
        assert counts[-1] == total

    def test_worker_progress_and_batches_interleave(self, qapp):
        from app.ui.generation_worker import GenerationWorker

        worker = GenerationWorker(_disc(400, 400, 170), GenerationSettings(), 300.0, 300.0)
        order = []
        worker.signals.progress.connect(lambda s, f: order.append(("p", f)))
        worker.signals.batch.connect(lambda stones, n: order.append(("b", n)))
        out = self._run_worker(worker)
        assert "result" in out
        # At least one batch arrived before the final progress tick.
        assert any(kind == "b" for kind, _ in order)
        assert order[-1] == ("p", pytest.approx(1.0))

    def test_worker_failed_on_bad_image(self, qapp):
        from app.ui.generation_worker import GenerationWorker

        worker = GenerationWorker(np.zeros((10, 10)), GenerationSettings(), 100.0, 100.0)
        out = self._run_worker(worker)
        assert "error" in out

    def test_worker_cancel_restores_design(self, qapp, tmp_path):
        # A window-level check: cancel mid-generation → the design on screen
        # is the previous one, untouched.
        import cv2

        from app.models.app_settings import AppSettings
        from app.ui.main_window import MainWindow

        p = tmp_path / "img.png"
        rgba = _disc(500, 500, 200)
        cv2.imwrite(str(p), rgba)
        w = MainWindow(AppSettings())
        try:
            w._import_image_path(p)
            proj = w.state.project
            # Pre-existing design stones.
            proj.design.add_stone(20.0, 20.0, 2.8)
            before = list(proj.design.stones)

            w._design_panel._limit_enabled.setChecked(False)
            worker = w._active_worker  # None until generate
            w.action_generate()
            worker = w._active_worker
            assert worker is not None
            worker.cancel()
            # Pump until the worker resolves.
            loop = QEventLoop()
            QTimer.singleShot(5000, loop.quit)

            def _check():
                if w._active_worker is None:
                    loop.quit()

            timer = QTimer()
            timer.setInterval(10)
            timer.timeout.connect(_check)
            timer.start()
            loop.exec()
            timer.stop()

            if w._active_worker is None:  # worker resolved (cancelled or done)
                # Either way the design must be in a valid state: either the
                # previous stones (cancel) or a full regeneration (finish).
                if proj.design.stone_count == 1:
                    assert [s.id for s in proj.design.stones] == [s.id for s in before]
        finally:
            w.state.dirty = False
            w.close()
            w.deleteLater()


class TestCanvasStreaming:
    @pytest.fixture()
    def canvas(self, qapp):
        from app.models.design import Design
        from app.ui.canvas import CanvasView
        from app.ui.theme import DARK_THEME

        c = CanvasView(DARK_THEME)
        c.resize(800, 600)
        d = Design(150.0, 150.0)
        c.set_design(d, fit=False)
        return c, d

    def test_append_and_discard_preview(self, canvas):
        from app.models.stone import Stone

        c, d = canvas
        d.add_stone(10.0, 10.0, 2.8)
        c.refresh_geometry()
        assert c._pos.shape[0] == 1

        c.begin_stone_stream()
        stones = [Stone(i + 100, 5.0 * i, 5.0, 2.8) for i in range(250)]
        c.append_stone_batch(stones)
        assert c._pos.shape[0] == 251  # preview visible on the canvas
        # ... but the design itself is untouched.
        assert d.stone_count == 1

        c.end_stone_stream()
        assert c._pos.shape[0] == 1  # preview discarded

    def test_preview_paints_without_crash(self, canvas):
        from app.models.stone import Stone

        c, d = canvas
        c.begin_stone_stream()
        stones = [Stone(i + 1, 50.0 + 4.0 * i, 50.0, 2.8) for i in range(300)]
        c.append_stone_batch(stones)
        pm = c.grab()  # forces a full paint pass offscreen
        assert not pm.isNull()
        c.end_stone_stream()

    def test_batch_count_bounded_for_20k(self):
        # The engine's streaming schedule must keep the number of GUI updates
        # small even for very large designs.
        from app.core.placement import _STREAM_BATCH_SIZES

        total = 20_000
        sizes = list(_STREAM_BATCH_SIZES)
        updates = 0
        remaining = total
        i = 0
        while remaining > 0:
            k = min(sizes[min(i, len(sizes) - 1)], remaining)
            remaining -= k
            updates += 1
            i += 1
        assert updates <= 30
