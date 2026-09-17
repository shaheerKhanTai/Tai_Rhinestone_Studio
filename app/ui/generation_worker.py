"""Background generation worker — keeps the UI responsive during generation.

Runs :func:`app.core.rhinestone_engine.generate` on a :class:`QThreadPool`
worker thread and marshals progress, completion, cancellation and errors back
to the GUI thread via signals. The engine's cooperative cancel hook lets the
user abort a long run. The worker copies the settings and holds a reference to
the (immutable-during-run) source image, so the GUI may keep editing metadata.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from app.core.placement import PlacementCancelled
from app.core.rhinestone_engine import GenerationResult, generate
from app.models.settings import GenerationSettings
from app.services.logging_config import get_logger

_log = get_logger(__name__)


class GenerationSignals(QObject):
    """Signals emitted by :class:`GenerationWorker` (owned on the GUI thread)."""

    progress = Signal(str, float)          # stage label, fraction 0..1
    batch = Signal(list, int)              # preview stones, running count (ISSUE 5)
    finished = Signal(object)              # GenerationResult
    cancelled = Signal()
    failed = Signal(str)                   # user-facing error message


class GenerationWorker(QRunnable):
    """A cancellable generation job for :class:`QThreadPool`."""

    def __init__(
        self,
        rgba: np.ndarray,
        settings: GenerationSettings,
        width_mm: float,
        height_mm: float,
        *,
        invert: bool | None = None,
        color: str = "#1F2933",
        analysis_max_dim: int = 1400,
        trace_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.signals = GenerationSignals()
        self._rgba = rgba
        self._settings = settings.copy()
        self._width_mm = width_mm
        self._height_mm = height_mm
        # None → the settings' own tone polarity decides (the normal path).
        self._invert = invert
        self._color = color
        self._analysis_max_dim = analysis_max_dim
        # The project's mask is painted in place on the GUI thread — snapshot
        # it so the worker's copy is stable for the whole run.
        self._trace_mask = (
            np.array(trace_mask, copy=True) if trace_mask is not None else None
        )
        self._cancel = False

    def cancel(self) -> None:
        """Request cooperative cancellation (safe to call from the GUI thread)."""
        self._cancel = True

    def _is_cancelled(self) -> bool:
        return self._cancel

    @Slot()
    def run(self) -> None:  # noqa: D401 - QRunnable entry point
        streamed = [0]  # running count of preview stones emitted so far

        def _on_batch(stones) -> None:
            streamed[0] += len(stones)
            self.signals.batch.emit(list(stones), streamed[0])

        try:
            result: GenerationResult = generate(
                self._rgba,
                self._settings,
                self._width_mm,
                self._height_mm,
                invert=self._invert,
                color=self._color,
                analysis_max_dim=self._analysis_max_dim,
                trace_mask=self._trace_mask,
                progress=self.signals.progress.emit,
                is_cancelled=self._is_cancelled,
                on_batch=_on_batch,
            )
        except PlacementCancelled:
            self.signals.cancelled.emit()
            return
        except MemoryError:
            _log.exception("Out of memory during generation")
            self.signals.failed.emit(
                "The design is too large to generate. Try a larger stone size, "
                "more spacing, or a smaller physical size."
            )
            return
        except Exception as exc:  # pragma: no cover - defensive catch-all
            _log.exception("Generation failed")
            self.signals.failed.emit(f"Generation failed: {exc}")
            return
        self.signals.finished.emit(result)
