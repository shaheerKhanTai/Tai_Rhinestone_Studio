"""Autosave and crash recovery.

A :class:`QTimer`-driven service periodically writes the current project to a
recovery file in the app-data ``autosave`` directory whenever there are unsaved
changes. On startup the app can offer to recover the most recent autosave.

The autosave file is a normal ``.rhinestone`` archive, so recovery reuses the
standard loader. Recovery files are pruned to a small number of recent copies.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from app.core.image_processor import LoadedImage
from app.models.project import Project
from app.services import project_service
from app.services.logging_config import get_logger
from app.services.paths import autosave_dir

_log = get_logger(__name__)

_RECOVERY_NAME = "recovery.rhinestone"
_KEEP_BACKUPS = 2


class AutosaveService(QObject):
    """Periodically persists a recovery copy of the active project."""

    autosaved = Signal(Path)
    autosave_failed = Signal(str)

    def __init__(self, interval_sec: int = 120, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._interval_ms = max(15, interval_sec) * 1000
        self._provider = None  # callable -> (Project|None, LoadedImage|None, dirty: bool)
        self._enabled = True

    def configure(self, *, enabled: bool, interval_sec: int) -> None:
        self._enabled = enabled
        self._interval_ms = max(15, interval_sec) * 1000
        if self._timer.isActive():
            self._timer.start(self._interval_ms)

    def set_provider(self, provider) -> None:
        """Register a callable returning ``(project, image, dirty)``."""
        self._provider = provider

    def start(self) -> None:
        if self._enabled:
            self._timer.start(self._interval_ms)

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        if not self._enabled or self._provider is None:
            return
        try:
            project, image, dirty = self._provider()
        except Exception:  # pragma: no cover - provider must never crash autosave
            return
        if project is None or not dirty or project.design.stone_count == 0:
            return
        self.save_now(project, image)

    def save_now(self, project: Project, image: LoadedImage | None) -> Path | None:
        """Write a recovery copy immediately (also used on graceful shutdown)."""
        try:
            target = autosave_dir() / _RECOVERY_NAME
            self._rotate(target)
            path = project_service.save_project(project, target, image)
            self.autosaved.emit(path)
            return path
        except Exception as exc:
            _log.warning("Autosave failed: %s", exc)
            self.autosave_failed.emit(str(exc))
            return None

    def _rotate(self, target: Path) -> None:
        """Keep a couple of previous recovery copies before overwriting."""
        if not target.exists():
            return
        try:
            for i in range(_KEEP_BACKUPS, 0, -1):
                older = target.with_suffix(f".{i}.rhinestone")
                newer = target if i == 1 else target.with_suffix(f".{i - 1}.rhinestone")
                if newer.exists():
                    older.unlink(missing_ok=True)
                    newer.replace(older)
        except OSError:
            pass

    @staticmethod
    def recovery_file() -> Path:
        return autosave_dir() / _RECOVERY_NAME

    @classmethod
    def has_recovery(cls) -> bool:
        return cls.recovery_file().exists()

    @classmethod
    def clear_recovery(cls) -> None:
        try:
            cls.recovery_file().unlink(missing_ok=True)
        except OSError:
            pass
