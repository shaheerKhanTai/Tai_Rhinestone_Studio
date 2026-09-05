"""Load and save :class:`AppSettings` and size presets as JSON.

Reads are fault-tolerant: a missing or corrupt file yields sensible defaults
(logged, never fatal). Writes are atomic.
"""
from __future__ import annotations

import json

from app.exporters.base import atomic_write_text
from app.models.app_settings import AppSettings
from app.models.presets import SizePreset, default_presets
from app.services.logging_config import get_logger
from app.services.paths import presets_file, settings_file

_log = get_logger(__name__)


def load_settings() -> AppSettings:
    """Load application settings, falling back to defaults on any problem."""
    path = settings_file()
    if not path.exists():
        return AppSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AppSettings.from_dict(data)
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("Could not read settings (%s); using defaults", exc)
        return AppSettings()


def save_settings(settings: AppSettings) -> None:
    """Persist application settings atomically."""
    try:
        atomic_write_text(settings_file(), json.dumps(settings.to_dict(), indent=2))
    except Exception as exc:  # pragma: no cover - disk failure
        _log.error("Could not save settings: %s", exc)


def load_presets() -> list[SizePreset]:
    """Load size presets, falling back to the built-in defaults."""
    path = presets_file()
    if not path.exists():
        return default_presets()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        presets = [SizePreset.from_dict(d) for d in data.get("presets", [])]
        return presets or default_presets()
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("Could not read presets (%s); using defaults", exc)
        return default_presets()


def save_presets(presets: list[SizePreset]) -> None:
    """Persist size presets atomically."""
    try:
        payload = {"presets": [p.to_dict() for p in presets]}
        atomic_write_text(presets_file(), json.dumps(payload, indent=2))
    except Exception as exc:  # pragma: no cover - disk failure
        _log.error("Could not save presets: %s", exc)
