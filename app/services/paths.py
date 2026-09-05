"""Per-user application data locations (logs, autosave, settings).

All writable runtime state lives under a single per-user directory — never
inside the installation folder, which may be read-only. On Windows this is
``%LOCALAPPDATA%\\RhinestoneStudio``. The ``RHINESTONE_DATA_DIR`` environment
variable overrides the base directory (used by the test suite so tests never
touch real user data).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_DIRNAME = "RhinestoneStudio"
_ENV_OVERRIDE = "RHINESTONE_DATA_DIR"


def _base_data_dir() -> Path:
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override)
    if sys.platform.startswith("win"):
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root) / _APP_DIRNAME
        return Path.home() / "AppData" / "Local" / _APP_DIRNAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_DIRNAME
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / _APP_DIRNAME


def app_data_dir() -> Path:
    """Return (creating if needed) the per-user application data directory."""
    d = _base_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _subdir(name: str) -> Path:
    d = app_data_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    return _subdir("logs")


def autosave_dir() -> Path:
    return _subdir("autosave")


def presets_file() -> Path:
    return app_data_dir() / "presets.json"


def settings_file() -> Path:
    return app_data_dir() / "settings.json"


def log_file() -> Path:
    return logs_dir() / "rhinestone_studio.log"
