"""Shared exporter utilities: errors and safe atomic file writes."""
from __future__ import annotations

import os
from pathlib import Path


class ExportError(Exception):
    """Raised when an export fails (bad output path, disk error, invalid data)."""


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically via a temp file + ``os.replace``.

    The parent directory is created if missing. On any error the partial temp
    file is removed and :class:`ExportError` is raised, so a failed export never
    leaves a half-written destination file.
    """
    path = Path(path)
    tmp = _tmp_path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExportError(f"Could not write {path}: {exc}") from exc
    return path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Atomically write ``text`` (UTF-8 by default) to ``path``."""
    return atomic_write_bytes(path, text.encode(encoding))
