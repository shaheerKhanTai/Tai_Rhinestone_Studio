"""Application logging setup.

Logs go to a rotating file in the per-user data directory (for customer
troubleshooting) and, in development, to the console. Technical detail is
logged; user-facing dialogs show friendly messages (see the UI error handler).
Sensitive data is not logged — only paths, sizes, counts and error text.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from app import __version__
from app.services.paths import log_file

_CONFIGURED = False
_MAX_BYTES = 2 * 1024 * 1024  # 2 MB per file
_BACKUPS = 3


def configure_logging(level: int = logging.INFO, *, console: bool = True) -> Path:
    """Configure root logging once. Returns the active log file path."""
    global _CONFIGURED
    path = log_file()
    if _CONFIGURED:
        return path

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        # Never let logging setup crash the app (e.g. locked/read-only dir).
        pass

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        root.addHandler(console_handler)

    _CONFIGURED = True
    logging.getLogger(__name__).info("Rhinestone Studio %s logging started", __version__)
    return path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
