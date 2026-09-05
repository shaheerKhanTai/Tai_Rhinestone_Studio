"""Shared pytest fixtures and environment isolation.

Sets Qt to the offscreen platform and redirects application data to a temp
directory *before* any app module imports it, so the test suite never opens a
real window or touches real user settings/logs.
"""
from __future__ import annotations

import os
import tempfile

# Must be set before any PySide6 / app.services.paths import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault(
    "RHINESTONE_DATA_DIR", tempfile.mkdtemp(prefix="rhinestone_test_data_")
)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """A session-wide QApplication for GUI tests (offscreen)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
