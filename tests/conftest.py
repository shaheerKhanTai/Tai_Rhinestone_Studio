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


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """Record modal dialogs instead of blocking on them.

    A failure dialog *should* be modal in the app, but a test that provokes a
    failure would otherwise sit on it until the test session times out — several
    suites here deliberately generate designs that place no stones, which is
    exactly the case that now reports itself. Stubbing ``exec`` keeps the
    behaviour observable (the dialog is still constructed and its text is
    recorded) without the block; the tests that care about the wording assert on
    it. Static helpers like ``QMessageBox.question`` are deliberately left
    alone — tests that use them depend on the answer they return.
    """
    from PySide6.QtWidgets import QMessageBox

    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: (shown.append(self.text()), 0)[1]
    )
    yield shown


@pytest.fixture(autouse=True)
def _reap_queued_widgets():
    """Destroy the widgets each test queued for deletion.

    The GUI tests build real ``MainWindow``s and call ``deleteLater()`` on
    teardown, but with no event loop running in a test session those deletions
    are only *queued* — the windows stay alive and keep accumulating. Every
    application-wide restyle (``QApplication.setStyleSheet``, which a theme
    change and every window's construction trigger) then walks all of them, so
    a theme-switch test late in a run cost 10.8s where the same file finishes in
    seconds on its own. Flushing the deferred deletes after each test keeps the
    cost linear.
    """
    yield
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
