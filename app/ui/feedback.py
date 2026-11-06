"""Audible feedback for failures.

A design tool can fail "quietly" in a way that looks like success: the dialog
appears behind the window, or the user is looking at the canvas rather than the
status bar. The error tone is a second channel for the same message, so a run
that produced nothing cannot be mistaken for one that worked.

Sound is best-effort by design — it is a *reinforcement* of an on-screen
message, never the message itself, so every failure path here is swallowed. A
machine with no audio device, a locked-down session, or a headless test run
must never be broken by trying to beep.
"""
from __future__ import annotations

import os

#: ``winsound`` exists only on Windows; imported lazily so the module loads
#: everywhere.
_WINSOUND_AVAILABLE = False
try:  # pragma: no cover - platform dependent
    import winsound

    _WINSOUND_AVAILABLE = True
except ImportError:  # pragma: no cover - non-Windows
    winsound = None  # type: ignore[assignment]


def _is_headless() -> bool:
    """True when running under an offscreen/test Qt platform.

    Tests drive the real UI objects, so error paths are exercised routinely;
    beeping the developer's machine on every one of them would be noise, not
    feedback.
    """
    return os.environ.get("QT_QPA_PLATFORM", "").lower() in ("offscreen", "minimal")


def play_error_sound() -> None:
    """Play the platform's error tone. Never raises.

    On Windows this is the real shell error sound (``MB_ICONHAND``) — the same
    one the system uses for a failed operation, which users already recognise.
    Elsewhere it falls back to Qt's own bell, and if that is unavailable too
    the call simply does nothing.
    """
    if _is_headless():
        return
    if _WINSOUND_AVAILABLE:
        try:  # pragma: no cover - platform dependent
            winsound.MessageBeep(winsound.MB_ICONHAND)
            return
        except Exception:
            # Fall through to the Qt bell — an audio-less session, a policy
            # that blocks shell sounds, or winsound itself refusing.
            pass
    try:  # pragma: no cover - platform dependent
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.beep()
    except Exception:
        pass
