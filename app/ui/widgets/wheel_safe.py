"""Spin and combo widgets that ignore the scroll wheel.

Qt gives ``QComboBox`` and the spin boxes wheel-event handling by default, so
scrolling a panel that contains them silently changes the selected option or
value — a frequent trap on laptops, where touchpad scrolling commonly passes
over these widgets. These variants decline the wheel event (it propagates to
the enclosing scroll area) and require an explicit click to change the value.
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class WheelIgnoredMixin:
    """Shared wheel suppression for wheel-grabbing input widgets."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        # Ignore instead of accepting, so the scroll propagates to the
        # surrounding panel rather than being swallowed.
        event.ignore()


class NoWheelComboBox(WheelIgnoredMixin, QComboBox):
    """A combo box that only changes selection via mouse/keyboard, not wheel."""


class NoWheelSpinBox(WheelIgnoredMixin, QSpinBox):
    """A spin box that only changes value via click/typing, not wheel."""


class NoWheelDoubleSpinBox(WheelIgnoredMixin, QDoubleSpinBox):
    """A double spin box that only changes value via click/typing, not wheel."""


__all__ = [
    "NoWheelComboBox",
    "NoWheelSpinBox",
    "NoWheelDoubleSpinBox",
]
