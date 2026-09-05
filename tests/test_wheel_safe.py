"""Wheel-safe input widgets must not change value on scroll.

Regression guard for the laptop trackpad trap: scrolling a panel that contains
a combo box or spin box used to silently change the option/value. The
``NoWheel*`` variants must ignore wheel events entirely while normal click and
keyboard interaction keeps working.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt, QEvent
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from app.ui.widgets.wheel_safe import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


def _wheel_event() -> QWheelEvent:
    pos = QPoint(0, 0)
    delta = QPoint(0, 120)
    return QWheelEvent(
        pos, pos, delta, delta, Qt.NoButton, Qt.NoModifier,
        Qt.ScrollUpdate, False,
    )


def _scroll(widget, times: int = 1) -> None:
    for _ in range(times):
        event = _wheel_event()
        QApplication.sendEvent(widget, event)


def test_combo_ignores_wheel(qapp):
    combo = NoWheelComboBox()
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(0)
    _scroll(combo, 3)
    assert combo.currentIndex() == 0


def test_spin_ignores_wheel(qapp):
    spin = NoWheelSpinBox()
    spin.setRange(0, 100)
    spin.setValue(10)
    _scroll(spin, 3)
    assert spin.value() == 10


def test_double_spin_ignores_wheel(qapp):
    spin = NoWheelDoubleSpinBox()
    spin.setRange(0.0, 100.0)
    spin.setValue(10.0)
    _scroll(spin, 3)
    assert spin.value() == pytest.approx(10.0)


def test_combo_keyboard_still_works(qapp):
    combo = NoWheelComboBox()
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(0)
    combo.show()
    combo.setFocus()
    QApplication.sendEvent(
        combo,
        QKeyEvent(QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier),
    )
    assert combo.currentIndex() == 1


def test_sidebar_uses_wheel_safe_widgets(qapp):
    from app.core.geometry import Unit
    from app.models.presets import SizePreset
    from app.ui.sidebar import DesignPanel

    presets = [SizePreset(name="A4", width=210.0, height=297.0, unit=Unit.MM)]
    panel = DesignPanel(presets)
    assert isinstance(panel._preset_combo, NoWheelComboBox)
    assert isinstance(panel._width_spin, NoWheelDoubleSpinBox)
    assert isinstance(panel._mode_combo, NoWheelComboBox)
    assert isinstance(panel._max_stones, NoWheelSpinBox)
