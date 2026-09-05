"""A labelled slider mapping an integer track to a ``0.0``–``1.0`` value.

Shows a title, a live value read-out and low/high end captions — the intuitive
control the density engine exposes instead of raw algorithm numbers.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

_STEPS = 100


class LabeledSlider(QWidget):
    """A titled 0..1 slider with a percentage read-out and end captions."""

    valueChanged = Signal(float)          # continuous, while dragging
    valueCommitted = Signal(float)        # on release / keyboard change

    def __init__(
        self,
        title: str,
        value: float = 0.5,
        *,
        low_label: str = "Low",
        high_label: str = "High",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(3)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self._title_label = QLabel(title)
        self._title_label.setProperty("role", "label")
        self._value_label = QLabel()
        self._value_label.setProperty("role", "caption")
        self._value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(self._title_label)
        header.addStretch(1)
        header.addWidget(self._value_label)
        layout.addLayout(header)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, _STEPS)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.setValue(int(round(value * _STEPS)))
        self._slider.valueChanged.connect(self._on_changed)
        self._slider.sliderReleased.connect(
            lambda: self.valueCommitted.emit(self.value())
        )
        layout.addWidget(self._slider)

        ends = QHBoxLayout()
        ends.setContentsMargins(0, 0, 0, 0)
        lo = QLabel(low_label)
        lo.setProperty("role", "caption")
        hi = QLabel(high_label)
        hi.setProperty("role", "caption")
        hi.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ends.addWidget(lo)
        ends.addStretch(1)
        ends.addWidget(hi)
        layout.addLayout(ends)

        self._update_value_label()

    def _on_changed(self, _int_value: int) -> None:
        self._update_value_label()
        self.valueChanged.emit(self.value())
        # Keyboard changes don't fire sliderReleased; treat them as commits.
        if not self._slider.isSliderDown():
            self.valueCommitted.emit(self.value())

    def _update_value_label(self) -> None:
        self._value_label.setText(f"{self._slider.value()}%")

    def value(self) -> float:
        return self._slider.value() / _STEPS

    def set_value(self, value: float, *, emit: bool = False) -> None:
        target = max(0, min(_STEPS, int(round(value * _STEPS))))
        if emit:
            self._slider.setValue(target)
        else:
            self._slider.blockSignals(True)
            self._slider.setValue(target)
            self._slider.blockSignals(False)
            self._update_value_label()
