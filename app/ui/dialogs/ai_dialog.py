"""AI Assistant dialog (ISSUE 6) — suggest → preview → Apply / Reject.

The dialog never edits the design itself. It shows a suggestion's summary and
a ghost preview on the canvas; the user must click Apply, at which point the
window records the whole suggestion as ONE undoable command. Reject discards
the preview and changes nothing.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ai.assistant import AIAssistant
from app.ai.validator import ValidatedSuggestion
from app.models.design import Design
from app.ui.canvas import CanvasView

_PRESETS = (
    "Add more stones in dark areas",
    "Reduce the number of stones",
    "Fill this empty region",
    "Clean up this region",
    "Move stones away from this region",
    "Improve the rhinestone arrangement",
)


class AIAssistantDialog(QDialog):
    """Ask the assistant for a suggestion, preview it, then apply or reject."""

    #: Emitted with the validated suggestion when the user clicks Apply.
    applied = Signal(object)
    #: Emitted when the user explicitly rejects the suggestion (distinct from
    #: QDialog's built-in ``rejected`` to avoid double emission).
    discard_requested = Signal()

    def __init__(
        self,
        assistant: AIAssistant,
        design: Design,
        canvas: CanvasView,
        *,
        gap_mm: float,
        stone_diameter_mm: float,
        density=None,
        region=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._assistant = assistant
        self._design = design
        self._canvas = canvas
        self._gap = gap_mm
        self._diameter = stone_diameter_mm
        self._density = density
        self._region = region
        self._validated: ValidatedSuggestion | None = None

        self.setWindowTitle("AI Assistant")
        self.setMinimumWidth(460)
        root = QVBoxLayout(self)
        root.setSpacing(10)

        header = QLabel(assistant.provider_label)
        header.setProperty("role", "caption")
        root.addWidget(header)
        note = QLabel(
            "Runs entirely on your computer — nothing is uploaded. "
            "Suggestions are geometry-checked before they can be applied."
        )
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        root.addWidget(note)

        self._command = QLineEdit()
        self._command.setPlaceholderText("e.g. “add more stones around the dark areas”")
        self._command.returnPressed.connect(self._suggest)
        root.addWidget(self._command)

        presets = QWidget()
        pl = QHBoxLayout(presets)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(4)
        for text in _PRESETS:
            btn = QPushButton(text)
            btn.setToolTip(f"Use “{text}” as the command")
            btn.clicked.connect(lambda _=False, t=text: self._use_preset(t))
            pl.addWidget(btn)
        root.addWidget(presets)

        if region is not None:
            rx, ry, rw, rh = region
            rlab = QLabel(
                f"Region: {rw:.0f} × {rh:.0f} mm at ({rx:.0f}, {ry:.0f}) "
                "(from your selection)"
            )
        else:
            rlab = QLabel("Region: whole design (select stones to focus a region)")
        rlab.setProperty("role", "caption")
        root.addWidget(rlab)

        self._summary = QLabel("Enter a command and press Suggest.")
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        buttons = QHBoxLayout()
        self._suggest_btn = QPushButton("Suggest")
        self._suggest_btn.setProperty("variant", "primary")
        self._suggest_btn.clicked.connect(self._suggest)
        buttons.addWidget(self._suggest_btn)
        buttons.addStretch(1)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        buttons.addWidget(self._apply_btn)
        reject = QPushButton("Reject")
        reject.clicked.connect(self._reject)
        buttons.addWidget(reject)
        root.addLayout(buttons)

    # ------------------------------------------------------------------ #
    def _use_preset(self, text: str) -> None:
        self._command.setText(text)
        self._suggest()

    def _suggest(self) -> None:
        command = self._command.text().strip()
        if not command:
            return
        self._validated = self._assistant.preview(
            self._design, command,
            gap_mm=self._gap,
            stone_diameter_mm=self._diameter,
            density=self._density,
            region=self._region,
        )
        if self._validated is None:
            self._summary.setText("No AI provider is available.")
            self._apply_btn.setEnabled(False)
            return
        adds = [
            (op.x, op.y)
            for op in self._validated.valid_ops
            if op.kind.value == "add_stone" and op.x is not None and op.y is not None
        ]
        moves = []
        for op in self._validated.valid_ops:
            if op.kind.value == "move_stone" and op.x is not None and op.y is not None:
                stone = self._design.get_by_id(op.stone_id) if op.stone_id else None
                if stone is not None:
                    moves.append((stone.x, stone.y, op.x, op.y))
        import numpy as np

        add_arr = np.array(adds, dtype=np.float64) if adds else None
        self._canvas.set_ai_preview(add_arr, moves, diameter=self._diameter)
        self._summary.setText(
            f"{self._validated.suggestion.description}\n\n"
            f"{self._validated.summary()}"
        )
        self._apply_btn.setEnabled(not self._validated.is_empty)

    def _apply(self) -> None:
        if self._validated is not None and not self._validated.is_empty:
            self.applied.emit(self._validated)
            self._canvas.set_ai_preview(None, None)
            self.accept()

    def _reject(self) -> None:
        self._canvas.set_ai_preview(None, None)
        self.discard_requested.emit()
        self.reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._canvas.set_ai_preview(None, None)
        super().closeEvent(event)
