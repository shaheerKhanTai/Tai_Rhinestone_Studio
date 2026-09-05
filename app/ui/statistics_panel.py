"""The right-hand panel: live design statistics and stone information.

A read-only view that reflects the current project and selection. Multi-colour
breakdown is included now so V2's multi-colour work needs no restructuring.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.core.geometry import from_mm
from app.models.project import Project


class StatisticsPanel(QWidget):
    """Displays design dimensions, counts, stone spec and colour breakdown."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        info = QGroupBox("Design Information")
        form = QFormLayout(info)
        form.setLabelAlignment(Qt.AlignRight)
        self._dim_label = self._value_label()
        self._count_label = self._value_label(strong=True)
        self._size_label = self._value_label()
        self._dia_label = self._value_label()
        self._spacing_label = self._value_label()
        self._coverage_label = self._value_label()
        form.addRow("Size", self._dim_label)
        form.addRow("Total stones", self._count_label)
        form.addRow("Stone size", self._size_label)
        form.addRow("Diameter", self._dia_label)
        form.addRow("Spacing", self._spacing_label)
        form.addRow("Coverage", self._coverage_label)
        root.addWidget(info)

        self._colors_group = QGroupBox("Colors")
        self._colors_layout = QVBoxLayout(self._colors_group)
        self._colors_layout.setSpacing(4)
        root.addWidget(self._colors_group)
        self._colors_group.setVisible(False)

        self._selection_group = QGroupBox("Selection")
        sel_layout = QVBoxLayout(self._selection_group)
        self._selection_label = QLabel("Nothing selected")
        self._selection_label.setProperty("role", "muted")
        sel_layout.addWidget(self._selection_label)
        root.addWidget(self._selection_group)

        root.addStretch(1)

    def _value_label(self, strong: bool = False) -> QLabel:
        lbl = QLabel("—")
        lbl.setProperty("role", "subheading" if strong else None)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return lbl

    # ------------------------------------------------------------------ #
    def bind_project(self, project: Project | None) -> None:
        self._project = project
        self.refresh()

    def refresh(self) -> None:
        proj = self._project
        if proj is None:
            return
        unit = proj.units
        self._dim_label.setText(
            f"{from_mm(proj.width_mm, unit):.2f} × {from_mm(proj.height_mm, unit):.2f} {unit.symbol}"
        )
        design = proj.design
        self._count_label.setText(f"{design.stone_count:,}")

        s = proj.settings
        self._size_label.setText(s.stone_size_id)
        try:
            self._dia_label.setText(f"{s.effective_diameter():.2f} mm")
        except Exception:
            self._dia_label.setText("—")
        self._spacing_label.setText(f"{s.spacing_mm:.2f} mm")
        self._coverage_label.setText(self._coverage_text(proj))
        self._refresh_colors(design.color_counts())

    def _coverage_text(self, proj: Project) -> str:
        """Estimated stone-area coverage of the design rectangle."""
        design = proj.design
        if design.stone_count == 0:
            return "—"
        import math

        area_design = proj.width_mm * proj.height_mm
        if area_design <= 0:
            return "—"
        stone_area = sum(math.pi * (s.diameter / 2.0) ** 2 for s in design.stones)
        pct = 100.0 * stone_area / area_design
        # Honest label: this is an estimate of stone footprint vs canvas area.
        return f"~{pct:.0f}% of area"

    def _refresh_colors(self, counts: dict[str, int]) -> None:
        # Clear existing rows.
        while self._colors_layout.count():
            item = self._colors_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if len(counts) <= 1:
            self._colors_group.setVisible(False)
            return
        self._colors_group.setVisible(True)
        for color, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            row = QWidget()
            from PySide6.QtWidgets import QHBoxLayout

            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            swatch = QLabel()
            pm = QPixmap(14, 14)
            pm.fill(QColor(color))
            swatch.setPixmap(pm)
            name = QLabel(color)
            count_lbl = QLabel(f"{count:,}")
            count_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            h.addWidget(swatch)
            h.addWidget(name)
            h.addStretch(1)
            h.addWidget(count_lbl)
            self._colors_layout.addWidget(row)

    def set_selection_count(self, count: int) -> None:
        if count <= 0:
            self._selection_label.setText("Nothing selected")
            self._selection_label.setProperty("role", "muted")
        else:
            self._selection_label.setText(f"{count:,} stone{'s' if count != 1 else ''} selected")
            self._selection_label.setProperty("role", None)
        # Re-polish so the role change takes effect.
        self._selection_label.style().unpolish(self._selection_label)
        self._selection_label.style().polish(self._selection_label)
