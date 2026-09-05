"""The application Settings dialog.

Edits an :class:`~app.models.app_settings.AppSettings` value object. The dialog
never persists anything itself — it returns an updated ``AppSettings`` via
:meth:`SettingsDialog.result_settings`; the main window applies it (theme,
autosave, canvas) and saves through :mod:`app.services.settings_service`.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.geometry import Unit
from app.models.app_settings import AppSettings
from app.models.stone import STONE_SIZE_ORDER, CUSTOM_SIZE_NAME, get_stone_size
from app.ui.widgets.labeled_slider import LabeledSlider
from app.ui.widgets.wheel_safe import NoWheelComboBox, NoWheelSpinBox

#: (Unit, label) shown in the default-units combo — mirrors the sidebar.
_UNIT_ITEMS = [(Unit.INCH, "Inches (in)"), (Unit.MM, "Millimeters (mm)"), (Unit.CM, "Centimeters (cm)")]


class SettingsDialog(QDialog):
    """A grouped preferences dialog over :class:`AppSettings`."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Work on a clone so Cancel truly discards; from_dict deep-copies lists.
        self._source = settings
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(480)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel("Settings")
        title.setProperty("role", "subheading")
        root.addWidget(title)

        root.addWidget(self._build_appearance_group())
        root.addWidget(self._build_defaults_group())
        root.addWidget(self._build_autosave_group())
        root.addWidget(self._build_canvas_group())
        root.addWidget(self._build_performance_group())
        root.addWidget(self._build_confirmations_group())
        root.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.RestoreDefaults | QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setProperty("variant", "primary")
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore_defaults)
        root.addWidget(buttons)

        self._load_from(settings)

    # ------------------------------------------------------------------ #
    # Group construction
    # ------------------------------------------------------------------ #
    def _build_appearance_group(self) -> QGroupBox:
        box = QGroupBox("Appearance")
        form = QFormLayout(box)
        self._theme_combo = NoWheelComboBox()
        self._theme_combo.addItem("Dark", "dark")
        self._theme_combo.addItem("Light", "light")
        form.addRow("Theme", self._theme_combo)
        return box

    def _build_defaults_group(self) -> QGroupBox:
        box = QGroupBox("Defaults for new projects")
        form = QFormLayout(box)

        self._units_combo = NoWheelComboBox()
        for unit, label in _UNIT_ITEMS:
            self._units_combo.addItem(label, unit)
        form.addRow("Units", self._units_combo)

        self._stone_combo = NoWheelComboBox()
        for size_id in STONE_SIZE_ORDER:
            if size_id == CUSTOM_SIZE_NAME:
                continue  # a "Custom" default has no stored diameter
            size = get_stone_size(size_id)
            self._stone_combo.addItem(f"{size_id} · {size.diameter_mm:.1f} mm", size_id)
        form.addRow("Stone size", self._stone_combo)

        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        self._export_dir = QLineEdit()
        self._export_dir.setPlaceholderText("Ask each time")
        self._export_dir.setClearButtonEnabled(True)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_export_dir)
        export_row.addWidget(self._export_dir, 1)
        export_row.addWidget(browse)
        export_wrap = QWidget()
        export_wrap.setLayout(export_row)
        form.addRow("Export folder", export_wrap)
        return box

    def _build_autosave_group(self) -> QGroupBox:
        box = QGroupBox("Autosave & recovery")
        form = QFormLayout(box)
        self._autosave_enabled = QCheckBox("Automatically save a recovery copy")
        form.addRow(self._autosave_enabled)
        self._autosave_interval = NoWheelSpinBox()
        self._autosave_interval.setRange(15, 3600)
        self._autosave_interval.setSingleStep(15)
        self._autosave_interval.setSuffix(" sec")
        form.addRow("Interval", self._autosave_interval)
        self._autosave_enabled.toggled.connect(self._autosave_interval.setEnabled)
        return box

    def _build_canvas_group(self) -> QGroupBox:
        box = QGroupBox("Canvas")
        layout = QVBoxLayout(box)
        self._show_grid = QCheckBox("Show grid")
        self._show_rulers = QCheckBox("Show rulers")
        layout.addWidget(self._show_grid)
        layout.addWidget(self._show_rulers)
        self._bg_opacity = LabeledSlider(
            "Background image opacity", value=0.45, low_label="0%", high_label="100%"
        )
        layout.addWidget(self._bg_opacity)
        return box

    def _build_performance_group(self) -> QGroupBox:
        box = QGroupBox("Performance")
        form = QFormLayout(box)
        self._analysis_dim = NoWheelSpinBox()
        self._analysis_dim.setRange(400, 4000)
        self._analysis_dim.setSingleStep(100)
        self._analysis_dim.setSuffix(" px")
        hint = QLabel("Larger values analyse more image detail but generate slower.")
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        form.addRow("Analysis resolution", self._analysis_dim)
        form.addRow(hint)
        return box

    def _build_confirmations_group(self) -> QGroupBox:
        box = QGroupBox("Confirmations")
        form = QFormLayout(box)
        self._confirm_overwrite = QCheckBox("Warn before overwriting existing files")
        form.addRow(self._confirm_overwrite)
        self._delete_threshold = NoWheelSpinBox()
        self._delete_threshold.setRange(1, 100_000)
        self._delete_threshold.setSingleStep(5)
        form.addRow("Confirm bulk delete over", self._delete_threshold)
        return box

    # ------------------------------------------------------------------ #
    # Load / read
    # ------------------------------------------------------------------ #
    def _load_from(self, s: AppSettings) -> None:
        self._set_combo_data(self._theme_combo, "light" if s.theme == "light" else "dark")
        self._set_combo_data(self._units_combo, s.default_units)
        self._set_combo_data(self._stone_combo, s.default_stone_size_id)
        self._export_dir.setText(s.default_export_dir or "")
        self._autosave_enabled.setChecked(s.autosave_enabled)
        self._autosave_interval.setValue(int(s.autosave_interval_sec))
        self._autosave_interval.setEnabled(s.autosave_enabled)
        self._show_grid.setChecked(s.show_grid)
        self._show_rulers.setChecked(s.show_rulers)
        self._bg_opacity.set_value(max(0, min(100, s.background_opacity)) / 100.0)
        self._analysis_dim.setValue(int(s.analysis_max_dim))
        self._confirm_overwrite.setChecked(s.confirm_before_overwrite)
        self._delete_threshold.setValue(int(s.confirm_delete_threshold))

    def _restore_defaults(self) -> None:
        self._load_from(AppSettings())

    def result_settings(self) -> AppSettings:
        """Return a new :class:`AppSettings` reflecting the dialog's controls.

        Recent projects and any fields the dialog does not expose are carried
        over from the source settings; numeric fields are clamped to valid
        ranges via :meth:`AppSettings.clamp`.
        """
        # Clone the source (preserves recent_projects etc.), then overwrite.
        out = AppSettings.from_dict(self._source.to_dict())
        out.theme = self._theme_combo.currentData()
        out.default_units = self._units_combo.currentData()
        out.default_stone_size_id = self._stone_combo.currentData()
        text = self._export_dir.text().strip()
        out.default_export_dir = text or None
        out.autosave_enabled = self._autosave_enabled.isChecked()
        out.autosave_interval_sec = self._autosave_interval.value()
        out.show_grid = self._show_grid.isChecked()
        out.show_rulers = self._show_rulers.isChecked()
        out.background_opacity = int(round(self._bg_opacity.value() * 100))
        out.analysis_max_dim = self._analysis_dim.value()
        out.confirm_before_overwrite = self._confirm_overwrite.isChecked()
        out.confirm_delete_threshold = self._delete_threshold.value()
        out.clamp()
        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _browse_export_dir(self) -> None:
        start = self._export_dir.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Default Export Folder", start)
        if directory:
            self._export_dir.setText(directory)

    @staticmethod
    def _set_combo_data(combo: QComboBox, data) -> None:
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)
