"""The application Settings dialog.

Edits an :class:`~app.models.app_settings.AppSettings` value object. The dialog
never persists anything itself — it returns an updated ``AppSettings`` via
:meth:`SettingsDialog.result_settings`; the main window applies it (theme,
autosave, canvas) and saves through :mod:`app.services.settings_service`.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
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

#: Smallest the dialog is allowed to get. Kept modest so it fits a small or
#: heavily scaled laptop panel; on a roomy display the dialog opens larger (see
#: :meth:`SettingsDialog._fit_to_screen`).
_MIN_WIDTH = 520
_MIN_HEIGHT = 360

#: Share of the available screen the dialog may occupy when sizing itself.
_SCREEN_FRACTION = 0.9

#: Screen width at (and above) which the groups are laid out in two columns.
#: Below it the columns would be too narrow for the form rows to read, so the
#: groups stack and the dialog scrolls instead.
_TWO_COLUMN_MIN_WIDTH = 860


class SettingsDialog(QDialog):
    """A grouped preferences dialog over :class:`AppSettings`."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Work on a clone so Cancel truly discards; from_dict deep-copies lists.
        self._source = settings
        self.setWindowTitle("Settings")
        self.setModal(True)

        # Lay the groups out in two columns when the display is wide enough.
        # Stacked in one column this dialog is ~940 px tall, which does not fit
        # a 768-px-tall laptop panel at all — the user would be scrolling for
        # options that have plenty of room to sit side by side. Two columns
        # roughly halve the height, so on a normal display every option is
        # visible at once, which is the point of the dialog.
        screen = self._target_screen()
        available = screen.availableGeometry() if screen is not None else None
        two_columns = available is not None and available.width() >= _TWO_COLUMN_MIN_WIDTH

        # Width is settled in :meth:`_fit_to_screen`, which measures the content
        # it just built; only the height floor is fixed up front.
        self.setMinimumHeight(_MIN_HEIGHT)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel("Settings")
        title.setProperty("role", "subheading")
        root.addWidget(title)

        # The groups live inside a scroll area, while the title and the button
        # box stay outside it as fixed rows. That is what guarantees OK / Cancel
        # are always reachable: however short the display, the worst case is a
        # scrollbar through the settings rather than controls pushed silently
        # off the bottom edge with no way to get to them.
        groups = [
            self._build_appearance_group(),
            self._build_defaults_group(),
            self._build_autosave_group(),
            self._build_canvas_group(),
            self._build_performance_group(),
            self._build_confirmations_group(),
        ]

        content = QWidget()
        if two_columns:
            grid = QGridLayout(content)
            grid.setContentsMargins(0, 0, 8, 0)  # room for the scrollbar
            grid.setSpacing(12)
            grid.setHorizontalSpacing(14)
            for i, group in enumerate(groups):
                grid.addWidget(group, i // 2, i % 2, Qt.AlignTop)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
        else:
            column = QVBoxLayout(content)
            column.setContentsMargins(0, 0, 8, 0)
            column.setSpacing(14)
            for group in groups:
                column.addWidget(group)
            column.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setWidget(content)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        root.addWidget(self._scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.RestoreDefaults | QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setProperty("variant", "primary")
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore_defaults)
        root.addWidget(buttons)

        self._load_from(settings)
        self._fit_to_screen()

    def _target_screen(self):
        """The screen this dialog will open on (its parent's, else the primary)."""
        parent = self.parentWidget()
        if parent is not None:
            screen = parent.screen()
            if screen is not None:
                return screen
        return QGuiApplication.primaryScreen()

    def _fit_to_screen(self) -> None:
        """Size the dialog to its content, capped to the screen it will open on.

        A fixed geometry cannot work here: the same dialog is opened on desktop
        monitors and on small, heavily scaled laptop panels, and the group boxes
        grow with the system font. Sizing to the content and clamping to the
        available screen keeps every option visible where there is room, and
        falls back on the scroll area where there is not.
        """
        screen = self._target_screen()
        available = screen.availableGeometry() if screen is not None else None
        max_w = int(available.width() * _SCREEN_FRACTION) if available else 10 ** 6
        max_h = int(available.height() * _SCREEN_FRACTION) if available else 10 ** 6

        hint = self.sizeHint()
        width, height = hint.width(), hint.height()

        # A scroll area's own size hint is deliberately modest — it never asks
        # for enough room to show all of its content, because that is the whole
        # point of scrolling. Left at that, the dialog would open showing only
        # the first few groups on a display with plenty of room for every one of
        # them. So ask the *content* what it needs and add back everything that
        # is not the scroller (title, spacing, button row, margins) — measured
        # from the hint rather than hardcoded, so it stays correct if the
        # dialog's chrome changes.
        content = self._scroll.widget()
        if content is not None:
            chrome_h = hint.height() - self._scroll.sizeHint().height()
            chrome_w = hint.width() - self._scroll.sizeHint().width()
            height = max(height, content.sizeHint().height() + chrome_h)
            # The content's own minimum width is the floor: sizing below it is
            # exactly what introduces a horizontal scrollbar across the form
            # rows, which is what makes a row of settings unreadable.
            width = max(width, content.minimumSizeHint().width() + chrome_w)

        # The minimum tracks the size we actually settled on, so the layout can
        # never be squeezed narrower than the content it was measured for —
        # except on a screen too small to honour it, where the cap wins and the
        # scroll area takes over.
        self.setMinimumWidth(min(max(_MIN_WIDTH, width), max_w))
        self.resize(min(width, max_w), max(_MIN_HEIGHT, min(height, max_h)))

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
        self._wheel_zooms = QCheckBox("Mouse wheel zooms the canvas")
        self._wheel_zooms.setToolTip(
            "Off: the wheel (and a touchpad's two-finger scroll) pans the canvas "
            "and Ctrl/Alt+wheel zooms. On: the wheel zooms, as in earlier versions."
        )
        layout.addWidget(self._show_grid)
        layout.addWidget(self._show_rulers)
        layout.addWidget(self._wheel_zooms)
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
        self._wheel_zooms.setChecked(s.wheel_zooms)
        self._bg_opacity.set_value(max(0, min(100, s.background_opacity)) / 100.0)
        self._analysis_dim.setValue(int(s.analysis_max_dim))
        self._confirm_overwrite.setChecked(s.confirm_before_overwrite)
        self._delete_threshold.setValue(int(s.confirm_delete_threshold))

    def _restore_defaults(self) -> None:
        self._load_from(AppSettings())
        # Defaults can change which rows are enabled/visible, so re-fit rather
        # than leaving the dialog sized for the previous state.
        self._fit_to_screen()

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
        out.wheel_zooms = self._wheel_zooms.isChecked()
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
