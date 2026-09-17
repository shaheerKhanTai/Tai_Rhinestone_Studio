"""The left workflow panel: the import → size → stones → generate controls.

Presents the primary workflow top-to-bottom with progressive disclosure of
advanced density controls. It is a *view*: it reflects the current project's
settings and emits signals; the main window applies changes and triggers
generation. It contains no generation algorithms.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.geometry import Unit, from_mm, to_mm
from app.models.presets import SizePreset
from app.models.project import Project
from app.models.settings import GenerationSettings, PlacementMode
from app.models.stone import CUSTOM_SIZE_NAME, STONE_SIZE_ORDER, get_stone_size
from app.ui.widgets.labeled_slider import LabeledSlider
from app.ui.widgets.wheel_safe import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)

_UNIT_ITEMS = [(Unit.INCH, "in"), (Unit.MM, "mm"), (Unit.CM, "cm")]


class DesignPanel(QWidget):
    """Workflow controls for size, rhinestones, density and generation."""

    import_image_requested = Signal()
    size_changed = Signal()                 # physical size / unit changed
    settings_changed = Signal()             # generation settings changed
    generate_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, presets: list[SizePreset], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._presets = presets
        self._updating = False  # guard to suppress signals while syncing
        self._build()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        root.addWidget(self._build_image_group())
        root.addWidget(self._build_size_group())
        root.addWidget(self._build_stone_group())
        root.addWidget(self._build_density_group())
        root.addWidget(self._build_generate_block())
        root.addStretch(1)

    def _build_image_group(self) -> QWidget:
        box = QGroupBox("1 · Image")
        lay = QVBoxLayout(box)
        self._image_name = QLabel("No image imported")
        self._image_name.setProperty("role", "muted")
        self._image_name.setWordWrap(True)
        self._image_info = QLabel("")
        self._image_info.setProperty("role", "caption")
        # Wrap so a long "WxHpx · FORMAT · N KB" line can never widen the
        # panel past its dock — different resolutions must not stretch the UI.
        self._image_info.setWordWrap(True)
        import_btn = QPushButton("Import Image…")
        import_btn.clicked.connect(self.import_image_requested)
        lay.addWidget(self._image_name)
        lay.addWidget(self._image_info)
        lay.addWidget(import_btn)

        # Crop / rotate / flip controls — populated by the window with themed
        # QActions via :meth:`set_transform_actions` (kept empty until then).
        self._transform_caption = QLabel("Adjust")
        self._transform_caption.setProperty("role", "caption")
        lay.addWidget(self._transform_caption)
        self._transform_bar = QWidget()
        self._transform_layout = QHBoxLayout(self._transform_bar)
        self._transform_layout.setContentsMargins(0, 0, 0, 0)
        self._transform_layout.setSpacing(4)
        lay.addWidget(self._transform_bar)
        return box

    def set_transform_actions(self, actions: list[QAction]) -> None:
        """Fill the Image group's tool row with buttons bound to ``actions``."""
        while self._transform_layout.count():
            item = self._transform_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for act in actions:
            btn = QToolButton()
            btn.setDefaultAction(act)
            btn.setAutoRaise(True)
            btn.setIconSize(QSize(18, 18))
            self._transform_layout.addWidget(btn)
        self._transform_layout.addStretch(1)

    @staticmethod
    def _make_combo() -> NoWheelComboBox:
        """A combo that elides long items instead of widening the panel.

        Left to its default size policy a combo's hint is the *widest* item,
        so long preset names ("T-Shirt Front — 11 × 12 in") forced the sidebar
        past its dock width and clipped it. Capped to a reasonable content
        length, long text is ellipsised and the panel stays at dock width.
        """
        combo = NoWheelComboBox()
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(10)
        return combo

    def _build_size_group(self) -> QWidget:
        box = QGroupBox("2 · Design Size")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._preset_combo = self._make_combo()
        for preset in self._presets:
            self._preset_combo.addItem(preset.name, preset)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        form.addRow("Preset", self._preset_combo)

        self._width_spin = self._make_dim_spin()
        self._height_spin = self._make_dim_spin()
        self._width_spin.valueChanged.connect(lambda _: self._on_dim_edited(is_width=True))
        self._height_spin.valueChanged.connect(lambda _: self._on_dim_edited(is_width=False))
        form.addRow("Width", self._width_spin)
        form.addRow("Height", self._height_spin)

        self._unit_combo = self._make_combo()
        for unit, label in _UNIT_ITEMS:
            self._unit_combo.addItem(label, unit)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        form.addRow("Units", self._unit_combo)

        self._lock_aspect = QCheckBox("Lock aspect ratio")
        self._lock_aspect.setChecked(True)
        self._lock_aspect.toggled.connect(self._on_lock_toggled)
        form.addRow("", self._lock_aspect)
        return box

    def _make_dim_spin(self) -> NoWheelDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(0.1, 5000.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.25)
        spin.setValue(10.0)
        return spin

    def _build_stone_group(self) -> QWidget:
        box = QGroupBox("3 · Rhinestones")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._size_combo = self._make_combo()
        for name in STONE_SIZE_ORDER:
            d = get_stone_size(name).diameter_mm
            self._size_combo.addItem(f"{name}  ·  {d:.1f} mm", name)
        self._size_combo.addItem("Custom…", CUSTOM_SIZE_NAME)
        self._size_combo.currentIndexChanged.connect(self._on_stone_size_changed)
        form.addRow("Stone size", self._size_combo)

        self._custom_dia = NoWheelDoubleSpinBox()
        self._custom_dia.setRange(0.5, 20.0)
        self._custom_dia.setDecimals(2)
        self._custom_dia.setSingleStep(0.1)
        self._custom_dia.setValue(3.0)
        self._custom_dia.setSuffix(" mm")
        self._custom_dia.valueChanged.connect(lambda _: self._emit_settings())
        self._custom_dia_row = self._custom_dia
        form.addRow("Diameter", self._custom_dia)

        self._spacing = NoWheelDoubleSpinBox()
        self._spacing.setRange(0.0, 10.0)
        self._spacing.setDecimals(2)
        self._spacing.setSingleStep(0.1)
        self._spacing.setValue(0.5)
        self._spacing.setSuffix(" mm")
        self._spacing.valueChanged.connect(lambda _: self._emit_settings())
        form.addRow("Spacing (gap)", self._spacing)

        self._pitch_label = QLabel("—")
        self._pitch_label.setToolTip("Distance between neighbouring stone centres")
        self._pitch_label.setProperty("role", "caption")
        form.addRow("Pitch", self._pitch_label)

        self._push_all = QCheckBox("Push All")
        self._push_all.setToolTip(
            "Automatically moves neighboring rhinestones when needed to "
            "maintain the configured spacing."
        )
        self._push_all.toggled.connect(lambda _: self._emit_settings())
        form.addRow("", self._push_all)

        self._mode_combo = self._make_combo()
        for mode in PlacementMode:
            self._mode_combo.addItem(mode.label, mode)
        # Default follows the settings model (Outline + Fill) — selected by
        # data, never by index, so enum order can change freely.
        self._set_combo_data(self._mode_combo, GenerationSettings().placement_mode)
        self._mode_combo.currentIndexChanged.connect(lambda _: self._emit_settings())
        form.addRow("Placement", self._mode_combo)
        return box

    def _build_density_group(self) -> QWidget:
        box = QGroupBox("4 · Appearance")
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        self._detail = LabeledSlider("Detail", 0.5)
        self._density = LabeledSlider("Density", 0.5)
        self._contrast = LabeledSlider("Contrast", 0.5)
        self._smoothing = LabeledSlider("Smoothing", 0.3)
        self._edge = LabeledSlider("Edge Preservation", 0.5)
        for s in (self._detail, self._density, self._contrast, self._smoothing, self._edge):
            s.valueCommitted.connect(lambda _=0.0: self._emit_settings())
            lay.addWidget(s)

        self._invert = QCheckBox("Light subject on\ndark background")
        self._invert.setToolTip(
            "Place stones on the light parts of the image instead of the "
            "dark parts (e.g. white lettering on a black shirt)."
        )
        self._invert.toggled.connect(lambda _: self._emit_settings())
        lay.addWidget(self._invert)

        self._invert_areas = QCheckBox("Invert Rhinestone Areas")
        self._invert_areas.setToolTip(
            "Fill the empty parts of the design instead of the subject — the "
            "pattern shifts into the complementary regions of the same area.\n"
            "The stone size, spacing and margins are unchanged; transparent "
            "areas stay empty."
        )
        self._invert_areas.toggled.connect(lambda _: self._emit_settings())
        lay.addWidget(self._invert_areas)

        # Max stone count with an unlimited toggle.
        limit_row = QHBoxLayout()
        self._limit_enabled = QCheckBox("Limit stones")
        self._limit_enabled.toggled.connect(self._on_limit_toggled)
        self._max_stones = NoWheelSpinBox()
        self._max_stones.setRange(1, 1_000_000)
        self._max_stones.setValue(5000)
        self._max_stones.setSingleStep(100)
        self._max_stones.setEnabled(False)
        self._max_stones.valueChanged.connect(lambda _: self._emit_settings())
        limit_row.addWidget(self._limit_enabled)
        limit_row.addWidget(self._max_stones, 1)
        lay.addLayout(limit_row)
        return box

    def _build_generate_block(self) -> QWidget:
        frame = QFrame()
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._generate_btn = QPushButton("Generate Design")
        self._generate_btn.setProperty("variant", "primary")
        self._generate_btn.setMinimumHeight(40)
        self._generate_btn.clicked.connect(self.generate_requested)
        lay.addWidget(self._generate_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._stage_label = QLabel("")
        self._stage_label.setProperty("role", "caption")
        self._stage_label.setVisible(False)
        lay.addWidget(self._stage_label)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setProperty("variant", "destructive")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self.cancel_requested)
        lay.addWidget(self._cancel_btn)
        return frame

    # ------------------------------------------------------------------ #
    # Binding to a project
    # ------------------------------------------------------------------ #
    def bind_project(self, project: Project | None) -> None:
        self._project = project
        self._sync_from_project()

    def set_image_info(self, name: str, detail: str) -> None:
        self._image_name.setText(name)
        self._image_info.setText(detail)

    def _sync_from_project(self) -> None:
        if self._project is None:
            return
        self._updating = True
        try:
            proj = self._project
            unit = proj.units
            self._set_combo_data(self._unit_combo, unit)
            self._set_dim_spin_config(unit)
            self._width_spin.setValue(from_mm(proj.width_mm, unit))
            self._height_spin.setValue(from_mm(proj.height_mm, unit))
            self._lock_aspect.setChecked(proj.lock_aspect)

            s = proj.settings
            self._set_combo_data(self._size_combo, s.stone_size_id)
            self._custom_dia.setValue(s.custom_diameter_mm or 3.0)
            self._update_custom_visibility()
            self._spacing.setValue(s.spacing_mm)
            self._set_combo_data(self._mode_combo, s.placement_mode)
            self._push_all.setChecked(s.push_all)
            self._detail.set_value(s.detail)
            self._density.set_value(s.density)
            self._contrast.set_value(s.contrast)
            self._smoothing.set_value(s.smoothing)
            self._edge.set_value(s.edge_preservation)
            self._invert.setChecked(s.invert_tone)
            self._invert_areas.setChecked(s.invert_areas)
            enabled = s.max_stones is not None
            self._limit_enabled.setChecked(enabled)
            self._max_stones.setEnabled(enabled)
            if s.max_stones:
                self._max_stones.setValue(s.max_stones)
        finally:
            self._updating = False
        self._update_pitch_label()

    @staticmethod
    def _set_combo_data(combo: QComboBox, data) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _set_dim_spin_config(self, unit: Unit) -> None:
        step = 0.25 if unit is Unit.INCH else (5.0 if unit is Unit.MM else 0.5)
        for spin in (self._width_spin, self._height_spin):
            spin.setSingleStep(step)
            spin.setSuffix(f" {unit.symbol}")

    # ------------------------------------------------------------------ #
    # Signal handlers
    # ------------------------------------------------------------------ #
    def _on_preset_selected(self, _index: int) -> None:
        if self._updating or self._project is None:
            return
        preset: SizePreset | None = self._preset_combo.currentData()
        if preset is None or preset.name == "Custom":
            return
        self._updating = True
        try:
            self._set_combo_data(self._unit_combo, preset.unit)
            self._project.units = preset.unit
            self._set_dim_spin_config(preset.unit)
            self._width_spin.setValue(preset.width)
            self._height_spin.setValue(preset.height)
        finally:
            self._updating = False
        self._commit_size()

    def _on_dim_edited(self, is_width: bool) -> None:
        if self._updating or self._project is None:
            return
        if self._lock_aspect.isChecked() and self._project.has_image:
            img = self._project.image
            if img.original_width_px and img.original_height_px:
                aspect = img.original_height_px / img.original_width_px
                self._updating = True
                try:
                    if is_width:
                        self._height_spin.setValue(self._width_spin.value() * aspect)
                    else:
                        self._width_spin.setValue(self._height_spin.value() / aspect)
                finally:
                    self._updating = False
        self._commit_size()

    def _on_unit_changed(self, _index: int) -> None:
        if self._updating or self._project is None:
            return
        new_unit: Unit = self._unit_combo.currentData()
        # Convert the displayed numbers so the physical size is unchanged.
        old_w_mm = self._project.width_mm
        old_h_mm = self._project.height_mm
        self._project.units = new_unit
        self._updating = True
        try:
            self._set_dim_spin_config(new_unit)
            self._width_spin.setValue(from_mm(old_w_mm, new_unit))
            self._height_spin.setValue(from_mm(old_h_mm, new_unit))
        finally:
            self._updating = False
        self._commit_size()

    def _on_lock_toggled(self, checked: bool) -> None:
        if self._project is not None:
            self._project.lock_aspect = checked

    def _on_stone_size_changed(self, _index: int) -> None:
        self._update_custom_visibility()
        if not self._updating:
            self._emit_settings()

    def _on_limit_toggled(self, checked: bool) -> None:
        self._max_stones.setEnabled(checked)
        if not self._updating:
            self._emit_settings()

    def _update_custom_visibility(self) -> None:
        is_custom = self._size_combo.currentData() == CUSTOM_SIZE_NAME
        self._custom_dia.setVisible(is_custom)
        # Also toggle the row label via the form layout.
        form = self._custom_dia.parentWidget().layout()
        if isinstance(form, QFormLayout):
            label = form.labelForField(self._custom_dia)
            if label is not None:
                label.setVisible(is_custom)

    def _commit_size(self) -> None:
        if self._project is None:
            return
        unit = self._project.units
        w_mm = to_mm(self._width_spin.value(), unit)
        h_mm = to_mm(self._height_spin.value(), unit)
        if w_mm > 0 and h_mm > 0:
            self._project.set_physical_size(w_mm, h_mm)
            self.size_changed.emit()

    def build_settings(self) -> GenerationSettings:
        """Assemble a :class:`GenerationSettings` from the current controls."""
        size_id = self._size_combo.currentData()
        return GenerationSettings(
            stone_size_id=size_id,
            custom_diameter_mm=self._custom_dia.value() if size_id == CUSTOM_SIZE_NAME else None,
            spacing_mm=self._spacing.value(),
            placement_mode=self._mode_combo.currentData(),
            push_all=self._push_all.isChecked(),
            detail=self._detail.value(),
            density=self._density.value(),
            contrast=self._contrast.value(),
            smoothing=self._smoothing.value(),
            edge_preservation=self._edge.value(),
            invert_tone=self._invert.isChecked(),
            invert_areas=self._invert_areas.isChecked(),
            max_stones=self._max_stones.value() if self._limit_enabled.isChecked() else None,
        )

    def _emit_settings(self) -> None:
        if self._updating or self._project is None:
            return
        try:
            settings = self.build_settings()
        except Exception:
            return  # invalid transient state (e.g. custom size mid-edit)
        self._project.settings = settings
        self._update_pitch_label()
        self.settings_changed.emit()

    def _update_pitch_label(self) -> None:
        try:
            settings = self.build_settings()
            self._pitch_label.setText(f"{settings.center_distance():.2f} mm")
        except Exception:
            self._pitch_label.setText("—")

    # ------------------------------------------------------------------ #
    # Generation UI state
    # ------------------------------------------------------------------ #
    def set_generating(self, generating: bool) -> None:
        self._generate_btn.setVisible(not generating)
        self._progress.setVisible(generating)
        self._stage_label.setVisible(generating)
        self._cancel_btn.setVisible(generating)
        if generating:
            self._progress.setValue(0)

    def set_progress(self, stage: str, fraction: float) -> None:
        self._progress.setValue(int(fraction * 100))
        self._stage_label.setText(stage + "…")

    def set_generate_enabled(self, enabled: bool) -> None:
        self._generate_btn.setEnabled(enabled)
