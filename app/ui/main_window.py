"""The application main window: menus, toolbar, docks, status bar, welcome.

Hosts a stacked welcome/editor view. The editor is a dock-based workspace: a
left *Design* panel, a central canvas host, and a right *Properties* panel. The
real canvas and panel contents are injected by later phases via
:meth:`set_canvas`, :meth:`set_left_panel` and :meth:`set_right_panel`; the
window owns the shell, menus, actions, theming and window-state persistence.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app import __app_name__, __version__
from app.core.geometry import fit_dimensions, to_mm, Unit
from app.core.image_processor import ImageLoadError, apply_image_transforms, load_image
from app.models.app_settings import AppSettings
from app.models.project import Project
from app.services import settings_service
from app.services.logging_config import get_logger
from app.ui.icons import make_app_icon, make_icon
from app.ui.state import AppState
from app.ui.theme import Theme, ThemeMode, build_qss, get_theme
from app.ui.welcome import WelcomeScreen

_log = get_logger(__name__)

_IMAGE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.svg);;All files (*.*)"
)
_ORG = "Tai"
_APP = "RhinestoneStudio"


class MainWindow(QMainWindow):
    """Top-level application window."""

    #: Emitted after the active project changes (new/open/import/close).
    project_changed = Signal()
    #: Emitted when the theme changes so injected widgets can retint.
    theme_changed = Signal()

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.app_settings = settings
        self.state = AppState()
        self._theme: Theme = get_theme(
            ThemeMode.LIGHT if settings.theme == "light" else ThemeMode.DARK
        )
        self._themed_actions: list[tuple[QAction, str]] = []
        self._thread_pool = QThreadPool.globalInstance()
        self._active_worker = None  # current GenerationWorker, if any
        self._command_stack = None  # app.core.history.CommandStack for the project

        self.setWindowTitle(__app_name__)
        self.setMinimumSize(1024, 680)
        self.setWindowIcon(make_app_icon(self._theme.palette.primary))

        self._build_central()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self._build_docks()
        self._build_canvas()
        self._build_panels()

        self.apply_theme(self._theme)
        self._restore_window_state()
        self._setup_autosave()
        self._show_welcome()
        self.refresh_recent()
        self._update_action_states()
        # Offer recovery once the window is up (deferred so construction never
        # blocks on a modal — also keeps headless tests non-blocking).
        QTimer.singleShot(0, self._maybe_offer_recovery)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_central(self) -> None:
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._welcome = WelcomeScreen(self._theme)
        self._welcome.import_image_requested.connect(self.action_import_image)
        self._welcome.new_project_requested.connect(self.action_new_project)
        self._welcome.open_project_requested.connect(self.action_open_project)
        self._welcome.recent_project_requested.connect(self.open_project_path)
        self._stack.addWidget(self._welcome)

        # Editor canvas host — the real canvas is injected in Phase 7.
        self._canvas_host = QWidget()
        host_layout = QVBoxLayout(self._canvas_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_placeholder = QLabel("")
        self._canvas_placeholder.setAlignment(Qt.AlignCenter)
        host_layout.addWidget(self._canvas_placeholder)
        self._canvas_widget: QWidget | None = None
        self._stack.addWidget(self._canvas_host)

    def _add_action(
        self,
        text: str,
        *,
        icon: str | None = None,
        shortcut: QKeySequence | QKeySequence.StandardKey | str | None = None,
        tip: str | None = None,
        checkable: bool = False,
        slot=None,
    ) -> QAction:
        act = QAction(text, self)
        if icon:
            act.setIcon(make_icon(icon, self._theme.palette.text))
            self._themed_actions.append((act, icon))
        if shortcut is not None:
            act.setShortcut(shortcut)
        if tip:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        act.setCheckable(checkable)
        if slot is not None:
            act.triggered.connect(slot)
        return act

    def _build_actions(self) -> None:
        # File
        self.act_new = self._add_action("New Project", icon="new",
                                        shortcut=QKeySequence.New, slot=self.action_new_project)
        self.act_open = self._add_action("Open Project…", icon="open",
                                         shortcut=QKeySequence.Open, slot=self.action_open_project)
        self.act_save = self._add_action("Save", icon="save",
                                         shortcut=QKeySequence.Save, slot=self.action_save)
        self.act_save_as = self._add_action("Save As…",
                                            shortcut=QKeySequence.SaveAs, slot=self.action_save_as)
        self.act_import = self._add_action("Import Image…", icon="import",
                                           shortcut="Ctrl+I", slot=self.action_import_image)

        # Image transforms (enabled once an image is present).
        self.act_rotate_left = self._add_action(
            "Rotate Left", icon="rotate_left", shortcut="Ctrl+[",
            tip="Rotate the image 90° counter-clockwise",
            slot=lambda: self._transform_image("rotate_left"))
        self.act_rotate_right = self._add_action(
            "Rotate Right", icon="rotate_right", shortcut="Ctrl+]",
            tip="Rotate the image 90° clockwise",
            slot=lambda: self._transform_image("rotate_right"))
        self.act_flip_h = self._add_action(
            "Flip Horizontal", icon="flip_h", tip="Mirror the image left–right",
            slot=lambda: self._transform_image("flip_h"))
        self.act_flip_v = self._add_action(
            "Flip Vertical", icon="flip_v", tip="Mirror the image top–bottom",
            slot=lambda: self._transform_image("flip_v"))
        self.act_reset_transforms = self._add_action(
            "Reset Image Adjustments", icon="reset",
            tip="Clear crop, rotation and flips",
            slot=lambda: self._transform_image("reset"))
        self.act_export_pkg = self._add_action("Export Production Package…",
                                               slot=self.action_export_package)
        self.act_settings = self._add_action("Settings…", icon="settings",
                                             shortcut="Ctrl+,", slot=self.action_settings)
        self.act_exit = self._add_action("Exit", shortcut="Ctrl+Q", slot=self.close)

        # Export submenu actions
        self.act_export_svg = self._add_action("Export SVG…", icon="export",
                                               shortcut="Ctrl+E", slot=lambda: self.action_export("svg"))
        self.act_export_cricut = self._add_action("Export Cricut SVG…",
                                                  slot=lambda: self.action_export("cricut"))
        self.act_export_dxf = self._add_action("Export DXF…", slot=lambda: self.action_export("dxf"))
        self.act_export_png = self._add_action("Export PNG…", slot=lambda: self.action_export("png"))
        self.act_export_pdf = self._add_action("Export PDF Production Sheet…",
                                               slot=lambda: self.action_export("pdf"))

        # Edit
        self.act_undo = self._add_action("Undo", icon="undo",
                                         shortcut=QKeySequence.Undo, slot=self.action_undo)
        self.act_redo = self._add_action("Redo", icon="redo",
                                         shortcut=QKeySequence.Redo, slot=self.action_redo)
        self.act_select_all = self._add_action("Select All",
                                               shortcut=QKeySequence.SelectAll, slot=self.action_select_all)
        self.act_deselect = self._add_action("Deselect All",
                                             shortcut="Ctrl+Shift+A", slot=self.action_deselect_all)
        self.act_delete = self._add_action("Delete Selected", icon="delete",
                                           shortcut=QKeySequence.Delete, slot=self.action_delete_selected)

        # View
        self.act_zoom_in = self._add_action("Zoom In", icon="zoom_in",
                                            shortcut=QKeySequence.ZoomIn, slot=self.action_zoom_in)
        self.act_zoom_out = self._add_action("Zoom Out", icon="zoom_out",
                                             shortcut=QKeySequence.ZoomOut, slot=self.action_zoom_out)
        self.act_fit = self._add_action("Fit to Screen", icon="fit",
                                        shortcut="Ctrl+0", slot=self.action_fit)
        self.act_actual = self._add_action("Actual Size", icon="actual_size",
                                           shortcut="Ctrl+1", slot=self.action_actual_size)
        self.act_show_image = self._add_action("Show Original Image", icon="image",
                                               checkable=True, slot=self.action_toggle_image)
        self.act_show_image.setChecked(True)
        self.act_show_grid = self._add_action("Show Grid", icon="grid",
                                              checkable=True, slot=self.action_toggle_grid)
        self.act_show_grid.setChecked(self.app_settings.show_grid)
        self.act_reset_layout = self._add_action("Reset Panel Layout", slot=self.action_reset_layout)

        # Theme actions (exclusive)
        self.act_theme_dark = self._add_action("Dark", checkable=True,
                                               slot=lambda: self.set_theme_mode(ThemeMode.DARK))
        self.act_theme_light = self._add_action("Light", checkable=True,
                                                slot=lambda: self.set_theme_mode(ThemeMode.LIGHT))
        theme_group = QActionGroup(self)
        theme_group.addAction(self.act_theme_dark)
        theme_group.addAction(self.act_theme_light)
        (self.act_theme_dark if self._theme.mode is ThemeMode.DARK else self.act_theme_light).setChecked(True)

        # Design
        self.act_generate = self._add_action("Generate Design", icon="generate",
                                             shortcut="Ctrl+G", slot=self.action_generate)
        self.act_resolve = self._add_action("Resolve Collisions", icon="resolve",
                                            slot=self.action_resolve_collisions)
        self.act_clear_trace = self._add_action("Clear Traced Area", icon="reset",
                                                tip="Remove the traced region so Generate uses the whole image",
                                                slot=self.action_clear_trace)

        # Editing tools (exclusive checkable group).
        from app.ui.canvas import CanvasTool

        self.act_tool_select = self._add_action("Select / Move", icon="select",
                                                shortcut="V", checkable=True,
                                                slot=lambda: self._set_tool(CanvasTool.SELECT))
        self.act_tool_add = self._add_action("Add Stone", icon="add",
                                             shortcut="A", checkable=True,
                                             slot=lambda: self._set_tool(CanvasTool.ADD))
        self.act_tool_delete = self._add_action("Delete Stone", icon="delete",
                                                shortcut="X", checkable=True,
                                                slot=lambda: self._set_tool(CanvasTool.DELETE))
        self.act_tool_brush = self._add_action("Brush Add", icon="brush",
                                               shortcut="B", checkable=True,
                                               slot=lambda: self._set_tool(CanvasTool.BRUSH))
        self.act_tool_eraser = self._add_action("Eraser", icon="eraser",
                                                shortcut="E", checkable=True,
                                                slot=lambda: self._set_tool(CanvasTool.ERASER))
        self.act_tool_crop = self._add_action("Crop Image", icon="crop",
                                              shortcut="C", checkable=True,
                                              tip="Drag a crop area, then Enter to apply (Esc to cancel)",
                                              slot=self._enter_crop_mode)
        self.act_tool_trace = self._add_action("Trace Area", icon="trace",
                                               shortcut="T", checkable=True,
                                               tip="Paint the areas that should receive rhinestones; "
                                                   "right-drag erases. Generate fills only traced areas",
                                               slot=lambda: self._set_tool(CanvasTool.TRACE))
        self._tool_group = QActionGroup(self)
        for a in (self.act_tool_select, self.act_tool_add, self.act_tool_delete,
                  self.act_tool_brush, self.act_tool_eraser, self.act_tool_crop,
                  self.act_tool_trace):
            self._tool_group.addAction(a)
        self.act_tool_select.setChecked(True)

        # Help
        self.act_about = self._add_action("About Rhinestone Studio", slot=self.action_about)

    def _build_menus(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("&File")
        m_file.addAction(self.act_new)
        m_file.addAction(self.act_open)
        m_file.addAction(self.act_save)
        m_file.addAction(self.act_save_as)
        self._recent_menu = m_file.addMenu("Recent Projects")
        m_file.addSeparator()
        m_file.addAction(self.act_import)
        m_export = m_file.addMenu("Export")
        m_export.addAction(self.act_export_svg)
        m_export.addAction(self.act_export_cricut)
        m_export.addAction(self.act_export_dxf)
        m_export.addAction(self.act_export_png)
        m_export.addAction(self.act_export_pdf)
        m_file.addAction(self.act_export_pkg)
        m_file.addSeparator()
        m_file.addAction(self.act_settings)
        m_file.addSeparator()
        m_file.addAction(self.act_exit)

        m_edit = mb.addMenu("&Edit")
        m_edit.addAction(self.act_undo)
        m_edit.addAction(self.act_redo)
        m_edit.addSeparator()
        m_edit.addAction(self.act_select_all)
        m_edit.addAction(self.act_deselect)
        m_edit.addAction(self.act_delete)

        m_image = mb.addMenu("&Image")
        m_image.addAction(self.act_import)
        m_image.addSeparator()
        m_image.addAction(self.act_rotate_left)
        m_image.addAction(self.act_rotate_right)
        m_image.addAction(self.act_flip_h)
        m_image.addAction(self.act_flip_v)
        m_image.addSeparator()
        m_image.addAction(self.act_reset_transforms)

        m_view = mb.addMenu("&View")
        m_view.addAction(self.act_zoom_in)
        m_view.addAction(self.act_zoom_out)
        m_view.addAction(self.act_fit)
        m_view.addAction(self.act_actual)
        m_view.addSeparator()
        m_view.addAction(self.act_show_image)
        m_view.addAction(self.act_show_grid)
        m_view.addSeparator()
        m_theme = m_view.addMenu("Theme")
        m_theme.addAction(self.act_theme_dark)
        m_theme.addAction(self.act_theme_light)
        m_view.addSeparator()
        m_view.addAction(self.act_reset_layout)

        m_design = mb.addMenu("&Design")
        m_design.addAction(self.act_generate)
        m_design.addAction(self.act_resolve)
        m_design.addAction(self.act_clear_trace)
        m_design.addSeparator()
        m_tools = m_design.addMenu("Tools")
        for a in (self.act_tool_select, self.act_tool_add, self.act_tool_delete,
                  self.act_tool_brush, self.act_tool_eraser, self.act_tool_crop,
                  self.act_tool_trace):
            m_tools.addAction(a)

        m_help = mb.addMenu("&Help")
        m_help.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonIconOnly)

        for act in (self.act_new, self.act_open, self.act_save, self.act_import):
            tb.addWidget(self._tool_button(act))
        tb.addSeparator()
        for act in (self.act_generate,):
            tb.addWidget(self._tool_button(act))
        tb.addSeparator()
        for act in (self.act_tool_select, self.act_tool_add, self.act_tool_delete,
                    self.act_tool_brush, self.act_tool_eraser, self.act_tool_crop,
                    self.act_tool_trace):
            tb.addWidget(self._tool_button(act))
        tb.addSeparator()
        for act in (self.act_undo, self.act_redo):
            tb.addWidget(self._tool_button(act))
        tb.addSeparator()
        for act in (self.act_zoom_out, self.act_zoom_in, self.act_fit, self.act_actual):
            tb.addWidget(self._tool_button(act))
        tb.addSeparator()
        for act in (self.act_show_image, self.act_show_grid):
            tb.addWidget(self._tool_button(act))

        # Spacer pushes export to the right.
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding,
                             spacer.sizePolicy().verticalPolicy().Preferred)
        tb.addWidget(spacer)
        tb.addWidget(self._tool_button(self.act_export_svg))

        self.addToolBar(Qt.TopToolBarArea, tb)
        self._toolbar = tb

    def _tool_button(self, action: QAction):
        from PySide6.QtWidgets import QToolButton

        btn = QToolButton()
        btn.setDefaultAction(action)
        btn.setAutoRaise(True)
        return btn

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_dims = QLabel("—")
        self._status_count = QLabel("0 stones")
        self._status_zoom = QLabel("100%")
        for w in (self._status_dims, self._status_count, self._status_zoom):
            w.setProperty("role", "caption")
            w.setMinimumWidth(90)
            w.setAlignment(Qt.AlignCenter)
        sb.addPermanentWidget(self._status_dims)
        sb.addPermanentWidget(self._status_count)
        sb.addPermanentWidget(self._status_zoom)
        sb.showMessage("Ready")

    def _build_docks(self) -> None:
        self._left_dock = QDockWidget("Design", self)
        self._left_dock.setObjectName("left_dock")
        self._left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._left_scroll = QScrollArea()
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setFrameShape(QScrollArea.NoFrame)
        self._left_scroll.setMinimumWidth(300)
        self._left_dock.setWidget(self._left_scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._left_dock)

        self._right_dock = QDockWidget("Properties", self)
        self._right_dock.setObjectName("right_dock")
        self._right_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._right_scroll = QScrollArea()
        self._right_scroll.setWidgetResizable(True)
        self._right_scroll.setFrameShape(QScrollArea.NoFrame)
        self._right_scroll.setMinimumWidth(240)
        self._right_dock.setWidget(self._right_scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self._right_dock)

        self.resizeDocks([self._left_dock], [320], Qt.Horizontal)

    def _build_canvas(self) -> None:
        from app.ui.canvas import CanvasScrollHost, CanvasView

        self._canvas = CanvasView(self._theme)
        self._canvas.set_show_grid(self.app_settings.show_grid)
        self._canvas.set_show_rulers(self.app_settings.show_rulers)
        self._canvas.set_background_opacity(self.app_settings.background_opacity)
        self._canvas.set_wheel_zooms(self.app_settings.wheel_zooms)
        self._canvas.zoom_changed.connect(self.set_status_zoom)
        self._canvas.cursor_moved.connect(self._on_cursor_moved)
        self._canvas.selection_changed.connect(self._on_selection_changed)
        self._canvas.command_produced.connect(self._on_command_produced)
        self._canvas.stones_mutated.connect(self._on_stones_mutated)
        self._canvas.crop_committed.connect(self._on_crop_committed)
        self._canvas.crop_cancelled.connect(self._on_crop_cancelled)
        self._canvas.push_failed.connect(self._on_push_failed)
        self._canvas.trace_changed.connect(self._on_trace_changed)
        self._canvas.trace_mask_replaced.connect(self._on_trace_changed)
        # The canvas travels inside its scrollbar rails; ``_canvas`` stays the
        # CanvasView so every call site keeps talking to the view itself.
        self._canvas_scroller = CanvasScrollHost(self._canvas)
        self.set_canvas(self._canvas_scroller)

    def _set_tool(self, tool) -> None:
        self._canvas.set_tool(tool)
        from app.ui.canvas import CanvasTool

        if tool is CanvasTool.TRACE:
            self.show_status_message(
                "Trace: drag to paint areas for rhinestones — right-drag erases. "
                "Generate fills only the traced areas", 6000
            )

    def _on_trace_changed(self) -> None:
        """A trace stroke / clear / history swap happened — sync state."""
        if not self.state.has_project:
            return
        self._canvas.resync_trace_to_project()
        self.state.mark_dirty(True)
        self._update_title()
        self._update_action_states()

    def _on_command_produced(self, command) -> None:
        """Record an already-applied edit command from the canvas."""
        if self._command_stack is not None:
            self._command_stack.push_applied(command)

    def _on_stones_mutated(self) -> None:
        if not self.state.has_project:
            return
        self.set_status_count(self.state.project.design.stone_count)
        self._stats_panel.refresh()
        self.state.mark_dirty(True)
        self._update_title()
        self._update_action_states()

    def _on_cursor_moved(self, x_mm: float, y_mm: float) -> None:
        if not self.state.has_project:
            return
        unit = self.state.project.units
        self.show_status_message(
            f"x: {x_mm / unit.mm_per_unit:.2f} {unit.symbol}   "
            f"y: {y_mm / unit.mm_per_unit:.2f} {unit.symbol}"
        )

    def _on_selection_changed(self, count: int) -> None:
        if count > 0:
            self.show_status_message(f"{count:,} selected", 3000)
        self._stats_panel.set_selection_count(count)
        self._update_action_states()

    def _build_panels(self) -> None:
        from app.ui.sidebar import DesignPanel
        from app.ui.statistics_panel import StatisticsPanel

        self._presets = settings_service.load_presets()
        self._design_panel = DesignPanel(self._presets)
        self._design_panel.import_image_requested.connect(self.action_import_image)
        self._design_panel.size_changed.connect(self._on_size_changed)
        self._design_panel.settings_changed.connect(self._on_settings_changed)
        self._design_panel.generate_requested.connect(self.action_generate)
        self._design_panel.cancel_requested.connect(self._cancel_generation)
        self._design_panel.set_transform_actions([
            self.act_rotate_left, self.act_rotate_right,
            self.act_flip_h, self.act_flip_v, self.act_reset_transforms,
        ])
        self.set_left_panel(self._design_panel)

        self._stats_panel = StatisticsPanel()
        self.set_right_panel(self._stats_panel)

        # Fit the left dock to what the sidebar actually needs at this font /
        # DPI, instead of a hardcoded 320 px that clips wide-font systems.
        self._fit_left_dock()

    def _fit_left_dock(self, *, only_grow: bool = False) -> None:
        """Resize the Design dock so the sidebar fits without clipping.

        The scroll area's minimum width is set to what the sidebar content
        needs, so the layout itself guarantees no horizontal scrollbar rides
        across the options; ``resizeDocks`` then nudges the dock to that size
        (fonts differ per system, so the old hardcoded 320 px clipped on
        wide-font machines).

        With ``only_grow`` a restored user-chosen width is respected unless it
        is too narrow for the content.
        """
        if self._design_panel is None:
            return
        needed = self._design_panel.minimumSizeHint().width()
        # Scrollbar + scroll-area frame + dock margins.
        needed += self._left_scroll.verticalScrollBar().sizeHint().width() + 16
        # Never eat more than 45% of the window, never less than the old 320.
        max_w = int(self.width() * 0.45)
        width = max(320, min(needed, max_w))
        if only_grow and self._left_dock.width() >= width:
            return
        self._left_scroll.setMinimumWidth(min(needed, max_w))
        self.resizeDocks([self._left_dock], [width], Qt.Horizontal)

    def _on_size_changed(self) -> None:
        if not self.state.has_project:
            return
        proj = self.state.project
        self._canvas.set_unit(proj.units)
        self._canvas.set_design(proj.design, fit=True)
        self._resample_trace(proj)
        self.set_status_dimensions(
            f"{proj.width_in_units:.2f} × {proj.height_in_units:.2f} {proj.units.symbol}"
        )
        self._stats_panel.refresh()
        self.state.mark_dirty(True)
        self._update_title()

    def _on_settings_changed(self) -> None:
        # Settings edits update stats/pitch; regeneration is explicit (button).
        self._stats_panel.refresh()
        self._sync_canvas_stone_spec()
        self.state.mark_dirty(True)
        self._update_title()

    def _sync_canvas_stone_spec(self) -> None:
        """Tell the canvas what stone the add/brush tools should create."""
        if not self.state.has_project:
            return
        s = self.state.project.settings
        try:
            diameter = s.effective_diameter()
        except Exception:
            return
        color = self.state.project.design.stones[0].color if self.state.project.design.stone_count else "#1F2933"
        self._canvas.set_new_stone_spec(diameter, s.stone_size_id, color, s.spacing_mm)
        self._canvas.set_push_all(s.push_all)

    def _resample_trace(self, proj: Project) -> None:
        """Adopt the trace mask to the project's (possibly changed) aspect."""
        if proj.trace_mask is None:
            return
        from app.core.trace_mask import resample_mask

        proj.trace_mask = resample_mask(proj.trace_mask, proj.width_mm, proj.height_mm)
        self._canvas.set_trace_mask(proj.trace_mask)

    def _on_push_failed(self) -> None:
        """Push All could not resolve the movement; the canvas restored the layout."""
        self._canvas.refresh_geometry()
        self._canvas.update()
        self.show_status_message(
            "Push All: spacing could not be maintained here — move undone", 5000
        )

    def _on_history_changed(self) -> None:
        stack = self._command_stack
        if stack is None:
            return
        self.act_undo.setEnabled(stack.can_undo)
        self.act_redo.setEnabled(stack.can_redo)
        self.act_undo.setText(f"Undo {stack.undo_label}".strip())
        self.act_redo.setText(f"Redo {stack.redo_label}".strip())

    # ------------------------------------------------------------------ #
    # Injection points for later phases
    # ------------------------------------------------------------------ #
    def set_canvas(self, widget: QWidget) -> None:
        """Install the real design canvas (Phase 7)."""
        layout = self._canvas_host.layout()
        if self._canvas_widget is not None:
            layout.removeWidget(self._canvas_widget)
            self._canvas_widget.deleteLater()
        self._canvas_placeholder.setVisible(False)
        layout.addWidget(widget)
        self._canvas_widget = widget

    def set_left_panel(self, widget: QWidget) -> None:
        self._left_scroll.setWidget(widget)

    def set_right_panel(self, widget: QWidget) -> None:
        self._right_scroll.setWidget(widget)

    # ------------------------------------------------------------------ #
    # Theme
    # ------------------------------------------------------------------ #
    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_qss(theme))
        # Retint action icons for the new palette.
        for act, icon_name in self._themed_actions:
            act.setIcon(make_icon(icon_name, theme.palette.text))
        # Give the generate action the accent colour for prominence.
        self.act_generate.setIcon(make_icon("generate", theme.palette.primary))
        self.setWindowIcon(make_app_icon(theme.palette.primary))
        self._canvas_placeholder.setStyleSheet(f"background-color: {theme.palette.canvas_bg};")
        self.theme_changed.emit()

    def set_theme_mode(self, mode: ThemeMode) -> None:
        if mode is self._theme.mode:
            return
        self.apply_theme(get_theme(mode))
        self.app_settings.theme = mode.value
        settings_service.save_settings(self.app_settings)

    @property
    def theme(self) -> Theme:
        return self._theme

    # ------------------------------------------------------------------ #
    # View state
    # ------------------------------------------------------------------ #
    def _show_welcome(self) -> None:
        self._stack.setCurrentWidget(self._welcome)
        self._left_dock.setVisible(False)
        self._right_dock.setVisible(False)
        self._toolbar.setVisible(False)
        self._update_title()

    def _show_editor(self) -> None:
        self._stack.setCurrentWidget(self._canvas_host)
        self._left_dock.setVisible(True)
        self._right_dock.setVisible(True)
        self._toolbar.setVisible(True)
        # The docks were hidden while the welcome screen was up, and
        # ``resizeDocks`` is ignored on hidden docks — so the sidebar's width
        # is only now adjustable. Fit it before the user sees a clipped panel.
        self._fit_left_dock()
        self._update_title()

    # ------------------------------------------------------------------ #
    # Status helpers (called by canvas/panels in later phases)
    # ------------------------------------------------------------------ #
    def set_status_zoom(self, percent: float) -> None:
        self._status_zoom.setText(f"{percent:.0f}%")

    def set_status_count(self, count: int) -> None:
        self._status_count.setText(f"{count:,} stones")

    def set_status_dimensions(self, text: str) -> None:
        self._status_dims.setText(text)

    def show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        self.statusBar().showMessage(message, timeout_ms)

    # ------------------------------------------------------------------ #
    # File actions
    # ------------------------------------------------------------------ #
    def action_new_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        project = Project.new("Untitled")
        project.settings.stone_size_id = self.app_settings.default_stone_size_id
        project.units = self.app_settings.default_units
        self.state.source_image = None  # a fresh project has no image
        self._set_project(project, file_path=None)
        self.show_status_message("New project created", 4000)

    def action_import_image(self) -> None:
        start_dir = self.app_settings.default_export_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Import Image", start_dir, _IMAGE_FILTER)
        if path:
            self._import_image_path(Path(path))

    def _import_image_path(self, path: Path) -> bool:
        try:
            loaded = load_image(path)
        except ImageLoadError as exc:
            self.show_error("Could not open image", str(exc))
            return False
        except Exception as exc:  # pragma: no cover - unexpected decode failure
            _log.exception("Unexpected image load error")
            self.show_error("Could not open image", f"An unexpected error occurred:\n{exc}")
            return False

        project = self.state.project or Project.new(path.stem)
        if not self.state.has_project:
            project.name = path.stem
            project.settings.stone_size_id = self.app_settings.default_stone_size_id
            project.units = self.app_settings.default_units

        project.image.source_path = path
        project.image.original_width_px = loaded.width
        project.image.original_height_px = loaded.height
        project.image.reset_transforms()

        # Default the physical size to the image aspect at a 10-inch width.
        default_w_mm = to_mm(10.0, project.units) if project.units != Unit.MM else 254.0
        w_mm, h_mm = fit_dimensions(loaded.width, loaded.height, default_w_mm, lock_aspect=True)
        project.set_physical_size(w_mm, h_mm)

        self.state.source_image = loaded
        self._set_project(project, file_path=self.state.file_path, dirty=True)
        self.show_status_message(f"Imported {path.name} ({loaded.width}×{loaded.height}px)", 5000)
        return True

    # ------------------------------------------------------------------ #
    # Image transforms (rotate / flip / reset — engine already applies them)
    # ------------------------------------------------------------------ #
    def _transform_image(self, kind: str) -> None:
        if not self.state.has_project or not self.state.has_image:
            return
        img = self.state.project.image
        had_crop = img.crop is not None
        if kind == "rotate_left":
            img.rotation = (img.rotation - 90) % 360
            label = "Rotated left"
        elif kind == "rotate_right":
            img.rotation = (img.rotation + 90) % 360
            label = "Rotated right"
        elif kind == "flip_h":
            img.flip_h = not img.flip_h
            label = "Flipped horizontally"
        elif kind == "flip_v":
            img.flip_v = not img.flip_v
            label = "Flipped vertically"
        elif kind == "reset":
            if img.rotation == 0 and not img.flip_h and not img.flip_v and img.crop is None:
                self.show_status_message("No image adjustments to reset", 3000)
                return
            img.reset_transforms()
            had_crop = False  # already cleared; don't double-report
            label = "Image adjustments reset"
        else:
            return
        # A crop is stored in oriented coordinates, so any orientation change
        # invalidates it — clear it rather than keep a mismatched rectangle.
        if kind in ("rotate_left", "rotate_right", "flip_h", "flip_v") and had_crop:
            img.crop = None
            label += " · crop cleared"
        self._refresh_image_view(recompute_size=True)
        self.state.mark_dirty(True)
        self._update_title()
        self.show_status_message(label, 3000)

    def _current_oriented_rgba(self):
        """Return the source image with the project's transforms applied (or None)."""
        src = self.state.source_image
        if src is None or not self.state.has_project or not self.state.project.has_image:
            return None
        img = self.state.project.image
        if img.rotation == 0 and not img.flip_h and not img.flip_v and img.crop is None:
            return src.rgba
        return apply_image_transforms(
            src.rgba, rotation=img.rotation, flip_h=img.flip_h,
            flip_v=img.flip_v, crop=img.crop,
        )

    def _refresh_image_view(self, *, recompute_size: bool) -> None:
        """Repaint the canvas background from the oriented image and re-fit size."""
        proj = self.state.project
        oriented = self._current_oriented_rgba()
        self._canvas.set_background_image(oriented)
        if recompute_size and oriented is not None and proj.lock_aspect:
            h, w = oriented.shape[:2]
            w_mm, h_mm = fit_dimensions(w, h, proj.width_mm, lock_aspect=True)
            proj.set_physical_size(w_mm, h_mm)
            self._resample_trace(proj)
        self._canvas.set_unit(proj.units)
        self._canvas.set_design(proj.design, fit=True)
        self._design_panel.bind_project(proj)  # resync width/height fields
        self.set_status_dimensions(
            f"{proj.width_in_units:.2f} × {proj.height_in_units:.2f} {proj.units.symbol}"
        )
        self._stats_panel.refresh()

    # ------------------------------------------------------------------ #
    # Crop tool
    # ------------------------------------------------------------------ #
    def _enter_crop_mode(self) -> None:
        from app.ui.canvas import CanvasTool

        if not self.state.has_project or not self.state.has_image:
            self.show_status_message("Import an image before cropping", 4000)
            self.act_tool_select.setChecked(True)
            self._canvas.set_tool(CanvasTool.SELECT)
            return
        self._canvas.set_tool(CanvasTool.CROP)
        self._canvas.begin_crop()
        self.show_status_message(
            "Drag to select a crop area, then press Enter to apply (Esc to cancel)", 6000
        )

    def _on_crop_committed(self, fx: float, fy: float, fw: float, fh: float) -> None:
        from app.ui.canvas import CanvasTool
        from app.core.history import CropCommand

        proj = self.state.project
        if proj is None or not proj.has_image:
            return
        # The crop fractions are relative to the *currently displayed* (possibly
        # already-cropped) image; compose with any existing crop so coordinates
        # stay in the oriented-image space that apply_image_transforms expects.
        oriented = self._current_oriented_rgba()
        if oriented is None:
            return
        ch, cw = oriented.shape[:2]
        ox, oy = (proj.image.crop[0], proj.image.crop[1]) if proj.image.crop else (0, 0)
        x = ox + int(round(fx * cw))
        y = oy + int(round(fy * ch))
        w = max(1, int(round(fw * cw)))
        h = max(1, int(round(fh * ch)))

        # Capture the state needed to make the crop a normal undoable command:
        # the previous crop rect, physical size and stones, plus the post-crop
        # values. The command swaps these atomically on undo/redo, then the
        # on_change hook refreshes the canvas background and fit so the view
        # stays consistent with the restored state.
        old_crop = proj.image.crop
        new_crop = (x, y, w, h)
        old_size = (proj.width_mm, proj.height_mm)
        old_stones = list(proj.design.stones)

        # Apply the crop live so the recomputed size/stones reflect STATE B.
        proj.image.crop = new_crop
        self._refresh_image_view(recompute_size=True)
        self._resample_trace(proj)
        new_size = (proj.width_mm, proj.height_mm)
        # Cropping the image field invalidates the existing stone layout (the
        # image-to-canvas mapping changed), so the design is cleared — a fresh
        # Generate rebuilds it against the cropped image. Undo restores the
        # previous stones exactly.
        new_stones = []
        proj.design.replace_all(new_stones)
        self._canvas.set_design(proj.design, fit=True)

        cmd = CropCommand(
            proj.image,
            proj,
            old_crop=old_crop,
            new_crop=new_crop,
            old_size=old_size,
            new_size=new_size,
            old_stones=old_stones,
            new_stones=new_stones,
            on_change=lambda: self._refresh_image_view(recompute_size=False),
            label=f"Crop to {w}×{h}px",
        )
        if self._command_stack is not None:
            self._command_stack.push_applied(cmd)
        else:
            cmd.redo()

        self.act_tool_select.setChecked(True)
        self._canvas.set_tool(CanvasTool.SELECT)
        self._refresh_image_view(recompute_size=False)
        self.state.mark_dirty(True)
        self._update_title()
        self.set_status_count(proj.design.stone_count)
        self._stats_panel.refresh()
        self.show_status_message(f"Cropped to {w}×{h}px", 4000)

    def _on_crop_cancelled(self) -> None:
        from app.ui.canvas import CanvasTool

        self.act_tool_select.setChecked(True)
        self._canvas.set_tool(CanvasTool.SELECT)
        self.show_status_message("Crop cancelled", 3000)

    def action_open_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        start_dir = self.app_settings.default_export_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", start_dir, "Rhinestone projects (*.rhinestone);;All files (*.*)"
        )
        if path:
            self.open_project_path(path)

    def open_project_path(self, path: str) -> None:
        from app.services.project_service import ProjectIOError, load_project

        try:
            project, image = load_project(path)
        except ProjectIOError as exc:
            self.show_error("Could not open project", str(exc))
            self.app_settings.remove_recent_project(str(path))
            settings_service.save_settings(self.app_settings)
            self.refresh_recent()
            return
        self.state.source_image = image
        self._set_project(project, file_path=Path(path), dirty=False)
        self.app_settings.add_recent_project(str(path))
        settings_service.save_settings(self.app_settings)
        self.refresh_recent()
        self.show_status_message(f"Opened {Path(path).name}", 4000)

    def action_save(self) -> bool:
        if not self.state.has_project:
            return False
        if self.state.file_path is None:
            return self.action_save_as()
        return self._write_project(self.state.file_path)

    def action_save_as(self) -> bool:
        if not self.state.has_project:
            return False
        start = self._suggested_save_path()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", str(start),
            "Rhinestone projects (*.rhinestone)",
        )
        if not path:
            return False
        return self._write_project(Path(path))

    def _suggested_save_path(self) -> Path:
        base = self.app_settings.default_export_dir or str(Path.home())
        name = self.state.title_name()
        return Path(base) / f"{name}.rhinestone"

    def _write_project(self, path: Path) -> bool:
        from app.services.project_service import ProjectIOError, save_project

        try:
            saved = save_project(self.state.project, path, self.state.source_image)
        except ProjectIOError as exc:
            self.show_error("Could not save project", str(exc))
            return False
        self.state.file_path = saved
        self.state.mark_dirty(False)
        self.app_settings.add_recent_project(str(saved))
        settings_service.save_settings(self.app_settings)
        self.refresh_recent()
        self._update_title()
        self.show_status_message(f"Saved {saved.name}", 4000)
        return True

    def action_export(self, kind: str) -> None:
        from app.services import export_service

        if not self._validate_before_export():
            return
        ext = {"svg": ".svg", "cricut": ".svg", "dxf": ".dxf", "png": ".png", "pdf": ".pdf"}[kind]
        label = {"svg": "SVG", "cricut": "Cricut SVG", "dxf": "DXF",
                 "png": "PNG preview", "pdf": "PDF production sheet"}[kind]
        suffix = "-cricut" if kind == "cricut" else ""
        start = self._export_start_path(f"{self.state.title_name()}{suffix}{ext}")
        path, _ = QFileDialog.getSaveFileName(self, f"Export {label}", str(start),
                                              f"{label} (*{ext})")
        if not path:
            return
        try:
            out = export_service.export_single(self.state.project, kind, path)
        except Exception as exc:
            self.show_error(f"Could not export {label}", str(exc))
            return
        self._remember_export_dir(Path(out).parent)
        self.show_status_message(f"Exported {out.name}", 5000)

    def action_export_package(self) -> None:
        from app.services import export_service

        if not self._validate_before_export():
            return
        start = self.app_settings.default_export_dir or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Choose Output Folder", start)
        if not directory:
            return
        # Place the package in its own subfolder named after the project.
        from app.services.export_service import _safe_stem

        target = Path(directory) / _safe_stem(self.state.title_name())
        if target.exists() and any(target.iterdir()) and self.app_settings.confirm_before_overwrite:
            resp = QMessageBox.question(
                self, "Folder not empty",
                f"'{target.name}' already exists and is not empty. Overwrite its files?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return
        try:
            result = export_service.export_production_package(self.state.project, target)
        except Exception as exc:
            self.show_error("Could not export package", str(exc))
            return
        self._remember_export_dir(Path(directory))
        self.show_status_message(
            f"Exported {len(result.files)} files to {target.name}", 6000
        )

    def action_settings(self) -> None:
        from app.ui.dialogs.settings_dialog import SettingsDialog

        dlg = SettingsDialog(self.app_settings, self)
        if dlg.exec() != QDialog.Accepted:
            return
        new = dlg.result_settings()
        self._apply_app_settings(new)
        settings_service.save_settings(new)
        self.show_status_message("Settings saved", 3000)

    def _apply_app_settings(self, new: AppSettings) -> None:
        """Adopt updated preferences and apply them live (theme, autosave, canvas)."""
        self.app_settings = new
        # Theme (retint everything + keep the View▸Theme radios in sync).
        desired = ThemeMode.LIGHT if new.theme == "light" else ThemeMode.DARK
        if desired is not self._theme.mode:
            self.apply_theme(get_theme(desired))
        self.act_theme_light.setChecked(desired is ThemeMode.LIGHT)
        self.act_theme_dark.setChecked(desired is ThemeMode.DARK)
        # Autosave cadence.
        self._autosave.configure(
            enabled=new.autosave_enabled, interval_sec=new.autosave_interval_sec
        )
        # Canvas preferences.
        self._canvas.set_show_grid(new.show_grid)
        self._canvas.set_show_rulers(new.show_rulers)
        self._canvas.set_background_opacity(new.background_opacity)
        self._canvas.set_wheel_zooms(new.wheel_zooms)
        self.act_show_grid.setChecked(new.show_grid)
        self._update_action_states()

    # -- export helpers ---------------------------------------------------- #
    def _validate_before_export(self) -> bool:
        from app.services import export_service

        if not self.state.has_project:
            return False
        result = export_service.validate_for_export(self.state.project)
        if not result.ok:
            self.show_error("Cannot export yet", "\n".join(result.errors))
            return False
        if result.warnings:
            resp = QMessageBox.warning(
                self, "Export warnings",
                "\n".join(result.warnings) + "\n\nExport anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if resp != QMessageBox.Yes:
                return False
        return True

    def _export_start_path(self, filename: str) -> Path:
        base = self.app_settings.default_export_dir or str(Path.home())
        return Path(base) / filename

    def _remember_export_dir(self, directory: Path) -> None:
        self.app_settings.default_export_dir = str(directory)
        settings_service.save_settings(self.app_settings)

    # ------------------------------------------------------------------ #
    # Autosave / recovery
    # ------------------------------------------------------------------ #
    def _setup_autosave(self) -> None:
        from app.services.autosave import AutosaveService

        self._autosave = AutosaveService(self.app_settings.autosave_interval_sec, self)
        self._autosave.configure(
            enabled=self.app_settings.autosave_enabled,
            interval_sec=self.app_settings.autosave_interval_sec,
        )
        self._autosave.set_provider(
            lambda: (self.state.project, self.state.source_image, self.state.dirty)
        )
        self._autosave.autosaved.connect(
            lambda _p: self.show_status_message("Autosaved recovery copy", 2000)
        )
        self._autosave.start()

    def _maybe_offer_recovery(self) -> None:
        from app.services.autosave import AutosaveService

        if not AutosaveService.has_recovery():
            return
        resp = QMessageBox.question(
            self, "Recover unsaved work?",
            "Rhinestone Studio found an autosaved design from a previous session.\n\n"
            "Would you like to recover it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if resp == QMessageBox.Yes:
            try:
                from app.services.project_service import load_project

                project, image = load_project(AutosaveService.recovery_file())
                self.state.source_image = image
                self._set_project(project, file_path=None, dirty=True)
                self.show_status_message("Recovered autosaved design", 5000)
            except Exception as exc:
                self.show_error("Could not recover", str(exc))
        else:
            AutosaveService.clear_recovery()

    # ------------------------------------------------------------------ #
    # Edit / View / Design actions (implemented in later phases)
    # ------------------------------------------------------------------ #
    def action_undo(self) -> None:
        if self._command_stack is not None and self._command_stack.undo():
            self._after_history_change()

    def action_redo(self) -> None:
        if self._command_stack is not None and self._command_stack.redo():
            self._after_history_change()

    def _after_history_change(self) -> None:
        self._canvas.refresh_geometry()
        self._canvas.refresh_selection()
        self._canvas.update()
        self._on_stones_mutated()

    def action_delete_selected(self) -> None:
        n = self._canvas.delete_selected()
        if n:
            self.show_status_message(f"Deleted {n:,} stone{'s' if n != 1 else ''}", 3000)

    def action_zoom_in(self) -> None:
        self._canvas.zoom_in()

    def action_zoom_out(self) -> None:
        self._canvas.zoom_out()

    def action_fit(self) -> None:
        self._canvas.fit_to_view()

    def action_actual_size(self) -> None:
        self._canvas.actual_size()

    def action_toggle_image(self, checked: bool) -> None:
        self._canvas.set_show_image(checked)

    def action_toggle_grid(self, checked: bool) -> None:
        self.app_settings.show_grid = checked
        self._canvas.set_show_grid(checked)
        settings_service.save_settings(self.app_settings)

    def action_select_all(self) -> None:
        self._canvas.select_all()

    def action_deselect_all(self) -> None:
        self._canvas.clear_selection()

    def action_reset_layout(self) -> None:
        self.resizeDocks([self._right_dock], [260], Qt.Horizontal)
        self._fit_left_dock()
        self._left_dock.setVisible(True)
        self._right_dock.setVisible(True)

    def action_clear_trace(self) -> None:
        if self._canvas.has_trace():
            self._canvas.clear_trace()
            self.show_status_message("Traced area cleared — Generate now uses the whole image", 4000)

    def action_generate(self) -> None:
        if self._active_worker is not None:
            return  # already running
        if not self.state.has_project or not self.state.has_image:
            self.show_status_message("Import an image before generating", 4000)
            return
        from app.core.trace_mask import mask_is_empty

        if self.state.project.trace_mask is not None and mask_is_empty(self.state.project.trace_mask):
            self.show_status_message(
                "The traced area is empty — paint it with the Trace tool "
                "(T) or clear the trace to use the whole image", 6000
            )
            return
        from app.core.image_processor import apply_image_transforms
        from app.ui.generation_worker import GenerationWorker

        proj = self.state.project
        img = self.state.source_image
        try:
            settings = self._design_panel.build_settings()
            settings.validate()
        except Exception as exc:
            self.show_error("Invalid settings", str(exc))
            return
        proj.settings = settings

        # Apply the project's crop/rotate/flip to the source pixels first.
        rgba = apply_image_transforms(
            img.rgba,
            rotation=proj.image.rotation,
            flip_h=proj.image.flip_h,
            flip_v=proj.image.flip_v,
            crop=proj.image.crop,
        )

        # Tone polarity and area inversion ride on the settings (they are part
        # of the project document, so they survive save/reopen).
        worker = GenerationWorker(
            rgba, settings, proj.width_mm, proj.height_mm,
            analysis_max_dim=self.app_settings.analysis_max_dim,
            trace_mask=proj.trace_mask,
        )
        worker.signals.progress.connect(self._design_panel.set_progress)
        worker.signals.batch.connect(self._on_generation_batch)
        worker.signals.finished.connect(self._on_generation_finished)
        worker.signals.cancelled.connect(self._on_generation_cancelled)
        worker.signals.failed.connect(self._on_generation_failed)
        self._active_worker = worker
        self._design_panel.set_generating(True)
        # Progressive display: stones stream onto the canvas as they are
        # placed (the design itself is only replaced when generation succeeds).
        self._canvas.begin_stone_stream()
        self.show_status_message("Generating rhinestone design…")
        self._thread_pool.start(worker)

    def _on_generation_batch(self, stones, count: int) -> None:
        """A batch of preview stones arrived from the generation worker."""
        self._canvas.append_stone_batch(stones)
        self.set_status_count(count)

    def _cancel_generation(self) -> None:
        if self._active_worker is not None:
            self._active_worker.cancel()
            self.show_status_message("Cancelling…", 2000)

    def _on_generation_finished(self, result) -> None:
        from app.core.history import CompositeCommand, ReplaceDesignCommand

        self._active_worker = None
        proj = self.state.project
        if proj is None:
            return
        old_stones = list(proj.design.stones)
        proj.design.replace_all(result.stones)

        # FEATURE 1 — a successful generation consumes the traced region: the
        # overlay and the project's mask are both cleared, so the next Generate
        # uses the whole image rather than silently re-using a stale gate. Only
        # the trace *state* is touched — the stones just placed and the source
        # image are untouched.
        #
        # A run that placed nothing keeps the trace: the user almost certainly
        # wants to adjust density and Generate again over the same painting,
        # so throwing it away would discard their work for no gain.
        cleared_trace = False
        commands: list = []
        if result.stone_count > 0:
            clear_cmd = self._canvas.clear_trace_command(label="Clear traced area")
            if clear_cmd is not None:
                commands.append(clear_cmd)
                cleared_trace = True
        commands.append(
            ReplaceDesignCommand(proj.design, old_stones, list(result.stones))
        )
        label = "Generate design (traced area)" if cleared_trace else "Generate design"
        if self._command_stack is not None:
            # One undoable step: Ctrl+Z restores the previous stones *and* the
            # traced area together, so undo never lands on a half-reverted state.
            self._command_stack.push_applied(
                commands[0] if len(commands) == 1 else CompositeCommand(commands, label=label)
            )
        self._design_panel.set_generating(False)
        # Replaces the streamed preview with the authoritative final stones.
        self._canvas.end_stone_stream()
        self._canvas.update()
        self.set_status_count(proj.design.stone_count)
        self._stats_panel.refresh()
        self.state.mark_dirty(True)
        self._update_title()
        self._update_action_states()
        self.show_status_message(self._generation_summary(result, cleared_trace), 6000)

    def _generation_summary(self, result, cleared_trace: bool) -> str:
        """User-facing summary of a finished generation pass."""
        count = result.stone_count
        if count == 0:
            return (
                "Generated 0 stones — nothing matched the current settings. "
                "Try a higher Density, a lighter subject, or check the traced area."
            )
        msg = f"Generated {count:,} stones"
        if result.was_limited:
            msg += f" (limited from {result.total_before_limit:,})"
        if cleared_trace:
            msg += " · traced area cleared"
        return msg

    def _on_generation_cancelled(self) -> None:
        self._active_worker = None
        self._design_panel.set_generating(False)
        # Discard the partial preview; the design was never touched, so the
        # previous valid design is what remains.
        self._canvas.end_stone_stream()
        self.set_status_count(
            self.state.project.design.stone_count if self.state.has_project else 0
        )
        self.show_status_message("Generation cancelled", 4000)

    def _on_generation_failed(self, message: str) -> None:
        self._active_worker = None
        self._design_panel.set_generating(False)
        self._canvas.end_stone_stream()
        self.set_status_count(
            self.state.project.design.stone_count if self.state.has_project else 0
        )
        self.show_error("Generation failed", message)

    def action_resolve_collisions(self) -> None:
        from app.core.collision import detect_collisions, resolve_collisions
        from app.core.history import ReplaceDesignCommand

        proj = self.state.project
        if proj is None or proj.design.stone_count == 0:
            return
        gap = proj.settings.spacing_mm
        report = detect_collisions(proj.design, gap)
        if not report.has_collisions:
            self.show_status_message("No collisions found", 4000)
            self._canvas.set_collisions(set())
            return
        old_stones = [s for s in (self.state.project.design.stones)]
        # Snapshot positions for undo (resolve mutates in place).
        import copy

        snapshot = [copy.copy(s) for s in old_stones]
        result = resolve_collisions(proj.design, gap)
        if self._command_stack is not None:
            self._command_stack.push_applied(
                ReplaceDesignCommand(proj.design, snapshot, list(proj.design.stones),
                                     label="Resolve collisions")
            )
        self._canvas.refresh_geometry()
        self._canvas.set_collisions(set())
        self._canvas.update()
        self._on_stones_mutated()
        parts = []
        if result.moved_count:
            parts.append(f"moved {result.moved_count:,}")
        if result.removed_count:
            parts.append(f"removed {result.removed_count:,}")
        self.show_status_message("Resolved collisions: " + (", ".join(parts) or "no change"), 5000)

    def action_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {__app_name__}",
            f"<h3>{__app_name__}</h3>"
            f"<p>Version {__version__}</p>"
            "<p>Convert images into editable rhinestone designs and export "
            "production-ready files.</p>"
            "<p>Offline-first. Your images never leave your computer.</p>",
        )

    # ------------------------------------------------------------------ #
    # Project plumbing
    # ------------------------------------------------------------------ #
    def _set_project(self, project: Project, file_path: Path | None, dirty: bool = False) -> None:
        from app.core.history import CommandStack

        self.state.project = project
        self.state.file_path = file_path
        self.state.dirty = dirty
        # Note: the caller owns state.source_image (set before calling); we do
        # not clear it here, so embedded/in-memory images survive _set_project.
        # Fresh undo history per project.
        self._command_stack = CommandStack()
        self._command_stack.on_change = self._on_history_changed
        # Sync the canvas with the project.
        self._canvas.set_unit(project.units)
        self._canvas.set_project(project)
        self._canvas.set_background_image(self._current_oriented_rgba())
        self._canvas.set_trace_mask(project.trace_mask)
        self._canvas.set_design(project.design, fit=True)
        self._sync_canvas_stone_spec()
        # Bind the panels.
        self._design_panel.bind_project(project)
        self._update_image_info()
        self._stats_panel.bind_project(project)
        self._show_editor()
        self._update_title()
        self._update_action_states()
        self.set_status_count(project.design.stone_count)
        self.set_status_dimensions(
            f"{project.width_in_units:.2f} × {project.height_in_units:.2f} {project.units.symbol}"
        )
        self.project_changed.emit()

    def refresh_recent(self) -> None:
        self._welcome.set_recent_projects(self.app_settings.recent_projects)
        self._recent_menu.clear()
        recent = [p for p in self.app_settings.recent_projects if Path(p).exists()]
        if not recent:
            empty = self._recent_menu.addAction("No recent projects")
            empty.setEnabled(False)
            return
        for p in recent:
            act = self._recent_menu.addAction(Path(p).name)
            act.setToolTip(p)
            act.triggered.connect(lambda _=False, path=p: self.open_project_path(path))
        self._recent_menu.addSeparator()
        clear = self._recent_menu.addAction("Clear Recent")
        clear.triggered.connect(self._clear_recent)

    def _clear_recent(self) -> None:
        self.app_settings.clear_recent_projects()
        settings_service.save_settings(self.app_settings)
        self.refresh_recent()

    def _update_action_states(self) -> None:
        has_project = self.state.has_project
        has_design = has_project and self.state.project.design.stone_count > 0
        has_image = has_project and self.state.project.has_image
        for act in (self.act_save, self.act_save_as, self.act_import, self.act_settings,
                    self.act_zoom_in, self.act_zoom_out, self.act_fit, self.act_actual,
                    self.act_show_image, self.act_show_grid):
            act.setEnabled(has_project)
        self.act_generate.setEnabled(has_image)
        for act in (self.act_rotate_left, self.act_rotate_right, self.act_flip_h,
                    self.act_flip_v, self.act_reset_transforms):
            act.setEnabled(has_image)
        for act in (self.act_export_svg, self.act_export_cricut, self.act_export_dxf,
                    self.act_export_png, self.act_export_pdf, self.act_export_pkg,
                    self.act_resolve, self.act_select_all,
                    self.act_deselect, self.act_delete):
            act.setEnabled(has_design)
        # Editing tools require an open project.
        for act in (self.act_tool_select, self.act_tool_add, self.act_tool_delete,
                    self.act_tool_brush, self.act_tool_eraser, self.act_tool_trace):
            act.setEnabled(has_project)
        # Cropping needs an image to crop.
        self.act_tool_crop.setEnabled(has_image)
        # Clearing needs a painted trace.
        self.act_clear_trace.setEnabled(self._canvas.has_trace())
        # Undo/redo reflect the command stack.
        stack = self._command_stack
        self.act_undo.setEnabled(bool(stack and stack.can_undo))
        self.act_redo.setEnabled(bool(stack and stack.can_redo))

    def _update_title(self) -> None:
        if not self.state.has_project:
            self.setWindowTitle(__app_name__)
            return
        star = "*" if self.state.dirty else ""
        self.setWindowTitle(f"{self.state.title_name()}{star} — {__app_name__}")

    def _update_image_info(self) -> None:
        img = self.state.source_image
        proj = self.state.project
        if img is None or proj is None or not proj.has_image:
            self._design_panel.set_image_info("No image imported", "")
            return
        path = proj.image.source_path
        name = path.name if path else "Embedded image"
        size_kb = img.file_size_bytes / 1024.0
        detail = f"{img.width}×{img.height}px · {img.format} · {size_kb:,.0f} KB"
        self._design_panel.set_image_info(name, detail)

    # ------------------------------------------------------------------ #
    # Dialog helpers
    # ------------------------------------------------------------------ #
    def show_error(self, title: str, message: str) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(title)
        box.setText(message)
        box.exec()

    def _confirm_discard_changes(self) -> bool:
        """Return True if it is safe to proceed (changes saved or discarded)."""
        if not self.state.dirty:
            return True
        resp = QMessageBox.question(
            self,
            "Save changes?",
            f"Save changes to “{self.state.title_name()}” before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if resp == QMessageBox.Save:
            return self.action_save()
        if resp == QMessageBox.Discard:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Window-state persistence
    # ------------------------------------------------------------------ #
    def _restore_window_state(self) -> None:
        from PySide6.QtCore import QSettings

        qs = QSettings(_ORG, _APP)
        geom = qs.value("window/geometry")
        state = qs.value("window/state")
        if isinstance(geom, QByteArray) and not geom.isEmpty():
            self.restoreGeometry(geom)
        else:
            self.resize(1360, 860)
        if isinstance(state, QByteArray) and not state.isEmpty():
            self.restoreState(state)
        # A restored dock width may predate this machine's font metrics —
        # widen it if the sidebar no longer fits.
        self._fit_left_dock(only_grow=True)

    def _save_window_state(self) -> None:
        from PySide6.QtCore import QSettings

        qs = QSettings(_ORG, _APP)
        qs.setValue("window/geometry", self.saveGeometry())
        qs.setValue("window/state", self.saveState())

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._active_worker is not None:
            self._active_worker.cancel()
        if not self._confirm_discard_changes():
            event.ignore()
            return
        # A clean exit: no unsaved work to recover.
        from app.services.autosave import AutosaveService

        self._autosave.stop()
        AutosaveService.clear_recovery()
        self._save_window_state()
        settings_service.save_settings(self.app_settings)
        _log.info("Application closing")
        super().closeEvent(event)
