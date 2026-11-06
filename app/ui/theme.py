"""Design tokens and QSS generation for the Rhinestone Studio interface.

The palettes here are the contrast-audited *Professional Dark* and *Modern
Light* systems (all critical text/background pairs meet WCAG AA). Tokens are the
single source of truth; :func:`build_qss` renders a complete Qt stylesheet from
them, so the whole UI stays consistent and re-themeable. Nothing in the shipped
app depends on any external design service.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #


def _parse(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def mix(c1: str, c2: str, t: float) -> str:
    """Blend ``c1`` toward ``c2`` by ``t`` in ``[0, 1]``."""
    a = _parse(c1)
    b = _parse(c2)
    return _to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def lighten(color: str, t: float) -> str:
    return mix(color, "#ffffff", t)


def darken(color: str, t: float) -> str:
    return mix(color, "#000000", t)


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _parse(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


# --------------------------------------------------------------------------- #
# Token tables
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Palette:
    """Semantic colour tokens for one theme."""

    background: str
    surface: str
    elevated: str
    border: str
    text: str
    muted_text: str
    primary: str
    on_primary: str
    accent: str
    on_accent: str
    success: str
    warning: str
    error: str
    info: str

    # Derived interaction colours -------------------------------------------
    @property
    def is_dark(self) -> bool:
        return sum(_parse(self.background)) < 384  # < 128 avg

    @property
    def hover_overlay(self) -> str:
        return mix(self.surface, self.text, 0.08)

    @property
    def pressed_overlay(self) -> str:
        return mix(self.surface, self.text, 0.14)

    @property
    def selected(self) -> str:
        return mix(self.surface, self.primary, 0.28 if self.is_dark else 0.16)

    @property
    def primary_hover(self) -> str:
        return lighten(self.primary, 0.10) if self.is_dark else darken(self.primary, 0.08)

    @property
    def primary_pressed(self) -> str:
        return darken(self.primary, 0.14)

    @property
    def field_bg(self) -> str:
        return darken(self.surface, 0.18) if self.is_dark else self.surface

    @property
    def disabled_text(self) -> str:
        return mix(self.muted_text, self.background, 0.45)

    @property
    def canvas_bg(self) -> str:
        return darken(self.background, 0.25) if self.is_dark else mix(self.background, "#000000", 0.06)


DARK_PALETTE = Palette(
    background="#14181f",
    surface="#1c2129",
    elevated="#262d38",
    border="#2f3742",
    text="#e8edf3",
    muted_text="#a4b0be",
    primary="#4f86d6",
    on_primary="#0a0e14",
    accent="#3fae9a",
    on_accent="#06231e",
    success="#57b98a",
    warning="#d8a44b",
    error="#e07a7a",
    info="#6aa0e0",
)

LIGHT_PALETTE = Palette(
    background="#f8fafc",
    surface="#ffffff",
    elevated="#ffffff",
    border="#e2e8f0",
    text="#0f172a",
    muted_text="#546174",
    primary="#2563eb",
    on_primary="#ffffff",
    accent="#0e7490",
    on_accent="#ffffff",
    success="#15803d",
    warning="#b45309",
    error="#dc2626",
    info="#2563eb",
)


class Typography:
    """Font family and the type scale (px), matching the design system."""

    FAMILY = "Segoe UI, system-ui, sans-serif"
    MONO = "Cascadia Code, Consolas, monospace"
    DISPLAY = 28
    HEADING = 20
    SUBHEADING = 16
    BODY = 14
    CAPTION = 12
    BUTTON = 14
    LABEL = 13


class Spacing:
    """4px spacing scale."""

    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    XXL = 32


class Geometry:
    RADIUS_SM = 3
    RADIUS_MD = 6
    RADIUS_LG = 9
    CONTROL_SM = 28
    CONTROL_MD = 36
    CONTROL_LG = 44
    SIDEBAR_MIN = 240
    SIDEBAR_DEFAULT = 300
    TOOLBAR_H = 44


class ThemeMode(enum.Enum):
    DARK = "dark"
    LIGHT = "light"

    @property
    def label(self) -> str:
        return "Dark" if self is ThemeMode.DARK else "Light"


@dataclass(frozen=True, slots=True)
class Theme:
    """A named theme: a palette plus the shared type/space/geometry scales."""

    mode: ThemeMode
    palette: Palette

    @property
    def name(self) -> str:
        return f"Rhinestone Studio {self.mode.label}"


DARK_THEME = Theme(ThemeMode.DARK, DARK_PALETTE)
LIGHT_THEME = Theme(ThemeMode.LIGHT, LIGHT_PALETTE)

_THEMES = {ThemeMode.DARK: DARK_THEME, ThemeMode.LIGHT: LIGHT_THEME}


def get_theme(mode: ThemeMode) -> Theme:
    return _THEMES[mode]


# --------------------------------------------------------------------------- #
# QSS generation
# --------------------------------------------------------------------------- #


def _rule(selector: str, props: dict[str, str]) -> str:
    body = "".join(f"    {k}: {v};\n" for k, v in props.items())
    return f"{selector} {{\n{body}}}\n"


def build_qss(theme: Theme) -> str:
    """Render a complete Qt stylesheet from ``theme``'s tokens."""
    p = theme.palette
    t = Typography
    g = Geometry
    rules: list[str] = []

    # Base ------------------------------------------------------------------
    rules.append(_rule("QWidget", {
        "background-color": p.background,
        "color": p.text,
        "font-family": t.FAMILY,
        "font-size": f"{t.BODY}px",
        "selection-background-color": p.primary,
        "selection-color": p.on_primary,
    }))
    rules.append(_rule("QMainWindow, QDialog", {"background-color": p.background}))
    rules.append(_rule("QToolTip", {
        "background-color": p.elevated,
        "color": p.text,
        "border": f"1px solid {p.border}",
        "padding": "4px 8px",
        "border-radius": f"{g.RADIUS_SM}px",
    }))

    # Labels ----------------------------------------------------------------
    rules.append(_rule("QLabel", {"background-color": "transparent", "color": p.text}))
    rules.append(_rule("QLabel[role=\"heading\"]", {
        "font-size": f"{t.HEADING}px", "font-weight": "600", "color": p.text,
    }))
    rules.append(_rule("QLabel[role=\"subheading\"]", {
        "font-size": f"{t.SUBHEADING}px", "font-weight": "600", "color": p.text,
    }))
    rules.append(_rule("QLabel[role=\"caption\"]", {
        "font-size": f"{t.CAPTION}px", "color": p.muted_text,
    }))
    rules.append(_rule("QLabel[role=\"muted\"]", {"color": p.muted_text}))
    rules.append(_rule("QLabel[role=\"display\"]", {
        "font-size": f"{t.DISPLAY}px", "font-weight": "600", "color": p.text,
    }))

    # Push buttons ----------------------------------------------------------
    rules.append(_rule("QPushButton", {
        "background-color": p.elevated,
        "color": p.text,
        "border": f"1px solid {p.border}",
        "border-radius": f"{g.RADIUS_MD}px",
        "padding": "7px 16px",
        "font-size": f"{t.BUTTON}px",
        "font-weight": "600",
        "min-height": "22px",
    }))
    rules.append(_rule("QPushButton:hover", {"background-color": p.hover_overlay,
                                             "border-color": mix(p.border, p.text, 0.25)}))
    rules.append(_rule("QPushButton:pressed", {"background-color": p.pressed_overlay}))
    rules.append(_rule("QPushButton:focus", {"border": f"2px solid {p.primary}", "outline": "none"}))
    rules.append(_rule("QPushButton:disabled", {
        "background-color": mix(p.surface, p.background, 0.5),
        "color": p.disabled_text,
        "border-color": mix(p.border, p.background, 0.4),
    }))
    # Primary variant
    rules.append(_rule('QPushButton[variant="primary"]', {
        "background-color": p.primary, "color": p.on_primary, "border": "1px solid transparent",
    }))
    rules.append(_rule('QPushButton[variant="primary"]:hover', {"background-color": p.primary_hover}))
    rules.append(_rule('QPushButton[variant="primary"]:pressed', {"background-color": p.primary_pressed}))
    rules.append(_rule('QPushButton[variant="primary"]:disabled', {
        "background-color": mix(p.primary, p.background, 0.55), "color": rgba(p.on_primary, 0.6),
    }))
    # Destructive variant
    rules.append(_rule('QPushButton[variant="destructive"]', {
        "background-color": "transparent", "color": p.error, "border": f"1px solid {mix(p.error, p.background, 0.3)}",
    }))
    rules.append(_rule('QPushButton[variant="destructive"]:hover', {
        "background-color": rgba(p.error, 0.14),
    }))
    # Tertiary / flat
    rules.append(_rule('QPushButton[variant="tertiary"]', {
        "background-color": "transparent", "color": p.primary, "border": "1px solid transparent",
    }))
    rules.append(_rule('QPushButton[variant="tertiary"]:hover', {"background-color": rgba(p.primary, 0.12)}))

    # Line edits / spin boxes ----------------------------------------------
    field = {
        "background-color": p.field_bg,
        "color": p.text,
        "border": f"1px solid {p.border}",
        "border-radius": f"{g.RADIUS_MD}px",
        "padding": "6px 10px",
        "min-height": "20px",
        "selection-background-color": p.primary,
    }
    for sel in ("QLineEdit", "QSpinBox", "QDoubleSpinBox", "QPlainTextEdit", "QTextEdit"):
        rules.append(_rule(sel, field))
    rules.append(_rule("QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus", {
        "border": f"2px solid {p.primary}", "padding": "5px 9px",
    }))
    rules.append(_rule("QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled", {
        "color": p.disabled_text, "background-color": mix(p.field_bg, p.background, 0.5),
    }))
    rules.append(_rule('QLineEdit[state="error"], QDoubleSpinBox[state="error"], QSpinBox[state="error"]', {
        "border": f"2px solid {p.error}", "padding": "5px 9px",
    }))
    rules.append(_rule(
        "QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, "
        "QDoubleSpinBox::down-button",
        {"width": "16px", "background-color": "transparent", "border": "none"},
    ))

    # Combo box -------------------------------------------------------------
    rules.append(_rule("QComboBox", field))
    rules.append(_rule("QComboBox:focus", {"border": f"2px solid {p.primary}", "padding": "5px 9px"}))
    rules.append(_rule("QComboBox::drop-down", {"border": "none", "width": "22px"}))
    rules.append(_rule("QComboBox QAbstractItemView", {
        "background-color": p.elevated,
        "color": p.text,
        "border": f"1px solid {p.border}",
        "border-radius": f"{g.RADIUS_MD}px",
        "selection-background-color": p.primary,
        "selection-color": p.on_primary,
        "outline": "none",
        "padding": "4px",
    }))

    # Checkboxes / radios ---------------------------------------------------
    rules.append(_rule("QCheckBox, QRadioButton", {"spacing": "8px", "background": "transparent"}))
    rules.append(_rule("QCheckBox::indicator, QRadioButton::indicator", {
        "width": "16px", "height": "16px", "border": f"1px solid {mix(p.border, p.text, 0.2)}",
        "background-color": p.field_bg,
    }))
    rules.append(_rule("QCheckBox::indicator", {"border-radius": f"{g.RADIUS_SM}px"}))
    rules.append(_rule("QRadioButton::indicator", {"border-radius": "8px"}))
    rules.append(_rule("QCheckBox::indicator:checked, QRadioButton::indicator:checked", {
        "background-color": p.primary, "border-color": p.primary,
    }))
    rules.append(_rule("QCheckBox::indicator:hover, QRadioButton::indicator:hover", {
        "border-color": p.primary,
    }))

    # Sliders ---------------------------------------------------------------
    rules.append(_rule("QSlider::groove:horizontal", {
        "height": "4px", "background": p.border, "border-radius": "2px",
    }))
    rules.append(_rule("QSlider::sub-page:horizontal", {
        "background": p.primary, "height": "4px", "border-radius": "2px",
    }))
    rules.append(_rule("QSlider::handle:horizontal", {
        "background": p.text if not p.is_dark else "#ffffff",
        "border": f"2px solid {p.primary}",
        "width": "14px", "height": "14px", "margin": "-6px 0", "border-radius": "8px",
    }))
    rules.append(_rule("QSlider::handle:horizontal:hover", {"border-color": p.primary_hover}))

    # Group box -------------------------------------------------------------
    rules.append(_rule("QGroupBox", {
        "background-color": "transparent",
        "border": f"1px solid {p.border}",
        "border-radius": f"{g.RADIUS_LG}px",
        "margin-top": "18px",
        "padding": "12px",
        "font-weight": "600",
    }))
    rules.append(_rule("QGroupBox::title", {
        "subcontrol-origin": "margin", "left": "12px", "padding": "0 4px",
        "color": p.muted_text, "font-size": f"{t.LABEL}px",
    }))

    # Cards -----------------------------------------------------------------
    rules.append(_rule("QFrame#card", {
        "background-color": p.surface,
        "border": f"1px solid {p.border}",
        "border-radius": f"{g.RADIUS_LG}px",
    }))

    # Error banner (the persistent inline failure notice in the Design panel) --
    # Addressed by object name rather than by descendant selectors
    # (``QFrame[role="error"] QToolButton`` and friends). Qt matches descendant
    # selectors by walking the ancestry of every widget whenever the sheet is
    # applied, and this sheet is re-applied to the whole application on every
    # theme change; ID selectors are a direct lookup. Same appearance, without
    # the per-widget cost.
    rules.append(_rule("QFrame#errorBanner", {
        "background-color": rgba(p.error, 0.14 if p.is_dark else 0.08),
        "border": f"1px solid {rgba(p.error, 0.55)}",
        "border-radius": f"{g.RADIUS_MD}px",
    }))
    rules.append(_rule("QLabel#errorBannerTitle", {
        "color": p.error, "font-weight": "600", "background": "transparent",
    }))
    rules.append(_rule("QLabel#errorBannerDetail", {
        "color": mix(p.error, p.text, 0.45), "background": "transparent",
        "font-size": f"{t.CAPTION}px",
    }))
    rules.append(_rule("QToolButton#errorBannerClose", {
        "color": p.error, "font-weight": "600", "padding": "0px 4px",
        "min-width": "16px", "min-height": "16px",
    }))
    rules.append(_rule("QToolButton#errorBannerClose:hover", {
        "background-color": rgba(p.error, 0.22), "border-color": "transparent",
    }))

    # Menu bar / menus ------------------------------------------------------
    rules.append(_rule("QMenuBar", {
        "background-color": p.surface, "color": p.text,
        "border-bottom": f"1px solid {p.border}", "padding": "2px 4px",
    }))
    rules.append(_rule("QMenuBar::item", {"background": "transparent", "padding": "6px 10px",
                                          "border-radius": f"{g.RADIUS_SM}px"}))
    rules.append(_rule("QMenuBar::item:selected", {"background-color": p.hover_overlay}))
    rules.append(_rule("QMenu", {
        "background-color": p.elevated, "color": p.text,
        "border": f"1px solid {p.border}", "border-radius": f"{g.RADIUS_MD}px", "padding": "6px",
    }))
    rules.append(_rule("QMenu::item", {"padding": "6px 24px 6px 12px", "border-radius": f"{g.RADIUS_SM}px"}))
    rules.append(_rule("QMenu::item:selected", {"background-color": p.primary, "color": p.on_primary}))
    rules.append(_rule("QMenu::item:disabled", {"color": p.disabled_text}))
    rules.append(_rule("QMenu::separator", {"height": "1px", "background": p.border, "margin": "6px 8px"}))

    # Toolbar ---------------------------------------------------------------
    rules.append(_rule("QToolBar", {
        "background-color": p.surface, "border": "none",
        "border-bottom": f"1px solid {p.border}", "spacing": "4px", "padding": "4px 8px",
    }))
    rules.append(_rule("QToolBar::separator", {"width": "1px", "background": p.border, "margin": "6px 6px"}))
    rules.append(_rule("QToolButton", {
        "background-color": "transparent", "color": p.text,
        "border": "1px solid transparent", "border-radius": f"{g.RADIUS_MD}px",
        "padding": "6px", "min-width": "24px", "min-height": "24px",
    }))
    rules.append(_rule("QToolButton:hover", {"background-color": p.hover_overlay,
                                             "border-color": p.border}))
    rules.append(_rule("QToolButton:pressed, QToolButton:checked", {
        "background-color": rgba(p.primary, 0.2), "border-color": p.primary,
    }))
    rules.append(_rule("QToolButton:disabled", {"color": p.disabled_text}))

    # Status bar ------------------------------------------------------------
    rules.append(_rule("QStatusBar", {
        "background-color": p.surface, "color": p.muted_text,
        "border-top": f"1px solid {p.border}",
    }))
    rules.append(_rule("QStatusBar::item", {"border": "none"}))

    # Dock widgets ----------------------------------------------------------
    rules.append(_rule("QDockWidget", {
        "color": p.text, "titlebar-close-icon": "none", "titlebar-normal-icon": "none",
    }))
    rules.append(_rule("QDockWidget::title", {
        "background-color": p.surface, "padding": "8px 12px",
        "border-bottom": f"1px solid {p.border}", "font-weight": "600",
    }))

    # Scroll bars -----------------------------------------------------------
    for orient, size_prop in (("vertical", "width"), ("horizontal", "height")):
        rules.append(_rule(f"QScrollBar:{orient}", {
            "background": "transparent", size_prop: "12px", "margin": "0",
        }))
        handle = "vertical" if orient == "vertical" else "horizontal"
        rules.append(_rule(f"QScrollBar::handle:{handle}", {
            "background": mix(p.surface, p.text, 0.22),
            "border-radius": "5px",
            "min-height": "28px" if orient == "vertical" else "0",
            "min-width": "28px" if orient == "horizontal" else "0",
            "margin": "2px",
        }))
        rules.append(_rule(f"QScrollBar::handle:{handle}:hover", {"background": mix(p.surface, p.text, 0.34)}))
        rules.append(_rule(
            f"QScrollBar::add-line:{orient}, QScrollBar::sub-line:{orient}",
            {"height": "0", "width": "0", "background": "none", "border": "none"},
        ))
        rules.append(_rule(
            f"QScrollBar::add-page:{orient}, QScrollBar::sub-page:{orient}",
            {"background": "none"},
        ))

    # Progress bar ----------------------------------------------------------
    rules.append(_rule("QProgressBar", {
        "background-color": p.field_bg, "border": "none", "border-radius": "3px",
        "height": "6px", "text-align": "center", "color": p.muted_text,
    }))
    rules.append(_rule("QProgressBar::chunk", {"background-color": p.primary, "border-radius": "3px"}))

    # Tabs ------------------------------------------------------------------
    rules.append(_rule("QTabWidget::pane", {"border": f"1px solid {p.border}",
                                            "border-radius": f"{g.RADIUS_MD}px", "top": "-1px"}))
    rules.append(_rule("QTabBar::tab", {
        "background": "transparent", "color": p.muted_text,
        "padding": "8px 14px", "border": "none", "font-weight": "600",
    }))
    rules.append(_rule("QTabBar::tab:selected", {
        "color": p.text, "border-bottom": f"2px solid {p.accent}",
    }))
    rules.append(_rule("QTabBar::tab:hover", {"color": p.text}))

    # Lists / tables --------------------------------------------------------
    rules.append(_rule("QListWidget, QTreeWidget, QTableWidget", {
        "background-color": p.surface, "border": f"1px solid {p.border}",
        "border-radius": f"{g.RADIUS_MD}px", "outline": "none", "padding": "4px",
    }))
    rules.append(_rule("QListWidget::item, QTreeWidget::item", {
        "padding": "7px 8px", "border-radius": f"{g.RADIUS_SM}px",
    }))
    rules.append(_rule("QListWidget::item:hover, QTreeWidget::item:hover", {
        "background-color": p.hover_overlay,
    }))
    rules.append(_rule("QListWidget::item:selected, QTreeWidget::item:selected", {
        "background-color": p.selected, "color": p.text,
    }))
    rules.append(_rule("QHeaderView::section", {
        "background-color": p.surface, "color": p.muted_text, "padding": "6px 8px",
        "border": "none", "border-bottom": f"1px solid {p.border}", "font-weight": "600",
    }))

    # Splitter --------------------------------------------------------------
    rules.append(_rule("QSplitter::handle", {"background-color": p.border}))
    rules.append(_rule("QSplitter::handle:horizontal", {"width": "1px"}))
    rules.append(_rule("QSplitter::handle:vertical", {"height": "1px"}))

    return "".join(rules)
