"""Tests for the theme/token system and app settings (no GUI needed)."""
from __future__ import annotations

from app.ui.theme import (
    DARK_THEME,
    LIGHT_THEME,
    ThemeMode,
    build_qss,
    darken,
    get_theme,
    lighten,
    mix,
)


class TestColorMath:
    def test_mix_endpoints(self):
        assert mix("#000000", "#ffffff", 0.0) == "#000000"
        assert mix("#000000", "#ffffff", 1.0) == "#ffffff"

    def test_mix_midpoint(self):
        # 127.5 rounds to 128 (0x80).
        assert mix("#000000", "#ffffff", 0.5) == "#808080"

    def test_lighten_darken(self):
        assert lighten("#808080", 0.5) != "#808080"
        assert darken("#808080", 0.5) != "#808080"
        # Lighten increases channel values.
        assert int(lighten("#808080", 0.5).lstrip("#")[0:2], 16) > 0x80


class TestThemes:
    def test_dark_is_dark(self):
        assert DARK_THEME.palette.is_dark
        assert not LIGHT_THEME.palette.is_dark

    def test_get_theme(self):
        assert get_theme(ThemeMode.DARK) is DARK_THEME
        assert get_theme(ThemeMode.LIGHT) is LIGHT_THEME

    def test_qss_generates_for_both(self):
        for theme in (DARK_THEME, LIGHT_THEME):
            qss = build_qss(theme)
            assert "QPushButton" in qss
            assert "QMainWindow" in qss
            assert theme.palette.primary in qss
            assert len(qss) > 2000

    def test_derived_colors_valid_hex(self):
        for theme in (DARK_THEME, LIGHT_THEME):
            p = theme.palette
            for color in (p.hover_overlay, p.pressed_overlay, p.selected,
                          p.primary_hover, p.field_bg, p.canvas_bg, p.disabled_text):
                assert color.startswith("#") and len(color) == 7
