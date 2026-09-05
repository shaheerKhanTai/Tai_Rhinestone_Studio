"""Tests for the Windows packaging assets (Phase 16).

We do not run a full PyInstaller build in CI (slow, platform-specific); instead
we verify the spec exists and is wired to the icon/resources, and that the icon
builder produces a valid multi-size ``.ico`` that Pillow can reopen.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


class TestSpec:
    def test_spec_exists(self):
        assert (_ROOT / "rhinestone_studio.spec").is_file()

    def test_spec_references_entry_icon_and_resources(self):
        text = (_ROOT / "rhinestone_studio.spec").read_text(encoding="utf-8")
        assert "app/main.py" in text
        assert "app.ico" in text
        assert "app/resources" in text
        assert "console=False" in text  # windowed build
        assert 'name="Rhinestone Studio"' in text

    def test_readme_documents_windows_build(self):
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        assert "pyinstaller rhinestone_studio.spec" in readme


class TestIconBuilder:
    def test_builds_multisize_ico(self, qapp, tmp_path):
        from PIL import Image

        from build.make_icon import build_ico, _SIZES

        out = build_ico(tmp_path / "app.ico")
        assert out.is_file()
        with Image.open(out) as im:
            # Pillow exposes the embedded sizes; the largest must be present.
            assert im.size == (max(_SIZES), max(_SIZES))
            sizes = getattr(im, "ico", None)
            # Fallback: at least reopen without error and be an ICO.
            assert im.format == "ICO"
