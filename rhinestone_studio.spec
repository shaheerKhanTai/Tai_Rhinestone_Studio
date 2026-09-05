# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for Rhinestone Studio (Windows).

Build with::

    python build/make_icon.py            # generate the .ico first
    pyinstaller rhinestone_studio.spec    # → dist/Rhinestone Studio/

Produces a one-folder, windowed (no console) application. All writable runtime
state lives in the per-user data directory (see app/services/paths.py), so the
installation folder can be read-only.
"""
from pathlib import Path

_root = Path(SPECPATH)  # noqa: F821 - injected by PyInstaller
_icon = _root / "app" / "resources" / "icons" / "app.ico"

_datas = []
_resources = _root / "app" / "resources"
if _resources.exists():
    _datas.append((str(_resources), "app/resources"))

a = Analysis(
    ["app/main.py"],
    pathex=[str(_root)],
    binaries=[],
    datas=_datas,
    # OpenCV/ezdxf/reportlab pull these lazily; name them so they're bundled.
    hiddenimports=[
        "ezdxf",
        "reportlab.pdfgen",
        "reportlab.graphics",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtQuick", "PySide6.Qt3DCore"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Rhinestone Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed GUI application
    disable_windowed_traceback=False,
    icon=str(_icon) if _icon.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Rhinestone Studio",
)
