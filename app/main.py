"""Application entry point.

Run with ``python -m app.main``. Sets up logging, high-DPI behaviour, the
application-wide exception hook and the main window. A hidden ``--screenshot``
mode renders the window to a PNG for automated visual checks without a user
present (used during development).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app import __app_name__, __organization__, __version__


def _install_excepthook(app_window_getter) -> None:
    """Route uncaught exceptions to the log and a friendly dialog."""
    log = logging.getLogger("app.crash")

    def hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        try:
            from PySide6.QtWidgets import QMessageBox

            win = app_window_getter()
            box = QMessageBox(win)
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Unexpected error")
            box.setText(
                "Rhinestone Studio hit an unexpected problem.\n\n"
                "Your work is preserved where possible. Technical details have "
                "been written to the log file."
            )
            box.setDetailedText(f"{exc_type.__name__}: {exc_value}")
            box.exec()
        except Exception:  # pragma: no cover - dialog must never mask the crash
            pass

    sys.excepthook = hook


def _configure_high_dpi() -> None:
    """Enable crisp scaling on high-DPI Windows displays."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:  # pragma: no cover - policy unavailable on some builds
        pass


def build_application(argv: list[str]):
    """Create the QApplication and the main window; return ``(app, window)``."""
    from PySide6.QtWidgets import QApplication

    from app.models.app_settings import AppSettings
    from app.services import settings_service
    from app.services.logging_config import configure_logging
    from app.ui.main_window import MainWindow

    configure_logging(level=logging.INFO)

    _configure_high_dpi()
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(__app_name__)
    app.setApplicationDisplayName(__app_name__)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(__organization__)

    settings: AppSettings = settings_service.load_settings()
    window = MainWindow(settings)
    return app, window


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(prog="rhinestone-studio", add_help=True)
    parser.add_argument("--screenshot", metavar="PATH", default=None,
                        help="Render the window to PATH and exit (development aid).")
    parser.add_argument("--import", dest="import_image", metavar="IMAGE", default=None,
                        help="Import an image on launch.")
    parser.add_argument("--theme", choices=("dark", "light"), default=None)
    args, qt_argv = parser.parse_known_args(argv[1:])

    app, window = build_application([argv[0], *qt_argv])

    if args.theme:
        from app.ui.theme import ThemeMode

        window.set_theme_mode(ThemeMode.LIGHT if args.theme == "light" else ThemeMode.DARK)

    _install_excepthook(lambda: window)

    if args.import_image:
        window._import_image_path(Path(args.import_image))

    if args.screenshot:
        return _run_screenshot(app, window, Path(args.screenshot))

    window.show()
    return app.exec()


def _run_screenshot(app, window, out_path: Path) -> int:
    """Render the window offscreen to a PNG (development/QA aid)."""
    from PySide6.QtCore import QTimer

    window.resize(1360, 860)
    window.show()

    def capture() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(out_path))
        app.quit()

    QTimer.singleShot(350, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
