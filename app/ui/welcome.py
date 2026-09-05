"""Welcome screen shown when no project is open.

A clean, centred hero: the logo, the product name, the two primary starting
actions, a recent-projects list and a compact quick-start guide. Emits signals
the main window connects to real handlers.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app import __version__
from app.ui.icons import make_app_logo
from app.ui.theme import Theme


class WelcomeScreen(QWidget):
    """Landing view with primary actions and recent projects."""

    import_image_requested = Signal()
    new_project_requested = Signal()
    open_project_requested = Signal()
    recent_project_requested = Signal(str)

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 48, 48, 48)
        outer.addStretch(1)

        center = QHBoxLayout()
        center.addStretch(1)

        card = QFrame()
        card.setObjectName("card")
        card.setMaximumWidth(760)
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(40, 36, 40, 36)
        card_l.setSpacing(8)

        logo = QLabel()
        logo.setPixmap(
            make_app_logo(72, self._theme.palette.primary, self._theme.palette.accent)
        )
        logo.setAlignment(Qt.AlignCenter)
        card_l.addWidget(logo)

        title = QLabel("Rhinestone Studio")
        title.setProperty("role", "display")
        title.setAlignment(Qt.AlignCenter)
        card_l.addWidget(title)

        subtitle = QLabel("Create your next rhinestone design.")
        subtitle.setProperty("role", "muted")
        subtitle.setAlignment(Qt.AlignCenter)
        card_l.addWidget(subtitle)

        card_l.addSpacing(20)

        # Primary actions.
        actions = QHBoxLayout()
        actions.setSpacing(12)
        actions.addStretch(1)
        import_btn = QPushButton("Import Image")
        import_btn.setProperty("variant", "primary")
        import_btn.setMinimumWidth(160)
        import_btn.setMinimumHeight(40)
        import_btn.clicked.connect(self.import_image_requested)
        new_btn = QPushButton("New Project")
        new_btn.setMinimumWidth(160)
        new_btn.setMinimumHeight(40)
        new_btn.clicked.connect(self.new_project_requested)
        open_btn = QPushButton("Open Project")
        open_btn.setMinimumWidth(160)
        open_btn.setMinimumHeight(40)
        open_btn.clicked.connect(self.open_project_requested)
        actions.addWidget(import_btn)
        actions.addWidget(new_btn)
        actions.addWidget(open_btn)
        actions.addStretch(1)
        card_l.addLayout(actions)

        card_l.addSpacing(24)

        # Two columns: recent projects and quick start.
        columns = QHBoxLayout()
        columns.setSpacing(28)

        recent_col = QVBoxLayout()
        recent_col.setSpacing(8)
        recent_label = QLabel("Recent Projects")
        recent_label.setProperty("role", "subheading")
        recent_col.addWidget(recent_label)
        self._recent_list = QListWidget()
        self._recent_list.setFrameShape(QFrame.NoFrame)
        self._recent_list.itemActivated.connect(self._on_recent_activated)
        self._recent_list.itemClicked.connect(self._on_recent_activated)
        recent_col.addWidget(self._recent_list, 1)
        self._recent_empty = QLabel("No recent projects yet.")
        self._recent_empty.setProperty("role", "caption")
        recent_col.addWidget(self._recent_empty)
        columns.addLayout(recent_col, 1)

        quick_col = QVBoxLayout()
        quick_col.setSpacing(8)
        quick_label = QLabel("Quick Start")
        quick_label.setProperty("role", "subheading")
        quick_col.addWidget(quick_label)
        steps = [
            "1.  Import an image",
            "2.  Set your design size",
            "3.  Choose your rhinestone",
            "4.  Generate",
            "5.  Edit",
            "6.  Export",
        ]
        for s in steps:
            lbl = QLabel(s)
            lbl.setProperty("role", "muted")
            quick_col.addWidget(lbl)
        quick_col.addStretch(1)
        columns.addLayout(quick_col, 1)

        card_l.addLayout(columns)

        center.addWidget(card)
        center.addStretch(1)
        outer.addLayout(center)
        outer.addStretch(1)

        version = QLabel(f"Version {__version__}")
        version.setProperty("role", "caption")
        version.setAlignment(Qt.AlignCenter)
        outer.addWidget(version)

    def set_recent_projects(self, paths: list[str]) -> None:
        """Populate the recent-projects list (most recent first)."""
        self._recent_list.clear()
        existing = [p for p in paths if Path(p).exists()]
        for p in existing:
            item = QListWidgetItem(Path(p).name)
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            self._recent_list.addItem(item)
        has_any = bool(existing)
        self._recent_list.setVisible(has_any)
        self._recent_empty.setVisible(not has_any)

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            self.recent_project_requested.emit(str(path))
