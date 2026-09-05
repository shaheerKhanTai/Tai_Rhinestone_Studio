"""Application-wide preferences (distinct from a project's generation settings).

Persisted as JSON in the per-user data directory. Kept free of UI types — the
theme is stored as a plain string the UI maps to a :class:`ThemeMode`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.geometry import Unit

#: Maximum number of recent-project entries retained.
MAX_RECENT = 10


@dataclass(slots=True)
class AppSettings:
    """User preferences for the whole application."""

    theme: str = "dark"                      # "dark" | "light"
    default_units: Unit = Unit.INCH
    default_stone_size_id: str = "SS10"
    default_export_dir: str | None = None
    autosave_enabled: bool = True
    autosave_interval_sec: int = 120
    recent_projects: list[str] = field(default_factory=list)
    show_grid: bool = False
    show_rulers: bool = True
    background_opacity: int = 45             # 0..100, canvas image overlay
    confirm_before_overwrite: bool = True
    confirm_delete_threshold: int = 25       # confirm bulk deletes above this
    analysis_max_dim: int = 1400             # performance: density analysis size

    def add_recent_project(self, path: str) -> None:
        """Record ``path`` as the most-recent project (delimited, capped)."""
        norm = str(path)
        self.recent_projects = [p for p in self.recent_projects if p != norm]
        self.recent_projects.insert(0, norm)
        del self.recent_projects[MAX_RECENT:]

    def remove_recent_project(self, path: str) -> None:
        self.recent_projects = [p for p in self.recent_projects if p != str(path)]

    def clear_recent_projects(self) -> None:
        self.recent_projects = []

    def clamp(self) -> None:
        """Coerce numeric fields into valid ranges (defensive after load)."""
        self.autosave_interval_sec = max(15, min(3600, int(self.autosave_interval_sec)))
        self.background_opacity = max(0, min(100, int(self.background_opacity)))
        self.confirm_delete_threshold = max(1, int(self.confirm_delete_threshold))
        self.analysis_max_dim = max(400, min(4000, int(self.analysis_max_dim)))
        if self.theme not in ("dark", "light"):
            self.theme = "dark"

    def to_dict(self) -> dict:
        return {
            "theme": self.theme,
            "default_units": self.default_units.value,
            "default_stone_size_id": self.default_stone_size_id,
            "default_export_dir": self.default_export_dir,
            "autosave_enabled": self.autosave_enabled,
            "autosave_interval_sec": self.autosave_interval_sec,
            "recent_projects": list(self.recent_projects),
            "show_grid": self.show_grid,
            "show_rulers": self.show_rulers,
            "background_opacity": self.background_opacity,
            "confirm_before_overwrite": self.confirm_before_overwrite,
            "confirm_delete_threshold": self.confirm_delete_threshold,
            "analysis_max_dim": self.analysis_max_dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppSettings":
        try:
            units = Unit.from_str(str(data.get("default_units", "in")))
        except Exception:
            units = Unit.INCH
        s = cls(
            theme=str(data.get("theme", "dark")),
            default_units=units,
            default_stone_size_id=str(data.get("default_stone_size_id", "SS10")),
            default_export_dir=data.get("default_export_dir"),
            autosave_enabled=bool(data.get("autosave_enabled", True)),
            autosave_interval_sec=int(data.get("autosave_interval_sec", 120)),
            recent_projects=[str(p) for p in data.get("recent_projects", [])],
            show_grid=bool(data.get("show_grid", False)),
            show_rulers=bool(data.get("show_rulers", True)),
            background_opacity=int(data.get("background_opacity", 45)),
            confirm_before_overwrite=bool(data.get("confirm_before_overwrite", True)),
            confirm_delete_threshold=int(data.get("confirm_delete_threshold", 25)),
            analysis_max_dim=int(data.get("analysis_max_dim", 1400)),
        )
        s.clamp()
        return s
