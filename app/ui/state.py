"""Mutable UI-session state shared across the main window and its panels.

Holds the open :class:`Project`, the loaded source image, the project's file
path and the unsaved-changes flag. Kept deliberately small; panels read from it
and the main window mutates it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.image_processor import LoadedImage
from app.models.project import Project


@dataclass(slots=True)
class AppState:
    """Runtime state for the current editing session."""

    project: Project | None = None
    source_image: LoadedImage | None = None
    file_path: Path | None = None
    dirty: bool = False

    @property
    def has_project(self) -> bool:
        return self.project is not None

    @property
    def has_image(self) -> bool:
        return self.source_image is not None

    def mark_dirty(self, value: bool = True) -> None:
        self.dirty = value

    def title_name(self) -> str:
        if self.file_path is not None:
            return self.file_path.stem
        if self.project is not None:
            return self.project.name
        return "Untitled"
