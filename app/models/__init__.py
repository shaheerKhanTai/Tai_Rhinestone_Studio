"""Typed domain models for Rhinestone Studio.

Import the common types directly from this package::

    from app.models import Project, Design, Stone, GenerationSettings
"""
from __future__ import annotations

from app.models.design import Design
from app.models.presets import (
    CUSTOM_PRESET_NAME,
    DEFAULT_PRESETS,
    SizePreset,
    default_presets,
)
from app.models.project import ImageState, Project
from app.models.settings import GenerationSettings, PlacementMode
from app.models.stone import (
    CUSTOM_SIZE_NAME,
    DEFAULT_STONE_COLOR,
    STANDARD_STONE_SIZES,
    STONE_SIZE_ORDER,
    Stone,
    StoneSize,
    get_stone_size,
    make_custom_size,
    resolve_diameter,
)

__all__ = [
    "Design",
    "Project",
    "ImageState",
    "GenerationSettings",
    "PlacementMode",
    "Stone",
    "StoneSize",
    "STANDARD_STONE_SIZES",
    "STONE_SIZE_ORDER",
    "CUSTOM_SIZE_NAME",
    "DEFAULT_STONE_COLOR",
    "get_stone_size",
    "make_custom_size",
    "resolve_diameter",
    "SizePreset",
    "DEFAULT_PRESETS",
    "CUSTOM_PRESET_NAME",
    "default_presets",
]
