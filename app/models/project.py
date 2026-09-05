"""The :class:`Project` — the complete, serialisable application document.

A project bundles the source-image reference and its transforms, the physical
design size and units, the generation settings, and the generated
:class:`~app.models.design.Design`. It is the unit of save/load and reopens to
the exact design state (the ``.rhinestone`` container and image embedding are
handled by :mod:`app.services.project_service` in a later phase).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app import PROJECT_FORMAT_VERSION, __version__
from app.core.geometry import GeometryError, Unit, from_mm
from app.models.design import Design
from app.models.settings import GenerationSettings

_VALID_ROTATIONS = (0, 90, 180, 270)


@dataclass(slots=True)
class ImageState:
    """Reference to the source image plus the non-destructive transforms.

    The user's original file is never modified; crop/rotate/flip are recorded
    here and applied by the image-processing pipeline. ``crop`` is a pixel
    rectangle ``(x, y, w, h)`` in the *original* image's coordinate space, or
    ``None`` for the full image.
    """

    source_path: Path | None = None
    original_width_px: int | None = None
    original_height_px: int | None = None
    crop: tuple[int, int, int, int] | None = None
    rotation: int = 0
    flip_h: bool = False
    flip_v: bool = False

    def __post_init__(self) -> None:
        if self.rotation not in _VALID_ROTATIONS:
            raise GeometryError(f"Rotation must be one of {_VALID_ROTATIONS}, got {self.rotation}")

    def reset_transforms(self) -> None:
        self.crop = None
        self.rotation = 0
        self.flip_h = False
        self.flip_v = False

    def to_dict(self) -> dict:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "original_width_px": self.original_width_px,
            "original_height_px": self.original_height_px,
            "crop": list(self.crop) if self.crop else None,
            "rotation": self.rotation,
            "flip_h": self.flip_h,
            "flip_v": self.flip_v,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageState":
        crop = data.get("crop")
        return cls(
            source_path=Path(data["source_path"]) if data.get("source_path") else None,
            original_width_px=data.get("original_width_px"),
            original_height_px=data.get("original_height_px"),
            crop=tuple(int(v) for v in crop) if crop else None,  # type: ignore[arg-type]
            rotation=int(data.get("rotation", 0)),
            flip_h=bool(data.get("flip_h", False)),
            flip_v=bool(data.get("flip_v", False)),
        )


@dataclass(slots=True)
class Project:
    """A complete Rhinestone Studio document."""

    name: str = "Untitled"
    image: ImageState = field(default_factory=ImageState)
    width_mm: float = 254.0   # 10 in
    height_mm: float = 203.2  # 8 in
    units: Unit = Unit.INCH
    lock_aspect: bool = True
    settings: GenerationSettings = field(default_factory=GenerationSettings)
    design: Design = field(default_factory=lambda: Design(254.0, 203.2))
    app_version: str = __version__
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise GeometryError("Project dimensions must be positive")
        # Keep the design's physical canvas in sync with the project size.
        self.design.resize(self.width_mm, self.height_mm)

    # -- convenience -------------------------------------------------------- #
    @classmethod
    def new(cls, name: str = "Untitled") -> "Project":
        """Create a fresh, empty project with default settings and size."""
        return cls(name=name)

    def set_physical_size(self, width_mm: float, height_mm: float) -> None:
        """Update the authoritative physical size and mirror it to the design."""
        if width_mm <= 0 or height_mm <= 0:
            raise GeometryError("Project dimensions must be positive")
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.design.resize(width_mm, height_mm)

    @property
    def width_in_units(self) -> float:
        return from_mm(self.width_mm, self.units)

    @property
    def height_in_units(self) -> float:
        return from_mm(self.height_mm, self.units)

    @property
    def has_image(self) -> bool:
        return self.image.source_path is not None

    @property
    def has_design(self) -> bool:
        return self.design.stone_count > 0

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (image bytes handled by service)."""
        return {
            "format_version": PROJECT_FORMAT_VERSION,
            "app_version": self.app_version,
            "name": self.name,
            "image": self.image.to_dict(),
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "units": self.units.value,
            "lock_aspect": self.lock_aspect,
            "settings": self.settings.to_dict(),
            "design": self.design.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        project = cls(
            name=str(data.get("name", "Untitled")),
            image=ImageState.from_dict(data.get("image", {})),
            width_mm=float(data.get("width_mm", 254.0)),
            height_mm=float(data.get("height_mm", 203.2)),
            units=Unit.from_str(str(data.get("units", "in"))),
            lock_aspect=bool(data.get("lock_aspect", True)),
            settings=GenerationSettings.from_dict(data.get("settings", {})),
            design=Design.from_dict(data.get("design", {"width_mm": 254.0, "height_mm": 203.2})),
            app_version=str(data.get("app_version", __version__)),
            metadata=dict(data.get("metadata", {})),
        )
        return project
