"""Export orchestration: pre-export validation, single-format export and the
full production package.

Centralises the rules the UI needs so exporters stay pure. All physical
dimensions flow from the project's millimetre size into each format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.exporters.base import ExportError
from app.exporters.dxf_exporter import DxfOptions, export_dxf
from app.exporters.pdf_exporter import ProductionSheetInfo, export_pdf
from app.exporters.png_exporter import PngOptions, export_png
from app.exporters.svg_exporter import SvgOptions, export_cricut_svg, export_svg
from app.models.project import Project
from app.services.logging_config import get_logger

_log = get_logger(__name__)


@dataclass(slots=True)
class ValidationResult:
    """Outcome of pre-export validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_for_export(project: Project) -> ValidationResult:
    """Check that ``project`` can be exported; collect blocking errors + warnings."""
    result = ValidationResult()
    design = project.design
    if design.stone_count == 0:
        result.errors.append("The design has no stones. Generate a design first.")
    if project.width_mm <= 0 or project.height_mm <= 0:
        result.errors.append("The design has invalid physical dimensions.")
    try:
        project.settings.validate()
    except Exception as exc:
        result.errors.append(f"Invalid rhinestone settings: {exc}")

    # Warn if any stone falls outside the physical canvas.
    if design.stone_count:
        outside = sum(
            1 for s in design.stones
            if not (0 <= s.x <= project.width_mm and 0 <= s.y <= project.height_mm)
        )
        if outside:
            result.warnings.append(f"{outside:,} stone(s) lie outside the design boundary.")
    return result


def _sheet_info(project: Project) -> ProductionSheetInfo:
    s = project.settings
    try:
        diameter = s.effective_diameter()
    except Exception:
        diameter = 0.0
    return ProductionSheetInfo(
        design_name=project.name,
        unit=project.units,
        stone_size_id=s.stone_size_id,
        stone_diameter_mm=diameter,
        spacing_mm=s.spacing_mm,
        color_counts=project.design.color_counts(),
    )


def export_single(project: Project, kind: str, path: str | Path) -> Path:
    """Export one format. ``kind`` ∈ {svg, cricut, dxf, png, pdf}."""
    design = project.design
    kind = kind.lower()
    if kind == "svg":
        return export_svg(design, path, SvgOptions())
    if kind == "cricut":
        return export_cricut_svg(design, path)
    if kind == "dxf":
        return export_dxf(design, path, DxfOptions(include_boundary=True))
    if kind == "png":
        return export_png(design, path, PngOptions(width_px=2400))
    if kind == "pdf":
        return export_pdf(design, path, _sheet_info(project))
    raise ExportError(f"Unknown export type: {kind}")


@dataclass(slots=True)
class PackageResult:
    """Files written by a production-package export."""

    directory: Path
    files: list[Path] = field(default_factory=list)


def export_production_package(
    project: Project, out_dir: str | Path, *, include_cricut: bool = True
) -> PackageResult:
    """Write SVG, DXF, PNG, PDF (and optional Cricut SVG) into ``out_dir``.

    The base file name is derived from the (sanitised) project name.
    """
    out_dir = Path(out_dir)
    name = _safe_stem(project.name)
    result = PackageResult(directory=out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"Could not create output folder {out_dir}: {exc}") from exc

    plan: list[tuple[str, Path]] = [
        ("svg", out_dir / f"{name}.svg"),
        ("dxf", out_dir / f"{name}.dxf"),
        ("png", out_dir / f"{name}.png"),
        ("pdf", out_dir / f"{name}.pdf"),
    ]
    if include_cricut:
        plan.append(("cricut", out_dir / f"{name}-cricut.svg"))

    for kind, path in plan:
        result.files.append(export_single(project, kind, path))
    _log.info("Exported production package to %s (%d files)", out_dir, len(result.files))
    return result


def _safe_stem(name: str) -> str:
    """Sanitise a project name into a filesystem-safe base file name."""
    cleaned = "".join(c if (c.isalnum() or c in " -_") else "_" for c in name).strip()
    cleaned = "_".join(cleaned.split())  # collapse whitespace to underscores
    return cleaned or "rhinestone_design"
