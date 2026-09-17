"""Read and write the native ``.rhinestone`` project format.

A ``.rhinestone`` file is a ZIP container holding:

* ``manifest.json`` — format/app version for quick inspection and migration;
* ``project.json`` — the full :class:`~app.models.project.Project` state
  (physical size, settings, every stone), so a project reopens to the exact
  design state with no dependency on temporary preview files;
* ``image.png`` — the **embedded** source image (optional), so the project is
  self-contained and portable even if the original file moves;
* ``trace.png`` — the **embedded** trace-tool mask (optional, grayscale),
  painted regions that gate where generation places stones.

Loading treats the archive as untrusted input: only known member names are
read (never ``extractall`` to disk), and a per-member size cap guards against
decompression bombs.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np

from app import PROJECT_FORMAT_VERSION, __version__
from app.core.image_processor import LoadedImage, loaded_from_array
from app.models.project import Project
from app.services.logging_config import get_logger

_log = get_logger(__name__)

PROJECT_SUFFIX = ".rhinestone"

_MANIFEST = "manifest.json"
_PROJECT = "project.json"
_IMAGE = "image.png"
_TRACE = "trace.png"

#: Reject any archive member that decompresses beyond this (guards zip bombs).
_MAX_MEMBER_BYTES = 256 * 1024 * 1024  # 256 MB


class ProjectIOError(Exception):
    """Raised when a project file cannot be read or written."""


def _encode_image_png(image: LoadedImage) -> bytes:
    bgra = image.rgba[:, :, [2, 1, 0, 3]].copy()
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise ProjectIOError("Could not encode embedded image")
    return buf.tobytes()


def _decode_image_rgba(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ProjectIOError("Embedded image is corrupt")
    if arr.ndim == 2:  # grayscale
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA)
    elif arr.shape[2] == 3:  # BGR
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
    else:  # BGRA
        arr = arr[:, :, [2, 1, 0, 3]]
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _encode_trace_png(mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(mask, dtype=np.uint8))
    if not ok:
        raise ProjectIOError("Could not encode trace mask")
    return buf.tobytes()


def _decode_trace_mask(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None or arr.ndim != 2:
        raise ProjectIOError("Embedded trace mask is corrupt")
    return np.ascontiguousarray((arr >= 128).astype(np.uint8) * 255)


def save_project(project: Project, path: str | Path, source_image: LoadedImage | None = None) -> Path:
    """Write ``project`` (and optional embedded image) to a ``.rhinestone`` file.

    The archive is written to a temporary sibling then atomically moved into
    place, so an interrupted save never corrupts an existing project.
    """
    path = Path(path)
    if path.suffix.lower() != PROJECT_SUFFIX:
        path = path.with_suffix(PROJECT_SUFFIX)

    project.app_version = __version__
    manifest = {
        "format_version": PROJECT_FORMAT_VERSION,
        "app_version": __version__,
        "has_image": source_image is not None,
        "has_trace": project.trace_mask is not None,
    }

    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_MANIFEST, json.dumps(manifest, indent=2))
            zf.writestr(_PROJECT, json.dumps(project.to_dict(), indent=2))
            if source_image is not None:
                zf.writestr(_IMAGE, _encode_image_png(source_image))
            if project.trace_mask is not None:
                zf.writestr(_TRACE, _encode_trace_png(project.trace_mask))
        os.replace(tmp, path)
    except (OSError, zipfile.BadZipFile) as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise ProjectIOError(f"Could not save project to {path}: {exc}") from exc
    _log.info("Saved project %s (%d stones)", path.name, project.design.stone_count)
    return path


def load_project(path: str | Path) -> tuple[Project, LoadedImage | None]:
    """Load a ``.rhinestone`` file, returning ``(project, embedded_image)``.

    Raises :class:`ProjectIOError` for missing/corrupt files or an invalid
    schema. The embedded image is ``None`` if the project was saved without one.
    """
    path = Path(path)
    if not path.exists():
        raise ProjectIOError(f"Project file not found: {path}")
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if _PROJECT not in names:
                raise ProjectIOError("Not a valid Rhinestone project (missing project.json)")
            _guard_member_size(zf, _PROJECT)
            project = Project.from_dict(json.loads(zf.read(_PROJECT)))

            image: LoadedImage | None = None
            if _IMAGE in names:
                _guard_member_size(zf, _IMAGE)
                image = loaded_from_array(_decode_image_rgba(zf.read(_IMAGE)))

            # Trace mask (optional; absent in projects saved before the tool).
            if _TRACE in names:
                _guard_member_size(zf, _TRACE)
                try:
                    project.trace_mask = _decode_trace_mask(zf.read(_TRACE))
                except ProjectIOError:
                    _log.warning("Dropping corrupt trace mask in %s", path.name)
                    project.trace_mask = None
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProjectIOError(f"Could not open {path.name}: {exc}") from exc
    except (ValueError, KeyError) as exc:
        raise ProjectIOError(f"Project file {path.name} is invalid: {exc}") from exc
    _log.info("Loaded project %s (%d stones)", path.name, project.design.stone_count)
    return project, image


def _guard_member_size(zf: zipfile.ZipFile, name: str) -> None:
    info = zf.getinfo(name)
    if info.file_size > _MAX_MEMBER_BYTES:
        raise ProjectIOError(f"Project member {name} is unreasonably large; refusing to load")
