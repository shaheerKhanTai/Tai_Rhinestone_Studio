"""Rhinestone Studio — image-to-rhinestone design application.

Top-level package. Business logic lives in :mod:`app.core`, :mod:`app.models`,
:mod:`app.exporters` and :mod:`app.services`; the PySide6 interface lives in
:mod:`app.ui`. The engine layers never import from :mod:`app.ui`.
"""
from __future__ import annotations

__app_name__ = "Rhinestone Studio"
__version__ = "0.1.0"
__organization__ = "Tai"
__organization_domain__ = "tai-rhinestone-studio.local"

#: Project-file format version. Bumped when the on-disk schema changes so that
#: :mod:`app.services.project_service` can migrate older files.
PROJECT_FORMAT_VERSION = 1

__all__ = [
    "__app_name__",
    "__version__",
    "__organization__",
    "__organization_domain__",
    "PROJECT_FORMAT_VERSION",
]
