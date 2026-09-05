"""Helpers to bridge NumPy image arrays and Qt image types."""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def numpy_rgba_to_qimage(rgba: np.ndarray) -> QImage:
    """Convert a contiguous ``(H, W, 4)`` uint8 RGBA array to a QImage.

    The returned image owns its pixel data (a deep copy), so the source array
    can be freed safely.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected an (H, W, 4) RGBA array")
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = arr.shape[:2]
    image = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return image.copy()  # detach from the temporary buffer
