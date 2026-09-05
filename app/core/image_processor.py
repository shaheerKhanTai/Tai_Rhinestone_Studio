"""Image-processing pipeline: raster in, density map out.

This module is pure (no PySide6) and independently testable. It implements the
processing pipeline described in the spec:

    Input image -> resize -> crop/transform -> luminance/alpha analysis ->
    contrast normalisation -> optional smoothing -> edge/detail analysis ->
    density map

The **density map** is a ``float32`` array in ``[0, 1]`` describing how strongly
each region of the (transformed) image should receive rhinestones. The
placement engine (Phase 3) samples it in physical space via
:class:`~app.core.geometry.DesignTransform`.

Design decisions
----------------
* Images are loaded through Pillow (broad format support) and immediately
  normalised to an ``RGBA`` ``uint8`` array so every downstream step has a
  predictable layout.
* Processing runs in ``float32`` and is fully vectorised (NumPy / OpenCV) — no
  Python-level per-pixel loops.
* **Transparency is always respected**: fully transparent pixels always receive
  zero density, so a logo exported as a transparent PNG works without a white
  background bleeding stones.
* The default polarity assumes a *dark subject on a light/transparent
  background* (the common case for logos, silhouettes and text). An ``invert``
  flag flips this for light-on-dark artwork — a clean extension point for the UI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from PySide6.QtCore import QByteArray, QRectF, QSize
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

# --------------------------------------------------------------------------- #
# Errors and supported formats
# --------------------------------------------------------------------------- #


class ImageLoadError(Exception):
    """Raised when an image cannot be opened or decoded."""


class ImageProcessingError(Exception):
    """Raised when a processing step receives invalid data."""


#: File extensions accepted by :func:`load_image` (lower-case, with dot).
#: ``.svg`` is rasterised through QtSvg (:func:`_load_svg`) before entering
#: the same pipeline as raster formats — see ISSUE 1 in the spec.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".svg"}
)

#: Rec.601 luma weights for RGB -> perceived brightness.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Loaded image
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LoadedImage:
    """A decoded image plus its source metadata.

    ``rgba`` is always an ``(H, W, 4)`` ``uint8`` array. ``has_alpha`` reflects
    whether the *source* carried meaningful (non-fully-opaque) transparency.
    """

    rgba: np.ndarray
    path: Path | None
    format: str
    file_size_bytes: int
    has_alpha: bool

    @property
    def width(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgba.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


def load_image(path: str | Path) -> LoadedImage:
    """Load an image from disk into a :class:`LoadedImage`.

    Raises :class:`ImageLoadError` for missing files, unsupported formats or
    corrupt data. The user's original file is never modified.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported image format {path.suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    file_size = path.stat().st_size

    # SVG is vector data: rasterise to RGBA via QtSvg before the shared
    # pipeline. ``QSvgRenderer`` honours viewBox/width/height/aspect ratio and
    # transparency, and does not execute scripts or fetch external resources.
    if path.suffix.lower() == ".svg":
        try:
            rgba, had_alpha = _load_svg(path)
        except ImageLoadError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ImageLoadError(f"Could not read SVG {path.name}: {exc}") from exc
        if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.size == 0:
            raise ImageLoadError(f"SVG {path.name} produced no pixel data")
        if had_alpha and bool(np.all(rgba[:, :, 3] == 255)):
            had_alpha = False
        return LoadedImage(
            rgba=np.ascontiguousarray(rgba),
            path=path,
            format="SVG",
            file_size_bytes=file_size,
            has_alpha=had_alpha,
        )

    try:
        with Image.open(path) as im:
            im_format = im.format or path.suffix.lstrip(".").upper()
            im.load()
            had_alpha = im.mode in ("RGBA", "LA", "PA") or (
                im.mode == "P" and "transparency" in im.info
            )
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Could not read image {path.name}: {exc}") from exc

    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.size == 0:
        raise ImageLoadError(f"Image {path.name} produced no pixel data")

    # Treat a fully-opaque alpha channel as "no meaningful transparency".
    if had_alpha and bool(np.all(rgba[:, :, 3] == 255)):
        had_alpha = False

    return LoadedImage(
        rgba=np.ascontiguousarray(rgba),
        path=path,
        format=im_format,
        file_size_bytes=file_size,
        has_alpha=had_alpha,
    )


def loaded_from_array(rgba: np.ndarray, *, has_alpha: bool | None = None) -> LoadedImage:
    """Wrap an in-memory ``(H, W, 3|4)`` array as a :class:`LoadedImage`.

    Convenience for tests and for images restored from an embedded project.
    """
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ImageProcessingError("Expected an (H, W, 3) or (H, W, 4) array")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    if has_alpha is None:
        has_alpha = not bool(np.all(arr[:, :, 3] == 255))
    return LoadedImage(np.ascontiguousarray(arr), None, "ARRAY", arr.nbytes, has_alpha)


# --------------------------------------------------------------------------- #
# SVG loading (rasterised via QtSvg before the shared pipeline)
# --------------------------------------------------------------------------- #

#: Default longest-side (px) an SVG is rasterised at. Large enough to preserve
#: vector detail for density analysis without making huge RGBA buffers.
_SVG_DEFAULT_MAX_DIM = 1400

#: Conservative upper bound on SVG pixel area, guarding pathologic / billion-
#: laughs-style vectors that claim enormous dimensions.
_SVG_MAX_PIXELS = 4_000_000


def _parse_svg_dimensions(path: Path) -> tuple[int, int]:
    """Inspect an SVG's root element for an intrinsic size.

    Returns ``(width_px, height_px)`` derived (in priority order) from the
    ``width``/``height`` attributes, then the ``viewBox``. Falls back to
    ``(0, 0)`` when neither is present — the caller then uses a default.
    Values in physical units (in/cm/pt/mm) are converted to pixels at 96 DPI.
    """
    try:
        tree = ElementTree.parse(path)
    except (ElementTree.ParseError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Could not parse SVG {path.name}: {exc}") from exc

    root = tree.getroot()
    # Strip the SVG namespace from the tag for a robust local-name match.
    tag = root.tag.rsplit("}", 1)[-1].lower()
    if tag != "svg":
        raise ImageLoadError(f"{path.name} is not a valid SVG (root element is {root.tag!r})")

    w = _length_to_px(root.attrib.get("width"))
    h = _length_to_px(root.attrib.get("height"))
    if w > 0 and h > 0:
        return int(round(w)), int(round(h))

    viewBox = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if viewBox:
        parts = viewBox.replace(",", " ").split()
        if len(parts) == 4:
            try:
                vb_w = float(parts[2])
                vb_h = float(parts[3])
            except ValueError:
                raise ImageLoadError(f"{path.name} has a malformed viewBox: {viewBox!r}")
            if vb_w > 0 and vb_h > 0:
                # If one physical dimension is given, derive the other from the
                # viewBox aspect ratio (no X/Y stretching). Otherwise fall back
                # to the viewBox size in user units as pixels.
                if w > 0 and h <= 0:
                    return int(round(w)), int(round(w * vb_h / vb_w))
                if h > 0 and w <= 0:
                    return int(round(h * vb_w / vb_h)), int(round(h))
                return int(round(vb_w)), int(round(vb_h))

    return 0, 0


_SVG_UNIT_TO_PX = {
    "px": 1.0, "": 1.0, "pt": 96.0 / 72.0, "pc": 96.0 / 6.0,
    "in": 96.0, "mm": 96.0 / 25.4, "cm": 96.0 / 2.54, "q": 96.0 / 25.4 / 4.0,
}


def _length_to_px(value: str | None) -> float:
    """Parse an SVG length (number + optional unit) into pixels at 96 DPI."""
    if value is None:
        return 0.0
    text = value.strip().lower()
    if not text:
        return 0.0
    # Percentages are relative and cannot be resolved without a viewport —
    # treat as "unspecified" so the caller falls back to the viewBox.
    if text.endswith("%"):
        return 0.0
    unit = ""
    for u in ("px", "pt", "pc", "in", "mm", "cm", "q", "ex", "em"):
        if text.endswith(u):
            unit = u
            text = text[: -len(u)].strip()
            break
    try:
        num = float(text)
    except ValueError:
        return 0.0
    if unit in ("em", "ex"):
        # Font-relative units need a font context we don't have; treat the
        # numeric value as user-unit pixels as a safe approximation.
        return max(0.0, num)
    return max(0.0, num * _SVG_UNIT_TO_PX.get(unit, 1.0))


def _load_svg(path: Path) -> tuple[np.ndarray, bool]:
    """Rasterise an SVG to an ``(H, W, 4)`` uint8 RGBA array via QtSvg.

    Resolves an intrinsic size from ``width``/``height`` (then ``viewBox``),
    preserving the aspect ratio — the SVG is never stretched independently in
    X/Y. Transparent regions stay transparent. Qt's ``QSvgRenderer`` is a static
    renderer: it does not execute scripts, animate, or fetch external resources,
    so untrusted SVGs cannot run code.

    Returns ``(rgba, has_alpha)``.
    """
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    data = path.read_bytes()
    if not data.strip():
        raise ImageLoadError(f"SVG {path.name} is empty")

    w, h = _parse_svg_dimensions(path)
    if w <= 0 or h <= 0:
        # No intrinsic size: default to a square canvas, then scale to the
        # configured maximum dimension.
        w = h = _SVG_DEFAULT_MAX_DIM

    scale = min(1.0, _SVG_DEFAULT_MAX_DIM / float(max(w, h)))
    rw = max(1, int(round(w * scale)))
    rh = max(1, int(round(h * scale)))
    if rw * rh > _SVG_MAX_PIXELS:
        raise ImageLoadError(
            f"SVG {path.name} is too large to rasterise ({rw}x{rh}px); "
            f"reduce its dimensions and retry."
        )

    renderer = QSvgRenderer(QByteArray(data))
    if not renderer.isValid():
        raise ImageLoadError(f"SVG {path.name} could not be rendered (invalid or unsupported)")

    # ARGB32_Premultiplied is Qt's fastest render target; we convert to
    # straight RGBA8888 for the shared pipeline (matching numpy_rgba_to_qimage).
    image = QImage(rw, rh, QImage.Format_ARGB32_Premultiplied)
    image.fill(0)  # transparent — preserves SVG transparency
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()

    rgba = _qimage_to_rgba(image)
    had_alpha = not bool(np.all(rgba[:, :, 3] == 255))
    return np.ascontiguousarray(rgba), had_alpha


def _qimage_to_rgba(image) -> np.ndarray:
    """Convert a Qt ``QImage`` (any format) to an ``(H, W, 4)`` uint8 RGBA array."""
    from PySide6.QtGui import QImage

    # Normalise to RGBA8888 (non-premultiplied) so the pipeline sees straight
    # alpha — QSvgRenderer renders premultiplied, which would distort colours
    # if blended naively downstream.
    if image.format() != QImage.Format_RGBA8888:
        image = image.convertToFormat(QImage.Format_RGBA8888)
    # In PySide6, constBits() returns a memoryview over the pixel buffer; its
    # length is at least sizeInBytes(), so no manual resize is needed. We copy
    # the expected byte count to detach from Qt's buffer and guarantee ownership.
    expected = image.height() * image.width() * 4
    buf = bytes(image.constBits())
    if len(buf) < expected:
        raise ImageLoadError("Rasterised SVG produced an incomplete pixel buffer")
    arr = np.frombuffer(buf[:expected], dtype=np.uint8).reshape(image.height(), image.width(), 4)
    return arr.copy()


# --------------------------------------------------------------------------- #
# Geometric transforms (non-destructive to the source file)
# --------------------------------------------------------------------------- #


def apply_orientation(
    rgba: np.ndarray, rotation: int = 0, flip_h: bool = False, flip_v: bool = False
) -> np.ndarray:
    """Rotate (clockwise degrees: 0/90/180/270) then flip an RGBA array."""
    if rotation not in (0, 90, 180, 270):
        raise ImageProcessingError(f"Rotation must be 0/90/180/270, got {rotation}")
    out = rgba
    if rotation == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rotation == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip_h:
        out = np.fliplr(out)
    if flip_v:
        out = np.flipud(out)
    return np.ascontiguousarray(out)


def apply_crop(rgba: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Crop an RGBA array to ``(x, y, w, h)`` pixels (clamped to bounds)."""
    h, w = rgba.shape[:2]
    x, y, cw, ch = crop
    x0 = max(0, min(int(x), w))
    y0 = max(0, min(int(y), h))
    x1 = max(x0, min(int(x + cw), w))
    y1 = max(y0, min(int(y + ch), h))
    if x1 <= x0 or y1 <= y0:
        raise ImageProcessingError(f"Crop {crop} is empty for a {w}x{h} image")
    return np.ascontiguousarray(rgba[y0:y1, x0:x1])


def apply_image_transforms(
    rgba: np.ndarray,
    *,
    rotation: int = 0,
    flip_h: bool = False,
    flip_v: bool = False,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Apply orientation then crop. ``crop`` is in *oriented*-image coordinates."""
    out = apply_orientation(rgba, rotation, flip_h, flip_v)
    if crop is not None:
        out = apply_crop(out, crop)
    return out


def resize_max(rgba: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale so the longest side is at most ``max_dim`` (never upscales)."""
    h, w = rgba.shape[:2]
    longest = max(h, w)
    if longest <= max_dim or max_dim <= 0:
        return rgba
    scale = max_dim / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Channel analysis
# --------------------------------------------------------------------------- #


def to_luminance01(rgba: np.ndarray) -> np.ndarray:
    """Perceived brightness of the RGB channels as ``float32`` in ``[0, 1]``."""
    rgb = rgba[:, :, :3].astype(np.float32)
    lum = rgb @ _LUMA_WEIGHTS
    return np.clip(lum / 255.0, 0.0, 1.0)


def extract_alpha01(rgba: np.ndarray) -> np.ndarray:
    """Alpha channel as ``float32`` in ``[0, 1]`` (1 == opaque)."""
    if rgba.shape[2] < 4:
        return np.ones(rgba.shape[:2], dtype=np.float32)
    return (rgba[:, :, 3].astype(np.float32)) / 255.0


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def apply_contrast(x: np.ndarray, amount: float) -> np.ndarray:
    """Apply a symmetric S-curve contrast to values in ``[0, 1]``.

    ``amount`` is ``0`` (identity) to ``1`` (strong). The curve is monotonic
    and maps ``0->0`` and ``1->1`` exactly.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1e-6:
        return x.astype(np.float32, copy=True)
    k = 1.0 + amount * 9.0
    centered = x - 0.5
    s = 1.0 / (1.0 + np.exp(-k * centered))
    s0 = 1.0 / (1.0 + np.exp(k * 0.5))   # sigmoid at x = 0
    s1 = 1.0 / (1.0 + np.exp(-k * 0.5))  # sigmoid at x = 1
    out = (s - s0) / (s1 - s0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def smooth(gray01: np.ndarray, amount: float) -> np.ndarray:
    """Gaussian-blur a single-channel image. ``amount`` in ``[0, 1]``."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1e-6:
        return gray01.astype(np.float32, copy=True)
    h, w = gray01.shape[:2]
    # Sigma scales with image size so the effect is resolution-independent.
    sigma = amount * max(1.0, 0.01 * min(h, w))
    return cv2.GaussianBlur(gray01, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)


def edge_magnitude(gray01: np.ndarray) -> np.ndarray:
    """Normalised Sobel gradient magnitude in ``[0, 1]``."""
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    peak = float(mag.max())
    if peak <= 1e-9:
        return np.zeros_like(mag, dtype=np.float32)
    return (mag / peak).astype(np.float32)


def local_texture(gray01: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Normalised local standard deviation — a measure of fine detail/texture."""
    ksize = max(3, ksize | 1)  # force odd, >= 3
    mean = cv2.boxFilter(gray01, ddepth=-1, ksize=(ksize, ksize), normalize=True)
    mean_sq = cv2.boxFilter(gray01 * gray01, ddepth=-1, ksize=(ksize, ksize), normalize=True)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    std = np.sqrt(var)
    peak = float(std.max())
    if peak <= 1e-9:
        return np.zeros_like(std, dtype=np.float32)
    return (std / peak).astype(np.float32)


# --------------------------------------------------------------------------- #
# Density map
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DensityParams:
    """Slider values (all ``0..1``) that shape the density map.

    Mirrors the UI controls. ``invert`` treats a light subject as foreground.
    The rhinestone engine builds this from
    :class:`~app.models.settings.GenerationSettings`.
    """

    detail: float = 0.5
    density: float = 0.5
    contrast: float = 0.5
    smoothing: float = 0.3
    edge_preservation: float = 0.5
    invert: bool = False


@dataclass(slots=True)
class DensityField:
    """Result of the pipeline: the density map plus useful preview layers."""

    density: np.ndarray       # (H, W) float32 in [0, 1]
    grayscale: np.ndarray     # (H, W) float32 in [0, 1] (smoothed luminance)
    edges: np.ndarray         # (H, W) float32 in [0, 1]

    @property
    def width(self) -> int:
        return int(self.density.shape[1])

    @property
    def height(self) -> int:
        return int(self.density.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def coverage(self) -> float:
        """Mean density — a rough proxy for how much of the field is 'active'."""
        return float(self.density.mean()) if self.density.size else 0.0


def build_density_map(
    luminance01: np.ndarray, alpha01: np.ndarray, params: DensityParams
) -> DensityField:
    """Combine luminance, alpha and edges into a density map.

    Each slider has a clear, monotonic effect:

    * **smoothing** blurs the input to suppress speckle;
    * **contrast** widens the light/dark separation;
    * **edge_preservation** reinforces outlines so they are not lost in flat
      low-density areas;
    * **detail** adds fine-texture regions to the coverage;
    * **density** applies a global gamma so higher values raise overall coverage.

    Transparent pixels (alpha ``0``) always yield density ``0``.
    """
    if luminance01.shape != alpha01.shape:
        raise ImageProcessingError(
            f"Luminance {luminance01.shape} and alpha {alpha01.shape} shapes differ"
        )

    g = smooth(luminance01.astype(np.float32), params.smoothing)

    # Darkness drives density by default; invert for light-on-dark artwork.
    tone = g.copy() if params.invert else (1.0 - g)
    tone = apply_contrast(tone, params.contrast)

    edges = edge_magnitude(g)
    texture = local_texture(g)

    density = tone.astype(np.float32)
    density = density + float(params.edge_preservation) * edges
    density = density + float(params.detail) * 0.5 * texture
    density = np.clip(density, 0.0, 1.0)

    # Global coverage gain: higher density slider -> smaller gamma -> brighter.
    gamma = 2.0 - 1.6 * float(params.density)
    density = np.power(density, gamma, dtype=np.float32)

    # Respect transparency as a hard mask.
    density = density * alpha01.astype(np.float32)
    density = np.clip(density, 0.0, 1.0).astype(np.float32)

    return DensityField(density=density, grayscale=g, edges=edges)


# --------------------------------------------------------------------------- #
# High-level convenience
# --------------------------------------------------------------------------- #


def process_to_density(
    rgba: np.ndarray, params: DensityParams, *, max_dim: int = 1400
) -> DensityField:
    """Run the full pipeline on an already-transformed RGBA array.

    The array is downscaled so its longest side is at most ``max_dim`` (keeping
    analysis fast); the resulting density field covers exactly the same visual
    field, so the placement engine can build a :class:`DesignTransform` from
    ``density.shape`` directly.
    """
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ImageProcessingError("Expected an (H, W, 3|4) RGBA array")
    scaled = resize_max(rgba, max_dim)
    luminance = to_luminance01(scaled)
    alpha = extract_alpha01(scaled)
    return build_density_map(luminance, alpha, params)
