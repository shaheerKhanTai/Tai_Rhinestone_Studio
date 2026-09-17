"""Stones must follow the artwork — its real outline, and only its area.

Two user-visible defects are pinned here:

* **Outline + Fill did not line up with the PNG.** The shape was inferred from
  how dark the pixels were and then eroded at stone scale, so the row of
  stones sat inside the artwork with rounded-off points, and artwork that
  carries no darkness at all — white, pastel or flat-coloured lettering on
  transparency — produced *no stones whatsoever*.
* **Tracing an area then generating added nothing** (or scattered stones over
  the traced region rather than over the artwork inside it).

The fix this suite protects: a transparent image states its own outline, so
that silhouette is what placement uses — to trace the shape, to bound where
stones may land, and to fill artwork that is too pale for a tone threshold to
see. Opaque images have no such signal and keep the tone-driven behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest
import cv2

from app.core.collision import find_collisions
from app.core.geometry import DesignTransform
from app.core.image_processor import DensityParams, process_to_density
from app.core.placement import place_stones
from app.core.rhinestone_engine import generate
from app.core.trace_mask import blank_mask, paint_stroke
from app.models.settings import GenerationSettings, PlacementMode

D = 2.8      # SS10
GAP = 0.5
PITCH = D + GAP
MM = 101.6   # 4-inch square artboard
SIZE = 400   # analysis pixels


# --------------------------------------------------------------------------- #
# Artwork
# --------------------------------------------------------------------------- #


def star_polygon(size: int = SIZE, r_out: int = 150, r_in: int = 65) -> np.ndarray:
    cx = cy = size // 2
    points = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = -np.pi / 2 + i * np.pi / 5
        points.append([cx + int(r * np.cos(a)), cy + int(r * np.sin(a))])
    return np.array(points, np.int32)


def star_alpha(size: int = SIZE) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask, [star_polygon(size)], 255)
    return mask


def rgba_logo(rgb, size: int = SIZE, transparent: bool = True) -> np.ndarray:
    """A star of colour ``rgb``, on transparency or on a white background."""
    img = np.full((size, size, 4), 255, np.uint8)
    img[:, :, :3] = rgb
    if transparent:
        img[star_alpha(size) == 0] = (0, 0, 0, 0)
    return img


def star_on_white(rgb, size: int = SIZE) -> np.ndarray:
    """A star of colour ``rgb`` on an opaque white sheet (no transparency)."""
    img = np.full((size, size, 4), 255, np.uint8)
    img[star_alpha(size) > 0, :3] = rgb
    return img


def disc_image(size: int = 300, radius: int = 100, rgb=(25, 25, 35),
               transparent: bool = True) -> np.ndarray:
    img = np.full((size, size, 4), 255, np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (*rgb, 255), -1)
    if transparent:
        outside = np.zeros((size, size), np.uint8)
        cv2.circle(outside, (size // 2, size // 2), radius, 255, -1)
        img[outside == 0] = (0, 0, 0, 0)
    return img


def mm_to_px(pos_mm: np.ndarray, shape: tuple[int, int],
             size_mm: float = MM, height_mm: float | None = None) -> np.ndarray:
    """Map stone centres (mm) onto the artwork's pixel grid."""
    h, w = shape
    y_mm = size_mm if height_mm is None else height_mm
    ix = np.clip(np.round(pos_mm[:, 0] / size_mm * (w - 1)).astype(int), 0, w - 1)
    iy = np.clip(np.round(pos_mm[:, 1] / y_mm * (h - 1)).astype(int), 0, h - 1)
    return np.column_stack([ix, iy])


def run(rgba, mode=PlacementMode.OUTLINE, trace=None, size_mm=MM, **kw):
    settings = GenerationSettings(placement_mode=mode, **kw)
    return generate(rgba, settings, size_mm, size_mm, trace_mask=trace,
                    analysis_max_dim=600)


def assert_all_on_artwork(result, alpha: np.ndarray, size_mm: float = MM) -> None:
    """Every stone must sit on the artwork, inside the stone's own radius."""
    pos = result.placement.positions_mm
    assert pos.shape[0] > 0
    # Grow the silhouette by one stone radius: an outline stone is centred on
    # the artwork's edge, so half of it legitimately overhangs.
    r_px = int(round((D / 2) / (size_mm / alpha.shape[1]))) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
    reach = cv2.dilate(alpha, kernel)
    px = mm_to_px(pos, alpha.shape, size_mm)
    outside = int(np.sum(reach[px[:, 1], px[:, 0]] == 0))
    assert outside == 0, f"{outside} of {pos.shape[0]} stones left the artwork"


def assert_no_overlaps(positions_mm: np.ndarray, settings) -> None:
    dia = np.full(positions_mm.shape[0], settings.effective_diameter())
    assert find_collisions(positions_mm, dia, settings.spacing_mm) == []


# --------------------------------------------------------------------------- #
# The artwork's own outline drives the shape
# --------------------------------------------------------------------------- #


class TestPaleArtworkIsStillARhinestoneDesign:
    """Artwork a tone threshold cannot see must still be converted.

    White or pale lettering on transparency has an alpha outline and no
    darkness anywhere: the tone-driven pipeline found nothing and generated
    zero stones, in every mode.
    """

    @pytest.mark.parametrize("mode", list(PlacementMode))
    def test_white_logo_gets_stones_in_every_mode(self, mode):
        res = run(rgba_logo((255, 255, 255)), mode)
        assert res.stone_count > 0, f"{mode.value} placed nothing on a white logo"
        assert_all_on_artwork(res, star_alpha())

    @pytest.mark.parametrize("rgb", [(255, 255, 255), (250, 240, 210), (250, 235, 60)])
    def test_pale_colours_all_convert(self, rgb):
        res = run(rgba_logo(rgb))
        assert res.stone_count > 100, f"{rgb} converted to {res.stone_count} stones"
        assert_all_on_artwork(res, star_alpha())

    def test_pale_artwork_fills_its_area(self):
        # Not a token stone: the shape is filled, like any other solid design.
        white = run(rgba_logo((255, 255, 255)))
        dark = run(rgba_logo((25, 25, 35)))
        assert white.stone_count >= 0.8 * dark.stone_count

    def test_shading_still_varies_inside_pale_artwork(self):
        # A pale artwork with light internal shading keeps its relative tone
        # rather than being flattened to a uniform blob.
        img = rgba_logo((240, 240, 240))
        band = np.zeros((SIZE, SIZE), np.uint8)
        cv2.rectangle(band, (140, 140), (260, 350), 255, -1)
        img[band > 0, :3] = 120   # a darker block inside the shape
        res = run(img, PlacementMode.HEXAGONAL)
        pos = res.placement.positions_mm
        px = mm_to_px(pos, (SIZE, SIZE))
        inside_band = int(np.sum(band[px[:, 1], px[:, 0]] > 0))
        assert inside_band > 0


class TestOutlineFollowsTheArtwork:
    def test_outline_row_sits_on_the_silhouette(self):
        # A transparent disc: the ring stones must lie on the artwork's own
        # edge, not somewhere inside it (the stone-scale opening used to pull
        # the whole boundary a half-pitch inwards).
        img = disc_image(radius=100)
        res = run(img)
        pos = res.placement.positions_mm
        r = np.hypot(pos[:, 0] - MM / 2, pos[:, 1] - MM / 2)
        # 100 px of a 300 px image on a 101.6 mm design → 33.87 mm radius.
        edge = 100 / 300 * MM
        # Stones reach the edge ...
        assert float(r.max()) > edge - 0.6
        # ... and never sit further out than half a stone past it.
        assert float(r.max()) < edge + D / 2 + 0.6
        # The row is populated the whole way round.
        ring = r[r > edge - D]
        assert ring.shape[0] >= 40, f"only {ring.shape[0]} stones near the edge"
        assert_all_on_artwork(res, cv2.circle(np.zeros((300, 300), np.uint8),
                                              (150, 150), 100, 255, -1))

    def test_sharp_points_of_the_artwork_are_reached(self):
        # The stone-scale opening used to shave thin features off, leaving the
        # star's points empty. The row now follows them to the tip.
        res = run(rgba_logo((25, 25, 35)))
        pos = res.placement.positions_mm
        top_tip = 50 / SIZE * MM       # the star's points, in design mm
        left_tip = 57 / SIZE * MM
        assert float(pos[:, 1].min()) < top_tip + PITCH, (
            f"topmost stone at {float(pos[:, 1].min()):.1f}mm, tip at {top_tip:.1f}mm"
        )
        assert float(pos[:, 0].min()) < left_tip + PITCH, (
            f"leftmost stone at {float(pos[:, 0].min()):.1f}mm, tip at {left_tip:.1f}mm"
        )

    def test_thin_strokes_keep_a_row_of_stones(self):
        # A stroke thinner than a stone: a row of stones must follow it.
        img = np.zeros((300, 300, 4), np.uint8)
        cv2.putText(img, "I", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 6.0,
                    (255, 255, 255, 255), 9, cv2.LINE_AA)
        res = run(img, PlacementMode.OUTLINE)
        assert res.stone_count >= 2
        alpha = (img[:, :, 3] >= 128).astype(np.uint8) * 255
        assert_all_on_artwork(res, alpha)

    def test_pale_internal_detail_is_still_a_hole(self):
        # A pale, flat detail inside the artwork is the same white as its
        # background — it stays empty, as it always did.
        img = disc_image(radius=140)
        cv2.circle(img, (150, 110), 35, (250, 250, 250, 255), -1)
        res = run(img)
        pos = res.placement.positions_mm
        hole_r = np.hypot(pos[:, 0] - 150 / 300 * MM, pos[:, 1] - 110 / 300 * MM)
        assert int((hole_r < 30 / 300 * MM).sum()) == 0

    def test_shaded_internal_detail_stays_filled(self):
        # A *graded* interior (a photo's shading) is not a hole: cutting it out
        # would punch the artwork full of holes.
        img = disc_image(radius=140)
        yy, xx = np.mgrid[0:300, 0:300].astype(np.float32)
        grad = np.clip(1.0 - np.hypot(xx - 150, yy - 110) / 90.0, 0, 1) ** 0.7
        patch = (yy > 60) & (yy < 160) & (xx > 90) & (xx < 210)
        img[patch, :3] = (60 + grad * 190)[patch][:, None]
        res = run(img)
        pos = res.placement.positions_mm
        inside_patch = int(np.sum((pos[:, 0] > 90 / 300 * MM) & (pos[:, 0] < 210 / 300 * MM)
                                  & (pos[:, 1] > 60 / 300 * MM) & (pos[:, 1] < 160 / 300 * MM)))
        assert inside_patch > 20


class TestNothingLandsOffTheArtwork:
    @pytest.mark.parametrize("mode", list(PlacementMode))
    def test_every_mode_stays_on_the_artwork(self, mode):
        res = run(rgba_logo((205, 45, 60)), mode)
        assert_all_on_artwork(res, star_alpha())

    def test_dithered_modes_do_not_scatter_on_transparency(self):
        # A pale, noisy artwork: the noise must not turn into background
        # stones outside the shape.
        rng = np.random.default_rng(4)
        img = np.zeros((SIZE, SIZE, 4), np.uint8)
        img[:, :, :3] = np.clip(
            230 + rng.normal(0, 18, (SIZE, SIZE, 1)), 0, 255
        ).astype(np.uint8)
        img[:, :, 3] = (star_alpha() > 0) * 255
        res = run(img, PlacementMode.HEXAGONAL)
        assert_all_on_artwork(res, star_alpha())


class TestOpaqueArtworkIsUnchanged:
    def test_opaque_images_carry_no_silhouette(self):
        field = process_to_density(rgba_logo((25, 25, 35), transparent=False),
                                   DensityParams())
        assert field.subject is None

    def test_transparent_images_carry_one(self):
        field = process_to_density(rgba_logo((25, 25, 35), transparent=True),
                                   DensityParams())
        assert field.subject is not None
        alpha = star_alpha()
        assert np.array_equal(field.subject > 0.5, alpha > 0)

    def test_a_photo_on_a_background_keeps_its_halftone(self):
        # Opaque tonal artwork is not a shape to fill: shading still drives
        # density, as it always did — the dark centre is covered more densely
        # than the pale rim.
        yy, xx = np.mgrid[0:300, 0:300].astype(np.float32)
        shade = np.clip(1.2 - np.hypot(xx - 150, yy - 150) / 126.0, 0, 1) ** 0.8
        gray = (255 * (1 - shade * 0.95)).astype(np.uint8)
        img = np.stack([gray] * 3 + [np.full((300, 300), 255, np.uint8)], -1)
        res = run(img, PlacementMode.HEXAGONAL)
        pos = res.placement.positions_mm
        r = np.hypot(pos[:, 0] - MM / 2, pos[:, 1] - MM / 2)
        near = int(np.sum(r < 12.0)) / (np.pi * 12.0 ** 2)
        far = int(np.sum(r > 34.0)) / (np.pi * (46.0 ** 2 - 34.0 ** 2))
        assert near > far, f"coverage/mm² near {near:.3f} vs far {far:.3f}"

    def test_stray_transparency_does_not_define_the_silhouette(self):
        # 99.5% opaque: a few soft pixels (a resaved JPEG, an exporter that
        # touched the alpha) must not hand placement a frame-filling "shape".
        img = rgba_logo((25, 25, 35), transparent=False)
        img[0:4, 0:4, 3] = 40
        field = process_to_density(img, DensityParams())
        assert field.subject is None


def design_fraction(result, artwork: np.ndarray,
                    height_mm: float | None = None) -> float:
    """Share of the stones that landed on ``artwork`` (a pixel mask)."""
    pos = result.placement.positions_mm
    px = mm_to_px(pos, artwork.shape, height_mm=height_mm)
    return float((artwork[px[:, 1], px[:, 0]] > 0).mean())


class TestDesignOnADarkBackdrop:
    """A light design on a dark backdrop must convert to the *design*.

    Exporting a rhinestone design as white lettering on a black background is
    the usual way to present one, and dark-on-light is the engine's default
    reading — so those files used to be traced as their *background*: the
    stones landed on the surround and the design itself stayed empty, which is
    what forced users to tick Invert Rhinestone Areas by hand. Outline + Fill
    now follows the design on its own.
    """

    @staticmethod
    def _on_backdrop(design_rgb, backdrop_rgb=(15, 15, 15)) -> np.ndarray:
        img = np.full((SIZE, SIZE, 4), 255, np.uint8)
        img[:, :, :3] = backdrop_rgb
        img[star_alpha() > 0, :3] = design_rgb
        return img

    def test_light_design_on_a_black_backdrop(self):
        artwork = star_alpha()
        res = run(self._on_backdrop((250, 250, 250)))
        assert res.stone_count > 100
        assert design_fraction(res, artwork) > 0.7, "stones landed on the backdrop"

    def test_white_lettering_on_a_black_backdrop(self):
        # The classic presentation: no alpha channel at all.
        img = np.full((300, 600, 3), 12, np.uint8)
        cv2.putText(img, "TIGER", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 4.0,
                    (255, 255, 255), 12, cv2.LINE_AA)
        img = np.dstack([img, np.full((300, 600), 255, np.uint8)])
        letters = np.zeros((300, 600), np.uint8)
        cv2.putText(letters, "TIGER", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 4.0,
                    255, 12, cv2.LINE_AA)
        res = generate(img, GenerationSettings(placement_mode=PlacementMode.OUTLINE),
                       MM, MM / 2, analysis_max_dim=700)
        assert res.stone_count > 20
        assert design_fraction(res, letters, height_mm=MM / 2) > 0.7

    def test_mid_tone_design_on_a_dark_backdrop(self):
        res = run(self._on_backdrop((150, 150, 150)))
        assert design_fraction(res, star_alpha()) > 0.7

    def test_dark_design_on_a_light_backdrop_is_unchanged(self):
        res = run(star_on_white((25, 25, 35)))
        assert design_fraction(res, star_alpha()) > 0.7

    def test_dithered_modes_keep_their_own_polarity(self):
        # Only the shape conversion follows the design by itself; the halftone
        # modes reproduce the image's shading with the polarity the user
        # chooses (that is what the invert controls are for).
        res = run(self._on_backdrop((250, 250, 250)), PlacementMode.HEXAGONAL)
        assert design_fraction(res, star_alpha()) < 0.5

    def test_explicit_inversion_still_wins(self):
        # Ticking Invert Rhinestone Areas stones the backdrop instead — the
        # user's own instruction, not something to second-guess.
        res = run(star_on_white((25, 25, 35)), PlacementMode.OUTLINE, invert_areas=True)
        assert res.stone_count > 100
        assert design_fraction(res, star_alpha()) < 0.3

    def test_explicit_tone_polarity_still_wins(self):
        img = self._on_backdrop((250, 250, 250))
        res = run(img, PlacementMode.OUTLINE, invert_tone=True)
        assert design_fraction(res, star_alpha()) > 0.7  # no double inversion

    def test_full_bleed_artwork_has_no_backdrop(self):
        # Nothing to separate: the whole tonal field is the design.
        img = np.full((SIZE, SIZE, 4), 25, np.uint8)
        img[:, :, 3] = 255
        field = process_to_density(img, DensityParams())
        assert field.backdrop is None
        assert run(img).stone_count > 300

    def test_tonal_artwork_has_no_backdrop(self):
        # A photo's dark areas vary far too much to read as a backdrop.
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        shade = np.clip(1.2 - np.hypot(xx - 200, yy - 200) / 168.0, 0, 1) ** 0.8
        gray = (255 * (1 - shade * 0.95)).astype(np.uint8)
        img = np.stack([gray] * 3 + [np.full((SIZE, SIZE), 255, np.uint8)], -1)
        field = process_to_density(img, DensityParams())
        assert field.backdrop is None

    def test_transparent_artwork_uses_its_alpha(self):
        # With transparency the silhouette is the artwork, so the backdrop rule
        # has nothing to say.
        img = np.zeros((SIZE, SIZE, 4), np.uint8)
        img[star_alpha() > 0] = (250, 250, 250, 255)
        field = process_to_density(img, DensityParams())
        assert field.backdrop is None
        res = run(img)
        assert design_fraction(res, star_alpha()) > 0.7


class TestTraceUsesTheArtwork:
    """Tracing an area must add stones on the artwork inside it — not nothing,
    and not a blanket of stones over the whole painted region."""

    @staticmethod
    def _band(x_mm: float, radius_mm: float = 11.0):
        mask = blank_mask(MM, MM)
        paint_stroke(mask, (x_mm, 0.12 * MM), (x_mm, 0.92 * MM), radius_mm, MM, MM)
        return mask

    @pytest.mark.parametrize("mode", list(PlacementMode))
    def test_every_mode_places_stones_inside_the_trace(self, mode):
        trace = self._band(0.28 * MM)
        res = run(rgba_logo((255, 255, 255)), mode, trace=trace)
        assert res.stone_count > 0, f"{mode.value} placed nothing after tracing"
        pos = res.placement.positions_mm
        # Strictly inside the painted region.
        px = mm_to_px(pos, trace.shape)
        assert bool(np.all(trace[px[:, 1], px[:, 0]] == 255))

    def test_trace_does_not_stone_the_empty_background(self):
        # The band runs the full height, but only the part of it that crosses
        # the artwork may receive stones.
        trace = self._band(0.5 * MM, radius_mm=16.0)
        res = run(rgba_logo((205, 45, 60)), PlacementMode.OUTLINE, trace=trace)
        assert res.stone_count > 0
        assert_all_on_artwork(res, star_alpha())

    def test_traced_artwork_is_reconverted_whole(self):
        # Whatever mode is used, the traced part of the shape is filled rather
        # than left as a few stray stones.
        full = run(rgba_logo((205, 45, 60)))
        trace = self._band(0.5 * MM, radius_mm=16.0)
        traced = run(rgba_logo((205, 45, 60)), PlacementMode.OUTLINE, trace=trace)
        assert traced.stone_count > 0.2 * full.stone_count

    def test_opaque_photo_trace_still_follows_shading(self):
        # An opaque image has no silhouette: inside the trace the tone still
        # drives density.
        img = np.full((300, 300, 4), 255, np.uint8)
        img[:, :150, :3] = 30     # dark left half
        img[:, 150:, :3] = 210    # light right half
        trace = blank_mask(MM, MM)
        paint_stroke(trace, (MM / 2, MM / 2), (MM / 2, MM / 2), MM / 2, MM, MM)
        res = run(img, PlacementMode.HEXAGONAL, trace=trace)
        pos = res.placement.positions_mm
        dark = int(np.sum(pos[:, 0] < MM / 2))
        light = int(np.sum(pos[:, 0] >= MM / 2))
        assert dark > light * 2


class TestPlumbing:
    def test_uniform_gate_raster_is_not_scaled_twice(self):
        # The trace raster is 0/255 uint8. A normalised 0/1 gate is accepted
        # too, but a 0/255 one must not be multiplied by 255 again (it used to
        # wrap at uint8 and leave a 1/255 gate, which silently killed every
        # stone).
        img = disc_image()
        field = process_to_density(img, DensityParams())
        transform = DesignTransform(field.width, field.height, MM, MM)
        settings = GenerationSettings(placement_mode=PlacementMode.HEXAGONAL)
        raster = blank_mask(MM, MM)
        paint_stroke(raster, (MM / 2, MM / 2), (MM / 2, MM / 2), MM / 2, MM, MM)

        from_raster = place_stones(field.density, transform, settings,
                                   gate=raster.astype(np.float64))
        from_unit = place_stones(field.density, transform, settings,
                                 gate=raster.astype(np.float64) / 255.0)
        assert from_raster.count == from_unit.count > 0

    def test_subject_is_optional(self):
        # Callers that pass only a density map (the placement unit tests, the
        # exporters) keep working unchanged.
        img = disc_image(transparent=False)
        field = process_to_density(img, DensityParams())
        transform = DesignTransform(field.width, field.height, MM, MM)
        settings = GenerationSettings(placement_mode=PlacementMode.OUTLINE)
        res = place_stones(field.density, transform, settings)
        assert res.count > 0
        assert_no_overlaps(res.positions_mm, settings)

    def test_deterministic_with_the_subject(self):
        img = rgba_logo((250, 240, 210))
        a = run(img)
        b = run(img)
        np.testing.assert_array_equal(a.placement.positions_mm, b.placement.positions_mm)

    def test_alpha_below_half_is_background(self):
        # Anti-aliased edges: the artwork's edge is where the pixel is half
        # opaque, so the row lands there and not on the faint fringe.
        img = np.zeros((200, 200, 4), np.uint8)
        cv2.circle(img, (100, 100), 60, (30, 30, 30, 255), -1)
        img[:, :, 3] = np.where(img[:, :, 3] > 0, 255, 0)
        img[0:5, 0:5, 3] = 90     # a faint speck outside the artwork
        res = run(img, PlacementMode.HEXAGONAL, size_mm=MM)
        pos = res.placement.positions_mm
        assert pos.shape[0] > 0
        assert float(np.hypot(pos[:, 0] - MM / 2, pos[:, 1] - MM / 2).max()) < 60 / 200 * MM + D
