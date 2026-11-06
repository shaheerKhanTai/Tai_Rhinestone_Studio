"""Why did generation produce nothing? — and what should the user change?

A run that places zero stones is the one failure a rhinestone design tool must
never report silently: the canvas simply stays empty and the user is left to
guess. This module turns the state a finished (or failed) generation left
behind into a :class:`GenerationIssue` — a headline, an explanation of what was
found, the causes that explain it, and the concrete changes that fix it.

It is deliberately **pure** (no PySide6, no dialogs, no logging) so the whole
diagnosis is unit-testable and the UI layer only has to render the result. The
checks run in priority order: the first one that matches is the answer, because
the earlier conditions make the later ones moot (a design whose traced area is
empty has no meaningful density to reason about).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.image_processor import DensityField, _TONE_FLOOR
from app.core.placement import solid_shape
from app.core.trace_mask import density_gate, mask_is_empty
from app.models.settings import GenerationSettings, PlacementMode

#: A traced region with less than this many times a single stone's footprint
#: of area cannot hold a stone at all — used to spot a trace that is painted,
#: but far too small for the chosen stone size.
_MIN_TRACE_STONE_AREA = 1.0

#: Stone counts at or below this are treated as "the limit is the reason you
#: got almost nothing" rather than as a normal cap.
_TINY_STONE_LIMIT = 10


@dataclass(frozen=True, slots=True)
class GenerationIssue:
    """A user-facing explanation of a generation that produced nothing, or failed.

    ``title`` is the dialog/dialog-banner headline; ``summary`` says what the
    engine found; ``causes`` explains why it happened; ``fixes`` lists the
    concrete changes that make it work — phrased as things to do, and things to
    avoid, so the message is actionable rather than merely descriptive.
    """

    title: str
    summary: str
    causes: tuple[str, ...]
    fixes: tuple[str, ...]
    #: ``False`` for purely advisory issues, which should not raise an alarm.
    sound: bool = True


def _pitch(settings: GenerationSettings) -> float:
    return settings.center_distance()


def _describe_design(width_mm: float, height_mm: float, unit_symbol: str = "mm") -> str:
    return f"{width_mm:.1f} × {height_mm:.1f} {unit_symbol}"


def diagnose_empty_generation(
    settings: GenerationSettings,
    field: DensityField,
    *,
    width_mm: float,
    height_mm: float,
    trace_mask: np.ndarray | None = None,
) -> GenerationIssue:
    """Explain a generation pass that placed no stones.

    ``field`` is the density field the run produced (its ``density``,
    ``subject`` and ``backdrop`` carry everything needed to tell "the image has
    no subject" from "the subject is there but the stones were too big").
    """
    try:
        pitch = _pitch(settings)
    except Exception:  # pragma: no cover - settings are validated before a run
        pitch = 0.0

    stone_desc = settings.stone_size_id
    d = np.asarray(field.density, dtype=np.float64)
    h, w = d.shape
    scale_x = width_mm / w if w else 0.0
    scale_y = height_mm / h if h else 0.0

    # -- 1. The traced area is painted but empty ----------------------------- #
    if trace_mask is not None and mask_is_empty(trace_mask):
        return GenerationIssue(
            title="No rhinestones were generated",
            summary=(
                "The traced area is empty, so there was nowhere for stones to go. "
                "Generate only fills the areas you paint with the Trace tool."
            ),
            causes=(
                "A trace exists on this design but no pixels were painted into it.",
                "The Trace tool was selected but no stroke landed on the canvas.",
            ),
            fixes=(
                "Paint the areas that should receive rhinestones with the Trace tool (T), "
                "then press Generate.",
                "Or use Design ▸ Clear Traced Area to remove the trace and generate "
                "from the whole image.",
            ),
        )

    # -- 2. The traced area is too small for this stone ----------------------- #
    if trace_mask is not None and pitch > 0:
        gate = density_gate(trace_mask, h, w)
        painted = int(np.count_nonzero(gate > 0.5))
        painted_mm2 = painted * scale_x * scale_y
        if painted > 0 and painted_mm2 < _MIN_TRACE_STONE_AREA * pitch * pitch:
            return GenerationIssue(
                title="No rhinestones were generated",
                summary=(
                    f"The traced area is too small to hold a {stone_desc} stone. "
                    f"It holds about {painted_mm2:.1f} mm² of paint, and one stone "
                    f"needs {pitch * pitch:.1f} mm² (a {pitch:.2f} mm pitch)."
                ),
                causes=(
                    "The painted region is smaller than a single stone plus its spacing.",
                    f"A {stone_desc} stone at {settings.spacing_mm:.2f} mm spacing needs "
                    f"a {pitch:.2f} mm pitch.",
                ),
                fixes=(
                    "Paint a larger area with the Trace tool, or",
                    "choose a smaller stone size in the Rhinestones panel, or",
                    "increase the design size so the same paint covers more millimetres.",
                ),
            )

    # -- 3. The stone does not fit the design at all -------------------------- #
    smallest_side = min(width_mm, height_mm)
    if pitch > 0 and smallest_side > 0 and pitch > smallest_side:
        return GenerationIssue(
            title="No rhinestones were generated",
            summary=(
                f"A single {stone_desc} stone plus its spacing needs a {pitch:.2f} mm pitch, "
                f"but the design is only {_describe_design(width_mm, height_mm)}."
            ),
            causes=(
                "The stone is larger than the entire design.",
                f"Each centre must also keep half a pitch ({pitch / 2:.2f} mm) from "
                "the edge, so no centre position is left.",
            ),
            fixes=(
                f"Choose a smaller stone size — the design needs at least "
                f"{pitch:.2f} mm on its shorter side.",
                "Or increase the design size in the Design Size panel.",
                f"Narrower spacing also helps: you currently have "
                f"{settings.spacing_mm:.2f} mm.",
            ),
        )

    # -- 4. The image is fully transparent ------------------------------------ #
    # Checked before the tone test below: a fully transparent image also has no
    # tone anywhere, so the tone test would claim it is merely "too light" and
    # send the user off to adjust sliders that cannot possibly help.
    subject = field.subject
    if subject is not None and not bool(np.any(subject)):
        return GenerationIssue(
            title="No rhinestones were generated",
            summary=(
                "This image is fully transparent, so it has no visible artwork to "
                "convert."
            ),
            causes=(
                "The alpha channel is empty across the whole image.",
                "A file can end up like this when an editor exported only the "
                "transparent layer.",
            ),
            fixes=(
                "Re-export the image with its artwork on a visible background.",
                "Open the file in an image editor to confirm the design is actually "
                "visible before importing it.",
            ),
        )

    # -- 5. The image carries no usable tone anywhere ------------------------- #
    peak = float(d.max()) if d.size else 0.0
    if d.size == 0 or peak <= _TONE_FLOOR:
        return GenerationIssue(
            title="No rhinestones were generated",
            summary=(
                "Nothing in this image is dark enough to place a stone on. "
                f"The strongest tone found was {peak:.3f}, below the {_TONE_FLOOR:.2f} "
                "floor that separates a subject from background haze."
            ),
            causes=(
                "The image is very light overall — a pale, white or pastel design on a "
                "light background.",
                "Or the subject is light and the background is dark, so stones were "
                "placed on the background instead (which then found nothing solid).",
                "Or Density and Contrast are both low.",
            ),
            fixes=(
                "Tick “Light subject on dark background” in the Appearance panel if the "
                "artwork is pale on a dark background.",
                "Raise the Density and Contrast sliders in the Appearance panel.",
                "Or raise Edge Preservation, which reinforces the outlines that are "
                "otherwise too faint to reach the floor.",
                "Avoid scanning or exporting artwork with a washed-out, low-contrast look.",
            ),
        )

    # -- 6. No solid shape survived the stone-scale cleanup ------------------- #
    if pitch > 0:
        try:
            pitch_px = pitch / max(min(scale_x, scale_y), 1e-9)
            shape_mask = solid_shape(
                d, subject if subject is not None else None, None, pitch_px
            )
            survived = bool(np.any(shape_mask))
        except Exception:  # pragma: no cover - defensive; shape derivation is total
            survived = True
        if not survived:
            return GenerationIssue(
                title="No rhinestones were generated",
                summary=(
                    "No solid shape survived the stone-size analysis: everything the "
                    "engine found was smaller than one stone."
                ),
                causes=(
                    "The artwork's features are finer than a single stone — thin lines, "
                    "small text or fine detail.",
                    "Or the Smoothing slider has blurred the design away.",
                ),
                fixes=(
                    "Choose a smaller stone size so the detail is larger than one stone.",
                    "Lower the Smoothing slider in the Appearance panel.",
                    "Raise the Detail slider to keep fine image structure.",
                    "Avoid artwork whose smallest features are thinner than a stone.",
                ),
            )

    # -- 7. The stone limit is the reason almost nothing appeared ------------- #
    if settings.max_stones is not None and 0 < settings.max_stones <= _TINY_STONE_LIMIT:
        return GenerationIssue(
            title="No rhinestones were generated",
            summary=(
                f"“Limit stones” is switched on and set to {settings.max_stones}, which "
                "is too low to build a design."
            ),
            causes=(
                "The maximum stone count is capping the result before any stones can "
                "form a pattern.",
            ),
            fixes=(
                "Raise the stone limit in the Appearance panel, or",
                "untick “Limit stones” to let the engine place as many as the design "
                "needs.",
            ),
        )

    # -- 8. Fallback: something about the density/size combination ------------- #
    return GenerationIssue(
        title="No rhinestones were generated",
        summary=(
            "The engine analysed the image but found no position that satisfied the "
            "stone size, spacing and density together."
        ),
        causes=(
            f"A {stone_desc} stone at {settings.spacing_mm:.2f} mm spacing gives a "
            f"{pitch:.2f} mm pitch on a {_describe_design(width_mm, height_mm)} design.",
            "The areas above the Density floor were too small or too thin to hold a "
            "stone at that pitch.",
            "Or the traced area (if any) excludes everything the density selected.",
        ),
        fixes=(
            "Raise the Density slider so more of the image qualifies.",
            "Choose a smaller stone size or wider spacing.",
            "Check the traced area covers the part of the design you want filled.",
            "If the artwork is pale on a dark background, tick “Light subject on dark "
            "background”.",
        ),
    )


def diagnose_failure(
    message: str,
    *,
    placement_mode: PlacementMode | None = None,
) -> GenerationIssue:
    """Explain a generation that raised, from the worker's error text.

    The worker forwards a plain string, so the mapping is textual. Anything
    unrecognised is passed through verbatim under a clear headline rather than
    being swallowed — an unexplained failure the user can quote is far better
    than a reassuring one that hides the cause.
    """
    text = (message or "").strip()
    lowered = text.lower()

    if "memory" in lowered or "too large" in lowered:
        return GenerationIssue(
            title="Generation ran out of memory",
            summary=(
                "The design needs more memory than is available, so the engine "
                "stopped before it could place any stones."
            ),
            causes=(
                "The stone pitch and design size together imply an enormous number of "
                "candidate positions.",
                "Or the analysis resolution is set very high for a large design.",
            ),
            fixes=(
                "Choose a larger stone size, which reduces the number of positions.",
                "Increase the spacing between stones.",
                "Reduce the design size.",
                "In Settings ▸ Performance, lower the analysis resolution.",
            ),
        )

    if "density" in lowered and "shape" in lowered:
        return GenerationIssue(
            title="Generation failed",
            summary=(
                "The density map did not match the design size, so placement could "
                "not run. This is an internal inconsistency rather than a problem "
                "with your artwork."
            ),
            causes=("The image analysis and the design dimensions disagreed.",),
            fixes=(
                "Press Generate again — re-analysing usually resolves it.",
                "If it persists, re-import the image so the design size is recomputed.",
            ),
        )

    return GenerationIssue(
        title="Generation failed",
        summary="The engine stopped before it could place any rhinestones.",
        causes=(text or "No further detail was reported by the engine.",),
        fixes=(
            "Press Generate again to retry.",
            "If it happens every time, try a different stone size or a smaller design.",
            "The status bar and the log file record the full error if you need to "
            "report it.",
        ),
    )
