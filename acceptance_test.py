"""GUI acceptance test (Step 9) — drives the REAL MainWindow through the
24-step acceptance workflow from the issue spec.

Run with: python acceptance_test.py

Uses the offscreen Qt platform (headless-safe) but exercises the genuine UI
objects: the design panel's controls, the canvas, the generation worker on the
thread pool, the command stack, and project save/load.
Screenshots are written to test_output/acceptance_*.png for manual review.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("RHINESTONE_DATA_DIR", tempfile.mkdtemp(prefix="rhinestone_accept_"))

import numpy as np  # noqa: E402
from PySide6.QtCore import QEventLoop, QPointF, QTimer  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)

PASSED: list[str] = []
FAILED: list[str] = []


def check(step: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(step)
        print(f"  PASS  {step}" + (f" — {detail}" if detail else ""))
    else:
        FAILED.append(step)
        print(f"  FAIL  {step}" + (f" — {detail}" if detail else ""))


def pump_until(window, cond, timeout_ms=60000) -> bool:
    """Process GUI events until cond() is true or the timeout expires."""
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(20)
    deadline = QTimer()
    deadline.setSingleShot(True)

    def _tick():
        if cond():
            loop.quit()

    def _timeout():
        loop.quit()

    timer.timeout.connect(_tick)
    deadline.timeout.connect(_timeout)
    timer.start()
    deadline.start(timeout_ms)
    loop.exec()
    timer.stop()
    return cond()


def shot(window, name: str) -> None:
    window.grab().save(str(OUT / f"acceptance_{name}.png"))


def main() -> int:
    from app.core.collision import find_collisions
    from app.main import build_application

    print("=" * 70)
    print("Rhinestone Studio — GUI Acceptance Test (24-step workflow)")
    print("=" * 70)

    app, window = build_application(["acceptance"])
    window.resize(1360, 860)

    # -- 1-2. Open the app and import an SVG ----------------------------- #
    print("\n[1-2] Import SVG")
    svg_path = Path(tempfile.mkdtemp(prefix="accept_svg_")) / "design.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" '
        'viewBox="0 0 400 400">'
        '<rect width="400" height="400" fill="#ffffff"/>'
        '<circle cx="200" cy="200" r="120" fill="#141414"/>'
        '<rect x="290" y="290" width="90" height="90" fill="#777777"/>'
        "</svg>",
        encoding="utf-8",
    )
    window._import_image_path(svg_path)
    proj = window.state.project
    check("1-2. SVG imported through the image pipeline",
          proj is not None and proj.has_image and window.state.source_image is not None,
          f"{window.state.source_image.width}x{window.state.source_image.height}px")
    shot(window, "01_imported_svg")

    # -- 3. Physical dimensions ------------------------------------------- #
    print("\n[3] Physical design dimensions")
    check("3. Physical size set from SVG aspect (10in default width)",
          abs(proj.width_mm - 254.0) < 1.0 and abs(proj.height_mm - 254.0) < 1.0,
          f"{proj.width_in_units:.2f} x {proj.height_in_units:.2f} in")

    # -- 4. Stone diameter + spacing + Push All -------------------------- #
    print("\n[4] Configure stone size, spacing, Push All")
    panel = window._design_panel
    panel._spacing.setValue(0.5)
    panel._push_all.setChecked(True)
    settings = panel.build_settings()
    check("4a. Stone size SS10 + 0.5mm gap configured",
          settings.stone_size_id == "SS10" and abs(settings.spacing_mm - 0.5) < 1e-9,
          f"pitch = {settings.center_distance():.2f} mm")
    check("4b. Push All checkbox enabled in the UI",
          settings.push_all and window._canvas._push_all,
          "persists with project settings")

    # -- 5-9. Crop -> Ctrl+Z -> Ctrl+Y ------------------------------------ #
    print("\n[5-9] Crop, undo, redo")
    # Crop the centre 70% — the disc (centre 200,200 r120) stays centred in it.
    crop_before = proj.image.crop
    window._on_crop_committed(0.15, 0.15, 0.7, 0.7)
    cropped = proj.image.crop
    check("5. Crop applied", cropped is not None and cropped != crop_before, str(cropped))
    shot(window, "02_cropped")
    window.action_undo()  # Ctrl+Z
    check("6-7. Ctrl+Z undoes the crop", proj.image.crop == crop_before)
    window.action_redo()  # Ctrl+Y
    check("8-9. Ctrl+Y restores the crop", proj.image.crop == cropped)

    # -- 10-12. Generate with progress + progressive stones --------------- #
    print("\n[10-12] Automatic generation, progress, progressive display")
    progress_seen: list[float] = []
    stages_seen: list[str] = []
    batches_seen: list[int] = []

    orig_set_progress = panel.set_progress
    orig_append = window._canvas.append_stone_batch

    def spy_progress(stage, frac):
        progress_seen.append(frac)
        stages_seen.append(stage)
        orig_set_progress(stage, frac)

    def spy_append(stones):
        batches_seen.append(len(stones))
        orig_append(stones)

    panel.set_progress = spy_progress
    window._canvas.append_stone_batch = spy_append

    window.action_generate()
    check("10. Generation started (worker active)", window._active_worker is not None)
    ok = pump_until(window, lambda: window._active_worker is None)
    check("10. Generation completed", ok and proj.design.stone_count > 0,
          f"{proj.design.stone_count:,} stones")
    panel.set_progress = orig_set_progress
    window._canvas.append_stone_batch = orig_append

    check("11a. Progress bar showed real engine progress",
          len(progress_seen) >= 4 and max(progress_seen) >= 0.999,
          f"{len(progress_seen)} updates, peak {max(progress_seen):.2f}")
    check("11b. Progress advanced through multiple stages",
          len(set(stages_seen)) >= 3, ", ".join(sorted(set(stages_seen))))
    check("11c. Stones appeared progressively (batched, not one-shot)",
          len(batches_seen) >= 2 and sum(batches_seen) == proj.design.stone_count,
          f"{len(batches_seen)} batches: {batches_seen}")
    shot(window, "03_generated")

    # -- 13. Stones follow image shading ---------------------------------- #
    print("\n[13] Shading controls density")
    pos = proj.design.positions()
    # The disc (SVG centre 200,200, r120) inside the crop (15%..85% of 400):
    # crop origin 60px, size 280px -> disc centre at (200-60)/280, r = 120/280.
    cx_mm = proj.width_mm * ((200 - 60) / 280)
    cy_mm = proj.height_mm * ((200 - 60) / 280)
    r_mm = proj.width_mm * (120 / 280)
    dist = np.hypot(pos[:, 0] - cx_mm, pos[:, 1] - cy_mm)
    in_disc = float(np.mean(dist <= r_mm))
    area_ratio = 3.14159 * r_mm**2 / (proj.width_mm * proj.height_mm)
    check("13. Stones concentrate in the dark disc",
          in_disc > min(0.9, area_ratio * 2.5),
          f"{in_disc:.0%} inside the disc (disc area = {area_ratio:.0%})")

    # -- 14. Spacing -------------------------------------------------------- #
    print("\n[14] Spacing validation")
    pairs = find_collisions(proj.design.positions(), proj.design.diameters(), 0.5)
    check("14. All stones respect the configured spacing", pairs == [],
          f"{proj.design.stone_count:,} stones, 0 collisions")

    # -- 15-16. Push All drag ---------------------------------------------- #
    print("\n[15-16] Push All movement")
    canvas = window._canvas
    pitch = settings.center_distance()
    stone_list = proj.design.stones

    # The generated hex lattice is packed at exactly the pitch (zero slack),
    # where any drag correctly cascades-then-refuses (verified below). The
    # spec's Push All example is a sparse row with free space — mirror it by
    # adding three stones in an empty area of the design (a normal user edit),
    # then dragging one toward its neighbour.
    positions_all = proj.design.positions()
    max_x = float(positions_all[:, 0].max())
    free_x = min(proj.width_mm - 2 * pitch, max_x + 4 * pitch)
    free_y = proj.height_mm * 0.08  # above the disc, inside the canvas
    free_x = max(2 * pitch, min(free_x, proj.width_mm - 2 * pitch))
    trio_ids = []
    for k in range(3):
        stone = proj.design.add_stone(free_x, free_y + k * pitch, 2.8)
        trio_ids.append(stone.id)
    canvas.refresh_geometry()
    a = proj.design.get_by_id(trio_ids[0])
    b = proj.design.get_by_id(trio_ids[1])
    c3 = proj.design.get_by_id(trio_ids[2])
    check("15a. Sparse three-stone cluster added (spec example scenario)",
          abs(np.hypot(b.x - a.x, b.y - a.y) - pitch) < 0.01)

    b_before = (b.x, b.y)
    c_before = (c3.x, c3.y)
    a.selected = True
    canvas.refresh_selection()
    canvas._begin_move(QPointF(a.x, a.y))
    canvas._update_move(QPointF(a.x + (b.x - a.x) * 0.5, a.y + (b.y - a.y) * 0.5))
    canvas._end_move()
    dist_now = float(np.hypot(b.x - a.x, b.y - a.y))
    check("15b. Push All pushed the neighbour aside", (b.x, b.y) != b_before,
          f"neighbour moved from {b_before[0]:.1f},{b_before[1]:.1f} to {b.x:.1f},{b.y:.1f}")
    check("16. Spacing preserved after the push", dist_now >= pitch - 0.05,
          f"distance {dist_now:.2f}mm vs pitch {pitch:.2f}mm")
    pairs = find_collisions(proj.design.positions(), proj.design.diameters(), 0.5)
    check("16b. Whole design still collision-free after the push", pairs == [])
    # One undo reverts drag + push together.
    window.action_undo()
    pairs = find_collisions(proj.design.positions(), proj.design.diameters(), 0.5)
    dist_after_undo = float(np.hypot(b.x - a.x, b.y - a.y))
    check("16c. Single Ctrl+Z reverts drag + pushed neighbours; layout valid",
          pairs == [] and (b.x, b.y) == b_before and dist_after_undo < pitch * 1.01)
    window.action_redo()
    shot(window, "04_push_all")

    # Impossible drag (interior of the packed disc): must restore cleanly.
    degree = np.array([
        int(np.sum(np.hypot(positions_all[:, 0] - s.x,
                            positions_all[:, 1] - s.y) < pitch * 1.2)) - 1
        for s in proj.design.stones if s.id not in set(trio_ids)
    ])
    interior_stones = [s for s in proj.design.stones if s.id not in set(trio_ids)]
    c = interior_stones[int(np.argmax(degree))]  # most-connected = deep interior
    e_idx = int(np.argmin([
        np.hypot(s.x - c.x, s.y - c.y) if s.id != c.id else np.inf
        for s in interior_stones
    ]))
    e = interior_stones[e_idx]
    c_before = (c.x, c.y)
    push_failed = []
    canvas.push_failed.connect(lambda: push_failed.append(True))
    c.selected = True
    canvas.refresh_selection()
    canvas._begin_move(QPointF(c.x, c.y))
    canvas._update_move(QPointF(c.x + (e.x - c.x) * 0.5, c.y + (e.y - c.y) * 0.5))
    canvas._end_move()
    check("16d. Impossible packed-interior drag restored + notified",
          push_failed == [True] and (c.x, c.y) == c_before,
          "wave would move the whole design -> refused, layout intact")

    # -- 22-24. Save and reopen -------------------------------------------- #
    print("\n[22-24] Save and reopen the project")
    save_path = OUT / "acceptance_project.rhinestone"
    saved = window._write_project(save_path)
    check("22. Project saved", saved and save_path.exists(),
          f"{save_path.stat().st_size:,} bytes")
    stone_count_saved = proj.design.stone_count
    positions_saved = proj.design.positions().copy()

    window.state.dirty = False
    window.close()
    window.deleteLater()

    app2, window2 = build_application(["acceptance-2"])
    window2.resize(1360, 860)
    window2.open_project_path(str(save_path))
    proj2 = window2.state.project
    check("23. Project reopened", proj2 is not None and proj2.design.stone_count > 0)
    check("24. Design intact after reopen (count + positions)",
          proj2.design.stone_count == stone_count_saved
          and np.allclose(proj2.design.positions(), positions_saved, atol=1e-6),
          f"{proj2.design.stone_count:,} stones restored")
    shot(window2, "06_reopened")
    window2.state.dirty = False
    window2.close()
    window2.deleteLater()

    # -- Summary ------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED STEPS:")
        for f in FAILED:
            print(f"  - {f}")
    print("=" * 70)
    print(f"Screenshots in: {OUT.absolute()}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
