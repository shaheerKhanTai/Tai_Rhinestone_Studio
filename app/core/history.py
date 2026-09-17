"""Undo/redo command stack and design edit commands.

A small command pattern kept deliberately memory-light (spec §20): commands
store only the data needed to reverse themselves — moved positions or the small
set of affected stones — never a whole-project snapshot per edit. The single
exception is :class:`ReplaceDesignCommand` (used for a full regeneration), which
must hold both stone sets because the operation replaces everything at once, but
that happens once per *generate*, not once per mouse event.

Pure and Qt-free, so the whole system is unit-testable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable

from app.models.design import Design
from app.models.stone import Stone


class Command(ABC):
    """A reversible edit. ``redo`` applies it; ``undo`` reverses it."""

    label: str = "Edit"

    @abstractmethod
    def redo(self) -> None:
        """Apply (or re-apply) the change."""

    @abstractmethod
    def undo(self) -> None:
        """Reverse the change."""


class CommandStack:
    """Undo/redo history with a bounded capacity.

    ``do`` executes and records a command. ``push_applied`` records a command
    whose effect the caller already performed live (e.g. an interactive drag),
    so it is not executed again but can still be undone/redone.
    """

    def __init__(self, capacity: int = 200) -> None:
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._capacity = max(1, capacity)
        self.on_change: Callable[[], None] | None = None

    # -- recording ---------------------------------------------------------- #
    def do(self, command: Command) -> None:
        command.redo()
        self._record(command)

    def push_applied(self, command: Command) -> None:
        self._record(command)

    def _record(self, command: Command) -> None:
        self._undo.append(command)
        self._redo.clear()
        if len(self._undo) > self._capacity:
            del self._undo[: len(self._undo) - self._capacity]
        self._notify()

    # -- traversal ---------------------------------------------------------- #
    def undo(self) -> bool:
        if not self._undo:
            return False
        command = self._undo.pop()
        command.undo()
        self._redo.append(command)
        self._notify()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        command = self._redo.pop()
        command.redo()
        self._undo.append(command)
        self._notify()
        return True

    # -- queries ------------------------------------------------------------ #
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> str:
        return self._undo[-1].label if self._undo else ""

    @property
    def redo_label(self) -> str:
        return self._redo[-1].label if self._redo else ""

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._notify()

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()


# --------------------------------------------------------------------------- #
# Concrete commands
# --------------------------------------------------------------------------- #


class AddStonesCommand(Command):
    """Add pre-built stones (ids already allocated) to a design."""

    def __init__(self, design: Design, stones: list[Stone]) -> None:
        self._design = design
        self._stones = list(stones)
        self.label = f"Add {len(self._stones)} stone{'s' if len(self._stones) != 1 else ''}"

    def redo(self) -> None:
        for s in self._stones:
            self._design.add(s)

    def undo(self) -> None:
        self._design.remove_many([s.id for s in self._stones])


class DeleteStonesCommand(Command):
    """Delete stones by id, capturing them so the deletion can be undone."""

    def __init__(self, design: Design, stone_ids: Iterable[int]) -> None:
        self._design = design
        self._ids = list(stone_ids)
        self._removed: list[Stone] = []
        n = len(self._ids)
        self.label = f"Delete {n} stone{'s' if n != 1 else ''}"

    @classmethod
    def from_removed(cls, design: Design, removed: list[Stone], label: str | None = None) -> "DeleteStonesCommand":
        """Build a command for stones already removed live (e.g. the eraser).

        ``redo`` re-removes them; ``undo`` restores them. Because the effect is
        already applied, push it with :meth:`CommandStack.push_applied`.
        """
        cmd = cls(design, [s.id for s in removed])
        cmd._removed = list(removed)
        if label:
            cmd.label = label
        return cmd

    def redo(self) -> None:
        self._removed = self._design.remove_many(self._ids)

    def undo(self) -> None:
        for s in self._removed:
            self._design.add(s)


class MoveStonesCommand(Command):
    """Move stones. ``moves`` maps stone id → ``((old_x, old_y), (new_x, new_y))``.

    Records only coordinates, so a drag of thousands of stones costs a couple of
    floats each — not a copy of every stone.
    """

    def __init__(self, design: Design, moves: dict[int, tuple[tuple[float, float], tuple[float, float]]]) -> None:
        self._design = design
        self._moves = moves
        n = len(moves)
        self.label = f"Move {n} stone{'s' if n != 1 else ''}"

    def _apply(self, index: int) -> None:
        for sid, pair in self._moves.items():
            stone = self._design.get_by_id(sid)
            if stone is not None:
                stone.x, stone.y = pair[index]

    def redo(self) -> None:
        self._apply(1)

    def undo(self) -> None:
        self._apply(0)


class ReplaceDesignCommand(Command):
    """Replace all stones (used by full regeneration). Holds both stone sets."""

    def __init__(self, design: Design, old_stones: list[Stone], new_stones: list[Stone],
                 label: str = "Generate design") -> None:
        self._design = design
        self._old = list(old_stones)
        self._new = list(new_stones)
        self.label = label

    def redo(self) -> None:
        self._design.replace_all(self._new)

    def undo(self) -> None:
        self._design.replace_all(self._old)


class CompositeCommand(Command):
    """Several commands applied as ONE undoable step (e.g. an AI suggestion).

    ``redo`` runs the children in order; ``undo`` reverses them in reverse
    order, so the whole group reverts as a unit. Children must be independent
    enough to redo/undo in this fixed sequence (the AI assistant orders its
    children deletes → moves → adds, whose undo order is exactly correct).
    """

    def __init__(self, commands: list[Command], label: str = "Edit") -> None:
        self._commands = list(commands)
        self.label = label

    def redo(self) -> None:
        for command in self._commands:
            command.redo()

    def undo(self) -> None:
        for command in reversed(self._commands):
            command.undo()


class PaintTraceCommand(Command):
    """An undoable trace-tool stroke over an existing mask (paint or erase).

    The canvas snapshots the whole mask at stroke start (a transient copy);
    the command keeps only that snapshot's *changed bounding box* patch, so a
    long session of strokes costs a few KB each, not a full raster per stroke.

    Undo/redo swap the patch **in place** on the project's mask array, so
    every reference to that array (the canvas overlay, a running generation
    gate) stays valid. The first-ever stroke (mask was ``None``) is recorded
    as a :class:`SetTraceMaskCommand` by the canvas instead — this command
    requires a pre-existing mask.
    """

    def __init__(self, project, before, bbox, label: str = "Trace area", on_change=None) -> None:
        from app.core.trace_mask import extract_patch

        if before is None or bbox is None:
            raise ValueError("PaintTraceCommand needs a before-mask and a changed bbox")
        self._project = project
        self._bbox = bbox
        self._before_patch = extract_patch(before, bbox)
        self._on_change = on_change
        self.label = label

    def _swap(self) -> None:
        from app.core.trace_mask import apply_patch, extract_patch

        mask = self._project.trace_mask
        if mask is None:
            return
        current = extract_patch(mask, self._bbox)
        apply_patch(mask, self._bbox, self._before_patch)
        self._before_patch = current
        if self._on_change is not None:
            self._on_change()

    def redo(self) -> None:
        self._swap()

    def undo(self) -> None:
        self._swap()


class SetTraceMaskCommand(Command):
    """Swap the whole trace mask (Clear Traced Area, first stroke, its undo)."""

    def __init__(self, project, before, after, on_change=None, label: str = "Clear trace") -> None:
        self._project = project
        self._before = before
        self._after = after
        self._on_change = on_change
        self.label = label

    def _apply(self, mask) -> None:
        self._project.trace_mask = mask
        if self._on_change is not None:
            self._on_change()

    def redo(self) -> None:
        self._apply(self._after)

    def undo(self) -> None:
        self._apply(self._before)


class CropCommand(Command):
    """An undoable image crop.

    Crop mutates three coupled pieces of project state at once:

    * the image ``crop`` rectangle (in oriented-image pixel space),
    * the physical design size (the crop changes aspect, so a locked-aspect
      resize follows), and
    * the generated design (stones are cleared/regenerated against the new
      image field).

    Rather than snapshot a raster (which would be lossy and large), the command
    captures the *values* of these fields before and after the crop and swaps
    them on undo/redo. The caller is responsible for recomputing the canvas view
    (background image, fit) after the command is applied — via the optional
    ``on_change`` hook, fired once per apply.
    """

    def __init__(
        self,
        image_state,
        project,
        *,
        old_crop,
        new_crop,
        old_size,
        new_size,
        old_stones,
        new_stones,
        on_change=None,
        label: str = "Crop image",
    ) -> None:
        self._image = image_state
        self._project = project
        self._old_crop = old_crop
        self._new_crop = new_crop
        self._old_size = old_size
        self._new_size = new_size
        self._old_stones = list(old_stones)
        self._new_stones = list(new_stones)
        self._on_change = on_change
        self.label = label

    def _apply(self, crop, size, stones) -> None:
        self._image.crop = crop
        self._project.width_mm, self._project.height_mm = size
        self._project.design.resize(*size)
        self._project.design.replace_all(list(stones))
        if self._on_change is not None:
            self._on_change()

    def redo(self) -> None:
        self._apply(self._new_crop, self._new_size, self._new_stones)

    def undo(self) -> None:
        self._apply(self._old_crop, self._old_size, self._old_stones)
