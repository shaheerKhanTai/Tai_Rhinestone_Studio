"""AI provider abstraction and registry.

:class:`AIProvider` is the single integration point for any suggestion source.
The application ships with exactly one provider — the deterministic, offline
*local rule-based assistant* — which is a heuristic interpreter, **not** a
machine-learning model, and is labelled as such everywhere it appears.

A cloud/LLM provider would implement the same interface; it must be explicitly
registered (and its endpoint/key configured) by the user — nothing registers
one automatically, and no image or project data is ever transmitted unless a
user-configured provider chooses to do so.

``AIRequest`` deliberately carries *summaries* (design geometry, a density
array, the user's command) rather than file paths: a provider can never touch
the user's files.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.ai.operations import AISuggestion, Region


@dataclass(slots=True)
class AIRequest:
    """Everything a provider may see. No file paths, no project handles."""

    command: str                                   # the user's natural-language request
    design_width_mm: float
    design_height_mm: float
    stone_ids: np.ndarray                          # (N,) int
    stone_positions_mm: np.ndarray                 # (N, 2) float
    stone_diameters_mm: np.ndarray                 # (N,) float
    gap_mm: float                                  # configured spacing gap
    stone_diameter_mm: float                       # configured default stone size
    #: Optional density map (H, W) in [0, 1] aligned with the design area —
    #: the same continuous field the generator uses. May be ``None``.
    density: np.ndarray | None = None
    region: Region | None = None                   # focus region (mm) or whole design


class AIProvider(ABC):
    """A source of structured stone suggestions."""

    #: Human-readable label shown in the UI (must not overstate capabilities).
    label: str = "AI provider"

    @abstractmethod
    def is_available(self) -> bool:
        """True when the provider can serve requests right now."""

    @abstractmethod
    def suggest(self, request: AIRequest) -> AISuggestion:
        """Turn a request into a structured suggestion (never code)."""


class ProviderRegistry:
    """Registry of available providers.

    The local assistant registers lazily on first access (to keep the import
    graph acyclic); any additional provider registers explicitly.
    """

    def __init__(self) -> None:
        self._providers: dict[str, AIProvider] = {}

    def _ensure_local(self) -> None:
        if not self._providers:
            from app.ai.local_provider import LocalRuleProvider

            self.register(LocalRuleProvider())

    def register(self, provider: AIProvider) -> None:
        self._providers[provider.label] = provider

    def available(self) -> list[AIProvider]:
        self._ensure_local()
        return [p for p in self._providers.values() if p.is_available()]

    def get(self, label: str) -> AIProvider | None:
        self._ensure_local()
        return self._providers.get(label)


#: Global registry. Cloud providers register here only after explicit user setup.
provider_registry = ProviderRegistry()
