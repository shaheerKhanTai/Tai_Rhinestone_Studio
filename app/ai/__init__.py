"""AI assistant integration (ISSUE 6).

The AI layer produces *structured suggestions* — typed operation records, never
code — which are geometry-validated against the authoritative engine
(:mod:`app.core.collision`, :mod:`app.core.geometry`) before anything reaches
the design. The user previews every suggestion and must explicitly Apply it;
an applied suggestion becomes exactly ONE undoable history action.

Privacy: the default provider runs entirely offline. No image data, telemetry
or project content ever leaves the machine; a future cloud provider must be
explicitly registered by the user.
"""
from app.ai.assistant import AIAssistant
from app.ai.operations import AISuggestion, OpKind, StoneOp
from app.ai.provider import AIProvider, AIRequest, provider_registry
from app.ai.validator import ValidatedSuggestion, validate_suggestion

__all__ = [
    "AIAssistant",
    "AIProvider",
    "AIRequest",
    "AISuggestion",
    "OpKind",
    "StoneOp",
    "ValidatedSuggestion",
    "provider_registry",
    "validate_suggestion",
]
