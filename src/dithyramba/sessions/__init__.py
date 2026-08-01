"""Interactive, append-only research-session contracts."""

from .errors import ResearchSessionError, ResearchSessionReplayError
from .models import (
    GENESIS_EVENT_HASH,
    ResearchSession,
    ResearchSessionBrief,
    ResearchSessionState,
    SessionActorKind,
    SessionArtifactKind,
    SessionArtifactReference,
    SessionEvent,
    SessionEventKind,
    SessionStatus,
    SessionTextEntry,
)
from .service import ResearchSessionCoordinator

__all__ = [
    "GENESIS_EVENT_HASH",
    "ResearchSession",
    "ResearchSessionBrief",
    "ResearchSessionCoordinator",
    "ResearchSessionError",
    "ResearchSessionReplayError",
    "ResearchSessionState",
    "SessionActorKind",
    "SessionArtifactKind",
    "SessionArtifactReference",
    "SessionEvent",
    "SessionEventKind",
    "SessionStatus",
    "SessionTextEntry",
]
