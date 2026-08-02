"""Least-context Python facade for continuing source-grounded research."""

from .errors import AgentResearchError
from .models import (
    AgentAnswerPreparation,
    AgentEvidencePacket,
    AgentResearchTurn,
    AgentSessionContext,
    AgentSourceDrilldown,
    AgentSourceReference,
    SessionContextBudget,
    SessionContextOmissions,
)
from .service import AgentResearchFacade

__all__ = [
    "AgentAnswerPreparation",
    "AgentEvidencePacket",
    "AgentResearchError",
    "AgentResearchFacade",
    "AgentResearchTurn",
    "AgentSessionContext",
    "AgentSourceDrilldown",
    "AgentSourceReference",
    "SessionContextBudget",
    "SessionContextOmissions",
]
