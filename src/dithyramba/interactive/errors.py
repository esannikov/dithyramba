"""Typed failures for the agent-facing interactive research facade."""

from __future__ import annotations


class AgentResearchError(RuntimeError):
    """An agent research command violates the bounded session contract."""
