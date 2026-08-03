"""Typed failures for interactive research-session contracts."""

from __future__ import annotations

from dithyramba.contracts import ContractError


class ResearchSessionError(ContractError):
    """Base error for research-session contract and replay failures."""


class ResearchSessionReplayError(ResearchSessionError):
    """An event stream cannot be replayed as one exact session."""
