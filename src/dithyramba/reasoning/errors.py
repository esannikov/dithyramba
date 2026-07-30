"""Typed failures for public, evidence-grounded reasoning artifacts."""

from __future__ import annotations

from dithyramba.contracts import ContractError


class ReasoningError(ContractError):
    """Base class for IdeaTrace contract and closure failures."""


class ReasoningPersistenceError(ReasoningError):
    """An IdeaTrace or closure receipt could not be stored or reopened exactly."""


class ReasoningNotFoundError(ReasoningPersistenceError):
    """A requested reasoning artifact does not exist in the current Library."""
