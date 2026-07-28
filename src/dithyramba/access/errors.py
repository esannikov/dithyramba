"""Typed failures for policy compilation and permitted-token verification."""

from __future__ import annotations

from dithyramba.contracts import ContractError


class AccessContractError(ContractError):
    """Base class for deterministic access-boundary failures."""


class InvalidAccessPolicyError(AccessContractError):
    """An access-policy snapshot is internally inconsistent."""


class InvalidAccessScopeError(AccessContractError):
    """A request scope or snapshot candidate violates the access contract."""


class PurposeNotAllowedError(AccessContractError):
    """The selected policy does not authorize the request purpose."""


class PermittedTokenError(AccessContractError):
    """A permitted token is malformed, stale, or used outside its bound tuple."""


class PermittedTokenMismatchError(PermittedTokenError):
    """A permitted token does not match one or more expected tuple components."""
