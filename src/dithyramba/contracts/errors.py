"""Typed failures raised by Dithyramba's deterministic contracts."""

from __future__ import annotations


class ContractError(ValueError):
    """Base class for values that violate a deterministic contract."""


class CanonicalizationError(ContractError):
    """A value cannot be represented by Dithyramba's canonical JSON subset."""


class InvalidCanonicalKeyError(CanonicalizationError):
    """A mapping has a key that is not a string."""


class NormalizationCollisionError(CanonicalizationError):
    """Two distinct mapping keys collapse to the same NFC string."""


class UnsupportedCanonicalTypeError(CanonicalizationError):
    """A value uses a type outside the canonical JSON subset."""


class InvalidIdentifierPrefixError(ContractError):
    """An identifier prefix is outside the public prefix grammar."""


class InvalidPayloadError(ContractError):
    """A hashing or identifier helper received a non-bytes payload."""
