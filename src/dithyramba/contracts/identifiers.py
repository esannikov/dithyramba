"""Opaque and content-derived identifiers for public VS0 entities."""

from __future__ import annotations

import re
import uuid
from typing import Any

from .canonical import canonical_json_bytes
from .errors import InvalidIdentifierPrefixError
from .hashing import BytesLike, sha256_hex

_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_MAX_PREFIX_LENGTH = 32


def random_id(prefix: str) -> str:
    """Create ``<prefix>_<uuid4 hex>`` using 128 random bits."""

    valid_prefix = _validate_prefix(prefix)
    return f"{valid_prefix}_{uuid.uuid4().hex}"


def new_id(prefix: str) -> str:
    """Alias for :func:`random_id` for entity constructors."""

    return random_id(prefix)


def content_id(prefix: str, payload: BytesLike) -> str:
    """Create ``<prefix>_<first 32 SHA-256 hex chars>`` from bytes."""

    valid_prefix = _validate_prefix(prefix)
    return f"{valid_prefix}_{sha256_hex(payload)[:32]}"


def canonical_content_id(prefix: str, value: Any) -> str:
    """Create a content-derived ID from strict canonical JSON."""

    return content_id(prefix, canonical_json_bytes(value))


def _validate_prefix(prefix: str) -> str:
    if (
        type(prefix) is not str
        or len(prefix) > _MAX_PREFIX_LENGTH
        or _PREFIX_PATTERN.fullmatch(prefix) is None
    ):
        raise InvalidIdentifierPrefixError(
            "identifier prefix must be 1-32 lowercase ASCII letters/digits "
            "with single internal underscores, starting with a letter"
        )
    return prefix
