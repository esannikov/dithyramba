"""SHA-256 helpers with stable output formats."""

from __future__ import annotations

import hashlib
from typing import Any

from .canonical import canonical_json_bytes
from .errors import InvalidPayloadError

BytesLike = bytes | bytearray | memoryview


def sha256_hex(payload: BytesLike) -> str:
    """Return a lowercase, 64-character SHA-256 hexadecimal digest."""

    return hashlib.sha256(_as_bytes(payload)).hexdigest()


def sha256_digest(payload: BytesLike) -> str:
    """Return the algorithm-qualified digest ``sha256:<hex>``."""

    return f"sha256:{sha256_hex(payload)}"


def canonical_sha256_hex(value: Any) -> str:
    """Hash a value after strict canonical JSON encoding."""

    return sha256_hex(canonical_json_bytes(value))


def canonical_sha256_digest(value: Any) -> str:
    """Return an algorithm-qualified digest for canonical JSON."""

    return sha256_digest(canonical_json_bytes(value))


def _as_bytes(payload: BytesLike) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise InvalidPayloadError(
            f"SHA-256 payload must be bytes-like, got {type(payload).__name__}"
        )
    return bytes(payload)
