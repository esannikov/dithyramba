"""Canonical JSON encoding used for hashes, receipts, and stable artifacts.

The accepted data model is intentionally smaller than what ``json.dumps``
will coerce: null, booleans, integers, strings, lists, and dictionaries whose
keys are strings. Strings and dictionary keys are normalized to Unicode NFC
before serialization. Floats are forbidden because their cross-runtime
representation is not part of the VS0 contract.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from .errors import (
    CanonicalizationError,
    InvalidCanonicalKeyError,
    NormalizationCollisionError,
    UnsupportedCanonicalTypeError,
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical UTF-8 JSON representation of ``value``.

    Dictionary keys are sorted, insignificant whitespace is omitted, and
    non-ASCII text is emitted directly. Invalid values fail closed instead of
    being converted implicitly.
    """

    normalized = _normalize(value, path="$", active_containers=set())
    try:
        serialized = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return serialized.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        # The recursive validator should catch ordinary type errors. Wrapping
        # here also gives callers one stable exception for cyclic input or an
        # invalid Unicode scalar.
        raise CanonicalizationError(f"value cannot be serialized canonically: {exc}") from exc


def _normalize(value: Any, *, path: str, active_containers: set[int]) -> Any:
    if value is None or type(value) in (bool, int):
        return value

    if type(value) is str:
        return unicodedata.normalize("NFC", value)

    if type(value) is float:
        raise UnsupportedCanonicalTypeError(f"floats are forbidden at {path}")

    if type(value) is list:
        container_id = id(value)
        if container_id in active_containers:
            raise CanonicalizationError(f"cyclic container reference at {path}")
        active_containers.add(container_id)
        try:
            return [
                _normalize(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(container_id)

    if type(value) is dict:
        container_id = id(value)
        if container_id in active_containers:
            raise CanonicalizationError(f"cyclic container reference at {path}")
        active_containers.add(container_id)
        normalized: dict[str, Any] = {}
        original_keys: dict[str, str] = {}
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise InvalidCanonicalKeyError(
                        f"mapping key at {path} must be str, got {type(key).__name__}"
                    )

                normalized_key = unicodedata.normalize("NFC", key)
                if normalized_key in normalized:
                    first = original_keys[normalized_key]
                    raise NormalizationCollisionError(
                        f"keys {first!r} and {key!r} at {path} normalize to {normalized_key!r}"
                    )

                normalized[normalized_key] = _normalize(
                    item,
                    path=f"{path}.{normalized_key}",
                    active_containers=active_containers,
                )
                original_keys[normalized_key] = key
            return normalized
        finally:
            active_containers.remove(container_id)

    raise UnsupportedCanonicalTypeError(
        f"unsupported canonical type {type(value).__name__} at {path}"
    )
