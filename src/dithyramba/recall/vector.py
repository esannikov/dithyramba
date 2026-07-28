"""Deterministic float32 vector contracts and exact cosine reference scan."""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dithyramba.contracts import canonical_content_id, canonical_sha256_hex, sha256_hex

from .hybrid_models import HybridContractError

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_NORMALIZED_ABS_TOLERANCE = 1e-5
_MAX_DIMENSIONS = 8_192


class VectorContractError(HybridContractError):
    """A dense vector or exact-scan input is malformed."""


@dataclass(frozen=True, slots=True)
class PackedVector:
    """One normalized vector encoded as exact little-endian IEEE-754 float32."""

    dimensions: int
    blob: bytes
    vector_hash: str

    def __post_init__(self) -> None:
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= _MAX_DIMENSIONS:
            raise VectorContractError("vector dimensions are out of bounds")
        if type(self.blob) is not bytes:
            raise VectorContractError("vector blob must be immutable bytes")
        if type(self.vector_hash) is not str or _HASH_PATTERN.fullmatch(self.vector_hash) is None:
            raise VectorContractError("vector_hash must be lowercase SHA-256")
        if sha256_hex(self.blob) != self.vector_hash:
            raise VectorContractError("vector blob does not match vector_hash")
        _decode_normalized(self.blob, dimensions=self.dimensions)

    @property
    def values(self) -> tuple[float, ...]:
        return _decode_normalized(self.blob, dimensions=self.dimensions)


@dataclass(frozen=True, slots=True)
class DenseVectorItem:
    source_fragment_id: str
    source_id: str
    vector: PackedVector

    def __post_init__(self) -> None:
        _require_id(self.source_fragment_id, "fragment")
        _require_id(self.source_id, "source")
        if not isinstance(self.vector, PackedVector):
            raise VectorContractError("DenseVectorItem.vector must be PackedVector")


@dataclass(frozen=True, slots=True)
class DenseHit:
    source_fragment_id: str
    source_id: str
    rank: int
    score_hex: str

    @property
    def score(self) -> float:
        try:
            score = float.fromhex(self.score_hex)
        except (TypeError, ValueError) as error:
            raise VectorContractError("dense score is not an exact float.hex value") from error
        if not math.isfinite(score):
            raise VectorContractError("dense score must be finite")
        return score


class VectorManifestItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source_fragment_id: str
    text_sha256: str
    vector_sha256: str

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        return _require_id(value, "fragment")

    @field_validator("text_sha256", "vector_sha256")
    @classmethod
    def _hash(cls, value: str) -> str:
        return _require_hash(value)

    def payload(self) -> dict[str, str]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "text_sha256": self.text_sha256,
            "vector_sha256": self.vector_sha256,
        }


class CorpusVectorGeneration(BaseModel):
    """Rebuildable vectors bound to one exact authorized input capability."""

    SCHEMA: ClassVar[str] = "dithyramba.corpus_vector_generation/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    library_id: str
    corpus_snapshot_id: str
    snapshot_hash: str
    access_policy_id: str
    policy_hash: str
    exclusion_hash: str
    permitted_set_hash: str
    model_profile_hash: str
    runtime_profile_hash: str
    dimensions: int = Field(ge=1, le=_MAX_DIMENSIONS)
    items: tuple[VectorManifestItem, ...]

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("corpus_snapshot_id")
    @classmethod
    def _snapshot_id(cls, value: str) -> str:
        return _require_id(value, "snapshot")

    @field_validator("access_policy_id")
    @classmethod
    def _policy_id(cls, value: str) -> str:
        return _require_id(value, "policy")

    @field_validator(
        "snapshot_hash",
        "policy_hash",
        "exclusion_hash",
        "permitted_set_hash",
        "model_profile_hash",
        "runtime_profile_hash",
    )
    @classmethod
    def _hashes(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("items")
    @classmethod
    def _items(cls, value: tuple[VectorManifestItem, ...]) -> tuple[VectorManifestItem, ...]:
        ordered = tuple(sorted(value, key=lambda item: item.source_fragment_id.encode("ascii")))
        if len({item.source_fragment_id for item in ordered}) != len(ordered):
            raise VectorContractError("vector generation fragment IDs must be unique")
        return ordered

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "access_policy_id": self.access_policy_id,
            "policy_hash": self.policy_hash,
            "exclusion_hash": self.exclusion_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "model_profile_hash": self.model_profile_hash,
            "runtime_profile_hash": self.runtime_profile_hash,
            "dimensions": self.dimensions,
            "items": [item.payload() for item in self.items],
        }

    @property
    def generation_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def generation_id(self) -> str:
        return canonical_content_id("vector_generation", self.semantic_payload())


def pack_normalized_vector(values: Sequence[float]) -> PackedVector:
    """Encode an already-normalized vector without silently changing provider output."""

    if isinstance(values, (str, bytes, bytearray)):
        raise VectorContractError("vector values must be a numeric sequence")
    try:
        items = tuple(values)
    except TypeError as error:
        raise VectorContractError("vector values must be a finite sequence") from error
    if not 1 <= len(items) <= _MAX_DIMENSIONS:
        raise VectorContractError("vector dimensions are out of bounds")
    packed_parts: list[bytes] = []
    for value in items:
        if type(value) is not float or not math.isfinite(value):
            raise VectorContractError("vector values must be finite Python floats")
        try:
            encoded = struct.pack("<f", value)
        except (OverflowError, struct.error) as error:
            raise VectorContractError("vector value cannot be represented as float32") from error
        decoded = struct.unpack("<f", encoded)[0]
        if not math.isfinite(decoded):
            raise VectorContractError("float32 vector values must remain finite")
        packed_parts.append(encoded)
    blob = b"".join(packed_parts)
    _decode_normalized(blob, dimensions=len(items))
    return PackedVector(
        dimensions=len(items),
        blob=blob,
        vector_hash=sha256_hex(blob),
    )


def load_packed_vector(
    blob: bytes,
    *,
    dimensions: int,
    expected_hash: str,
) -> PackedVector:
    """Validate persisted bytes, dimension and hash before exact scan."""

    return PackedVector(
        dimensions=dimensions,
        blob=blob,
        vector_hash=expected_hash,
    )


def exact_cosine_scan(
    query: PackedVector,
    candidates: Iterable[DenseVectorItem],
    *,
    limit: int,
) -> tuple[DenseHit, ...]:
    """Reference exact scan over normalized vectors with deterministic ties."""

    if not isinstance(query, PackedVector):
        raise VectorContractError("query must be a PackedVector")
    if type(limit) is not int or not 1 <= limit <= 500:
        raise VectorContractError("exact scan limit must be between 1 and 500")
    query_values = query.values
    seen: set[str] = set()
    scored: list[tuple[float, DenseVectorItem]] = []
    for candidate in candidates:
        if not isinstance(candidate, DenseVectorItem):
            raise VectorContractError("exact scan candidates must be DenseVectorItem values")
        if candidate.source_fragment_id in seen:
            raise VectorContractError("exact scan candidate IDs must be unique")
        seen.add(candidate.source_fragment_id)
        if candidate.vector.dimensions != query.dimensions:
            raise VectorContractError("query and candidate vector dimensions differ")
        score = math.fsum(
            left * right for left, right in zip(query_values, candidate.vector.values, strict=True)
        )
        if not math.isfinite(score):
            raise VectorContractError("exact cosine score is not finite")
        if score < -1.000001 or score > 1.000001:
            raise VectorContractError("exact cosine score is outside the normalized range")
        scored.append((max(-1.0, min(1.0, score)), candidate))
    ordered = sorted(
        scored,
        key=lambda item: (-item[0], item[1].source_fragment_id.encode("ascii")),
    )[:limit]
    return tuple(
        DenseHit(
            source_fragment_id=candidate.source_fragment_id,
            source_id=candidate.source_id,
            rank=rank,
            score_hex=score.hex(),
        )
        for rank, (score, candidate) in enumerate(ordered, start=1)
    )


def _decode_normalized(blob: bytes, *, dimensions: int) -> tuple[float, ...]:
    expected_size = dimensions * 4
    if len(blob) != expected_size:
        raise VectorContractError("vector blob length does not match dimensions")
    try:
        values = tuple(item[0] for item in struct.iter_unpack("<f", blob))
    except struct.error as error:
        raise VectorContractError("vector blob is not little-endian float32") from error
    if len(values) != dimensions or any(not math.isfinite(value) for value in values):
        raise VectorContractError("vector blob contains invalid float32 values")
    norm_squared = math.fsum(value * value for value in values)
    if norm_squared == 0.0:
        raise VectorContractError("zero vectors are forbidden")
    if not math.isclose(
        norm_squared,
        1.0,
        rel_tol=_NORMALIZED_ABS_TOLERANCE,
        abs_tol=_NORMALIZED_ABS_TOLERANCE,
    ):
        raise VectorContractError("vector must already be L2-normalized")
    return values


def _require_hash(value: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise VectorContractError("hash must be 64 lowercase hexadecimal characters")
    return value


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise VectorContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise VectorContractError(f"identifier must use a canonical {expected} suffix")
    return value
