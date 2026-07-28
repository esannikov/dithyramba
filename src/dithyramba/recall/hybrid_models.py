"""Versioned, model-agnostic contracts for P7 hybrid recall."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.access import QueryExclusions
from dithyramba.contracts import canonical_content_id, canonical_json_bytes, canonical_sha256_hex

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_PURPOSE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class HybridContractError(ValueError):
    """A P7 hybrid artifact violates its frozen public contract."""


class CorpusLayer(StrEnum):
    """Declared source role; this never changes Collection semantics."""

    PRIMARY = "primary"
    DERIVED = "derived"
    SYNTHESIS = "synthesis"
    UNCLASSIFIED = "unclassified"


class RetrievalChannel(StrEnum):
    FTS = "fts"
    DENSE = "dense"
    FUSED = "fused"
    RERANKED = "reranked"


class EmbeddingModelProfile(BaseModel):
    """Immutable identity and input contract for one local embedding model."""

    SCHEMA: ClassVar[str] = "dithyramba.embedding_model_profile/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    provider: Literal["sentence_transformers"] = "sentence_transformers"
    model_id: str
    revision: str
    license: str
    dimensions: int = Field(ge=1, le=8_192)
    max_tokens: int = Field(ge=1, le=32_768)
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize_embeddings: Literal[True] = True
    trust_remote_code: Literal[False] = False

    @field_validator("model_id")
    @classmethod
    def _model_id(cls, value: str) -> str:
        if type(value) is not str or _MODEL_ID_PATTERN.fullmatch(value) is None:
            raise HybridContractError("model_id must be an exact owner/repository identifier")
        return value

    @field_validator("revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        if type(value) is not str or _REVISION_PATTERN.fullmatch(value) is None:
            raise HybridContractError("model revision must be a 40-character lowercase commit")
        return value

    @field_validator("license")
    @classmethod
    def _license(cls, value: str) -> str:
        return _require_label(value, "model license")

    @field_validator("query_prefix", "passage_prefix")
    @classmethod
    def _prefix(cls, value: str) -> str:
        if type(value) is not str or len(value) > 128 or "\x00" in value:
            raise HybridContractError("embedding prefixes must be bounded text without NUL")
        normalized = unicodedata.normalize("NFC", value)
        if normalized != value:
            raise HybridContractError("embedding prefixes must already be NFC")
        return value

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "dimensions": self.dimensions,
            "max_tokens": self.max_tokens,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "normalize_embeddings": self.normalize_embeddings,
            "trust_remote_code": self.trust_remote_code,
        }

    @property
    def profile_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def profile_id(self) -> str:
        return canonical_content_id("embedding_profile", self.semantic_payload())


class ModelRuntimeProfile(BaseModel):
    """Exact local runtime identity; network access is forbidden at query time."""

    SCHEMA: ClassVar[str] = "dithyramba.model_runtime_profile/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sentence_transformers_version: str
    transformers_version: str
    torch_version: str
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    dtype: Literal["float32"] = "float32"
    batch_size: int = Field(default=16, ge=1, le=512)
    local_files_only: Literal[True] = True
    trust_remote_code: Literal[False] = False

    @field_validator(
        "sentence_transformers_version",
        "transformers_version",
        "torch_version",
    )
    @classmethod
    def _version(cls, value: str) -> str:
        return _require_label(value, "runtime version")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "sentence_transformers_version": self.sentence_transformers_version,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }

    @property
    def profile_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def profile_id(self) -> str:
        return canonical_content_id("runtime_profile", self.semantic_payload())


class CorpusLayerItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source_id: str
    layer: CorpusLayer

    @field_validator("source_id")
    @classmethod
    def _source_id(cls, value: str) -> str:
        return _require_id(value, "source")

    def payload(self) -> dict[str, str]:
        return {"source_id": self.source_id, "layer": self.layer.value}


class CorpusLayerProfile(BaseModel):
    """Content-addressed Source-role map for one exact permitted manifest."""

    SCHEMA: ClassVar[str] = "dithyramba.corpus_layer_profile/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    library_id: str
    corpus_snapshot_id: str
    exclusion_hash: str
    permitted_set_hash: str
    items: tuple[CorpusLayerItem, ...]

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("corpus_snapshot_id")
    @classmethod
    def _snapshot_id(cls, value: str) -> str:
        return _require_id(value, "snapshot")

    @field_validator("exclusion_hash", "permitted_set_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("items")
    @classmethod
    def _items(cls, value: tuple[CorpusLayerItem, ...]) -> tuple[CorpusLayerItem, ...]:
        ordered = tuple(sorted(value, key=lambda item: item.source_id.encode("ascii")))
        if len({item.source_id for item in ordered}) != len(ordered):
            raise HybridContractError("CorpusLayerProfile Source IDs must be unique")
        return ordered

    def require_exact_sources(
        self,
        permitted_source_ids: tuple[str, ...],
        *,
        benchmark: bool = False,
    ) -> None:
        expected = tuple(sorted(permitted_source_ids, key=lambda value: value.encode("ascii")))
        if len(set(expected)) != len(expected):
            raise HybridContractError("permitted Source IDs must be unique")
        actual = tuple(item.source_id for item in self.items)
        if actual != expected:
            raise HybridContractError("CorpusLayerProfile must cover the exact permitted Sources")
        if benchmark:
            if not expected:
                raise HybridContractError("benchmark CorpusLayerProfile cannot be empty")
            if any(item.layer is CorpusLayer.UNCLASSIFIED for item in self.items):
                raise HybridContractError(
                    "benchmark CorpusLayerProfile cannot contain unclassified"
                )

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "exclusion_hash": self.exclusion_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "items": [item.payload() for item in self.items],
        }

    @property
    def profile_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def profile_id(self) -> str:
        return canonical_content_id("layer_profile", self.semantic_payload())


class FusionProfile(BaseModel):
    """Frozen exact reciprocal-rank fusion profile."""

    SCHEMA: ClassVar[str] = "dithyramba.fusion_profile/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    algorithm: Literal["rrf_v1"] = "rrf_v1"
    rrf_k: Literal[60] = 60
    fts_weight: Literal[1] = 1
    dense_weight: Literal[1] = 1
    max_per_derived_source_top_five: Literal[2] = 2

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "algorithm": self.algorithm,
            "rrf_k": self.rrf_k,
            "fts_weight": self.fts_weight,
            "dense_weight": self.dense_weight,
            "max_per_derived_source_top_five": self.max_per_derived_source_top_five,
        }

    @property
    def profile_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())


class HybridRetrievalBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_fts_candidates: int = Field(default=100, ge=1, le=500)
    max_dense_candidates: int = Field(default=100, ge=1, le=500)
    max_fused_candidates: int = Field(default=100, ge=1, le=500)
    max_source_fragments: int = Field(default=30, ge=1, le=100)

    @model_validator(mode="after")
    def _selection_fits_pool(self) -> HybridRetrievalBudget:
        if self.max_source_fragments > self.max_fused_candidates:
            raise HybridContractError("max_source_fragments must not exceed fused candidates")
        return self

    def payload(self) -> dict[str, int]:
        return {
            "max_fts_candidates": self.max_fts_candidates,
            "max_dense_candidates": self.max_dense_candidates,
            "max_fused_candidates": self.max_fused_candidates,
            "max_source_fragments": self.max_source_fragments,
        }


class HybridQueryRequest(BaseModel):
    """A new P7 request; frozen VS0 QueryRequest/1.0 stays byte-compatible."""

    SCHEMA: ClassVar[str] = "dithyramba.hybrid_query_request/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    question: str
    library_id: str
    collection_ids: tuple[str, ...]
    corpus_snapshot_id: str
    access_policy_id: str
    purpose: str
    model_profile_hash: str
    runtime_profile_hash: str
    layer_profile_hash: str
    fusion_profile_hash: str
    reranker_profile_hash: str | None = None
    exclusions: QueryExclusions = Field(default_factory=QueryExclusions)
    retrieval: HybridRetrievalBudget = Field(default_factory=HybridRetrievalBudget)
    result_format: Literal["hybrid_evidence_packet"] = "hybrid_evidence_packet"

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFC", value)
        if (
            type(value) is not str
            or normalized != value
            or not value.strip()
            or value != value.strip()
            or "\x00" in value
            or len(value) > 2_000
        ):
            raise HybridContractError("question must be bounded, unpadded NFC text without NUL")
        return value

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

    @field_validator("collection_ids")
    @classmethod
    def _collections(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 16:
            raise HybridContractError("HybridQueryRequest requires 1-16 Collections")
        for collection_id in value:
            _require_id(collection_id, "collection")
        if len(set(value)) != len(value):
            raise HybridContractError("HybridQueryRequest Collection IDs must be unique")
        return tuple(sorted(value, key=lambda item: item.encode("ascii")))

    @field_validator("purpose")
    @classmethod
    def _purpose(cls, value: str) -> str:
        if type(value) is not str or _PURPOSE_PATTERN.fullmatch(value) is None:
            raise HybridContractError("purpose must match [a-z][a-z0-9_-]{0,63}")
        return value

    @field_validator(
        "model_profile_hash",
        "runtime_profile_hash",
        "layer_profile_hash",
        "fusion_profile_hash",
        "reranker_profile_hash",
    )
    @classmethod
    def _artifact_hash(cls, value: str | None) -> str | None:
        return None if value is None else _require_hash(value)

    @field_validator("exclusions")
    @classmethod
    def _exclusions(cls, value: QueryExclusions) -> QueryExclusions:
        if type(value) is not QueryExclusions:
            raise HybridContractError("exclusions must be an exact QueryExclusions")
        for source_id in value.source_ids:
            _require_id(source_id, "source")
        for family_id in value.source_family_ids:
            _require_id(family_id, "family")
        for fragment_id in value.source_fragment_ids:
            _require_id(fragment_id, "fragment")
        return value

    @field_validator("retrieval")
    @classmethod
    def _retrieval(cls, value: HybridRetrievalBudget) -> HybridRetrievalBudget:
        if type(value) is not HybridRetrievalBudget:
            raise HybridContractError("retrieval must be an exact HybridRetrievalBudget")
        return value

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "question": self.question,
            "library_id": self.library_id,
            "collection_ids": list(self.collection_ids),
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "purpose": self.purpose,
            "model_profile_hash": self.model_profile_hash,
            "runtime_profile_hash": self.runtime_profile_hash,
            "layer_profile_hash": self.layer_profile_hash,
            "fusion_profile_hash": self.fusion_profile_hash,
            "reranker_profile_hash": self.reranker_profile_hash,
            "exclusions": self.exclusions.payload(),
            "retrieval": self.retrieval.payload(),
            "result_format": self.result_format,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def request_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def request_id(self) -> str:
        return canonical_content_id("hybrid_query", self.semantic_payload())


class HybridRankTraceItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source_fragment_id: str
    source_id: str
    rank: int = Field(ge=1, le=500)
    score_repr: str

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        return _require_id(value, "fragment")

    @field_validator("source_id")
    @classmethod
    def _source_id(cls, value: str) -> str:
        return _require_id(value, "source")

    @field_validator("score_repr")
    @classmethod
    def _score(cls, value: str) -> str:
        if type(value) is not str or not value or len(value) > 128 or "\x00" in value:
            raise HybridContractError("trace score representation is invalid")
        return value

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_id": self.source_id,
            "rank": self.rank,
            "score_repr": self.score_repr,
        }


class HybridRetrievalReceipt(BaseModel):
    """Deterministic channel trace; runtime timings deliberately live elsewhere."""

    SCHEMA: ClassVar[str] = "dithyramba.hybrid_retrieval_receipt/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request_id: str
    request_hash: str
    exclusion_hash: str
    permitted_set_hash: str
    model_profile_hash: str
    runtime_profile_hash: str
    layer_profile_hash: str
    fusion_profile_hash: str
    vector_generation_hash: str
    fts_profile_version: str
    fts_result_hash: str
    query_vector_hash: str
    reranker_profile_hash: str | None = None
    fts: tuple[HybridRankTraceItem, ...] = ()
    dense: tuple[HybridRankTraceItem, ...] = ()
    fused: tuple[HybridRankTraceItem, ...] = ()
    reranked: tuple[HybridRankTraceItem, ...] = ()

    @field_validator("request_id")
    @classmethod
    def _request_id(cls, value: str) -> str:
        return _require_id(value, "hybrid_query")

    @field_validator(
        "request_hash",
        "exclusion_hash",
        "permitted_set_hash",
        "model_profile_hash",
        "runtime_profile_hash",
        "layer_profile_hash",
        "fusion_profile_hash",
        "vector_generation_hash",
        "fts_result_hash",
        "query_vector_hash",
        "reranker_profile_hash",
    )
    @classmethod
    def _hashes(cls, value: str | None) -> str | None:
        return None if value is None else _require_hash(value)

    @field_validator("fts_profile_version")
    @classmethod
    def _fts_profile_version(cls, value: str) -> str:
        return _require_label(value, "FTS profile version")

    @model_validator(mode="after")
    def _trace_closure(self) -> HybridRetrievalReceipt:
        for name, trace in (
            ("fts", self.fts),
            ("dense", self.dense),
            ("fused", self.fused),
            ("reranked", self.reranked),
        ):
            _validate_trace(name, trace)
        input_ids = {item.source_fragment_id for item in (*self.fts, *self.dense)}
        fused_ids = {item.source_fragment_id for item in self.fused}
        if not fused_ids.issubset(input_ids):
            raise HybridContractError("fused trace may contain only FTS or dense candidates")
        reranked_ids = {item.source_fragment_id for item in self.reranked}
        if self.reranker_profile_hash is None:
            if reranked_ids:
                raise HybridContractError("reranked trace requires a reranker profile")
        elif reranked_ids != fused_ids:
            raise HybridContractError("reranker trace must preserve the exact fused candidate set")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "exclusion_hash": self.exclusion_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "model_profile_hash": self.model_profile_hash,
            "runtime_profile_hash": self.runtime_profile_hash,
            "layer_profile_hash": self.layer_profile_hash,
            "fusion_profile_hash": self.fusion_profile_hash,
            "vector_generation_hash": self.vector_generation_hash,
            "fts_profile_version": self.fts_profile_version,
            "fts_result_hash": self.fts_result_hash,
            "query_vector_hash": self.query_vector_hash,
            "reranker_profile_hash": self.reranker_profile_hash,
            "fts": [item.payload() for item in self.fts],
            "dense": [item.payload() for item in self.dense],
            "fused": [item.payload() for item in self.fused],
            "reranked": [item.payload() for item in self.reranked],
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def receipt_id(self) -> str:
        return canonical_content_id("hybrid_retrieval", self.semantic_payload())


class HybridPacketItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rank: int = Field(ge=1, le=100)
    source_fragment_id: str
    source_id: str
    source_family_id: str
    layer: CorpusLayer

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        return _require_id(value, "fragment")

    @field_validator("source_id")
    @classmethod
    def _source_id(cls, value: str) -> str:
        return _require_id(value, "source")

    @field_validator("source_family_id")
    @classmethod
    def _family_id(cls, value: str) -> str:
        return _require_id(value, "family")

    def payload(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "source_fragment_id": self.source_fragment_id,
            "source_id": self.source_id,
            "source_family_id": self.source_family_id,
            "layer": self.layer.value,
        }


class HybridEvidencePacket(BaseModel):
    """Text-free P7 packet manifest; exact source text remains packet-authorized storage."""

    SCHEMA: ClassVar[str] = "dithyramba.hybrid_evidence_packet/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request_id: str
    request_hash: str
    corpus_snapshot_id: str
    access_policy_id: str
    read_receipt_hash: str
    access_receipt_hash: str
    coverage_report_hash: str
    retrieval_receipt_hash: str
    result_status: Literal["evidence_found", "no_evidence"]
    items: tuple[HybridPacketItem, ...] = ()

    @field_validator("request_id")
    @classmethod
    def _request_id(cls, value: str) -> str:
        return _require_id(value, "hybrid_query")

    @field_validator("corpus_snapshot_id")
    @classmethod
    def _snapshot_id(cls, value: str) -> str:
        return _require_id(value, "snapshot")

    @field_validator("access_policy_id")
    @classmethod
    def _policy_id(cls, value: str) -> str:
        return _require_id(value, "policy")

    @field_validator(
        "request_hash",
        "read_receipt_hash",
        "access_receipt_hash",
        "coverage_report_hash",
        "retrieval_receipt_hash",
    )
    @classmethod
    def _hash(cls, value: str) -> str:
        return _require_hash(value)

    @model_validator(mode="after")
    def _items_match_status(self) -> HybridEvidencePacket:
        _validate_packet_items(self.items)
        if (self.result_status == "evidence_found") != bool(self.items):
            raise HybridContractError("packet result_status must match whether evidence exists")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "read_receipt_hash": self.read_receipt_hash,
            "access_receipt_hash": self.access_receipt_hash,
            "coverage_report_hash": self.coverage_report_hash,
            "retrieval_receipt_hash": self.retrieval_receipt_hash,
            "result_status": self.result_status,
            "items": [item.payload() for item in self.items],
        }

    @property
    def packet_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def packet_id(self) -> str:
        return canonical_content_id("hybrid_packet", self.semantic_payload())


class HybridRunObservation(BaseModel):
    """Non-semantic operational measurements, excluded from all retrieval hashes."""

    SCHEMA: ClassVar[str] = "dithyramba.hybrid_run_observation/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    processing_run_id: str
    request_id: str
    cold_model_load_ns: int = Field(ge=0)
    passage_embedding_ns: int = Field(ge=0)
    query_embedding_ns: int = Field(ge=0)
    retrieval_ns: int = Field(ge=0)
    reranker_ns: int = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
    cache_bytes: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    observed_at: str

    @field_validator("processing_run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        return _require_id(value, "run")

    @field_validator("request_id")
    @classmethod
    def _request_id(cls, value: str) -> str:
        return _require_id(value, "hybrid_query")

    @field_validator("observed_at")
    @classmethod
    def _timestamp(cls, value: str) -> str:
        if type(value) is not str or not value.endswith("Z"):
            raise HybridContractError("observed_at must be an explicit UTC timestamp")
        try:
            parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
        except ValueError as error:
            raise HybridContractError("observed_at must be a valid UTC timestamp") from error
        offset = parsed.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise HybridContractError("observed_at must be UTC")
        return value

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "processing_run_id": self.processing_run_id,
            "request_id": self.request_id,
            "cold_model_load_ns": self.cold_model_load_ns,
            "passage_embedding_ns": self.passage_embedding_ns,
            "query_embedding_ns": self.query_embedding_ns,
            "retrieval_ns": self.retrieval_ns,
            "reranker_ns": self.reranker_ns,
            "peak_rss_bytes": self.peak_rss_bytes,
            "cache_bytes": self.cache_bytes,
            "provider_calls": self.provider_calls,
            "observed_at": self.observed_at,
        }

    @property
    def observation_hash(self) -> str:
        return canonical_sha256_hex(self.payload())


def _validate_trace(name: str, trace: tuple[HybridRankTraceItem, ...]) -> None:
    ids = tuple(item.source_fragment_id for item in trace)
    if len(set(ids)) != len(ids):
        raise HybridContractError(f"{name} trace contains duplicate fragment IDs")
    if tuple(item.rank for item in trace) != tuple(range(1, len(trace) + 1)):
        raise HybridContractError(f"{name} trace ranks must be contiguous from one")


def _validate_packet_items(items: tuple[HybridPacketItem, ...]) -> None:
    if len(items) > 100:
        raise HybridContractError("hybrid packet may contain at most 100 items")
    if tuple(item.rank for item in items) != tuple(range(1, len(items) + 1)):
        raise HybridContractError("hybrid packet ranks must be contiguous from one")
    if len({item.source_fragment_id for item in items}) != len(items):
        raise HybridContractError("hybrid packet fragment IDs must be unique")


def _require_hash(value: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise HybridContractError("artifact hash must be 64 lowercase hexadecimal characters")
    return value


def _require_label(value: str, field: str) -> str:
    if type(value) is not str or _LABEL_PATTERN.fullmatch(value) is None:
        raise HybridContractError(f"{field} must be a bounded ASCII label")
    return value


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise HybridContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise HybridContractError(f"identifier must use a canonical {expected} suffix")
    return value
