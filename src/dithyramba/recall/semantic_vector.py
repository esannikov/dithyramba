"""Lossless SemanticSpan vector-v2 materialization and grouped exact scan.

Source text is accepted only at the materialization boundary.  Every durable
contract in this module contains identities, offsets already committed by a
``SemanticSpanPlan``, and hashes; it never serializes source or span text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)
from dithyramba.persistence.models import SourceFragmentText

from .hybrid_models import EmbeddingModelProfile, HybridContractError, ModelRuntimeProfile
from .provisioning import ModelProvisioningReceipt
from .semantic_audit import (
    SemanticAuditItemStatus,
    SemanticCoverageAudit,
)
from .semantic_spans import SemanticSpan, SemanticSpanPlan
from .vector import PackedVector

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_MAX_DIMENSIONS = 8_192
_MAX_EMBED_BATCH_ITEMS = 512
_MAX_EMBED_BATCH_CHARACTERS = 2_000_000

SemanticVectorNormalization = Literal["l2_unit_float32_v1"]
SEMANTIC_VECTOR_NORMALIZATION: SemanticVectorNormalization = "l2_unit_float32_v1"


class SemanticSpanVectorError(HybridContractError):
    """SemanticSpan vectors or their exact-scan bindings are inconsistent."""


class SemanticSpanEmbeddingProvider(Protocol):
    """Exact local provider seam needed by vector-v2 materialization."""

    @property
    def model_profile(self) -> EmbeddingModelProfile: ...

    @property
    def runtime_profile(self) -> ModelRuntimeProfile: ...

    @property
    def provisioning_receipt(self) -> ModelProvisioningReceipt: ...

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int: ...

    def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]: ...


@dataclass(frozen=True, slots=True)
class SemanticSpanVectorManifestItem:
    """One ordered, text-free SemanticSpan/vector identity binding."""

    semantic_span_id: str
    semantic_span_hash: str
    parent_fragment_id: str
    span_ordinal: int
    span_text_sha256: str
    vector_sha256: str

    def __post_init__(self) -> None:
        _require_id(self.semantic_span_id, "semantic_span")
        _require_id(self.parent_fragment_id, "fragment")
        for value in (
            self.semantic_span_hash,
            self.span_text_sha256,
            self.vector_sha256,
        ):
            _require_hash(value)
        if self.semantic_span_id != f"semantic_span_{self.semantic_span_hash[:32]}":
            raise SemanticSpanVectorError("SemanticSpan ID/hash pair is inconsistent")
        if type(self.span_ordinal) is not int or self.span_ordinal < 0:
            raise SemanticSpanVectorError("span_ordinal must be a non-negative integer")

    def payload(self) -> dict[str, object]:
        return {
            "semantic_span_id": self.semantic_span_id,
            "semantic_span_hash": self.semantic_span_hash,
            "parent_fragment_id": self.parent_fragment_id,
            "span_ordinal": self.span_ordinal,
            "span_text_sha256": self.span_text_sha256,
            "vector_sha256": self.vector_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticSpanVectorGeneration:
    """Complete vector-v2 closure for one authorized SemanticSpan plan."""

    SCHEMA: ClassVar[str] = "dithyramba.semantic_span_vector_generation/2.0"
    MANIFEST_SCHEMA: ClassVar[str] = "dithyramba.semantic_span_vector_manifest/2.0"

    library_id: str
    corpus_snapshot_id: str
    snapshot_hash: str
    access_policy_id: str
    policy_hash: str
    exclusion_hash: str
    permitted_set_hash: str
    semantic_audit_id: str
    semantic_audit_hash: str
    semantic_audit_binding_hash: str
    semantic_audit_fragment_manifest_hash: str
    semantic_span_plan_id: str
    semantic_span_plan_hash: str
    semantic_span_profile_hash: str
    semantic_span_parent_manifest_hash: str
    semantic_span_closure_hash: str
    model_profile_hash: str
    runtime_profile_hash: str
    provisioning_receipt_hash: str
    dimensions: int
    items: tuple[SemanticSpanVectorManifestItem, ...]
    normalization: SemanticVectorNormalization = SEMANTIC_VECTOR_NORMALIZATION

    def __post_init__(self) -> None:
        _require_id(self.library_id, "library")
        _require_id(self.corpus_snapshot_id, "snapshot")
        _require_id(self.access_policy_id, "policy")
        _require_id(self.semantic_audit_id, "semantic_audit")
        _require_id(self.semantic_span_plan_id, "semantic_span_plan")
        for value in (
            self.snapshot_hash,
            self.policy_hash,
            self.exclusion_hash,
            self.permitted_set_hash,
            self.semantic_audit_hash,
            self.semantic_audit_binding_hash,
            self.semantic_audit_fragment_manifest_hash,
            self.semantic_span_plan_hash,
            self.semantic_span_profile_hash,
            self.semantic_span_parent_manifest_hash,
            self.semantic_span_closure_hash,
            self.model_profile_hash,
            self.runtime_profile_hash,
            self.provisioning_receipt_hash,
        ):
            _require_hash(value)
        if self.corpus_snapshot_id != f"snapshot_{self.snapshot_hash[:32]}":
            raise SemanticSpanVectorError("snapshot ID/hash pair is inconsistent")
        if self.semantic_audit_id != f"semantic_audit_{self.semantic_audit_hash[:32]}":
            raise SemanticSpanVectorError("semantic audit ID/hash pair is inconsistent")
        if self.semantic_span_plan_id != (
            f"semantic_span_plan_{self.semantic_span_plan_hash[:32]}"
        ):
            raise SemanticSpanVectorError("SemanticSpan plan ID/hash pair is inconsistent")
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= _MAX_DIMENSIONS:
            raise SemanticSpanVectorError("vector dimensions are out of bounds")
        if self.normalization != SEMANTIC_VECTOR_NORMALIZATION:
            raise SemanticSpanVectorError("unsupported semantic vector normalization")
        if type(self.items) is not tuple:
            raise SemanticSpanVectorError("vector manifest items must be an exact tuple")
        if any(type(item) is not SemanticSpanVectorManifestItem for item in self.items):
            raise SemanticSpanVectorError("vector manifest items must use the exact contract")
        ordered = tuple(sorted(self.items, key=_manifest_sort_key))
        object.__setattr__(self, "items", ordered)
        span_ids = tuple(item.semantic_span_id for item in ordered)
        parent_ordinals = tuple((item.parent_fragment_id, item.span_ordinal) for item in ordered)
        if len(set(span_ids)) != len(span_ids):
            raise SemanticSpanVectorError("vector manifest SemanticSpan IDs must be unique")
        if len(set(parent_ordinals)) != len(parent_ordinals):
            raise SemanticSpanVectorError("vector manifest parent/span ordinals must be unique")
        ordinals_by_parent: dict[str, list[int]] = {}
        for item in ordered:
            ordinals_by_parent.setdefault(item.parent_fragment_id, []).append(item.span_ordinal)
        if any(
            tuple(ordinals) != tuple(range(len(ordinals)))
            for ordinals in ordinals_by_parent.values()
        ):
            raise SemanticSpanVectorError(
                "vector manifest span ordinals must be contiguous from zero per parent"
            )

    @property
    def span_manifest_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": self.MANIFEST_SCHEMA,
                "items": [item.payload() for item in self.items],
            }
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "access_policy_id": self.access_policy_id,
            "policy_hash": self.policy_hash,
            "exclusion_hash": self.exclusion_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "semantic_audit_id": self.semantic_audit_id,
            "semantic_audit_hash": self.semantic_audit_hash,
            "semantic_audit_binding_hash": self.semantic_audit_binding_hash,
            "semantic_audit_fragment_manifest_hash": (self.semantic_audit_fragment_manifest_hash),
            "semantic_span_plan_id": self.semantic_span_plan_id,
            "semantic_span_plan_hash": self.semantic_span_plan_hash,
            "semantic_span_profile_hash": self.semantic_span_profile_hash,
            "semantic_span_parent_manifest_hash": self.semantic_span_parent_manifest_hash,
            "semantic_span_closure_hash": self.semantic_span_closure_hash,
            "model_profile_hash": self.model_profile_hash,
            "runtime_profile_hash": self.runtime_profile_hash,
            "provisioning_receipt_hash": self.provisioning_receipt_hash,
            "dimensions": self.dimensions,
            "normalization": self.normalization,
            "span_manifest_hash": self.span_manifest_hash,
            "counts": {
                "parents": len({item.parent_fragment_id for item in self.items}),
                "spans": len(self.items),
            },
            "items": [item.payload() for item in self.items],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def generation_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    @property
    def generation_id(self) -> str:
        return canonical_content_id("semantic_span_vector_generation", self.payload())


@dataclass(frozen=True, slots=True)
class SemanticSpanDenseVectorItem:
    """One materialized SemanticSpan vector; no source text is retained."""

    semantic_span_id: str
    semantic_span_hash: str
    parent_fragment_id: str
    span_ordinal: int
    span_text_sha256: str
    vector: PackedVector

    def __post_init__(self) -> None:
        if type(self.vector) is not PackedVector:
            raise SemanticSpanVectorError("dense item vector must be an exact PackedVector")
        manifest = SemanticSpanVectorManifestItem(
            semantic_span_id=self.semantic_span_id,
            semantic_span_hash=self.semantic_span_hash,
            parent_fragment_id=self.parent_fragment_id,
            span_ordinal=self.span_ordinal,
            span_text_sha256=self.span_text_sha256,
            vector_sha256=self.vector.vector_hash,
        )
        del manifest
        _ = self.vector.values
        if sha256_hex(self.vector.blob) != self.vector.vector_hash:
            raise SemanticSpanVectorError("dense item vector hash differs from its bytes")

    def manifest_item(self) -> SemanticSpanVectorManifestItem:
        return SemanticSpanVectorManifestItem(
            semantic_span_id=self.semantic_span_id,
            semantic_span_hash=self.semantic_span_hash,
            parent_fragment_id=self.parent_fragment_id,
            span_ordinal=self.span_ordinal,
            span_text_sha256=self.span_text_sha256,
            vector_sha256=self.vector.vector_hash,
        )


@dataclass(frozen=True, slots=True)
class GroupedSemanticDenseHit:
    """One parent result represented by its best real SemanticSpan."""

    parent_fragment_id: str
    winning_semantic_span_id: str
    winning_span_ordinal: int
    rank: int
    score_hex: str

    def __post_init__(self) -> None:
        _require_id(self.parent_fragment_id, "fragment")
        _require_id(self.winning_semantic_span_id, "semantic_span")
        if type(self.winning_span_ordinal) is not int or self.winning_span_ordinal < 0:
            raise SemanticSpanVectorError("winning_span_ordinal must be a non-negative integer")
        if type(self.rank) is not int or self.rank < 1:
            raise SemanticSpanVectorError("dense hit rank must be a positive integer")
        score = _parse_score(self.score_hex)
        if not -1.0 <= score <= 1.0:
            raise SemanticSpanVectorError("dense score is outside the cosine range")

    @property
    def score(self) -> float:
        return _parse_score(self.score_hex)

    def payload(self) -> dict[str, object]:
        return {
            "parent_fragment_id": self.parent_fragment_id,
            "winning_semantic_span_id": self.winning_semantic_span_id,
            "winning_span_ordinal": self.winning_span_ordinal,
            "rank": self.rank,
            "score_hex": self.score_hex,
        }


@dataclass(frozen=True, slots=True)
class SemanticSpanDenseReceipt:
    """Content-addressed proof of an exact max-over-spans parent scan."""

    SCHEMA: ClassVar[str] = "dithyramba.semantic_span_dense_receipt/1.0"

    vector_generation_id: str
    vector_generation_hash: str
    span_manifest_hash: str
    query_vector_sha256: str
    query_model_profile_hash: str
    query_runtime_profile_hash: str
    query_provisioning_receipt_hash: str
    dimensions: int
    requested_parent_limit: int
    scanned_span_count: int
    parent_candidate_count: int
    hits: tuple[GroupedSemanticDenseHit, ...]
    normalization: SemanticVectorNormalization = SEMANTIC_VECTOR_NORMALIZATION
    grouping_algorithm: Literal["max_parent_span_v1"] = "max_parent_span_v1"

    def __post_init__(self) -> None:
        _require_id(self.vector_generation_id, "semantic_span_vector_generation")
        for value in (
            self.vector_generation_hash,
            self.span_manifest_hash,
            self.query_vector_sha256,
            self.query_model_profile_hash,
            self.query_runtime_profile_hash,
            self.query_provisioning_receipt_hash,
        ):
            _require_hash(value)
        if self.vector_generation_id != (
            f"semantic_span_vector_generation_{self.vector_generation_hash[:32]}"
        ):
            raise SemanticSpanVectorError("vector generation ID/hash pair is inconsistent")
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= _MAX_DIMENSIONS:
            raise SemanticSpanVectorError("receipt dimensions are out of bounds")
        if (
            type(self.requested_parent_limit) is not int
            or not 1 <= self.requested_parent_limit <= 500
        ):
            raise SemanticSpanVectorError("parent limit must be between 1 and 500")
        for numeric_value, label in (
            (self.scanned_span_count, "scanned_span_count"),
            (self.parent_candidate_count, "parent_candidate_count"),
        ):
            if type(numeric_value) is not int or numeric_value < 0:
                raise SemanticSpanVectorError(f"{label} must be a non-negative integer")
        if self.parent_candidate_count > self.scanned_span_count:
            raise SemanticSpanVectorError("parent count cannot exceed scanned span count")
        if (self.scanned_span_count == 0) != (self.parent_candidate_count == 0):
            raise SemanticSpanVectorError("empty span and parent candidate closures must agree")
        if self.normalization != SEMANTIC_VECTOR_NORMALIZATION:
            raise SemanticSpanVectorError("unsupported semantic vector normalization")
        if self.grouping_algorithm != "max_parent_span_v1":
            raise SemanticSpanVectorError("unsupported SemanticSpan grouping algorithm")
        if type(self.hits) is not tuple or any(
            type(hit) is not GroupedSemanticDenseHit for hit in self.hits
        ):
            raise SemanticSpanVectorError("dense receipt hits must use the exact contract")
        expected_hit_count = min(self.requested_parent_limit, self.parent_candidate_count)
        if len(self.hits) != expected_hit_count:
            raise SemanticSpanVectorError("dense receipt must contain the exact bounded parent set")
        if tuple(hit.rank for hit in self.hits) != tuple(range(1, len(self.hits) + 1)):
            raise SemanticSpanVectorError("dense receipt ranks must be contiguous from one")
        parent_ids = tuple(hit.parent_fragment_id for hit in self.hits)
        if len(set(parent_ids)) != len(parent_ids):
            raise SemanticSpanVectorError("dense receipt parent IDs must be unique")
        winning_span_ids = tuple(hit.winning_semantic_span_id for hit in self.hits)
        if len(set(winning_span_ids)) != len(winning_span_ids):
            raise SemanticSpanVectorError("dense receipt winning SemanticSpan IDs must be unique")
        expected = tuple(
            sorted(
                self.hits,
                key=lambda hit: (-hit.score, hit.parent_fragment_id.encode("ascii")),
            )
        )
        if self.hits != expected:
            raise SemanticSpanVectorError("dense receipt hits are not canonically ranked")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "vector_generation_id": self.vector_generation_id,
            "vector_generation_hash": self.vector_generation_hash,
            "span_manifest_hash": self.span_manifest_hash,
            "query_vector_sha256": self.query_vector_sha256,
            "query_model_profile_hash": self.query_model_profile_hash,
            "query_runtime_profile_hash": self.query_runtime_profile_hash,
            "query_provisioning_receipt_hash": self.query_provisioning_receipt_hash,
            "dimensions": self.dimensions,
            "normalization": self.normalization,
            "grouping_algorithm": self.grouping_algorithm,
            "requested_parent_limit": self.requested_parent_limit,
            "scanned_span_count": self.scanned_span_count,
            "parent_candidate_count": self.parent_candidate_count,
            "hits": [hit.payload() for hit in self.hits],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    @property
    def receipt_id(self) -> str:
        return canonical_content_id("semantic_span_dense", self.payload())


def materialize_semantic_span_vectors(
    fragments: tuple[SourceFragmentText, ...],
    *,
    audit: SemanticCoverageAudit,
    plan: SemanticSpanPlan,
    provider: SemanticSpanEmbeddingProvider,
) -> tuple[SemanticSpanVectorGeneration, tuple[SemanticSpanDenseVectorItem, ...]]:
    """Embed the exact authorized SemanticSpan closure after lossless preflight."""

    ordered_fragments, model, runtime, provisioning = _validate_materialization_inputs(
        fragments,
        audit=audit,
        plan=plan,
        provider=provider,
    )
    text_by_parent = {fragment.source_fragment_id: fragment.text for fragment in ordered_fragments}
    ordered_spans = tuple(sorted(plan.spans, key=_span_sort_key))
    span_texts: list[str] = []
    counter = provider.count_prepared_tokens_no_truncation
    for span in ordered_spans:
        parent_text = text_by_parent[span.parent_fragment_id]
        span_text = parent_text[span.char_start : span.char_end]
        if sha256_hex(span_text.encode("utf-8")) != span.span_text_sha256:
            raise SemanticSpanVectorError("SemanticSpan text slice differs from its committed hash")
        try:
            observed_count = counter(f"{model.passage_prefix}{span_text}")
        except Exception as exc:
            raise SemanticSpanVectorError(
                "exact no-truncation SemanticSpan preflight failed"
            ) from exc
        if type(observed_count) is not int or observed_count < 1:
            raise SemanticSpanVectorError("SemanticSpan preflight returned an invalid token count")
        if observed_count > model.max_tokens:
            raise SemanticSpanVectorError("SemanticSpan exceeds the embedding model token limit")
        if observed_count != span.prepared_token_count:
            raise SemanticSpanVectorError("SemanticSpan prepared-token count drifted")
        span_texts.append(span_text)

    vectors = _embed_all(provider, tuple(span_texts), dimensions=model.dimensions)
    dense_items = tuple(
        SemanticSpanDenseVectorItem(
            semantic_span_id=span.span_id,
            semantic_span_hash=span.span_hash,
            parent_fragment_id=span.parent_fragment_id,
            span_ordinal=span.span_ordinal,
            span_text_sha256=span.span_text_sha256,
            vector=vector,
        )
        for span, vector in zip(ordered_spans, vectors, strict=True)
    )
    generation = SemanticSpanVectorGeneration(
        library_id=audit.binding.library_id,
        corpus_snapshot_id=audit.binding.corpus_snapshot_id,
        snapshot_hash=audit.binding.snapshot_hash,
        access_policy_id=audit.binding.access_policy_id,
        policy_hash=audit.binding.policy_hash,
        exclusion_hash=audit.binding.exclusion_hash,
        permitted_set_hash=audit.binding.permitted_set_hash,
        semantic_audit_id=audit.audit_id,
        semantic_audit_hash=audit.audit_hash,
        semantic_audit_binding_hash=audit.binding.binding_hash,
        semantic_audit_fragment_manifest_hash=audit.fragment_manifest_hash,
        semantic_span_plan_id=plan.plan_id,
        semantic_span_plan_hash=plan.plan_hash,
        semantic_span_profile_hash=plan.profile.profile_hash,
        semantic_span_parent_manifest_hash=plan.parent_manifest_hash,
        semantic_span_closure_hash=plan.span_closure_hash,
        model_profile_hash=model.profile_hash,
        runtime_profile_hash=runtime.profile_hash,
        provisioning_receipt_hash=provisioning.receipt_hash,
        dimensions=model.dimensions,
        items=tuple(item.manifest_item() for item in dense_items),
    )
    return generation, dense_items


def exact_grouped_semantic_span_scan(
    query: PackedVector,
    candidates: tuple[SemanticSpanDenseVectorItem, ...],
    *,
    generation: SemanticSpanVectorGeneration,
    query_model_profile_hash: str,
    query_runtime_profile_hash: str,
    query_provisioning_receipt_hash: str,
    limit: int,
    normalization: SemanticVectorNormalization = SEMANTIC_VECTOR_NORMALIZATION,
) -> SemanticSpanDenseReceipt:
    """Score every real span, then retain the best real span per parent."""

    if type(query) is not PackedVector:
        raise SemanticSpanVectorError("query must be an exact PackedVector")
    if type(generation) is not SemanticSpanVectorGeneration:
        raise SemanticSpanVectorError("generation must use the exact vector-v2 contract")
    if type(candidates) is not tuple or any(
        type(candidate) is not SemanticSpanDenseVectorItem for candidate in candidates
    ):
        raise SemanticSpanVectorError("scan candidates must use the exact dense item contract")
    for value in (
        query_model_profile_hash,
        query_runtime_profile_hash,
        query_provisioning_receipt_hash,
    ):
        _require_hash(value)
    if (
        query_model_profile_hash != generation.model_profile_hash
        or query_runtime_profile_hash != generation.runtime_profile_hash
        or query_provisioning_receipt_hash != generation.provisioning_receipt_hash
    ):
        raise SemanticSpanVectorError("query model/runtime/provisioning binding differs")
    if normalization != generation.normalization or normalization != SEMANTIC_VECTOR_NORMALIZATION:
        raise SemanticSpanVectorError("query vector normalization differs from the generation")
    if query.dimensions != generation.dimensions:
        raise SemanticSpanVectorError("query dimensions differ from the vector generation")
    if sha256_hex(query.blob) != query.vector_hash:
        raise SemanticSpanVectorError("query vector hash differs from its bytes")
    if type(limit) is not int or not 1 <= limit <= 500:
        raise SemanticSpanVectorError("parent limit must be between 1 and 500")
    ordered = _validate_candidate_closure(candidates, generation=generation)
    query_values = query.values
    best_by_parent: dict[str, tuple[float, SemanticSpanDenseVectorItem]] = {}
    for candidate in ordered:
        if candidate.vector.dimensions != query.dimensions:
            raise SemanticSpanVectorError("candidate dimensions differ from the query")
        score = math.fsum(
            left * right for left, right in zip(query_values, candidate.vector.values, strict=True)
        )
        if not math.isfinite(score) or not -1.000001 <= score <= 1.000001:
            raise SemanticSpanVectorError("exact cosine score is outside the normalized range")
        score = max(-1.0, min(1.0, score))
        incumbent = best_by_parent.get(candidate.parent_fragment_id)
        if incumbent is None or _span_winner_key(score, candidate) < _span_winner_key(*incumbent):
            best_by_parent[candidate.parent_fragment_id] = (score, candidate)
    ranked = sorted(
        best_by_parent.values(),
        key=lambda item: (-item[0], item[1].parent_fragment_id.encode("ascii")),
    )
    hits = tuple(
        GroupedSemanticDenseHit(
            parent_fragment_id=candidate.parent_fragment_id,
            winning_semantic_span_id=candidate.semantic_span_id,
            winning_span_ordinal=candidate.span_ordinal,
            rank=rank,
            score_hex=score.hex(),
        )
        for rank, (score, candidate) in enumerate(ranked[:limit], start=1)
    )
    return SemanticSpanDenseReceipt(
        vector_generation_id=generation.generation_id,
        vector_generation_hash=generation.generation_hash,
        span_manifest_hash=generation.span_manifest_hash,
        query_vector_sha256=query.vector_hash,
        query_model_profile_hash=query_model_profile_hash,
        query_runtime_profile_hash=query_runtime_profile_hash,
        query_provisioning_receipt_hash=query_provisioning_receipt_hash,
        dimensions=query.dimensions,
        requested_parent_limit=limit,
        scanned_span_count=len(ordered),
        parent_candidate_count=len(best_by_parent),
        hits=hits,
        normalization=normalization,
    )


def _validate_materialization_inputs(
    fragments: tuple[SourceFragmentText, ...],
    *,
    audit: SemanticCoverageAudit,
    plan: SemanticSpanPlan,
    provider: SemanticSpanEmbeddingProvider,
) -> tuple[
    tuple[SourceFragmentText, ...],
    EmbeddingModelProfile,
    ModelRuntimeProfile,
    ModelProvisioningReceipt,
]:
    if type(audit) is not SemanticCoverageAudit:
        raise SemanticSpanVectorError("audit must use the exact SemanticCoverageAudit contract")
    if type(plan) is not SemanticSpanPlan:
        raise SemanticSpanVectorError("plan must use the exact SemanticSpanPlan contract")
    if type(fragments) is not tuple or any(
        type(fragment) is not SourceFragmentText for fragment in fragments
    ):
        raise SemanticSpanVectorError("fragments must be exact SourceFragmentText values")
    ordered = tuple(sorted(fragments, key=lambda item: item.source_fragment_id.encode("ascii")))
    if len({fragment.source_fragment_id for fragment in ordered}) != len(ordered):
        raise SemanticSpanVectorError("authorized fragment IDs must be unique")
    for fragment in ordered:
        _validate_source_fragment(fragment)
    try:
        model = provider.model_profile
        runtime = provider.runtime_profile
        provisioning = provider.provisioning_receipt
        counter = provider.count_prepared_tokens_no_truncation
        embedder = provider.embed_passages
    except Exception as exc:
        raise SemanticSpanVectorError("semantic provider bindings are unavailable") from exc
    if (
        type(model) is not EmbeddingModelProfile
        or type(runtime) is not ModelRuntimeProfile
        or type(provisioning) is not ModelProvisioningReceipt
        or not callable(counter)
        or not callable(embedder)
    ):
        raise SemanticSpanVectorError("semantic provider must expose exact callable contracts")
    if (
        provisioning.embedding_profile_hash != model.profile_hash
        or provisioning.embedding_profile_id != model.profile_id
        or provisioning.model_id != model.model_id
        or provisioning.revision != model.revision
    ):
        raise SemanticSpanVectorError("provider provisioning differs from its model profile")
    binding = audit.binding
    if (
        binding.model_profile_hash != model.profile_hash
        or binding.runtime_profile_hash != runtime.profile_hash
        or binding.provisioning_receipt_hash != provisioning.receipt_hash
        or binding.max_tokens != model.max_tokens
        or binding.passage_prefix != model.passage_prefix
    ):
        raise SemanticSpanVectorError("semantic audit differs from the provider binding")
    profile = plan.profile
    if (
        profile.model_profile_hash != model.profile_hash
        or profile.runtime_profile_hash != runtime.profile_hash
        or profile.provisioning_receipt_hash != provisioning.receipt_hash
        or profile.max_prepared_tokens != model.max_tokens
        or profile.passage_prefix != model.passage_prefix
    ):
        raise SemanticSpanVectorError("SemanticSpan plan differs from the provider binding")
    _validate_parent_closure(ordered, audit=audit, plan=plan)
    return ordered, model, runtime, provisioning


def _validate_parent_closure(
    fragments: tuple[SourceFragmentText, ...],
    *,
    audit: SemanticCoverageAudit,
    plan: SemanticSpanPlan,
) -> None:
    fragment_identity = tuple(_fragment_identity(fragment) for fragment in fragments)
    audit_identity = tuple(
        (
            item.source_fragment_id,
            item.source_version_id,
            item.source_id,
            item.ordinal,
            item.text_sha256,
        )
        for item in audit.items
    )
    parent_identity = tuple(
        (
            parent.source_fragment_id,
            parent.source_version_id,
            parent.source_id,
            parent.ordinal,
            parent.text_sha256,
        )
        for parent in plan.parents
    )
    if fragment_identity != audit_identity or fragment_identity != parent_identity:
        raise SemanticSpanVectorError("audit, plan, and authorized parent closure differ")
    fragment_by_id = {fragment.source_fragment_id: fragment for fragment in fragments}
    for audit_item, parent in zip(audit.items, plan.parents, strict=True):
        if audit_item.status is SemanticAuditItemStatus.FAILED:
            raise SemanticSpanVectorError("failed semantic audit cannot authorize vectors")
        if audit_item.prepared_token_count != parent.full_prepared_token_count:
            raise SemanticSpanVectorError("semantic audit and parent token counts differ")
        fragment = fragment_by_id[parent.source_fragment_id]
        if parent.text_char_count != len(fragment.text):
            raise SemanticSpanVectorError("SemanticSpan parent character count differs")
        for span in parent.spans:
            _validate_span_against_parent(span, fragment)


def _validate_source_fragment(fragment: SourceFragmentText) -> None:
    _require_id(fragment.source_fragment_id, "fragment")
    _require_id(fragment.source_version_id, "source_version")
    _require_id(fragment.source_id, "source")
    if type(fragment.ordinal) is not int or fragment.ordinal < 0:
        raise SemanticSpanVectorError("fragment ordinal must be a non-negative integer")
    if type(fragment.text) is not str or not fragment.text:
        raise SemanticSpanVectorError("fragment text must be a non-empty exact string")
    _require_hash(fragment.text_sha256)
    try:
        actual_hash = sha256_hex(fragment.text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise SemanticSpanVectorError("fragment text must be valid UTF-8") from exc
    if actual_hash != fragment.text_sha256:
        raise SemanticSpanVectorError("fragment text differs from text_sha256")


def _validate_span_against_parent(span: SemanticSpan, fragment: SourceFragmentText) -> None:
    if type(span) is not SemanticSpan:
        raise SemanticSpanVectorError("plan spans must use the exact SemanticSpan contract")
    if (
        span.parent_fragment_id != fragment.source_fragment_id
        or span.parent_text_sha256 != fragment.text_sha256
        or span.parent_ordinal != fragment.ordinal
        or span.char_end > len(fragment.text)
    ):
        raise SemanticSpanVectorError("SemanticSpan parent binding differs")
    span_text = fragment.text[span.char_start : span.char_end]
    if not span_text or sha256_hex(span_text.encode("utf-8")) != span.span_text_sha256:
        raise SemanticSpanVectorError("SemanticSpan slice differs from its committed hash")


def _embed_all(
    provider: SemanticSpanEmbeddingProvider,
    texts: tuple[str, ...],
    *,
    dimensions: int,
) -> tuple[PackedVector, ...]:
    if not texts:
        return ()
    results: list[PackedVector] = []
    cursor = 0
    while cursor < len(texts):
        end = cursor
        character_count = 0
        while end < len(texts) and end - cursor < _MAX_EMBED_BATCH_ITEMS:
            next_count = len(texts[end])
            if end > cursor and character_count + next_count > _MAX_EMBED_BATCH_CHARACTERS:
                break
            character_count += next_count
            end += 1
        batch = texts[cursor:end]
        try:
            output = provider.embed_passages(batch)
        except Exception as exc:
            raise SemanticSpanVectorError("SemanticSpan embedding provider failed") from exc
        if type(output) is not tuple or len(output) != len(batch):
            raise SemanticSpanVectorError("embedding output cardinality differs from its batch")
        for vector in output:
            if type(vector) is not PackedVector:
                raise SemanticSpanVectorError("embedding output must use exact PackedVector values")
            if vector.dimensions != dimensions:
                raise SemanticSpanVectorError("embedding output dimensions differ from the profile")
            # Access revalidates persisted bytes, finiteness and unit normalization.
            _ = vector.values
            if sha256_hex(vector.blob) != vector.vector_hash:
                raise SemanticSpanVectorError("embedding vector hash differs from its bytes")
            results.append(vector)
        cursor = end
    return tuple(results)


def _validate_candidate_closure(
    candidates: tuple[SemanticSpanDenseVectorItem, ...],
    *,
    generation: SemanticSpanVectorGeneration,
) -> tuple[SemanticSpanDenseVectorItem, ...]:
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.parent_fragment_id.encode("ascii"),
                item.span_ordinal,
                item.semantic_span_id.encode("ascii"),
            ),
        )
    )
    span_ids = tuple(candidate.semantic_span_id for candidate in ordered)
    if len(set(span_ids)) != len(span_ids):
        raise SemanticSpanVectorError("scan candidate SemanticSpan IDs must be unique")
    manifest = {item.semantic_span_id: item for item in generation.items}
    if set(span_ids) != set(manifest):
        raise SemanticSpanVectorError("scan candidates must equal the complete span manifest")
    for candidate in ordered:
        expected = manifest[candidate.semantic_span_id]
        if candidate.manifest_item() != expected:
            raise SemanticSpanVectorError("scan candidate differs from the generation manifest")
        if candidate.vector.dimensions != generation.dimensions:
            raise SemanticSpanVectorError("scan candidate dimensions differ from the generation")
        if sha256_hex(candidate.vector.blob) != candidate.vector.vector_hash:
            raise SemanticSpanVectorError("scan candidate vector hash differs from its bytes")
    return ordered


def _span_winner_key(
    score: float,
    candidate: SemanticSpanDenseVectorItem,
) -> tuple[float, int, bytes]:
    return (-score, candidate.span_ordinal, candidate.semantic_span_id.encode("ascii"))


def _span_sort_key(span: SemanticSpan) -> tuple[bytes, int, bytes]:
    return (
        span.parent_fragment_id.encode("ascii"),
        span.span_ordinal,
        span.span_id.encode("ascii"),
    )


def _manifest_sort_key(item: SemanticSpanVectorManifestItem) -> tuple[bytes, int, bytes]:
    return (
        item.parent_fragment_id.encode("ascii"),
        item.span_ordinal,
        item.semantic_span_id.encode("ascii"),
    )


def _fragment_identity(fragment: SourceFragmentText) -> tuple[str, str, str, int, str]:
    return (
        fragment.source_fragment_id,
        fragment.source_version_id,
        fragment.source_id,
        fragment.ordinal,
        fragment.text_sha256,
    )


def _parse_score(value: object) -> float:
    if type(value) is not str:
        raise SemanticSpanVectorError("dense score must be an exact float.hex string")
    try:
        score = float.fromhex(value)
    except ValueError as exc:
        raise SemanticSpanVectorError("dense score must be an exact float.hex string") from exc
    if not math.isfinite(score):
        raise SemanticSpanVectorError("dense score must be finite")
    if score.hex() != value:
        raise SemanticSpanVectorError("dense score must use canonical float.hex spelling")
    return score


def _require_hash(value: object) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise SemanticSpanVectorError("artifact hash must be lowercase SHA-256 hex")
    return value


def _require_id(value: object, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise SemanticSpanVectorError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise SemanticSpanVectorError(f"identifier must use a canonical {expected} suffix")
    return value
