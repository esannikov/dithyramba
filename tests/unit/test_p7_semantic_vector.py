"""SemanticSpan vector-v2 closure, materialization, and grouped scan tests."""

from __future__ import annotations

import math
import re
import struct
from copy import deepcopy
from dataclasses import fields, replace
from typing import Any, cast

import pytest

import dithyramba.recall.semantic_vector as semantic_vector
from dithyramba.contracts import sha256_hex
from dithyramba.persistence.models import SourceFragmentText
from dithyramba.recall.hybrid_models import EmbeddingModelProfile, ModelRuntimeProfile
from dithyramba.recall.provisioning import ModelProvisioningReceipt, ProvisionedModelFile
from dithyramba.recall.semantic_audit import (
    SemanticAuditBinding,
    SemanticAuditFailureReason,
    SemanticAuditItemStatus,
    SemanticCoverageAudit,
    audit_semantic_coverage,
)
from dithyramba.recall.semantic_spans import (
    ContentTokenOffset,
    SemanticSpan,
    SemanticSpanPlan,
    SemanticSpanProfile,
    plan_semantic_spans,
)
from dithyramba.recall.semantic_vector import (
    SEMANTIC_VECTOR_NORMALIZATION,
    SemanticSpanDenseReceipt,
    SemanticSpanDenseVectorItem,
    SemanticSpanVectorError,
    SemanticSpanVectorGeneration,
    SemanticSpanVectorManifestItem,
    exact_grouped_semantic_span_scan,
    materialize_semantic_span_vectors,
)
from dithyramba.recall.vector import PackedVector, VectorContractError, pack_normalized_vector

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
SNAPSHOT_HASH = "1" * 64
POLICY_HASH = "2" * 64
EXCLUSION_HASH = "3" * 64
PERMITTED_HASH = "4" * 64


def _model(*, dimensions: int = 2, max_tokens: int = 6) -> EmbeddingModelProfile:
    return EmbeddingModelProfile(
        model_id="example/semantic-test",
        revision="a" * 40,
        license="MIT",
        dimensions=dimensions,
        max_tokens=max_tokens,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )


def _runtime() -> ModelRuntimeProfile:
    return ModelRuntimeProfile(
        sentence_transformers_version="5.6.0",
        transformers_version="4.0.0",
        torch_version="2.0.0",
        device="cpu",
        dtype="float32",
        batch_size=16,
        local_files_only=True,
        trust_remote_code=False,
    )


def _receipt(model: EmbeddingModelProfile) -> ModelProvisioningReceipt:
    return ModelProvisioningReceipt(
        embedding_profile_id=model.profile_id,
        embedding_profile_hash=model.profile_hash,
        model_id=model.model_id,
        revision=model.revision,
        files=(
            ProvisionedModelFile(
                relative_path="model.safetensors",
                size_bytes=1,
                sha256=HASH_A,
            ),
        ),
    )


def _fragment(
    fragment_id: str,
    text: str,
    *,
    ordinal: int = 0,
    version: str = "source_version_one",
    source: str = "source_one",
) -> SourceFragmentText:
    return SourceFragmentText(
        source_fragment_id=fragment_id,
        source_version_id=version,
        source_id=source,
        ordinal=ordinal,
        fragment_kind="paragraph",
        text=text,
        text_sha256=sha256_hex(text.encode("utf-8")),
        source_address_json="{}",
    )


class LocalProvider:
    def __init__(
        self,
        *,
        model: EmbeddingModelProfile | None = None,
        runtime: ModelRuntimeProfile | None = None,
        receipt: ModelProvisioningReceipt | None = None,
    ) -> None:
        self.model_profile = model or _model()
        self.runtime_profile = runtime or _runtime()
        self.provisioning_receipt = receipt or _receipt(self.model_profile)
        self.count_bias = 0
        self.count_override: int | None = None
        self.output_override: object | None = None
        self.count_inputs: list[str] = []
        self.embed_batches: list[tuple[str, ...]] = []

    def content_token_offsets_no_special_tokens(
        self,
        text: str,
    ) -> tuple[ContentTokenOffset, ...]:
        return tuple(
            ContentTokenOffset(match.start(), match.end()) for match in re.finditer(r"\S+", text)
        )

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
        self.count_inputs.append(prepared_text)
        if self.count_override is not None:
            return self.count_override
        return len(re.findall(r"\S+", prepared_text)) + 1 + self.count_bias

    def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
        self.embed_batches.append(texts)
        if self.output_override is not None:
            return cast(tuple[PackedVector, ...], self.output_override)
        vectors = (
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
        )
        return tuple(
            pack_normalized_vector(vectors[text.encode("utf-8")[0] % len(vectors)])
            for text in texts
        )


def _binding(provider: LocalProvider, **overrides: object) -> SemanticAuditBinding:
    values: dict[str, object] = {
        "library_id": "library_one",
        "corpus_snapshot_id": f"snapshot_{SNAPSHOT_HASH[:32]}",
        "snapshot_hash": SNAPSHOT_HASH,
        "access_policy_id": "policy_one",
        "policy_hash": POLICY_HASH,
        "exclusion_hash": EXCLUSION_HASH,
        "permitted_set_hash": PERMITTED_HASH,
        "model_profile_hash": provider.model_profile.profile_hash,
        "runtime_profile_hash": provider.runtime_profile.profile_hash,
        "provisioning_receipt_hash": provider.provisioning_receipt.receipt_hash,
        "max_tokens": provider.model_profile.max_tokens,
        "passage_prefix": provider.model_profile.passage_prefix,
    }
    values.update(overrides)
    return SemanticAuditBinding(**values)  # type: ignore[arg-type]


def _plan(
    fragments: tuple[SourceFragmentText, ...],
    provider: LocalProvider,
) -> SemanticSpanPlan:
    profile = SemanticSpanProfile(
        model_profile_hash=provider.model_profile.profile_hash,
        runtime_profile_hash=provider.runtime_profile.profile_hash,
        provisioning_receipt_hash=provider.provisioning_receipt.receipt_hash,
        max_prepared_tokens=provider.model_profile.max_tokens,
        overlap_content_tokens=1,
        passage_prefix=provider.model_profile.passage_prefix,
    )
    return plan_semantic_spans(fragments, profile=profile, tokenizer=provider)


def _case(
    fragments: tuple[SourceFragmentText, ...] | None = None,
) -> tuple[
    tuple[SourceFragmentText, ...],
    LocalProvider,
    SemanticCoverageAudit,
    SemanticSpanPlan,
]:
    values = fragments or (
        _fragment("fragment_alpha", "таємний alpha beta", ordinal=0),
        _fragment("fragment_beta", "classified one two three four five", ordinal=1),
    )
    values = tuple(sorted(values, key=lambda item: item.source_fragment_id.encode("ascii")))
    provider = LocalProvider()
    audit = audit_semantic_coverage(values, binding=_binding(provider), token_counter=provider)
    plan = _plan(values, provider)
    provider.count_inputs.clear()
    return values, provider, audit, plan


def _materialized() -> tuple[
    tuple[SourceFragmentText, ...],
    LocalProvider,
    SemanticCoverageAudit,
    SemanticSpanPlan,
    SemanticSpanVectorGeneration,
    tuple[SemanticSpanDenseVectorItem, ...],
]:
    fragments, provider, audit, plan = _case()
    generation, vectors = materialize_semantic_span_vectors(
        fragments,
        audit=audit,
        plan=plan,
        provider=provider,
    )
    return fragments, provider, audit, plan, generation, vectors


def _scan(
    generation: SemanticSpanVectorGeneration,
    vectors: tuple[SemanticSpanDenseVectorItem, ...],
    *,
    query: PackedVector | None = None,
    limit: int = 10,
) -> SemanticSpanDenseReceipt:
    return exact_grouped_semantic_span_scan(
        query or pack_normalized_vector((1.0, 0.0)),
        vectors,
        generation=generation,
        query_model_profile_hash=generation.model_profile_hash,
        query_runtime_profile_hash=generation.runtime_profile_hash,
        query_provisioning_receipt_hash=generation.provisioning_receipt_hash,
        limit=limit,
    )


def _manual_item(
    label: str,
    parent: str,
    ordinal: int,
    vector: PackedVector,
) -> SemanticSpanDenseVectorItem:
    span_hash = sha256_hex(f"span:{label}".encode())
    return SemanticSpanDenseVectorItem(
        semantic_span_id=f"semantic_span_{span_hash[:32]}",
        semantic_span_hash=span_hash,
        parent_fragment_id=parent,
        span_ordinal=ordinal,
        span_text_sha256=sha256_hex(f"text:{label}".encode()),
        vector=vector,
    )


def _with_items(
    generation: SemanticSpanVectorGeneration,
    items: tuple[SemanticSpanDenseVectorItem, ...],
) -> SemanticSpanVectorGeneration:
    return replace(generation, items=tuple(item.manifest_item() for item in items))


def _forge_vector(
    values: tuple[float, ...],
    *,
    vector_hash: str | None = None,
) -> PackedVector:
    blob = struct.pack(f"<{len(values)}f", *values)
    forged = object.__new__(PackedVector)
    object.__setattr__(forged, "dimensions", len(values))
    object.__setattr__(forged, "blob", blob)
    object.__setattr__(forged, "vector_hash", vector_hash or sha256_hex(blob))
    return forged


def test_materialization_is_complete_deterministic_content_addressed_and_text_free() -> None:
    fragments, provider, audit, plan = _case()

    first, first_vectors = materialize_semantic_span_vectors(
        tuple(reversed(fragments)), audit=audit, plan=plan, provider=provider
    )
    second, second_vectors = materialize_semantic_span_vectors(
        fragments, audit=audit, plan=plan, provider=provider
    )

    assert first == second
    assert first_vectors == second_vectors
    assert first.generation_id.startswith("semantic_span_vector_generation_")
    assert len(first.generation_hash) == 64
    assert len(first.span_manifest_hash) == 64
    assert first.semantic_audit_id == audit.audit_id
    assert first.semantic_span_plan_id == plan.plan_id
    assert first.semantic_span_closure_hash == plan.span_closure_hash
    assert len(first.items) == len(plan.spans) == len(first_vectors)
    assert tuple(item.semantic_span_id for item in first.items) == tuple(
        item.semantic_span_id for item in first_vectors
    )
    assert first.normalization == SEMANTIC_VECTOR_NORMALIZATION
    for secret in ("таємний", "classified", "one two", "four five"):
        assert secret.encode() not in first.canonical_bytes
        assert secret not in repr(first_vectors)


def test_materialization_preflights_every_exact_slice_before_embedding() -> None:
    fragments, provider, audit, plan = _case()

    generation, _vectors = materialize_semantic_span_vectors(
        fragments, audit=audit, plan=plan, provider=provider
    )

    text_by_id = {item.source_fragment_id: item.text for item in fragments}
    expected = [
        f"{provider.model_profile.passage_prefix}"
        f"{text_by_id[span.parent_fragment_id][span.char_start : span.char_end]}"
        for span in sorted(
            plan.spans,
            key=lambda item: (
                item.parent_fragment_id.encode("ascii"),
                item.span_ordinal,
                item.span_id.encode("ascii"),
            ),
        )
    ]
    assert provider.count_inputs == expected
    assert sum(len(batch) for batch in provider.embed_batches) == len(plan.spans)
    assert generation.dimensions == provider.model_profile.dimensions


def test_canonical_empty_authorized_closure_creates_no_embedding_calls_and_scans_empty() -> None:
    fragments: tuple[SourceFragmentText, ...] = ()
    provider = LocalProvider()
    audit = audit_semantic_coverage(fragments, binding=_binding(provider), token_counter=provider)
    plan = _plan(fragments, provider)

    generation, vectors = materialize_semantic_span_vectors(
        fragments, audit=audit, plan=plan, provider=provider
    )
    receipt = _scan(generation, vectors)

    assert generation.items == ()
    assert vectors == ()
    assert provider.embed_batches == []
    assert receipt.hits == ()
    assert receipt.scanned_span_count == 0
    assert receipt.parent_candidate_count == 0
    assert b"text" not in generation.canonical_bytes


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate"])
def test_materialization_rejects_incomplete_authorized_parent_closure(mode: str) -> None:
    fragments, provider, audit, plan = _case()
    extra = _fragment(
        "fragment_extra", "extra", ordinal=0, version="source_version_extra", source="source_extra"
    )
    broken = {
        "missing": fragments[:-1],
        "extra": (*fragments, extra),
        "duplicate": (fragments[0], fragments[0], fragments[1]),
    }[mode]

    with pytest.raises(SemanticSpanVectorError, match=r"closure|unique"):
        materialize_semantic_span_vectors(broken, audit=audit, plan=plan, provider=provider)


def test_materialization_rejects_wrong_parent_text_hash_and_span_slice() -> None:
    fragments, provider, audit, plan = _case()
    changed = replace(fragments[0], text="changed", text_sha256=sha256_hex(b"changed"))
    wrong_hash = replace(fragments[0], text_sha256=HASH_B)

    with pytest.raises(SemanticSpanVectorError, match="closure"):
        materialize_semantic_span_vectors(
            (changed, fragments[1]), audit=audit, plan=plan, provider=provider
        )
    with pytest.raises(SemanticSpanVectorError, match="text_sha256"):
        materialize_semantic_span_vectors(
            (wrong_hash, fragments[1]), audit=audit, plan=plan, provider=provider
        )


def test_materialization_rejects_wrong_audit_model_runtime_provisioning_and_plan() -> None:
    fragments, provider, audit, plan = _case()
    wrong_model_audit = SemanticCoverageAudit(
        binding=_binding(provider, model_profile_hash=HASH_B),
        items=audit.items,
    )
    other_runtime = provider.runtime_profile.model_copy(update={"batch_size": 17})
    runtime_provider = LocalProvider(
        model=provider.model_profile,
        runtime=other_runtime,
        receipt=provider.provisioning_receipt,
    )
    wrong_receipt_model = _model(dimensions=2, max_tokens=6)
    wrong_receipt = provider.provisioning_receipt.model_copy(
        update={"embedding_profile_hash": HASH_B},
    )
    receipt_provider = LocalProvider(
        model=wrong_receipt_model,
        runtime=provider.runtime_profile,
        receipt=wrong_receipt,
    )
    other_fragments = (fragments[0],)
    other_audit = audit_semantic_coverage(
        other_fragments,
        binding=_binding(provider),
        token_counter=provider,
    )

    with pytest.raises(SemanticSpanVectorError, match="audit differs"):
        materialize_semantic_span_vectors(
            fragments, audit=wrong_model_audit, plan=plan, provider=provider
        )
    with pytest.raises(SemanticSpanVectorError, match="audit differs"):
        materialize_semantic_span_vectors(
            fragments, audit=audit, plan=plan, provider=runtime_provider
        )
    with pytest.raises(SemanticSpanVectorError, match="provisioning differs"):
        materialize_semantic_span_vectors(
            fragments, audit=audit, plan=plan, provider=receipt_provider
        )
    with pytest.raises(SemanticSpanVectorError, match="closure"):
        materialize_semantic_span_vectors(
            fragments, audit=other_audit, plan=plan, provider=provider
        )


def test_materialization_rejects_noncontracts_and_unavailable_provider_bindings() -> None:
    fragments, provider, audit, plan = _case()

    with pytest.raises(SemanticSpanVectorError, match="exact SemanticCoverageAudit"):
        materialize_semantic_span_vectors(
            fragments, audit=cast(Any, object()), plan=plan, provider=provider
        )
    with pytest.raises(SemanticSpanVectorError, match="exact SemanticSpanPlan"):
        materialize_semantic_span_vectors(
            fragments, audit=audit, plan=cast(Any, object()), provider=provider
        )
    with pytest.raises(SemanticSpanVectorError, match="exact SourceFragmentText"):
        materialize_semantic_span_vectors(
            cast(Any, [*fragments]), audit=audit, plan=plan, provider=provider
        )
    with pytest.raises(SemanticSpanVectorError, match="bindings are unavailable"):
        materialize_semantic_span_vectors(
            fragments, audit=audit, plan=plan, provider=cast(Any, object())
        )


def test_count_drift_and_overbudget_fail_before_any_embedding() -> None:
    fragments, provider, audit, plan = _case((_fragment("fragment_one", "one two", ordinal=0),))
    committed = plan.spans[0].prepared_token_count
    provider.count_override = committed + 1
    with pytest.raises(SemanticSpanVectorError, match="drifted"):
        materialize_semantic_span_vectors(fragments, audit=audit, plan=plan, provider=provider)
    assert provider.embed_batches == []

    provider.count_override = provider.model_profile.max_tokens + 1
    with pytest.raises(SemanticSpanVectorError, match="token limit"):
        materialize_semantic_span_vectors(fragments, audit=audit, plan=plan, provider=provider)
    assert provider.embed_batches == []


@pytest.mark.parametrize(
    ("output", "match"),
    [
        ((), "cardinality"),
        ([], "cardinality"),
        ((pack_normalized_vector((1.0, 0.0, 0.0)),), "dimensions"),
        ((object(),), "PackedVector"),
        ((_forge_vector((math.nan, 0.0)),), "invalid"),
        ((_forge_vector((0.5, 0.5)),), "normalized"),
        ((_forge_vector((1.0, 0.0), vector_hash=HASH_C),), "hash"),
    ],
)
def test_provider_cardinality_dimension_normalization_finiteness_and_hash_fail_closed(
    output: object,
    match: str,
) -> None:
    fragments, provider, audit, plan = _case((_fragment("fragment_one", "one two", ordinal=0),))
    provider.output_override = output

    with pytest.raises((SemanticSpanVectorError, VectorContractError), match=match):
        materialize_semantic_span_vectors(fragments, audit=audit, plan=plan, provider=provider)


def test_embedding_provider_exception_is_source_free() -> None:
    fragments, provider, audit, plan = _case(
        (_fragment("fragment_one", "secret payload", ordinal=0),)
    )

    class ExplodingProvider(LocalProvider):
        def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
            raise RuntimeError(texts[0])

    exploding = ExplodingProvider(
        model=provider.model_profile,
        runtime=provider.runtime_profile,
        receipt=provider.provisioning_receipt,
    )
    with pytest.raises(SemanticSpanVectorError, match="provider failed") as caught:
        materialize_semantic_span_vectors(fragments, audit=audit, plan=plan, provider=exploding)
    assert "secret" not in str(caught.value)


def test_exact_grouping_scans_all_spans_before_parent_limit() -> None:
    _fragments, _provider, _audit, _plan, base_generation, _vectors = _materialized()
    query = pack_normalized_vector((1.0, 0.0))
    parent_a = tuple(_manual_item(f"a{index}", "fragment_a", index, query) for index in range(50))
    parent_b = (
        _manual_item(
            "b0",
            "fragment_b",
            0,
            pack_normalized_vector((0.8, 0.6)),
        ),
    )
    candidates = (*parent_a, *parent_b)
    generation = _with_items(base_generation, candidates)

    receipt = _scan(generation, tuple(reversed(candidates)), query=query, limit=2)

    assert receipt.scanned_span_count == 51
    assert receipt.parent_candidate_count == 2
    assert tuple(hit.parent_fragment_id for hit in receipt.hits) == (
        "fragment_a",
        "fragment_b",
    )
    assert receipt.hits[1].score == pytest.approx(0.8)


def test_grouping_ties_choose_lowest_span_ordinal_then_ascii_parent() -> None:
    _fragments, _provider, _audit, _plan, base_generation, _vectors = _materialized()
    query = pack_normalized_vector((1.0, 0.0))
    a_high_ordinal = _manual_item("a-high", "fragment_a", 1, query)
    a_low_ordinal = _manual_item("z-low", "fragment_a", 0, query)
    b = _manual_item("b", "fragment_b", 0, query)
    candidates = (b, a_high_ordinal, a_low_ordinal)
    generation = _with_items(base_generation, candidates)

    receipt = _scan(generation, candidates, query=query)

    assert tuple(hit.parent_fragment_id for hit in receipt.hits) == (
        "fragment_a",
        "fragment_b",
    )
    assert receipt.hits[0].winning_semantic_span_id == a_low_ordinal.semantic_span_id
    assert receipt.hits[0].winning_span_ordinal == 0
    assert {hit.winning_semantic_span_id for hit in receipt.hits}.issubset(
        {item.semantic_span_id for item in candidates}
    )


def test_scan_is_deterministic_across_candidate_order_and_receipted_without_text() -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()

    first = _scan(generation, vectors)
    second = _scan(generation, tuple(reversed(vectors)))

    assert first == second
    assert first.receipt_id.startswith("semantic_span_dense_")
    assert len(first.receipt_hash) == 64
    assert first.vector_generation_hash == generation.generation_hash
    assert first.span_manifest_hash == generation.span_manifest_hash
    assert first.query_vector_sha256 == pack_normalized_vector((1.0, 0.0)).vector_hash
    assert first.grouping_algorithm == "max_parent_span_v1"
    assert b"classified" not in first.canonical_bytes
    assert b"passage" not in first.canonical_bytes


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "wrong_parent", "wrong_hash"])
def test_scan_rejects_incomplete_or_modified_span_vector_closure(mode: str) -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()
    extra = _manual_item("extra", "fragment_extra", 0, pack_normalized_vector((1.0, 0.0)))
    first = vectors[0]
    changed_parent = replace(first, parent_fragment_id="fragment_changed")
    replacement_values = (0.0, 1.0) if first.vector.values == (1.0, 0.0) else (1.0, 0.0)
    changed_vector = replace(first, vector=pack_normalized_vector(replacement_values))
    broken = {
        "missing": vectors[:-1],
        "extra": (*vectors, extra),
        "duplicate": (*vectors, vectors[0]),
        "wrong_parent": (changed_parent, *vectors[1:]),
        "wrong_hash": (changed_vector, *vectors[1:]),
    }[mode]

    with pytest.raises(SemanticSpanVectorError, match=r"unique|complete|manifest"):
        _scan(generation, broken)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("query_model_profile_hash", HASH_B, "binding"),
        ("query_runtime_profile_hash", HASH_B, "binding"),
        ("query_provisioning_receipt_hash", HASH_B, "binding"),
        ("normalization", cast(Any, "raw"), "normalization"),
        ("limit", 0, "limit"),
    ],
)
def test_scan_rejects_query_binding_and_limit_mismatches(
    field: str,
    value: object,
    match: str,
) -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()
    arguments: dict[str, object] = {
        "generation": generation,
        "query_model_profile_hash": generation.model_profile_hash,
        "query_runtime_profile_hash": generation.runtime_profile_hash,
        "query_provisioning_receipt_hash": generation.provisioning_receipt_hash,
        "limit": 10,
        "normalization": SEMANTIC_VECTOR_NORMALIZATION,
    }
    arguments[field] = value

    with pytest.raises(SemanticSpanVectorError, match=match):
        exact_grouped_semantic_span_scan(
            pack_normalized_vector((1.0, 0.0)),
            vectors,
            **arguments,  # type: ignore[arg-type]
        )


def test_scan_rejects_dimension_and_exact_contract_impostors() -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()

    class DerivedDenseItem(SemanticSpanDenseVectorItem):
        pass

    class DerivedGeneration(SemanticSpanVectorGeneration):
        pass

    derived_item = DerivedDenseItem(
        semantic_span_id=vectors[0].semantic_span_id,
        semantic_span_hash=vectors[0].semantic_span_hash,
        parent_fragment_id=vectors[0].parent_fragment_id,
        span_ordinal=vectors[0].span_ordinal,
        span_text_sha256=vectors[0].span_text_sha256,
        vector=vectors[0].vector,
    )
    derived_generation = DerivedGeneration(
        **{
            field: getattr(generation, field)
            for field in generation.__dataclass_fields__
            if field not in {"SCHEMA", "MANIFEST_SCHEMA"}
        }
    )

    with pytest.raises(SemanticSpanVectorError, match="query"):
        exact_grouped_semantic_span_scan(
            cast(Any, object()),
            vectors,
            generation=generation,
            query_model_profile_hash=generation.model_profile_hash,
            query_runtime_profile_hash=generation.runtime_profile_hash,
            query_provisioning_receipt_hash=generation.provisioning_receipt_hash,
            limit=2,
        )
    with pytest.raises(SemanticSpanVectorError, match="dense item"):
        _scan(generation, (derived_item, *vectors[1:]))
    with pytest.raises(SemanticSpanVectorError, match="generation"):
        _scan(cast(Any, derived_generation), vectors)
    with pytest.raises(SemanticSpanVectorError, match="dimensions"):
        _scan(generation, vectors, query=pack_normalized_vector((1.0, 0.0, 0.0)))


def test_generation_manifest_is_sorted_validated_and_canonical_when_empty() -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()
    reversed_generation = replace(
        generation,
        items=tuple(item.manifest_item() for item in reversed(vectors)),
    )
    empty = replace(generation, items=())

    assert reversed_generation == generation
    assert empty.items == ()
    assert empty.payload()["counts"] == {"parents": 0, "spans": 0}
    with pytest.raises(SemanticSpanVectorError, match="unique"):
        replace(generation, items=(generation.items[0], generation.items[0]))
    with pytest.raises(SemanticSpanVectorError, match="parent/span"):
        replace(
            generation,
            items=(
                generation.items[0],
                replace(
                    generation.items[1],
                    parent_fragment_id=generation.items[0].parent_fragment_id,
                    span_ordinal=generation.items[0].span_ordinal,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("corpus_snapshot_id", "snapshot_wrong", "snapshot"),
        ("semantic_audit_id", "semantic_audit_wrong", "audit"),
        ("semantic_span_plan_id", "semantic_span_plan_wrong", "plan"),
        ("dimensions", 0, "dimensions"),
        ("normalization", cast(Any, "raw"), "normalization"),
        ("items", cast(Any, []), "tuple"),
    ],
)
def test_generation_rejects_identity_dimension_normalization_and_collection_errors(
    field: str,
    value: object,
    match: str,
) -> None:
    _fragments, _provider, _audit, _plan, generation, _vectors = _materialized()

    with pytest.raises(SemanticSpanVectorError, match=match):
        replace(generation, **cast(Any, {field: value}))


def test_manifest_and_dense_item_reject_invalid_identity_hash_ordinal_and_vector() -> None:
    _fragments, _provider, _audit, _plan, _generation, vectors = _materialized()
    item = vectors[0].manifest_item()

    with pytest.raises(SemanticSpanVectorError, match="ID/hash"):
        replace(item, semantic_span_id="semantic_span_wrong")
    with pytest.raises(SemanticSpanVectorError, match="span_ordinal"):
        replace(item, span_ordinal=-1)
    with pytest.raises(SemanticSpanVectorError, match="hash"):
        replace(item, vector_sha256="bad")
    with pytest.raises(SemanticSpanVectorError, match="PackedVector"):
        replace(vectors[0], vector=cast(Any, object()))


def test_hit_and_receipt_validate_canonical_score_rank_order_and_counts() -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()
    receipt = _scan(generation, vectors)
    hit = receipt.hits[0]

    assert hit.score == float.fromhex(hit.score_hex)
    with pytest.raises(SemanticSpanVectorError, match=r"float\.hex"):
        replace(hit, score_hex="0.5")
    with pytest.raises(SemanticSpanVectorError, match="finite"):
        replace(hit, score_hex="inf")
    with pytest.raises(SemanticSpanVectorError, match="cosine"):
        replace(hit, score_hex=(2.0).hex())
    with pytest.raises(SemanticSpanVectorError, match="positive"):
        replace(hit, rank=0)
    with pytest.raises(SemanticSpanVectorError, match="contiguous"):
        replace(receipt, hits=(replace(hit, rank=2), *receipt.hits[1:]))
    with pytest.raises(SemanticSpanVectorError, match="parent count"):
        replace(receipt, scanned_span_count=0)
    with pytest.raises(SemanticSpanVectorError, match=r"closure|exact bounded"):
        replace(receipt, parent_candidate_count=0)
    with pytest.raises(SemanticSpanVectorError, match="ID/hash"):
        replace(receipt, vector_generation_id="semantic_span_vector_generation_wrong")


def test_receipt_rejects_duplicate_parents_and_noncanonical_parent_order() -> None:
    _fragments, _provider, _audit, _plan, generation, _vectors = _materialized()
    query = pack_normalized_vector((1.0, 0.0))
    a = _manual_item("a", "fragment_a", 0, query)
    b = _manual_item("b", "fragment_b", 0, query)
    manual_generation = _with_items(generation, (a, b))
    receipt = _scan(manual_generation, (a, b))
    first, second = receipt.hits

    with pytest.raises(SemanticSpanVectorError, match="unique"):
        replace(receipt, hits=(first, replace(second, parent_fragment_id=first.parent_fragment_id)))
    with pytest.raises(SemanticSpanVectorError, match="winning SemanticSpan IDs"):
        replace(
            receipt,
            hits=(
                first,
                replace(second, winning_semantic_span_id=first.winning_semantic_span_id),
            ),
        )
    with pytest.raises(SemanticSpanVectorError, match="canonically ranked"):
        replace(receipt, hits=(replace(second, rank=1), replace(first, rank=2)))


def test_scan_rejects_invalid_candidate_dimensions_even_with_forged_manifest_match() -> None:
    _fragments, _provider, _audit, _plan, generation, _vectors = _materialized()
    wrong = _manual_item(
        "wrong-dim",
        "fragment_wrong",
        0,
        pack_normalized_vector((1.0, 0.0, 0.0)),
    )
    generation = _with_items(generation, (wrong,))

    with pytest.raises(SemanticSpanVectorError, match="dimensions"):
        _scan(generation, (wrong,))


def test_exact_contract_validators_reject_manifest_and_hit_subclasses_and_bad_hit_ordinal() -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()

    class DerivedManifest(SemanticSpanVectorManifestItem):
        pass

    derived = DerivedManifest(
        **{
            field.name: getattr(generation.items[0], field.name)
            for field in fields(generation.items[0])
        }
    )
    with pytest.raises(SemanticSpanVectorError, match="exact contract"):
        replace(generation, items=(derived, *generation.items[1:]))

    receipt = _scan(generation, vectors)
    with pytest.raises(SemanticSpanVectorError, match="winning_span_ordinal"):
        replace(receipt.hits[0], winning_span_ordinal=-1)
    with pytest.raises(SemanticSpanVectorError, match="exact contract"):
        replace(receipt, hits=cast(Any, [*receipt.hits]))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dimensions", 0, "dimensions"),
        ("requested_parent_limit", 0, "parent limit"),
        ("scanned_span_count", -1, "scanned_span_count"),
        ("normalization", cast(Any, "raw"), "normalization"),
        ("grouping_algorithm", cast(Any, "sum_v1"), "grouping"),
    ],
)
def test_receipt_rejects_invalid_numeric_algorithm_and_normalization_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()
    receipt = _scan(generation, vectors)
    with pytest.raises(SemanticSpanVectorError, match=match):
        replace(receipt, **cast(Any, {field: value}))


@pytest.mark.parametrize("invalid", [0, cast(Any, False)])
def test_materialization_rejects_invalid_preflight_counts(invalid: object) -> None:
    fragments, provider, audit, plan = _case((_fragment("fragment_one", "one two", ordinal=0),))
    provider.count_override = cast(int, invalid)
    with pytest.raises(SemanticSpanVectorError, match="invalid token count"):
        materialize_semantic_span_vectors(fragments, audit=audit, plan=plan, provider=provider)


def test_materialization_classifies_token_counter_exception_without_source_text() -> None:
    fragments, provider, audit, plan = _case(
        (_fragment("fragment_one", "private phrase", ordinal=0),)
    )

    class ExplodingCounter(LocalProvider):
        def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
            raise RuntimeError(prepared_text)

    exploding = ExplodingCounter(
        model=provider.model_profile,
        runtime=provider.runtime_profile,
        receipt=provider.provisioning_receipt,
    )
    with pytest.raises(SemanticSpanVectorError, match="preflight failed") as caught:
        materialize_semantic_span_vectors(fragments, audit=audit, plan=plan, provider=exploding)
    assert "private phrase" not in str(caught.value)


def test_provider_exact_contract_and_corrupt_plan_profile_fail_closed() -> None:
    fragments, provider, audit, plan = _case()

    class WrongContractProvider:
        model_profile = object()
        runtime_profile = provider.runtime_profile
        provisioning_receipt = provider.provisioning_receipt

        def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
            return len(prepared_text)

        def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
            return tuple(pack_normalized_vector((1.0, 0.0)) for _text in texts)

    with pytest.raises(SemanticSpanVectorError, match="exact callable"):
        materialize_semantic_span_vectors(
            fragments,
            audit=audit,
            plan=plan,
            provider=cast(Any, WrongContractProvider()),
        )

    corrupt_plan = deepcopy(plan)
    wrong_profile = replace(plan.profile, runtime_profile_hash=HASH_C)
    object.__setattr__(corrupt_plan, "profile", wrong_profile)
    with pytest.raises(SemanticSpanVectorError, match="plan differs"):
        materialize_semantic_span_vectors(
            fragments, audit=audit, plan=corrupt_plan, provider=provider
        )


def test_failed_audit_count_drift_and_corrupt_parent_geometry_fail_closed() -> None:
    fragments, provider, audit, plan = _case()
    failed_item = replace(
        audit.items[0],
        status=SemanticAuditItemStatus.FAILED,
        prepared_token_count=None,
        failure_reason=SemanticAuditFailureReason.TOKENIZER_ERROR,
    )
    failed_audit = SemanticCoverageAudit(
        binding=audit.binding,
        items=(failed_item, *audit.items[1:]),
    )
    with pytest.raises(SemanticSpanVectorError, match="failed semantic audit"):
        materialize_semantic_span_vectors(
            fragments, audit=failed_audit, plan=plan, provider=provider
        )

    committed_count = audit.items[0].prepared_token_count
    assert committed_count is not None
    drift_item = replace(audit.items[0], prepared_token_count=committed_count - 1)
    drift_audit = SemanticCoverageAudit(binding=audit.binding, items=(drift_item, *audit.items[1:]))
    with pytest.raises(SemanticSpanVectorError, match="parent token counts"):
        materialize_semantic_span_vectors(
            fragments, audit=drift_audit, plan=plan, provider=provider
        )

    corrupt_plan = deepcopy(plan)
    corrupt_parent = corrupt_plan.parents[0]
    object.__setattr__(corrupt_parent, "text_char_count", corrupt_parent.text_char_count + 1)
    with pytest.raises(SemanticSpanVectorError, match="character count"):
        materialize_semantic_span_vectors(
            fragments, audit=audit, plan=corrupt_plan, provider=provider
        )


@pytest.mark.parametrize(
    "fragment",
    [
        replace(_fragment("fragment_bad_ordinal", "text"), ordinal=-1),
        replace(_fragment("fragment_empty", "text"), text="", text_sha256=sha256_hex(b"")),
        replace(_fragment("fragment_surrogate", "text"), text="\ud800", text_sha256=HASH_A),
    ],
)
def test_invalid_source_fragment_primitives_fail_before_closure(
    fragment: SourceFragmentText,
) -> None:
    _fragments, provider, audit, plan = _case()
    with pytest.raises(SemanticSpanVectorError, match=r"ordinal|non-empty|UTF-8"):
        materialize_semantic_span_vectors((fragment,), audit=audit, plan=plan, provider=provider)


def _plan_with_corrupt_first_span(
    plan: SemanticSpanPlan,
    **changes: object,
) -> SemanticSpanPlan:
    corrupt_plan = deepcopy(plan)
    parent = corrupt_plan.parents[0]
    changed = replace(parent.spans[0], **cast(Any, changes))
    object.__setattr__(parent, "spans", (changed, *parent.spans[1:]))
    return corrupt_plan


def test_corrupt_span_type_parent_and_slice_bindings_fail_closed() -> None:
    fragments, provider, audit, plan = _case()

    class DerivedSpan(SemanticSpan):
        pass

    corrupt_type_plan = deepcopy(plan)
    parent = corrupt_type_plan.parents[0]
    derived = DerivedSpan(
        **{field.name: getattr(parent.spans[0], field.name) for field in fields(parent.spans[0])}
    )
    object.__setattr__(parent, "spans", (derived, *parent.spans[1:]))
    with pytest.raises(SemanticSpanVectorError, match="exact SemanticSpan"):
        materialize_semantic_span_vectors(
            fragments, audit=audit, plan=corrupt_type_plan, provider=provider
        )

    wrong_parent_plan = _plan_with_corrupt_first_span(plan, parent_fragment_id="fragment_wrong")
    with pytest.raises(SemanticSpanVectorError, match="parent binding"):
        materialize_semantic_span_vectors(
            fragments, audit=audit, plan=wrong_parent_plan, provider=provider
        )

    wrong_slice_plan = _plan_with_corrupt_first_span(plan, span_text_sha256=HASH_C)
    with pytest.raises(SemanticSpanVectorError, match="slice"):
        materialize_semantic_span_vectors(
            fragments, audit=audit, plan=wrong_slice_plan, provider=provider
        )


def test_embedding_batches_split_before_character_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    fragments, provider, audit, plan = _case()
    monkeypatch.setattr(semantic_vector, "_MAX_EMBED_BATCH_CHARACTERS", 8)

    _generation, vectors = materialize_semantic_span_vectors(
        fragments, audit=audit, plan=plan, provider=provider
    )

    assert len(vectors) > 1
    assert len(provider.embed_batches) == len(vectors)


def test_scan_rejects_cosine_roundoff_outside_normalized_tolerance() -> None:
    _fragments, _provider, _audit, _plan, generation, _vectors = _materialized()
    almost_unit = pack_normalized_vector((1.000002, 0.0))
    candidate = _manual_item("roundoff", "fragment_roundoff", 0, almost_unit)
    generation = _with_items(generation, (candidate,))

    with pytest.raises(SemanticSpanVectorError, match="cosine score"):
        _scan(generation, (candidate,), query=almost_unit)


@pytest.mark.parametrize(
    ("score", "match"),
    [
        (cast(Any, 1.0), r"float\.hex"),
        ("not-hex", r"float\.hex"),
    ],
)
def test_hit_score_rejects_non_string_and_unparseable_values(score: object, match: str) -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()
    hit = _scan(generation, vectors).hits[0]
    with pytest.raises(SemanticSpanVectorError, match=match):
        replace(hit, score_hex=cast(Any, score))


@pytest.mark.parametrize("parent_id", ["wrong", "fragment_UP"])
def test_identifier_validation_rejects_wrong_prefix_and_noncanonical_suffix(parent_id: str) -> None:
    _fragments, _provider, _audit, _plan, generation, _vectors = _materialized()
    with pytest.raises(SemanticSpanVectorError, match="identifier"):
        replace(generation.items[0], parent_fragment_id=parent_id)


def test_generation_rejects_noncontiguous_per_parent_span_ordinals() -> None:
    _fragments, _provider, _audit, _plan, generation, _vectors = _materialized()
    parent_counts: dict[str, int] = {}
    for item in generation.items:
        parent_counts[item.parent_fragment_id] = parent_counts.get(item.parent_fragment_id, 0) + 1
    target_index = next(
        index
        for index, item in enumerate(generation.items)
        if parent_counts[item.parent_fragment_id] > 1 and item.span_ordinal == 1
    )
    items = list(generation.items)
    items[target_index] = replace(items[target_index], span_ordinal=10)

    with pytest.raises(SemanticSpanVectorError, match="contiguous"):
        replace(generation, items=tuple(items))


def test_receipt_rejects_omitted_ranked_parent_and_forged_query_hash() -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()
    receipt = _scan(generation, vectors)
    assert len(receipt.hits) >= 2

    with pytest.raises(SemanticSpanVectorError, match="exact bounded"):
        replace(receipt, hits=receipt.hits[:-1])
    forged_query = _forge_vector((1.0, 0.0), vector_hash=HASH_C)
    with pytest.raises(SemanticSpanVectorError, match="query vector hash"):
        _scan(generation, vectors, query=forged_query)


def test_dense_item_revalidates_forged_vector_hash_at_contract_boundary() -> None:
    _fragments, _provider, _audit, _plan, _generation, vectors = _materialized()
    forged = _forge_vector(vectors[0].vector.values, vector_hash=HASH_C)
    with pytest.raises(SemanticSpanVectorError, match="vector hash"):
        replace(vectors[0], vector=forged)


def test_receipt_requires_empty_span_and_parent_closures_to_agree() -> None:
    _fragments, _provider, _audit, _plan, generation, vectors = _materialized()
    receipt = _scan(generation, vectors)
    with pytest.raises(SemanticSpanVectorError, match="closures must agree"):
        replace(
            receipt,
            scanned_span_count=receipt.scanned_span_count + 1,
            parent_candidate_count=0,
            hits=(),
        )
