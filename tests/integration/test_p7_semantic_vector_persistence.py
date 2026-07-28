"""SemanticSpan vector-v2 persistence, authorization, and rollback tests."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import dithyramba.persistence.repository as repository_module
from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    QueryExclusions,
    RequestScope,
)
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    AuthorizationError,
    SQLiteHybridStore,
    SQLiteSemanticSpanStore,
)
from dithyramba.persistence.errors import (
    PersistenceConflictError,
    PersistenceIntegrityError,
)
from dithyramba.persistence.models import AuthorizedRead, SourceFragmentText
from dithyramba.persistence.repository import LibraryRepository, initialize_library
from dithyramba.persistence.semantic import (
    _blob,
    _boolean,
    _integer,
    _nonnegative_integer,
    _positive_integer,
    _require_hash,
    _sqlite_boolean,
    _text,
)
from dithyramba.recall.hybrid_models import (
    CorpusLayer,
    CorpusLayerItem,
    CorpusLayerProfile,
    EmbeddingModelProfile,
    ModelRuntimeProfile,
)
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
    SemanticSpanPlan,
    SemanticSpanProfile,
    plan_semantic_spans,
)
from dithyramba.recall.semantic_vector import (
    SemanticSpanDenseVectorItem,
    SemanticSpanVectorGeneration,
    materialize_semantic_span_vectors,
)
from dithyramba.recall.vector import (
    CorpusVectorGeneration,
    DenseVectorItem,
    PackedVector,
    VectorManifestItem,
    pack_normalized_vector,
)
from dithyramba.snapshots import CorpusSnapshot

NOW = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-21T12:00:00.000000Z"
REVISION = "c" * 40
SEMANTIC_TABLES = (
    "semantic_span_profiles",
    "semantic_span_plans",
    "semantic_span_parents",
    "semantic_spans",
    "semantic_span_vector_generations",
    "semantic_span_vector_items",
)


def _clock() -> datetime:
    return NOW


@dataclass(frozen=True, slots=True)
class _Context:
    repository: LibraryRepository
    semantic: SQLiteSemanticSpanStore
    hybrid: SQLiteHybridStore
    collection_id: str
    source_id: str
    fragments: tuple[SourceFragmentText, ...]
    snapshot: CorpusSnapshot
    policy: AccessPolicySnapshot
    scope: RequestScope
    authorization: AuthorizedRead
    model: EmbeddingModelProfile
    runtime: ModelRuntimeProfile
    provider: _Provider
    audit: SemanticCoverageAudit
    plan: SemanticSpanPlan
    generation: SemanticSpanVectorGeneration
    vectors: tuple[SemanticSpanDenseVectorItem, ...]


class _Provider:
    def __init__(
        self,
        model: EmbeddingModelProfile,
        runtime: ModelRuntimeProfile,
        receipt: ModelProvisioningReceipt,
    ) -> None:
        self.model_profile = model
        self.runtime_profile = runtime
        self.provisioning_receipt = receipt

    def content_token_offsets_no_special_tokens(
        self,
        text: str,
    ) -> tuple[ContentTokenOffset, ...]:
        return tuple(
            ContentTokenOffset(match.start(), match.end()) for match in re.finditer(r"\S+", text)
        )

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
        return len(prepared_text.split()) + 2

    def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
        return tuple(
            pack_normalized_vector((1.0, 0.0))
            if "Alpha" in text
            else pack_normalized_vector((0.0, 1.0))
            for text in texts
        )


def _context(tmp_path: Path, *, suffix: str = "main") -> _Context:
    data_root = tmp_path / f"data_{suffix}"
    source_root = tmp_path / f"source_{suffix}"
    source_root.mkdir()
    repository = initialize_library(
        LibraryConfig(name=f"Semantic {suffix}"),
        data_root=data_root,
        clock=_clock,
    )
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name=f"Corpus {suffix}",
            kind=CollectionKind.CORPUS,
            roots=(build_collection_root(source_root, data_root=data_root),),
        )
    )
    source_id = f"source_{suffix}"
    _seed_source(
        repository,
        collection_id=collection.config.collection_id,
        suffix=suffix,
    )
    snapshot = repository.freeze_snapshot((collection.config.collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id=f"policy_{suffix}",
        library_id=repository.library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),),
    )
    repository.persist_access_policy(name="Research", snapshot=policy)
    scope = RequestScope(
        library_id=repository.library_id,
        snapshot_hash=snapshot.manifest_hash,
        purpose="research",
        collection_ids=(collection.config.collection_id,),
    )
    authorization = repository.authorize_read(
        access_policy_id=policy.access_policy_id,
        scope=scope,
    )
    fragments = repository.read_permitted_fragments(authorization)
    model = EmbeddingModelProfile(
        model_id="intfloat/multilingual-e5-small",
        revision=REVISION,
        license="MIT",
        dimensions=2,
        max_tokens=8,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    runtime = ModelRuntimeProfile(
        sentence_transformers_version="5.6.0",
        transformers_version="4.53.2",
        torch_version="2.7.1",
    )
    receipt = ModelProvisioningReceipt(
        embedding_profile_id=model.profile_id,
        embedding_profile_hash=model.profile_hash,
        model_id=model.model_id,
        revision=model.revision,
        files=(
            ProvisionedModelFile(
                relative_path="config.json",
                size_bytes=1,
                sha256="a" * 64,
            ),
        ),
    )
    provider = _Provider(model, runtime, receipt)
    binding = SemanticAuditBinding(
        library_id=repository.library_id,
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        snapshot_hash=snapshot.manifest_hash,
        access_policy_id=policy.access_policy_id,
        policy_hash=policy.policy_hash,
        exclusion_hash=scope.exclusion_hash,
        permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
        model_profile_hash=model.profile_hash,
        runtime_profile_hash=runtime.profile_hash,
        provisioning_receipt_hash=receipt.receipt_hash,
        max_tokens=model.max_tokens,
        passage_prefix=model.passage_prefix,
    )
    audit = audit_semantic_coverage(fragments, binding=binding, token_counter=provider)
    profile = SemanticSpanProfile(
        model_profile_hash=model.profile_hash,
        runtime_profile_hash=runtime.profile_hash,
        provisioning_receipt_hash=receipt.receipt_hash,
        max_prepared_tokens=model.max_tokens,
        overlap_content_tokens=2,
        passage_prefix=model.passage_prefix,
    )
    plan = plan_semantic_spans(fragments, profile=profile, tokenizer=provider)
    generation, vectors = materialize_semantic_span_vectors(
        fragments,
        audit=audit,
        plan=plan,
        provider=provider,
    )
    hybrid = SQLiteHybridStore(repository)
    hybrid.persist_embedding_model_profile(model)
    hybrid.persist_model_runtime_profile(runtime)
    return _Context(
        repository=repository,
        semantic=SQLiteSemanticSpanStore(repository),
        hybrid=hybrid,
        collection_id=collection.config.collection_id,
        source_id=source_id,
        fragments=fragments,
        snapshot=snapshot,
        policy=policy,
        scope=scope,
        authorization=authorization,
        model=model,
        runtime=runtime,
        provider=provider,
        audit=audit,
        plan=plan,
        generation=generation,
        vectors=vectors,
    )


def _seed_source(
    repository: LibraryRepository,
    *,
    collection_id: str,
    suffix: str,
) -> None:
    connection = repository._store.connection
    source_id = f"source_{suffix}"
    version_id = f"source_version_{suffix}"
    family_id = f"family_{suffix}"
    texts = ("Alpha evidence", "Beta evidence")
    source_bytes = "\n".join(texts).encode()
    content_hash = sha256_hex(source_bytes)
    connection.execute(
        "INSERT INTO blobs VALUES (?, ?, ?, ?)",
        (content_hash, len(source_bytes), f"{suffix}/blob", NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO sources VALUES (?, ?, ?, 'text/markdown', ?, ?)",
        (source_id, repository.library_id, f"file:///{suffix}.md", suffix, NOW_TEXT),
    )
    connection.execute(
        """
        INSERT INTO source_versions VALUES (
            ?, ?, 1, ?, ?, ?, NULL, 'markdown_v1', 'processed', NULL
        )
        """,
        (version_id, source_id, content_hash, len(source_bytes), NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_heads VALUES (?, ?, ?, NULL)",
        (source_id, version_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_families VALUES (?, ?, ?, ?)",
        (family_id, repository.library_id, suffix, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_family_members VALUES (?, ?, 'root', NULL, ?)",
        (family_id, source_id, NOW_TEXT),
    )
    for ordinal, text in enumerate(texts):
        address = {"kind": "markdown", "heading_path": [], "paragraph_index": ordinal}
        connection.execute(
            "INSERT INTO source_fragments VALUES (?, ?, ?, 'paragraph', ?, ?, ?, ?)",
            (
                f"fragment_{suffix}_{ordinal}",
                version_id,
                ordinal,
                text,
                sha256_hex(text.encode()),
                canonical_json_bytes(address).decode(),
                canonical_sha256_hex(address),
            ),
        )
    connection.execute(
        "INSERT INTO collection_memberships VALUES (?, ?, 'active', ?)",
        (collection_id, source_id, NOW_TEXT),
    )


def _close(*contexts: _Context) -> None:
    for context in contexts:
        context.repository.close()


def _generation_for(
    context: _Context,
    audit: SemanticCoverageAudit,
    plan: SemanticSpanPlan,
    vectors: tuple[SemanticSpanDenseVectorItem, ...],
    *,
    dimensions: int | None = None,
) -> SemanticSpanVectorGeneration:
    return SemanticSpanVectorGeneration(
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
        model_profile_hash=plan.profile.model_profile_hash,
        runtime_profile_hash=plan.profile.runtime_profile_hash,
        provisioning_receipt_hash=plan.profile.provisioning_receipt_hash,
        dimensions=context.model.dimensions if dimensions is None else dimensions,
        items=tuple(item.manifest_item() for item in vectors),
    )


def _counts(context: _Context) -> tuple[int, ...]:
    connection = context.repository._store.connection
    return tuple(
        int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in SEMANTIC_TABLES
    )


def test_round_trip_is_exact_idempotent_append_only_and_v1_coexists(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        expected = (context.audit, context.plan, context.generation, context.vectors)
        assert context.semantic.library_id == context.repository.library_id
        assert (
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                context.generation,
                context.vectors,
            )
            == expected
        )
        first_counts = _counts(context)
        assert first_counts == (1, 1, 2, 2, 1, 2)
        assert (
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                context.generation,
                tuple(reversed(context.vectors)),
            )
            == expected
        )
        assert _counts(context) == first_counts
        assert (
            context.semantic.load_semantic_span_vector_generation(
                context.authorization,
                context.generation.generation_hash,
            )
            == expected
        )

        _persist_v1(context)
        connection = context.repository._store.connection
        assert (
            connection.execute("SELECT COUNT(*) FROM corpus_vector_generations").fetchone()[0] == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM semantic_span_vector_generations").fetchone()[
                0
            ]
            == 1
        )
        assert all(
            "text" not in str(row[1]).lower()
            or str(row[1])
            in {
                "text_sha256",
                "parent_text_sha256",
                "span_text_sha256",
                "text_char_count",
            }
            for table in SEMANTIC_TABLES
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        for table in (*SEMANTIC_TABLES, "source_fragment_text_metrics"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f"UPDATE {table} SET rowid = rowid")
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f"DELETE FROM {table}")
    finally:
        _close(context)


def _persist_v1(context: _Context) -> None:
    layers = CorpusLayerProfile(
        library_id=context.repository.library_id,
        corpus_snapshot_id=context.snapshot.corpus_snapshot_id,
        exclusion_hash=context.scope.exclusion_hash,
        permitted_set_hash=context.authorization.compiled.manifest.permitted_set_hash,
        items=(CorpusLayerItem(source_id=context.source_id, layer=CorpusLayer.PRIMARY),),
    )
    context.hybrid.persist_corpus_layer_profile(layers)
    vectors = tuple(
        DenseVectorItem(fragment.source_fragment_id, fragment.source_id, vector.vector)
        for fragment, vector in zip(context.fragments, context.vectors, strict=True)
    )
    generation = CorpusVectorGeneration(
        library_id=context.repository.library_id,
        corpus_snapshot_id=context.snapshot.corpus_snapshot_id,
        snapshot_hash=context.snapshot.manifest_hash,
        access_policy_id=context.policy.access_policy_id,
        policy_hash=context.policy.policy_hash,
        exclusion_hash=context.scope.exclusion_hash,
        permitted_set_hash=context.authorization.compiled.manifest.permitted_set_hash,
        model_profile_hash=context.model.profile_hash,
        runtime_profile_hash=context.runtime.profile_hash,
        dimensions=context.model.dimensions,
        items=tuple(
            VectorManifestItem(
                source_fragment_id=fragment.source_fragment_id,
                text_sha256=fragment.text_sha256,
                vector_sha256=vector.vector.vector_hash,
            )
            for fragment, vector in zip(context.fragments, vectors, strict=True)
        ),
    )
    context.hybrid.persist_corpus_vector_generation(generation, vectors)


def test_authorization_is_same_object_current_and_blob_gate_stays_closed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        context.semantic.persist_semantic_span_vector_generation(
            context.authorization,
            context.audit,
            context.plan,
            context.generation,
            context.vectors,
        )
        connection = context.repository._store.connection
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("SELECT vector_blob FROM semantic_span_vector_items").fetchall()
        copied = replace(context.authorization)
        with pytest.raises(AuthorizationError, match="not issued"):
            context.semantic.load_semantic_span_vector_generation(
                copied,
                context.generation.generation_hash,
            )
        context.repository._authorizations.pop(context.authorization.authorization_id)
        with pytest.raises(AuthorizationError, match="not issued"):
            context.semantic.load_semantic_span_vector_generation(
                context.authorization,
                context.generation.generation_hash,
            )
        assert context.repository._vector_blob_read_depth == 0
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("SELECT vector_blob FROM semantic_span_vector_items").fetchall()
    finally:
        _close(context)


def test_issued_authorization_rejects_in_place_scope_and_manifest_mutation(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, suffix="issuance_snapshot")
    try:
        restricted_scope = replace(
            context.scope,
            exclusions=QueryExclusions(
                source_fragment_ids=(context.fragments[0].source_fragment_id,)
            ),
        )
        restricted = context.repository.authorize_read(
            access_policy_id=context.policy.access_policy_id,
            scope=restricted_scope,
        )
        assert len(restricted.compiled.manifest.items) < len(
            context.authorization.compiled.manifest.items
        )

        object.__setattr__(restricted, "scope", context.scope)
        object.__setattr__(restricted, "compiled", context.authorization.compiled)

        with pytest.raises(AuthorizationError, match="changed after issuance"):
            context.repository.revalidate_authorized_read(restricted)
        with pytest.raises(AuthorizationError, match="changed after issuance"):
            context.repository.read_permitted_fragments(restricted)
    finally:
        _close(context)


def test_revalidation_uses_issuance_snapshot_if_capability_changes_mid_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, suffix="issuance_race")
    try:
        restricted_scope = replace(
            context.scope,
            exclusions=QueryExclusions(
                source_fragment_ids=(context.fragments[0].source_fragment_id,)
            ),
        )
        restricted = context.repository.authorize_read(
            access_policy_id=context.policy.access_policy_id,
            scope=restricted_scope,
        )
        original_fingerprint = repository_module._authorization_fingerprint
        changed = False

        def change_after_check(authorization: AuthorizedRead) -> str:
            nonlocal changed
            fingerprint = original_fingerprint(authorization)
            if authorization is restricted and not changed:
                changed = True
                object.__setattr__(restricted, "scope", context.scope)
                object.__setattr__(restricted, "compiled", context.authorization.compiled)
            return fingerprint

        monkeypatch.setattr(
            repository_module,
            "_authorization_fingerprint",
            change_after_check,
        )

        with pytest.raises(AuthorizationError, match="changed during revalidation"):
            context.repository.revalidate_authorized_read(restricted)
        assert changed
    finally:
        _close(context)


@pytest.mark.parametrize(
    "stage",
    ("profile", "plan", "parents", "spans", "generation", "vectors", "verified"),
)
def test_injected_failure_after_each_stage_rolls_back_all_semantic_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    context = _context(tmp_path, suffix=stage)
    try:

        def fail_after(observed: str) -> None:
            if observed == stage:
                raise RuntimeError(f"forced {stage}")

        monkeypatch.setattr(context.semantic, "_after_persist_stage", fail_after)
        with pytest.raises(RuntimeError, match=f"forced {stage}"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                context.generation,
                context.vectors,
            )
        assert _counts(context) == (0, 0, 0, 0, 0, 0)
        assert context.repository._vector_blob_read_depth == 0
    finally:
        _close(context)


def test_manifest_model_access_and_nondeterministic_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        with pytest.raises(PersistenceIntegrityError, match="complete generation manifest"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                context.generation,
                context.vectors[:1],
            )
        with pytest.raises(PersistenceIntegrityError, match="unique"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                context.generation,
                (context.vectors[0], context.vectors[0]),
            )
        wrong_generation = replace(context.generation, policy_hash="b" * 64)
        with pytest.raises(PersistenceIntegrityError, match="bindings differ"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                wrong_generation,
                context.vectors,
            )
        assert _counts(context) == (0, 0, 0, 0, 0, 0)

        context.semantic.persist_semantic_span_vector_generation(
            context.authorization,
            context.audit,
            context.plan,
            context.generation,
            context.vectors,
        )
        changed_vectors = tuple(
            replace(item, vector=pack_normalized_vector((-1.0, 0.0))) for item in context.vectors
        )
        changed_manifest = tuple(item.manifest_item() for item in changed_vectors)
        conflicting = replace(context.generation, items=changed_manifest)
        with pytest.raises(PersistenceConflictError, match="nondeterministic"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                conflicting,
                changed_vectors,
            )
        assert _counts(context) == (1, 1, 2, 2, 1, 2)
    finally:
        _close(context)


def test_audit_plan_parent_and_vector_semantics_fail_before_writing(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        failed_item = replace(
            context.audit.items[0],
            status=SemanticAuditItemStatus.FAILED,
            prepared_token_count=None,
            failure_reason=SemanticAuditFailureReason.TOKENIZER_ERROR,
        )
        failed_audit = SemanticCoverageAudit(
            binding=context.audit.binding,
            items=(failed_item, *context.audit.items[1:]),
        )
        with pytest.raises(PersistenceIntegrityError, match="failed semantic audit"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                failed_audit,
                context.plan,
                _generation_for(context, failed_audit, context.plan, context.vectors),
                context.vectors,
            )

        profile_mismatch_audit = SemanticCoverageAudit(
            binding=replace(context.audit.binding, max_tokens=7),
            items=context.audit.items,
        )
        with pytest.raises(PersistenceIntegrityError, match="profile bindings differ"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                profile_mismatch_audit,
                context.plan,
                _generation_for(
                    context,
                    profile_mismatch_audit,
                    context.plan,
                    context.vectors,
                ),
                context.vectors,
            )

        identity_items = tuple(
            replace(item, source_id="source_other") for item in context.audit.items
        )
        identity_audit = SemanticCoverageAudit(
            binding=context.audit.binding,
            items=identity_items,
        )
        with pytest.raises(PersistenceIntegrityError, match="parent manifests differ"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                identity_audit,
                context.plan,
                _generation_for(context, identity_audit, context.plan, context.vectors),
                context.vectors,
            )

        count_item = replace(
            context.audit.items[0],
            prepared_token_count=cast(int, context.audit.items[0].prepared_token_count) + 1,
        )
        count_audit = SemanticCoverageAudit(
            binding=context.audit.binding,
            items=(count_item, *context.audit.items[1:]),
        )
        with pytest.raises(PersistenceIntegrityError, match="token counts differ"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                count_audit,
                context.plan,
                _generation_for(context, count_audit, context.plan, context.vectors),
                context.vectors,
            )

        three_dimensional = tuple(
            replace(item, vector=pack_normalized_vector((1.0, 0.0, 0.0)))
            for item in context.vectors
        )
        with pytest.raises(PersistenceIntegrityError, match="dimensions differ"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                _generation_for(
                    context,
                    context.audit,
                    context.plan,
                    three_dimensional,
                ),
                three_dimensional,
            )

        with pytest.raises(PersistenceIntegrityError, match="model, runtime, token"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                _generation_for(
                    context,
                    context.audit,
                    context.plan,
                    three_dimensional,
                    dimensions=3,
                ),
                three_dimensional,
            )

        fewer_vectors = context.vectors[:1]
        with pytest.raises(PersistenceIntegrityError, match="manifest counts differ"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                _generation_for(
                    context,
                    context.audit,
                    context.plan,
                    fewer_vectors,
                ),
                fewer_vectors,
            )

        foreign_audit = SemanticCoverageAudit(
            binding=replace(context.audit.binding, library_id="library_other"),
            items=context.audit.items,
        )
        with pytest.raises(PersistenceIntegrityError, match="current AuthorizedRead"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                foreign_audit,
                context.plan,
                _generation_for(
                    context,
                    foreign_audit,
                    context.plan,
                    context.vectors,
                ),
                context.vectors,
            )

        first = context.vectors[0]
        forged_hash = "d" * 64
        forged = replace(
            first,
            semantic_span_id=f"semantic_span_{forged_hash[:32]}",
            semantic_span_hash=forged_hash,
        )
        forged_vectors = (forged, *context.vectors[1:])
        with pytest.raises(PersistenceIntegrityError, match="differs from its plan"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                _generation_for(context, context.audit, context.plan, forged_vectors),
                forged_vectors,
            )

        subset_audit = SemanticCoverageAudit(
            binding=context.audit.binding,
            items=context.audit.items[:1],
        )
        subset_plan = SemanticSpanPlan(
            profile=context.plan.profile,
            parents=context.plan.parents[:1],
        )
        subset_vectors = context.vectors[:1]
        with pytest.raises(PersistenceIntegrityError, match="complete permitted manifest"):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                subset_audit,
                subset_plan,
                _generation_for(context, subset_audit, subset_plan, subset_vectors),
                subset_vectors,
            )
        assert _counts(context) == (0, 0, 0, 0, 0, 0)
    finally:
        _close(context)


def test_stale_authorization_and_structural_plan_collision_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _context(tmp_path, suffix="stale")
    collision = _context(tmp_path, suffix="collision")
    try:
        monkeypatch.setattr(
            stale.repository,
            "_compile_db_access",
            lambda **_kwargs: collision.authorization.compiled,
        )
        with pytest.raises(AuthorizationError, match="metadata changed"):
            stale.semantic.persist_semantic_span_vector_generation(
                stale.authorization,
                stale.audit,
                stale.plan,
                stale.generation,
                stale.vectors,
            )
        assert _counts(stale) == (0, 0, 0, 0, 0, 0)

        collision.semantic.persist_semantic_span_vector_generation(
            collision.authorization,
            collision.audit,
            collision.plan,
            collision.generation,
            collision.vectors,
        )
        changed_parent = replace(
            collision.plan.parents[0],
            content_offsets_hash="e" * 64,
        )
        changed_plan = SemanticSpanPlan(
            profile=collision.plan.profile,
            parents=(changed_parent, *collision.plan.parents[1:]),
        )
        changed_generation = _generation_for(
            collision,
            collision.audit,
            changed_plan,
            collision.vectors,
        )
        with pytest.raises(PersistenceConflictError, match="structural closure"):
            collision.semantic.persist_semantic_span_vector_generation(
                collision.authorization,
                collision.audit,
                changed_plan,
                changed_generation,
                collision.vectors,
            )
    finally:
        _close(stale, collision)


def test_metadata_and_vector_corruption_are_detected_before_return(tmp_path: Path) -> None:
    first = _context(tmp_path, suffix="metric")
    second = _context(tmp_path, suffix="blob")
    try:
        connection = first.repository._store.connection
        connection.execute("DROP TRIGGER source_fragment_text_metrics_no_update")
        connection.execute(
            "UPDATE source_fragment_text_metrics SET text_char_count = text_char_count + 1"
        )
        with pytest.raises(PersistenceIntegrityError, match="character count"):
            first.semantic.persist_semantic_span_vector_generation(
                first.authorization,
                first.audit,
                first.plan,
                first.generation,
                first.vectors,
            )
        assert _counts(first) == (0, 0, 0, 0, 0, 0)

        second.semantic.persist_semantic_span_vector_generation(
            second.authorization,
            second.audit,
            second.plan,
            second.generation,
            second.vectors,
        )
        connection = second.repository._store.connection
        connection.execute("DROP TRIGGER semantic_span_vector_items_no_update")
        with second.repository._permit_vector_blob_read():
            connection.execute(
                "UPDATE semantic_span_vector_items SET vector_blob = ? WHERE span_ordinal = 0",
                (sqlite3.Binary(pack_normalized_vector((-1.0, 0.0)).blob),),
            )
        with pytest.raises(PersistenceIntegrityError, match="vector bytes"):
            second.semantic.load_semantic_span_vector_generation(
                second.authorization,
                second.generation.generation_hash,
            )
        assert second.repository._vector_blob_read_depth == 0
    finally:
        _close(first, second)


def test_canonical_empty_authorized_closure_round_trips(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        scope = RequestScope(
            library_id=context.repository.library_id,
            snapshot_hash=context.snapshot.manifest_hash,
            purpose="research",
            collection_ids=(context.collection_id,),
            exclusions=QueryExclusions(source_ids=(context.source_id,)),
        )
        authorization = context.repository.authorize_read(
            access_policy_id=context.policy.access_policy_id,
            scope=scope,
        )
        binding = replace(
            context.audit.binding,
            exclusion_hash=scope.exclusion_hash,
            permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
        )
        audit = audit_semantic_coverage((), binding=binding, token_counter=context.provider)
        plan = plan_semantic_spans((), profile=context.plan.profile, tokenizer=context.provider)
        generation, vectors = materialize_semantic_span_vectors(
            (),
            audit=audit,
            plan=plan,
            provider=context.provider,
        )
        expected = (audit, plan, generation, vectors)
        assert (
            context.semantic.persist_semantic_span_vector_generation(
                authorization,
                audit,
                plan,
                generation,
                vectors,
            )
            == expected
        )
        assert (
            context.semantic.load_semantic_span_vector_generation(
                authorization,
                generation.generation_hash,
            )
            == expected
        )
        assert _counts(context) == (1, 1, 0, 0, 1, 0)
    finally:
        _close(context)


@pytest.mark.parametrize(
    ("table", "sql", "parameters", "message"),
    (
        (
            "semantic_span_profiles",
            "UPDATE semantic_span_profiles SET profile_version = '2.0'",
            (),
            "Profile is invalid",
        ),
        (
            "semantic_span_profiles",
            "UPDATE semantic_span_profiles SET overlap_content_tokens = 1",
            (),
            "Profile identity mismatch",
        ),
        (
            "semantic_span_plans",
            "UPDATE semantic_span_plans SET parent_count = 3",
            (),
            "persisted counts differ",
        ),
        (
            "semantic_spans",
            "UPDATE semantic_spans SET semantic_span_hash = ? "
            "WHERE rowid = (SELECT MIN(rowid) FROM semantic_spans)",
            ("f" * 64,),
            "SemanticSpan identity mismatch",
        ),
        (
            "semantic_spans",
            "UPDATE semantic_spans SET char_end = 0 WHERE span_ordinal = 0",
            (),
            "must be positive",
        ),
        (
            "semantic_spans",
            "UPDATE semantic_spans SET char_start = 2, char_end = 1 "
            "WHERE rowid = (SELECT MIN(rowid) FROM semantic_spans)",
            (),
            "SemanticSpanPlan is invalid",
        ),
        (
            "semantic_span_parents",
            "UPDATE semantic_span_parents SET span_count = 2 "
            "WHERE rowid = (SELECT MIN(rowid) FROM semantic_span_parents)",
            (),
            "parent span count differs",
        ),
        (
            "semantic_span_plans",
            "UPDATE semantic_span_plans SET parent_manifest_hash = ?",
            ("d" * 64,),
            "SemanticSpanPlan identity mismatch",
        ),
        (
            "semantic_span_vector_generations",
            "UPDATE semantic_span_vector_generations SET span_manifest_hash = ?",
            ("e" * 64,),
            "generation identity mismatch",
        ),
        (
            "semantic_span_vector_generations",
            "UPDATE semantic_span_vector_generations SET access_policy_id = 'bad'",
            (),
            "vector generation is invalid",
        ),
        (
            "semantic_span_vector_generations",
            "UPDATE semantic_span_vector_generations SET dimensions = 3",
            (),
            "persisted plan/model/generation closure differs",
        ),
        (
            "semantic_span_vector_items",
            "UPDATE semantic_span_vector_items SET semantic_span_hash = 'bad'",
            (),
            "vector manifest is invalid",
        ),
        (
            "embedding_model_profiles",
            "UPDATE embedding_model_profiles SET provider = 'bad'",
            (),
            "EmbeddingModelProfile is invalid",
        ),
        (
            "embedding_model_profiles",
            "UPDATE embedding_model_profiles SET dimensions = 3",
            (),
            "EmbeddingModelProfile identity mismatch",
        ),
        (
            "model_runtime_profiles",
            "UPDATE model_runtime_profiles SET device = 'bad'",
            (),
            "ModelRuntimeProfile is invalid",
        ),
        (
            "model_runtime_profiles",
            "UPDATE model_runtime_profiles SET batch_size = 17",
            (),
            "ModelRuntimeProfile identity mismatch",
        ),
    ),
)
def test_every_reconstructed_layer_fails_closed_on_corruption(
    tmp_path: Path,
    table: str,
    sql: str,
    parameters: tuple[object, ...],
    message: str,
) -> None:
    context = _context(tmp_path)
    try:
        context.semantic.persist_semantic_span_vector_generation(
            context.authorization,
            context.audit,
            context.plan,
            context.generation,
            context.vectors,
        )
        connection = context.repository._store.connection
        connection.execute(f"DROP TRIGGER {table}_no_update")
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        if table == "semantic_span_vector_items":
            with context.repository._permit_vector_blob_read():
                connection.execute(sql, parameters)
        else:
            connection.execute(sql, parameters)
        with pytest.raises(PersistenceIntegrityError, match=message):
            context.semantic.load_semantic_span_vector_generation(
                context.authorization,
                context.generation.generation_hash,
            )
        assert context.repository._vector_blob_read_depth == 0
    finally:
        _close(context)


def test_missing_items_bad_blob_and_wrong_item_plan_are_detected(tmp_path: Path) -> None:
    missing = _context(tmp_path, suffix="missing")
    bad_blob = _context(tmp_path, suffix="bad_blob")
    wrong_plan = _context(tmp_path, suffix="wrong_plan")
    try:
        for context in (missing, bad_blob, wrong_plan):
            context.semantic.persist_semantic_span_vector_generation(
                context.authorization,
                context.audit,
                context.plan,
                context.generation,
                context.vectors,
            )

        connection = missing.repository._store.connection
        connection.execute("DROP TRIGGER semantic_span_vector_items_no_delete")
        connection.execute("DELETE FROM semantic_span_vector_items WHERE span_ordinal = 0")
        with pytest.raises(PersistenceIntegrityError, match="item count differs"):
            missing.semantic.load_semantic_span_vector_generation(
                missing.authorization,
                missing.generation.generation_hash,
            )

        connection = bad_blob.repository._store.connection
        connection.execute("DROP TRIGGER semantic_span_vector_items_no_update")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        with bad_blob.repository._permit_vector_blob_read():
            connection.execute("UPDATE semantic_span_vector_items SET vector_blob = 'bad'")
        with pytest.raises(PersistenceIntegrityError, match="immutable bytes"):
            bad_blob.semantic.load_semantic_span_vector_generation(
                bad_blob.authorization,
                bad_blob.generation.generation_hash,
            )

        connection = wrong_plan.repository._store.connection
        connection.execute("DROP TRIGGER semantic_span_vector_items_no_update")
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "UPDATE semantic_span_vector_items SET semantic_span_plan_id = 'semantic_span_plan_bad'"
        )
        with pytest.raises(PersistenceIntegrityError, match="item plan closure differs"):
            wrong_plan.semantic.load_semantic_span_vector_generation(
                wrong_plan.authorization,
                wrong_plan.generation.generation_hash,
            )
    finally:
        _close(missing, bad_blob, wrong_plan)


def test_defensive_loader_collisions_and_sqlite_conflict_roll_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_collision = _context(tmp_path, suffix="plan_loader")
    bundle_collision = _context(tmp_path, suffix="bundle_loader")
    profile_collision = _context(tmp_path, suffix="profile_loader")
    sqlite_conflict = _context(tmp_path, suffix="sqlite_conflict")
    try:
        monkeypatch.setattr(
            plan_collision.semantic,
            "_load_plan",
            lambda *_args, **_kwargs: SemanticSpanPlan(
                profile=plan_collision.plan.profile,
                parents=(),
            ),
        )
        with pytest.raises(PersistenceIntegrityError, match="Plan conflicts"):
            plan_collision.semantic.persist_semantic_span_vector_generation(
                plan_collision.authorization,
                plan_collision.audit,
                plan_collision.plan,
                plan_collision.generation,
                plan_collision.vectors,
            )
        assert _counts(plan_collision) == (0, 0, 0, 0, 0, 0)

        with monkeypatch.context() as patch:
            patch.setattr(
                bundle_collision.semantic,
                "_load_bundle",
                lambda *_args, **_kwargs: (
                    bundle_collision.audit,
                    bundle_collision.plan,
                    bundle_collision.generation,
                    (),
                ),
            )
            with pytest.raises(PersistenceIntegrityError, match="bundle conflicts"):
                bundle_collision.semantic.persist_semantic_span_vector_generation(
                    bundle_collision.authorization,
                    bundle_collision.audit,
                    bundle_collision.plan,
                    bundle_collision.generation,
                    bundle_collision.vectors,
                )
        assert _counts(bundle_collision) == (0, 0, 0, 0, 0, 0)

        with monkeypatch.context() as patch:
            patch.setattr(
                profile_collision.semantic,
                "_load_profile",
                lambda *_args, **_kwargs: replace(
                    profile_collision.plan.profile,
                    overlap_content_tokens=1,
                ),
            )
            with pytest.raises(PersistenceIntegrityError, match="Profile conflicts"):
                profile_collision.semantic.persist_semantic_span_vector_generation(
                    profile_collision.authorization,
                    profile_collision.audit,
                    profile_collision.plan,
                    profile_collision.generation,
                    profile_collision.vectors,
                )
        assert _counts(profile_collision) == (0, 0, 0, 0, 0, 0)

        with monkeypatch.context() as patch:
            patch.setattr(
                sqlite_conflict.semantic,
                "_insert_parents",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    sqlite3.IntegrityError("forced conflict")
                ),
            )
            with pytest.raises(PersistenceConflictError, match="insert conflicted"):
                sqlite_conflict.semantic.persist_semantic_span_vector_generation(
                    sqlite_conflict.authorization,
                    sqlite_conflict.audit,
                    sqlite_conflict.plan,
                    sqlite_conflict.generation,
                    sqlite_conflict.vectors,
                )
        assert _counts(sqlite_conflict) == (0, 0, 0, 0, 0, 0)
    finally:
        _close(plan_collision, bundle_collision, profile_collision, sqlite_conflict)


def test_private_loader_absence_and_scalar_decoders_are_fail_closed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        connection = context.repository._store.connection
        for operation in (
            lambda: context.semantic._load_profile(connection, "0" * 64),
            lambda: context.semantic._load_plan(
                connection,
                plan_hash="0" * 64,
                expected_profile=context.plan.profile,
            ),
            lambda: context.semantic._load_embedding_model_profile(connection, "0" * 64),
            lambda: context.semantic._load_runtime_profile(connection, "0" * 64),
        ):
            with pytest.raises(PersistenceIntegrityError, match="does not exist"):
                operation()
        assert _require_hash("a" * 64, "hash") == "a" * 64
        assert _text("text", "label") == "text"
        assert _integer(1, "label") == 1
        assert _positive_integer(1, "label") == 1
        assert _nonnegative_integer(0, "label") == 0
        assert _sqlite_boolean(1, "label") == 1
        assert _boolean(0, "label") is False
        assert _blob(b"blob") == b"blob"
        invalid = (
            lambda: _text(1, "label"),
            lambda: _integer(True, "label"),
            lambda: _positive_integer(0, "label"),
            lambda: _nonnegative_integer(-1, "label"),
            lambda: _sqlite_boolean(2, "label"),
            lambda: _blob(bytearray(b"bad")),
        )
        for invalid_operation in invalid:
            with pytest.raises(PersistenceIntegrityError):
                invalid_operation()
    finally:
        _close(context)


def test_entrypoint_types_hashes_and_foreign_authorization_fail_closed(tmp_path: Path) -> None:
    first = _context(tmp_path, suffix="first")
    second = _context(tmp_path, suffix="second")
    try:
        with pytest.raises(TypeError, match="LibraryRepository"):
            SQLiteSemanticSpanStore(cast(Any, object()))
        invalid_calls: tuple[Callable[[], object], ...] = (
            lambda: first.semantic.persist_semantic_span_vector_generation(
                cast(Any, object()), first.audit, first.plan, first.generation, first.vectors
            ),
            lambda: first.semantic.persist_semantic_span_vector_generation(
                first.authorization,
                cast(Any, object()),
                first.plan,
                first.generation,
                first.vectors,
            ),
            lambda: first.semantic.persist_semantic_span_vector_generation(
                first.authorization,
                first.audit,
                cast(Any, object()),
                first.generation,
                first.vectors,
            ),
            lambda: first.semantic.persist_semantic_span_vector_generation(
                first.authorization, first.audit, first.plan, cast(Any, object()), first.vectors
            ),
            lambda: first.semantic.persist_semantic_span_vector_generation(
                first.authorization, first.audit, first.plan, first.generation, cast(Any, [])
            ),
        )
        for call in invalid_calls[1:]:
            with pytest.raises(TypeError):
                call()
        with pytest.raises(AuthorizationError):
            invalid_calls[0]()
        for digest in ("bad", "0" * 64):
            expected = "SHA-256" if digest == "bad" else "does not exist"
            with pytest.raises(PersistenceIntegrityError, match=expected):
                first.semantic.load_semantic_span_vector_generation(
                    first.authorization,
                    digest,
                )
        with pytest.raises(AuthorizationError, match="not issued"):
            first.semantic.persist_semantic_span_vector_generation(
                second.authorization,
                first.audit,
                first.plan,
                first.generation,
                first.vectors,
            )
    finally:
        _close(first, second)
