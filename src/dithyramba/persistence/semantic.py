"""Exact, text-free SQLite persistence for SemanticSpan vector-v2 artifacts.

The adapter accepts only a live :class:`AuthorizedRead` issued by the same
``LibraryRepository`` instance.  It recompiles that capability inside every
read or write transaction, validates the complete permitted parent closure
using metadata only, and materializes protected vector bytes only after the
non-blob closure has been independently reconstructed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import ValidationError

from dithyramba.access import CompiledAccess, RequestScope
from dithyramba.persistence.errors import (
    PersistenceConflictError,
    PersistenceIntegrityError,
)
from dithyramba.persistence.models import AuthorizedRead
from dithyramba.recall.hybrid_models import (
    EmbeddingModelProfile,
    HybridContractError,
    ModelRuntimeProfile,
)
from dithyramba.recall.semantic_audit import (
    SemanticAuditBinding,
    SemanticAuditContractError,
    SemanticAuditItem,
    SemanticAuditItemStatus,
    SemanticAuditStatus,
    SemanticCoverageAudit,
)
from dithyramba.recall.semantic_spans import (
    SemanticSpan,
    SemanticSpanContractError,
    SemanticSpanParentPlan,
    SemanticSpanPlan,
    SemanticSpanProfile,
)
from dithyramba.recall.semantic_vector import (
    SEMANTIC_VECTOR_NORMALIZATION,
    SemanticSpanDenseVectorItem,
    SemanticSpanVectorError,
    SemanticSpanVectorGeneration,
    SemanticSpanVectorManifestItem,
)
from dithyramba.recall.vector import VectorContractError, load_packed_vector

from .repository import LibraryRepository, _timestamp, _validated_timestamp

SemanticSpanVectorBundle = tuple[
    SemanticCoverageAudit,
    SemanticSpanPlan,
    SemanticSpanVectorGeneration,
    tuple[SemanticSpanDenseVectorItem, ...],
]

_CHUNK_SIZE = 500


@dataclass(frozen=True, slots=True)
class _GenerationHeader:
    generation_id: str
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
    normalization: str
    parent_count: int
    span_count: int
    span_manifest_hash: str
    generation_hash: str
    created_at: str


class SQLiteSemanticSpanStore:
    """Persist and reconstruct one exact authorized SemanticSpan closure."""

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteSemanticSpanStore requires a LibraryRepository")
        self._repository = repository

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def persist_semantic_span_vector_generation(
        self,
        authorization: AuthorizedRead,
        audit: SemanticCoverageAudit,
        plan: SemanticSpanPlan,
        generation: SemanticSpanVectorGeneration,
        vectors: tuple[SemanticSpanDenseVectorItem, ...],
    ) -> SemanticSpanVectorBundle:
        """Atomically persist one fully reconstructed vector-v2 bundle."""

        ordered_vectors = self._validate_input_bundle(audit, plan, generation, vectors)
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                authorized = self._repository._revalidate_authorized_read_context(authorization)
                compiled = authorized.compiled
                self._validate_authorized_closure(
                    connection,
                    authorized.access_policy_id,
                    authorized.scope,
                    compiled,
                    generation,
                    plan,
                )
                model = self._load_embedding_model_profile(
                    connection,
                    generation.model_profile_hash,
                )
                runtime = self._load_runtime_profile(
                    connection,
                    generation.runtime_profile_hash,
                )
                self._validate_model_closure(plan, generation, model, runtime)

                self._insert_or_verify_profile(connection, plan.profile, created_at)
                self._after_persist_stage("profile")
                plan_inserted = self._insert_plan_header(connection, plan, created_at)
                self._after_persist_stage("plan")
                if plan_inserted:
                    self._insert_parents(connection, plan)
                    self._after_persist_stage("parents")
                    self._insert_spans(connection, plan)
                    self._after_persist_stage("spans")
                else:
                    self._after_persist_stage("parents")
                    self._after_persist_stage("spans")
                persisted_plan = self._load_plan(
                    connection,
                    plan_hash=plan.plan_hash,
                    expected_profile=plan.profile,
                )
                if persisted_plan != plan:
                    raise PersistenceIntegrityError(
                        "persisted SemanticSpanPlan conflicts with its content address"
                    )

                generation_inserted = self._insert_generation_header(
                    connection,
                    audit,
                    plan,
                    generation,
                    created_at,
                )
                self._after_persist_stage("generation")
                if generation_inserted:
                    self._insert_vectors(connection, generation, ordered_vectors)
                self._after_persist_stage("vectors")

                persisted = self._load_bundle(
                    connection,
                    authorized.access_policy_id,
                    authorized.scope,
                    compiled,
                    generation.generation_hash,
                )
                if persisted != (audit, plan, generation, ordered_vectors):
                    raise PersistenceIntegrityError(
                        "persisted SemanticSpan vector bundle conflicts with its content address"
                    )
                self._after_persist_stage("verified")
                return persisted
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError(
                "SemanticSpan vector generation insert conflicted"
            ) from error

    def load_semantic_span_vector_generation(
        self,
        authorization: AuthorizedRead,
        generation_hash: str,
    ) -> SemanticSpanVectorBundle:
        """Load only after current access and complete parent closure revalidate."""

        digest = _require_hash(generation_hash, "SemanticSpan generation hash")
        with self._repository._store.transaction() as connection:
            authorized = self._repository._revalidate_authorized_read_context(authorization)
            return self._load_bundle(
                connection,
                authorized.access_policy_id,
                authorized.scope,
                authorized.compiled,
                digest,
            )

    def _after_persist_stage(self, stage: str) -> None:
        """Fault-injection seam used to prove transaction rollback boundaries."""

        del stage

    def _validate_input_bundle(
        self,
        audit: SemanticCoverageAudit,
        plan: SemanticSpanPlan,
        generation: SemanticSpanVectorGeneration,
        vectors: tuple[SemanticSpanDenseVectorItem, ...],
    ) -> tuple[SemanticSpanDenseVectorItem, ...]:
        if type(audit) is not SemanticCoverageAudit:
            raise TypeError("audit must be an exact SemanticCoverageAudit")
        if type(plan) is not SemanticSpanPlan:
            raise TypeError("plan must be an exact SemanticSpanPlan")
        if type(generation) is not SemanticSpanVectorGeneration:
            raise TypeError("generation must be an exact SemanticSpanVectorGeneration")
        if type(vectors) is not tuple or any(
            type(item) is not SemanticSpanDenseVectorItem for item in vectors
        ):
            raise TypeError("vectors must be a tuple of exact SemanticSpanDenseVectorItem values")
        if audit.status is SemanticAuditStatus.FAILED:
            raise PersistenceIntegrityError("failed semantic audit cannot authorize persistence")

        ordered_vectors = tuple(sorted(vectors, key=_vector_sort_key))
        vector_ids = tuple(item.semantic_span_id for item in ordered_vectors)
        if len(set(vector_ids)) != len(vector_ids):
            raise PersistenceIntegrityError("semantic vector items must be unique")
        vector_manifest = tuple(item.manifest_item() for item in ordered_vectors)
        if vector_manifest != generation.items:
            raise PersistenceIntegrityError(
                "semantic vector items must equal the complete generation manifest"
            )
        if any(item.vector.dimensions != generation.dimensions for item in ordered_vectors):
            raise PersistenceIntegrityError("semantic vector dimensions differ from generation")

        binding = audit.binding
        profile = plan.profile
        if (
            profile.model_profile_hash != binding.model_profile_hash
            or profile.runtime_profile_hash != binding.runtime_profile_hash
            or profile.provisioning_receipt_hash != binding.provisioning_receipt_hash
            or profile.max_prepared_tokens != binding.max_tokens
            or profile.passage_prefix != binding.passage_prefix
        ):
            raise PersistenceIntegrityError("semantic audit and span profile bindings differ")
        if (
            generation.library_id != binding.library_id
            or generation.corpus_snapshot_id != binding.corpus_snapshot_id
            or generation.snapshot_hash != binding.snapshot_hash
            or generation.access_policy_id != binding.access_policy_id
            or generation.policy_hash != binding.policy_hash
            or generation.exclusion_hash != binding.exclusion_hash
            or generation.permitted_set_hash != binding.permitted_set_hash
            or generation.semantic_audit_id != audit.audit_id
            or generation.semantic_audit_hash != audit.audit_hash
            or generation.semantic_audit_binding_hash != binding.binding_hash
            or generation.semantic_audit_fragment_manifest_hash != audit.fragment_manifest_hash
            or generation.semantic_span_plan_id != plan.plan_id
            or generation.semantic_span_plan_hash != plan.plan_hash
            or generation.semantic_span_profile_hash != profile.profile_hash
            or generation.semantic_span_parent_manifest_hash != plan.parent_manifest_hash
            or generation.semantic_span_closure_hash != plan.span_closure_hash
            or generation.model_profile_hash != profile.model_profile_hash
            or generation.runtime_profile_hash != profile.runtime_profile_hash
            or generation.provisioning_receipt_hash != profile.provisioning_receipt_hash
            or generation.normalization != SEMANTIC_VECTOR_NORMALIZATION
        ):
            raise PersistenceIntegrityError("semantic generation, audit, and plan bindings differ")

        audit_identity = tuple(_audit_identity(item) for item in audit.items)
        parent_identity = tuple(_parent_identity(parent) for parent in plan.parents)
        if audit_identity != parent_identity:
            raise PersistenceIntegrityError("semantic audit and parent manifests differ")
        for audit_item, parent in zip(audit.items, plan.parents, strict=True):
            if audit_item.prepared_token_count != parent.full_prepared_token_count:
                raise PersistenceIntegrityError("semantic audit and parent token counts differ")
        spans = plan.spans
        if len(spans) != len(generation.items):
            raise PersistenceIntegrityError("semantic span and vector manifest counts differ")
        for span, item in zip(spans, generation.items, strict=True):
            if (
                span.span_id != item.semantic_span_id
                or span.span_hash != item.semantic_span_hash
                or span.parent_fragment_id != item.parent_fragment_id
                or span.span_ordinal != item.span_ordinal
                or span.span_text_sha256 != item.span_text_sha256
            ):
                raise PersistenceIntegrityError("semantic vector manifest differs from its plan")
        return ordered_vectors

    def _validate_authorized_closure(
        self,
        connection: sqlite3.Connection,
        issued_access_policy_id: str,
        issued_scope: RequestScope,
        compiled: CompiledAccess,
        generation: SemanticSpanVectorGeneration,
        plan: SemanticSpanPlan,
    ) -> None:
        token = compiled.token
        manifest = compiled.manifest
        if (
            generation.library_id != self.library_id
            or manifest.library_id != self.library_id
            or generation.library_id != token.library_id
            or generation.snapshot_hash != token.snapshot_hash
            or generation.snapshot_hash != manifest.snapshot_hash
            or generation.snapshot_hash != issued_scope.snapshot_hash
            or generation.access_policy_id != issued_access_policy_id
            or generation.policy_hash != token.policy_hash
            or generation.exclusion_hash != token.exclusion_hash
            or generation.exclusion_hash != issued_scope.exclusion_hash
            or generation.permitted_set_hash != token.permitted_set_hash
            or generation.permitted_set_hash != manifest.permitted_set_hash
        ):
            raise PersistenceIntegrityError(
                "SemanticSpan generation differs from the current AuthorizedRead"
            )
        snapshot_rows = connection.execute(
            """
            SELECT corpus_snapshot_id
            FROM corpus_snapshots
            WHERE library_id = ? AND manifest_hash = ?
            ORDER BY corpus_snapshot_id COLLATE BINARY
            """,
            (self.library_id, generation.snapshot_hash),
        ).fetchall()
        if len(snapshot_rows) != 1 or _text(snapshot_rows[0][0], "snapshot ID") != (
            generation.corpus_snapshot_id
        ):
            raise PersistenceIntegrityError("SemanticSpan snapshot ID/hash closure differs")

        compiled_identity = tuple(
            (
                item.source_fragment_id,
                item.source_version_id,
                item.source_id,
            )
            for item in manifest.items
        )
        parent_identity = tuple(
            (
                parent.source_fragment_id,
                parent.source_version_id,
                parent.source_id,
            )
            for parent in plan.parents
        )
        if compiled_identity != parent_identity:
            raise PersistenceIntegrityError(
                "SemanticSpan parents must equal the complete permitted manifest"
            )
        self._validate_parent_database_closure(
            connection,
            generation.corpus_snapshot_id,
            plan.parents,
        )

    def _validate_parent_database_closure(
        self,
        connection: sqlite3.Connection,
        snapshot_id: str,
        parents: tuple[SemanticSpanParentPlan, ...],
    ) -> None:
        expected = {parent.source_fragment_id: parent for parent in parents}
        observed: dict[str, tuple[str, str, int, str, int, int]] = {}
        identifiers = tuple(expected)
        for chunk in _chunks(identifiers):
            placeholders = ", ".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT fragment.source_fragment_id, fragment.source_version_id,
                       version.source_id, fragment.ordinal, fragment.text_sha256,
                       metrics.text_char_count,
                       EXISTS (
                           SELECT 1 FROM snapshot_members AS member
                           WHERE member.corpus_snapshot_id = ?
                             AND member.source_version_id = fragment.source_version_id
                             AND member.membership_state = 'active'
                       )
                FROM source_fragments AS fragment
                JOIN source_fragment_text_metrics AS metrics
                  ON metrics.source_fragment_id = fragment.source_fragment_id
                 AND metrics.text_sha256 = fragment.text_sha256
                JOIN source_versions AS version
                  ON version.source_version_id = fragment.source_version_id
                JOIN sources AS source ON source.source_id = version.source_id
                WHERE source.library_id = ?
                  AND fragment.source_fragment_id IN ({placeholders})
                ORDER BY fragment.source_fragment_id COLLATE BINARY
                """,
                (snapshot_id, self.library_id, *chunk),
            ).fetchall()
            for row in rows:
                fragment_id = _text(row[0], "parent fragment ID")
                if fragment_id in observed:
                    raise PersistenceIntegrityError("duplicate persisted SemanticSpan parent")
                observed[fragment_id] = (
                    _text(row[1], "parent SourceVersion ID"),
                    _text(row[2], "parent Source ID"),
                    _nonnegative_integer(row[3], "parent ordinal"),
                    _require_hash(row[4], "parent text hash"),
                    _positive_integer(row[5], "parent text character count"),
                    _sqlite_boolean(row[6], "active snapshot membership"),
                )
        if set(observed) != set(expected):
            raise PersistenceIntegrityError(
                "SemanticSpan parents do not resolve to the exact persisted fragment set"
            )
        for fragment_id, parent in expected.items():
            version_id, source_id, ordinal, text_hash, text_char_count, active = observed[
                fragment_id
            ]
            if (
                version_id != parent.source_version_id
                or source_id != parent.source_id
                or ordinal != parent.ordinal
                or text_hash != parent.text_sha256
                or text_char_count != parent.text_char_count
                or active != 1
            ):
                raise PersistenceIntegrityError(
                    "SemanticSpan parent lineage, ordinal, hash, character count, "
                    "or snapshot state differs"
                )

    def _validate_model_closure(
        self,
        plan: SemanticSpanPlan,
        generation: SemanticSpanVectorGeneration,
        model: EmbeddingModelProfile,
        runtime: ModelRuntimeProfile,
    ) -> None:
        profile = plan.profile
        if (
            profile.model_profile_hash != model.profile_hash
            or profile.runtime_profile_hash != runtime.profile_hash
            or profile.max_prepared_tokens != model.max_tokens
            or profile.passage_prefix != model.passage_prefix
            or generation.dimensions != model.dimensions
        ):
            raise PersistenceIntegrityError(
                "SemanticSpan model, runtime, token, prefix, or dimension closure differs"
            )

    def _insert_or_verify_profile(
        self,
        connection: sqlite3.Connection,
        profile: SemanticSpanProfile,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO semantic_span_profiles(
                semantic_span_profile_id, profile_version, model_profile_hash,
                runtime_profile_hash, provisioning_receipt_hash,
                max_prepared_tokens, overlap_content_tokens, passage_prefix,
                profile_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.profile_id,
                profile.profile_version,
                profile.model_profile_hash,
                profile.runtime_profile_hash,
                profile.provisioning_receipt_hash,
                profile.max_prepared_tokens,
                profile.overlap_content_tokens,
                profile.passage_prefix,
                profile.profile_hash,
                created_at,
            ),
        )
        persisted = self._load_profile(connection, profile.profile_hash)
        if persisted != profile:
            raise PersistenceIntegrityError(
                "persisted SemanticSpanProfile conflicts with its content address"
            )

    def _insert_plan_header(
        self,
        connection: sqlite3.Connection,
        plan: SemanticSpanPlan,
        created_at: str,
    ) -> bool:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO semantic_span_plans(
                semantic_span_plan_id, plan_hash, semantic_span_profile_hash,
                parent_manifest_hash, span_closure_hash, parent_count,
                span_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.plan_id,
                plan.plan_hash,
                plan.profile.profile_hash,
                plan.parent_manifest_hash,
                plan.span_closure_hash,
                len(plan.parents),
                len(plan.spans),
                created_at,
            ),
        )
        if cursor.rowcount == 0:
            exists = connection.execute(
                "SELECT 1 FROM semantic_span_plans WHERE plan_hash = ?",
                (plan.plan_hash,),
            ).fetchone()
            if exists is None:
                raise PersistenceConflictError(
                    "SemanticSpanPlan conflicts with an existing structural closure"
                )
            return False
        return True

    def _insert_parents(
        self,
        connection: sqlite3.Connection,
        plan: SemanticSpanPlan,
    ) -> None:
        plan_id = plan.plan_id
        connection.executemany(
            """
            INSERT INTO semantic_span_parents(
                semantic_span_plan_id, source_fragment_id, source_version_id,
                source_id, ordinal, text_sha256, text_char_count,
                content_token_count, full_prepared_token_count,
                content_offsets_hash, span_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    plan_id,
                    parent.source_fragment_id,
                    parent.source_version_id,
                    parent.source_id,
                    parent.ordinal,
                    parent.text_sha256,
                    parent.text_char_count,
                    parent.content_token_count,
                    parent.full_prepared_token_count,
                    parent.content_offsets_hash,
                    len(parent.spans),
                )
                for parent in plan.parents
            ),
        )

    def _insert_spans(
        self,
        connection: sqlite3.Connection,
        plan: SemanticSpanPlan,
    ) -> None:
        plan_id = plan.plan_id
        connection.executemany(
            """
            INSERT INTO semantic_spans(
                semantic_span_plan_id, semantic_span_id, semantic_span_hash,
                parent_fragment_id, parent_text_sha256, parent_ordinal,
                span_ordinal, char_start, char_end, content_token_start,
                content_token_end, span_text_sha256, prepared_token_count,
                profile_hash, model_profile_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    plan_id,
                    span.span_id,
                    span.span_hash,
                    span.parent_fragment_id,
                    span.parent_text_sha256,
                    span.parent_ordinal,
                    span.span_ordinal,
                    span.char_start,
                    span.char_end,
                    span.content_token_start,
                    span.content_token_end,
                    span.span_text_sha256,
                    span.prepared_token_count,
                    span.profile_hash,
                    span.model_profile_hash,
                )
                for span in plan.spans
            ),
        )

    def _insert_generation_header(
        self,
        connection: sqlite3.Connection,
        audit: SemanticCoverageAudit,
        plan: SemanticSpanPlan,
        generation: SemanticSpanVectorGeneration,
        created_at: str,
    ) -> bool:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO semantic_span_vector_generations(
                semantic_span_vector_generation_id, library_id,
                corpus_snapshot_id, snapshot_hash, access_policy_id,
                policy_hash, exclusion_hash, permitted_set_hash,
                semantic_audit_id, semantic_audit_hash,
                semantic_audit_binding_hash,
                semantic_audit_fragment_manifest_hash,
                semantic_span_plan_id, semantic_span_plan_hash,
                semantic_span_profile_hash, semantic_span_parent_manifest_hash,
                semantic_span_closure_hash, model_profile_hash,
                runtime_profile_hash, provisioning_receipt_hash, dimensions,
                normalization, parent_count, span_count, span_manifest_hash,
                generation_hash, created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                generation.generation_id,
                generation.library_id,
                generation.corpus_snapshot_id,
                generation.snapshot_hash,
                generation.access_policy_id,
                generation.policy_hash,
                generation.exclusion_hash,
                generation.permitted_set_hash,
                audit.audit_id,
                audit.audit_hash,
                audit.binding.binding_hash,
                audit.fragment_manifest_hash,
                plan.plan_id,
                plan.plan_hash,
                plan.profile.profile_hash,
                plan.parent_manifest_hash,
                plan.span_closure_hash,
                generation.model_profile_hash,
                generation.runtime_profile_hash,
                generation.provisioning_receipt_hash,
                generation.dimensions,
                generation.normalization,
                len(plan.parents),
                len(plan.spans),
                generation.span_manifest_hash,
                generation.generation_hash,
                created_at,
            ),
        )
        if cursor.rowcount == 0:
            exists = connection.execute(
                """
                SELECT 1 FROM semantic_span_vector_generations
                WHERE generation_hash = ? AND library_id = ?
                """,
                (generation.generation_hash, self.library_id),
            ).fetchone()
            if exists is None:
                raise PersistenceConflictError(
                    "conflicting nondeterministic SemanticSpan vector generation"
                )
            return False
        return True

    def _insert_vectors(
        self,
        connection: sqlite3.Connection,
        generation: SemanticSpanVectorGeneration,
        vectors: tuple[SemanticSpanDenseVectorItem, ...],
    ) -> None:
        generation_id = generation.generation_id
        with self._repository._permit_vector_blob_read():
            connection.executemany(
                """
                INSERT INTO semantic_span_vector_items(
                    semantic_span_vector_generation_id, semantic_span_plan_id,
                    semantic_span_id, parent_fragment_id, span_ordinal,
                    semantic_span_hash, span_text_sha256, vector_sha256,
                    vector_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        generation_id,
                        generation.semantic_span_plan_id,
                        item.semantic_span_id,
                        item.parent_fragment_id,
                        item.span_ordinal,
                        item.semantic_span_hash,
                        item.span_text_sha256,
                        item.vector.vector_hash,
                        sqlite3.Binary(item.vector.blob),
                    )
                    for item in vectors
                ),
            )

    def _load_bundle(
        self,
        connection: sqlite3.Connection,
        issued_access_policy_id: str,
        issued_scope: RequestScope,
        compiled: CompiledAccess,
        generation_hash: str,
    ) -> SemanticSpanVectorBundle:
        header = self._load_generation_header(connection, generation_hash)
        profile = self._load_profile(connection, header.semantic_span_profile_hash)
        plan = self._load_plan(
            connection,
            plan_hash=header.semantic_span_plan_hash,
            expected_profile=profile,
        )
        model = self._load_embedding_model_profile(connection, header.model_profile_hash)
        runtime = self._load_runtime_profile(connection, header.runtime_profile_hash)
        self._validate_model_header_closure(header, plan, model, runtime)
        manifest = self._load_vector_manifest(connection, header)
        generation = self._reconstruct_generation(header, manifest)
        audit = self._reconstruct_audit(header, plan, model)
        self._validate_authorized_closure(
            connection,
            issued_access_policy_id,
            issued_scope,
            compiled,
            generation,
            plan,
        )
        vectors = self._load_protected_vectors(connection, header, manifest)
        ordered = self._validate_input_bundle(audit, plan, generation, vectors)
        return audit, plan, generation, ordered

    def _load_profile(
        self,
        connection: sqlite3.Connection,
        profile_hash: str,
    ) -> SemanticSpanProfile:
        digest = _require_hash(profile_hash, "SemanticSpanProfile hash")
        row = connection.execute(
            """
            SELECT semantic_span_profile_id, profile_version, model_profile_hash,
                   runtime_profile_hash, provisioning_receipt_hash,
                   max_prepared_tokens, overlap_content_tokens, passage_prefix,
                   profile_hash, created_at
            FROM semantic_span_profiles WHERE profile_hash = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("SemanticSpanProfile does not exist")
        try:
            profile = SemanticSpanProfile(
                model_profile_hash=_text(row[2], "SemanticSpan model hash"),
                runtime_profile_hash=_text(row[3], "SemanticSpan runtime hash"),
                provisioning_receipt_hash=_text(row[4], "SemanticSpan provisioning hash"),
                profile_version=cast(Literal["1.0"], _text(row[1], "profile version")),
                max_prepared_tokens=_positive_integer(row[5], "max prepared tokens"),
                overlap_content_tokens=_nonnegative_integer(row[6], "overlap tokens"),
                passage_prefix=_text(row[7], "passage prefix"),
            )
        except (SemanticSpanContractError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted SemanticSpanProfile is invalid") from error
        if (
            profile.profile_id != _text(row[0], "SemanticSpanProfile ID")
            or profile.profile_hash != _text(row[8], "SemanticSpanProfile hash")
            or profile.profile_hash != digest
        ):
            raise PersistenceIntegrityError("persisted SemanticSpanProfile identity mismatch")
        _validated_timestamp(_text(row[9], "SemanticSpanProfile timestamp"))
        return profile

    def _load_plan(
        self,
        connection: sqlite3.Connection,
        *,
        plan_hash: str,
        expected_profile: SemanticSpanProfile,
    ) -> SemanticSpanPlan:
        digest = _require_hash(plan_hash, "SemanticSpanPlan hash")
        row = connection.execute(
            """
            SELECT semantic_span_plan_id, plan_hash, semantic_span_profile_hash,
                   parent_manifest_hash, span_closure_hash, parent_count,
                   span_count, created_at
            FROM semantic_span_plans WHERE plan_hash = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("SemanticSpanPlan does not exist")
        plan_id = _text(row[0], "SemanticSpanPlan ID")
        parent_count = _nonnegative_integer(row[5], "SemanticSpanPlan parent count")
        span_count = _nonnegative_integer(row[6], "SemanticSpanPlan span count")
        parent_rows = connection.execute(
            """
            SELECT source_fragment_id, source_version_id, source_id, ordinal,
                   text_sha256, text_char_count, content_token_count,
                   full_prepared_token_count, content_offsets_hash, span_count
            FROM semantic_span_parents
            WHERE semantic_span_plan_id = ?
            ORDER BY source_fragment_id COLLATE BINARY
            """,
            (plan_id,),
        ).fetchall()
        span_rows = connection.execute(
            """
            SELECT semantic_span_id, semantic_span_hash, parent_fragment_id,
                   parent_text_sha256, parent_ordinal, span_ordinal,
                   char_start, char_end, content_token_start, content_token_end,
                   span_text_sha256, prepared_token_count, profile_hash,
                   model_profile_hash
            FROM semantic_spans
            WHERE semantic_span_plan_id = ?
            ORDER BY parent_fragment_id COLLATE BINARY, span_ordinal,
                     semantic_span_id COLLATE BINARY
            """,
            (plan_id,),
        ).fetchall()
        if len(parent_rows) != parent_count or len(span_rows) != span_count:
            raise PersistenceIntegrityError("SemanticSpanPlan persisted counts differ")
        spans_by_parent: dict[str, list[SemanticSpan]] = {}
        try:
            for span_row in span_rows:
                span = SemanticSpan(
                    parent_fragment_id=_text(span_row[2], "span parent fragment ID"),
                    parent_text_sha256=_text(span_row[3], "span parent text hash"),
                    parent_ordinal=_nonnegative_integer(span_row[4], "span parent ordinal"),
                    span_ordinal=_nonnegative_integer(span_row[5], "span ordinal"),
                    char_start=_nonnegative_integer(span_row[6], "span char start"),
                    char_end=_positive_integer(span_row[7], "span char end"),
                    content_token_start=_nonnegative_integer(
                        span_row[8], "span content token start"
                    ),
                    content_token_end=_nonnegative_integer(span_row[9], "span content token end"),
                    span_text_sha256=_text(span_row[10], "span text hash"),
                    prepared_token_count=_positive_integer(
                        span_row[11], "span prepared token count"
                    ),
                    profile_hash=_text(span_row[12], "span profile hash"),
                    model_profile_hash=_text(span_row[13], "span model hash"),
                )
                if span.span_id != _text(span_row[0], "SemanticSpan ID") or span.span_hash != _text(
                    span_row[1], "SemanticSpan hash"
                ):
                    raise PersistenceIntegrityError("persisted SemanticSpan identity mismatch")
                spans_by_parent.setdefault(span.parent_fragment_id, []).append(span)
            parents = tuple(
                SemanticSpanParentPlan(
                    source_fragment_id=_text(parent[0], "parent fragment ID"),
                    source_version_id=_text(parent[1], "parent SourceVersion ID"),
                    source_id=_text(parent[2], "parent Source ID"),
                    ordinal=_nonnegative_integer(parent[3], "parent ordinal"),
                    text_sha256=_text(parent[4], "parent text hash"),
                    text_char_count=_positive_integer(parent[5], "parent char count"),
                    content_token_count=_nonnegative_integer(
                        parent[6], "parent content token count"
                    ),
                    full_prepared_token_count=_positive_integer(
                        parent[7], "parent prepared token count"
                    ),
                    content_offsets_hash=_text(parent[8], "parent offsets hash"),
                    spans=tuple(spans_by_parent.get(_text(parent[0], "parent fragment ID"), ())),
                )
                for parent in parent_rows
            )
            plan = SemanticSpanPlan(profile=expected_profile, parents=parents)
        except (
            SemanticSpanContractError,
            PersistenceIntegrityError,
            TypeError,
            ValueError,
        ) as error:
            if isinstance(error, PersistenceIntegrityError):
                raise
            raise PersistenceIntegrityError("persisted SemanticSpanPlan is invalid") from error
        if set(spans_by_parent) != {parent.source_fragment_id for parent in parents}:
            raise PersistenceIntegrityError("SemanticSpanPlan contains an orphan span")
        for parent_row, parent in zip(parent_rows, parents, strict=True):
            if _positive_integer(parent_row[9], "parent span count") != len(parent.spans):
                raise PersistenceIntegrityError("SemanticSpan parent span count differs")
        if (
            _text(row[2], "SemanticSpanProfile hash") != expected_profile.profile_hash
            or plan.plan_id != plan_id
            or plan.plan_hash != _text(row[1], "SemanticSpanPlan hash")
            or plan.plan_hash != digest
            or plan.parent_manifest_hash != _text(row[3], "parent manifest hash")
            or plan.span_closure_hash != _text(row[4], "span closure hash")
        ):
            raise PersistenceIntegrityError("persisted SemanticSpanPlan identity mismatch")
        _validated_timestamp(_text(row[7], "SemanticSpanPlan timestamp"))
        return plan

    def _load_generation_header(
        self,
        connection: sqlite3.Connection,
        generation_hash: str,
    ) -> _GenerationHeader:
        digest = _require_hash(generation_hash, "SemanticSpan generation hash")
        row = connection.execute(
            """
            SELECT semantic_span_vector_generation_id, library_id,
                   corpus_snapshot_id, snapshot_hash, access_policy_id,
                   policy_hash, exclusion_hash, permitted_set_hash,
                   semantic_audit_id, semantic_audit_hash,
                   semantic_audit_binding_hash,
                   semantic_audit_fragment_manifest_hash,
                   semantic_span_plan_id, semantic_span_plan_hash,
                   semantic_span_profile_hash, semantic_span_parent_manifest_hash,
                   semantic_span_closure_hash, model_profile_hash,
                   runtime_profile_hash, provisioning_receipt_hash, dimensions,
                   normalization, parent_count, span_count, span_manifest_hash,
                   generation_hash, created_at
            FROM semantic_span_vector_generations
            WHERE generation_hash = ? AND library_id = ?
            """,
            (digest, self.library_id),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError(
                "SemanticSpan vector generation does not exist in this Library"
            )
        header = _GenerationHeader(
            generation_id=_text(row[0], "SemanticSpan generation ID"),
            library_id=_text(row[1], "SemanticSpan Library ID"),
            corpus_snapshot_id=_text(row[2], "SemanticSpan snapshot ID"),
            snapshot_hash=_text(row[3], "SemanticSpan snapshot hash"),
            access_policy_id=_text(row[4], "SemanticSpan policy ID"),
            policy_hash=_text(row[5], "SemanticSpan policy hash"),
            exclusion_hash=_text(row[6], "SemanticSpan exclusion hash"),
            permitted_set_hash=_text(row[7], "SemanticSpan permitted-set hash"),
            semantic_audit_id=_text(row[8], "semantic audit ID"),
            semantic_audit_hash=_text(row[9], "semantic audit hash"),
            semantic_audit_binding_hash=_text(row[10], "semantic audit binding hash"),
            semantic_audit_fragment_manifest_hash=_text(
                row[11], "semantic audit fragment manifest hash"
            ),
            semantic_span_plan_id=_text(row[12], "SemanticSpanPlan ID"),
            semantic_span_plan_hash=_text(row[13], "SemanticSpanPlan hash"),
            semantic_span_profile_hash=_text(row[14], "SemanticSpanProfile hash"),
            semantic_span_parent_manifest_hash=_text(row[15], "SemanticSpan parent manifest hash"),
            semantic_span_closure_hash=_text(row[16], "SemanticSpan closure hash"),
            model_profile_hash=_text(row[17], "embedding model hash"),
            runtime_profile_hash=_text(row[18], "model runtime hash"),
            provisioning_receipt_hash=_text(row[19], "model provisioning hash"),
            dimensions=_positive_integer(row[20], "SemanticSpan generation dimensions"),
            normalization=_text(row[21], "SemanticSpan normalization"),
            parent_count=_nonnegative_integer(row[22], "SemanticSpan parent count"),
            span_count=_nonnegative_integer(row[23], "SemanticSpan span count"),
            span_manifest_hash=_text(row[24], "SemanticSpan manifest hash"),
            generation_hash=_text(row[25], "SemanticSpan generation hash"),
            created_at=_text(row[26], "SemanticSpan generation timestamp"),
        )
        if header.generation_hash != digest:
            raise PersistenceIntegrityError("persisted SemanticSpan generation hash mismatch")
        _validated_timestamp(header.created_at)
        return header

    def _load_vector_manifest(
        self,
        connection: sqlite3.Connection,
        header: _GenerationHeader,
    ) -> tuple[SemanticSpanVectorManifestItem, ...]:
        rows = connection.execute(
            """
            SELECT semantic_span_id, semantic_span_hash, parent_fragment_id,
                   span_ordinal, span_text_sha256, vector_sha256,
                   semantic_span_plan_id
            FROM semantic_span_vector_items
            WHERE semantic_span_vector_generation_id = ?
            ORDER BY parent_fragment_id COLLATE BINARY, span_ordinal,
                     semantic_span_id COLLATE BINARY
            """,
            (header.generation_id,),
        ).fetchall()
        if len(rows) != header.span_count:
            raise PersistenceIntegrityError("SemanticSpan vector item count differs")
        try:
            items = tuple(
                SemanticSpanVectorManifestItem(
                    semantic_span_id=_text(row[0], "vector SemanticSpan ID"),
                    semantic_span_hash=_text(row[1], "vector SemanticSpan hash"),
                    parent_fragment_id=_text(row[2], "vector parent fragment ID"),
                    span_ordinal=_nonnegative_integer(row[3], "vector span ordinal"),
                    span_text_sha256=_text(row[4], "vector span text hash"),
                    vector_sha256=_text(row[5], "vector hash"),
                )
                for row in rows
            )
        except (SemanticSpanVectorError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError(
                "persisted semantic vector manifest is invalid"
            ) from error
        if any(_text(row[6], "vector plan ID") != header.semantic_span_plan_id for row in rows):
            raise PersistenceIntegrityError("semantic vector item plan closure differs")
        return items

    def _reconstruct_generation(
        self,
        header: _GenerationHeader,
        manifest: tuple[SemanticSpanVectorManifestItem, ...],
    ) -> SemanticSpanVectorGeneration:
        try:
            generation = SemanticSpanVectorGeneration(
                library_id=header.library_id,
                corpus_snapshot_id=header.corpus_snapshot_id,
                snapshot_hash=header.snapshot_hash,
                access_policy_id=header.access_policy_id,
                policy_hash=header.policy_hash,
                exclusion_hash=header.exclusion_hash,
                permitted_set_hash=header.permitted_set_hash,
                semantic_audit_id=header.semantic_audit_id,
                semantic_audit_hash=header.semantic_audit_hash,
                semantic_audit_binding_hash=header.semantic_audit_binding_hash,
                semantic_audit_fragment_manifest_hash=(
                    header.semantic_audit_fragment_manifest_hash
                ),
                semantic_span_plan_id=header.semantic_span_plan_id,
                semantic_span_plan_hash=header.semantic_span_plan_hash,
                semantic_span_profile_hash=header.semantic_span_profile_hash,
                semantic_span_parent_manifest_hash=(header.semantic_span_parent_manifest_hash),
                semantic_span_closure_hash=header.semantic_span_closure_hash,
                model_profile_hash=header.model_profile_hash,
                runtime_profile_hash=header.runtime_profile_hash,
                provisioning_receipt_hash=header.provisioning_receipt_hash,
                dimensions=header.dimensions,
                items=manifest,
                normalization=cast(Literal["l2_unit_float32_v1"], header.normalization),
            )
        except (SemanticSpanVectorError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError(
                "persisted SemanticSpan vector generation is invalid"
            ) from error
        if (
            generation.generation_id != header.generation_id
            or generation.generation_hash != header.generation_hash
            or generation.span_manifest_hash != header.span_manifest_hash
            or len({item.parent_fragment_id for item in manifest}) != header.parent_count
        ):
            raise PersistenceIntegrityError(
                "persisted SemanticSpan vector generation identity mismatch"
            )
        return generation

    def _reconstruct_audit(
        self,
        header: _GenerationHeader,
        plan: SemanticSpanPlan,
        model: EmbeddingModelProfile,
    ) -> SemanticCoverageAudit:
        try:
            binding = SemanticAuditBinding(
                library_id=header.library_id,
                corpus_snapshot_id=header.corpus_snapshot_id,
                snapshot_hash=header.snapshot_hash,
                access_policy_id=header.access_policy_id,
                policy_hash=header.policy_hash,
                exclusion_hash=header.exclusion_hash,
                permitted_set_hash=header.permitted_set_hash,
                model_profile_hash=header.model_profile_hash,
                runtime_profile_hash=header.runtime_profile_hash,
                provisioning_receipt_hash=header.provisioning_receipt_hash,
                max_tokens=model.max_tokens,
                passage_prefix=model.passage_prefix,
            )
            items = tuple(
                SemanticAuditItem(
                    source_fragment_id=parent.source_fragment_id,
                    source_version_id=parent.source_version_id,
                    source_id=parent.source_id,
                    ordinal=parent.ordinal,
                    text_sha256=parent.text_sha256,
                    status=(
                        SemanticAuditItemStatus.FIT
                        if parent.full_prepared_token_count <= binding.max_tokens
                        else SemanticAuditItemStatus.WINDOW_REQUIRED
                    ),
                    prepared_token_count=parent.full_prepared_token_count,
                )
                for parent in plan.parents
            )
            audit = SemanticCoverageAudit(binding=binding, items=items)
        except (SemanticAuditContractError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted semantic audit is invalid") from error
        if (
            audit.audit_id != header.semantic_audit_id
            or audit.audit_hash != header.semantic_audit_hash
            or audit.binding.binding_hash != header.semantic_audit_binding_hash
            or audit.fragment_manifest_hash != header.semantic_audit_fragment_manifest_hash
        ):
            raise PersistenceIntegrityError("persisted semantic audit identity mismatch")
        return audit

    def _validate_model_header_closure(
        self,
        header: _GenerationHeader,
        plan: SemanticSpanPlan,
        model: EmbeddingModelProfile,
        runtime: ModelRuntimeProfile,
    ) -> None:
        if (
            header.semantic_span_plan_id != plan.plan_id
            or header.semantic_span_profile_hash != plan.profile.profile_hash
            or header.semantic_span_parent_manifest_hash != plan.parent_manifest_hash
            or header.semantic_span_closure_hash != plan.span_closure_hash
            or header.parent_count != len(plan.parents)
            or header.span_count != len(plan.spans)
            or header.model_profile_hash != model.profile_hash
            or header.runtime_profile_hash != runtime.profile_hash
            or header.provisioning_receipt_hash != plan.profile.provisioning_receipt_hash
            or plan.profile.max_prepared_tokens != model.max_tokens
            or plan.profile.passage_prefix != model.passage_prefix
            or header.dimensions != model.dimensions
            or header.normalization != SEMANTIC_VECTOR_NORMALIZATION
        ):
            raise PersistenceIntegrityError(
                "SemanticSpan persisted plan/model/generation closure differs"
            )

    def _load_protected_vectors(
        self,
        connection: sqlite3.Connection,
        header: _GenerationHeader,
        manifest: tuple[SemanticSpanVectorManifestItem, ...],
    ) -> tuple[SemanticSpanDenseVectorItem, ...]:
        with self._repository._permit_vector_blob_read():
            rows = connection.execute(
                """
                SELECT semantic_span_id, semantic_span_hash, parent_fragment_id,
                       span_ordinal, span_text_sha256, vector_sha256, vector_blob
                FROM semantic_span_vector_items
                WHERE semantic_span_vector_generation_id = ?
                ORDER BY parent_fragment_id COLLATE BINARY, span_ordinal,
                         semantic_span_id COLLATE BINARY
                """,
                (header.generation_id,),
            ).fetchall()
        if len(rows) != len(manifest):
            raise PersistenceIntegrityError("protected semantic vector closure changed")
        try:
            vectors = tuple(
                SemanticSpanDenseVectorItem(
                    semantic_span_id=_text(row[0], "vector SemanticSpan ID"),
                    semantic_span_hash=_text(row[1], "vector SemanticSpan hash"),
                    parent_fragment_id=_text(row[2], "vector parent fragment ID"),
                    span_ordinal=_nonnegative_integer(row[3], "vector span ordinal"),
                    span_text_sha256=_text(row[4], "vector span text hash"),
                    vector=load_packed_vector(
                        _blob(row[6]),
                        dimensions=header.dimensions,
                        expected_hash=_text(row[5], "vector hash"),
                    ),
                )
                for row in rows
            )
        except (SemanticSpanVectorError, VectorContractError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError(
                "persisted semantic vector bytes are invalid"
            ) from error
        if tuple(item.manifest_item() for item in vectors) != manifest:
            raise PersistenceIntegrityError("protected semantic vector manifest differs")
        return vectors

    def _load_embedding_model_profile(
        self,
        connection: sqlite3.Connection,
        profile_hash: str,
    ) -> EmbeddingModelProfile:
        digest = _require_hash(profile_hash, "EmbeddingModelProfile hash")
        row = connection.execute(
            """
            SELECT embedding_model_profile_id, provider, model_id, revision,
                   model_license, dimensions, max_tokens, query_prefix,
                   passage_prefix, normalize_embeddings, trust_remote_code,
                   profile_hash, created_at
            FROM embedding_model_profiles WHERE profile_hash = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("EmbeddingModelProfile does not exist")
        try:
            profile = EmbeddingModelProfile(
                provider=cast(Literal["sentence_transformers"], _text(row[1], "provider")),
                model_id=_text(row[2], "model ID"),
                revision=_text(row[3], "model revision"),
                license=_text(row[4], "model license"),
                dimensions=_positive_integer(row[5], "model dimensions"),
                max_tokens=_positive_integer(row[6], "model max tokens"),
                query_prefix=_text(row[7], "query prefix"),
                passage_prefix=_text(row[8], "passage prefix"),
                normalize_embeddings=cast(Literal[True], _boolean(row[9], "normalize")),
                trust_remote_code=cast(Literal[False], _boolean(row[10], "trust remote code")),
            )
        except (HybridContractError, ValidationError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted EmbeddingModelProfile is invalid") from error
        if (
            profile.profile_id != _text(row[0], "EmbeddingModelProfile ID")
            or profile.profile_hash != _text(row[11], "EmbeddingModelProfile hash")
            or profile.profile_hash != digest
        ):
            raise PersistenceIntegrityError("persisted EmbeddingModelProfile identity mismatch")
        _validated_timestamp(_text(row[12], "EmbeddingModelProfile timestamp"))
        return profile

    def _load_runtime_profile(
        self,
        connection: sqlite3.Connection,
        profile_hash: str,
    ) -> ModelRuntimeProfile:
        digest = _require_hash(profile_hash, "ModelRuntimeProfile hash")
        row = connection.execute(
            """
            SELECT model_runtime_profile_id, sentence_transformers_version,
                   transformers_version, torch_version, device, dtype,
                   batch_size, local_files_only, trust_remote_code,
                   profile_hash, created_at
            FROM model_runtime_profiles WHERE profile_hash = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("ModelRuntimeProfile does not exist")
        try:
            profile = ModelRuntimeProfile(
                sentence_transformers_version=_text(row[1], "sentence-transformers version"),
                transformers_version=_text(row[2], "transformers version"),
                torch_version=_text(row[3], "torch version"),
                device=cast(Literal["cpu", "mps", "cuda"], _text(row[4], "device")),
                dtype=cast(Literal["float32"], _text(row[5], "dtype")),
                batch_size=_positive_integer(row[6], "runtime batch size"),
                local_files_only=cast(Literal[True], _boolean(row[7], "local files only")),
                trust_remote_code=cast(Literal[False], _boolean(row[8], "trust remote code")),
            )
        except (HybridContractError, ValidationError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted ModelRuntimeProfile is invalid") from error
        if (
            profile.profile_id != _text(row[0], "ModelRuntimeProfile ID")
            or profile.profile_hash != _text(row[9], "ModelRuntimeProfile hash")
            or profile.profile_hash != digest
        ):
            raise PersistenceIntegrityError("persisted ModelRuntimeProfile identity mismatch")
        _validated_timestamp(_text(row[10], "ModelRuntimeProfile timestamp"))
        return profile


def _vector_sort_key(item: SemanticSpanDenseVectorItem) -> tuple[bytes, int, bytes]:
    return (
        item.parent_fragment_id.encode("ascii"),
        item.span_ordinal,
        item.semantic_span_id.encode("ascii"),
    )


def _audit_identity(item: SemanticAuditItem) -> tuple[str, str, str, int, str]:
    return (
        item.source_fragment_id,
        item.source_version_id,
        item.source_id,
        item.ordinal,
        item.text_sha256,
    )


def _parent_identity(parent: SemanticSpanParentPlan) -> tuple[str, str, str, int, str]:
    return (
        parent.source_fragment_id,
        parent.source_version_id,
        parent.source_id,
        parent.ordinal,
        parent.text_sha256,
    )


def _chunks(values: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        values[offset : offset + _CHUNK_SIZE] for offset in range(0, len(values), _CHUNK_SIZE)
    )


def _require_hash(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PersistenceIntegrityError(f"{label} must be lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        raise PersistenceIntegrityError(f"persisted {label} must be text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise PersistenceIntegrityError(f"persisted {label} must be an integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result < 1:
        raise PersistenceIntegrityError(f"persisted {label} must be positive")
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise PersistenceIntegrityError(f"persisted {label} must be nonnegative")
    return result


def _sqlite_boolean(value: object, label: str) -> int:
    result = _integer(value, label)
    if result not in (0, 1):
        raise PersistenceIntegrityError(f"persisted {label} must be a SQLite boolean")
    return result


def _boolean(value: object, label: str) -> bool:
    return bool(_sqlite_boolean(value, label))


def _blob(value: object) -> bytes:
    if type(value) is not bytes:
        raise PersistenceIntegrityError("persisted vector blob must be immutable bytes")
    return value
