"""Exact, append-only SQLite persistence for P7 profiles and corpus vectors.

This adapter deliberately persists only rebuildable hybrid-recall inputs. Query,
run, receipt, and packet persistence belong to a separate lifecycle boundary.
Vector bytes are materialized only inside the repository's dedicated vector
gate; this module never selects SourceFragment text or stores arbitrary JSON.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Literal, cast

from pydantic import ValidationError

from dithyramba.persistence.errors import (
    PersistenceConflictError,
    PersistenceIntegrityError,
)
from dithyramba.recall.hybrid_models import (
    CorpusLayer,
    CorpusLayerItem,
    CorpusLayerProfile,
    EmbeddingModelProfile,
    FusionProfile,
    HybridContractError,
    ModelRuntimeProfile,
)
from dithyramba.recall.rerank import RerankerContractError, RerankerProfile
from dithyramba.recall.vector import (
    CorpusVectorGeneration,
    DenseVectorItem,
    VectorContractError,
    VectorManifestItem,
    load_packed_vector,
)

from .repository import LibraryRepository, _timestamp, _validated_timestamp

_CHUNK_SIZE = 500


class SQLiteHybridStore:
    """Persist and independently reconstruct immutable P7 input artifacts."""

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteHybridStore requires a LibraryRepository")
        self._repository = repository

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def persist_embedding_model_profile(
        self,
        profile: EmbeddingModelProfile,
    ) -> EmbeddingModelProfile:
        """Insert a content-addressed embedding profile or verify the exact row."""

        if type(profile) is not EmbeddingModelProfile:
            raise TypeError("persist_embedding_model_profile requires an EmbeddingModelProfile")
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO embedding_model_profiles(
                        embedding_model_profile_id, provider, model_id, revision,
                        model_license, dimensions, max_tokens, query_prefix,
                        passage_prefix, normalize_embeddings, trust_remote_code,
                        profile_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.profile_id,
                        profile.provider,
                        profile.model_id,
                        profile.revision,
                        profile.license,
                        profile.dimensions,
                        profile.max_tokens,
                        profile.query_prefix,
                        profile.passage_prefix,
                        int(profile.normalize_embeddings),
                        int(profile.trust_remote_code),
                        profile.profile_hash,
                        created_at,
                    ),
                )
                persisted = self._load_embedding_model_profile(connection, profile.profile_hash)
                if persisted != profile:
                    raise PersistenceIntegrityError(
                        "persisted EmbeddingModelProfile conflicts with its content address"
                    )
                return persisted
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("EmbeddingModelProfile insert conflicted") from error

    def load_embedding_model_profile(self, profile_hash: str) -> EmbeddingModelProfile:
        """Load and independently re-hash one embedding profile."""

        with self._repository._store.transaction() as connection:
            return self._load_embedding_model_profile(connection, profile_hash)

    def persist_model_runtime_profile(
        self,
        profile: ModelRuntimeProfile,
    ) -> ModelRuntimeProfile:
        """Insert a content-addressed runtime profile or verify the exact row."""

        if type(profile) is not ModelRuntimeProfile:
            raise TypeError("persist_model_runtime_profile requires a ModelRuntimeProfile")
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO model_runtime_profiles(
                        model_runtime_profile_id, sentence_transformers_version,
                        transformers_version, torch_version, device, dtype,
                        batch_size, local_files_only, trust_remote_code,
                        profile_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.profile_id,
                        profile.sentence_transformers_version,
                        profile.transformers_version,
                        profile.torch_version,
                        profile.device,
                        profile.dtype,
                        profile.batch_size,
                        int(profile.local_files_only),
                        int(profile.trust_remote_code),
                        profile.profile_hash,
                        created_at,
                    ),
                )
                persisted = self._load_model_runtime_profile(connection, profile.profile_hash)
                if persisted != profile:
                    raise PersistenceIntegrityError(
                        "persisted ModelRuntimeProfile conflicts with its content address"
                    )
                return persisted
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("ModelRuntimeProfile insert conflicted") from error

    def load_model_runtime_profile(self, profile_hash: str) -> ModelRuntimeProfile:
        """Load and independently re-hash one runtime profile."""

        with self._repository._store.transaction() as connection:
            return self._load_model_runtime_profile(connection, profile_hash)

    def persist_fusion_profile(self, profile: FusionProfile) -> FusionProfile:
        """Insert the exact frozen fusion profile or verify the existing row."""

        if type(profile) is not FusionProfile:
            raise TypeError("persist_fusion_profile requires a FusionProfile")
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fusion_profiles(
                        profile_hash, algorithm, rrf_k, fts_weight, dense_weight,
                        max_per_derived_source_top_five, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.profile_hash,
                        profile.algorithm,
                        profile.rrf_k,
                        profile.fts_weight,
                        profile.dense_weight,
                        profile.max_per_derived_source_top_five,
                        created_at,
                    ),
                )
                persisted = self._load_fusion_profile(connection, profile.profile_hash)
                if persisted != profile:
                    raise PersistenceIntegrityError(
                        "persisted FusionProfile conflicts with its content address"
                    )
                return persisted
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("FusionProfile insert conflicted") from error

    def load_fusion_profile(self, profile_hash: str) -> FusionProfile:
        """Load and independently re-hash one fusion profile."""

        with self._repository._store.transaction() as connection:
            return self._load_fusion_profile(connection, profile_hash)

    def persist_reranker_profile(self, profile: RerankerProfile) -> RerankerProfile:
        """Insert one exact offline reranker profile or verify the existing row."""

        if type(profile) is not RerankerProfile:
            raise TypeError("persist_reranker_profile requires a RerankerProfile")
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO reranker_profiles(
                        reranker_profile_id, provider, model_id, revision,
                        model_license, max_tokens, local_files_only,
                        trust_remote_code, profile_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.profile_id,
                        profile.provider,
                        profile.model_id,
                        profile.revision,
                        profile.license,
                        profile.max_tokens,
                        int(profile.local_files_only),
                        int(profile.trust_remote_code),
                        profile.profile_hash,
                        created_at,
                    ),
                )
                persisted = self._load_reranker_profile(connection, profile.profile_hash)
                if persisted != profile:
                    raise PersistenceIntegrityError(
                        "persisted RerankerProfile conflicts with its content address"
                    )
                return persisted
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("RerankerProfile insert conflicted") from error

    def load_reranker_profile(self, profile_hash: str) -> RerankerProfile:
        """Load and independently re-hash one offline reranker profile."""

        with self._repository._store.transaction() as connection:
            return self._load_reranker_profile(connection, profile_hash)

    def persist_corpus_layer_profile(self, profile: CorpusLayerProfile) -> CorpusLayerProfile:
        """Persist one exact Library/snapshot-bound Source role map."""

        if type(profile) is not CorpusLayerProfile:
            raise TypeError("persist_corpus_layer_profile requires a CorpusLayerProfile")
        self._require_local_library(profile.library_id, "CorpusLayerProfile")
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                self._validate_layer_scope(connection, profile)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO corpus_layer_profiles(
                        corpus_layer_profile_id, library_id, corpus_snapshot_id,
                        exclusion_hash, permitted_set_hash, item_count,
                        profile_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.profile_id,
                        profile.library_id,
                        profile.corpus_snapshot_id,
                        profile.exclusion_hash,
                        profile.permitted_set_hash,
                        len(profile.items),
                        profile.profile_hash,
                        created_at,
                    ),
                )
                if cursor.rowcount == 1:
                    connection.executemany(
                        """
                        INSERT INTO corpus_layer_items(
                            corpus_layer_profile_id, source_id, layer
                        ) VALUES (?, ?, ?)
                        """,
                        (
                            (profile.profile_id, item.source_id, item.layer.value)
                            for item in profile.items
                        ),
                    )
                persisted = self._load_corpus_layer_profile(connection, profile.profile_hash)
                if persisted != profile:
                    raise PersistenceIntegrityError(
                        "persisted CorpusLayerProfile conflicts with its content address"
                    )
                return persisted
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("CorpusLayerProfile insert conflicted") from error

    def load_corpus_layer_profile(self, profile_hash: str) -> CorpusLayerProfile:
        """Load one complete Source role map and revalidate its snapshot closure."""

        with self._repository._store.transaction() as connection:
            return self._load_corpus_layer_profile(connection, profile_hash)

    def persist_corpus_vector_generation(
        self,
        generation: CorpusVectorGeneration,
        vectors: tuple[DenseVectorItem, ...],
    ) -> tuple[CorpusVectorGeneration, tuple[DenseVectorItem, ...]]:
        """Persist exact normalized bytes for one authorized vector manifest."""

        if type(generation) is not CorpusVectorGeneration:
            raise TypeError("persist_corpus_vector_generation requires a CorpusVectorGeneration")
        if type(vectors) is not tuple or any(type(item) is not DenseVectorItem for item in vectors):
            raise TypeError("vectors must be a tuple of DenseVectorItem values")
        self._require_local_library(generation.library_id, "CorpusVectorGeneration")
        ordered_vectors = tuple(
            sorted(vectors, key=lambda item: item.source_fragment_id.encode("ascii"))
        )
        self._validate_vector_manifest(generation, ordered_vectors)
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                self._validate_generation_closure(connection, generation, ordered_vectors)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO corpus_vector_generations(
                        corpus_vector_generation_id, library_id, corpus_snapshot_id,
                        snapshot_hash, access_policy_id, policy_hash, exclusion_hash,
                        permitted_set_hash, model_profile_hash, runtime_profile_hash,
                        dimensions, item_count, generation_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        generation.model_profile_hash,
                        generation.runtime_profile_hash,
                        generation.dimensions,
                        len(generation.items),
                        generation.generation_hash,
                        created_at,
                    ),
                )
                if cursor.rowcount == 1:
                    manifest = {item.source_fragment_id: item for item in generation.items}
                    # The schema's CHECK/trigger evaluates vector_blob during
                    # insertion, which SQLite reports to the authorizer as a
                    # column read. Keep that evaluation inside the same narrow
                    # adapter-owned gate as materialization.
                    with self._repository._permit_vector_blob_read():
                        connection.executemany(
                            """
                            INSERT INTO corpus_vector_items(
                                corpus_vector_generation_id, source_fragment_id,
                                text_sha256, vector_sha256, vector_blob
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                (
                                    generation.generation_id,
                                    item.source_fragment_id,
                                    manifest[item.source_fragment_id].text_sha256,
                                    item.vector.vector_hash,
                                    sqlite3.Binary(item.vector.blob),
                                )
                                for item in ordered_vectors
                            ),
                        )
                persisted = self._load_corpus_vector_generation(
                    connection,
                    generation.generation_hash,
                )
                if persisted != (generation, ordered_vectors):
                    raise PersistenceIntegrityError(
                        "persisted CorpusVectorGeneration conflicts with its content address"
                    )
                return persisted
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("CorpusVectorGeneration insert conflicted") from error

    def load_corpus_vector_generation(
        self,
        generation_hash: str,
    ) -> tuple[CorpusVectorGeneration, tuple[DenseVectorItem, ...]]:
        """Load and verify one generation; vector bytes cross only their narrow gate."""

        with self._repository._store.transaction() as connection:
            return self._load_corpus_vector_generation(connection, generation_hash)

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
            FROM embedding_model_profiles
            WHERE profile_hash = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("EmbeddingModelProfile does not exist")
        try:
            profile = EmbeddingModelProfile(
                provider=cast(
                    Literal["sentence_transformers"],
                    _text(row[1], "embedding provider"),
                ),
                model_id=_text(row[2], "embedding model ID"),
                revision=_text(row[3], "embedding revision"),
                license=_text(row[4], "embedding license"),
                dimensions=_integer(row[5], "embedding dimensions"),
                max_tokens=_integer(row[6], "embedding max tokens"),
                query_prefix=_text(row[7], "query prefix"),
                passage_prefix=_text(row[8], "passage prefix"),
                normalize_embeddings=cast(
                    Literal[True],
                    _boolean(row[9], "normalize_embeddings"),
                ),
                trust_remote_code=cast(
                    Literal[False],
                    _boolean(row[10], "trust_remote_code"),
                ),
            )
        except (HybridContractError, ValidationError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted EmbeddingModelProfile is invalid") from error
        if (
            _text(row[0], "EmbeddingModelProfile ID") != profile.profile_id
            or _text(row[11], "EmbeddingModelProfile hash") != profile.profile_hash
            or profile.profile_hash != digest
        ):
            raise PersistenceIntegrityError("persisted EmbeddingModelProfile identity mismatch")
        _validated_timestamp(_text(row[12], "EmbeddingModelProfile timestamp"))
        return profile

    def _load_model_runtime_profile(
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
            FROM model_runtime_profiles
            WHERE profile_hash = ?
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
                device=cast(
                    Literal["cpu", "mps", "cuda"],
                    _text(row[4], "runtime device"),
                ),
                dtype=cast(Literal["float32"], _text(row[5], "runtime dtype")),
                batch_size=_integer(row[6], "runtime batch size"),
                local_files_only=cast(
                    Literal[True],
                    _boolean(row[7], "local_files_only"),
                ),
                trust_remote_code=cast(
                    Literal[False],
                    _boolean(row[8], "trust_remote_code"),
                ),
            )
        except (HybridContractError, ValidationError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted ModelRuntimeProfile is invalid") from error
        if (
            _text(row[0], "ModelRuntimeProfile ID") != profile.profile_id
            or _text(row[9], "ModelRuntimeProfile hash") != profile.profile_hash
            or profile.profile_hash != digest
        ):
            raise PersistenceIntegrityError("persisted ModelRuntimeProfile identity mismatch")
        _validated_timestamp(_text(row[10], "ModelRuntimeProfile timestamp"))
        return profile

    def _load_fusion_profile(
        self,
        connection: sqlite3.Connection,
        profile_hash: str,
    ) -> FusionProfile:
        digest = _require_hash(profile_hash, "FusionProfile hash")
        row = connection.execute(
            """
            SELECT profile_hash, algorithm, rrf_k, fts_weight, dense_weight,
                   max_per_derived_source_top_five, created_at
            FROM fusion_profiles
            WHERE profile_hash = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("FusionProfile does not exist")
        try:
            profile = FusionProfile(
                algorithm=cast(Literal["rrf_v1"], _text(row[1], "fusion algorithm")),
                rrf_k=cast(Literal[60], _integer(row[2], "fusion rrf_k")),
                fts_weight=cast(Literal[1], _integer(row[3], "fusion FTS weight")),
                dense_weight=cast(Literal[1], _integer(row[4], "fusion dense weight")),
                max_per_derived_source_top_five=cast(
                    Literal[2],
                    _integer(row[5], "fusion layer cap"),
                ),
            )
        except (HybridContractError, ValidationError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted FusionProfile is invalid") from error
        if (
            _text(row[0], "FusionProfile hash") != profile.profile_hash
            or profile.profile_hash != digest
        ):
            raise PersistenceIntegrityError("persisted FusionProfile identity mismatch")
        _validated_timestamp(_text(row[6], "FusionProfile timestamp"))
        return profile

    def _load_reranker_profile(
        self,
        connection: sqlite3.Connection,
        profile_hash: str,
    ) -> RerankerProfile:
        digest = _require_hash(profile_hash, "RerankerProfile hash")
        row = connection.execute(
            """
            SELECT reranker_profile_id, provider, model_id, revision,
                   model_license, max_tokens, local_files_only,
                   trust_remote_code, profile_hash, created_at
            FROM reranker_profiles
            WHERE profile_hash = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("RerankerProfile does not exist")
        try:
            profile = RerankerProfile(
                provider=_text(row[1], "reranker provider"),
                model_id=_text(row[2], "reranker model ID"),
                revision=_text(row[3], "reranker revision"),
                license=_text(row[4], "reranker license"),
                max_tokens=_positive_integer(row[5], "reranker max tokens"),
                local_files_only=_boolean(row[6], "reranker local_files_only"),
                trust_remote_code=_boolean(row[7], "reranker trust_remote_code"),
            )
        except (RerankerContractError, ValidationError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted RerankerProfile is invalid") from error
        if (
            profile.profile_id != _text(row[0], "RerankerProfile ID")
            or profile.profile_hash != _text(row[8], "RerankerProfile hash")
            or profile.profile_hash != digest
        ):
            raise PersistenceIntegrityError("persisted RerankerProfile identity mismatch")
        _validated_timestamp(_text(row[9], "RerankerProfile timestamp"))
        return profile

    def _load_corpus_layer_profile(
        self,
        connection: sqlite3.Connection,
        profile_hash: str,
    ) -> CorpusLayerProfile:
        digest = _require_hash(profile_hash, "CorpusLayerProfile hash")
        row = connection.execute(
            """
            SELECT corpus_layer_profile_id, library_id, corpus_snapshot_id,
                   exclusion_hash, permitted_set_hash, item_count,
                   profile_hash, created_at
            FROM corpus_layer_profiles
            WHERE profile_hash = ? AND library_id = ?
            """,
            (digest, self.library_id),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("CorpusLayerProfile does not exist in this Library")
        profile_id = _text(row[0], "CorpusLayerProfile ID")
        item_count = _nonnegative_integer(row[5], "CorpusLayerProfile item count")
        item_rows = connection.execute(
            """
            SELECT source_id, layer
            FROM corpus_layer_items
            WHERE corpus_layer_profile_id = ?
            ORDER BY source_id COLLATE BINARY
            """,
            (profile_id,),
        ).fetchall()
        if len(item_rows) != item_count:
            raise PersistenceIntegrityError("CorpusLayerProfile item count mismatch")
        try:
            profile = CorpusLayerProfile(
                library_id=_text(row[1], "CorpusLayerProfile Library ID"),
                corpus_snapshot_id=_text(row[2], "CorpusLayerProfile snapshot ID"),
                exclusion_hash=_text(row[3], "CorpusLayerProfile exclusion hash"),
                permitted_set_hash=_text(row[4], "CorpusLayerProfile permitted-set hash"),
                items=tuple(
                    CorpusLayerItem(
                        source_id=_text(item[0], "CorpusLayerItem Source ID"),
                        layer=CorpusLayer(_text(item[1], "CorpusLayerItem layer")),
                    )
                    for item in item_rows
                ),
            )
        except (HybridContractError, ValidationError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted CorpusLayerProfile is invalid") from error
        if (
            profile.library_id != self.library_id
            or profile.profile_id != profile_id
            or profile.profile_hash != _text(row[6], "CorpusLayerProfile hash")
            or profile.profile_hash != digest
        ):
            raise PersistenceIntegrityError("persisted CorpusLayerProfile identity mismatch")
        _validated_timestamp(_text(row[7], "CorpusLayerProfile timestamp"))
        self._validate_layer_scope(connection, profile)
        return profile

    def _load_corpus_vector_generation(
        self,
        connection: sqlite3.Connection,
        generation_hash: str,
    ) -> tuple[CorpusVectorGeneration, tuple[DenseVectorItem, ...]]:
        digest = _require_hash(generation_hash, "CorpusVectorGeneration hash")
        row = connection.execute(
            """
            SELECT corpus_vector_generation_id, library_id, corpus_snapshot_id,
                   snapshot_hash, access_policy_id, policy_hash, exclusion_hash,
                   permitted_set_hash, model_profile_hash, runtime_profile_hash,
                   dimensions, item_count, generation_hash, created_at
            FROM corpus_vector_generations
            WHERE generation_hash = ? AND library_id = ?
            """,
            (digest, self.library_id),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("CorpusVectorGeneration does not exist in this Library")
        generation_id = _text(row[0], "CorpusVectorGeneration ID")
        dimensions = _positive_integer(row[10], "CorpusVectorGeneration dimensions")
        item_count = _nonnegative_integer(row[11], "CorpusVectorGeneration item count")
        with self._repository._permit_vector_blob_read():
            item_rows = connection.execute(
                """
                SELECT item.source_fragment_id, item.text_sha256,
                       item.vector_sha256, item.vector_blob, source.source_id
                FROM corpus_vector_items AS item
                JOIN source_fragments AS fragment
                  ON fragment.source_fragment_id = item.source_fragment_id
                JOIN source_versions AS version
                  ON version.source_version_id = fragment.source_version_id
                JOIN sources AS source ON source.source_id = version.source_id
                WHERE item.corpus_vector_generation_id = ?
                  AND source.library_id = ?
                ORDER BY item.source_fragment_id COLLATE BINARY
                """,
                (generation_id, self.library_id),
            ).fetchall()
        if len(item_rows) != item_count:
            raise PersistenceIntegrityError("CorpusVectorGeneration item count mismatch")
        try:
            manifest_items = tuple(
                VectorManifestItem(
                    source_fragment_id=_text(item[0], "vector fragment ID"),
                    text_sha256=_text(item[1], "vector text hash"),
                    vector_sha256=_text(item[2], "vector hash"),
                )
                for item in item_rows
            )
            vectors = tuple(
                DenseVectorItem(
                    source_fragment_id=_text(item[0], "vector fragment ID"),
                    source_id=_text(item[4], "vector Source ID"),
                    vector=load_packed_vector(
                        _blob(item[3]),
                        dimensions=dimensions,
                        expected_hash=_text(item[2], "vector hash"),
                    ),
                )
                for item in item_rows
            )
            generation = CorpusVectorGeneration(
                library_id=_text(row[1], "CorpusVectorGeneration Library ID"),
                corpus_snapshot_id=_text(row[2], "CorpusVectorGeneration snapshot ID"),
                snapshot_hash=_text(row[3], "CorpusVectorGeneration snapshot hash"),
                access_policy_id=_text(row[4], "CorpusVectorGeneration policy ID"),
                policy_hash=_text(row[5], "CorpusVectorGeneration policy hash"),
                exclusion_hash=_text(row[6], "CorpusVectorGeneration exclusion hash"),
                permitted_set_hash=_text(row[7], "CorpusVectorGeneration permitted-set hash"),
                model_profile_hash=_text(row[8], "CorpusVectorGeneration model hash"),
                runtime_profile_hash=_text(row[9], "CorpusVectorGeneration runtime hash"),
                dimensions=dimensions,
                items=manifest_items,
            )
        except (
            HybridContractError,
            ValidationError,
            VectorContractError,
            TypeError,
            ValueError,
        ) as error:
            raise PersistenceIntegrityError(
                "persisted CorpusVectorGeneration is invalid"
            ) from error
        if (
            generation.library_id != self.library_id
            or generation.generation_id != generation_id
            or generation.generation_hash != _text(row[12], "CorpusVectorGeneration hash")
            or generation.generation_hash != digest
        ):
            raise PersistenceIntegrityError("persisted CorpusVectorGeneration identity mismatch")
        _validated_timestamp(_text(row[13], "CorpusVectorGeneration timestamp"))
        self._validate_vector_manifest(generation, vectors)
        self._validate_generation_closure(connection, generation, vectors)
        return generation, vectors

    def _validate_layer_scope(
        self,
        connection: sqlite3.Connection,
        profile: CorpusLayerProfile,
    ) -> None:
        self._require_local_library(profile.library_id, "CorpusLayerProfile")
        snapshot = self._repository.get_corpus_snapshot(profile.corpus_snapshot_id)
        if snapshot.library_id != self.library_id:
            raise PersistenceIntegrityError(
                "CorpusLayerProfile snapshot belongs to another Library"
            )
        requested = {item.source_id for item in profile.items}
        if not requested:
            return
        active_sources: set[str] = set()
        for chunk in _chunks(tuple(sorted(requested))):
            placeholders = ", ".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT DISTINCT version.source_id
                FROM snapshot_members AS member
                JOIN source_versions AS version
                  ON version.source_version_id = member.source_version_id
                JOIN sources AS source ON source.source_id = version.source_id
                WHERE member.corpus_snapshot_id = ?
                  AND member.membership_state = 'active'
                  AND source.library_id = ?
                  AND version.source_id IN ({placeholders})
                """,
                (profile.corpus_snapshot_id, self.library_id, *chunk),
            ).fetchall()
            active_sources.update(_text(row[0], "snapshot Source ID") for row in rows)
        if active_sources != requested:
            raise PersistenceIntegrityError(
                "CorpusLayerProfile Sources are not active members of its snapshot"
            )

    def _validate_generation_closure(
        self,
        connection: sqlite3.Connection,
        generation: CorpusVectorGeneration,
        vectors: tuple[DenseVectorItem, ...],
    ) -> None:
        self._require_local_library(generation.library_id, "CorpusVectorGeneration")
        snapshot = self._repository.get_corpus_snapshot(generation.corpus_snapshot_id)
        if (
            snapshot.library_id != self.library_id
            or snapshot.manifest_hash != generation.snapshot_hash
        ):
            raise PersistenceIntegrityError("CorpusVectorGeneration snapshot closure mismatch")
        policy = self._repository.get_access_policy(generation.access_policy_id).snapshot
        if policy.library_id != self.library_id or policy.policy_hash != generation.policy_hash:
            raise PersistenceIntegrityError("CorpusVectorGeneration policy closure mismatch")
        model = self._load_embedding_model_profile(connection, generation.model_profile_hash)
        if model.dimensions != generation.dimensions:
            raise PersistenceIntegrityError("CorpusVectorGeneration model dimensions mismatch")
        self._load_model_runtime_profile(connection, generation.runtime_profile_hash)

        layer_rows = connection.execute(
            """
            SELECT profile_hash
            FROM corpus_layer_profiles
            WHERE library_id = ? AND corpus_snapshot_id = ?
              AND exclusion_hash = ? AND permitted_set_hash = ?
            ORDER BY profile_hash
            """,
            (
                self.library_id,
                generation.corpus_snapshot_id,
                generation.exclusion_hash,
                generation.permitted_set_hash,
            ),
        ).fetchall()
        if len(layer_rows) != 1:
            raise PersistenceIntegrityError(
                "CorpusVectorGeneration exclusion/permitted-set closure is absent or ambiguous"
            )
        layer = self._load_corpus_layer_profile(
            connection,
            _text(layer_rows[0][0], "CorpusLayerProfile hash"),
        )
        layer_source_ids = {item.source_id for item in layer.items}

        expected_manifest = {item.source_fragment_id: item for item in generation.items}
        persisted_fragments: dict[str, tuple[str, str]] = {}
        for chunk in _chunks(tuple(expected_manifest)):
            placeholders = ", ".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT DISTINCT fragment.source_fragment_id, fragment.text_sha256,
                                version.source_id
                FROM source_fragments AS fragment
                JOIN source_versions AS version
                  ON version.source_version_id = fragment.source_version_id
                JOIN sources AS source ON source.source_id = version.source_id
                JOIN snapshot_members AS member
                  ON member.source_version_id = version.source_version_id
                WHERE member.corpus_snapshot_id = ?
                  AND member.membership_state = 'active'
                  AND source.library_id = ?
                  AND fragment.source_fragment_id IN ({placeholders})
                """,
                (generation.corpus_snapshot_id, self.library_id, *chunk),
            ).fetchall()
            for row in rows:
                fragment_id = _text(row[0], "vector fragment ID")
                value = (
                    _text(row[1], "persisted fragment text hash"),
                    _text(row[2], "persisted fragment Source ID"),
                )
                persisted_fragments[fragment_id] = value
        if set(persisted_fragments) != set(expected_manifest):
            raise PersistenceIntegrityError(
                "CorpusVectorGeneration fragments are not exact active snapshot members"
            )
        vectors_by_fragment = {item.source_fragment_id: item for item in vectors}
        for fragment_id, manifest_item in expected_manifest.items():
            persisted_text_hash, persisted_source_id = persisted_fragments[fragment_id]
            vector_item = vectors_by_fragment[fragment_id]
            if (
                persisted_text_hash != manifest_item.text_sha256
                or persisted_source_id != vector_item.source_id
                or persisted_source_id not in layer_source_ids
            ):
                raise PersistenceIntegrityError(
                    "CorpusVectorGeneration Fragment text, Source, or layer closure mismatch"
                )

    def _validate_vector_manifest(
        self,
        generation: CorpusVectorGeneration,
        vectors: tuple[DenseVectorItem, ...],
    ) -> None:
        manifest = {item.source_fragment_id: item for item in generation.items}
        vector_ids = tuple(item.source_fragment_id for item in vectors)
        if len(set(vector_ids)) != len(vector_ids) or set(vector_ids) != set(manifest):
            raise PersistenceIntegrityError(
                "vector items must exactly match the generation manifest Fragment IDs"
            )
        for item in vectors:
            expected = manifest[item.source_fragment_id]
            if item.vector.dimensions != generation.dimensions:
                raise PersistenceIntegrityError(
                    "vector item dimensions do not match the generation"
                )
            if item.vector.vector_hash != expected.vector_sha256:
                raise PersistenceIntegrityError("vector item hash does not match the manifest")

    def _require_local_library(self, library_id: str, label: str) -> None:
        if library_id != self.library_id:
            raise PersistenceIntegrityError(f"{label} belongs to a different Library")


def _chunks(values: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    for offset in range(0, len(values), _CHUNK_SIZE):
        yield values[offset : offset + _CHUNK_SIZE]


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


def _boolean(value: object, label: str) -> bool:
    result = _integer(value, label)
    if result not in (0, 1):
        raise PersistenceIntegrityError(f"persisted {label} must be a SQLite boolean")
    return bool(result)


def _blob(value: object) -> bytes:
    if type(value) is not bytes:
        raise PersistenceIntegrityError("persisted vector blob must be immutable bytes")
    return value
