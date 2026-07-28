"""Durable P7 profile/vector persistence and protected-byte boundary tests."""

from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

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
from dithyramba.persistence.errors import PersistenceConflictError, PersistenceIntegrityError
from dithyramba.persistence.hybrid import (
    SQLiteHybridStore,
    _blob,
    _boolean,
    _chunks,
    _integer,
    _nonnegative_integer,
    _positive_integer,
    _require_hash,
    _text,
)
from dithyramba.persistence.recall import SQLiteRecallBackend
from dithyramba.persistence.repository import LibraryRepository, initialize_library
from dithyramba.recall.hybrid_models import (
    CorpusLayer,
    CorpusLayerItem,
    CorpusLayerProfile,
    EmbeddingModelProfile,
    FusionProfile,
    ModelRuntimeProfile,
)
from dithyramba.recall.models import QueryRequest
from dithyramba.recall.rerank import RerankerProfile
from dithyramba.recall.vector import (
    CorpusVectorGeneration,
    DenseVectorItem,
    VectorManifestItem,
    pack_normalized_vector,
)
from dithyramba.snapshots import CorpusSnapshot
from dithyramba.store.database import Store

NOW = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-21T12:00:00.000000Z"
REVISION = "c" * 40


def _clock() -> datetime:
    return NOW


@dataclass(frozen=True, slots=True)
class HybridContext:
    repository: LibraryRepository
    store: SQLiteHybridStore
    collection_id: str
    source_id: str
    fragment_ids: tuple[str, ...]
    text_hashes: dict[str, str]
    snapshot: CorpusSnapshot
    policy: AccessPolicySnapshot
    scope: RequestScope
    model: EmbeddingModelProfile
    runtime: ModelRuntimeProfile
    fusion: FusionProfile
    layers: CorpusLayerProfile
    generation: CorpusVectorGeneration
    vectors: tuple[DenseVectorItem, ...]


def _context(tmp_path: Path, *, suffix: str = "main") -> HybridContext:
    data_root = tmp_path / f"data_{suffix}"
    source_root = tmp_path / f"source_{suffix}"
    source_root.mkdir()
    repository = initialize_library(
        LibraryConfig(name=f"Hybrid {suffix}"),
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
    fragment_ids, text_hashes = _seed_source(
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
    model = EmbeddingModelProfile(
        model_id="intfloat/multilingual-e5-small",
        revision=REVISION,
        license="MIT",
        dimensions=2,
        max_tokens=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    runtime = ModelRuntimeProfile(
        sentence_transformers_version="5.6.0",
        transformers_version="4.53.2",
        torch_version="2.7.1",
    )
    fusion = FusionProfile()
    layers = CorpusLayerProfile(
        library_id=repository.library_id,
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        exclusion_hash=scope.exclusion_hash,
        permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
        items=(CorpusLayerItem(source_id=source_id, layer=CorpusLayer.PRIMARY),),
    )
    vectors_by_id = {
        fragment_ids[0]: pack_normalized_vector((1.0, 0.0)),
        fragment_ids[1]: pack_normalized_vector((0.0, 1.0)),
    }
    generation = CorpusVectorGeneration(
        library_id=repository.library_id,
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        snapshot_hash=snapshot.manifest_hash,
        access_policy_id=policy.access_policy_id,
        policy_hash=policy.policy_hash,
        exclusion_hash=scope.exclusion_hash,
        permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
        model_profile_hash=model.profile_hash,
        runtime_profile_hash=runtime.profile_hash,
        dimensions=2,
        items=tuple(
            VectorManifestItem(
                source_fragment_id=fragment_id,
                text_sha256=text_hashes[fragment_id],
                vector_sha256=vectors_by_id[fragment_id].vector_hash,
            )
            for fragment_id in reversed(fragment_ids)
        ),
    )
    vectors = tuple(
        DenseVectorItem(fragment_id, source_id, vectors_by_id[fragment_id])
        for fragment_id in reversed(fragment_ids)
    )
    store = SQLiteHybridStore(repository)
    store.persist_embedding_model_profile(model)
    store.persist_model_runtime_profile(runtime)
    store.persist_fusion_profile(fusion)
    store.persist_corpus_layer_profile(layers)
    store.persist_corpus_vector_generation(generation, vectors)
    return HybridContext(
        repository=repository,
        store=store,
        collection_id=collection.config.collection_id,
        source_id=source_id,
        fragment_ids=fragment_ids,
        text_hashes=text_hashes,
        snapshot=snapshot,
        policy=policy,
        scope=scope,
        model=model,
        runtime=runtime,
        fusion=fusion,
        layers=layers,
        generation=generation,
        vectors=vectors,
    )


def _seed_source(
    repository: LibraryRepository,
    *,
    collection_id: str,
    suffix: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
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
    fragment_ids: list[str] = []
    text_hashes: dict[str, str] = {}
    for ordinal, text in enumerate(texts):
        fragment_id = f"fragment_{suffix}_{ordinal}"
        text_hash = sha256_hex(text.encode())
        address = {
            "kind": "markdown",
            "heading_path": [],
            "paragraph_index": ordinal,
        }
        connection.execute(
            "INSERT INTO source_fragments VALUES (?, ?, ?, 'paragraph', ?, ?, ?, ?)",
            (
                fragment_id,
                version_id,
                ordinal,
                text,
                text_hash,
                canonical_json_bytes(address).decode(),
                canonical_sha256_hex(address),
            ),
        )
        fragment_ids.append(fragment_id)
        text_hashes[fragment_id] = text_hash
    connection.execute(
        "INSERT INTO collection_memberships VALUES (?, ?, 'active', ?)",
        (collection_id, source_id, NOW_TEXT),
    )
    return tuple(fragment_ids), text_hashes


def _close(*contexts: HybridContext) -> None:
    for context in contexts:
        context.repository.close()


def _reranker_profile() -> RerankerProfile:
    return RerankerProfile(
        model_id="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        revision="d" * 40,
        license="Apache-2.0",
        max_tokens=512,
    )


def _drop_append_trigger(
    connection: sqlite3.Connection,
    table: str,
    operation: str,
) -> None:
    assert table in {
        "embedding_model_profiles",
        "model_runtime_profiles",
        "fusion_profiles",
        "corpus_layer_items",
        "corpus_vector_generations",
        "corpus_vector_items",
    }
    assert operation in {"update", "delete"}
    connection.execute(f"DROP TRIGGER {table}_no_{operation}")


def test_exact_round_trip_is_idempotent_sorted_and_append_only(tmp_path: Path) -> None:
    context = _context(tmp_path)
    reranker = _reranker_profile()
    try:
        assert context.store.library_id == context.repository.library_id
        assert context.store.persist_embedding_model_profile(context.model) == context.model
        assert context.store.persist_model_runtime_profile(context.runtime) == context.runtime
        assert context.store.persist_fusion_profile(context.fusion) == context.fusion
        assert context.store.persist_reranker_profile(reranker) == reranker
        assert context.store.persist_corpus_layer_profile(context.layers) == context.layers

        generation, vectors = context.store.persist_corpus_vector_generation(
            context.generation,
            context.vectors,
        )
        assert generation == context.generation
        assert tuple(item.source_fragment_id for item in vectors) == context.fragment_ids
        assert tuple(item.vector.blob for item in vectors) == (
            pack_normalized_vector((1.0, 0.0)).blob,
            pack_normalized_vector((0.0, 1.0)).blob,
        )
        assert (
            context.store.load_embedding_model_profile(context.model.profile_hash) == context.model
        )
        assert (
            context.store.load_model_runtime_profile(context.runtime.profile_hash)
            == context.runtime
        )
        assert context.store.load_fusion_profile(context.fusion.profile_hash) == context.fusion
        assert context.store.load_reranker_profile(reranker.profile_hash) == reranker
        assert (
            context.store.load_corpus_layer_profile(context.layers.profile_hash) == context.layers
        )
        assert context.store.load_corpus_vector_generation(context.generation.generation_hash) == (
            context.generation,
            vectors,
        )

        connection = context.repository._store.connection
        for table, expected in (
            ("embedding_model_profiles", 1),
            ("model_runtime_profiles", 1),
            ("fusion_profiles", 1),
            ("reranker_profiles", 1),
            ("corpus_layer_profiles", 1),
            ("corpus_layer_items", 1),
            ("corpus_vector_generations", 1),
            ("corpus_vector_items", 2),
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE fusion_profiles SET created_at = created_at")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE reranker_profiles SET created_at = created_at")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM corpus_vector_generations")
    finally:
        _close(context)


def test_vector_blob_and_fragment_text_have_independent_narrow_read_gates(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        connection = context.repository._store.connection
        assert connection.execute(
            "SELECT vector_sha256 FROM corpus_vector_items ORDER BY source_fragment_id"
        ).fetchall()
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("SELECT vector_blob FROM corpus_vector_items").fetchall()
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("SELECT text FROM source_fragments").fetchall()

        with context.repository._permit_fragment_text_read():
            assert connection.execute("SELECT text FROM source_fragments").fetchall()
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute("SELECT vector_blob FROM corpus_vector_items").fetchall()
        with context.repository._permit_vector_blob_read():
            assert connection.execute("SELECT vector_blob FROM corpus_vector_items").fetchall()
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute("SELECT text FROM source_fragments").fetchall()

        loaded = context.store.load_corpus_vector_generation(context.generation.generation_hash)
        assert loaded[0] == context.generation
        assert context.repository._vector_blob_read_depth == 0
        assert context.repository._fragment_text_read_depth == 0
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("SELECT vector_blob FROM corpus_vector_items").fetchall()
    finally:
        _close(context)


def test_empty_generation_round_trips_without_dummy_blob_or_source_text(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        exclusions = QueryExclusions(source_ids=(context.source_id,))
        scope = RequestScope(
            library_id=context.repository.library_id,
            snapshot_hash=context.snapshot.manifest_hash,
            purpose="research",
            collection_ids=(context.collection_id,),
            exclusions=exclusions,
        )
        authorization = context.repository.authorize_read(
            access_policy_id=context.policy.access_policy_id,
            scope=scope,
        )
        assert authorization.compiled.manifest.items == ()
        layers = CorpusLayerProfile(
            library_id=context.repository.library_id,
            corpus_snapshot_id=context.snapshot.corpus_snapshot_id,
            exclusion_hash=scope.exclusion_hash,
            permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
            items=(),
        )
        context.store.persist_corpus_layer_profile(layers)
        generation = CorpusVectorGeneration(
            library_id=context.repository.library_id,
            corpus_snapshot_id=context.snapshot.corpus_snapshot_id,
            snapshot_hash=context.snapshot.manifest_hash,
            access_policy_id=context.policy.access_policy_id,
            policy_hash=context.policy.policy_hash,
            exclusion_hash=scope.exclusion_hash,
            permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
            model_profile_hash=context.model.profile_hash,
            runtime_profile_hash=context.runtime.profile_hash,
            dimensions=context.model.dimensions,
            items=(),
        )

        assert context.store.persist_corpus_vector_generation(generation, ()) == (generation, ())
        assert context.store.load_corpus_vector_generation(generation.generation_hash) == (
            generation,
            (),
        )
        assert context.store.persist_corpus_vector_generation(generation, ()) == (generation, ())
        connection = context.repository._store.connection
        assert (
            connection.execute(
                "SELECT item_count FROM corpus_vector_generations WHERE generation_hash = ?",
                (generation.generation_hash,),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM corpus_vector_items WHERE corpus_vector_generation_id = ?",
                (generation.generation_id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        _close(context)


def test_cross_library_scope_is_rejected_and_global_profiles_coexist(tmp_path: Path) -> None:
    first = _context(tmp_path, suffix="first")
    second = _context(tmp_path, suffix="second")
    try:
        assert first.model.profile_hash == second.model.profile_hash
        assert second.store.persist_embedding_model_profile(first.model) == first.model
        with pytest.raises(PersistenceIntegrityError, match="different Library"):
            second.store.persist_corpus_layer_profile(first.layers)
        with pytest.raises(PersistenceIntegrityError, match="different Library"):
            second.store.persist_corpus_vector_generation(first.generation, first.vectors)
        with pytest.raises(PersistenceIntegrityError, match="does not exist"):
            second.store.load_corpus_layer_profile(first.layers.profile_hash)
        with pytest.raises(PersistenceIntegrityError, match="does not exist"):
            second.store.load_corpus_vector_generation(first.generation.generation_hash)
    finally:
        _close(first, second)


@pytest.mark.parametrize(
    ("kind", "update_sql", "message"),
    [
        (
            "embedding",
            "UPDATE embedding_model_profiles SET model_id = 'owner/changed'",
            "identity mismatch",
        ),
        (
            "embedding",
            "UPDATE embedding_model_profiles SET provider = 'invalid'",
            "is invalid",
        ),
        (
            "runtime",
            "UPDATE model_runtime_profiles SET batch_size = 17",
            "identity mismatch",
        ),
        (
            "runtime",
            "UPDATE model_runtime_profiles SET device = 'invalid'",
            "is invalid",
        ),
        (
            "fusion",
            "UPDATE fusion_profiles SET created_at = 'bad'",
            "timestamp",
        ),
        (
            "fusion",
            "UPDATE fusion_profiles SET algorithm = 'invalid'",
            "is invalid",
        ),
        (
            "layer",
            "UPDATE corpus_layer_items SET layer = 'derived'",
            "identity mismatch",
        ),
        (
            "layer",
            "UPDATE corpus_layer_items SET layer = 'invalid'",
            "is invalid",
        ),
    ],
)
def test_profile_loaders_fail_closed_on_relational_or_hash_corruption(
    tmp_path: Path,
    kind: str,
    update_sql: str,
    message: str,
) -> None:
    context = _context(tmp_path)
    try:
        connection = context.repository._store.connection
        tables = {
            "embedding": "embedding_model_profiles",
            "runtime": "model_runtime_profiles",
            "fusion": "fusion_profiles",
            "layer": "corpus_layer_items",
        }
        _drop_append_trigger(connection, tables[kind], "update")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(update_sql)
        loaders = {
            "embedding": (
                context.store.load_embedding_model_profile,
                context.model.profile_hash,
            ),
            "runtime": (
                context.store.load_model_runtime_profile,
                context.runtime.profile_hash,
            ),
            "fusion": (context.store.load_fusion_profile, context.fusion.profile_hash),
            "layer": (
                context.store.load_corpus_layer_profile,
                context.layers.profile_hash,
            ),
        }
        loader, content_hash = loaders[kind]
        with pytest.raises(PersistenceIntegrityError, match=message):
            loader(content_hash)
    finally:
        _close(context)


@pytest.mark.parametrize(
    ("update_sql", "parameters", "message"),
    [
        (
            "UPDATE corpus_vector_items SET vector_blob = ? WHERE source_fragment_id LIKE ?",
            (sqlite3.Binary(pack_normalized_vector((-1.0, 0.0)).blob), "%_0"),
            "invalid",
        ),
        (
            "UPDATE corpus_vector_items SET text_sha256 = ? WHERE source_fragment_id LIKE ?",
            ("d" * 64, "%_0"),
            "identity mismatch",
        ),
        (
            "UPDATE corpus_vector_generations SET item_count = 3",
            (),
            "item count mismatch",
        ),
        (
            "UPDATE corpus_vector_generations SET created_at = 'bad'",
            (),
            "timestamp",
        ),
    ],
)
def test_generation_load_fails_closed_on_bytes_manifest_count_or_time_corruption(
    tmp_path: Path,
    update_sql: str,
    parameters: tuple[object, ...],
    message: str,
) -> None:
    context = _context(tmp_path)
    try:
        connection = context.repository._store.connection
        table = (
            "corpus_vector_items"
            if update_sql.startswith("UPDATE corpus_vector_items")
            else "corpus_vector_generations"
        )
        _drop_append_trigger(connection, table, "update")
        if "vector_blob" in update_sql:
            with context.repository._permit_vector_blob_read():
                connection.execute(update_sql, parameters)
        else:
            connection.execute(update_sql, parameters)
        with pytest.raises(PersistenceIntegrityError, match=message):
            context.store.load_corpus_vector_generation(context.generation.generation_hash)
        assert context.repository._vector_blob_read_depth == 0
    finally:
        _close(context)


def test_generation_persist_rejects_manifest_vector_and_relational_mismatches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        with pytest.raises(PersistenceIntegrityError, match="exactly match"):
            context.store.persist_corpus_vector_generation(
                context.generation,
                context.vectors[:1],
            )
        duplicate = (context.vectors[0], context.vectors[0])
        with pytest.raises(PersistenceIntegrityError, match="exactly match"):
            context.store.persist_corpus_vector_generation(context.generation, duplicate)

        wrong_vector_hash = CorpusVectorGeneration(
            **{
                **context.generation.model_dump(),
                "items": tuple(
                    VectorManifestItem(
                        source_fragment_id=item.source_fragment_id,
                        text_sha256=item.text_sha256,
                        vector_sha256=("e" * 64 if index == 0 else item.vector_sha256),
                    )
                    for index, item in enumerate(context.generation.items)
                ),
            }
        )
        with pytest.raises(PersistenceIntegrityError, match="hash"):
            context.store.persist_corpus_vector_generation(wrong_vector_hash, context.vectors)

        wrong_dimensions = CorpusVectorGeneration(
            **{**context.generation.model_dump(), "dimensions": 1}
        )
        with pytest.raises(PersistenceIntegrityError, match="dimensions"):
            context.store.persist_corpus_vector_generation(
                wrong_dimensions,
                context.vectors,
            )

        missing_fragment_id = "fragment_missing"
        missing_vector = DenseVectorItem(
            missing_fragment_id,
            context.source_id,
            context.vectors[0].vector,
        )
        kept_vector = next(
            item
            for item in context.vectors
            if item.source_fragment_id == context.generation.items[0].source_fragment_id
        )
        missing_fragment_generation = CorpusVectorGeneration(
            **{
                **context.generation.model_dump(),
                "items": (
                    context.generation.items[0],
                    VectorManifestItem(
                        source_fragment_id=missing_fragment_id,
                        text_sha256="9" * 64,
                        vector_sha256=missing_vector.vector.vector_hash,
                    ),
                ),
            }
        )
        with pytest.raises(PersistenceIntegrityError, match="active snapshot members"):
            context.store.persist_corpus_vector_generation(
                missing_fragment_generation,
                (kept_vector, missing_vector),
            )

        wrong_source_vectors = tuple(
            DenseVectorItem(item.source_fragment_id, "source_forged", item.vector)
            for item in context.vectors
        )
        with pytest.raises(PersistenceIntegrityError, match="closure mismatch"):
            context.store.persist_corpus_vector_generation(
                context.generation,
                wrong_source_vectors,
            )

        wrong_text = CorpusVectorGeneration(
            **{
                **context.generation.model_dump(),
                "items": tuple(
                    VectorManifestItem(
                        source_fragment_id=item.source_fragment_id,
                        text_sha256=("f" * 64 if index == 0 else item.text_sha256),
                        vector_sha256=item.vector_sha256,
                    )
                    for index, item in enumerate(context.generation.items)
                ),
            }
        )
        with pytest.raises(PersistenceIntegrityError, match="closure mismatch"):
            context.store.persist_corpus_vector_generation(wrong_text, context.vectors)

        wrong_snapshot = CorpusVectorGeneration(
            **{**context.generation.model_dump(), "snapshot_hash": "a" * 64}
        )
        with pytest.raises(PersistenceIntegrityError, match="snapshot closure"):
            context.store.persist_corpus_vector_generation(wrong_snapshot, context.vectors)
        wrong_policy = CorpusVectorGeneration(
            **{**context.generation.model_dump(), "policy_hash": "b" * 64}
        )
        with pytest.raises(PersistenceIntegrityError, match="policy closure"):
            context.store.persist_corpus_vector_generation(wrong_policy, context.vectors)
        missing_permitted = CorpusVectorGeneration(
            **{**context.generation.model_dump(), "permitted_set_hash": "d" * 64}
        )
        with pytest.raises(PersistenceIntegrityError, match="closure is absent"):
            context.store.persist_corpus_vector_generation(missing_permitted, context.vectors)

        one_dimension = EmbeddingModelProfile(
            model_id="owner/one-dimension",
            revision="a" * 40,
            license="MIT",
            dimensions=1,
            max_tokens=16,
        )
        context.store.persist_embedding_model_profile(one_dimension)
        wrong_model = CorpusVectorGeneration(
            **{
                **context.generation.model_dump(),
                "model_profile_hash": one_dimension.profile_hash,
            }
        )
        with pytest.raises(PersistenceIntegrityError, match="model dimensions"):
            context.store.persist_corpus_vector_generation(wrong_model, context.vectors)
    finally:
        _close(context)


def test_layer_requires_active_sources_in_its_exact_snapshot(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        connection = context.repository._store.connection
        connection.execute(
            "INSERT INTO sources VALUES ('source_outside', ?, 'file:///outside.md', "
            "'text/markdown', 'Outside', ?)",
            (context.repository.library_id, NOW_TEXT),
        )
        invalid = CorpusLayerProfile(
            library_id=context.repository.library_id,
            corpus_snapshot_id=context.snapshot.corpus_snapshot_id,
            exclusion_hash="1" * 64,
            permitted_set_hash="2" * 64,
            items=(
                CorpusLayerItem(
                    source_id="source_outside",
                    layer=CorpusLayer.UNCLASSIFIED,
                ),
            ),
        )
        with pytest.raises(PersistenceIntegrityError, match="active members"):
            context.store.persist_corpus_layer_profile(invalid)
    finally:
        _close(context)


def test_existing_partial_rows_are_rejected_not_repaired(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        connection = context.repository._store.connection
        _drop_append_trigger(connection, "corpus_vector_items", "delete")
        connection.execute(
            "DELETE FROM corpus_vector_items WHERE source_fragment_id = ?",
            (context.fragment_ids[0],),
        )
        with pytest.raises(PersistenceIntegrityError, match="item count mismatch"):
            context.store.persist_corpus_vector_generation(
                context.generation,
                context.vectors,
            )
        assert connection.execute("SELECT COUNT(*) FROM corpus_vector_items").fetchone()[0] == 1

        _drop_append_trigger(connection, "corpus_layer_items", "delete")
        connection.execute(
            "DELETE FROM corpus_layer_items WHERE corpus_layer_profile_id = ?",
            (context.layers.profile_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="item count mismatch"):
            context.store.persist_corpus_layer_profile(context.layers)
        assert connection.execute("SELECT COUNT(*) FROM corpus_layer_items").fetchone()[0] == 0
    finally:
        _close(context)


def test_p7_rows_coexist_with_frozen_vs0_query_persistence(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        backend = SQLiteRecallBackend(context.repository)
        request = QueryRequest(
            question="What is the evidence?",
            library_id=context.repository.library_id,
            collection_ids=(context.collection_id,),
            corpus_snapshot_id=context.snapshot.corpus_snapshot_id,
            access_policy_id=context.policy.access_policy_id,
            purpose="research",
        )
        backend.persist_query_request(request)
        assert backend.load_query_request(request.query_request_id) == request
        assert (
            context.store.load_corpus_vector_generation(context.generation.generation_hash)[0]
            == context.generation
        )
        connection = context.repository._store.connection
        assert connection.execute("SELECT COUNT(*) FROM query_requests").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM hybrid_query_requests").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM corpus_vector_generations").fetchone()[0] == 1
        )
    finally:
        _close(context)


def test_constructor_types_bad_hashes_and_vector_input_types_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="LibraryRepository"):
        SQLiteHybridStore(cast(Any, object()))

    context = _context(tmp_path)
    try:
        with pytest.raises(TypeError, match="EmbeddingModelProfile"):
            context.store.persist_embedding_model_profile(cast(Any, object()))
        with pytest.raises(TypeError, match="ModelRuntimeProfile"):
            context.store.persist_model_runtime_profile(cast(Any, object()))
        with pytest.raises(TypeError, match="FusionProfile"):
            context.store.persist_fusion_profile(cast(Any, object()))
        with pytest.raises(TypeError, match="RerankerProfile"):
            context.store.persist_reranker_profile(cast(Any, object()))
        with pytest.raises(TypeError, match="CorpusLayerProfile"):
            context.store.persist_corpus_layer_profile(cast(Any, object()))
        with pytest.raises(TypeError, match="CorpusVectorGeneration"):
            context.store.persist_corpus_vector_generation(cast(Any, object()), ())
        with pytest.raises(TypeError, match="tuple"):
            context.store.persist_corpus_vector_generation(
                context.generation,
                cast(Any, list(context.vectors)),
            )
        with pytest.raises(TypeError, match="DenseVectorItem"):
            context.store.persist_corpus_vector_generation(
                context.generation,
                cast(Any, (object(),)),
            )
        for loader in (
            context.store.load_embedding_model_profile,
            context.store.load_model_runtime_profile,
            context.store.load_fusion_profile,
            context.store.load_reranker_profile,
            context.store.load_corpus_layer_profile,
            context.store.load_corpus_vector_generation,
        ):
            with pytest.raises(PersistenceIntegrityError, match="SHA-256"):
                loader("bad")
            with pytest.raises(PersistenceIntegrityError, match="does not exist"):
                loader("0" * 64)
    finally:
        _close(context)


def test_defensive_content_collision_guards_reject_inexact_loader_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    try:
        other_model = EmbeddingModelProfile(
            model_id="owner/other",
            revision="a" * 40,
            license="MIT",
            dimensions=3,
            max_tokens=16,
        )
        other_runtime = ModelRuntimeProfile(
            sentence_transformers_version="1",
            transformers_version="1",
            torch_version="1",
            batch_size=1,
        )
        other_layers = CorpusLayerProfile(
            library_id=context.repository.library_id,
            corpus_snapshot_id=context.snapshot.corpus_snapshot_id,
            exclusion_hash=context.layers.exclusion_hash,
            permitted_set_hash=context.layers.permitted_set_hash,
            items=(),
        )

        with monkeypatch.context() as patch:
            patch.setattr(
                SQLiteHybridStore,
                "_load_embedding_model_profile",
                lambda _self, _connection, _hash: other_model,
            )
            with pytest.raises(PersistenceIntegrityError, match="conflicts"):
                context.store.persist_embedding_model_profile(context.model)
        with monkeypatch.context() as patch:
            patch.setattr(
                SQLiteHybridStore,
                "_load_model_runtime_profile",
                lambda _self, _connection, _hash: other_runtime,
            )
            with pytest.raises(PersistenceIntegrityError, match="conflicts"):
                context.store.persist_model_runtime_profile(context.runtime)
        with monkeypatch.context() as patch:
            patch.setattr(
                SQLiteHybridStore,
                "_load_fusion_profile",
                lambda _self, _connection, _hash: cast(FusionProfile, object()),
            )
            with pytest.raises(PersistenceIntegrityError, match="conflicts"):
                context.store.persist_fusion_profile(context.fusion)
        with monkeypatch.context() as patch:
            patch.setattr(
                SQLiteHybridStore,
                "_load_corpus_layer_profile",
                lambda _self, _connection, _hash: other_layers,
            )
            with pytest.raises(PersistenceIntegrityError, match="conflicts"):
                context.store.persist_corpus_layer_profile(context.layers)
        with monkeypatch.context() as patch:
            patch.setattr(
                SQLiteHybridStore,
                "_load_corpus_vector_generation",
                lambda _self, _connection, _hash: (context.generation, ()),
            )
            with pytest.raises(PersistenceIntegrityError, match="conflicts"):
                context.store.persist_corpus_vector_generation(
                    context.generation,
                    context.vectors,
                )
    finally:
        _close(context)


def test_sqlite_integrity_failures_are_typed_for_every_append_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)

    class ConflictContext(AbstractContextManager[sqlite3.Connection]):
        def __enter__(self) -> sqlite3.Connection:
            raise sqlite3.IntegrityError("forced conflict")

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object,
        ) -> None:
            return None

    def conflicting_transaction(
        _store: Store,
        *,
        immediate: bool = False,
    ) -> AbstractContextManager[sqlite3.Connection]:
        del immediate
        return ConflictContext()

    try:
        monkeypatch.setattr(Store, "transaction", conflicting_transaction)
        operations = (
            lambda: context.store.persist_embedding_model_profile(context.model),
            lambda: context.store.persist_model_runtime_profile(context.runtime),
            lambda: context.store.persist_fusion_profile(context.fusion),
            lambda: context.store.persist_corpus_layer_profile(context.layers),
            lambda: context.store.persist_corpus_vector_generation(
                context.generation,
                context.vectors,
            ),
        )
        for operation in operations:
            with pytest.raises(PersistenceConflictError, match="conflicted"):
                operation()
    finally:
        _close(context)


def test_scalar_decoders_and_chunker_reject_noncanonical_persisted_types() -> None:
    assert tuple(_chunks(tuple(str(index) for index in range(501)))) == (
        tuple(str(index) for index in range(500)),
        ("500",),
    )
    assert _require_hash("a" * 64, "hash") == "a" * 64
    assert _text("value", "text") == "value"
    assert _integer(1, "integer") == 1
    assert _positive_integer(1, "positive") == 1
    assert _nonnegative_integer(0, "nonnegative") == 0
    assert _boolean(1, "boolean") is True
    assert _boolean(0, "boolean") is False
    assert _blob(b"bytes") == b"bytes"

    for value in (None, "A" * 64):
        with pytest.raises(PersistenceIntegrityError, match="SHA-256"):
            _require_hash(value, "hash")
    with pytest.raises(PersistenceIntegrityError, match="text"):
        _text(1, "text")
    with pytest.raises(PersistenceIntegrityError, match="integer"):
        _integer(True, "integer")
    with pytest.raises(PersistenceIntegrityError, match="positive"):
        _positive_integer(0, "positive")
    with pytest.raises(PersistenceIntegrityError, match="nonnegative"):
        _nonnegative_integer(-1, "nonnegative")
    with pytest.raises(PersistenceIntegrityError, match="SQLite boolean"):
        _boolean(2, "boolean")
    with pytest.raises(PersistenceIntegrityError, match="immutable bytes"):
        _blob(bytearray(b"bytes"))
