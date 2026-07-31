from __future__ import annotations

import dataclasses
import os
import sqlite3
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import dithyramba.store as store_package
from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    CompiledAccess,
    PolicyEffect,
    RequestScope,
)
from dithyramba.collections import (
    CollectionConfig,
    CollectionKind,
    build_collection_root,
)
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.library import (
    LibraryConfig,
    LibraryLayoutError,
    create_library_layout,
    library_paths,
    validate_library_layout,
)
from dithyramba.library import paths as library_paths_module
from dithyramba.persistence import (
    AccessPolicyNotFoundError,
    AuthorizationError,
    BackupError,
    CollectionNotFoundError,
    LibraryNotFoundError,
    OutboxExportError,
    PersistenceConflictError,
    PersistenceIntegrityError,
    initialize_library,
    library_logical_identity_hash,
    list_libraries,
    open_library,
)
from dithyramba.persistence import repository as repository_module
from dithyramba.store import Migration, MigrationRunner, MigrationStateError, Store, StoreOpenError
from dithyramba.store import database as database_module
from dithyramba.store import migrations as migrations_module

NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-20T12:00:00.000000Z"
HASH_A = "a" * 64
HASH_B = "b" * 64


class EventIds:
    def __init__(self) -> None:
        self._next = 0

    def __call__(self) -> str:
        event_id = f"event_edge_{self._next:04d}"
        self._next += 1
        return event_id


def fixed_clock() -> datetime:
    return NOW


def _create_collection(
    repository: repository_module.LibraryRepository,
    tmp_path: Path,
    *,
    name: str = "Research corpus",
) -> CollectionConfig:
    source_root = tmp_path / f"source-{uuid.uuid4().hex}"
    source_root.mkdir()
    config = CollectionConfig(
        library_id=repository.library_id,
        name=name,
        kind=CollectionKind.CORPUS,
        roots=(
            build_collection_root(source_root, data_root=repository.paths.application_data_root),
        ),
    )
    return repository.create_collection(config).config


def _policy(
    library_id: str, collection_id: str, *, policy_id: str = "policy_research"
) -> AccessPolicySnapshot:
    return AccessPolicySnapshot(
        access_policy_id=policy_id,
        library_id=library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
    )


def _insert_snapshot_fixture(
    repository: repository_module.LibraryRepository,
    *,
    library_id: str,
    collection_id: str,
    suffix: str = "edge",
    text: str = "Grounded evidence",
) -> str:
    connection = repository._store.connection
    content_hash = sha256_hex(text.encode("utf-8"))
    source_id = f"source_{suffix}"
    source_version_id = f"source_version_{suffix}"
    fragment_id = f"fragment_{suffix}"
    address = {"kind": "markdown", "heading_path": [], "paragraph_index": 0}
    address_json = canonical_json_bytes(address).decode("utf-8")
    connection.execute(
        "INSERT INTO blobs VALUES (?, ?, ?, ?)",
        (content_hash, len(text.encode("utf-8")), f"{suffix}/blob", NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
        (source_id, library_id, f"file:///{suffix}.md", "text/markdown", suffix, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_version_id,
            source_id,
            1,
            content_hash,
            len(text.encode("utf-8")),
            NOW_TEXT,
            None,
            "markdown_v1",
            "processed",
            None,
        ),
    )
    connection.execute(
        "INSERT INTO source_heads VALUES (?, ?, ?, ?)",
        (source_id, source_version_id, NOW_TEXT, None),
    )
    connection.execute(
        "INSERT INTO source_families VALUES (?, ?, ?, ?)",
        (f"family_{suffix}", library_id, suffix, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_family_members VALUES (?, ?, 'root', NULL, ?)",
        (f"family_{suffix}", source_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_fragments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            fragment_id,
            source_version_id,
            0,
            "paragraph",
            text,
            content_hash,
            address_json,
            canonical_sha256_hex(address),
        ),
    )
    connection.execute(
        "INSERT INTO collection_memberships VALUES (?, ?, 'active', ?)",
        (collection_id, source_id, NOW_TEXT),
    )
    return repository.freeze_snapshot((collection_id,)).manifest_hash


def _authorized_repository(
    tmp_path: Path,
) -> tuple[repository_module.LibraryRepository, CollectionConfig, Any]:
    library = LibraryConfig(name="Authorization")
    repository = initialize_library(
        library,
        data_root=tmp_path / "data",
        clock=fixed_clock,
        event_id_factory=EventIds(),
    )
    collection = _create_collection(repository, tmp_path)
    snapshot_hash = _insert_snapshot_fixture(
        repository,
        library_id=library.library_id,
        collection_id=collection.collection_id,
    )
    policy = _policy(library.library_id, collection.collection_id)
    repository.persist_access_policy(name="Research", snapshot=policy)
    scope = RequestScope(
        library_id=library.library_id,
        snapshot_hash=snapshot_hash,
        purpose="research",
        collection_ids=(collection.collection_id,),
    )
    authorization = repository.authorize_read(
        access_policy_id=policy.access_policy_id,
        scope=scope,
    )
    return repository, collection, authorization


def _authorized_large_repository(
    tmp_path: Path,
    *,
    fragment_count: int = 450,
) -> tuple[repository_module.LibraryRepository, Any, tuple[str, ...]]:
    library = LibraryConfig(name="Large authorization")
    repository = initialize_library(
        library,
        data_root=tmp_path / "data",
        clock=fixed_clock,
        event_id_factory=EventIds(),
    )
    collection = _create_collection(repository, tmp_path)
    connection = repository._store.connection
    source_id = "source_large_manifest"
    source_version_id = "source_version_large_manifest"
    family_id = "family_large_manifest"
    source_bytes = b"large manifest source"
    content_hash = sha256_hex(source_bytes)
    connection.execute(
        "INSERT INTO blobs VALUES (?, ?, ?, ?)",
        (content_hash, len(source_bytes), "large/blob", NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
        (
            source_id,
            library.library_id,
            "file:///large.md",
            "text/markdown",
            "Large manifest",
            NOW_TEXT,
        ),
    )
    connection.execute(
        "INSERT INTO source_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_version_id,
            source_id,
            1,
            content_hash,
            len(source_bytes),
            NOW_TEXT,
            None,
            "markdown_v1",
            "processed",
            None,
        ),
    )
    connection.execute(
        "INSERT INTO source_heads VALUES (?, ?, ?, ?)",
        (source_id, source_version_id, NOW_TEXT, None),
    )
    connection.execute(
        "INSERT INTO source_families VALUES (?, ?, ?, ?)",
        (family_id, library.library_id, "Large manifest", NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_family_members VALUES (?, ?, 'root', NULL, ?)",
        (family_id, source_id, NOW_TEXT),
    )
    fragment_ids: list[str] = []
    fragment_rows: list[tuple[object, ...]] = []
    for index in range(fragment_count):
        fragment_id = f"fragment_large_{index:04d}"
        text = f"Mars evidence paragraph {index:04d}"
        address = {
            "kind": "markdown",
            "heading_path": [],
            "paragraph_index": index,
        }
        fragment_ids.append(fragment_id)
        fragment_rows.append(
            (
                fragment_id,
                source_version_id,
                index,
                "paragraph",
                text,
                sha256_hex(text.encode("utf-8")),
                canonical_json_bytes(address).decode("utf-8"),
                canonical_sha256_hex(address),
            )
        )
    connection.executemany(
        "INSERT INTO source_fragments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        fragment_rows,
    )
    connection.execute(
        "INSERT INTO collection_memberships VALUES (?, ?, 'active', ?)",
        (collection.collection_id, source_id, NOW_TEXT),
    )
    snapshot = repository.freeze_snapshot((collection.collection_id,))
    policy = _policy(library.library_id, collection.collection_id)
    repository.persist_access_policy(name="Research", snapshot=policy)
    scope = RequestScope(
        library_id=library.library_id,
        snapshot_hash=snapshot.manifest_hash,
        purpose="research",
        collection_ids=(collection.collection_id,),
    )
    authorization = repository.authorize_read(
        access_policy_id=policy.access_policy_id,
        scope=scope,
    )
    return repository, authorization, tuple(fragment_ids)


def test_initialization_failure_closes_store_and_removes_only_new_library(tmp_path: Path) -> None:
    config = LibraryConfig(name="Invalid clock")
    paths = library_paths(config.library_id, data_root=tmp_path / "data")

    with pytest.raises(PersistenceIntegrityError, match="timezone-aware"):
        initialize_library(
            config,
            data_root=tmp_path / "data",
            clock=lambda: datetime(2026, 7, 20, 12, 0, 0),
        )

    assert not paths.root.exists()
    assert paths.libraries_root.is_dir()


def test_open_missing_library_and_list_missing_root_are_explicit(tmp_path: Path) -> None:
    missing = LibraryConfig(name="Missing")

    assert list_libraries(data_root=tmp_path / "data") == ()
    with pytest.raises(LibraryNotFoundError, match="does not exist"):
        open_library(missing.library_id, data_root=tmp_path / "data")


def test_list_libraries_rejects_non_directory_registry_surface(tmp_path: Path) -> None:
    libraries_root = tmp_path / "data" / "libraries"
    libraries_root.parent.mkdir()
    libraries_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(PersistenceIntegrityError, match="real directory"):
        list_libraries(data_root=tmp_path / "data")


def test_list_libraries_rejects_library_symlink_and_ignores_unrelated_entries(
    tmp_path: Path,
) -> None:
    libraries_root = tmp_path / "data" / "libraries"
    libraries_root.mkdir(parents=True)
    target = tmp_path / "target"
    target.mkdir()
    (libraries_root / "notes.txt").write_text("ignore", encoding="utf-8")
    (libraries_root / "not-a-library").mkdir()
    (libraries_root / "alias").symlink_to(target, target_is_directory=True)

    assert list_libraries(data_root=tmp_path / "data") == ()

    symlink_name = LibraryConfig(name="Alias").library_id
    (libraries_root / symlink_name).symlink_to(target, target_is_directory=True)
    with pytest.raises(PersistenceIntegrityError, match="may not be a symlink"):
        list_libraries(data_root=tmp_path / "data")


@pytest.mark.parametrize(
    "case",
    ["missing-row", "invalid-id", "physical-mismatch", "hash-mismatch", "bad-timestamp"],
)
def test_open_library_revalidates_persisted_identity(tmp_path: Path, case: str) -> None:
    expected = LibraryConfig(name="Expected")
    paths = create_library_layout(expected, data_root=tmp_path / case)
    with Store.open(paths.database) as store:
        if case == "missing-row":
            pass
        else:
            persisted = expected
            library_id = expected.library_id
            identity_hash = library_logical_identity_hash(expected)
            created_at = NOW_TEXT
            if case == "invalid-id":
                library_id = "library_invalid"
                identity_hash = HASH_A
            elif case == "physical-mismatch":
                persisted = LibraryConfig(name="Other")
                library_id = persisted.library_id
                identity_hash = library_logical_identity_hash(persisted)
            elif case == "hash-mismatch":
                identity_hash = HASH_B
            elif case == "bad-timestamp":
                created_at = "2026-02-30T12:00:00.000000Z"
            store.connection.execute(
                "INSERT INTO libraries VALUES (?, ?, ?, ?)",
                (library_id, persisted.name, identity_hash, created_at),
            )

    with pytest.raises(PersistenceIntegrityError):
        open_library(expected.library_id, data_root=tmp_path / case)


def test_repository_verify_detects_changed_in_memory_identity(tmp_path: Path) -> None:
    config = LibraryConfig(name="Original")
    with initialize_library(config, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        changed_config = LibraryConfig(library_id=config.library_id, name="Changed")
        repository.library = dataclasses.replace(repository.library, config=changed_config)

        with pytest.raises(PersistenceIntegrityError, match="identity changed"):
            repository.verify()


def test_collection_operations_reject_cross_library_and_missing_ids(tmp_path: Path) -> None:
    first = LibraryConfig(name="First")
    other = LibraryConfig(name="Other")
    with initialize_library(first, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        foreign = CollectionConfig(
            library_id=other.library_id,
            name="Foreign",
            kind=CollectionKind.CORPUS,
        )
        with pytest.raises(PersistenceIntegrityError, match="different Library"):
            repository.create_collection(foreign)
        with pytest.raises(CollectionNotFoundError, match="does not exist"):
            repository.get_collection("collection_missing")


def test_collection_loader_rejects_invalid_identity_and_timestamp(tmp_path: Path) -> None:
    library = LibraryConfig(name="Collection integrity")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        repository._store.connection.execute(
            "INSERT INTO collections VALUES (?, ?, ?, ?, ?)",
            ("collection_invalid", library.library_id, "Invalid ID", "corpus", NOW_TEXT),
        )
        valid_id = "collection_" + uuid.uuid4().hex
        repository._store.connection.execute(
            "INSERT INTO collections VALUES (?, ?, ?, ?, ?)",
            (
                valid_id,
                library.library_id,
                "Bad timestamp",
                "corpus",
                "2026-02-30T12:00:00.000000Z",
            ),
        )

        with pytest.raises(PersistenceIntegrityError, match="persisted Collection"):
            repository.get_collection("collection_invalid")
        with pytest.raises(PersistenceIntegrityError, match="real date-time"):
            repository.get_collection(valid_id)


@pytest.mark.parametrize(
    "corrupt_json",
    ["not-json", "{}", '["ok",1]', '[ "not-canonical" ]'],
)
def test_collection_loader_rejects_corrupt_persisted_globs(
    tmp_path: Path, corrupt_json: str
) -> None:
    library = LibraryConfig(name="Corrupt roots")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        collection = _create_collection(repository, tmp_path)
        repository._store.connection.execute(
            "UPDATE collection_roots SET include_globs_json = ? WHERE collection_id = ?",
            (corrupt_json, collection.collection_id),
        )

        with pytest.raises(PersistenceIntegrityError):
            repository.get_collection(collection.collection_id)


def test_collection_loader_revalidates_root_presence_and_data_overlap(tmp_path: Path) -> None:
    library = LibraryConfig(name="Root revalidation")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        collection = _create_collection(repository, tmp_path)
        source_root = collection.roots[0].path
        source_root.rmdir()
        with pytest.raises(PersistenceIntegrityError, match="no longer satisfies"):
            repository.get_collection(collection.collection_id)

        repository._store.connection.execute(
            "UPDATE collection_roots SET resolved_path = ? WHERE collection_id = ?",
            (str(repository.paths.cache), collection.collection_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="no longer satisfies"):
            repository.get_collection(collection.collection_id)


def test_collection_loader_rejects_symlink_rebinding(tmp_path: Path) -> None:
    library = LibraryConfig(name="Root rebinding")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        collection = _create_collection(repository, tmp_path)
        persisted_root = collection.roots[0].path
        replacement = tmp_path / "replacement-root"
        persisted_root.rmdir()
        replacement.mkdir()
        persisted_root.symlink_to(replacement, target_is_directory=True)

        with pytest.raises(PersistenceIntegrityError, match="resolves to a different path"):
            repository.get_collection(collection.collection_id)


def test_policy_operations_reject_bad_name_cross_library_and_missing_references(
    tmp_path: Path,
) -> None:
    library = LibraryConfig(name="Policies")
    other = LibraryConfig(name="Other")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        collection = _create_collection(repository, tmp_path)
        with pytest.raises(PersistenceIntegrityError, match="unpadded"):
            repository.persist_access_policy(
                name=" padded",
                snapshot=_policy(library.library_id, collection.collection_id),
            )
        with pytest.raises(PersistenceIntegrityError, match="different Library"):
            repository.persist_access_policy(
                name="Foreign",
                snapshot=_policy(other.library_id, collection.collection_id),
            )
        with pytest.raises(CollectionNotFoundError, match="absent"):
            repository.persist_access_policy(
                name="Missing Collection",
                snapshot=_policy(library.library_id, "collection_missing"),
            )
        with pytest.raises(AccessPolicyNotFoundError, match="does not exist"):
            repository.get_access_policy("policy_missing")


def test_duplicate_access_policy_is_an_explicit_conflict(tmp_path: Path) -> None:
    library = LibraryConfig(name="Policy conflict")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        collection = _create_collection(repository, tmp_path)
        policy = _policy(library.library_id, collection.collection_id)
        repository.persist_access_policy(name="Research", snapshot=policy)

        with pytest.raises(PersistenceConflictError, match="conflicted"):
            repository.persist_access_policy(name="Research duplicate", snapshot=policy)


@pytest.mark.parametrize("case", ["invalid-policy", "hash", "name", "timestamp"])
def test_policy_loader_revalidates_semantics_hash_label_and_timestamp(
    tmp_path: Path, case: str
) -> None:
    library = LibraryConfig(name="Policy loader")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        collection = _create_collection(repository, tmp_path)
        policy_id = f"policy_{case.replace('-', '_')}"
        purposes = "[]" if case == "invalid-policy" else '["research"]'
        policy_hash = HASH_B
        name = "Research"
        created_at = NOW_TEXT
        if case == "name":
            name = " padded"
        if case == "timestamp":
            created_at = "invalid"
        repository._store.connection.execute(
            "INSERT INTO access_policies VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                policy_id,
                library.library_id,
                name,
                purposes,
                0,
                0,
                policy_hash,
                created_at,
            ),
        )
        repository._store.connection.execute(
            "INSERT INTO access_policy_collection_rules VALUES (?, ?, 'allow')",
            (policy_id, collection.collection_id),
        )

        with pytest.raises(PersistenceIntegrityError):
            repository.get_access_policy(policy_id)


def test_authorization_rejects_invalid_scope_library_collection_policy_and_snapshot(
    tmp_path: Path,
) -> None:
    library = LibraryConfig(name="Authorization boundary")
    other = LibraryConfig(name="Other")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        collection = _create_collection(repository, tmp_path)
        policy = _policy(library.library_id, collection.collection_id)
        repository.persist_access_policy(name="Research", snapshot=policy)
        valid_scope = RequestScope(
            library_id=library.library_id,
            snapshot_hash=HASH_A,
            purpose="research",
            collection_ids=(collection.collection_id,),
        )
        with pytest.raises(AuthorizationError, match="RequestScope"):
            repository.authorize_read(
                access_policy_id=policy.access_policy_id,
                scope=cast(Any, object()),
            )
        with pytest.raises(AuthorizationError, match="different Library"):
            repository.authorize_read(
                access_policy_id=policy.access_policy_id,
                scope=dataclasses.replace(valid_scope, library_id=other.library_id),
            )
        with pytest.raises(CollectionNotFoundError, match="absent"):
            repository.authorize_read(
                access_policy_id=policy.access_policy_id,
                scope=dataclasses.replace(valid_scope, collection_ids=("collection_missing",)),
            )
        with pytest.raises(AccessPolicyNotFoundError):
            repository.authorize_read(access_policy_id="policy_missing", scope=valid_scope)
        with pytest.raises(AuthorizationError, match="snapshot"):
            repository.authorize_read(
                access_policy_id=policy.access_policy_id,
                scope=valid_scope,
            )


def test_authorization_rejects_corrupt_snapshot_scope_json(tmp_path: Path) -> None:
    library = LibraryConfig(name="Snapshot JSON")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        collection = _create_collection(repository, tmp_path)
        policy = _policy(library.library_id, collection.collection_id)
        repository.persist_access_policy(name="Research", snapshot=policy)
        repository._store.connection.execute(
            "INSERT INTO corpus_snapshots VALUES (?, ?, ?, ?, ?)",
            ("snapshot_corrupt", library.library_id, "not-json", HASH_A, NOW_TEXT),
        )
        scope = RequestScope(
            library_id=library.library_id,
            snapshot_hash=HASH_A,
            purpose="research",
            collection_ids=(collection.collection_id,),
        )

        with pytest.raises(PersistenceIntegrityError, match="not valid JSON"):
            repository.authorize_read(access_policy_id=policy.access_policy_id, scope=scope)


def test_empty_snapshot_authorization_returns_no_text(tmp_path: Path) -> None:
    library = LibraryConfig(name="Empty snapshot")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        collection = _create_collection(repository, tmp_path)
        policy = _policy(library.library_id, collection.collection_id)
        repository.persist_access_policy(name="Research", snapshot=policy)
        snapshot = repository.freeze_snapshot((collection.collection_id,))
        scope = RequestScope(
            library_id=library.library_id,
            snapshot_hash=snapshot.manifest_hash,
            purpose="research",
            collection_ids=(collection.collection_id,),
        )
        authorization = repository.authorize_read(
            access_policy_id=policy.access_policy_id,
            scope=scope,
        )

        assert repository.read_permitted_fragments(authorization) == ()


def test_authorized_subset_is_chunked_without_a_false_two_thousand_item_ceiling(
    tmp_path: Path,
) -> None:
    repository, _collection, authorization = _authorized_repository(tmp_path)
    try:
        identifiers = tuple(f"fragment_probe_{index}" for index in range(2_001))
        with pytest.raises(AuthorizationError, match="outside the permitted manifest"):
            repository.read_permitted_fragment_subset(authorization, identifiers)
    finally:
        repository.close()


def test_large_authorized_subset_preserves_exact_caller_order_and_hashes(
    tmp_path: Path,
) -> None:
    repository, authorization, fragment_ids = _authorized_large_repository(tmp_path)
    try:
        requested = tuple(reversed(fragment_ids))
        fragments = repository.read_permitted_fragment_subset(authorization, requested)

        assert tuple(item.source_fragment_id for item in fragments) == requested
        assert tuple(item.text for item in fragments) == tuple(
            f"Mars evidence paragraph {index:04d}" for index in reversed(range(len(fragment_ids)))
        )
        assert all(item.text_sha256 == sha256_hex(item.text.encode("utf-8")) for item in fragments)
    finally:
        repository.close()


def test_large_authorized_subset_fails_closed_and_cleans_temporary_read_gate(
    tmp_path: Path,
) -> None:
    repository, authorization, fragment_ids = _authorized_large_repository(tmp_path)
    try:
        assert repository.read_permitted_fragment_subset(authorization, ()) == ()
        with pytest.raises(AuthorizationError, match="outside"):
            repository.read_permitted_fragment_subset(
                authorization,
                (*fragment_ids, "fragment_not_permitted"),
            )

        connection = repository._store.connection
        expected = {item.source_fragment_id: item for item in authorization.compiled.manifest.items}
        missing_family = dict(expected)
        missing_family[fragment_ids[0]] = dataclasses.replace(
            missing_family[fragment_ids[0]],
            source_family_id=None,
        )
        with pytest.raises(PersistenceIntegrityError, match="lacks SourceFamily"):
            repository._read_large_permitted_fragment_subset(
                connection,
                fragment_ids,
                missing_family,
            )

        wrong_lineage = dict(expected)
        wrong_lineage[fragment_ids[0]] = dataclasses.replace(
            wrong_lineage[fragment_ids[0]],
            source_version_id="source_version_wrong",
        )
        with pytest.raises(PersistenceIntegrityError, match="lineage differs"):
            repository._read_large_permitted_fragment_subset(
                connection,
                fragment_ids,
                wrong_lineage,
            )

        connection.execute("DROP TRIGGER source_fragments_no_update")
        connection.execute(
            "UPDATE source_fragments SET text = 'tampered' WHERE source_fragment_id = ?",
            (fragment_ids[0],),
        )
        with pytest.raises(PersistenceIntegrityError, match="text hash"):
            repository.read_permitted_fragment_subset(authorization, fragment_ids)

        assert repository._fragment_text_read_depth == 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_temp_master "
                "WHERE type = 'table' AND name = 'dithyramba_fragment_read_permit'"
            ).fetchone()[0]
            == 0
        )
    finally:
        repository.close()


def test_authorization_rejects_content_addressed_snapshot_membership_tampering(
    tmp_path: Path,
) -> None:
    repository, _collection, authorization = _authorized_repository(tmp_path)
    try:
        repository._store.connection.execute("DROP TRIGGER snapshot_members_no_update")
        repository._store.connection.execute(
            "UPDATE snapshot_members SET membership_state = 'holdout'"
        )

        with pytest.raises(PersistenceIntegrityError, match="CorpusSnapshot ID mismatch"):
            repository.read_permitted_fragments(authorization)
    finally:
        repository.close()


def test_authorization_normalizes_corrupt_membership_state(tmp_path: Path) -> None:
    repository, _collection, authorization = _authorized_repository(tmp_path)
    try:
        connection = repository._store.connection
        connection.execute("DROP TRIGGER snapshot_members_no_update")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("UPDATE snapshot_members SET membership_state = 'corrupt'")

        with pytest.raises(PersistenceIntegrityError, match="manifest is invalid"):
            repository.read_permitted_fragments(authorization)
    finally:
        repository.close()


def _freeze_recompile(
    repository: repository_module.LibraryRepository,
    compiled: CompiledAccess,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def return_original(**_kwargs: object) -> CompiledAccess:
        return compiled

    monkeypatch.setattr(repository, "_compile_db_access", return_original)


def test_authorized_read_detects_missing_fragment_after_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _collection, authorization = _authorized_repository(tmp_path)
    try:
        _freeze_recompile(repository, authorization.compiled, monkeypatch)
        repository._store.connection.execute("DROP TRIGGER source_fragment_text_metrics_no_delete")
        repository._store.connection.execute("DELETE FROM source_fragment_text_metrics")
        repository._store.connection.execute("DROP TRIGGER source_fragments_no_delete")
        repository._store.connection.execute("DELETE FROM source_fragments")

        with pytest.raises(PersistenceIntegrityError, match="exact persisted fragment set"):
            repository.read_permitted_fragments(authorization)
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("source_address_json", "not-json", "SourceAddress"),
        ("text", "Tampered evidence", "text hash mismatch"),
    ],
)
def test_authorized_read_revalidates_address_and_text_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: str,
    expected: str,
) -> None:
    repository, _collection, authorization = _authorized_repository(tmp_path)
    try:
        _freeze_recompile(repository, authorization.compiled, monkeypatch)
        repository._store.connection.execute("DROP TRIGGER source_fragments_no_update")
        repository._store.connection.execute(
            f'UPDATE source_fragments SET "{column}" = ?', (value,)
        )

        with pytest.raises(PersistenceIntegrityError, match=expected):
            repository.read_permitted_fragments(authorization)
    finally:
        repository.close()


def test_pending_outbox_revalidates_payload_hash(tmp_path: Path) -> None:
    library = LibraryConfig(name="Outbox integrity")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        repository._store.connection.execute(
            "INSERT INTO event_outbox VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                "event_corrupt",
                "corrupt.created",
                "library",
                library.library_id,
                "{}",
                HASH_A,
                NOW_TEXT,
            ),
        )

        with pytest.raises(PersistenceIntegrityError, match="payload hash mismatch"):
            repository.pending_outbox_events()


def test_outbox_export_rejects_conflicting_existing_event(tmp_path: Path) -> None:
    library = LibraryConfig(name="Outbox conflict")
    with initialize_library(
        library,
        data_root=tmp_path / "data",
        clock=fixed_clock,
        event_id_factory=EventIds(),
    ) as repository:
        event = repository.pending_outbox_events()[0]
        conflicting = {**event.envelope(), "event_type": "tampered.created"}
        repository.paths.events.write_bytes(canonical_json_bytes(conflicting) + b"\n")

        with pytest.raises(OutboxExportError, match="conflicting event ID"):
            repository.export_outbox()


def test_outbox_export_detects_concurrent_delivery_after_append(tmp_path: Path) -> None:
    library = LibraryConfig(name="Concurrent outbox")
    with initialize_library(
        library,
        data_root=tmp_path / "data",
        clock=fixed_clock,
        event_id_factory=EventIds(),
    ) as repository:

        def deliver_concurrently(event: object) -> None:
            repository._store.connection.execute(
                "UPDATE event_outbox SET delivered_at = ? WHERE event_id = ?",
                (NOW_TEXT, cast(Any, event).event_id),
            )

        with pytest.raises(OutboxExportError, match="changed during delivery"):
            repository.export_outbox(after_append=deliver_concurrently)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"partial", "partial line"),
        (b"\xff\n", "invalid canonical line"),
        (b"not-json\n", "invalid canonical line"),
        (b"[]\n", "invalid canonical line"),
        (b'{ "event_id":"event_one"}\n', "invalid canonical line"),
        (b'{"event_id":"wrong"}\n', "invalid event_id"),
        (b'{"event_id":"event_one"}\n{"event_id":"event_one"}\n', "duplicate event ID"),
    ],
)
def test_event_file_reader_rejects_partial_invalid_or_duplicate_lines(
    tmp_path: Path, payload: bytes, expected: str
) -> None:
    events = tmp_path / "events.jsonl"
    events.write_bytes(payload)
    events.chmod(0o600)

    with pytest.raises(OutboxExportError, match=expected):
        repository_module._read_event_file(events)


def test_event_file_read_and_append_os_errors_are_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "events.jsonl"
    directory.mkdir()
    with pytest.raises(OutboxExportError, match=r"cannot be opened|private regular file"):
        repository_module._read_event_file(directory)

    events = tmp_path / "real-events.jsonl"
    events.write_bytes(b"")
    events.chmod(0o600)

    def fail_open(*_args: object, **_kwargs: object) -> int:
        raise OSError("append unavailable")

    monkeypatch.setattr("dithyramba.persistence.repository.os.open", fail_open)
    with pytest.raises(OutboxExportError, match="append failed"):
        repository_module._append_and_fsync(events, b"{}\n")


def test_online_backup_rejects_active_transaction(tmp_path: Path) -> None:
    library = LibraryConfig(name="Backup transaction")
    with (
        initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository,
        repository._store.transaction(),
        pytest.raises(BackupError, match="inside a transaction"),
    ):
        repository.create_online_backup()


def test_online_backup_wraps_temp_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = LibraryConfig(name="Backup temp failure")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:

        def fail_mkstemp(**_kwargs: object) -> tuple[int, str]:
            raise OSError("no temporary files")

        monkeypatch.setattr("dithyramba.persistence.repository.tempfile.mkstemp", fail_mkstemp)
        with pytest.raises(BackupError, match="failed verification"):
            repository.create_online_backup()
        assert tuple(repository.paths.backups.iterdir()) == ()


def test_online_backup_wraps_verification_failure_and_cleans_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = LibraryConfig(name="Backup verification")
    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:

        def fail_verification(_connection: sqlite3.Connection) -> None:
            raise RuntimeError("verification failed")

        monkeypatch.setattr(repository_module, "verify_integrity", fail_verification)
        with pytest.raises(BackupError, match="failed verification"):
            repository.create_online_backup()

        assert tuple(repository.paths.backups.iterdir()) == ()


def test_online_backup_preserves_specific_stale_schema_error_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = LibraryConfig(name="Stale backup")

    class StaleRunner:
        latest_version = 1
        instances = 0

        def __init__(self, _connection: sqlite3.Connection) -> None:
            type(self).instances += 1
            self._instance = type(self).instances

        def verify(self) -> tuple[object, ...]:
            return ()

        def current_version(self) -> int:
            return 1 if self._instance == 1 else 0

    with initialize_library(library, data_root=tmp_path / "data", clock=fixed_clock) as repository:
        monkeypatch.setattr(repository_module, "MigrationRunner", StaleRunner)
        with pytest.raises(BackupError, match="schema is not current"):
            repository.create_online_backup()

        assert tuple(repository.paths.backups.iterdir()) == ()


def test_store_open_library_requires_typed_validated_paths(tmp_path: Path) -> None:
    with pytest.raises(StoreOpenError, match="requires LibraryPaths"):
        Store.open_library(cast(Any, tmp_path / "memory.sqlite3"))

    paths = create_library_layout(LibraryConfig(name="Typed paths"), data_root=tmp_path / "data")
    with Store.open_library(paths) as store:
        assert store.schema_version == 12


def test_store_rejects_missing_or_symlinked_parent(tmp_path: Path) -> None:
    with pytest.raises(StoreOpenError, match="parent must already exist"):
        Store.open(tmp_path / "missing" / "memory.sqlite3")

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(StoreOpenError, match="symlink component"):
        Store.open(alias_parent / "memory.sqlite3")


@pytest.mark.parametrize("already_exists", [False, True])
def test_store_chmod_failure_fails_closed_and_removes_only_new_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_exists: bool,
) -> None:
    database = tmp_path / "memory.sqlite3"
    if already_exists:
        database.write_bytes(b"")

    original_chmod = Path.chmod

    def fail_target_chmod(
        path: Path,
        mode: int,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        if path == database:
            raise OSError("chmod unavailable")
        original_chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "chmod", fail_target_chmod)
    with pytest.raises(StoreOpenError, match="could not secure"):
        Store.open(database)

    assert database.exists() is already_exists


def test_store_profile_verifies_every_required_pragma() -> None:
    class Result:
        def __init__(self, value: object = None) -> None:
            self.value = value

        def fetchone(self) -> tuple[object]:
            return (self.value,)

    class ProfileConnection:
        def __init__(self, *, busy_timeout: int) -> None:
            self.busy_timeout = busy_timeout

        def execute(self, sql: str) -> Result:
            values: dict[str, object] = {
                "PRAGMA foreign_keys": 1,
                "PRAGMA journal_mode": "wal",
                "PRAGMA synchronous": 2,
                "PRAGMA trusted_schema": 0,
                "PRAGMA busy_timeout": self.busy_timeout,
            }
            return Result(values.get(sql))

    database_module._apply_profile(cast(sqlite3.Connection, ProfileConnection(busy_timeout=5000)))
    with pytest.raises(StoreOpenError, match="required VS0 profile"):
        database_module._apply_profile(cast(sqlite3.Connection, ProfileConnection(busy_timeout=1)))


def test_layout_rejects_symlink_hardlink_escaped_file_and_wrong_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = create_library_layout(LibraryConfig(name="First"), data_root=tmp_path / "data")
    second = create_library_layout(LibraryConfig(name="Second"), data_root=tmp_path / "data")

    first.events.unlink()
    first.events.symlink_to(second.events)
    with pytest.raises(LibraryLayoutError, match="member escaped"):
        validate_library_layout(first)
    first.events.unlink()
    first.events.write_bytes(b"")
    first.events.chmod(0o600)

    hardlink = tmp_path / "events-hardlink"
    os.link(first.events, hardlink)
    with pytest.raises(LibraryLayoutError, match="hardlinked"):
        validate_library_layout(first)
    hardlink.unlink()

    escaped = tmp_path / "escaped-events"
    escaped.write_bytes(b"")
    escaped.chmod(0o600)
    with pytest.raises(LibraryLayoutError, match="member escaped"):
        validate_library_layout(dataclasses.replace(first, events=escaped))

    fake_metadata = SimpleNamespace(st_uid=os.geteuid() + 1, st_mode=stat.S_IFDIR | 0o700)
    monkeypatch.setattr(Path, "lstat", lambda _path: fake_metadata)
    with pytest.raises(LibraryLayoutError, match="different owner"):
        library_paths_module._require_private_posix_permissions(
            first.root,
            expected_mode=0o700,
        )


def test_layout_missing_low_level_components_are_wrapped(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(LibraryLayoutError, match="is missing"):
        library_paths_module._require_real_directory(missing, label="test directory")
    with pytest.raises(LibraryLayoutError, match="is missing"):
        library_paths_module._require_real_file(missing, label="test file")


def _migration_dir(tmp_path: Path, sql: str) -> Path:
    directory = tmp_path / f"migrations-{uuid.uuid4().hex}"
    directory.mkdir()
    (directory / "0001_test.sql").write_text(sql, encoding="utf-8")
    return directory


def test_migration_base_exception_rolls_back_active_schema_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema_sql = """
    CREATE TABLE schema_migrations(
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        applied_at TEXT NOT NULL
    );
    """
    connection = sqlite3.connect(":memory:", isolation_level=None)

    def abort_iteration(_sql: str) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(migrations_module, "_iter_sql_statements", abort_iteration)
    try:
        with pytest.raises(KeyboardInterrupt):
            MigrationRunner(connection, _migration_dir(tmp_path, schema_sql)).apply_all()
        assert connection.in_transaction is False
    finally:
        connection.close()


def test_expected_schema_fingerprint_rejects_bad_count_and_bad_trusted_sql() -> None:
    with pytest.raises(MigrationStateError, match="invalid applied migration count"):
        migrations_module._expected_schema_fingerprint((), 0)

    migration = Migration(1, "0001_bad.sql", HASH_A, "BROKEN STATEMENT;")
    with pytest.raises(MigrationStateError, match="cannot reconstruct"):
        migrations_module._expected_schema_fingerprint((migration,), 1)


def test_migration_cli_main_and_lazy_store_attribute_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert migrations_module.main(["verify"]) == 0
    payload = capsys.readouterr().out
    assert '"status":"ok"' in payload

    with pytest.raises(AttributeError, match="has no attribute"):
        store_package.__getattr__("not_a_store_symbol")


@pytest.mark.parametrize(
    ("value", "loader", "expected"),
    [
        ("[]", repository_module._load_canonical_object, "JSON object"),
        ('{ "key":1}', repository_module._load_canonical_object, "canonical JSON"),
        ("not-json", repository_module._load_canonical_string_tuple, "valid JSON"),
        ("{}", repository_module._load_canonical_string_tuple, "string array"),
        ('["ok",1]', repository_module._load_canonical_string_tuple, "string array"),
        ('[ "ok" ]', repository_module._load_canonical_string_tuple, "canonical JSON"),
    ],
)
def test_persisted_json_decoders_reject_wrong_shapes_and_noncanonical_text(
    value: str,
    loader: Any,
    expected: str,
) -> None:
    with pytest.raises(PersistenceIntegrityError, match=expected):
        loader(value, "persisted value")


@pytest.mark.parametrize("clock_value", ["not-a-date", datetime(2026, 7, 20, 12, 0, 0)])
def test_operational_clock_requires_aware_datetime(clock_value: object) -> None:
    with pytest.raises(PersistenceIntegrityError, match="timezone-aware"):
        repository_module._timestamp(lambda: cast(Any, clock_value))


@pytest.mark.parametrize("value", [None, True, 2, "1"])
def test_persisted_boolean_decoder_is_strict(value: object) -> None:
    with pytest.raises(PersistenceIntegrityError, match="0 or 1"):
        repository_module._strict_bool(value, "flag")


def test_fsync_directory_has_a_portable_noop_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr("dithyramba.persistence.repository.os.O_DIRECTORY")

    repository_module._fsync_directory(tmp_path)
