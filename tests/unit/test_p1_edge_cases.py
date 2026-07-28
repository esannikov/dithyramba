from __future__ import annotations

import dataclasses
import sqlite3
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    CompiledAccess,
    InvalidAccessPolicyError,
    InvalidAccessScopeError,
    MembershipState,
    PermittedManifest,
    PermittedManifestItem,
    PermittedToken,
    PermittedTokenError,
    PolicyEffect,
    PublicAccessResult,
    QueryExclusions,
    RequestScope,
    SnapshotCandidate,
    SnapshotMembership,
    SourceRule,
    compile_access,
    verify_permitted_token,
)
from dithyramba.collections import (
    CollectionConfig,
    CollectionKind,
    CollectionRoot,
    InvalidCollectionGlobError,
    InvalidCollectionIdError,
    validate_collection_id,
    validate_globs,
)
from dithyramba.library import (
    InvalidLibraryIdError,
    LibraryAlreadyExistsError,
    LibraryConfig,
    LibraryLayoutError,
    create_library_layout,
    library_paths,
    validate_library_id,
    validate_library_layout,
)
from dithyramba.library import paths as library_paths_module
from dithyramba.store import (
    IntegrityCheckError,
    MigrationApplyError,
    MigrationBackupRequiredError,
    MigrationDiscoveryError,
    MigrationRunner,
    MigrationStateError,
    Store,
    StoreClosedError,
    StoreOpenError,
    TransactionStateError,
    discover_migrations,
    migration_checksums,
    packaged_migration_source,
    verify_integrity,
)
from dithyramba.store import database as database_module
from dithyramba.store import migrations as migrations_module

LIBRARY = "library_alpha"
OTHER_LIBRARY = "library_beta"
COLLECTION = "collection_alpha"
SNAPSHOT_HASH = "a" * 64
OTHER_HASH = "b" * 64


def _policy() -> AccessPolicySnapshot:
    return AccessPolicySnapshot(
        access_policy_id="policy_research",
        library_id=LIBRARY,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(COLLECTION, PolicyEffect.ALLOW),),
    )


def _scope() -> RequestScope:
    return RequestScope(
        library_id=LIBRARY,
        snapshot_hash=SNAPSHOT_HASH,
        purpose="research",
        collection_ids=(COLLECTION,),
    )


def _candidate() -> SnapshotCandidate:
    return SnapshotCandidate(
        source_fragment_id="fragment_one",
        source_version_id="source_version_one",
        source_id="source_one",
        source_family_id=None,
        memberships=(SnapshotMembership(COLLECTION, MembershipState.ACTIVE),),
    )


@pytest.mark.parametrize(
    "allowed_purposes",
    [(), ("Not Valid",), ("research", "research")],
)
def test_policy_rejects_empty_invalid_or_duplicate_purposes(
    allowed_purposes: tuple[str, ...],
) -> None:
    with pytest.raises(InvalidAccessPolicyError):
        AccessPolicySnapshot(
            access_policy_id="policy_research",
            library_id=LIBRARY,
            allowed_purposes=allowed_purposes,
            collection_rules=(CollectionRule(COLLECTION, PolicyEffect.ALLOW),),
        )


@pytest.mark.parametrize("field", ["allow_export", "allow_external_provider"])
def test_policy_capability_flags_require_real_booleans(field: str) -> None:
    values: dict[str, Any] = {
        "access_policy_id": "policy_research",
        "library_id": LIBRARY,
        "allowed_purposes": ("research",),
        "collection_rules": (CollectionRule(COLLECTION, PolicyEffect.ALLOW),),
        field: 1,
    }

    with pytest.raises(InvalidAccessPolicyError, match="booleans"):
        AccessPolicySnapshot(**values)


def test_policy_rules_reject_wrong_types_and_duplicate_identifiers() -> None:
    with pytest.raises(InvalidAccessPolicyError, match="invalid rule type"):
        AccessPolicySnapshot(
            access_policy_id="policy_research",
            library_id=LIBRARY,
            allowed_purposes=("research",),
            collection_rules=cast(Any, (SourceRule("source_one", PolicyEffect.ALLOW),)),
        )

    with pytest.raises(InvalidAccessPolicyError, match="one rule"):
        AccessPolicySnapshot(
            access_policy_id="policy_research",
            library_id=LIBRARY,
            allowed_purposes=("research",),
            collection_rules=(
                CollectionRule(COLLECTION, PolicyEffect.ALLOW),
                CollectionRule(COLLECTION, PolicyEffect.DENY),
            ),
        )


def test_access_identifiers_enums_and_iterables_fail_closed() -> None:
    with pytest.raises(InvalidAccessScopeError, match="identifiers"):
        CollectionRule("collection_bad id", PolicyEffect.ALLOW)
    with pytest.raises(InvalidAccessScopeError, match="PolicyEffect"):
        CollectionRule(COLLECTION, cast(Any, "allow"))
    with pytest.raises(InvalidAccessScopeError, match="MembershipState"):
        SnapshotMembership(COLLECTION, cast(Any, "active"))
    with pytest.raises(InvalidAccessScopeError, match="iterable"):
        QueryExclusions(source_ids=cast(Any, "source_one"))
    with pytest.raises(InvalidAccessScopeError, match="unique"):
        QueryExclusions(source_ids=("source_one", "source_one"))


@pytest.mark.parametrize("purpose", ["", "Research", "a" * 65, 1])
def test_request_scope_rejects_invalid_purpose(purpose: object) -> None:
    with pytest.raises(InvalidAccessScopeError, match="purpose"):
        RequestScope(
            library_id=LIBRARY,
            snapshot_hash=SNAPSHOT_HASH,
            purpose=cast(Any, purpose),
            collection_ids=(COLLECTION,),
        )


def test_request_scope_rejects_invalid_collection_selection_and_exclusions() -> None:
    with pytest.raises(InvalidAccessScopeError, match="at least one"):
        RequestScope(LIBRARY, SNAPSHOT_HASH, "research", ())
    with pytest.raises(InvalidAccessScopeError, match="at most 16"):
        RequestScope(
            LIBRARY,
            SNAPSHOT_HASH,
            "research",
            tuple(f"collection_{index}" for index in range(17)),
        )
    with pytest.raises(InvalidAccessScopeError, match="unique"):
        RequestScope(LIBRARY, SNAPSHOT_HASH, "research", (COLLECTION, COLLECTION))
    with pytest.raises(InvalidAccessScopeError, match="QueryExclusions"):
        RequestScope(
            LIBRARY,
            SNAPSHOT_HASH,
            "research",
            (COLLECTION,),
            exclusions=cast(Any, ()),
        )
    with pytest.raises(InvalidAccessScopeError, match="SHA-256"):
        RequestScope(LIBRARY, "A" * 64, "research", (COLLECTION,))


def test_snapshot_candidate_requires_unique_typed_membership_metadata() -> None:
    with pytest.raises(InvalidAccessScopeError, match="must have a membership"):
        SnapshotCandidate("fragment_one", "source_version_one", "source_one", None, ())
    with pytest.raises(InvalidAccessScopeError, match="invalid value"):
        SnapshotCandidate(
            "fragment_one",
            "source_version_one",
            "source_one",
            None,
            cast(Any, ("collection_alpha",)),
        )
    duplicate = SnapshotMembership(COLLECTION, MembershipState.ACTIVE)
    with pytest.raises(InvalidAccessScopeError, match="one state"):
        SnapshotCandidate(
            "fragment_one",
            "source_version_one",
            "source_one",
            None,
            (duplicate, duplicate),
        )
    with pytest.raises(InvalidAccessScopeError, match="prefix 'family_'"):
        SnapshotCandidate(
            "fragment_one",
            "source_version_one",
            "source_one",
            "source_family_one",
            (duplicate,),
        )


def test_manifest_contract_rejects_empty_or_duplicate_fragment_entries() -> None:
    with pytest.raises(InvalidAccessScopeError, match="allowed Collection"):
        PermittedManifestItem(
            "fragment_one",
            "source_version_one",
            "source_one",
            None,
            (),
        )

    item = PermittedManifestItem(
        "fragment_one",
        "source_version_one",
        "source_one",
        "family_one",
        (COLLECTION,),
    )
    with pytest.raises(InvalidAccessScopeError, match="fragment IDs"):
        PermittedManifest(LIBRARY, SNAPSHOT_HASH, (item, item))
    with pytest.raises(InvalidAccessScopeError, match="invalid item"):
        PermittedManifest(LIBRARY, SNAPSHOT_HASH, cast(Any, ("fragment_one",)))


def test_token_public_result_and_compiled_tuple_validate_all_integrity_fields() -> None:
    with pytest.raises(PermittedTokenError, match="SHA-256"):
        PermittedToken(LIBRARY, SNAPSHOT_HASH, OTHER_HASH, "bad", OTHER_HASH)
    with pytest.raises(InvalidAccessScopeError, match="boolean"):
        PublicAccessResult(cast(Any, 1))

    manifest = PermittedManifest(LIBRARY, SNAPSHOT_HASH, ())
    public = PublicAccessResult(False)
    with pytest.raises(PermittedTokenError, match="Library"):
        CompiledAccess(
            manifest,
            PermittedToken(
                OTHER_LIBRARY,
                SNAPSHOT_HASH,
                OTHER_HASH,
                OTHER_HASH,
                manifest.permitted_set_hash,
            ),
            public,
        )
    with pytest.raises(PermittedTokenError, match="snapshot"):
        CompiledAccess(
            manifest,
            PermittedToken(
                LIBRARY,
                OTHER_HASH,
                OTHER_HASH,
                OTHER_HASH,
                manifest.permitted_set_hash,
            ),
            public,
        )
    with pytest.raises(PermittedTokenError, match="permitted set"):
        CompiledAccess(
            manifest,
            PermittedToken(LIBRARY, SNAPSHOT_HASH, OTHER_HASH, OTHER_HASH, OTHER_HASH),
            public,
        )


def test_compiler_rejects_cross_library_and_untyped_candidates() -> None:
    cross_library_scope = dataclasses.replace(_scope(), library_id=OTHER_LIBRARY)
    with pytest.raises(InvalidAccessScopeError, match="different Libraries"):
        compile_access(policy=_policy(), scope=cross_library_scope, candidates=())
    with pytest.raises(InvalidAccessScopeError, match="SnapshotCandidate"):
        compile_access(policy=_policy(), scope=_scope(), candidates=cast(Any, ("fragment",)))


def test_compiler_ignores_candidates_outside_requested_collections() -> None:
    candidate = dataclasses.replace(
        _candidate(),
        memberships=(SnapshotMembership("collection_other", MembershipState.ACTIVE),),
    )

    compiled = compile_access(policy=_policy(), scope=_scope(), candidates=(candidate,))

    assert compiled.manifest.items == ()
    assert compiled.public_result.policy_omission_present is False


def test_permitted_token_rejects_wrong_type_and_integrity_tampering() -> None:
    compiled = compile_access(policy=_policy(), scope=_scope(), candidates=(_candidate(),))
    token = compiled.token
    expected = {
        "library_id": token.library_id,
        "snapshot_hash": token.snapshot_hash,
        "policy_hash": token.policy_hash,
        "exclusion_hash": token.exclusion_hash,
        "permitted_set_hash": token.permitted_set_hash,
    }
    with pytest.raises(PermittedTokenError, match="requires"):
        verify_permitted_token(cast(Any, object()), **expected)

    object.__setattr__(token, "token_hash", OTHER_HASH)
    with pytest.raises(PermittedTokenError, match="integrity"):
        verify_permitted_token(token, **expected)


def test_explicit_model_ids_run_uuid4_validators(tmp_path: Path) -> None:
    library = LibraryConfig(name="Research")
    explicit_library = LibraryConfig(library_id=library.id, name="Explicit")
    collection = CollectionConfig(
        collection_id="collection_" + uuid.uuid4().hex,
        library_id=explicit_library.id,
        name="Corpus",
        kind=CollectionKind.CORPUS,
        roots=(CollectionRoot(path=tmp_path),),
    )

    assert explicit_library.id == library.id
    assert validate_collection_id(collection.id) == collection.id


def test_uuid_shaped_non_v4_ids_are_rejected() -> None:
    with pytest.raises(InvalidLibraryIdError, match="UUID4"):
        validate_library_id("library_" + uuid.uuid1().hex)
    with pytest.raises(InvalidCollectionIdError, match="UUID4"):
        validate_collection_id("collection_" + uuid.uuid1().hex)


def test_collection_root_and_globs_reject_direct_ambiguity(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="absolute"):
        CollectionRoot(path=Path("relative"))
    with pytest.raises(ValidationError, match="absolute"):
        CollectionRoot(path=tmp_path / "base" / ".." / "source")
    with pytest.raises(InvalidCollectionGlobError, match="non-empty"):
        validate_globs(("bad\x00glob",))


def test_layout_creation_rolls_back_partial_root_after_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = LibraryConfig(name="Research")
    planned = library_paths(config.id, data_root=tmp_path / "data")

    def fail_file_creation(_path: Path) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(library_paths_module, "_create_private_file", fail_file_creation)
    with pytest.raises(LibraryLayoutError, match="could not create"):
        create_library_layout(config, data_root=tmp_path / "data")

    assert not planned.root.exists()


def test_layout_creation_rolls_back_partial_root_after_invariant_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = LibraryConfig(name="Research")
    planned = library_paths(config.id, data_root=tmp_path / "data")

    def fail_containment(_paths: object) -> None:
        raise RuntimeError("containment verifier failed")

    monkeypatch.setattr(library_paths_module, "_verify_layout_containment", fail_containment)
    with pytest.raises(RuntimeError, match="containment verifier"):
        create_library_layout(config, data_root=tmp_path / "data")

    assert not planned.root.exists()


def test_layout_creation_preserves_typed_already_exists_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_directory_creation(_path: Path) -> None:
        raise LibraryAlreadyExistsError("raced with another creator")

    monkeypatch.setattr(library_paths_module, "_create_private_directory", fail_directory_creation)
    with pytest.raises(LibraryAlreadyExistsError, match="raced"):
        create_library_layout(LibraryConfig(name="Research"), data_root=tmp_path / "data")


def test_layout_creation_wraps_os_error_before_library_root_exists(tmp_path: Path) -> None:
    data_root = tmp_path / "occupied"
    data_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(LibraryLayoutError, match="could not create"):
        create_library_layout(LibraryConfig(name="Research"), data_root=data_root)

    assert data_root.read_text(encoding="utf-8") == "not a directory"


def test_layout_rejects_symlinked_application_data_component(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    real_libraries = tmp_path / "real-libraries"
    real_libraries.mkdir()
    (data_root / "libraries").symlink_to(real_libraries, target_is_directory=True)

    with pytest.raises(LibraryLayoutError, match="real directory"):
        create_library_layout(LibraryConfig(name="Research"), data_root=data_root)


def test_layout_validation_rejects_missing_root_directory_and_escaped_members(
    tmp_path: Path,
) -> None:
    config = LibraryConfig(name="Research")
    planned = library_paths(config.id, data_root=tmp_path / "missing-data")
    with pytest.raises(LibraryLayoutError, match="root is not a directory"):
        validate_library_layout(planned)

    paths = create_library_layout(config, data_root=tmp_path / "data")
    paths.blobs.rmdir()
    with pytest.raises(LibraryLayoutError, match="directory is missing"):
        validate_library_layout(paths)

    escaped_root = dataclasses.replace(paths, root=tmp_path / "escaped-root")
    with pytest.raises(LibraryLayoutError, match="not a directory"):
        validate_library_layout(escaped_root)

    escaped_blobs = tmp_path / "escaped-blobs"
    escaped_blobs.mkdir()
    escaped_member = dataclasses.replace(paths, blobs=escaped_blobs)
    with pytest.raises(LibraryLayoutError, match="member escaped"):
        validate_library_layout(escaped_member)


def test_store_can_verify_existing_schema_without_applying_migrations(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    with Store.open(database):
        pass

    with Store.open(database, apply_migrations=False) as store:
        assert store.schema_version == 9


def test_store_close_is_idempotent_and_closed_access_fails(tmp_path: Path) -> None:
    store = Store.open(tmp_path / "memory.sqlite3")
    store.close()
    store.close()

    with pytest.raises(StoreClosedError, match="closed"):
        _ = store.connection


def test_store_rejects_nested_transactions(tmp_path: Path) -> None:
    with (
        Store.open(tmp_path / "memory.sqlite3") as store,
        store.transaction(),
        pytest.raises(TransactionStateError, match="nested"),
        store.transaction(),
    ):
        pass


def test_store_wraps_sqlite_open_errors(tmp_path: Path) -> None:
    database_directory = tmp_path / "directory.sqlite3"
    database_directory.mkdir()

    with pytest.raises(StoreOpenError, match="regular file"):
        Store.open(database_directory)


def test_store_closes_connection_after_profile_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = sqlite3.connect(tmp_path / "tracked.sqlite3", isolation_level=None)
    monkeypatch.setattr(
        "dithyramba.store.database.sqlite3.connect", lambda *_args, **_kwargs: connection
    )

    def reject_profile(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("profile rejected")

    monkeypatch.setattr(database_module, "_apply_profile", reject_profile)
    with pytest.raises(StoreOpenError, match="could not open"):
        Store.open(tmp_path / "ignored.sqlite3")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_store_closes_connection_after_non_sqlite_setup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = sqlite3.connect(tmp_path / "tracked.sqlite3", isolation_level=None)
    original_connect = sqlite3.connect
    monkeypatch.setattr(
        "dithyramba.store.database.sqlite3.connect", lambda *_args, **_kwargs: connection
    )
    empty_migrations = tmp_path / "empty-migrations"
    empty_migrations.mkdir()

    with pytest.raises(MigrationDiscoveryError, match="no numbered"):
        Store.open(tmp_path / "ignored.sqlite3", migration_source=empty_migrations)

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    assert original_connect is not None


def test_in_memory_connection_is_rejected_when_it_cannot_accept_wal() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        with pytest.raises(StoreOpenError, match="required VS0 profile"):
            database_module._apply_profile(connection)
    finally:
        connection.close()


class _Rows:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows


class _IntegrityFailureConnection:
    def execute(self, sql: str) -> _Rows:
        if sql == "PRAGMA integrity_check":
            return _Rows([("broken page",), ("missing index",)])
        return _Rows([])


def test_integrity_check_reports_all_sqlite_failures() -> None:
    connection = cast(sqlite3.Connection, _IntegrityFailureConnection())

    with pytest.raises(IntegrityCheckError, match="broken page; missing index"):
        verify_integrity(connection)


def _migration_directory(tmp_path: Path, *sql_files: tuple[str, bytes | str]) -> Path:
    directory = tmp_path / str(uuid.uuid4())
    directory.mkdir()
    for name, payload in sql_files:
        if isinstance(payload, bytes):
            (directory / name).write_bytes(payload)
        else:
            (directory / name).write_text(payload, encoding="utf-8")
    return directory


def _history_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute(
        """
        CREATE TABLE schema_migrations(
            version,
            name TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    return connection


MIGRATION_TABLE_SQL = """
CREATE TABLE schema_migrations(
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
"""


def test_migration_discovery_rejects_missing_empty_and_invalid_utf8_sources(tmp_path: Path) -> None:
    with pytest.raises(MigrationDiscoveryError, match="cannot be read"):
        discover_migrations(tmp_path / "missing")

    empty = _migration_directory(tmp_path)
    with pytest.raises(MigrationDiscoveryError, match="no numbered"):
        discover_migrations(empty)

    undecodable = _migration_directory(tmp_path, ("0001_bad.sql", b"\xff\xfe"))
    with pytest.raises(MigrationDiscoveryError, match="cannot be decoded"):
        discover_migrations(undecodable)


def test_migration_discovery_rejects_empty_zero_and_duplicate_versions(tmp_path: Path) -> None:
    empty_sql = _migration_directory(tmp_path, ("0001_empty.sql", " \n"))
    with pytest.raises(MigrationDiscoveryError, match="empty"):
        discover_migrations(empty_sql)

    zero = _migration_directory(tmp_path, ("0000_zero.sql", "SELECT 1;"))
    with pytest.raises(MigrationDiscoveryError, match="invalid migration version"):
        discover_migrations(zero)

    duplicate = _migration_directory(
        tmp_path,
        ("0001_first.sql", "SELECT 1;"),
        ("0001_second.sql", "SELECT 2;"),
    )
    with pytest.raises(MigrationDiscoveryError, match="duplicate"):
        discover_migrations(duplicate)


def test_migration_discovery_ignores_non_sql_entries_and_accepts_traversable(
    tmp_path: Path,
) -> None:
    source = _migration_directory(tmp_path, ("0001_first.sql", "SELECT 1;"))
    (source / "notes.txt").write_text("ignore", encoding="utf-8")
    (source / "ignored.sql").mkdir()

    migrations = discover_migrations(source)
    packaged = discover_migrations(packaged_migration_source())

    assert migrations[0].version == 1
    assert packaged[0].name == "0001_core.sql"
    assert packaged[1].name == "0002_source_heads.sql"
    assert packaged[2].name == "0003_recall_run_artifacts.sql"
    assert packaged[3].name == "0004_meaning_core.sql"
    assert packaged[4].name == "0005_hybrid_recall.sql"
    assert packaged[5].name == "0006_structure_relations.sql"
    assert packaged[6].name == "0007_semantic_span_vectors.sql"
    assert packaged[7].name == "0008_relation_admission.sql"
    assert migration_checksums(migrations) == {1: migrations[0].sha256}


@pytest.mark.parametrize(
    ("version", "name", "expected"),
    [
        ("1", "0001_first.sql", "not an integer"),
        (2, "0001_first.sql", "non-contiguous"),
        (1, "0001_changed.sql", "name changed"),
    ],
)
def test_migration_history_rejects_invalid_version_order_or_name(
    tmp_path: Path, version: object, name: str, expected: str
) -> None:
    source = _migration_directory(tmp_path, ("0001_first.sql", "SELECT 1;"))
    runner_for_hash = MigrationRunner(sqlite3.connect(":memory:"), source)
    expected_hash = runner_for_hash.migrations[0].sha256
    runner_for_hash._connection.close()
    connection = _history_connection()
    try:
        connection.execute(
            "INSERT INTO schema_migrations VALUES (?, ?, ?, 'now')",
            (version, name, expected_hash),
        )
        with pytest.raises(MigrationStateError, match=expected):
            MigrationRunner(connection, source).verify()
    finally:
        connection.close()


def test_migration_history_with_unreadable_shape_fails_closed(tmp_path: Path) -> None:
    source = _migration_directory(tmp_path, ("0001_first.sql", "SELECT 1;"))
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        connection.execute("CREATE TABLE schema_migrations(unexpected TEXT)")
        with pytest.raises(MigrationStateError, match="cannot be read"):
            MigrationRunner(connection, source).verify()
    finally:
        connection.close()


def _two_step_migrations(
    tmp_path: Path, second_sql: str = "CREATE TABLE second(value TEXT);"
) -> Path:
    return _migration_directory(
        tmp_path,
        ("0001_initial.sql", MIGRATION_TABLE_SQL),
        ("0002_second.sql", second_sql),
    )


def _apply_initial_migration(connection: sqlite3.Connection, tmp_path: Path) -> None:
    source = _migration_directory(tmp_path, ("0001_initial.sql", MIGRATION_TABLE_SQL))
    assert [migration.version for migration in MigrationRunner(connection, source).apply_all()] == [
        1
    ]


def test_fresh_database_reaches_multi_migration_head_without_backup(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        applied = MigrationRunner(connection, _two_step_migrations(tmp_path)).apply_all()
        assert [migration.version for migration in applied] == [1, 2]
    finally:
        connection.close()


def test_later_migration_requires_verified_backup(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        _apply_initial_migration(connection, tmp_path)
        with pytest.raises(MigrationBackupRequiredError, match="backup hook"):
            MigrationRunner(connection, _two_step_migrations(tmp_path)).apply_all()
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 1
    finally:
        connection.close()


def test_backup_failure_is_wrapped_and_does_not_apply_later_migration(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)

    def fail_backup(_connection: sqlite3.Connection, _migration: object) -> None:
        raise OSError("backup unavailable")

    try:
        _apply_initial_migration(connection, tmp_path)
        with pytest.raises(MigrationApplyError, match="backup verification"):
            MigrationRunner(connection, _two_step_migrations(tmp_path)).apply_all(
                backup_hook=fail_backup
            )
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 1
    finally:
        connection.close()


def test_backup_hook_must_not_leave_transaction_open(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)

    def leave_transaction_open(connection: sqlite3.Connection, _migration: object) -> None:
        connection.execute("BEGIN")

    try:
        _apply_initial_migration(connection, tmp_path)
        with pytest.raises(TransactionStateError, match="active transaction"):
            MigrationRunner(connection, _two_step_migrations(tmp_path)).apply_all(
                backup_hook=leave_transaction_open
            )
        assert connection.in_transaction is False
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 1
    finally:
        connection.close()


def test_verified_backup_allows_later_migration(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    observed_versions: list[int] = []

    def record_backup(_connection: sqlite3.Connection, migration: object) -> None:
        observed_versions.append(cast(Any, migration).version)

    try:
        _apply_initial_migration(connection, tmp_path)
        applied = MigrationRunner(connection, _two_step_migrations(tmp_path)).apply_all(
            backup_hook=record_backup
        )
        assert [migration.version for migration in applied] == [2]
        assert observed_versions == [2]
        assert connection.execute("SELECT count(*) FROM second").fetchone()[0] == 0
    finally:
        connection.close()


def test_locked_migration_guard_runs_after_begin_and_before_first_sql(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    observed_versions: list[int] = []

    def locked_guard(connection: sqlite3.Connection, migration: object) -> None:
        assert connection.in_transaction is True
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name = 'second'"
            ).fetchone()[0]
            == 0
        )
        observed_versions.append(cast(Any, migration).version)

    try:
        _apply_initial_migration(connection, tmp_path)
        applied = MigrationRunner(connection, _two_step_migrations(tmp_path)).apply_all(
            backup_hook=lambda _connection, _migration: None,
            locked_guard=locked_guard,
        )
        assert [migration.version for migration in applied] == [2]
        assert observed_versions == [2]
        assert connection.in_transaction is False
        assert connection.execute("SELECT COUNT(*) FROM second").fetchone()[0] == 0
    finally:
        connection.close()


def test_locked_guard_failure_rolls_back_only_its_migration_transaction(
    tmp_path: Path,
) -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    source = _migration_directory(
        tmp_path,
        ("0001_initial.sql", MIGRATION_TABLE_SQL),
        ("0002_second.sql", "CREATE TABLE second(value TEXT);"),
        ("0003_third.sql", "CREATE TABLE third(value TEXT);"),
    )

    def fail_third(_connection: sqlite3.Connection, migration: object) -> None:
        if cast(Any, migration).version == 3:
            raise OSError("locked state proof failed")

    try:
        _apply_initial_migration(connection, tmp_path)
        with pytest.raises(MigrationApplyError, match="locked migration guard") as captured:
            MigrationRunner(connection, source).apply_all(
                backup_hook=lambda _connection, _migration: None,
                locked_guard=fail_third,
            )
        assert isinstance(captured.value.__cause__, OSError)
        assert connection.in_transaction is False
        assert MigrationRunner(connection, source).current_version() == 2
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            ).fetchall()
        }
        assert "second" in tables
        assert "third" not in tables
    finally:
        connection.close()


def test_locked_guard_cannot_end_runner_owned_transaction(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)

    def commit_guard(connection: sqlite3.Connection, _migration: object) -> None:
        connection.commit()

    try:
        _apply_initial_migration(connection, tmp_path)
        with pytest.raises(TransactionStateError, match="ended the transaction"):
            MigrationRunner(connection, _two_step_migrations(tmp_path)).apply_all(
                backup_hook=lambda _connection, _migration: None,
                locked_guard=commit_guard,
            )
        assert connection.in_transaction is False
        assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name = 'second'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_migration_cannot_start_inside_caller_transaction(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute("BEGIN")
    try:
        with pytest.raises(TransactionStateError, match="active transaction"):
            MigrationRunner(
                connection,
                _migration_directory(tmp_path, ("0001_initial.sql", MIGRATION_TABLE_SQL)),
            ).apply_all()
    finally:
        connection.rollback()
        connection.close()


def test_failed_migration_rolls_back_schema_and_history_atomically(tmp_path: Path) -> None:
    broken_sql = MIGRATION_TABLE_SQL + "\nCREATE TABLE partial(value TEXT);\nBROKEN STATEMENT;\n"
    source = _migration_directory(tmp_path, ("0001_broken.sql", broken_sql))
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        with pytest.raises(MigrationApplyError, match="failed to apply"):
            MigrationRunner(connection, source).apply_all()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == set()
    finally:
        connection.close()


def test_sql_statement_parser_ignores_comments_and_rejects_incomplete_sql() -> None:
    comments_only = "-- a line comment;\n/* a block comment; */\n"
    assert list(migrations_module._iter_sql_statements(comments_only)) == []

    with pytest.raises(MigrationDiscoveryError, match="incomplete"):
        list(migrations_module._iter_sql_statements("CREATE TABLE unfinished(value TEXT)"))
