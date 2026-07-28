from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    RequestScope,
    SourceRule,
)
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.library import LibraryConfig, PathOverlapError
from dithyramba.persistence import (
    AuthorizationError,
    AuthorizedRead,
    LibraryRepository,
    PersistenceConflictError,
    PersistenceIntegrityError,
    initialize_library,
    library_logical_identity_hash,
    list_libraries,
    open_library,
)
from dithyramba.store import Store

NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-20T12:00:00.000000Z"


class EventIds:
    def __init__(self) -> None:
        self._next = 0

    def __call__(self) -> str:
        value = f"event_{self._next:04d}"
        self._next += 1
        return value


def fixed_clock() -> datetime:
    return NOW


def test_two_libraries_are_physically_separate_and_safely_listed(tmp_path: Path) -> None:
    data_root = tmp_path / "application-data"
    first = LibraryConfig(name="Research")
    second = LibraryConfig(name="Creative")

    with initialize_library(first, data_root=data_root, clock=fixed_clock) as first_repo:
        first_database = first_repo.paths.database
    with initialize_library(second, data_root=data_root, clock=fixed_clock) as second_repo:
        second_database = second_repo.paths.database

    records = list_libraries(data_root=data_root)

    assert first_database != second_database
    assert first_database.is_file() and second_database.is_file()
    assert {record.config.library_id for record in records} == {first.library_id, second.library_id}
    assert all(record.paths.application_data_root == data_root for record in records)
    assert not (data_root / "registry.sqlite3").exists()
    assert library_logical_identity_hash(first) == library_logical_identity_hash(
        LibraryConfig(library_id=first.library_id, name="Renamed display label")
    )


def test_initialization_never_places_live_data_under_a_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "synchronized-source"
    source_root.mkdir()
    unsafe_data_root = source_root / "runtime"

    with pytest.raises(PathOverlapError):
        initialize_library(
            LibraryConfig(name="Unsafe"),
            data_root=unsafe_data_root,
            declared_source_roots=(source_root,),
        )

    assert not unsafe_data_root.exists()


def test_collection_root_round_trip_and_failed_outbox_insert_roll_back_together(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    source_root.mkdir()
    config = LibraryConfig(name="Research")
    event_ids = EventIds()

    with initialize_library(
        config,
        data_root=data_root,
        clock=fixed_clock,
        event_id_factory=event_ids,
    ) as repository:
        root = build_collection_root(
            source_root,
            data_root=data_root,
            include_globs=("papers/**/*.md",),
            exclude_globs=("private/**",),
        )
        collection = CollectionConfig(
            library_id=config.library_id,
            name="PhD",
            kind=CollectionKind.CORPUS,
            roots=(root,),
        )
        persisted = repository.create_collection(collection)
        loaded = repository.get_collection(collection.collection_id)

        assert loaded == persisted
        assert loaded.config.roots == (root,)
        assert len(loaded.collection_root_ids) == 1
        before_events = len(repository.pending_outbox_events())

    failed_root = tmp_path / "failed-source"
    failed_root.mkdir()
    with open_library(
        config.library_id,
        data_root=data_root,
        clock=fixed_clock,
        event_id_factory=lambda: "invalid",
    ) as repository:
        failed = CollectionConfig(
            library_id=config.library_id,
            name="Must roll back",
            kind=CollectionKind.CASE,
            roots=(build_collection_root(failed_root, data_root=data_root),),
        )
        with pytest.raises(PersistenceConflictError):
            repository.create_collection(failed)

        assert {item.config.name for item in repository.list_collections()} == {"PhD"}
        assert len(repository.pending_outbox_events()) == before_events


def test_policy_round_trip_rehash_compile_and_missing_source_rule_fail_closed(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    source_root.mkdir()
    library = LibraryConfig(name="Research")

    with initialize_library(library, data_root=data_root, clock=fixed_clock) as repository:
        collection = repository.create_collection(
            CollectionConfig(
                library_id=library.library_id,
                name="PhD",
                kind=CollectionKind.CORPUS,
                roots=(build_collection_root(source_root, data_root=data_root),),
            )
        )
        policy = AccessPolicySnapshot(
            access_policy_id="policy_research",
            library_id=library.library_id,
            allowed_purposes=("teaching", "research"),
            collection_rules=(CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),),
            allow_export=True,
        )
        persisted = repository.persist_access_policy(name="Local research", snapshot=policy)
        loaded = repository.get_access_policy(policy.access_policy_id)

        assert loaded == persisted
        assert loaded.snapshot == policy
        assert loaded.snapshot.policy_hash == policy.policy_hash
        assert repository.list_access_policies() == (loaded,)

        invalid = AccessPolicySnapshot(
            access_policy_id="policy_invalid_source",
            library_id=library.library_id,
            allowed_purposes=("research",),
            collection_rules=policy.collection_rules,
            source_rules=(SourceRule("source_missing", PolicyEffect.DENY),),
        )
        before = len(repository.pending_outbox_events())
        with pytest.raises(PersistenceIntegrityError, match="Sources"):
            repository.persist_access_policy(name="Invalid", snapshot=invalid)
        assert len(repository.pending_outbox_events()) == before
        assert {item.snapshot.access_policy_id for item in repository.list_access_policies()} == {
            policy.access_policy_id
        }


def test_outbox_export_is_ordered_and_idempotent_across_append_mark_crash(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    library = LibraryConfig(name="Research")
    event_ids = EventIds()

    with initialize_library(
        library,
        data_root=data_root,
        clock=fixed_clock,
        event_id_factory=event_ids,
    ) as repository:
        for name, root in (("First", first_root), ("Second", second_root)):
            repository.create_collection(
                CollectionConfig(
                    library_id=library.library_id,
                    name=name,
                    kind=CollectionKind.CORPUS,
                    roots=(build_collection_root(root, data_root=data_root),),
                )
            )
        pending = repository.pending_outbox_events()
        assert tuple(event.event_id for event in pending) == (
            "event_0000",
            "event_0001",
            "event_0002",
        )

        def crash_after_append(_event: object) -> None:
            raise RuntimeError("simulated crash")

        with pytest.raises(RuntimeError, match="simulated crash"):
            repository.export_outbox(after_append=crash_after_append)
        assert len(repository.pending_outbox_events()) == 3

        result = repository.export_outbox()
        assert result.appended_count == 2
        assert result.deduplicated_count == 1
        assert result.delivered_count == 3
        assert repository.pending_outbox_events() == ()

        lines = repository.paths.events.read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines]
        assert [event["event_id"] for event in events] == [
            "event_0000",
            "event_0001",
            "event_0002",
        ]
        assert len({event["event_id"] for event in events}) == len(events)
        assert all(
            canonical_json_bytes(event).decode("utf-8") == line
            for event, line in zip(events, lines, strict=True)
        )
        assert repository.export_outbox().delivered_count == 0


def test_two_exporters_serialize_one_event_without_duplicate_jsonl(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    library = LibraryConfig(name="Concurrent export")
    with initialize_library(library, data_root=data_root, clock=fixed_clock):
        pass
    barrier = threading.Barrier(2)

    def export_once() -> tuple[int, int]:
        barrier.wait()
        with open_library(library.library_id, data_root=data_root, clock=fixed_clock) as repository:
            result = repository.export_outbox()
            return result.appended_count, result.delivered_count

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: export_once(), range(2)))

    assert sum(item[0] for item in results) == 1
    assert sum(item[1] for item in results) == 1
    events = next((data_root / "libraries" / library.library_id).glob("events.jsonl"))
    lines = events.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_id"].startswith("event_")


def test_empty_library_online_backup_opens_and_verifies(tmp_path: Path) -> None:
    library = LibraryConfig(name="Empty")
    with initialize_library(
        library,
        data_root=tmp_path / "data",
        clock=fixed_clock,
    ) as repository:
        backup = repository.create_online_backup()

        assert backup.path.parent == repository.paths.backups
        assert backup.path.is_file()
        assert len(backup.sha256) == 64
        assert backup.schema_version == 9
        assert not tuple(repository.paths.backups.glob("*.tmp"))

    with Store.open(backup.path, apply_migrations=False) as backup_store:
        backup_store.verify()
        assert backup_store.connection.execute("SELECT count(*) FROM libraries").fetchone()[0] == 1
        assert (
            backup_store.connection.execute("SELECT count(*) FROM event_outbox").fetchone()[0] == 1
        )


def test_db_derived_authorization_blocks_forgery_before_text_select_and_scopes_rows(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    public_root = tmp_path / "public"
    private_root = tmp_path / "private"
    public_root.mkdir()
    private_root.mkdir()
    library = LibraryConfig(name="Isolation")

    with initialize_library(library, data_root=data_root, clock=fixed_clock) as repository:
        public = repository.create_collection(
            CollectionConfig(
                library_id=library.library_id,
                name="Public",
                kind=CollectionKind.CORPUS,
                roots=(build_collection_root(public_root, data_root=data_root),),
            )
        )
        private = repository.create_collection(
            CollectionConfig(
                library_id=library.library_id,
                name="Private",
                kind=CollectionKind.HOLDOUT,
                roots=(build_collection_root(private_root, data_root=data_root),),
            )
        )
        snapshot_hash = _insert_snapshot_fixture(
            repository,
            library_id=library.library_id,
            memberships=(
                (public.config.collection_id, "public", "Public evidence", "active"),
                (private.config.collection_id, "private", "Private canary", "active"),
            ),
        )
        policy = AccessPolicySnapshot(
            access_policy_id="policy_public_only",
            library_id=library.library_id,
            allowed_purposes=("research",),
            collection_rules=(
                CollectionRule(public.config.collection_id, PolicyEffect.ALLOW),
                CollectionRule(private.config.collection_id, PolicyEffect.DENY),
            ),
        )
        repository.persist_access_policy(name="Public only", snapshot=policy)
        scope = RequestScope(
            library_id=library.library_id,
            snapshot_hash=snapshot_hash,
            purpose="research",
            collection_ids=(public.config.collection_id, private.config.collection_id),
        )
        authorization = repository.authorize_read(
            access_policy_id=policy.access_policy_id,
            scope=scope,
        )

        with pytest.raises(sqlite3.DatabaseError):
            repository._store.connection.execute("SELECT text FROM source_fragments").fetchall()

        statements: list[str] = []
        repository._store.connection.set_trace_callback(statements.append)
        copied = dataclasses.replace(authorization)
        with pytest.raises(AuthorizationError, match="not issued"):
            repository.read_permitted_fragments(copied)
        with pytest.raises(AuthorizationError):
            repository.read_permitted_fragments(cast(AuthorizedRead, None))
        assert not any("sf.text" in statement.lower() for statement in statements)

        fragments = repository.read_permitted_fragments(authorization)
        repository._store.connection.set_trace_callback(None)

        assert tuple(fragment.source_fragment_id for fragment in fragments) == ("fragment_public",)
        assert fragments[0].text == "Public evidence"
        assert authorization.compiled.public_result.policy_omission_present is True
        assert all("Private canary" not in statement for statement in statements)
        assert any("sf.text" in statement.lower() for statement in statements)

        # SQLite authorizers run when a statement is prepared. The Store
        # deliberately disables prepared-statement caching so a statement that
        # was prepared inside an allowed scope cannot be replayed after it.
        repeated_sql = "SELECT text FROM source_fragments ORDER BY source_fragment_id"
        with repository._permit_fragment_text_read():
            permitted_text = repository._store.connection.execute(repeated_sql).fetchall()
        assert [str(row[0]) for row in permitted_text] == ["Private canary", "Public evidence"]
        with pytest.raises(sqlite3.DatabaseError):
            repository._store.connection.execute(repeated_sql).fetchall()


def _insert_snapshot_fixture(
    repository: LibraryRepository,
    *,
    library_id: str,
    memberships: tuple[tuple[str, str, str, str], ...],
) -> str:
    connection = repository._store.connection
    for collection_id, suffix, text, state in memberships:
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
            (
                source_id,
                library_id,
                f"file:///{suffix}.md",
                "text/markdown",
                suffix.title(),
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
            (f"family_{suffix}", library_id, suffix.title(), NOW_TEXT),
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
            "INSERT INTO collection_memberships VALUES (?, ?, ?, ?)",
            (collection_id, source_id, state, NOW_TEXT),
        )
    snapshot = repository.freeze_snapshot(item[0] for item in memberships)
    return snapshot.manifest_hash
