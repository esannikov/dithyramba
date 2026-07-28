from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    RequestScope,
)
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    AuthorizationError,
    CollectionNotFoundError,
    CorpusSnapshotNotFoundError,
    LibraryRepository,
    PersistenceConflictError,
    PersistenceIntegrityError,
    initialize_library,
    open_library,
)


def _collection(
    repository: LibraryRepository,
    root: Path,
    *,
    name: str,
) -> str:
    return repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name=name,
            kind=CollectionKind.CORPUS,
            roots=(
                build_collection_root(
                    root,
                    data_root=repository.paths.application_data_root,
                ),
            ),
        )
    ).config.collection_id


def test_empty_collection_scope_round_trips_and_authorizes_without_members(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    library = LibraryConfig(name="Empty snapshot scope")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection_id = _collection(repository, root, name="Empty")
        snapshot = repository.freeze_snapshot((collection_id,))
        loaded = repository.get_corpus_snapshot(snapshot.corpus_snapshot_id)

        assert loaded == snapshot
        assert snapshot.collection_ids == (collection_id,)
        assert snapshot.members == ()
        assert b"created_at" not in snapshot.canonical_bytes
        assert b"processing_run_id" not in snapshot.canonical_bytes
        scope_json = repository._store.connection.execute(
            "SELECT scope_json FROM corpus_snapshots WHERE corpus_snapshot_id = ?",
            (snapshot.corpus_snapshot_id,),
        ).fetchone()[0]
        assert json.loads(str(scope_json))["collection_ids"] == [collection_id]

        policy = AccessPolicySnapshot(
            access_policy_id="policy_empty_collection",
            library_id=library.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
        )
        repository.persist_access_policy(name="Empty collection", snapshot=policy)
        authorization = repository.authorize_read(
            access_policy_id=policy.access_policy_id,
            scope=RequestScope(
                library_id=library.library_id,
                snapshot_hash=snapshot.manifest_hash,
                purpose="research",
                collection_ids=(collection_id,),
            ),
        )

        assert authorization.compiled.manifest.items == ()
        assert repository.read_permitted_fragments(authorization) == ()

        other_root = tmp_path / "other-empty"
        other_root.mkdir()
        other_collection_id = _collection(repository, other_root, name="Other empty")
        with pytest.raises(AuthorizationError, match="Collection absent from the snapshot"):
            repository.authorize_read(
                access_policy_id=policy.access_policy_id,
                scope=RequestScope(
                    library_id=library.library_id,
                    snapshot_hash=snapshot.manifest_hash,
                    purpose="research",
                    collection_ids=(other_collection_id,),
                ),
            )


def test_freeze_uses_reverted_source_head_instead_of_latest_version_number(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    path = root / "paper.md"
    original = "# Thesis\n\nOriginal formulation.\n"
    path.write_text(original, encoding="utf-8")
    library = LibraryConfig(name="Reverted head")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection_id = _collection(repository, root, name="Corpus")
        service = IngestService(repository)
        first_version_id = (
            service.ingest_path(collection_id, "paper.md").outcomes[0].source_version_id
        )
        path.write_text("# Thesis\n\nChanged formulation.\n", encoding="utf-8")
        second_version_id = (
            service.ingest_path(collection_id, "paper.md").outcomes[0].source_version_id
        )
        path.write_text(original, encoding="utf-8")
        reverted_version_id = (
            service.ingest_path(collection_id, "paper.md").outcomes[0].source_version_id
        )

        snapshot = repository.freeze_snapshot((collection_id,))

        assert first_version_id == reverted_version_id
        assert second_version_id != first_version_id
        assert tuple(member.source_version_id for member in snapshot.members) == (first_version_id,)


def test_same_current_version_can_be_member_of_two_collections(tmp_path: Path) -> None:
    root = tmp_path / "shared-source"
    root.mkdir()
    (root / "paper.md").write_text("# Shared\n\nOne source.\n", encoding="utf-8")
    library = LibraryConfig(name="Shared membership")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        first_id = _collection(repository, root, name="First")
        second_id = _collection(repository, root, name="Second")
        service = IngestService(repository)
        first_version = service.ingest_path(first_id, "paper.md").outcomes[0].source_version_id
        second_version = service.ingest_path(second_id, "paper.md").outcomes[0].source_version_id

        snapshot = repository.freeze_snapshot((second_id, first_id))

        assert first_version == second_version
        assert snapshot.collection_ids == tuple(sorted((first_id, second_id)))
        assert len(snapshot.members) == 2
        assert {member.collection_id for member in snapshot.members} == {first_id, second_id}
        assert {member.source_version_id for member in snapshot.members} == {first_version}
        assert len({member.source_family_id for member in snapshot.members}) == 1
        assert len({member.root_source_id for member in snapshot.members}) == 1


def test_snapshot_reconstructs_root_and_duplicate_lineage_from_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "duplicate-source"
    root.mkdir()
    payload = "# Same\n\nExact bytes.\n"
    (root / "a.md").write_text(payload, encoding="utf-8")
    (root / "b.md").write_text(payload, encoding="utf-8")
    library = LibraryConfig(name="Snapshot lineage")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection_id = _collection(repository, root, name="Corpus")
        IngestService(repository).ingest_collection(collection_id)

        snapshot = repository.freeze_snapshot((collection_id,))
        members_by_source = {member.source_id: member for member in snapshot.members}
        sources = repository.list_sources(collection_id=collection_id)
        root_source = next(source for source in sources if source.family_role.value == "root")
        duplicate = next(source for source in sources if source.family_role.value == "duplicate")

        assert members_by_source[root_source.source_id].root_source_id == root_source.source_id
        duplicate_member = members_by_source[duplicate.source_id]
        assert duplicate_member.root_source_id == root_source.source_id
        assert duplicate_member.source_family_id == root_source.source_family_id
        assert duplicate_member.family_role.value == "duplicate"


def test_identical_concurrent_freeze_reuses_one_snapshot_and_one_event(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    root = tmp_path / "source"
    root.mkdir()
    (root / "paper.txt").write_text("Concurrent evidence", encoding="utf-8")
    library = LibraryConfig(name="Concurrent freeze")
    with initialize_library(library, data_root=data_root) as repository:
        collection_id = _collection(repository, root, name="Corpus")
        IngestService(repository).ingest_path(collection_id, "paper.txt")

    barrier = threading.Barrier(2)

    def freeze_once() -> tuple[str, str]:
        with open_library(library.library_id, data_root=data_root) as repository:
            barrier.wait()
            snapshot = repository.freeze_snapshot((collection_id,))
            return snapshot.corpus_snapshot_id, snapshot.manifest_hash

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: freeze_once(), range(2)))

    assert results[0] == results[1]
    with open_library(library.library_id, data_root=data_root) as repository:
        connection = repository._store.connection
        assert connection.execute("SELECT COUNT(*) FROM corpus_snapshots").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM snapshot_members").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM event_outbox WHERE event_type = 'corpus_snapshot.frozen'"
            ).fetchone()[0]
            == 1
        )


def test_snapshot_write_rolls_back_members_when_outbox_insert_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    library = LibraryConfig(name="Atomic snapshot")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection_id = _collection(repository, root, name="Corpus")
        existing_event_id = repository.pending_outbox_events()[0].event_id
        repository._event_id_factory = lambda: existing_event_id

        with pytest.raises(PersistenceConflictError, match="CorpusSnapshot transaction"):
            repository.freeze_snapshot((collection_id,))

        connection = repository._store.connection
        assert connection.execute("SELECT COUNT(*) FROM corpus_snapshots").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM snapshot_members").fetchone()[0] == 0


def test_get_snapshot_rejects_scope_corruption_and_cross_library_ids(tmp_path: Path) -> None:
    first_root = tmp_path / "first-root"
    second_root = tmp_path / "second-root"
    first_root.mkdir()
    second_root.mkdir()
    data_root = tmp_path / "data"
    first = LibraryConfig(name="First")
    second = LibraryConfig(name="Second")

    with initialize_library(first, data_root=data_root) as repository:
        first_collection_id = _collection(repository, first_root, name="First corpus")
        snapshot = repository.freeze_snapshot((first_collection_id,))

    with initialize_library(second, data_root=data_root) as repository:
        _collection(repository, second_root, name="Second corpus")
        with pytest.raises(CorpusSnapshotNotFoundError, match="does not exist"):
            repository.get_corpus_snapshot(snapshot.corpus_snapshot_id)
        with pytest.raises(CollectionNotFoundError, match="absent"):
            repository.freeze_snapshot((first_collection_id,))

    with open_library(first.library_id, data_root=data_root) as repository:
        connection = repository._store.connection
        connection.execute("DROP TRIGGER corpus_snapshots_no_update")
        persisted_scope = snapshot.semantic_payload()["scope"]
        assert isinstance(persisted_scope, dict)
        corrupted_scope = {**persisted_scope, "unexpected": True}
        connection.execute(
            "UPDATE corpus_snapshots SET scope_json = ? WHERE corpus_snapshot_id = ?",
            (
                json.dumps(
                    corrupted_scope,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                snapshot.corpus_snapshot_id,
            ),
        )

        with pytest.raises(PersistenceIntegrityError, match="scope differs"):
            repository.get_corpus_snapshot(snapshot.corpus_snapshot_id)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("missing_library", "scope is invalid"),
        ("non_string_collection", "scope is invalid"),
        ("foreign_library", "crosses the Library boundary"),
        ("missing_collection", "absent Collection"),
        ("manifest_hash", "manifest hash mismatch"),
        ("created_at", "operational timestamp"),
    ],
)
def test_get_snapshot_fails_closed_for_corrupt_persisted_projection(
    tmp_path: Path,
    case: str,
    expected: str,
) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    library = LibraryConfig(name=f"Corruption {case}")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection_id = _collection(repository, root, name="Corpus")
        snapshot = repository.freeze_snapshot((collection_id,))
        connection = repository._store.connection
        connection.execute("DROP TRIGGER corpus_snapshots_no_update")

        if case == "manifest_hash":
            connection.execute(
                "UPDATE corpus_snapshots SET manifest_hash = ? WHERE corpus_snapshot_id = ?",
                ("f" * 64, snapshot.corpus_snapshot_id),
            )
        elif case == "created_at":
            connection.execute(
                "UPDATE corpus_snapshots SET created_at = 'not-a-time' "
                "WHERE corpus_snapshot_id = ?",
                (snapshot.corpus_snapshot_id,),
            )
        else:
            scope: dict[str, object] = {
                "schema": "dithyramba.corpus_snapshot_scope/1.0",
                "library_id": library.library_id,
                "collection_ids": [collection_id],
            }
            if case == "missing_library":
                del scope["library_id"]
            elif case == "non_string_collection":
                scope["collection_ids"] = [1]
            elif case == "foreign_library":
                scope["library_id"] = "library_foreign"
            elif case == "missing_collection":
                scope["collection_ids"] = ["collection_missing"]
            connection.execute(
                "UPDATE corpus_snapshots SET scope_json = ? WHERE corpus_snapshot_id = ?",
                (
                    json.dumps(scope, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                    snapshot.corpus_snapshot_id,
                ),
            )

        with pytest.raises(PersistenceIntegrityError, match=expected):
            repository.get_corpus_snapshot(snapshot.corpus_snapshot_id)


def test_freeze_and_load_reject_missing_heads_and_cross_library_members(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "paper.txt").write_text("Evidence", encoding="utf-8")
    library = LibraryConfig(name="Broken current state")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection_id = _collection(repository, root, name="Corpus")
        outcome = IngestService(repository).ingest_path(collection_id, "paper.txt").outcomes[0]
        connection = repository._store.connection
        connection.execute("DROP TRIGGER source_heads_no_delete")
        connection.execute("DELETE FROM source_heads WHERE source_id = ?", (outcome.source_id,))

        with pytest.raises(PersistenceIntegrityError, match="no valid current SourceVersion"):
            repository.freeze_snapshot((collection_id,))

        connection.execute(
            "INSERT INTO source_heads VALUES (?, ?, ?, NULL)",
            (outcome.source_id, outcome.source_version_id, "2026-07-20T12:00:00.000000Z"),
        )
        snapshot = repository.freeze_snapshot((collection_id,))
        connection.execute("DROP TRIGGER libraries_one_row_insert")
        connection.execute(
            "INSERT INTO libraries VALUES (?, ?, ?, ?)",
            (
                "library_foreign",
                "Foreign",
                "a" * 64,
                "2026-07-20T12:00:00.000000Z",
            ),
        )
        connection.execute("DROP TRIGGER sources_no_update")
        connection.execute(
            "UPDATE sources SET library_id = 'library_foreign' WHERE source_id = ?",
            (outcome.source_id,),
        )

        with pytest.raises(PersistenceIntegrityError, match="members cross"):
            repository.get_corpus_snapshot(snapshot.corpus_snapshot_id)


def test_freeze_rejects_invalid_scope_and_corrupt_membership_state(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "paper.txt").write_text("Evidence", encoding="utf-8")
    library = LibraryConfig(name="Invalid freeze state")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection_id = _collection(repository, root, name="Corpus")
        IngestService(repository).ingest_path(collection_id, "paper.txt")

        with pytest.raises(PersistenceIntegrityError, match="scope is invalid"):
            repository.freeze_snapshot(())

        connection = repository._store.connection
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("UPDATE collection_memberships SET state = 'corrupt'")
        with pytest.raises(PersistenceIntegrityError, match="cannot form"):
            repository.freeze_snapshot((collection_id,))
