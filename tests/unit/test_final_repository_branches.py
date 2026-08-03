from __future__ import annotations

import json
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
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig, library_paths
from dithyramba.persistence import (
    LibraryRepository,
    PersistenceIntegrityError,
    initialize_library,
)
from dithyramba.provenance import (
    IngestInputOutcome,
    ProcessingRunStatus,
    TerminalInputOutcome,
    build_ingest_coverage,
)
from dithyramba.store import Store

NOW_TEXT = "2026-07-20T12:00:00.000000Z"


def _repository_with_collection(
    tmp_path: Path,
    *,
    name: str = "Repository branches",
) -> tuple[LibraryRepository, str, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    library = LibraryConfig(name=name)
    repository = initialize_library(library, data_root=tmp_path / "data")
    collection = repository.create_collection(
        CollectionConfig(
            library_id=library.library_id,
            name="Corpus",
            kind=CollectionKind.CORPUS,
            roots=(
                build_collection_root(
                    source_root,
                    data_root=repository.paths.application_data_root,
                ),
            ),
        )
    )
    return repository, collection.config.collection_id, source_root


def _completed_coverage(repository: LibraryRepository) -> tuple[str, str]:
    run = repository.begin_ingest_run(code_version="test", profile_version="test/1.0")
    draft = build_ingest_coverage(
        processing_run_id=run.processing_run_id,
        collection_id="collection_fixture",
        outcomes=(
            IngestInputOutcome(
                collection_root_id="root_fixture",
                relative_path="skipped.txt",
                terminal_outcome=TerminalInputOutcome.SKIPPED,
                failure_code="no_extractable_text",
            ),
        ),
    )
    completed, coverage = repository.complete_ingest_run(
        coverage=draft,
        status=ProcessingRunStatus.SUCCEEDED,
    )
    return coverage.coverage_report_id, completed.processing_run_id


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("invalid-run-id", "processing_run_id is invalid"),
        ("invalid-collection-id", "collection_id is invalid"),
        ("hash", "hash mismatch"),
        ("identity", "ID mismatch"),
        ("relational-scalar", "relational fields"),
        ("policy-projection", "policy omission projection"),
        ("omissions-projection", "omissions projection"),
        ("omission-identity", "Omission ID mismatch"),
        ("run-output", "terminal ingest ProcessingRun"),
        ("cancelled-run", "terminal ingest run"),
    ],
)
def test_coverage_report_loader_fails_closed_for_independent_corruption(
    tmp_path: Path,
    case: str,
    expected: str,
) -> None:
    library = LibraryConfig(name=f"Coverage corruption {case}")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        coverage_id, run_id = _completed_coverage(repository)
        connection = repository._store.connection
        row = connection.execute(
            "SELECT report_json FROM coverage_reports WHERE coverage_report_id = ?",
            (coverage_id,),
        ).fetchone()
        assert row is not None
        payload = cast(dict[str, object], json.loads(str(row[0])))

        connection.execute("DROP TRIGGER coverage_reports_no_update")
        if case == "invalid-run-id":
            payload["processing_run_id"] = None
            connection.execute(
                "UPDATE coverage_reports SET report_json = ? WHERE coverage_report_id = ?",
                (canonical_json_bytes(payload).decode("utf-8"), coverage_id),
            )
        elif case == "invalid-collection-id":
            payload["collection_id"] = ""
            connection.execute(
                "UPDATE coverage_reports SET report_json = ? WHERE coverage_report_id = ?",
                (canonical_json_bytes(payload).decode("utf-8"), coverage_id),
            )
        elif case == "hash":
            connection.execute(
                "UPDATE coverage_reports SET report_hash = ? WHERE coverage_report_id = ?",
                ("f" * 64, coverage_id),
            )
        elif case == "identity":
            payload["collection_id"] = "collection_rebound"
            connection.execute(
                "UPDATE coverage_reports SET report_json = ?, report_hash = ? "
                "WHERE coverage_report_id = ?",
                (
                    canonical_json_bytes(payload).decode("utf-8"),
                    canonical_sha256_hex(payload),
                    coverage_id,
                ),
            )
        elif case == "relational-scalar":
            connection.execute(
                "UPDATE coverage_reports SET stage = 'recall' WHERE coverage_report_id = ?",
                (coverage_id,),
            )
        elif case == "policy-projection":
            connection.execute(
                "UPDATE coverage_reports SET policy_omission_present = 1 "
                "WHERE coverage_report_id = ?",
                (coverage_id,),
            )
        elif case == "omissions-projection":
            connection.execute(
                "INSERT INTO omissions VALUES (?, ?, ?, ?, ?, ?)",
                ("omission_extra", coverage_id, "unsupported", "counted", 1, "extra"),
            )
        elif case == "omission-identity":
            connection.execute("DROP TRIGGER omissions_no_update")
            connection.execute(
                "UPDATE omissions SET omission_id = 'omission_tampered' "
                "WHERE coverage_report_id = ?",
                (coverage_id,),
            )
        elif case == "run-output":
            connection.execute(
                "UPDATE processing_runs SET output_hash = ? WHERE processing_run_id = ?",
                ("f" * 64, run_id),
            )
        else:
            assert case == "cancelled-run"
            connection.execute(
                "UPDATE processing_runs SET status = 'cancelled' WHERE processing_run_id = ?",
                (run_id,),
            )

        with pytest.raises(PersistenceIntegrityError, match=expected):
            repository.get_coverage_report(coverage_id)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("media-type", "ingest_persistence_failed"),
        ("missing-versions", "ingest_persistence_failed"),
        ("missing-head", "ingest_persistence_failed"),
    ],
)
def test_reingest_reports_persisted_source_corruption(
    tmp_path: Path,
    case: str,
    expected: str,
) -> None:
    repository, collection_id, source_root = _repository_with_collection(tmp_path)
    try:
        source_path = source_root / "paper.txt"
        source_path.write_text("Stable source bytes", encoding="utf-8")
        first = IngestService(repository).ingest_path(collection_id, source_path.name)
        source_id = first.outcomes[0].source_id
        source_version_id = first.outcomes[0].source_version_id
        assert source_id is not None and source_version_id is not None
        connection = repository._store.connection

        if case == "media-type":
            connection.execute("DROP TRIGGER sources_no_update")
            connection.execute(
                "UPDATE sources SET media_type = 'text/markdown' WHERE source_id = ?",
                (source_id,),
            )
        elif case == "missing-versions":
            connection.execute("DROP TRIGGER source_heads_no_delete")
            connection.execute("DELETE FROM source_heads WHERE source_id = ?", (source_id,))
            connection.execute("DROP TRIGGER source_fragments_no_delete")
            connection.execute(
                "DELETE FROM source_fragments WHERE source_version_id = ?",
                (source_version_id,),
            )
            connection.execute("DROP TRIGGER source_versions_no_delete")
            connection.execute(
                "DELETE FROM source_versions WHERE source_version_id = ?",
                (source_version_id,),
            )
        else:
            assert case == "missing-head"
            connection.execute("DROP TRIGGER source_heads_no_delete")
            connection.execute("DELETE FROM source_heads WHERE source_id = ?", (source_id,))

        second = IngestService(repository).ingest_path(collection_id, source_path.name)

        assert second.run.status is ProcessingRunStatus.FAILED
        assert second.outcomes[0].failure_code == expected
    finally:
        repository.close()


def test_ingest_reports_a_conflicting_blob_projection(tmp_path: Path) -> None:
    repository, collection_id, source_root = _repository_with_collection(tmp_path)
    try:
        payload = b"Blob projection integrity"
        source_path = source_root / "blob.txt"
        source_path.write_bytes(payload)
        digest = sha256_hex(payload)
        repository._store.connection.execute(
            "INSERT INTO blobs VALUES (?, ?, ?, ?)",
            (digest, len(payload) + 1, "corrupt/blob-path", NOW_TEXT),
        )

        result = IngestService(repository).ingest_path(collection_id, source_path.name)

        assert result.run.status is ProcessingRunStatus.FAILED
        assert result.outcomes[0].failure_code == "ingest_persistence_failed"
        assert repository.list_sources() == ()
    finally:
        repository.close()


def test_source_only_policy_validates_an_existing_source_without_collections(
    tmp_path: Path,
) -> None:
    repository, collection_id, source_root = _repository_with_collection(tmp_path)
    try:
        source_path = source_root / "source-rule.txt"
        source_path.write_text("Source-scoped policy", encoding="utf-8")
        result = IngestService(repository).ingest_path(collection_id, source_path.name)
        source_id = result.outcomes[0].source_id
        assert source_id is not None
        policy = AccessPolicySnapshot(
            access_policy_id="policy_source_only",
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(),
            source_rules=(SourceRule(source_id, PolicyEffect.ALLOW),),
        )

        persisted = repository.persist_access_policy(name="Source only", snapshot=policy)

        assert persisted.snapshot == policy
    finally:
        repository.close()


def test_authorization_merges_one_fragment_membership_from_two_collections(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "shared-source"
    source_root.mkdir()
    source_path = source_root / "shared.txt"
    source_path.write_text("Shared collection evidence", encoding="utf-8")
    library = LibraryConfig(name="Multiple memberships")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        root = build_collection_root(
            source_root,
            data_root=repository.paths.application_data_root,
        )
        collection_ids: list[str] = []
        for name in ("First", "Second"):
            collection = repository.create_collection(
                CollectionConfig(
                    library_id=library.library_id,
                    name=name,
                    kind=CollectionKind.CORPUS,
                    roots=(root,),
                )
            )
            collection_ids.append(collection.config.collection_id)
            result = IngestService(repository).ingest_path(
                collection.config.collection_id,
                source_path.name,
            )
            assert result.run.status is ProcessingRunStatus.SUCCEEDED

        selected = tuple(collection_ids)
        snapshot = repository.freeze_snapshot(selected)
        policy = AccessPolicySnapshot(
            access_policy_id="policy_multiple_memberships",
            library_id=library.library_id,
            allowed_purposes=("research",),
            collection_rules=tuple(
                CollectionRule(collection_id, PolicyEffect.ALLOW) for collection_id in selected
            ),
        )
        repository.persist_access_policy(name="Multiple memberships", snapshot=policy)
        authorization = repository.authorize_read(
            access_policy_id=policy.access_policy_id,
            scope=RequestScope(
                library_id=library.library_id,
                snapshot_hash=snapshot.manifest_hash,
                purpose="research",
                collection_ids=selected,
            ),
        )

        assert len(authorization.compiled.manifest.items) == 1
        assert authorization.compiled.manifest.items[0].collection_ids == tuple(sorted(selected))


def test_initialize_cleans_layout_when_store_open_fails_before_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = LibraryConfig(name="Store open failure")
    paths = library_paths(library.library_id, data_root=tmp_path / "data")

    def fail_open(_paths: object) -> Store:
        raise RuntimeError("simulated store open failure")

    monkeypatch.setattr(Store, "open_library", fail_open)

    with pytest.raises(RuntimeError, match="simulated store open failure"):
        initialize_library(library, data_root=tmp_path / "data")

    assert not paths.root.exists()
