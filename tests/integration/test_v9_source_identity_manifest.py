"""Schema-v9 connector-declared logical Source identity integration tests."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from dithyramba.collections import (
    CollectionConfig,
    CollectionKind,
    CollectionRoot,
    SourceIdentityManifestError,
    load_source_identity_manifest,
)
from dithyramba.collections import identity_manifest as identity_manifest_module
from dithyramba.contracts import canonical_json_bytes, new_id, sha256_hex
from dithyramba.ingest.errors import InvalidSourcePathError
from dithyramba.ingest.service import (
    IngestService,
    _canonical_source_lock_key,
    _replace_or_append_outcome,
)
from dithyramba.library import LibraryConfig
from dithyramba.persistence import LibraryRepository, PersistenceIntegrityError, initialize_library
from dithyramba.provenance import (
    IngestDisposition,
    IngestInputOutcome,
    SourceFamilyRole,
    SourceLineageIntegrityError,
    TerminalInputOutcome,
)


def _manifest(root: Path, identities: list[dict[str, object]]) -> tuple[Path, str]:
    path = root / "identity-manifest.json"
    payload = {
        "schema": "dithyramba.source_identity_manifest/1.0",
        "identities": identities,
    }
    data = canonical_json_bytes(payload)
    path.write_bytes(data)
    return path, sha256_hex(data)


def _identity(uri: str, parts: list[tuple[str, int]]) -> dict[str, object]:
    return {
        "logical_source_uri": uri,
        "connector": "books",
        "connector_revision": "books/1.0",
        "representation": "ocr-text",
        "parts": [{"path": path, "part": part} for path, part in parts],
    }


def _collection(
    repository: LibraryRepository,
    root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    name: str,
) -> str:
    created = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name=name,
            kind=CollectionKind.CORPUS,
            roots=(
                CollectionRoot(
                    path=root,
                    identity_manifest_path=manifest_path,
                    identity_manifest_sha256=manifest_sha256,
                ),
            ),
        )
    )
    return created.config.collection_id


def test_declared_identity_is_library_wide_and_first_representation_is_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "source.md").write_text("# First\n\nOriginal scan.\n", encoding="utf-8")
    (second_root / "source.md").write_text("# Second\n\nCorrected OCR.\n", encoding="utf-8")
    uri = "urn:dithyramba:book:cross-collection"
    first_manifest, first_hash = _manifest(first_root, [_identity(uri, [("source.md", 0)])])
    second_manifest, second_hash = _manifest(second_root, [_identity(uri, [("source.md", 0)])])

    with initialize_library(
        LibraryConfig(name="V9 cross collection"), data_root=tmp_path / "data"
    ) as repo:
        first = _collection(repo, first_root, first_manifest, first_hash, "First")
        second = _collection(repo, second_root, second_manifest, second_hash, "Second")
        first_outcome = IngestService(repo).ingest_collection(first).outcomes[0]
        second_outcome = IngestService(repo).ingest_collection(second).outcomes[0]
        repeated = IngestService(repo).ingest_collection(second).outcomes[0]

        assert first_outcome.source_family_id == second_outcome.source_family_id
        assert first_outcome.root_source_id == first_outcome.source_id
        assert second_outcome.root_source_id == first_outcome.source_id
        records = {record.source_id: record for record in repo.list_sources()}
        assert records[first_outcome.source_id or ""].family_role is SourceFamilyRole.ROOT
        assert records[second_outcome.source_id or ""].family_role is SourceFamilyRole.DERIVATIVE
        assert repeated.disposition is IngestDisposition.UNCHANGED
        assert (
            repo._store.connection.execute(
                "SELECT count(*) FROM source_identity_bindings"
            ).fetchone()[0]
            == 2
        )
        assert (
            repo._store.connection.execute(
                "SELECT count(*) FROM event_outbox "
                "WHERE event_type = 'source_identity.binding_recorded'"
            ).fetchone()[0]
            == 2
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repo._store.connection.execute("UPDATE source_identity_bindings SET rowid = rowid")

        declared = load_source_identity_manifest(
            CollectionRoot(
                path=first_root,
                identity_manifest_path=first_manifest,
                identity_manifest_sha256=first_hash,
            )
        )
        assert declared is not None
        declaration = declared.declaration_for("source.md")
        with pytest.raises(SourceLineageIntegrityError, match="rebinding is forbidden"):
            repo._declared_lineage_for_existing_source(
                repo._store.connection,
                source_id=first_outcome.source_id or "",
                declaration=replace(declaration, logical_source_uri="urn:dithyramba:book:wrong"),
            )
        with pytest.raises(SourceLineageIntegrityError, match="immutable SourceVersion binding"):
            repo._require_identity_binding(
                repo._store.connection,
                source_id=first_outcome.source_id or "",
                source_version_id=first_outcome.source_version_id or "",
                declaration=replace(declaration, connector="different"),
            )
        monkeypatch.setattr(
            repo,
            "_lineage_for_source",
            lambda _connection, _source_id: (
                new_id("family"),
                first_outcome.source_id or "",
                SourceFamilyRole.ROOT,
            ),
        )
        with pytest.raises(SourceLineageIntegrityError, match="binding disagrees"):
            repo._declared_lineage_for_existing_source(
                repo._store.connection,
                source_id=first_outcome.source_id or "",
                declaration=declaration,
            )
        orphan_family_id = new_id("family")
        repo._store.connection.execute(
            "INSERT INTO source_families(source_family_id, library_id, label, created_at) "
            "VALUES (?, ?, ?, ?)",
            (orphan_family_id, repo.library_id, "orphan", "2026-07-26T00:00:00.000000Z"),
        )
        repo._store.connection.execute(
            "INSERT INTO source_family_identities(source_family_id, library_id, "
            "logical_source_uri, created_at) VALUES (?, ?, ?, ?)",
            (
                orphan_family_id,
                repo.library_id,
                "urn:dithyramba:book:orphan",
                "2026-07-26T00:00:00.000000Z",
            ),
        )
        with pytest.raises(SourceLineageIntegrityError, match="exactly one technical root"):
            repo._declared_lineage_for_new_source(
                repo._store.connection,
                declaration=replace(declaration, logical_source_uri="urn:dithyramba:book:orphan"),
                content_sha256="0" * 64,
            )


def test_identical_bytes_with_distinct_declared_identities_do_not_merge(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    payload = "# Shared\n\nByte-identical physical representation.\n"
    (root / "a.md").write_text(payload, encoding="utf-8")
    (root / "b.md").write_text(payload, encoding="utf-8")
    manifest, manifest_hash = _manifest(
        root,
        [
            _identity("urn:dithyramba:book:one", [("a.md", 0)]),
            _identity("urn:dithyramba:book:two", [("b.md", 0)]),
        ],
    )

    with initialize_library(
        LibraryConfig(name="V9 distinct identity"), data_root=tmp_path / "data"
    ) as repo:
        collection = _collection(repo, root, manifest, manifest_hash, "Corpus")
        outcomes = IngestService(repo).ingest_collection(collection).outcomes

        assert outcomes[0].source_family_id != outcomes[1].source_family_id
        assert {item.root_source_id for item in outcomes} == {item.source_id for item in outcomes}


def test_pinned_identity_manifest_tamper_fails_closed_on_collection_reload(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nBody.\n", encoding="utf-8")
    manifest, manifest_hash = _manifest(
        root,
        [_identity("urn:dithyramba:book:tamper", [("a.md", 0)])],
    )

    with initialize_library(LibraryConfig(name="V9 tamper"), data_root=tmp_path / "data") as repo:
        collection = _collection(repo, root, manifest, manifest_hash, "Corpus")
        manifest.write_bytes(
            canonical_json_bytes(
                {
                    "schema": "dithyramba.source_identity_manifest/1.0",
                    "identities": [_identity("urn:dithyramba:book:changed", [("a.md", 0)])],
                }
            )
        )
        with pytest.raises(PersistenceIntegrityError, match="identity manifest"):
            repo.get_collection(collection)


def test_identity_manifest_symlink_is_rejected_even_when_its_hash_matches(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nBody.\n", encoding="utf-8")
    manifest, manifest_hash = _manifest(
        root,
        [_identity("urn:dithyramba:book:symlink", [("a.md", 0)])],
    )
    symlink = root / "identity-link.json"
    symlink.symlink_to(manifest.name)

    with (
        initialize_library(LibraryConfig(name="V9 symlink"), data_root=tmp_path / "data") as repo,
        pytest.raises(SourceIdentityManifestError, match="non-symlink"),
    ):
        _collection(repo, root, symlink, manifest_hash, "Corpus")


def test_manifest_requires_complete_mapping_and_contiguous_parts(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nBody.\n", encoding="utf-8")
    (root / "b.md").write_text("# B\n\nBody.\n", encoding="utf-8")
    manifest, manifest_hash = _manifest(
        root,
        [_identity("urn:dithyramba:book:incomplete", [("a.md", 1)])],
    )

    with (
        initialize_library(
            LibraryConfig(name="V9 complete map"), data_root=tmp_path / "data"
        ) as repo,
        pytest.raises(SourceIdentityManifestError, match="contiguous"),
    ):
        _collection(repo, root, manifest, manifest_hash, "Corpus")


def test_loader_rejects_malformed_administrative_contracts_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nBody.\n", encoding="utf-8")
    path = root / "identity-manifest.json"
    assert load_source_identity_manifest(CollectionRoot(path=root)) is None

    def reject(payload: object, message: str) -> None:
        data = canonical_json_bytes(payload)
        path.write_bytes(data)
        configured = CollectionRoot(
            path=root,
            identity_manifest_path=path,
            identity_manifest_sha256=sha256_hex(data),
        )
        with pytest.raises(SourceIdentityManifestError, match=message):
            load_source_identity_manifest(configured)

    base = {
        "schema": "dithyramba.source_identity_manifest/1.0",
        "identities": [_identity("urn:dithyramba:book:valid", [("a.md", 0)])],
    }
    data = canonical_json_bytes(base)
    path.write_bytes(data)
    configured = CollectionRoot(
        path=root,
        identity_manifest_path=path,
        identity_manifest_sha256=sha256_hex(data),
    )
    manifest = load_source_identity_manifest(configured)
    assert manifest is not None
    with pytest.raises(SourceIdentityManifestError, match="no declaration"):
        manifest.declaration_for("missing.md")

    with pytest.raises(SourceIdentityManifestError, match="hash has no manifest path"):
        load_source_identity_manifest(CollectionRoot(path=root, identity_manifest_sha256="0" * 64))
    with pytest.raises(SourceIdentityManifestError, match="path has no pinned"):
        load_source_identity_manifest(CollectionRoot(path=root, identity_manifest_path=path))
    with pytest.raises(SourceIdentityManifestError, match="cannot be read"):
        load_source_identity_manifest(
            CollectionRoot(
                path=root,
                identity_manifest_path=root / "missing.json",
                identity_manifest_sha256="0" * 64,
            )
        )

    path.write_bytes(b"{")
    with pytest.raises(SourceIdentityManifestError, match="valid UTF-8 JSON"):
        load_source_identity_manifest(
            CollectionRoot(
                path=root,
                identity_manifest_path=path,
                identity_manifest_sha256=sha256_hex(b"{"),
            )
        )
    reject([], "exactly schema")
    reject({"schema": "wrong", "identities": []}, "schema is unsupported")
    reject({"schema": base["schema"], "identities": {}}, "non-empty list")
    reject({"schema": base["schema"], "identities": ["not-an-object"]}, "declared fields")

    invalid_parts = _identity("urn:dithyramba:book:parts", [("a.md", 0)])
    invalid_parts["parts"] = []
    reject({"schema": base["schema"], "identities": [invalid_parts]}, "non-empty list")
    invalid_part = _identity("urn:dithyramba:book:parts", [("a.md", 0)])
    invalid_part["parts"] = ["not-an-object"]
    reject({"schema": base["schema"], "identities": [invalid_part]}, "exactly path and part")
    negative_part = _identity("urn:dithyramba:book:parts", [("a.md", 0)])
    negative_part["parts"] = [{"path": "a.md", "part": -1}]
    reject({"schema": base["schema"], "identities": [negative_part]}, "non-negative")
    reject(
        {
            "schema": base["schema"],
            "identities": [
                _identity("urn:dithyramba:book:one", [("a.md", 0)]),
                _identity("urn:dithyramba:book:two", [("a.md", 0)]),
            ],
        },
        "more than once",
    )

    invalid_path = _identity("urn:dithyramba:book:path", [("../a.md", 0)])
    reject({"schema": base["schema"], "identities": [invalid_path]}, "stay below")
    empty_path = _identity("urn:dithyramba:book:path", [("", 0)])
    reject({"schema": base["schema"], "identities": [empty_path]}, "POSIX relative")
    invalid_uri = _identity("", [("a.md", 0)])
    reject({"schema": base["schema"], "identities": [invalid_uri]}, "non-empty unpadded")
    file_uri = _identity("file:///a.md", [("a.md", 0)])
    reject({"schema": base["schema"], "identities": [file_uri]}, "non-file absolute")
    invalid_connector = _identity("urn:dithyramba:book:connector", [("a.md", 0)])
    invalid_connector["connector"] = "Books"
    reject({"schema": base["schema"], "identities": [invalid_connector]}, "label grammar")
    invalid_revision = _identity("urn:dithyramba:book:revision", [("a.md", 0)])
    invalid_revision["connector_revision"] = " "
    reject({"schema": base["schema"], "identities": [invalid_revision]}, "non-empty unpadded")

    (root / "b.md").write_text("# B\n\nBody.\n", encoding="utf-8")
    reject(base, "map every eligible")


def test_loader_rejects_declared_symlink_path_and_path_check_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    declaration = identity_manifest_module.SourceIdentityDeclaration(
        logical_source_uri="urn:dithyramba:book:direct",
        connector="books",
        connector_revision="books/1.0",
        representation="ocr-text",
        part_number=0,
        part_metadata_hash="0" * 64,
        manifest_sha256="1" * 64,
    )
    configured = CollectionRoot(path=root)
    from dithyramba.ingest import reader as reader_module

    monkeypatch.setattr(reader_module, "discover_source_paths", lambda _root: ("a.md",))
    monkeypatch.setattr(os.path, "islink", lambda _path: True)
    with pytest.raises(SourceIdentityManifestError, match="must not be a symlink"):
        identity_manifest_module._validate_complete_root_mapping(configured, {"a.md": declaration})

    monkeypatch.setattr(
        os.path,
        "islink",
        lambda _path: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(SourceIdentityManifestError, match="cannot be checked"):
        identity_manifest_module._validate_complete_root_mapping(configured, {"a.md": declaration})


def test_root_pin_models_and_ingest_manifest_failure_are_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nBody.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity_manifest_path"):
        CollectionRoot(path=root, identity_manifest_path=Path("relative.json"))
    with pytest.raises(ValueError, match="identity_manifest_sha256"):
        CollectionRoot(path=root, identity_manifest_sha256="not-a-hash")

    manifest, manifest_hash = _manifest(
        root,
        [_identity("urn:dithyramba:book:service", [("a.md", 0)])],
    )
    with initialize_library(
        LibraryConfig(name="V9 service failure"), data_root=tmp_path / "data"
    ) as repo:
        collection = _collection(repo, root, manifest, manifest_hash, "Corpus")
        record = repo.get_collection(collection)
        outcome = IngestService(repo)._ingest_one(
            collection_id=collection,
            collection_root_id=record.collection_root_ids[0],
            root=CollectionRoot(
                path=root,
                identity_manifest_path=manifest,
                identity_manifest_sha256="0" * 64,
            ),
            relative_path="a.md",
        )
        assert outcome.failure_code == "source_identity_manifest_invalid"


def test_ingest_path_and_terminal_replacement_guards_are_exact(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    configured = CollectionRoot(path=root)
    with pytest.raises(InvalidSourcePathError, match="non-empty text"):
        _canonical_source_lock_key(configured, "")
    with pytest.raises(InvalidSourcePathError, match="POSIX separators"):
        _canonical_source_lock_key(configured, "a\\b.md")

    original = IngestInputOutcome(
        collection_root_id="root_test",
        relative_path="a.md",
        terminal_outcome=TerminalInputOutcome.FAILED,
        failure_code="first_failure",
    )
    replacement = IngestInputOutcome(
        collection_root_id="root_test",
        relative_path="a.md",
        terminal_outcome=TerminalInputOutcome.FAILED,
        failure_code="second_failure",
    )
    outcomes = [original]
    _replace_or_append_outcome(outcomes, replacement)
    assert outcomes == [replacement]
