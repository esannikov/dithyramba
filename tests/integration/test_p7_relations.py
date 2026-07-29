"""P7 Relation import, policy-before-graph and complete path receipt tests."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

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
from dithyramba.persistence import SQLiteRelationRepository
from dithyramba.persistence.repository import LibraryRepository, initialize_library
from dithyramba.relations import (
    RelationAuthorizationError,
    RelationDirection,
    RelationImportRequest,
    RelationIntegrityError,
    RelationNodeRef,
    RelationNodeType,
    RelationPathRequest,
    RelationProposal,
    RelationType,
)

NOW = datetime(2026, 7, 21, 15, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-21T15:00:00.000000Z"


def _clock() -> datetime:
    return NOW


def _context(
    tmp_path: Path,
) -> tuple[
    LibraryRepository,
    SQLiteRelationRepository,
    str,
    AccessPolicySnapshot,
    RequestScope,
    tuple[str, ...],
    tuple[str, ...],
]:
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    source_root.mkdir()
    repository = initialize_library(
        LibraryConfig(name="Relations"), data_root=data_root, clock=_clock
    )
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="Corpus",
            kind=CollectionKind.CORPUS,
            roots=(build_collection_root(source_root, data_root=data_root),),
        )
    )
    fragment_ids, text_hashes = _seed_source(repository, collection.config.collection_id)
    snapshot = repository.freeze_snapshot((collection.config.collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id="policy_relations",
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
    statement_ids, evidence_ids = _seed_meaning(
        repository,
        collection.config.collection_id,
        snapshot.corpus_snapshot_id,
        policy.access_policy_id,
        fragment_ids,
        text_hashes,
    )
    return (
        repository,
        SQLiteRelationRepository(repository),
        snapshot.corpus_snapshot_id,
        policy,
        scope,
        statement_ids,
        evidence_ids,
    )


def _seed_source(
    repository: LibraryRepository, collection_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    connection = repository._store.connection
    texts = (
        "Embodiment supports situated agency.",
        "Situated agency enables adaptive action.",
        "Disembodied prediction can still guide action.",
    )
    source_bytes = "\n".join(texts).encode()
    content_hash = sha256_hex(source_bytes)
    connection.execute(
        "INSERT INTO blobs VALUES (?, ?, 'bb/blob', ?)",
        (content_hash, len(source_bytes), NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO sources VALUES "
        "('source_relations', ?, 'file:///relations.md', 'text/markdown', 'Relations', ?)",
        (repository.library_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_versions VALUES "
        "('source_version_relations', 'source_relations', 1, ?, ?, ?, NULL, "
        "'markdown_v1', 'processed', NULL)",
        (content_hash, len(source_bytes), NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_heads VALUES ('source_relations', 'source_version_relations', ?, NULL)",
        (NOW_TEXT,),
    )
    connection.execute(
        "INSERT INTO source_families VALUES ('family_relations', ?, 'Family', ?)",
        (repository.library_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_family_members VALUES "
        "('family_relations', 'source_relations', 'root', NULL, ?)",
        (NOW_TEXT,),
    )
    fragment_ids: list[str] = []
    hashes: list[str] = []
    for ordinal, text in enumerate(texts):
        fragment_id = f"fragment_relations_{ordinal}"
        text_hash = sha256_hex(text.encode())
        address = {"kind": "markdown", "heading_path": [], "paragraph_index": ordinal}
        connection.execute(
            "INSERT INTO source_fragments VALUES (?, 'source_version_relations', ?, "
            "'paragraph', ?, ?, ?, ?)",
            (
                fragment_id,
                ordinal,
                text,
                text_hash,
                canonical_json_bytes(address).decode(),
                canonical_sha256_hex(address),
            ),
        )
        fragment_ids.append(fragment_id)
        hashes.append(text_hash)
    connection.execute(
        "INSERT INTO collection_memberships VALUES (?, 'source_relations', 'active', ?)",
        (collection_id, NOW_TEXT),
    )
    return tuple(fragment_ids), tuple(hashes)


def _seed_meaning(
    repository: LibraryRepository,
    collection_id: str,
    snapshot_id: str,
    policy_id: str,
    fragment_ids: tuple[str, ...],
    text_hashes: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    connection = repository._store.connection
    hashes = (f"{number:064x}" for number in range(100, 112))
    values = tuple(hashes)
    connection.execute(
        """
        INSERT INTO meaning_processing_runs VALUES (
            'run_meaning_relations', ?, 'meaning_import', ?, ?, ?, ?, 'deep',
            'p7', '{}', ?, '{}', ?, ?, ?, ?, 'succeeded', 7, 0, 0, NULL,
            '{}', ?, ?, ?, ?
        )
        """,
        (
            repository.library_id,
            snapshot_id,
            policy_id,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            NOW_TEXT,
            NOW_TEXT,
        ),
    )
    connection.execute(
        "INSERT INTO voices VALUES "
        "('voice_relations', ?, 'author', 'Author', 'candidate', ?, "
        "'run_meaning_relations', ?)",
        (repository.library_id, values[9], NOW_TEXT),
    )
    statement_ids = tuple(f"statement_relations_{number}" for number in range(3))
    evidence_ids = tuple(f"evidence_relations_{number}" for number in range(3))
    texts = (
        "Embodiment supports situated agency.",
        "Situated agency enables adaptive action.",
        "Disembodied prediction can still guide action.",
    )
    with repository._permit_fragment_text_read():
        for number, (statement_id, evidence_id, fragment_id, text, text_hash) in enumerate(
            zip(statement_ids, evidence_ids, fragment_ids, texts, text_hashes, strict=True)
        ):
            connection.execute(
                "INSERT INTO statements VALUES (?, ?, ?, 'source_claim', ?, "
                "'voice_relations', NULL, 'candidate', 'active', ?, "
                "'run_meaning_relations', ?)",
                (
                    statement_id,
                    repository.library_id,
                    collection_id,
                    text,
                    f"{200 + number:064x}",
                    NOW_TEXT,
                ),
            )
            connection.execute(
                "INSERT INTO evidence_links VALUES (?, ?, ?, 'voice_relations', "
                "'supports', 'exact', NULL, 0, ?, ?, ?, 'human_import/1.0', "
                "'candidate', ?, 'run_meaning_relations', ?)",
                (
                    evidence_id,
                    statement_id,
                    fragment_id,
                    len(text),
                    text,
                    text_hash,
                    f"{300 + number:064x}",
                    NOW_TEXT,
                ),
            )
    return statement_ids, evidence_ids


def _seed_private_entity(
    repository: LibraryRepository,
    tmp_path: Path,
) -> tuple[str, str]:
    data_root = tmp_path / "data"
    private_root = tmp_path / "private-source"
    private_root.mkdir(exist_ok=True)
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="Private corpus",
            kind=CollectionKind.CORPUS,
            roots=(build_collection_root(private_root, data_root=data_root),),
        )
    )
    collection_id = collection.config.collection_id
    text = "Private person is never named in corpus A."
    content_hash = sha256_hex(text.encode())
    address = {"kind": "markdown", "heading_path": [], "paragraph_index": 0}
    connection = repository._store.connection
    connection.execute(
        "INSERT INTO blobs VALUES (?, ?, 'cc/private-entity', ?)",
        (content_hash, len(text.encode()), NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO sources VALUES "
        "('source_private_b', ?, 'file:///private-b.md', 'text/markdown', "
        "'Private B', ?)",
        (repository.library_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_versions VALUES "
        "('source_version_private_b', 'source_private_b', 1, ?, ?, ?, NULL, "
        "'markdown_v1', 'processed', NULL)",
        (content_hash, len(text.encode()), NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_heads VALUES ('source_private_b', 'source_version_private_b', ?, NULL)",
        (NOW_TEXT,),
    )
    connection.execute(
        "INSERT INTO source_families VALUES ('family_private_b', ?, 'Private B', ?)",
        (repository.library_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_family_members VALUES "
        "('family_private_b', 'source_private_b', 'root', NULL, ?)",
        (NOW_TEXT,),
    )
    connection.execute(
        "INSERT INTO source_fragments VALUES "
        "('fragment_private_b', 'source_version_private_b', 0, 'paragraph', ?, ?, ?, ?)",
        (
            text,
            content_hash,
            canonical_json_bytes(address).decode(),
            canonical_sha256_hex(address),
        ),
    )
    connection.execute(
        "INSERT INTO collection_memberships VALUES (?, 'source_private_b', 'active', ?)",
        (collection_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO entities VALUES "
        "('entity_private_b', ?, 'person', 'Private Person', 'candidate', ?, "
        "'run_meaning_relations', ?)",
        (repository.library_id, "6" * 64, NOW_TEXT),
    )
    quote = "Private person"
    with repository._permit_fragment_text_read():
        connection.execute(
            "INSERT INTO entity_mentions VALUES "
            "('entity_mention_private_b', 'entity_private_b', ?, "
            "'fragment_private_b', 0, ?, ?, ?, ?)",
            (
                collection_id,
                len(quote),
                quote,
                sha256_hex(quote.encode()),
                "7" * 64,
            ),
        )
    return collection_id, "entity_private_b"


def _import_request(
    snapshot_id: str, policy: AccessPolicySnapshot, scope: RequestScope
) -> RelationImportRequest:
    return RelationImportRequest(
        snapshot_id,
        policy.access_policy_id,
        scope,
        "p7",
        "relations/1.0",
    )


def _proposals(
    statement_ids: tuple[str, ...], evidence_ids: tuple[str, ...]
) -> tuple[RelationProposal, ...]:
    nodes = tuple(RelationNodeRef(RelationNodeType.STATEMENT, item) for item in statement_ids)
    return (
        RelationProposal(
            RelationType.SUPPORTS,
            nodes[0],
            nodes[1],
            evidence_ids[:2],
            "Embodiment grounds the transition to situated agency.",
            "human_import/1.0",
        ),
        RelationProposal(
            RelationType.CONTRADICTS,
            nodes[1],
            nodes[2],
            evidence_ids[1:],
            "The two statements disagree about whether embodiment is required.",
            "human_import/1.0",
        ),
    )


def test_relation_import_and_two_hop_receipt_round_trip(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        result = relations.import_relations(
            _import_request(snapshot_id, policy, scope), _proposals(statements, evidence)
        )
        assert (
            relations.import_relations(
                _import_request(snapshot_id, policy, scope), _proposals(statements, evidence)
            )
            == result
        )
        assert relations.load_import_result(result.receipt.import_run_id) == result
        assert relations.load_relation(result.relations[0].relation_id) == result.relations[0]

        request = RelationPathRequest(
            snapshot_id,
            policy.access_policy_id,
            scope,
            (RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),),
            (RelationType.SUPPORTS, RelationType.CONTRADICTS),
            RelationDirection.OUTGOING,
            2,
            10,
        )
        receipt = relations.traverse(request)

        assert sorted(len(path.steps) for path in receipt.paths) == [1, 2]
        assert all(step.evidence_link_ids for path in receipt.paths for step in path.steps)
        assert relations.load_path_receipt(receipt.receipt_id) == receipt
        assert relations.traverse(request) == receipt
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repository._store.connection.execute(
                "UPDATE relations SET reason_text = 'changed' WHERE relation_id = ?",
                (result.relations[0].relation_id,),
            )
    finally:
        repository.close()


def test_policy_exclusion_removes_relation_before_recursive_traversal(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        relations.import_relations(
            _import_request(snapshot_id, policy, scope), _proposals(statements, evidence)
        )
        denied_scope = RequestScope(
            library_id=scope.library_id,
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
            exclusions=QueryExclusions(source_fragment_ids=("fragment_relations_2",)),
        )
        receipt = relations.traverse(
            RelationPathRequest(
                snapshot_id,
                policy.access_policy_id,
                denied_scope,
                (RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),),
                (RelationType.SUPPORTS, RelationType.CONTRADICTS),
                RelationDirection.OUTGOING,
                3,
                10,
            )
        )

        assert len(receipt.paths) == 1
        assert len(receipt.paths[0].steps) == 1
        assert receipt.paths[0].steps[0].relation_type is RelationType.SUPPORTS
        assert evidence[2] not in receipt.payload().__repr__()
    finally:
        repository.close()


def test_import_refuses_evidence_or_statement_outside_exact_scope(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        denied_scope = RequestScope(
            library_id=scope.library_id,
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
            exclusions=QueryExclusions(source_fragment_ids=("fragment_relations_2",)),
        )
        with pytest.raises(RelationAuthorizationError, match="outside"):
            relations.import_relations(
                _import_request(snapshot_id, policy, denied_scope),
                (_proposals(statements, evidence)[1],),
            )
        bad = RelationProposal(
            RelationType.SUPPORTS,
            RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),
            RelationNodeRef(RelationNodeType.STATEMENT, statements[1]),
            (evidence[0],),
            "Missing object evidence.",
            "human_import/1.0",
        )
        with pytest.raises(RelationIntegrityError, match="own exact"):
            relations.import_relations(_import_request(snapshot_id, policy, scope), (bad,))

        _private_collection_id, private_entity_id = _seed_private_entity(repository, tmp_path)
        private_endpoint = RelationProposal(
            RelationType.SUPPORTS,
            RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),
            RelationNodeRef(RelationNodeType.ENTITY, private_entity_id),
            (evidence[0],),
            "A public statement is incorrectly linked to a private-only entity.",
            "human_import/1.0",
        )
        with pytest.raises(RelationAuthorizationError, match="endpoint grounding"):
            relations.import_relations(
                _import_request(snapshot_id, policy, scope),
                (private_endpoint,),
            )
    finally:
        repository.close()


def test_import_revalidates_endpoint_grounding_inside_write_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    second_connection: sqlite3.Connection | None = None
    try:
        private_collection_id, _private_entity_id = _seed_private_entity(repository, tmp_path)
        connection = repository._store.connection
        connection.execute(
            "INSERT INTO entities VALUES "
            "('entity_grounding_race', ?, 'person', 'Grounding race', 'candidate', ?, "
            "'run_meaning_relations', ?)",
            (repository.library_id, "8" * 64, NOW_TEXT),
        )
        public_quote = "Embodiment"
        with repository._permit_fragment_text_read():
            connection.execute(
                "INSERT INTO entity_mentions VALUES "
                "('entity_mention_grounding_race_public', 'entity_grounding_race', ?, "
                "'fragment_relations_0', 0, ?, ?, ?, ?)",
                (
                    scope.collection_ids[0],
                    len(public_quote),
                    public_quote,
                    sha256_hex(public_quote.encode()),
                    "9" * 64,
                ),
            )

        database_path = str(connection.execute("PRAGMA database_list").fetchone()[2])
        second_connection = sqlite3.connect(database_path, isolation_level=None)
        second_connection.execute("PRAGMA foreign_keys = ON")
        injected = False

        def inject_private_grounding(_clock: object) -> str:
            nonlocal injected
            private_quote = "Private person"
            second_connection.execute(
                "INSERT INTO entity_mentions VALUES "
                "('entity_mention_grounding_race_private', 'entity_grounding_race', ?, "
                "'fragment_private_b', 0, ?, ?, ?, ?)",
                (
                    private_collection_id,
                    len(private_quote),
                    private_quote,
                    sha256_hex(private_quote.encode()),
                    "a" * 64,
                ),
            )
            injected = True
            return NOW_TEXT

        monkeypatch.setattr(
            "dithyramba.persistence.relations._timestamp",
            inject_private_grounding,
        )
        proposal = RelationProposal(
            RelationType.SUPPORTS,
            RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),
            RelationNodeRef(RelationNodeType.ENTITY, "entity_grounding_race"),
            (evidence[0],),
            "A Relation whose endpoint grounding expands before its write transaction.",
            "human_import/1.0",
        )

        with pytest.raises(RelationAuthorizationError, match="endpoint grounding"):
            relations.import_relations(
                _import_request(snapshot_id, policy, scope),
                (proposal,),
            )

        assert injected is True
        assert set(relations._node_grounding_fragments(connection, proposal.object_node)) == {
            "fragment_relations_0",
            "fragment_private_b",
        }
        assert connection.execute("SELECT COUNT(*) FROM relation_import_runs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0] == 0
    finally:
        if second_connection is not None:
            second_connection.close()
        repository.close()


def test_import_and_traversal_reject_snapshot_id_hash_mismatch(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        mismatched_scope = RequestScope(
            library_id=scope.library_id,
            snapshot_hash="f" * 64,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
        )
        with pytest.raises(RelationAuthorizationError, match="ID/hash"):
            relations.import_relations(
                _import_request(snapshot_id, policy, mismatched_scope),
                _proposals(statements, evidence),
            )
        with pytest.raises(RelationAuthorizationError, match="ID/hash"):
            relations.traverse(
                RelationPathRequest(
                    snapshot_id,
                    policy.access_policy_id,
                    mismatched_scope,
                    (RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),),
                    (RelationType.SUPPORTS,),
                    RelationDirection.OUTGOING,
                    1,
                    10,
                )
            )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM relation_import_runs"
            ).fetchone()[0]
            == 0
        )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM relation_path_receipts"
            ).fetchone()[0]
            == 0
        )
    finally:
        repository.close()


def test_traversal_excludes_relation_with_denied_endpoint_grounding(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        connection = repository._store.connection
        connection.execute(
            "INSERT INTO entities VALUES "
            "('entity_denied_grounding', ?, 'work', 'Denied work', 'candidate', ?, "
            "'run_meaning_relations', ?)",
            (repository.library_id, "4" * 64, NOW_TEXT),
        )
        quote = "Disembodied prediction"
        with repository._permit_fragment_text_read():
            connection.execute(
                "INSERT INTO entity_mentions VALUES "
                "('entity_mention_denied_grounding', 'entity_denied_grounding', "
                "(SELECT collection_id FROM collections WHERE library_id = ?), "
                "'fragment_relations_2', 0, ?, ?, ?, ?)",
                (
                    repository.library_id,
                    len(quote),
                    quote,
                    sha256_hex(quote.encode()),
                    "5" * 64,
                ),
            )
        proposal = RelationProposal(
            RelationType.SUPPORTS,
            RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),
            RelationNodeRef(RelationNodeType.ENTITY, "entity_denied_grounding"),
            (evidence[0],),
            "The source statement names an entity grounded elsewhere.",
            "human_import/1.0",
        )
        relations.import_relations(_import_request(snapshot_id, policy, scope), (proposal,))

        denied_scope = RequestScope(
            library_id=scope.library_id,
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
            exclusions=QueryExclusions(source_fragment_ids=("fragment_relations_2",)),
        )
        receipt = relations.traverse(
            RelationPathRequest(
                snapshot_id,
                policy.access_policy_id,
                denied_scope,
                (RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),),
                (RelationType.SUPPORTS,),
                RelationDirection.OUTGOING,
                1,
                10,
            )
        )

        assert receipt.paths == ()
        assert "entity_denied_grounding" not in receipt.payload().__repr__()
    finally:
        repository.close()
