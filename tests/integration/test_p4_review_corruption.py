"""Fail-closed reconstruction tests for corrupted persisted review targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    new_id,
)
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import LibraryRepository, initialize_library
from dithyramba.persistence.recall import SQLiteRecallBackend
from dithyramba.persistence.review import ReviewDecisionRepository
from dithyramba.recall import QueryRequest, RecallService, current_fts_runtime_profile
from dithyramba.review import (
    ReviewAction,
    ReviewDecisionNotFoundError,
    ReviewDecisionRequest,
    ReviewIntegrityError,
    ReviewScope,
    ReviewTargetType,
)
from dithyramba.review.service import ReviewService


def _collection(repository: LibraryRepository, root: Path, name: str) -> str:
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


def _source(
    repository: LibraryRepository,
    root: Path,
    *,
    collection_name: str = "Corruption corpus",
    filename: str = "evidence.md",
) -> tuple[str, str, str]:
    collection_id = _collection(repository, root, collection_name)
    outcome = IngestService(repository).ingest_path(collection_id, filename).outcomes[0]
    assert outcome.source_version_id is not None
    fragment = repository.list_source_fragments(outcome.source_version_id)[0]
    return collection_id, outcome.source_version_id, fragment.source_fragment_id


def _source_review(
    repository: LibraryRepository,
    *,
    collection_id: str,
    fragment_id: str,
) -> tuple[str, str]:
    service = ReviewService(repository)
    target = service.target(ReviewTargetType.SOURCE_FRAGMENT, fragment_id)
    decision = service.decide(
        ReviewDecisionRequest(
            target_type=ReviewTargetType.SOURCE_FRAGMENT,
            target_id=fragment_id,
            target_hash=target.target_hash,
            action=ReviewAction.ACCEPT,
            reason="The exact persisted fragment supports this bounded use.",
            authority="researcher:coverage",
            scope=ReviewScope(collection_ids=(collection_id,), use="research"),
        )
    )
    return decision.review_decision_id, target.target_hash


def _prepare_packet(
    repository: LibraryRepository,
    *,
    collection_id: str,
    question: str,
) -> str:
    snapshot = repository.freeze_snapshot((collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id=new_id("policy"),
        library_id=repository.library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
    )
    existing_policy = repository._store.connection.execute(
        "SELECT access_policy_id FROM access_policies WHERE policy_hash = ?",
        (policy.policy_hash,),
    ).fetchone()
    if existing_policy is None:
        repository.persist_access_policy(name="Corruption policy", snapshot=policy)
    else:
        policy = AccessPolicySnapshot(
            access_policy_id=str(existing_policy[0]),
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
        )
    profile = current_fts_runtime_profile()
    result = RecallService(
        SQLiteRecallBackend(repository),
        profile_version=profile.profile_version,
    ).recall(
        QueryRequest(
            question=question,
            library_id=repository.library_id,
            collection_ids=(collection_id,),
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            purpose="research",
        )
    )
    assert result.packet.source_fragments
    return result.packet.evidence_packet_id


def _packet_payload(repository: LibraryRepository, packet_id: str) -> dict[str, object]:
    row = repository._store.connection.execute(
        "SELECT packet_json FROM evidence_packets WHERE evidence_packet_id = ?",
        (packet_id,),
    ).fetchone()
    assert row is not None
    return cast(dict[str, object], json.loads(str(row[0])))


def _reidentify_packet(payload: dict[str, object]) -> tuple[str, str, str]:
    semantic = dict(payload)
    semantic.pop("evidence_packet_id", None)
    semantic.pop("packet_hash", None)
    packet_hash = canonical_sha256_hex(semantic)
    packet_id = canonical_content_id("packet", semantic)
    stored = {**semantic, "evidence_packet_id": packet_id, "packet_hash": packet_hash}
    return packet_id, packet_hash, canonical_json_bytes(stored).decode("utf-8")


def _drop_update_trigger(repository: LibraryRepository, table: str) -> None:
    repository._store.connection.execute(f"DROP TRIGGER {table}_no_update")


def test_repository_rejects_unknown_target_type_and_missing_decisions(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "evidence.md").write_text("# Evidence\n\nExact review target.\n", encoding="utf-8")

    with initialize_library(
        LibraryConfig(name="Review missing-state checks"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, _version_id, fragment_id = _source(repository, root)
        review_repository = ReviewDecisionRepository(repository)
        target = review_repository.get_target(ReviewTargetType.SOURCE_FRAGMENT, fragment_id)

        with pytest.raises(ReviewIntegrityError, match="unsupported"):
            review_repository.get_target(cast(ReviewTargetType, object()), fragment_id)
        with pytest.raises(ReviewDecisionNotFoundError, match="does not exist"):
            review_repository.get("review_absent")
        with pytest.raises(ReviewDecisionNotFoundError, match="does not exist"):
            review_repository.create(
                ReviewDecisionRequest(
                    target_type=ReviewTargetType.SOURCE_FRAGMENT,
                    target_id=fragment_id,
                    target_hash=target.target_hash,
                    action=ReviewAction.SUPERSEDE,
                    reason="A nonexistent prior decision cannot be reversed.",
                    authority="researcher:coverage",
                    scope=ReviewScope(collection_ids=(collection_id,), use="research"),
                    supersedes_review_decision_id="review_absent",
                )
            )
        assert review_repository.list() == ()


@pytest.mark.parametrize(
    "corruption",
    [
        "scope_shape",
        "scope_collection_type",
        "scope_noncanonical",
        "reason_noncanonical",
        "target_hash",
        "scope_outside_target",
    ],
)
def test_persisted_review_decision_corruption_is_rejected(
    tmp_path: Path,
    corruption: str,
) -> None:
    root = tmp_path / "corpus"
    other_root = tmp_path / "other"
    root.mkdir()
    other_root.mkdir()
    (root / "evidence.md").write_text("# Evidence\n\nPinned decision.\n", encoding="utf-8")
    (other_root / "other.md").write_text("# Other\n\nSeparate scope.\n", encoding="utf-8")

    with initialize_library(
        LibraryConfig(name=f"Review row corruption {corruption}"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, _version_id, fragment_id = _source(repository, root)
        unrelated_id = _collection(repository, other_root, "Unrelated corpus")
        decision_id, _target_hash = _source_review(
            repository,
            collection_id=collection_id,
            fragment_id=fragment_id,
        )
        connection = repository._store.connection
        _drop_update_trigger(repository, "review_decisions")

        if corruption == "scope_shape":
            scope: dict[str, object] = {
                "schema": ReviewScope.SCHEMA,
                "collection_ids": [collection_id],
                "use": "research",
                "unexpected": True,
            }
            connection.execute(
                "UPDATE review_decisions SET scope_json = ? WHERE review_decision_id = ?",
                (canonical_json_bytes(scope).decode("utf-8"), decision_id),
            )
        elif corruption == "scope_collection_type":
            scope = {
                "schema": ReviewScope.SCHEMA,
                "collection_ids": collection_id,
                "use": "research",
            }
            connection.execute(
                "UPDATE review_decisions SET scope_json = ? WHERE review_decision_id = ?",
                (canonical_json_bytes(scope).decode("utf-8"), decision_id),
            )
        elif corruption == "scope_noncanonical":
            scope = {
                "schema": "dithyramba.review_scope/9.9",
                "collection_ids": [collection_id],
                "use": "research",
            }
            connection.execute(
                "UPDATE review_decisions SET scope_json = ? WHERE review_decision_id = ?",
                (canonical_json_bytes(scope).decode("utf-8"), decision_id),
            )
        elif corruption == "reason_noncanonical":
            connection.execute(
                "UPDATE review_decisions SET reason = ? WHERE review_decision_id = ?",
                (" padded reason ", decision_id),
            )
        elif corruption == "target_hash":
            connection.execute(
                "UPDATE review_decisions SET target_hash = ? WHERE review_decision_id = ?",
                ("f" * 64, decision_id),
            )
        else:
            scope = ReviewScope(
                collection_ids=(unrelated_id,),
                use="research",
            ).semantic_payload()
            connection.execute(
                "UPDATE review_decisions SET scope_json = ? WHERE review_decision_id = ?",
                (canonical_json_bytes(scope).decode("utf-8"), decision_id),
            )

        with pytest.raises(ReviewIntegrityError):
            ReviewService(repository).get(decision_id)


@pytest.mark.parametrize(
    "corruption",
    [
        "address_hash",
        "content_identity",
        "missing_membership",
        "pdf_address_shape",
    ],
)
def test_persisted_source_fragment_corruption_is_rejected(
    tmp_path: Path,
    corruption: str,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "evidence.md").write_text("# Evidence\n\nPinned fragment.\n", encoding="utf-8")

    with initialize_library(
        LibraryConfig(name=f"Source target corruption {corruption}"),
        data_root=tmp_path / "data",
    ) as repository:
        _collection_id, version_id, fragment_id = _source(repository, root)
        connection = repository._store.connection

        if corruption == "missing_membership":
            source_row = connection.execute(
                "SELECT source_id FROM source_versions WHERE source_version_id = ?",
                (version_id,),
            ).fetchone()
            assert source_row is not None
            connection.execute(
                "DELETE FROM collection_memberships WHERE source_id = ?",
                (str(source_row[0]),),
            )
        else:
            _drop_update_trigger(repository, "source_fragments")
            if corruption == "address_hash":
                connection.execute(
                    "UPDATE source_fragments SET address_hash = ? WHERE source_fragment_id = ?",
                    ("f" * 64, fragment_id),
                )
            elif corruption == "content_identity":
                row = connection.execute(
                    "SELECT fragment_kind FROM source_fragments WHERE source_fragment_id = ?",
                    (fragment_id,),
                ).fetchone()
                assert row is not None
                changed_kind = "paragraph" if str(row[0]) == "heading" else "heading"
                connection.execute(
                    "UPDATE source_fragments SET fragment_kind = ? WHERE source_fragment_id = ?",
                    (changed_kind, fragment_id),
                )
            else:
                malformed_pdf: dict[str, object] = {
                    "schema": "dithyramba.source_address/1.0",
                    "kind": "pdf",
                    "page": 1,
                    "bbox": ["0.000", "0.000", "10.000", "10.000"],
                    "char_start": 0,
                    "char_end": 10,
                    "unexpected": True,
                }
                connection.execute(
                    """
                    UPDATE source_fragments SET source_address_json = ?
                    WHERE source_fragment_id = ?
                    """,
                    (canonical_json_bytes(malformed_pdf).decode("utf-8"), fragment_id),
                )

        with pytest.raises(ReviewIntegrityError):
            ReviewService(repository).target(ReviewTargetType.SOURCE_FRAGMENT, fragment_id)


@pytest.mark.parametrize(
    "corruption",
    [
        "json_identity",
        "json_hash_field",
        "content_identity",
        "query_relation",
        "snapshot_relation",
    ],
)
def test_persisted_evidence_packet_corruption_is_rejected(
    tmp_path: Path,
    corruption: str,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "evidence.md").write_text(
        "# Evidence\n\nPacket corruption evidence is exact.\n",
        encoding="utf-8",
    )

    with initialize_library(
        LibraryConfig(name=f"Packet target corruption {corruption}"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, _version_id, _fragment_id = _source(repository, root)
        packet_id = _prepare_packet(
            repository,
            collection_id=collection_id,
            question="packet corruption evidence",
        )
        payload = _packet_payload(repository, packet_id)
        connection = repository._store.connection
        _drop_update_trigger(repository, "evidence_packets")

        lookup_id = packet_id
        if corruption == "json_identity":
            payload["evidence_packet_id"] = "packet_corrupt"
            connection.execute(
                "UPDATE evidence_packets SET packet_json = ? WHERE evidence_packet_id = ?",
                (canonical_json_bytes(payload).decode("utf-8"), packet_id),
            )
        elif corruption == "json_hash_field":
            payload["packet_hash"] = "f" * 64
            connection.execute(
                "UPDATE evidence_packets SET packet_json = ? WHERE evidence_packet_id = ?",
                (canonical_json_bytes(payload).decode("utf-8"), packet_id),
            )
        elif corruption == "content_identity":
            connection.execute("PRAGMA foreign_keys = OFF")
            lookup_id = "packet_corrupt"
            payload["evidence_packet_id"] = lookup_id
            connection.execute(
                """
                UPDATE evidence_packets SET evidence_packet_id = ?, packet_json = ?
                WHERE evidence_packet_id = ?
                """,
                (lookup_id, canonical_json_bytes(payload).decode("utf-8"), packet_id),
            )
        elif corruption == "query_relation":
            connection.execute("PRAGMA foreign_keys = OFF")
            payload["query_request_id"] = "query_unrelated"
            lookup_id, packet_hash, packet_json = _reidentify_packet(payload)
            connection.execute(
                """
                UPDATE evidence_packets
                SET evidence_packet_id = ?, packet_hash = ?, packet_json = ?
                WHERE evidence_packet_id = ?
                """,
                (lookup_id, packet_hash, packet_json, packet_id),
            )
        else:
            connection.execute("PRAGMA foreign_keys = OFF")
            payload["corpus_snapshot_id"] = "snapshot_unrelated"
            lookup_id, packet_hash, packet_json = _reidentify_packet(payload)
            connection.execute(
                """
                UPDATE evidence_packets
                SET evidence_packet_id = ?, packet_hash = ?, packet_json = ?
                WHERE evidence_packet_id = ?
                """,
                (lookup_id, packet_hash, packet_json, packet_id),
            )

        with pytest.raises(ReviewIntegrityError):
            ReviewService(repository).target(ReviewTargetType.EVIDENCE_PACKET, lookup_id)


def test_packet_query_library_drift_is_rejected_after_exact_reconstruction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "evidence.md").write_text(
        "# Evidence\n\nLibrary-bound packet evidence.\n",
        encoding="utf-8",
    )

    with initialize_library(
        LibraryConfig(name="Packet query Library corruption"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, _version_id, _fragment_id = _source(repository, root)
        packet_id = _prepare_packet(
            repository,
            collection_id=collection_id,
            question="library-bound packet evidence",
        )
        connection = repository._store.connection
        packet_payload = _packet_payload(repository, packet_id)
        original_query_id = str(packet_payload["query_request_id"])
        query_row = connection.execute(
            "SELECT request_json FROM query_requests WHERE query_request_id = ?",
            (original_query_id,),
        ).fetchone()
        assert query_row is not None
        query_payload = cast(dict[str, object], json.loads(str(query_row[0])))
        query_payload["library_id"] = "library_alien"
        query_hash = canonical_sha256_hex(query_payload)
        query_id = canonical_content_id("query", query_payload)

        connection.execute("PRAGMA foreign_keys = OFF")
        _drop_update_trigger(repository, "query_requests")
        _drop_update_trigger(repository, "evidence_packets")
        connection.execute(
            """
            UPDATE query_requests
            SET query_request_id = ?, request_json = ?, request_hash = ?
            WHERE query_request_id = ?
            """,
            (
                query_id,
                canonical_json_bytes(query_payload).decode("utf-8"),
                query_hash,
                original_query_id,
            ),
        )
        packet_payload["query_request_id"] = query_id
        packet_payload["query_request_hash"] = query_hash
        changed_packet_id, packet_hash, packet_json = _reidentify_packet(packet_payload)
        connection.execute(
            """
            UPDATE evidence_packets
            SET evidence_packet_id = ?, query_request_id = ?, packet_hash = ?, packet_json = ?
            WHERE evidence_packet_id = ?
            """,
            (changed_packet_id, query_id, packet_hash, packet_json, packet_id),
        )

        with pytest.raises(ReviewIntegrityError, match="another Library"):
            ReviewService(repository).target(
                ReviewTargetType.EVIDENCE_PACKET,
                changed_packet_id,
            )


def test_packet_item_scope_corruption_and_multi_packet_scan_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    other_root = tmp_path / "other"
    root.mkdir()
    other_root.mkdir()
    (root / "evidence.md").write_text(
        "# Evidence\n\nAlpha evidence.\n\nBeta evidence.\n",
        encoding="utf-8",
    )
    (other_root / "other.md").write_text(
        "# Other\n\nUnrelated fragment.\n",
        encoding="utf-8",
    )

    with initialize_library(
        LibraryConfig(name="Packet item corruption"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, _version_id, _fragment_id = _source(repository, root)
        first_packet = _prepare_packet(
            repository,
            collection_id=collection_id,
            question="alpha evidence",
        )
        second_packet = _prepare_packet(
            repository,
            collection_id=collection_id,
            question="beta evidence",
        )
        service = ReviewService(repository)
        ordered_packets = sorted((first_packet, second_packet))
        second_targets = service.packet_item_targets(ordered_packets[1])
        assert second_targets
        selected = second_targets[-1]
        assert service.target(ReviewTargetType.PACKET_ITEM, selected.target_id) == selected

        other_collection = _collection(repository, other_root, "Other corpus")
        outcome = IngestService(repository).ingest_path(other_collection, "other.md").outcomes[0]
        assert outcome.source_version_id is not None
        unrelated_fragment = repository.list_source_fragments(outcome.source_version_id)[0]
        connection = repository._store.connection
        _drop_update_trigger(repository, "packet_items")
        connection.execute(
            """
            UPDATE packet_items SET source_fragment_id = ?
            WHERE evidence_packet_id = ? AND rank = 1
            """,
            (unrelated_fragment.source_fragment_id, ordered_packets[0]),
        )

        with pytest.raises(ReviewIntegrityError, match="outside packet Collections"):
            service.packet_item_targets(ordered_packets[0])
