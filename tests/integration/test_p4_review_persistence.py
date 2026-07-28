"""ReviewDecision persistence, scope, immutability, and reversal tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import LibraryRepository, initialize_library
from dithyramba.persistence.recall import SQLiteRecallBackend
from dithyramba.recall import (
    QueryRequest,
    RecallService,
    RetrievalBudget,
    current_fts_runtime_profile,
)
from dithyramba.review import (
    ReviewAction,
    ReviewConflictError,
    ReviewDecisionRequest,
    ReviewIntegrityError,
    ReviewScope,
    ReviewTargetNotFoundError,
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


def _source_target(
    repository: LibraryRepository,
    root: Path,
    *,
    filename: str = "paper.md",
) -> tuple[str, str, str]:
    collection_id = _collection(repository, root, "Review corpus")
    outcome = IngestService(repository).ingest_path(collection_id, filename).outcomes[0]
    assert outcome.source_version_id is not None
    fragment = repository.list_source_fragments(outcome.source_version_id)[0]
    target = ReviewService(repository).target(
        ReviewTargetType.SOURCE_FRAGMENT,
        fragment.source_fragment_id,
    )
    return collection_id, target.target_id, target.target_hash


def _request(
    *,
    collection_id: str,
    target_id: str,
    target_hash: str,
    action: ReviewAction = ReviewAction.ACCEPT,
    supersedes: str | None = None,
) -> ReviewDecisionRequest:
    return ReviewDecisionRequest(
        target_type=ReviewTargetType.SOURCE_FRAGMENT,
        target_id=target_id,
        target_hash=target_hash,
        action=action,
        reason="Reviewed against the exact stored fragment and its source context.",
        authority="researcher:eugene",
        scope=ReviewScope(collection_ids=(collection_id,), use="research"),
        supersedes_review_decision_id=supersedes,
    )


def test_decision_and_supersede_are_append_only_without_source_mutation(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    source_path = root / "paper.md"
    source_path.write_text("# Evidence\n\nA bounded interpretation.\n", encoding="utf-8")
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()

    with initialize_library(
        LibraryConfig(name="Review persistence"), data_root=tmp_path / "data"
    ) as repository:
        collection_id, target_id, target_hash = _source_target(repository, root)
        fragment_rows_before = repository._store.connection.execute(
            """
            SELECT source_fragment_id, source_version_id, text_sha256, address_hash
            FROM source_fragments ORDER BY source_fragment_id
            """
        ).fetchall()
        service = ReviewService(repository)
        accepted = service.decide(
            _request(
                collection_id=collection_id,
                target_id=target_id,
                target_hash=target_hash,
            )
        )
        superseded = service.decide(
            _request(
                collection_id=collection_id,
                target_id=target_id,
                target_hash=target_hash,
                action=ReviewAction.SUPERSEDE,
                supersedes=accepted.review_decision_id,
            )
        )

        decisions = service.list(
            target_type=ReviewTargetType.SOURCE_FRAGMENT,
            target_id=target_id,
        )
        assert decisions == (accepted, superseded)
        assert superseded.supersedes_review_decision_id == accepted.review_decision_id
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM review_decisions"
            ).fetchone()[0]
            == 2
        )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM event_outbox WHERE event_type = 'review_decision.created'"
            ).fetchone()[0]
            == 2
        )
        assert (
            repository._store.connection.execute(
                """
                SELECT source_fragment_id, source_version_id, text_sha256, address_hash
                FROM source_fragments ORDER BY source_fragment_id
                """
            ).fetchall()
            == fragment_rows_before
        )
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before


def test_review_scope_and_library_are_fail_closed(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "paper.md").write_text("# First\n\nTarget text.\n", encoding="utf-8")
    (second_root / "other.md").write_text("# Other\n\nOther text.\n", encoding="utf-8")
    data_root = tmp_path / "data"
    first = LibraryConfig(name="First review Library")
    second = LibraryConfig(name="Second review Library")

    with initialize_library(first, data_root=data_root) as repository:
        collection_id, target_id, target_hash = _source_target(repository, first_root)
        unrelated = _collection(repository, second_root, "Unrelated")
        service = ReviewService(repository)
        with pytest.raises(ReviewIntegrityError, match="outside"):
            service.decide(
                _request(
                    collection_id=unrelated,
                    target_id=target_id,
                    target_hash=target_hash,
                )
            )
        with pytest.raises(ReviewConflictError, match="stale"):
            service.decide(
                _request(
                    collection_id=collection_id,
                    target_id=target_id,
                    target_hash="0" * 64,
                )
            )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM review_decisions"
            ).fetchone()[0]
            == 0
        )

    with (
        initialize_library(second, data_root=data_root) as repository,
        pytest.raises(ReviewTargetNotFoundError, match="this Library"),
    ):
        ReviewService(repository).target(ReviewTargetType.SOURCE_FRAGMENT, target_id)


def test_prior_decision_can_only_be_superseded_once(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "paper.md").write_text("# Evidence\n\nReview lineage.\n", encoding="utf-8")
    with initialize_library(
        LibraryConfig(name="Review lineage"), data_root=tmp_path / "data"
    ) as repository:
        collection_id, target_id, target_hash = _source_target(repository, root)
        service = ReviewService(repository)
        prior = service.decide(
            _request(
                collection_id=collection_id,
                target_id=target_id,
                target_hash=target_hash,
            )
        )
        reversal = _request(
            collection_id=collection_id,
            target_id=target_id,
            target_hash=target_hash,
            action=ReviewAction.SUPERSEDE,
            supersedes=prior.review_decision_id,
        )
        service.decide(reversal)
        with pytest.raises(ReviewConflictError, match="already been superseded"):
            service.decide(reversal)
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM review_decisions"
            ).fetchone()[0]
            == 2
        )


def test_decision_and_event_roll_back_together_and_rows_reject_mutation(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "paper.md").write_text("# Evidence\n\nAtomic review.\n", encoding="utf-8")
    with initialize_library(
        LibraryConfig(name="Atomic review"), data_root=tmp_path / "data"
    ) as repository:
        collection_id, target_id, target_hash = _source_target(repository, root)
        existing_event_id = repository.pending_outbox_events()[0].event_id
        repository._event_id_factory = lambda: existing_event_id
        with pytest.raises(ReviewConflictError, match="transaction conflicted"):
            ReviewService(repository).decide(
                _request(
                    collection_id=collection_id,
                    target_id=target_id,
                    target_hash=target_hash,
                )
            )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM review_decisions"
            ).fetchone()[0]
            == 0
        )

        repository._event_id_factory = lambda: "event_review_unique"
        decision = ReviewService(repository).decide(
            _request(
                collection_id=collection_id,
                target_id=target_id,
                target_hash=target_hash,
            )
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repository._store.connection.execute(
                "UPDATE review_decisions SET reason = 'changed' WHERE review_decision_id = ?",
                (decision.review_decision_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repository._store.connection.execute(
                "DELETE FROM review_decisions WHERE review_decision_id = ?",
                (decision.review_decision_id,),
            )


def test_packet_and_packet_item_targets_are_hash_pinned_and_reviewable(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "paper.md").write_text(
        "# Evidence\n\nPacket item review preserves exact context.\n",
        encoding="utf-8",
    )
    library = LibraryConfig(name="Packet review")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection_id = _collection(repository, root, "Packet corpus")
        IngestService(repository).ingest_path(collection_id, "paper.md")
        snapshot = repository.freeze_snapshot((collection_id,))
        policy = AccessPolicySnapshot(
            access_policy_id="policy_packet_review",
            library_id=library.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
        )
        repository.persist_access_policy(name="Packet review", snapshot=policy)
        profile = current_fts_runtime_profile()
        recalled = RecallService(
            SQLiteRecallBackend(repository),
            profile_version=profile.profile_version,
        ).recall(
            QueryRequest(
                question="What preserves exact context?",
                library_id=library.library_id,
                collection_ids=(collection_id,),
                corpus_snapshot_id=snapshot.corpus_snapshot_id,
                access_policy_id=policy.access_policy_id,
                purpose="research",
                retrieval=RetrievalBudget(max_candidates=10, max_source_fragments=5),
            )
        )
        service = ReviewService(repository)
        packet_target = service.target(
            ReviewTargetType.EVIDENCE_PACKET,
            recalled.packet.evidence_packet_id,
        )
        assert packet_target.target_hash == recalled.packet.packet_hash
        assert packet_target.collection_ids == (collection_id,)
        packet_decision = service.decide(
            ReviewDecisionRequest(
                target_type=ReviewTargetType.EVIDENCE_PACKET,
                target_id=packet_target.target_id,
                target_hash=packet_target.target_hash,
                action=ReviewAction.ACCEPT,
                reason="The packet is fit for this bounded research use.",
                authority="researcher:eugene",
                scope=ReviewScope(collection_ids=(collection_id,), use="research"),
            )
        )

        item_targets = service.packet_item_targets(recalled.packet.evidence_packet_id)
        assert len(item_targets) == len(recalled.packet.source_fragments)
        assert all(item.target_type is ReviewTargetType.PACKET_ITEM for item in item_targets)
        selected = item_targets[0]
        assert service.target(ReviewTargetType.PACKET_ITEM, selected.target_id) == selected
        item_decision = service.decide(
            ReviewDecisionRequest(
                target_type=ReviewTargetType.PACKET_ITEM,
                target_id=selected.target_id,
                target_hash=selected.target_hash,
                action=ReviewAction.REVISE,
                reason="The evidence is exact, but the interpretation needs narrower wording.",
                authority="researcher:eugene",
                scope=ReviewScope(collection_ids=(collection_id,), use="research"),
            )
        )
        assert service.get(packet_decision.review_decision_id) == packet_decision
        assert service.get(item_decision.review_decision_id) == item_decision
        assert service.list(
            target_type=ReviewTargetType.PACKET_ITEM,
            target_id=selected.target_id,
        ) == (item_decision,)


def test_review_repository_rejects_missing_filters_targets_and_invalid_limits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "paper.md").write_text("# Evidence\n\nOne fragment.\n", encoding="utf-8")
    with initialize_library(
        LibraryConfig(name="Review input errors"), data_root=tmp_path / "data"
    ) as repository:
        _source_target(repository, root)
        service = ReviewService(repository)
        with pytest.raises(ReviewTargetNotFoundError):
            service.target(ReviewTargetType.PACKET_ITEM, "packet_item_absent")
        with pytest.raises(ReviewTargetNotFoundError):
            service.target(ReviewTargetType.EVIDENCE_PACKET, "packet_absent")
        with pytest.raises(ReviewIntegrityError, match="supplied together"):
            service.list(target_type=ReviewTargetType.SOURCE_FRAGMENT)
        with pytest.raises(ReviewIntegrityError, match="between 1 and 500"):
            service.list(limit=0)
        with pytest.raises(ReviewIntegrityError, match="between 1 and 500"):
            service.list(limit=True)


def test_supersede_must_retain_exact_target_and_scope(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "paper.md").write_text("# Evidence\n\nScoped reversal.\n", encoding="utf-8")
    with initialize_library(
        LibraryConfig(name="Scoped reversal"), data_root=tmp_path / "data"
    ) as repository:
        collection_id, target_id, target_hash = _source_target(repository, root)
        service = ReviewService(repository)
        prior = service.decide(
            _request(
                collection_id=collection_id,
                target_id=target_id,
                target_hash=target_hash,
            )
        )
        with pytest.raises(ReviewIntegrityError, match="retain"):
            service.decide(
                ReviewDecisionRequest(
                    target_type=ReviewTargetType.SOURCE_FRAGMENT,
                    target_id=target_id,
                    target_hash=target_hash,
                    action=ReviewAction.SUPERSEDE,
                    reason="Attempted scope drift.",
                    authority="researcher:eugene",
                    scope=ReviewScope(
                        collection_ids=(collection_id,),
                        use="publication",
                    ),
                    supersedes_review_decision_id=prior.review_decision_id,
                )
            )


def test_corrupt_fragment_address_and_packet_json_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "paper.md").write_text("# Evidence\n\nCorruption canary.\n", encoding="utf-8")
    library = LibraryConfig(name="Review corruption")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection_id, fragment_id, _fragment_hash = _source_target(repository, root)
        connection = repository._store.connection
        connection.execute("DROP TRIGGER source_fragments_no_update")
        connection.execute(
            "UPDATE source_fragments SET source_address_json = ? WHERE source_fragment_id = ?",
            ("{}", fragment_id),
        )
        with pytest.raises(ReviewIntegrityError, match="address"):
            ReviewService(repository).target(ReviewTargetType.SOURCE_FRAGMENT, fragment_id)

    packet_root = tmp_path / "packet-corpus"
    packet_root.mkdir()
    (packet_root / "packet.md").write_text(
        "# Evidence\n\nPacket corruption canary.\n",
        encoding="utf-8",
    )
    packet_library = LibraryConfig(name="Packet corruption")
    with initialize_library(packet_library, data_root=tmp_path / "packet-data") as repository:
        collection_id = _collection(repository, packet_root, "Packet corpus")
        IngestService(repository).ingest_path(collection_id, "packet.md")
        snapshot = repository.freeze_snapshot((collection_id,))
        policy = AccessPolicySnapshot(
            access_policy_id="policy_packet_corruption",
            library_id=packet_library.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
        )
        repository.persist_access_policy(name="Packet corruption", snapshot=policy)
        profile = current_fts_runtime_profile()
        result = RecallService(
            SQLiteRecallBackend(repository), profile_version=profile.profile_version
        ).recall(
            QueryRequest(
                question="corruption canary",
                library_id=packet_library.library_id,
                collection_ids=(collection_id,),
                corpus_snapshot_id=snapshot.corpus_snapshot_id,
                access_policy_id=policy.access_policy_id,
                purpose="research",
            )
        )
        connection = repository._store.connection
        connection.execute("DROP TRIGGER evidence_packets_no_update")
        raw = json.loads(result.packet.canonical_bytes)
        raw["result_status"] = "no_evidence"
        connection.execute(
            "UPDATE evidence_packets SET packet_json = ? WHERE evidence_packet_id = ?",
            (
                json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                result.packet.evidence_packet_id,
            ),
        )
        with pytest.raises(ReviewIntegrityError, match="hash"):
            ReviewService(repository).target(
                ReviewTargetType.EVIDENCE_PACKET,
                result.packet.evidence_packet_id,
            )
