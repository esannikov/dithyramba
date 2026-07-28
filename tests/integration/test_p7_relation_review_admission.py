"""P7 semantic-review admission tests for candidate Relations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect, RequestScope
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.meaning import (
    MeaningBudgetError,
    MeaningNotFoundError,
    MeaningPersistenceError,
    MeaningReviewAction,
    MeaningReviewError,
    MeaningReviewRequest,
    MeaningReviewScope,
    MeaningReviewTargetType,
    RelationAdmission,
    ReviewBudget,
)
from dithyramba.persistence import SQLiteMeaningRepository
from dithyramba.relations import (
    RelationNodeRef,
    RelationNodeType,
    RelationProposal,
    RelationType,
)
from tests.integration.test_p7_relations import (
    NOW_TEXT,
    _context,
    _import_request,
    _proposals,
    _seed_private_entity,
)


def _review_request(
    *,
    session_id: str,
    relation_id: str,
    relation_hash: str,
    collection_ids: tuple[str, ...],
    action: MeaningReviewAction = MeaningReviewAction.ACCEPT,
    use: str = "research",
    replacement_target_id: str | None = None,
    supersedes_review_decision_id: str | None = None,
) -> MeaningReviewRequest:
    return MeaningReviewRequest(
        review_session_id=session_id,
        target_type=MeaningReviewTargetType.RELATION,
        target_id=relation_id,
        target_hash=relation_hash,
        action=action,
        reason="Exact evidence and comparison scope reviewed.",
        authority="human-reviewer",
        scope=MeaningReviewScope(collection_ids, use),
        replacement_target_id=replacement_target_id,
        supersedes_review_decision_id=supersedes_review_decision_id,
    )


def test_relation_enters_semantic_queue_and_only_exact_accept_admits(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        relation = relations.import_relations(
            _import_request(snapshot_id, policy, scope),
            (_proposals(statements, evidence)[0],),
        ).relations[0]
        meaning = SQLiteMeaningRepository(repository)
        queued = {
            (item.target_type, item.target_id): item
            for item in meaning.list_review_queue(limit=500)
        }
        queue_item = queued[(MeaningReviewTargetType.RELATION, relation.relation_id)]
        assert queue_item.target_hash == relation.content_hash
        assert queue_item.collection_ids == scope.collection_ids
        assert (
            meaning.resolve_relation_admission(
                relation.relation_id,
                use=scope.purpose,
                request_collection_ids=scope.collection_ids,
            )
            is None
        )

        session = meaning.start_review_session(ReviewBudget(review_decision_limit=3))
        decision = meaning.create_review_decision(
            _review_request(
                session_id=session.review_session_id,
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                collection_ids=scope.collection_ids,
            )
        )
        assert decision.target_type is MeaningReviewTargetType.RELATION
        assert decision.review_decision_id.startswith("review_relation_")
        assert meaning.get_review_decision(decision.review_decision_id) == decision

        admission = meaning.resolve_relation_admission(
            relation.relation_id,
            use=scope.purpose,
            request_collection_ids=scope.collection_ids,
        )
        assert isinstance(admission, RelationAdmission)
        assert admission.review_decision_id == decision.review_decision_id
        assert admission.target_hash == relation.content_hash
        assert admission.admission_hash == canonical_sha256_hex(admission.payload())
        assert relation.relation_id not in {
            item.target_id for item in meaning.list_review_queue(limit=500)
        }

        assert (
            meaning.resolve_relation_admission(
                relation.relation_id,
                use="publication",
                request_collection_ids=scope.collection_ids,
            )
            is None
        )
        assert (
            meaning.resolve_relation_admission(
                relation.relation_id,
                use=scope.purpose,
                request_collection_ids=("collection_other",),
            )
            is None
        )
    finally:
        repository.close()


def test_relation_review_scope_includes_complete_endpoint_grounding(
    tmp_path: Path,
) -> None:
    repository, relations, _snapshot_id, _policy, scope_a, statements, evidence = _context(tmp_path)
    try:
        collection_a = scope_a.collection_ids[0]
        collection_b, private_entity_id = _seed_private_entity(repository, tmp_path)
        snapshot = repository.freeze_snapshot((collection_a, collection_b))
        policy = AccessPolicySnapshot(
            access_policy_id="policy_relations_ab",
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(
                CollectionRule(collection_a, PolicyEffect.ALLOW),
                CollectionRule(collection_b, PolicyEffect.ALLOW),
            ),
        )
        repository.persist_access_policy(name="Research AB", snapshot=policy)
        scope_ab = RequestScope(
            library_id=repository.library_id,
            snapshot_hash=snapshot.manifest_hash,
            purpose="research",
            collection_ids=(collection_a, collection_b),
        )
        proposal = RelationProposal(
            RelationType.SUPPORTS,
            RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),
            RelationNodeRef(RelationNodeType.ENTITY, private_entity_id),
            (evidence[0],),
            "The relation joins evidence in A to an entity grounded in B.",
            "human_import/1.0",
        )
        relation = relations.import_relations(
            _import_request(snapshot.corpus_snapshot_id, policy, scope_ab),
            (proposal,),
        ).relations[0]
        meaning = SQLiteMeaningRepository(repository)
        queue_item = next(
            item
            for item in meaning.list_review_queue(limit=500)
            if item.target_type is MeaningReviewTargetType.RELATION
            and item.target_id == relation.relation_id
        )
        assert queue_item.collection_ids == tuple(sorted((collection_a, collection_b)))

        session = meaning.start_review_session()
        with pytest.raises(MeaningReviewError, match="exact Collection closure"):
            meaning.create_review_decision(
                _review_request(
                    session_id=session.review_session_id,
                    relation_id=relation.relation_id,
                    relation_hash=relation.content_hash,
                    collection_ids=(collection_a,),
                )
            )
        decision = meaning.create_review_decision(
            _review_request(
                session_id=session.review_session_id,
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                collection_ids=tuple(sorted((collection_a, collection_b))),
            )
        )
        assert decision.scope.collection_ids == tuple(sorted((collection_a, collection_b)))
        assert (
            meaning.resolve_relation_admission(
                relation.relation_id,
                use="research",
                request_collection_ids=(collection_a,),
            )
            is None
        )
        assert (
            meaning.resolve_relation_admission(
                relation.relation_id,
                use="research",
                request_collection_ids=tuple(sorted((collection_a, collection_b))),
            )
            is not None
        )
    finally:
        repository.close()


@pytest.mark.parametrize(
    "action",
    (MeaningReviewAction.REJECT, MeaningReviewAction.DEFER),
)
def test_non_accept_relation_dispositions_never_admit(
    tmp_path: Path,
    action: MeaningReviewAction,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        relation = relations.import_relations(
            _import_request(snapshot_id, policy, scope),
            (_proposals(statements, evidence)[0],),
        ).relations[0]
        meaning = SQLiteMeaningRepository(repository)
        session = meaning.start_review_session()
        meaning.create_review_decision(
            _review_request(
                session_id=session.review_session_id,
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                collection_ids=scope.collection_ids,
                action=action,
            )
        )

        assert (
            meaning.resolve_relation_admission(
                relation.relation_id,
                use=scope.purpose,
                request_collection_ids=scope.collection_ids,
            )
            is None
        )
        queued_ids = {item.target_id for item in meaning.list_review_queue(limit=500)}
        if action is MeaningReviewAction.REJECT:
            assert relation.relation_id not in queued_ids
        else:
            assert relation.relation_id in queued_ids
    finally:
        repository.close()


def test_relation_revise_does_not_accept_target_or_replacement(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        imported = relations.import_relations(
            _import_request(snapshot_id, policy, scope),
            _proposals(statements, evidence),
        )
        target, replacement = imported.relations
        meaning = SQLiteMeaningRepository(repository)
        session = meaning.start_review_session()
        meaning.create_review_decision(
            _review_request(
                session_id=session.review_session_id,
                relation_id=target.relation_id,
                relation_hash=target.content_hash,
                collection_ids=scope.collection_ids,
                action=MeaningReviewAction.REVISE,
                replacement_target_id=replacement.relation_id,
            )
        )

        for relation in (target, replacement):
            assert (
                meaning.resolve_relation_admission(
                    relation.relation_id,
                    use=scope.purpose,
                    request_collection_ids=scope.collection_ids,
                )
                is None
            )
    finally:
        repository.close()


def test_supersede_removes_accept_and_new_accept_can_restore_admission(
    tmp_path: Path,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        relation = relations.import_relations(
            _import_request(snapshot_id, policy, scope),
            (_proposals(statements, evidence)[0],),
        ).relations[0]
        meaning = SQLiteMeaningRepository(repository)
        session = meaning.start_review_session(ReviewBudget(review_decision_limit=4))
        accepted = meaning.create_review_decision(
            _review_request(
                session_id=session.review_session_id,
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                collection_ids=scope.collection_ids,
            )
        )
        meaning.create_review_decision(
            _review_request(
                session_id=session.review_session_id,
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                collection_ids=scope.collection_ids,
                action=MeaningReviewAction.SUPERSEDE,
                supersedes_review_decision_id=accepted.review_decision_id,
            )
        )
        assert (
            meaning.resolve_relation_admission(
                relation.relation_id,
                use=scope.purpose,
                request_collection_ids=scope.collection_ids,
            )
            is None
        )
        assert relation.relation_id in {
            item.target_id for item in meaning.list_review_queue(limit=500)
        }

        accepted_again = meaning.create_review_decision(
            _review_request(
                session_id=session.review_session_id,
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                collection_ids=scope.collection_ids,
            )
        )
        admission = meaning.resolve_relation_admission(
            relation.relation_id,
            use=scope.purpose,
            request_collection_ids=scope.collection_ids,
        )
        assert admission is not None
        assert admission.review_decision_id == accepted_again.review_decision_id
    finally:
        repository.close()


def test_relation_review_requires_fresh_hash_and_exact_collection_closure(
    tmp_path: Path,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        relation = relations.import_relations(
            _import_request(snapshot_id, policy, scope),
            (_proposals(statements, evidence)[0],),
        ).relations[0]
        meaning = SQLiteMeaningRepository(repository)
        session = meaning.start_review_session()
        with pytest.raises(MeaningReviewError, match="hash is stale"):
            meaning.create_review_decision(
                _review_request(
                    session_id=session.review_session_id,
                    relation_id=relation.relation_id,
                    relation_hash="f" * 64,
                    collection_ids=scope.collection_ids,
                )
            )
        with pytest.raises(MeaningReviewError, match="exact Collection closure"):
            meaning.create_review_decision(
                _review_request(
                    session_id=session.review_session_id,
                    relation_id=relation.relation_id,
                    relation_hash=relation.content_hash,
                    collection_ids=("collection_other",),
                )
            )
        with pytest.raises(MeaningReviewError, match="exact Collection closure"):
            meaning.create_review_decision(
                _review_request(
                    session_id=session.review_session_id,
                    relation_id=relation.relation_id,
                    relation_hash=relation.content_hash,
                    collection_ids=(*scope.collection_ids, "collection_other"),
                )
            )
        with pytest.raises(MeaningNotFoundError, match="does not exist"):
            meaning.resolve_relation_admission(
                "relation_missing",
                use=scope.purpose,
                request_collection_ids=scope.collection_ids,
            )
    finally:
        repository.close()


def test_relation_and_meaning_decisions_share_one_session_budget(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        relation = relations.import_relations(
            _import_request(snapshot_id, policy, scope),
            (_proposals(statements, evidence)[0],),
        ).relations[0]
        meaning = SQLiteMeaningRepository(repository)
        session = meaning.start_review_session(ReviewBudget(review_decision_limit=1))
        statement_hash = str(
            repository._store.connection.execute(
                "SELECT content_hash FROM statements WHERE statement_id = ?",
                (statements[0],),
            ).fetchone()[0]
        )
        meaning.create_review_decision(
            MeaningReviewRequest(
                review_session_id=session.review_session_id,
                target_type=MeaningReviewTargetType.STATEMENT,
                target_id=statements[0],
                target_hash=statement_hash,
                action=MeaningReviewAction.ACCEPT,
                reason="Statement reviewed.",
                authority="human-reviewer",
                scope=MeaningReviewScope(scope.collection_ids, scope.purpose),
            )
        )
        with pytest.raises(MeaningBudgetError, match="budget is exhausted"):
            meaning.create_review_decision(
                _review_request(
                    session_id=session.review_session_id,
                    relation_id=relation.relation_id,
                    relation_hash=relation.content_hash,
                    collection_ids=scope.collection_ids,
                )
            )
    finally:
        repository.close()


def test_relation_admission_fails_closed_on_multiple_active_decisions(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        relation = relations.import_relations(
            _import_request(snapshot_id, policy, scope),
            (_proposals(statements, evidence)[0],),
        ).relations[0]
        meaning = SQLiteMeaningRepository(repository)
        session = meaning.start_review_session(ReviewBudget(review_decision_limit=4))
        first = meaning.create_review_decision(
            _review_request(
                session_id=session.review_session_id,
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                collection_ids=scope.collection_ids,
            )
        )
        connection = repository._store.connection
        connection.execute("DROP TRIGGER relation_review_decisions_validate_active_insert")
        connection.execute(
            """
            INSERT INTO relation_review_decisions(
                review_decision_id, review_session_id, library_id, target_type,
                target_id, target_hash, action, reason, authority, scope_json,
                replacement_target_id, supersedes_review_decision_id, created_at
            ) VALUES (
                'review_relation_corrupt_second', ?, ?, 'relation', ?, ?, 'accept',
                'Injected corruption', 'test', ?, NULL, NULL, ?
            )
            """,
            (
                session.review_session_id,
                repository.library_id,
                relation.relation_id,
                relation.content_hash,
                canonical_json_bytes(first.scope.payload()).decode("utf-8"),
                NOW_TEXT,
            ),
        )
        with pytest.raises(MeaningPersistenceError, match="multiple active"):
            meaning.resolve_relation_admission(
                relation.relation_id,
                use=scope.purpose,
                request_collection_ids=scope.collection_ids,
            )
    finally:
        repository.close()


def test_relation_admission_fails_closed_on_noncanonical_equivalent_scope(
    tmp_path: Path,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        relation = relations.import_relations(
            _import_request(snapshot_id, policy, scope),
            (_proposals(statements, evidence)[0],),
        ).relations[0]
        meaning = SQLiteMeaningRepository(repository)
        session = meaning.start_review_session(ReviewBudget(review_decision_limit=4))
        accepted = meaning.create_review_decision(
            _review_request(
                session_id=session.review_session_id,
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                collection_ids=scope.collection_ids,
            )
        )
        noncanonical_scope_json = json.dumps(
            {
                "use": accepted.scope.use,
                "collection_ids": list(accepted.scope.collection_ids),
                "schema": "dithyramba.meaning_review_scope/1.0",
            },
            ensure_ascii=False,
        )
        assert noncanonical_scope_json != canonical_json_bytes(accepted.scope.payload()).decode(
            "utf-8"
        )

        repository._store.connection.execute(
            """
            INSERT INTO relation_review_decisions(
                review_decision_id, review_session_id, library_id, target_type,
                target_id, target_hash, action, reason, authority, scope_json,
                replacement_target_id, supersedes_review_decision_id, created_at
            ) VALUES (
                'review_relation_noncanonical_reject', ?, ?, 'relation', ?, ?, 'reject',
                'Injected equivalent scope', 'test', ?, NULL, NULL, ?
            )
            """,
            (
                session.review_session_id,
                repository.library_id,
                relation.relation_id,
                relation.content_hash,
                noncanonical_scope_json,
                NOW_TEXT,
            ),
        )

        with pytest.raises(MeaningPersistenceError, match="not canonical JSON"):
            meaning.resolve_relation_admission(
                relation.relation_id,
                use=scope.purpose,
                request_collection_ids=scope.collection_ids,
            )
    finally:
        repository.close()


def test_relation_review_creation_fails_closed_on_noncanonical_active_scope(
    tmp_path: Path,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        relation = relations.import_relations(
            _import_request(snapshot_id, policy, scope),
            (_proposals(statements, evidence)[0],),
        ).relations[0]
        meaning = SQLiteMeaningRepository(repository)
        session = meaning.start_review_session(ReviewBudget(review_decision_limit=4))
        requested_scope = MeaningReviewScope(scope.collection_ids, scope.purpose)
        noncanonical_scope_json = json.dumps(
            {
                "use": requested_scope.use,
                "collection_ids": list(requested_scope.collection_ids),
                "schema": "dithyramba.meaning_review_scope/1.0",
            },
            ensure_ascii=False,
        )
        repository._store.connection.execute(
            """
            INSERT INTO relation_review_decisions(
                review_decision_id, review_session_id, library_id, target_type,
                target_id, target_hash, action, reason, authority, scope_json,
                replacement_target_id, supersedes_review_decision_id, created_at
            ) VALUES (
                'review_relation_noncanonical_first', ?, ?, 'relation', ?, ?, 'reject',
                'Injected equivalent scope', 'test', ?, NULL, NULL, ?
            )
            """,
            (
                session.review_session_id,
                repository.library_id,
                relation.relation_id,
                relation.content_hash,
                noncanonical_scope_json,
                NOW_TEXT,
            ),
        )

        with pytest.raises(MeaningPersistenceError, match="not canonical JSON"):
            meaning.create_review_decision(
                _review_request(
                    session_id=session.review_session_id,
                    relation_id=relation.relation_id,
                    relation_hash=relation.content_hash,
                    collection_ids=scope.collection_ids,
                )
            )
    finally:
        repository.close()
