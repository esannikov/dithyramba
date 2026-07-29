from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

import dithyramba.persistence.meaning as meaning_persistence
from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect, RequestScope
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)
from dithyramba.library import LibraryConfig
from dithyramba.meaning import (
    ConceptMeaningProposal,
    ConceptProposal,
    EntityKind,
    EntityProposal,
    EvidenceLinkProposal,
    ExactMentionProposal,
    ExtractionProfile,
    GeneratorProfile,
    MeaningOutputManifest,
    MeaningProposal,
    MeaningReviewAction,
    MeaningReviewRequest,
    MeaningReviewScope,
    MeaningReviewTargetType,
    MeaningRunRequest,
    ReviewBudget,
    StatementKind,
    StatementProposal,
    TimeContextProposal,
    VoiceKind,
    VoiceProposal,
)
from dithyramba.persistence import LibraryRepository, SQLiteMeaningRepository, initialize_library
from dithyramba.reading_room import (
    ReadingRoomEdgeType,
    ReadingRoomLimits,
    ReadingRoomNodeType,
    ReadingRoomProjection,
    ReadingRoomService,
)

NOW = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-21T12:00:00.000000Z"
PUBLIC_TEXT = "Автор: пам'ять є практикою."
PRIVATE_TEXT = "PRIVATE-READING-ROOM-CANARY"


def _clock() -> datetime:
    return NOW


def _digest(value: str) -> str:
    return sha256_hex(value.encode("utf-8"))


@dataclass(slots=True)
class ReadingRoomFixture:
    repository: LibraryRepository
    meaning: SQLiteMeaningRepository
    request: MeaningRunRequest
    public_collection_id: str
    holdout_collection_id: str
    public_fragment_id: str
    holdout_fragment_id: str


@pytest.fixture
def reading_room_fixture(tmp_path: Path) -> Iterator[ReadingRoomFixture]:
    data_root = tmp_path / "data"
    public_root = tmp_path / "public"
    holdout_root = tmp_path / "holdout"
    public_root.mkdir()
    holdout_root.mkdir()
    library = LibraryConfig(name="Reading Room")
    with initialize_library(library, data_root=data_root, clock=_clock) as repository:
        public = repository.create_collection(
            CollectionConfig(
                library_id=library.library_id,
                name="Public",
                kind=CollectionKind.CORPUS,
                roots=(build_collection_root(public_root, data_root=data_root),),
            )
        )
        holdout = repository.create_collection(
            CollectionConfig(
                library_id=library.library_id,
                name="Private holdout",
                kind=CollectionKind.HOLDOUT,
                roots=(build_collection_root(holdout_root, data_root=data_root),),
            )
        )
        _insert_source(
            repository,
            suffix="public",
            collection_id=public.config.collection_id,
            text=PUBLIC_TEXT,
            membership_state="active",
        )
        _insert_source(
            repository,
            suffix="holdout",
            collection_id=holdout.config.collection_id,
            text=PRIVATE_TEXT,
            membership_state="holdout",
        )
        snapshot = repository.freeze_snapshot(
            (public.config.collection_id, holdout.config.collection_id)
        )
        policy = AccessPolicySnapshot(
            access_policy_id="policy_reading_room",
            library_id=library.library_id,
            allowed_purposes=("research",),
            collection_rules=(
                CollectionRule(public.config.collection_id, PolicyEffect.ALLOW),
                CollectionRule(holdout.config.collection_id, PolicyEffect.ALLOW),
            ),
        )
        repository.persist_access_policy(name="Reading Room policy", snapshot=policy)
        request = MeaningRunRequest(
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            scope=RequestScope(
                library_id=library.library_id,
                snapshot_hash=snapshot.manifest_hash,
                purpose="research",
                collection_ids=(public.config.collection_id, holdout.config.collection_id),
            ),
            profile=ExtractionProfile.DEEP,
            generator=GeneratorProfile(
                generator_id="fixture",
                revision="1",
                license="internal",
                prompt_version="reading-room-v1",
            ),
            review_budget=ReviewBudget(),
            code_version="test",
        )
        meaning = SQLiteMeaningRepository(repository)
        meaning.import_proposal(
            request=request,
            proposal=_proposal(public.config.collection_id),
        )
        yield ReadingRoomFixture(
            repository=repository,
            meaning=meaning,
            request=request,
            public_collection_id=public.config.collection_id,
            holdout_collection_id=holdout.config.collection_id,
            public_fragment_id="fragment_public",
            holdout_fragment_id="fragment_holdout",
        )


def test_projection_is_authorized_text_free_deterministic_and_current_schema(
    reading_room_fixture: ReadingRoomFixture,
) -> None:
    fixture = reading_room_fixture
    statements: list[str] = []
    connection = fixture.repository._store.connection
    connection.set_trace_callback(statements.append)
    try:
        first = _project(fixture)
        second = _project(fixture)
    finally:
        connection.set_trace_callback(None)

    assert first == second
    assert first.projection_hash == second.projection_hash
    assert first.library.storage_schema_version == 10
    assert first.payload()["schema"] == ReadingRoomProjection.SCHEMA
    assert first.corpus.source_count == 1
    assert first.corpus.source_version_count == 1
    assert first.corpus.source_fragment_count == 1
    assert first.corpus.collection_count == 1
    assert first.access.permitted_collection_ids == (fixture.public_collection_id,)
    assert len(first.corpus.fragments) == 1
    fragment = first.corpus.fragments[0]
    assert fragment.source_fragment_id == fixture.public_fragment_id
    assert fragment.source_version_id == "source_version_public"
    assert fragment.source_id == "source_public"
    assert fragment.source_family_id == "family_public"
    assert fragment.collection_ids == (fixture.public_collection_id,)
    assert fragment.text_sha256 == _digest(PUBLIC_TEXT)

    serialized = json.dumps(first.payload(), ensure_ascii=False, sort_keys=True)
    assert PUBLIC_TEXT not in serialized
    assert PRIVATE_TEXT not in serialized
    assert fixture.holdout_fragment_id not in serialized
    assert "source_holdout" not in serialized
    assert "source_version_holdout" not in serialized
    normalized_sql = "\n".join(statements).lower()
    assert "quote_text" not in normalized_sql
    assert not any("sf.text," in statement.lower() for statement in statements)


def test_typed_graph_has_only_persisted_relations_and_no_dangling_edges(
    reading_room_fixture: ReadingRoomFixture,
) -> None:
    projection = _project(reading_room_fixture)
    node_ids = {node.node_id for node in projection.graph.nodes}
    node_types = {node.node_type for node in projection.graph.nodes}
    edge_types = {edge.edge_type for edge in projection.graph.edges}

    assert {
        ReadingRoomNodeType.STATEMENT,
        ReadingRoomNodeType.EVIDENCE_LINK,
        ReadingRoomNodeType.SOURCE_FRAGMENT,
        ReadingRoomNodeType.VOICE,
        ReadingRoomNodeType.ENTITY,
        ReadingRoomNodeType.ENTITY_MENTION,
        ReadingRoomNodeType.CONCEPT,
        ReadingRoomNodeType.CONCEPT_MENTION,
        ReadingRoomNodeType.CONCEPT_MEANING,
        ReadingRoomNodeType.TIME_CONTEXT,
    }.issubset(node_types)
    assert {
        ReadingRoomEdgeType.STATEMENT_VOICE,
        ReadingRoomEdgeType.STATEMENT_EVIDENCE_LINK,
        ReadingRoomEdgeType.EVIDENCE_LINK_FRAGMENT,
        ReadingRoomEdgeType.ENTITY_MENTION_ENTITY,
        ReadingRoomEdgeType.ENTITY_MENTION_FRAGMENT,
        ReadingRoomEdgeType.CONCEPT_MENTION_CONCEPT,
        ReadingRoomEdgeType.CONCEPT_MENTION_FRAGMENT,
        ReadingRoomEdgeType.CONCEPT_MEANING_CONCEPT,
        ReadingRoomEdgeType.CONCEPT_MEANING_VOICE,
        ReadingRoomEdgeType.CONCEPT_MEANING_STATEMENT,
        ReadingRoomEdgeType.STATEMENT_TIME_CONTEXT,
        ReadingRoomEdgeType.CONCEPT_MEANING_TIME_CONTEXT,
    }.issubset(edge_types)
    assert all(
        edge.source_node_id in node_ids and edge.target_node_id in node_ids
        for edge in projection.graph.edges
    )
    assert not any("cooccurrence" in edge.edge_type.value for edge in projection.graph.edges)
    assert all(
        node.label is None
        for node in projection.graph.nodes
        if node.node_type
        in {
            ReadingRoomNodeType.EVIDENCE_LINK,
            ReadingRoomNodeType.SOURCE_FRAGMENT,
        }
    )


def test_all_denied_projection_is_empty_and_does_not_disclose_denied_counts_or_ids(
    reading_room_fixture: ReadingRoomFixture,
) -> None:
    fixture = reading_room_fixture
    denied_policy = AccessPolicySnapshot(
        access_policy_id="policy_reading_room_denied",
        library_id=fixture.repository.library_id,
        allowed_purposes=("research",),
        collection_rules=(
            CollectionRule(fixture.public_collection_id, PolicyEffect.DENY),
            CollectionRule(fixture.holdout_collection_id, PolicyEffect.DENY),
        ),
    )
    fixture.repository.persist_access_policy(name="Denied", snapshot=denied_policy)
    projection = ReadingRoomService(fixture.repository).project(
        corpus_snapshot_id=fixture.request.corpus_snapshot_id,
        access_policy_id=denied_policy.access_policy_id,
        scope=fixture.request.scope,
    )

    assert projection.corpus.collection_count == 0
    assert projection.corpus.source_count == 0
    assert projection.corpus.source_version_count == 0
    assert projection.corpus.source_fragment_count == 0
    assert projection.corpus.fragments == ()
    assert projection.graph.nodes == ()
    assert projection.graph.edges == ()
    assert projection.recent_runs == ()
    assert projection.review.visible_candidate_count == 0
    assert projection.access.policy_omission_present is True
    serialized = json.dumps(projection.payload(), ensure_ascii=False, sort_keys=True)
    for denied_value in (
        fixture.public_collection_id,
        fixture.holdout_collection_id,
        fixture.public_fragment_id,
        fixture.holdout_fragment_id,
        "source_public",
        "source_holdout",
    ):
        assert denied_value not in serialized


def test_mixed_evidence_closure_is_omitted_without_disclosing_denied_identity(
    reading_room_fixture: ReadingRoomFixture,
) -> None:
    fixture = reading_room_fixture
    connection = fixture.repository._store.connection
    run_id = str(
        connection.execute("SELECT processing_run_id FROM meaning_processing_runs").fetchone()[0]
    )
    voice_id = str(connection.execute("SELECT voice_id FROM voices").fetchone()[0])
    original_run = fixture.meaning.get_run(run_id).run
    import_key = str(
        connection.execute(
            "SELECT import_key FROM meaning_processing_runs WHERE processing_run_id = ?",
            (run_id,),
        ).fetchone()[0]
    )
    statement_id = "statement_mixed_probe"
    public_evidence_id = "evidence_mixed_public"
    private_evidence_id = "evidence_mixed_private"
    connection.execute(
        """
        INSERT INTO statements(
            statement_id, library_id, collection_id, kind, statement_text,
            voice_id, time_context_id, status, lifecycle, content_hash,
            first_processing_run_id, created_at
        ) VALUES (?, ?, ?, 'source_claim', 'mixed', ?, NULL, 'candidate',
                  'active', ?, ?, ?)
        """,
        (
            statement_id,
            fixture.repository.library_id,
            fixture.public_collection_id,
            voice_id,
            _digest(statement_id),
            run_id,
            NOW_TEXT,
        ),
    )
    with fixture.repository._permit_fragment_text_read():
        _insert_evidence(
            connection,
            evidence_id=public_evidence_id,
            statement_id=statement_id,
            fragment_id=fixture.public_fragment_id,
            voice_id=voice_id,
            quote="Автор",
            quote_start=0,
            run_id=run_id,
        )
        _insert_evidence(
            connection,
            evidence_id=private_evidence_id,
            statement_id=statement_id,
            fragment_id=fixture.holdout_fragment_id,
            voice_id=voice_id,
            quote=PRIVATE_TEXT,
            quote_start=0,
            run_id=run_id,
        )
    output = MeaningOutputManifest(
        voice_ids=(voice_id,),
        statement_ids=(statement_id,),
        evidence_link_ids=(public_evidence_id, private_evidence_id),
    )
    coverage_payload: dict[str, object] = {
        "schema": "dithyramba.meaning_coverage_report/2.0",
        "processing_run_id": run_id,
        "profile": "deep",
        "state": "complete",
        "read_fragment_count": 1,
        "candidate_count": output.candidate_count,
        "omissions": [],
    }
    connection.execute("DROP TRIGGER meaning_processing_runs_no_update")
    connection.execute("DROP TRIGGER meaning_coverage_reports_no_update")
    run_payload = meaning_persistence._run_receipt_payload(
        processing_run_id=run_id,
        kind=original_run.kind.value,
        corpus_snapshot_id=original_run.corpus_snapshot_id,
        access_policy_id=original_run.access_policy_id,
        scope_hash=original_run.scope_hash,
        permitted_set_hash=original_run.permitted_set_hash,
        extraction_profile=original_run.profile.value,
        code_version=original_run.code_version,
        generator_hash=original_run.generator_hash,
        review_budget_hash=original_run.review_budget_hash,
        input_hash=original_run.input_hash,
        proposal_hash=original_run.proposal_hash,
        import_key=import_key,
        status=original_run.status.value,
        candidate_count=output.candidate_count,
        unresolved_before=original_run.unresolved_before,
        backlog_warning=original_run.backlog_warning,
        error_code=original_run.error_code,
        output_hash=output.output_hash,
        started_at=original_run.started_at,
        finished_at=original_run.finished_at,
    )
    connection.execute(
        """
        UPDATE meaning_processing_runs
        SET candidate_count = ?, output_manifest_json = ?, output_hash = ?, receipt_hash = ?
        WHERE processing_run_id = ?
        """,
        (
            output.candidate_count,
            canonical_json_bytes(output.payload()).decode("utf-8"),
            output.output_hash,
            canonical_sha256_hex(run_payload),
            run_id,
        ),
    )
    connection.execute(
        """
        UPDATE meaning_coverage_reports
        SET coverage_report_id = ?, candidate_count = ?, report_hash = ?
        WHERE processing_run_id = ?
        """,
        (
            canonical_content_id("coverage", coverage_payload),
            output.candidate_count,
            canonical_sha256_hex(coverage_payload),
            run_id,
        ),
    )
    connection.execute(
        "CREATE TRIGGER meaning_processing_runs_no_update BEFORE UPDATE ON "
        "meaning_processing_runs BEGIN SELECT RAISE(ABORT, "
        "'meaning_processing_runs is append-only'); END"
    )
    connection.execute(
        "CREATE TRIGGER meaning_coverage_reports_no_update BEFORE UPDATE ON "
        "meaning_coverage_reports BEGIN SELECT RAISE(ABORT, "
        "'meaning_coverage_reports is append-only'); END"
    )

    projection = _project(fixture)

    assert projection.graph.nodes == ()
    assert projection.graph.edges == ()
    assert projection.omissions.closure_omission_present is True
    serialized = json.dumps(projection.payload(), ensure_ascii=False, sort_keys=True)
    assert fixture.holdout_fragment_id not in serialized
    assert private_evidence_id not in serialized
    assert PRIVATE_TEXT not in serialized


def test_graph_and_review_bounds_are_explicit_and_integrity_preserving(
    reading_room_fixture: ReadingRoomFixture,
) -> None:
    fixture = reading_room_fixture
    projection = ReadingRoomService(fixture.repository).project(
        corpus_snapshot_id=fixture.request.corpus_snapshot_id,
        access_policy_id=fixture.request.access_policy_id,
        scope=fixture.request.scope,
        limits=ReadingRoomLimits(
            node_limit=2,
            edge_limit=1,
            run_limit=1,
            review_limit=1,
            provenance_run_limit=1,
        ),
    )

    assert len(projection.graph.nodes) <= 2
    assert len(projection.graph.edges) <= 1
    assert projection.omissions.graph_nodes_truncated is True
    assert projection.omissions.graph_edges_truncated is True
    assert projection.omissions.review_queue_truncated is True
    assert len(projection.review.queue) == 1
    node_ids = {node.node_id for node in projection.graph.nodes}
    assert all(
        edge.source_node_id in node_ids and edge.target_node_id in node_ids
        for edge in projection.graph.edges
    )


def test_only_visible_review_candidates_and_scope_applicable_decisions_are_projected(
    reading_room_fixture: ReadingRoomFixture,
) -> None:
    fixture = reading_room_fixture
    queue = fixture.meaning.list_review_queue()
    statement = next(
        item for item in queue if item.target_type is MeaningReviewTargetType.STATEMENT
    )
    session = fixture.meaning.start_review_session()
    decision = fixture.meaning.create_review_decision(
        MeaningReviewRequest(
            review_session_id=session.review_session_id,
            target_type=statement.target_type,
            target_id=statement.target_id,
            target_hash=statement.target_hash,
            action=MeaningReviewAction.ACCEPT,
            reason="grounded",
            authority="reviewer",
            scope=MeaningReviewScope((fixture.public_collection_id,), "research"),
        )
    )

    projection = _project(fixture)
    reviewed = next(node for node in projection.graph.nodes if node.node_id == statement.target_id)
    assert reviewed.review_state == MeaningReviewAction.ACCEPT.value
    assert reviewed.latest_review_decision_id == decision.review_decision_id
    assert statement.target_id not in {item.target_id for item in projection.review.queue}
    assert sum(item.count for item in projection.review.status_counts) == (
        projection.review.visible_candidate_count
    )
    serialized = json.dumps(projection.payload(), ensure_ascii=False, sort_keys=True)
    assert "grounded" not in serialized
    assert "reviewer" not in serialized


def _project(fixture: ReadingRoomFixture) -> ReadingRoomProjection:
    return ReadingRoomService(fixture.repository, fixture.meaning).project(
        corpus_snapshot_id=fixture.request.corpus_snapshot_id,
        access_policy_id=fixture.request.access_policy_id,
        scope=fixture.request.scope,
    )


def _proposal(collection_id: str) -> MeaningProposal:
    author_label = "Автор"
    concept_label = "пам'ять"
    quote = "пам'ять є практикою"
    author_start = PUBLIC_TEXT.index(author_label)
    concept_start = PUBLIC_TEXT.index(concept_label)
    quote_start = PUBLIC_TEXT.index(quote)
    return MeaningProposal(
        voices=(VoiceProposal("author", VoiceKind.AUTHOR, author_label),),
        entities=(EntityProposal("author_entity", EntityKind.PERSON, author_label),),
        entity_mentions=(
            ExactMentionProposal(
                "author_entity",
                collection_id,
                "fragment_public",
                author_start,
                author_start + len(author_label),
                author_label,
                _digest(author_label),
            ),
        ),
        concepts=(ConceptProposal("memory", concept_label),),
        concept_mentions=(
            ExactMentionProposal(
                "memory",
                collection_id,
                "fragment_public",
                concept_start,
                concept_start + len(concept_label),
                concept_label,
                _digest(concept_label),
            ),
        ),
        time_contexts=(TimeContextProposal("time", recorded_at=NOW_TEXT),),
        statements=(
            StatementProposal(
                "statement",
                StatementKind.SOURCE_CLAIM,
                "Пам'ять є практикою",
                "author",
                collection_id,
                (
                    EvidenceLinkProposal(
                        "fragment_public",
                        quote_start,
                        quote_start + len(quote),
                        quote,
                        _digest(quote),
                        "author",
                    ),
                ),
                "time",
            ),
        ),
        concept_meanings=(
            ConceptMeaningProposal(
                "meaning",
                "memory",
                "author",
                "практика",
                ("statement",),
                "time",
            ),
        ),
    )


def _insert_source(
    repository: LibraryRepository,
    *,
    suffix: str,
    collection_id: str,
    text: str,
    membership_state: str,
) -> None:
    connection = repository._store.connection
    content_hash = _digest(text)
    address = {"kind": "markdown", "heading_path": [], "paragraph_index": 0}
    connection.execute(
        "INSERT INTO blobs VALUES (?, ?, ?, ?)",
        (content_hash, len(text.encode("utf-8")), f"{suffix}/blob", NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO sources VALUES (?, ?, ?, 'text/markdown', ?, ?)",
        (
            f"source_{suffix}",
            repository.library_id,
            f"file:///{suffix}.md",
            suffix,
            NOW_TEXT,
        ),
    )
    connection.execute(
        """
        INSERT INTO source_versions VALUES (?, ?, 1, ?, ?, ?, NULL, 'markdown_v1',
                                            'processed', NULL)
        """,
        (
            f"source_version_{suffix}",
            f"source_{suffix}",
            content_hash,
            len(text.encode("utf-8")),
            NOW_TEXT,
        ),
    )
    connection.execute(
        "INSERT INTO source_heads VALUES (?, ?, ?, NULL)",
        (f"source_{suffix}", f"source_version_{suffix}", NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_families VALUES (?, ?, ?, ?)",
        (f"family_{suffix}", repository.library_id, suffix, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_family_members VALUES (?, ?, 'root', NULL, ?)",
        (f"family_{suffix}", f"source_{suffix}", NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_fragments VALUES (?, ?, 0, 'paragraph', ?, ?, ?, ?)",
        (
            f"fragment_{suffix}",
            f"source_version_{suffix}",
            text,
            content_hash,
            canonical_json_bytes(address).decode("utf-8"),
            canonical_sha256_hex(address),
        ),
    )
    connection.execute(
        "INSERT INTO collection_memberships VALUES (?, ?, ?, ?)",
        (collection_id, f"source_{suffix}", membership_state, NOW_TEXT),
    )


def _insert_evidence(
    connection: sqlite3.Connection,
    *,
    evidence_id: str,
    statement_id: str,
    fragment_id: str,
    voice_id: str,
    quote: str,
    quote_start: int,
    run_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO evidence_links(
            evidence_link_id, statement_id, source_fragment_id,
            attributed_voice_id, polarity, alignment, limits_text,
            quote_start, quote_end, quote_text, quote_sha256,
            extraction_method, status, content_hash,
            first_processing_run_id, created_at
        ) VALUES (?, ?, ?, ?, 'supports', 'exact', NULL, ?, ?, ?, ?,
                  'test', 'candidate', ?, ?, ?)
        """,
        (
            evidence_id,
            statement_id,
            fragment_id,
            voice_id,
            quote_start,
            quote_start + len(quote),
            quote,
            _digest(quote),
            _digest(evidence_id),
            run_id,
            NOW_TEXT,
        ),
    )
