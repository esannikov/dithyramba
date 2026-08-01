"""End-to-end least-context agent facade tests over a cold Library reopen."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    QueryExclusions,
)
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.ingest.service import IngestService
from dithyramba.interactive import (
    AgentEvidencePacket,
    AgentResearchError,
    AgentResearchFacade,
    AgentResearchTurn,
    SessionContextBudget,
)
from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    LibraryRepository,
    ResearchSessionNotFoundError,
    ResearchSessionPersistenceError,
    SQLiteResearchSessionRepository,
    initialize_library,
    open_library,
)
from dithyramba.persistence.models import AuthorizedRead, SourceFragmentText
from dithyramba.persistence.sessions import (
    RecallCommandCompletion,
    RecallCommandStart,
    _artifact_error,
    _canonical_json_text,
    _load_event,
    _load_session,
    _payload_optional_text,
    _payload_text,
    _require_completion_matches_start,
    _require_same_command,
    _validated_command_id,
    _validated_event,
    _validated_session,
)
from dithyramba.recall import (
    EvidencePacket,
    QueryRequest,
    RetrievalBudget,
    current_fts_runtime_profile,
)
from dithyramba.sessions import (
    ResearchSession,
    ResearchSessionBrief,
    SessionEvent,
    SessionStatus,
)


class _TickingClock:
    def __init__(self) -> None:
        self._next = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self._next
        self._next += timedelta(microseconds=1)
        return current


def _prepare_library(repository: object, root: Path) -> tuple[str, str, str, str]:
    from dithyramba.persistence import LibraryRepository

    assert isinstance(repository, LibraryRepository)
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="Interactive corpus",
            kind=CollectionKind.CORPUS,
            roots=(
                build_collection_root(
                    root,
                    data_root=repository.paths.application_data_root,
                ),
            ),
        )
    )
    collection_id = collection.config.collection_id
    ingested = IngestService(repository).ingest_path(collection_id, "craft.md")
    source_id = ingested.outcomes[0].source_id
    assert source_id is not None
    snapshot = repository.freeze_snapshot((collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id="policy_interactive_research",
        library_id=repository.library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
    )
    repository.persist_access_policy(name="Interactive research", snapshot=policy)
    return collection_id, snapshot.corpus_snapshot_id, policy.access_policy_id, source_id


def test_agent_turn_reopens_with_same_compact_context_and_exact_packet(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "craft.md").write_text(
        "# Dialogue evidence\n\nBlocking changes the power relation inside a dialogue scene.\n",
        encoding="utf-8",
    )
    data_root = tmp_path / "data"
    budget = SessionContextBudget(
        recent_questions=1,
        recent_drafts=1,
        recent_gaps=1,
        recent_rejected_paths=1,
        recent_decisions=1,
        evidence_references=10,
        candidate_references=10,
    )
    profile = current_fts_runtime_profile().profile_version
    with initialize_library(
        LibraryConfig(name="Interactive research"), data_root=data_root
    ) as repository:
        library_id = repository.library_id
        collection_id, snapshot_id, policy_id, _source_id = _prepare_library(repository, root)
        facade = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=profile,
            clock=_TickingClock(),
            context_budget=budget,
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="How should dialogue staging reveal power?",
                intended_use="directorial decision",
                success_criteria=("find exact craft support", "retain open gaps"),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
        )
        assert opened.status is SessionStatus.OPEN
        turn = facade.recall(
            opened.session_id,
            "dialogue blocking power relation",
            command_id="command_dialogue_power_001",
            retrieval=RetrievalBudget(max_candidates=10, max_source_fragments=3),
        )
        assert turn.evidence_packet.source_fragments
        assert not hasattr(turn.evidence_packet, "read_receipt")
        assert turn.evidence_packet.read_receipt_hash
        tampered_projection = turn.evidence_packet.model_dump(mode="json")
        processed_count = tampered_projection["processed_count"]
        assert isinstance(processed_count, int)
        tampered_projection["processed_count"] = processed_count + 1
        with pytest.raises(ValidationError, match="projection_hash"):
            AgentEvidencePacket.model_validate_json(json.dumps(tampered_projection))

        full_packet = facade._recall_backend.load_evidence_packet(
            turn.evidence_packet.evidence_packet_id
        )
        legacy_turn = AgentResearchTurn.create(
            command_id=turn.command_id,
            question_event_id=turn.question_event_id,
            evidence_event_id=turn.evidence_event_id,
            evidence_packet=AgentEvidencePacket.create(full_packet),
            context=turn.context,
            schema_id=AgentResearchTurn.LEGACY_SCHEMA,
        )
        legacy_replay = facade._turn_from_completion(
            RecallCommandCompletion(
                command_id=turn.command_id,
                command_hash="a" * 64,
                question_event_id=turn.question_event_id,
                evidence_event_id=turn.evidence_event_id,
                evidence_packet_id=full_packet.evidence_packet_id,
                evidence_packet_hash=full_packet.packet_hash,
                agent_evidence_json=None,
                context_json=turn.context.model_dump_json(),
                turn_hash=legacy_turn.turn_hash,
            )
        )
        assert legacy_replay == legacy_turn
        assert turn.context.questions[-1].text == "dialogue blocking power relation"
        assert {item.artifact_kind.value for item in turn.context.evidence_references} == {
            "evidence_packet",
            "source_fragment",
        }
        facade.record_draft(
            opened.session_id,
            "Stage the change in status through movement; this remains a draft.",
        )
        facade.record_gap(opened.session_id, "Need a counterexample with static blocking.")
        facade.record_gap(opened.session_id, "Need evidence about eyelines.")
        facade.record_gap(opened.session_id, "Need a scene-specific production constraint.")
        final = facade.reject_path(
            opened.session_id,
            "Do not treat generic dialogue advice as evidence for this scene.",
        )
        assert final.omissions.gaps == 2
        assert final.gaps[0].text == "Need a scene-specific production constraint."
        assert final.drafts[0].text.endswith("this remains a draft.")
        assert final.evidence_references == turn.context.evidence_references
        final_hash = final.context_hash
        assert not hasattr(facade, "accept_candidate")
        assert not hasattr(facade, "record_decision")
        assert not hasattr(facade, "close")

    with open_library(library_id, data_root=data_root) as repository:
        reopened = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=profile,
            clock=_TickingClock(),
            context_budget=budget,
        ).context(opened.session_id)
        assert reopened.context_hash == final_hash
        assert reopened.state_hash == final.state_hash
        assert reopened.evidence_references == final.evidence_references


def test_agent_facade_preserves_pre_read_source_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "craft.md").write_text(
        "# Evidence\n\nThis sentence would match if its Source were permitted.\n",
        encoding="utf-8",
    )
    with initialize_library(
        LibraryConfig(name="Excluded interactive research"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, snapshot_id, policy_id, source_id = _prepare_library(repository, root)
        facade = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=current_fts_runtime_profile().profile_version,
            clock=_TickingClock(),
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="Does excluded material remain unread?",
                intended_use="access-boundary verification",
                success_criteria=("preserve exclusion",),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
            exclusions=QueryExclusions(source_ids=(source_id,)),
        )
        turn = facade.recall(
            opened.session_id,
            "sentence Source permitted",
            command_id="command_exclusion_001",
        )

        assert turn.evidence_packet.source_fragments == ()
        assert turn.context.evidence_references[0].artifact_kind.value == "evidence_packet"
        assert len(turn.context.evidence_references) == 1


def test_agent_facade_reuses_and_explicitly_destroys_scope_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "craft.md").write_text(
        "# Evidence\n\nBlocking changes power. Eyelines reveal attention.\n",
        encoding="utf-8",
    )
    with initialize_library(
        LibraryConfig(name="Cached interactive research"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, snapshot_id, policy_id, _source_id = _prepare_library(repository, root)
        facade = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=current_fts_runtime_profile().profile_version,
            clock=_TickingClock(),
        )
        reads = 0
        original_read = facade._recall_backend.read_permitted_fragments

        def counting_read(authorization: AuthorizedRead) -> tuple[SourceFragmentText, ...]:
            nonlocal reads
            reads += 1
            return original_read(authorization)

        monkeypatch.setattr(
            facade._recall_backend,
            "read_permitted_fragments",
            counting_read,
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="How do staging choices carry dramatic information?",
                intended_use="session-cache verification",
                success_criteria=("reuse one authorized read",),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
        )
        facade.recall(
            opened.session_id,
            "blocking changes power",
            command_id="command_cache_blocking_001",
        )
        facade.recall(
            opened.session_id,
            "eyelines reveal attention",
            command_id="command_cache_eyelines_001",
        )
        stats = facade.cache_stats(opened.session_id)
        assert stats is not None
        assert reads == 1
        assert stats.index_build_count == 1
        assert stats.search_count == 2

        facade.close_scope_sessions()
        assert facade.cache_stats(opened.session_id) is None
        facade.recall(
            opened.session_id,
            "power attention",
            command_id="command_cache_rebuild_001",
        )
        assert reads == 2


def test_recall_command_retry_is_exact_and_rejects_changed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "craft.md").write_text(
        "# Evidence\n\nRepeated commands must not duplicate session evidence.\n",
        encoding="utf-8",
    )
    with initialize_library(
        LibraryConfig(name="Idempotent interactive research"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, snapshot_id, policy_id, _source_id = _prepare_library(repository, root)
        facade = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=current_fts_runtime_profile().profile_version,
            clock=_TickingClock(),
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="Can a retried transport preserve one exact research turn?",
                intended_use="retry verification",
                success_criteria=("no duplicate question", "no duplicate evidence"),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
        )
        first = facade.recall(
            opened.session_id,
            "repeated commands duplicate session evidence",
            command_id="command_retry_001",
        )
        repeated = facade.recall(
            opened.session_id,
            "repeated commands duplicate session evidence",
            command_id="command_retry_001",
        )

        assert repeated == first
        assert repeated.turn_hash == first.turn_hash
        assert len(SQLiteResearchSessionRepository(repository).list_events(opened.session_id)) == 2
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM processing_runs WHERE kind = 'recall'"
            ).fetchone()[0]
            == 1
        )

        def _forbid_full_packet_load(_packet_id: str) -> object:
            raise AssertionError("compact replay must not load the full EvidencePacket")

        monkeypatch.setattr(
            facade._recall_backend,
            "load_evidence_packet",
            _forbid_full_packet_load,
        )
        compact_replay = facade.recall(
            opened.session_id,
            "repeated commands duplicate session evidence",
            command_id="command_retry_001",
        )
        assert compact_replay == first
        with pytest.raises(
            ResearchSessionPersistenceError,
            match="different recall input",
        ):
            facade.recall(
                opened.session_id,
                "a changed question must fail closed",
                command_id="command_retry_001",
            )


def test_partial_recall_command_resumes_without_duplicate_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "craft.md").write_text(
        "# Evidence\n\nA partial command can resume from its durable question receipt.\n",
        encoding="utf-8",
    )
    with initialize_library(
        LibraryConfig(name="Resumable interactive research"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, snapshot_id, policy_id, _source_id = _prepare_library(repository, root)
        facade = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=current_fts_runtime_profile().profile_version,
            clock=_TickingClock(),
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="Can a command resume after recall fails?",
                intended_use="partial-turn recovery",
                success_criteria=("one question event", "one evidence event"),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
        )
        real_recall = facade._recall.recall_in_session

        def _fail_once(_request: object, _scope_session: object) -> object:
            raise RuntimeError("simulated transport interruption")

        monkeypatch.setattr(facade._recall, "recall_in_session", _fail_once)
        with pytest.raises(RuntimeError, match="simulated transport interruption"):
            facade.recall(
                opened.session_id,
                "partial command durable question receipt",
                command_id="command_resume_001",
            )
        assert len(SQLiteResearchSessionRepository(repository).list_events(opened.session_id)) == 1

        monkeypatch.setattr(facade._recall, "recall_in_session", real_recall)
        resumed = facade.recall(
            opened.session_id,
            "partial command durable question receipt",
            command_id="command_resume_001",
        )
        events = SQLiteResearchSessionRepository(repository).list_events(opened.session_id)
        assert len(events) == 2
        assert resumed.question_event_id == events[0].event_id
        assert resumed.evidence_event_id == events[1].event_id


def test_interactive_contracts_and_facade_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "craft.md").write_text(
        "# Evidence\n\nA compact projection remains bound to its full audit packet.\n",
        encoding="utf-8",
    )
    with initialize_library(
        LibraryConfig(name="Fail-closed interactive research"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, snapshot_id, policy_id, _source_id = _prepare_library(repository, root)
        profile = current_fts_runtime_profile().profile_version
        with pytest.raises(TypeError, match="LibraryRepository"):
            AgentResearchFacade(
                cast(LibraryRepository, object()),
                agent_id="agent:test",
                profile_version=profile,
            )
        with pytest.raises(AgentResearchError, match="agent_id"):
            AgentResearchFacade(repository, agent_id="", profile_version=profile)
        with pytest.raises(AgentResearchError, match="scope_session_capacity"):
            AgentResearchFacade(
                repository,
                agent_id="agent:test",
                profile_version=profile,
                scope_session_capacity=0,
            )

        facade = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=profile,
            clock=_TickingClock(),
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="Does the compact projection fail closed?",
                intended_use="contract verification",
                success_criteria=("reject drift",),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
        )
        turn = facade.recall(
            opened.session_id,
            "compact projection full audit packet",
            command_id="command_contract_edges_001",
        )
        completion = facade._sessions.load_recall_command_completion(turn.command_id)
        assert completion is not None

        for key, value, match in (
            ("schema_id", "dithyramba.agent_evidence_packet/9.9", "schema_id"),
            ("result_status", "no_evidence", "result_status"),
            ("retrieval_selected_count", 0, "retrieval_selected_count"),
        ):
            payload = turn.evidence_packet.model_dump(mode="json")
            payload[key] = value
            semantic = turn.evidence_packet.semantic_payload()
            if key == "retrieval_selected_count":
                retrieval = semantic["retrieval"]
                assert isinstance(retrieval, dict)
                retrieval["selected_count"] = value
            else:
                semantic[key] = value
            payload["projection_hash"] = canonical_sha256_hex(semantic)
            with pytest.raises(ValidationError, match=match):
                AgentEvidencePacket.model_validate_json(json.dumps(payload))
        with pytest.raises(TypeError, match="exact EvidencePacket"):
            AgentEvidencePacket.create(cast(EvidencePacket, object()))

        context_payload = turn.context.model_dump(mode="json")
        context_payload["schema_id"] = "dithyramba.agent_session_context/9.9"
        with pytest.raises(ValidationError, match="schema_id"):
            type(turn.context).model_validate_json(json.dumps(context_payload))
        context_payload = turn.context.model_dump(mode="json")
        context_payload["context_hash"] = "b" * 64
        with pytest.raises(ValidationError, match="context_hash"):
            type(turn.context).model_validate_json(json.dumps(context_payload))
        session = facade._sessions.get_session(opened.session_id)
        state = facade._sessions.get_state(opened.session_id)
        with pytest.raises(ValueError, match="another ResearchSession"):
            type(turn.context).create(
                session=session,
                state=state.model_copy(update={"session_hash": "b" * 64}),
            )

        turn_payload = turn.model_dump(mode="json")
        turn_payload["schema_id"] = "dithyramba.agent_research_turn/9.9"
        with pytest.raises(ValidationError, match="schema_id"):
            AgentResearchTurn.model_validate_json(json.dumps(turn_payload))
        turn_payload = turn.model_dump(mode="json")
        turn_payload["turn_hash"] = "b" * 64
        with pytest.raises(ValidationError, match="turn_hash"):
            AgentResearchTurn.model_validate_json(json.dumps(turn_payload))

        with pytest.raises(AgentResearchError, match="packet hash drifted"):
            facade._turn_from_completion(
                replace(
                    completion,
                    agent_evidence_json=None,
                    evidence_packet_hash="b" * 64,
                )
            )
        with pytest.raises(AgentResearchError, match="projection drifted"):
            facade._turn_from_completion(replace(completion, evidence_packet_hash="b" * 64))
        with pytest.raises(AgentResearchError, match="turn hash drifted"):
            facade._turn_from_completion(replace(completion, turn_hash="b" * 64))

        original_completion_loader = facade._sessions.load_recall_command_completion

        def _mismatched_completion(
            _command_id: str,
            *,
            connection: object | None = None,
        ) -> RecallCommandCompletion:
            del connection
            return replace(completion, command_hash="b" * 64)

        monkeypatch.setattr(
            facade._sessions,
            "load_recall_command_completion",
            _mismatched_completion,
        )
        with pytest.raises(AgentResearchError, match="disagrees with its start"):
            facade.recall(
                opened.session_id,
                "compact projection full audit packet",
                command_id="command_contract_edges_001",
            )
        monkeypatch.setattr(
            facade._sessions,
            "load_recall_command_completion",
            original_completion_loader,
        )

        bad_clock = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=profile,
            clock=lambda: datetime(2026, 8, 1),
        )
        with pytest.raises(AgentResearchError, match="timezone-aware"):
            bad_clock._timestamp()

        facade.attach_evidence(
            opened.session_id,
            summary="Reattach the exact packet for a separate journal purpose.",
            references=(turn.context.evidence_references[0],),
        )

        def _ignore_append(*_args: object, **_kwargs: object) -> object:
            return object()

        def _same_context(_session_id: str) -> object:
            return turn.context

        monkeypatch.setattr(facade, "_append", _ignore_append)
        monkeypatch.setattr(facade, "context", _same_context)
        linked = facade.link_candidates(
            opened.session_id,
            summary="No promotion; exercise the typed candidate journal route.",
            references=(),
        )
        assert linked == turn.context


def test_command_receipt_helpers_and_corruption_fail_closed(
    tmp_path: Path,
) -> None:
    assert _validated_command_id("command_helper_001") == "command_helper_001"
    for invalid in (cast(str, 1), "command-uncanonical", "command_" + "a" * 100):
        with pytest.raises(ResearchSessionPersistenceError, match="command_id"):
            _validated_command_id(invalid)
    assert _payload_text({"value": "ok"}, "value") == "ok"
    invalid_text_payloads: tuple[dict[str, object], ...] = (
        {},
        {"value": 1},
        {"value": ""},
    )
    for payload in invalid_text_payloads:
        with pytest.raises(ValueError, match="value"):
            _payload_text(payload, "value")
    assert _payload_optional_text({}, "value") is None
    assert _payload_optional_text({"value": "ok"}, "value") == "ok"
    with pytest.raises(ValueError, match="value"):
        _payload_optional_text({"value": 1}, "value")
    assert _canonical_json_text('{"a":1}', "payload") == '{"a":1}'
    for value, match in (("", "nonblank"), ("{", "valid JSON"), ('{"a": 1}', "canonical")):
        with pytest.raises(ResearchSessionPersistenceError, match=match):
            _canonical_json_text(value, "payload")

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "craft.md").write_text("# Evidence\n\nCommand receipts are immutable.\n")
    with initialize_library(
        LibraryConfig(name="Command receipt corruption"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, snapshot_id, policy_id, _source_id = _prepare_library(repository, root)
        facade = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=current_fts_runtime_profile().profile_version,
            clock=_TickingClock(),
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="Do command receipts fail closed?",
                intended_use="corruption verification",
                success_criteria=("reject drift",),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
        )
        session = facade._sessions.get_session(opened.session_id)
        request = QueryRequest(
            question="command receipts immutable",
            library_id=session.library_id,
            collection_ids=session.collection_ids,
            corpus_snapshot_id=session.corpus_snapshot_id,
            access_policy_id=session.access_policy_id,
            purpose=session.purpose,
            exclusions=session.scope.exclusions,
        )
        with pytest.raises(TypeError, match="QueryRequest"):
            facade._sessions.begin_recall_command(
                session_id=session.session_id,
                command_id="command_invalid_request_001",
                request=cast(QueryRequest, object()),
                actor_id="agent:test",
                occurred_at="2026-08-01T12:00:00.000000Z",
            )
        with pytest.raises(ResearchSessionPersistenceError, match="scope differs"):
            facade._sessions.begin_recall_command(
                session_id=session.session_id,
                command_id="command_wrong_scope_001",
                request=request.model_copy(update={"purpose": "other"}),
                actor_id="agent:test",
                occurred_at="2026-08-01T12:00:00.000000Z",
            )
        with pytest.raises(TypeError, match="RecallCommandStart"):
            facade._sessions.complete_recall_command(
                start=cast(RecallCommandStart, object()),
                evidence_event=cast(SessionEvent, object()),
                evidence_packet_id="packet_invalid",
                evidence_packet_hash="a" * 64,
                agent_evidence_json="{}",
                context_json="{}",
                turn_hash="a" * 64,
                occurred_at="2026-08-01T12:00:00.000000Z",
            )

        turn = facade.recall(
            opened.session_id,
            "command receipts immutable",
            command_id="command_receipt_edges_001",
        )
        start = facade._sessions._load_command_start(turn.command_id)
        completion = facade._sessions.load_recall_command_completion(turn.command_id)
        assert start is not None and completion is not None
        assert facade._sessions.persist_session(session) == session
        with pytest.raises(TypeError, match="ResearchSession"):
            facade._sessions.persist_session(cast(ResearchSession, object()))
        with pytest.raises(TypeError, match="LibraryRepository"):
            SQLiteResearchSessionRepository(cast(LibraryRepository, object()))
        with pytest.raises(ResearchSessionNotFoundError, match="does not exist"):
            facade._sessions.get_session("research_session_" + "b" * 32)
        with pytest.raises(TypeError, match="SessionEvent"):
            facade._sessions.append_event(cast(SessionEvent, object()))
        events = facade._sessions.list_events(opened.session_id)
        assert facade._sessions.append_event(events[0]) == events[0]

        with pytest.raises(ResearchSessionPersistenceError, match="content identity"):
            _validated_session(session.model_copy(update={"session_hash": "b" * 64}))
        with pytest.raises(ResearchSessionPersistenceError, match="content identity"):
            _validated_event(events[0].model_copy(update={"event_hash": "b" * 64}))
        with pytest.raises(ResearchSessionPersistenceError, match="persisted ResearchSession"):
            _load_session("{")
        with pytest.raises(ResearchSessionPersistenceError, match="persisted SessionEvent"):
            _load_event("{")
        session_json = canonical_json_bytes(session.model_dump(mode="json")).decode("utf-8")
        event_json = canonical_json_bytes(events[0].model_dump(mode="json")).decode("utf-8")
        with pytest.raises(ResearchSessionPersistenceError, match="not canonical JSON"):
            _load_session(session_json.replace(":", ": ", 1))
        with pytest.raises(ResearchSessionPersistenceError, match="not canonical JSON"):
            _load_event(event_json.replace(":", ": ", 1))

        with pytest.raises(ResearchSessionPersistenceError, match="different Library"):
            facade._sessions._validate_session_scope(
                session.model_copy(
                    update={
                        "scope": replace(
                            session.scope,
                            library_id="library_" + "b" * 32,
                        )
                    }
                )
            )
        with pytest.raises(ResearchSessionPersistenceError, match="snapshot ID/hash"):
            facade._sessions._validate_session_scope(
                session.model_copy(update={"scope": replace(session.scope, snapshot_hash="b" * 64)})
            )
        with pytest.raises(ResearchSessionPersistenceError, match="exceed"):
            facade._sessions._validate_session_scope(
                session.model_copy(
                    update={
                        "scope": replace(
                            session.scope,
                            collection_ids=("collection_" + "b" * 32,),
                        )
                    }
                )
            )
        with pytest.raises(ResearchSessionPersistenceError, match="not authorized"):
            facade._sessions._validate_session_scope(
                session.model_copy(update={"access_policy_id": "policy_absent"})
            )

        packet_reference = turn.context.evidence_references[0]
        with pytest.raises(ResearchSessionPersistenceError, match="absent, stale"):
            facade._sessions._validate_artifacts(
                session,
                (packet_reference.model_copy(update={"artifact_hash": "b" * 64}),),
            )
        with pytest.raises(ResearchSessionPersistenceError, match="absent, stale"):
            facade._sessions._validate_artifacts(
                session,
                (packet_reference.model_copy(update={"artifact_id": "packet_" + "b" * 32}),),
            )
        source_reference = turn.context.evidence_references[1]
        with pytest.raises(ResearchSessionPersistenceError, match="absent, stale"):
            facade._sessions._validate_artifacts(
                session,
                (source_reference.model_copy(update={"artifact_hash": "b" * 64}),),
            )
        assert "absent, stale" in str(_artifact_error(packet_reference))

        same = facade._sessions.complete_recall_command(
            start=start,
            evidence_event=facade._sessions.list_events(opened.session_id)[1],
            evidence_packet_id=completion.evidence_packet_id,
            evidence_packet_hash=completion.evidence_packet_hash,
            agent_evidence_json=completion.agent_evidence_json or "{}",
            context_json=completion.context_json,
            turn_hash=completion.turn_hash,
            occurred_at="2026-08-01T12:00:00.000000Z",
        )
        assert same == completion
        with pytest.raises(ResearchSessionPersistenceError, match="different recall input"):
            _require_same_command(start, "b" * 64, start.query_request_id)
        with pytest.raises(ResearchSessionPersistenceError, match="disagrees"):
            _require_completion_matches_start(
                replace(completion, question_event_id="session_event_" + "b" * 32),
                start,
            )

        connection = repository._store.connection

        second = facade.open(
            brief=ResearchSessionBrief.create(
                question="Is a foreign session event rejected?",
                intended_use="scope verification",
                success_criteria=("reject foreign event",),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
        )
        second_session = facade._sessions.get_session(second.session_id)
        second_request = QueryRequest(
            question="foreign session event",
            library_id=second_session.library_id,
            collection_ids=second_session.collection_ids,
            corpus_snapshot_id=second_session.corpus_snapshot_id,
            access_policy_id=second_session.access_policy_id,
            purpose=second_session.purpose,
            exclusions=second_session.scope.exclusions,
        )
        second_start = facade._sessions.begin_recall_command(
            session_id=second.session_id,
            command_id="command_foreign_session_001",
            request=second_request,
            actor_id="agent:test",
            occurred_at="2026-08-01T12:00:00.000000Z",
        )
        with pytest.raises(ResearchSessionPersistenceError, match="stale ResearchSession"):
            facade._sessions._insert_event(connection, session, second_start.question_event)
        unfinished_request = request.model_copy(
            update={"question": "uncompleted command must reject a foreign event"}
        )
        unfinished_start = facade._sessions.begin_recall_command(
            session_id=session.session_id,
            command_id="command_uncompleted_001",
            request=unfinished_request,
            actor_id="agent:test",
            occurred_at="2026-08-01T12:00:00.000000Z",
        )
        with pytest.raises(ResearchSessionPersistenceError, match="another session"):
            facade._sessions.complete_recall_command(
                start=unfinished_start,
                evidence_event=second_start.question_event,
                evidence_packet_id=completion.evidence_packet_id,
                evidence_packet_hash=completion.evidence_packet_hash,
                agent_evidence_json=completion.agent_evidence_json or "{}",
                context_json=completion.context_json,
                turn_hash=completion.turn_hash,
                occurred_at="2026-08-01T12:00:00.000000Z",
            )

        def insert_outbox(
            *,
            event_id: str,
            event_type: str,
            command_id: str,
            payload_json: str,
            payload_hash: str,
        ) -> None:
            connection.execute(
                """
                INSERT INTO event_outbox(
                    event_id, event_type, aggregate_type, aggregate_id,
                    payload_json, payload_hash, occurred_at
                ) VALUES (?, ?, 'recall_command', ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    command_id,
                    payload_json,
                    payload_hash,
                    "2026-08-01T12:00:00.000000Z",
                ),
            )

        assert (
            facade._sessions._load_command_payload(
                "test.missing",
                "command_missing_001",
            )
            is None
        )
        insert_outbox(
            event_id="event_invalid_json",
            event_type="test.invalid_json",
            command_id="command_invalid_json_001",
            payload_json="{",
            payload_hash="a" * 64,
        )
        with pytest.raises(ResearchSessionPersistenceError, match="valid JSON"):
            facade._sessions._load_command_payload(
                "test.invalid_json",
                "command_invalid_json_001",
            )
        insert_outbox(
            event_id="event_noncanonical_json",
            event_type="test.noncanonical",
            command_id="command_noncanonical_001",
            payload_json='{"a": 1}',
            payload_hash=canonical_sha256_hex({"a": 1}),
        )
        with pytest.raises(ResearchSessionPersistenceError, match="not canonical"):
            facade._sessions._load_command_payload(
                "test.noncanonical",
                "command_noncanonical_001",
            )
        canonical = canonical_json_bytes({"a": 1}).decode("utf-8")
        insert_outbox(
            event_id="event_hash_drift",
            event_type="test.hash_drift",
            command_id="command_hash_drift_001",
            payload_json=canonical,
            payload_hash="b" * 64,
        )
        with pytest.raises(ResearchSessionPersistenceError, match="hash drifted"):
            facade._sessions._load_command_payload(
                "test.hash_drift",
                "command_hash_drift_001",
            )
        for index in range(2):
            insert_outbox(
                event_id=f"event_duplicate_{index}",
                event_type="test.duplicate",
                command_id="command_duplicate_001",
                payload_json=canonical,
                payload_hash=canonical_sha256_hex({"a": 1}),
            )
        with pytest.raises(ResearchSessionPersistenceError, match="more than one"):
            facade._sessions._load_command_payload(
                "test.duplicate",
                "command_duplicate_001",
            )

        invalid_completion_payload = canonical_json_bytes(
            {"schema": "dithyramba.recall_command_completed/1.1"}
        ).decode("utf-8")
        insert_outbox(
            event_id="event_invalid_completion",
            event_type="research_session.recall_command_completed",
            command_id="command_invalid_completion_001",
            payload_json=invalid_completion_payload,
            payload_hash=canonical_sha256_hex(
                {"schema": "dithyramba.recall_command_completed/1.1"}
            ),
        )
        with pytest.raises(ResearchSessionPersistenceError, match="completion is invalid"):
            facade._sessions.load_recall_command_completion("command_invalid_completion_001")

        invalid_start = {
            "schema": "dithyramba.recall_command_started/1.0",
            "command_id": "command_invalid_start_001",
            "command_hash": "a" * 64,
            "question_event_id": events[0].event_id,
            "question_event_hash": events[0].event_hash,
        }
        invalid_start_json = canonical_json_bytes(invalid_start).decode("utf-8")
        insert_outbox(
            event_id="event_invalid_start",
            event_type="research_session.recall_command_started",
            command_id="command_invalid_start_001",
            payload_json=invalid_start_json,
            payload_hash=canonical_sha256_hex(invalid_start),
        )
        with pytest.raises(ResearchSessionPersistenceError, match="start is invalid"):
            facade._sessions._load_command_start("command_invalid_start_001")
