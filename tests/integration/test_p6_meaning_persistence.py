from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest

import dithyramba.persistence.meaning as meaning_persistence
from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    CompiledAccess,
    PermittedManifest,
    PermittedManifestItem,
    PermittedToken,
    PolicyEffect,
    RequestScope,
)
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
    CoverageState,
    EntityKind,
    EntityProposal,
    EvidenceLinkProposal,
    ExactMentionProposal,
    ExtractionProfile,
    GeneratorProfile,
    MeaningBudgetError,
    MeaningContractError,
    MeaningGenerator,
    MeaningInputFragment,
    MeaningNotFoundError,
    MeaningOmission,
    MeaningOutputManifest,
    MeaningPersistenceError,
    MeaningProposal,
    MeaningReviewAction,
    MeaningReviewError,
    MeaningReviewQueueItem,
    MeaningReviewRequest,
    MeaningReviewScope,
    MeaningReviewTargetType,
    MeaningRunRequest,
    MeaningRunResult,
    MeaningRunStatus,
    OmissionCategory,
    ReviewBudget,
    StatementKind,
    StatementProposal,
    TimeContextProposal,
    VoiceKind,
    VoiceProposal,
)
from dithyramba.persistence import (
    AuthorizedRead,
    LibraryRepository,
    SQLiteMeaningRepository,
    initialize_library,
    open_library,
)

NOW = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-21T12:00:00.000000Z"
PUBLIC_TEXT = "Автор: пам'ять є практикою. Оповідач: пам'ять є монтажем."
HOLDOUT_TEXT = "PRIVATE-CANARY-HOLDOUT"
SECOND_TEXT = "Автор: пам'ять є повторенням."
MEANING_TEXT = "практика і монтаж"  # noqa: RUF001 - intentional Ukrainian text


def _clock() -> datetime:
    return NOW


def _digest(value: str) -> str:
    return sha256_hex(value.encode("utf-8"))


@dataclass(slots=True)
class SemanticFixture:
    repository: LibraryRepository
    meaning: SQLiteMeaningRepository
    request: MeaningRunRequest
    proposal: MeaningProposal
    collection_id: str
    public_fragment_id: str
    holdout_fragment_id: str


@pytest.fixture
def semantic_fixture(tmp_path: Path) -> Iterator[SemanticFixture]:
    data_root = tmp_path / "data"
    public_root = tmp_path / "public"
    holdout_root = tmp_path / "holdout"
    public_root.mkdir()
    holdout_root.mkdir()
    library = LibraryConfig(name="P6 semantic")
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
                name="Holdout",
                kind=CollectionKind.HOLDOUT,
                roots=(build_collection_root(holdout_root, data_root=data_root),),
            )
        )
        _insert_source_fixture(
            repository,
            suffix="public",
            collection_id=public.config.collection_id,
            text=PUBLIC_TEXT,
            membership_state="active",
        )
        _insert_source_fixture(
            repository,
            suffix="holdout",
            collection_id=holdout.config.collection_id,
            text=HOLDOUT_TEXT,
            membership_state="holdout",
        )
        snapshot = repository.freeze_snapshot(
            (public.config.collection_id, holdout.config.collection_id)
        )
        policy = AccessPolicySnapshot(
            access_policy_id="policy_semantic",
            library_id=library.library_id,
            allowed_purposes=("research",),
            collection_rules=(
                CollectionRule(public.config.collection_id, PolicyEffect.ALLOW),
                CollectionRule(holdout.config.collection_id, PolicyEffect.ALLOW),
            ),
        )
        repository.persist_access_policy(name="Semantic local", snapshot=policy)
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
                prompt_version="p6-test-v1",
            ),
            review_budget=ReviewBudget(),
            code_version="test",
        )
        yield SemanticFixture(
            repository=repository,
            meaning=SQLiteMeaningRepository(repository),
            request=request,
            proposal=_deep_proposal(public.config.collection_id),
            collection_id=public.config.collection_id,
            public_fragment_id="fragment_public",
            holdout_fragment_id="fragment_holdout",
        )


def test_policy_first_extract_persists_exact_voice_grounded_candidates_without_holdout(
    semantic_fixture: SemanticFixture,
) -> None:
    fixture = semantic_fixture
    seen_inputs: list[tuple[str, str]] = []
    events_before = fixture.repository._store.connection.execute(
        "SELECT count(*) FROM event_outbox"
    ).fetchone()[0]

    def generator(
        inputs: tuple[MeaningInputFragment, ...],
        _request: MeaningRunRequest,
    ) -> MeaningProposal:
        seen_inputs.extend((item.source_fragment_id, item.text) for item in inputs)
        return fixture.proposal

    result = fixture.meaning.extract(request=fixture.request, generator=generator)

    assert result.run.status is MeaningRunStatus.SUCCEEDED
    assert result.run.scope_hash == fixture.request.scope.exclusion_hash
    assert (
        result.run.permitted_set_hash
        == fixture.meaning._authorize(fixture.request).compiled.manifest.permitted_set_hash
    )
    assert result.run.code_version == fixture.request.code_version
    assert len(result.run.receipt_hash) == 64
    assert result.coverage_report.state is CoverageState.COMPLETE
    assert seen_inputs == [(fixture.public_fragment_id, PUBLIC_TEXT)]
    assert tuple(item.source_fragment_id for item in result.read_receipt.items) == (
        fixture.public_fragment_id,
    )
    assert result.read_receipt.items[0].source_version_id == "source_version_public"
    assert result.read_receipt.items[0].source_id == "source_public"
    assert result.read_receipt.items[0].collection_ids == (fixture.collection_id,)
    assert fixture.holdout_fragment_id not in {
        item.source_fragment_id for item in result.read_receipt.items
    }
    assert (
        fixture.repository._store.connection.execute(
            "SELECT count(*) FROM event_outbox"
        ).fetchone()[0]
        == events_before
    )

    statement_rows = fixture.repository._store.connection.execute(
        """
        SELECT s.statement_id, s.voice_id, count(e.evidence_link_id)
        FROM statements AS s
        LEFT JOIN evidence_links AS e ON e.statement_id = s.statement_id
        GROUP BY s.statement_id, s.voice_id
        """
    ).fetchall()
    assert statement_rows and all(int(row[2]) >= 1 for row in statement_rows)
    evidence_rows = fixture.repository._store.connection.execute(
        """
        SELECT e.source_fragment_id, e.quote_start, e.quote_end,
               e.quote_text, e.quote_sha256, e.attributed_voice_id, s.voice_id
        FROM evidence_links AS e JOIN statements AS s ON s.statement_id = e.statement_id
        """
    ).fetchall()
    assert evidence_rows
    for row in evidence_rows:
        assert str(row[0]) == fixture.public_fragment_id
        quote = PUBLIC_TEXT[int(row[1]) : int(row[2])]
        assert quote == str(row[3])
        assert _digest(quote) == str(row[4])
        assert str(row[5]) == str(row[6])
    meaning_rows = fixture.repository._store.connection.execute(
        "SELECT meaning_text, voice_id FROM concept_meanings ORDER BY voice_id"
    ).fetchall()
    assert [str(row[0]) for row in meaning_rows] == [MEANING_TEXT, MEANING_TEXT]
    assert len({str(row[1]) for row in meaning_rows}) == 2

    semantic_values = _semantic_values(fixture.repository._store.connection)
    assert HOLDOUT_TEXT not in semantic_values
    assert fixture.holdout_fragment_id not in semantic_values


def test_import_is_idempotent_and_partial_or_denied_proposals_create_no_candidates(
    semantic_fixture: SemanticFixture,
) -> None:
    fixture = semantic_fixture
    first = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    counts_before = _candidate_counts(fixture.repository._store.connection)
    second = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)

    assert first.run.processing_run_id == second.run.processing_run_id
    assert second.deduplicated is True
    assert _candidate_counts(fixture.repository._store.connection) == counts_before

    partial_request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="partial-v1"),
    )
    partial = MeaningProposal(
        state=CoverageState.PARTIAL,
        omissions=(MeaningOmission(OmissionCategory.FAILURE, "extract_partial", 1),),
        failure_code="extract_partial",
    )
    partial_result = fixture.meaning.import_proposal(
        request=partial_request,
        proposal=partial,
    )
    assert partial_result.run.status is MeaningRunStatus.PARTIAL
    assert partial_result.run.output.candidate_count == 0
    assert partial_result.coverage_report.state is CoverageState.PARTIAL
    assert _candidate_counts(fixture.repository._store.connection) == counts_before
    assert (
        fixture.repository._store.connection.execute(
            "SELECT count(*) FROM sqlite_schema WHERE name = 'evidence_gaps'"
        ).fetchone()[0]
        == 0
    )

    denied_request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="denied-v1"),
    )
    denied_quote = HOLDOUT_TEXT
    denied = MeaningProposal(
        voices=(VoiceProposal("author", VoiceKind.AUTHOR, "Автор"),),
        statements=(
            StatementProposal(
                "denied_statement",
                StatementKind.SOURCE_CLAIM,
                "Заборонене твердження",
                "author",
                fixture.collection_id,
                (
                    EvidenceLinkProposal(
                        fixture.holdout_fragment_id,
                        0,
                        len(denied_quote),
                        denied_quote,
                        _digest(denied_quote),
                        "author",
                    ),
                ),
            ),
        ),
    )
    denied_result = fixture.meaning.import_proposal(request=denied_request, proposal=denied)
    assert denied_result.run.status is MeaningRunStatus.FAILED
    assert denied_result.run.error_code == "invalid_proposal"
    assert denied_result.run.output.candidate_count == 0
    assert tuple(item.source_fragment_id for item in denied_result.read_receipt.items) == (
        fixture.public_fragment_id,
    )
    assert _candidate_counts(fixture.repository._store.connection) == counts_before


def test_review_queue_budget_revise_append_only_and_source_preservation(
    semantic_fixture: SemanticFixture,
) -> None:
    fixture = semantic_fixture
    original = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    before_source = _source_texts(fixture.repository)
    queue = fixture.meaning.list_review_queue(limit=100)
    statements = [item for item in queue if item.target_type is MeaningReviewTargetType.STATEMENT]
    statement = statements[0]
    revision_target = statements[1]
    other = next(item for item in queue if item.target_id != statement.target_id)

    session = fixture.meaning.start_review_session(ReviewBudget(review_decision_limit=1))
    decision = fixture.meaning.create_review_decision(
        MeaningReviewRequest(
            review_session_id=session.review_session_id,
            target_type=statement.target_type,
            target_id=statement.target_id,
            target_hash=statement.target_hash,
            action=MeaningReviewAction.ACCEPT,
            reason="source grounded",
            authority="reviewer",
            scope=MeaningReviewScope((fixture.collection_id,), "research"),
        )
    )
    assert decision.action is MeaningReviewAction.ACCEPT
    assert statement.target_id not in {
        item.target_id for item in fixture.meaning.list_review_queue(limit=100)
    }
    with pytest.raises(MeaningBudgetError, match="exhausted"):
        fixture.meaning.create_review_decision(
            MeaningReviewRequest(
                review_session_id=session.review_session_id,
                target_type=other.target_type,
                target_id=other.target_id,
                target_hash=other.target_hash,
                action=MeaningReviewAction.DEFER,
                reason="later",
                authority="reviewer",
                scope=MeaningReviewScope((fixture.collection_id,), "research"),
            )
        )

    revision_request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="revision-v1"),
    )
    revision_proposal = _deep_proposal(
        fixture.collection_id,
        statement_suffix=" — уточнено",
    )
    revision = fixture.meaning.import_proposal(
        request=revision_request,
        proposal=revision_proposal,
    )
    replacement_id = next(
        candidate
        for candidate in revision.run.output.statement_ids
        if candidate not in original.run.output.statement_ids
    )
    revision_session = fixture.meaning.start_review_session()
    revise = fixture.meaning.create_review_decision(
        MeaningReviewRequest(
            review_session_id=revision_session.review_session_id,
            target_type=revision_target.target_type,
            target_id=revision_target.target_id,
            target_hash=revision_target.target_hash,
            action=MeaningReviewAction.REVISE,
            reason="replacement preserves evidence",
            authority="reviewer",
            scope=MeaningReviewScope((fixture.collection_id,), "research"),
            replacement_target_id=replacement_id,
        )
    )
    assert revise.replacement_target_id == replacement_id
    assert _source_texts(fixture.repository) == before_source

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        fixture.repository._store.connection.execute(
            "UPDATE meaning_review_decisions SET reason = 'mutated' WHERE review_decision_id = ?",
            (decision.review_decision_id,),
        )
    with pytest.raises(sqlite3.DatabaseError):
        fixture.repository._store.connection.execute(
            "UPDATE source_fragments SET text = text WHERE source_fragment_id = ?",
            (fixture.public_fragment_id,),
        )


def test_backlog_refusal_is_receipted_and_direct_sql_cannot_link_source_text(
    semantic_fixture: SemanticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = semantic_fixture
    first = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    assert fixture.meaning.unresolved_candidate_count() >= 1
    monkeypatch.setattr(meaning_persistence, "_BACKLOG_HARD_LIMIT", 1)
    called = False

    def must_not_run(
        _inputs: tuple[object, ...],
        _request: MeaningRunRequest,
    ) -> MeaningProposal:
        nonlocal called
        called = True
        return fixture.proposal

    refusal_request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="backlog-v1"),
    )
    refused = fixture.meaning.extract(request=refusal_request, generator=must_not_run)
    assert called is False
    assert refused.run.status is MeaningRunStatus.REFUSED
    assert refused.run.error_code == "review_backlog_limit"
    assert refused.coverage_report.state is CoverageState.REFUSED
    assert refused.read_receipt.items == ()
    assert refused.run.review_budget_hash == refusal_request.review_budget.budget_hash

    evidence = fixture.repository._store.connection.execute(
        """
        SELECT statement_id, source_fragment_id, attributed_voice_id,
               polarity, alignment, quote_start, quote_end, quote_text,
               quote_sha256, extraction_method
        FROM evidence_links LIMIT 1
        """
    ).fetchone()
    assert evidence is not None
    direct_id = "evidence_direct_forbidden"
    with pytest.raises(sqlite3.DatabaseError):
        fixture.repository._store.connection.execute(
            """
            INSERT INTO evidence_links(
                evidence_link_id, statement_id, source_fragment_id,
                attributed_voice_id, polarity, alignment, limits_text,
                quote_start, quote_end, quote_text, quote_sha256,
                extraction_method, status, content_hash,
                first_processing_run_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?)
            """,
            (
                direct_id,
                str(evidence[0]),
                str(evidence[1]),
                str(evidence[2]),
                str(evidence[3]),
                str(evidence[4]),
                int(evidence[5]),
                int(evidence[6]),
                str(evidence[7]),
                str(evidence[8]),
                str(evidence[9]),
                _digest(direct_id),
                first.run.processing_run_id,
                NOW_TEXT,
            ),
        )
    assert (
        fixture.repository._store.connection.execute(
            "SELECT count(*) FROM evidence_links WHERE evidence_link_id = ?",
            (direct_id,),
        ).fetchone()[0]
        == 0
    )


def test_public_repository_boundaries_fail_closed(semantic_fixture: SemanticFixture) -> None:
    fixture = semantic_fixture
    with pytest.raises(TypeError, match="MeaningRunRequest"):
        fixture.meaning.import_proposal(
            request=cast(MeaningRunRequest, object()),
            proposal=fixture.proposal,
        )
    with pytest.raises(TypeError, match="MeaningProposal"):
        fixture.meaning.import_proposal(
            request=fixture.request,
            proposal=cast(MeaningProposal, object()),
        )
    with pytest.raises(TypeError, match="callable"):
        fixture.meaning.extract(
            request=fixture.request,
            generator=cast(MeaningGenerator, object()),
        )
    with pytest.raises(MeaningNotFoundError, match="ProcessingRun"):
        fixture.meaning.get_run("run_missing")
    with pytest.raises(MeaningReviewError, match="limit"):
        fixture.meaning.list_review_queue(limit=0)
    with pytest.raises(MeaningReviewError, match="limit"):
        fixture.meaning.list_review_queue(limit=True)
    with pytest.raises(MeaningReviewError, match="ReviewBudget"):
        fixture.meaning.start_review_session(object())  # type: ignore[arg-type]
    with pytest.raises(MeaningNotFoundError, match="review session"):
        fixture.meaning.get_review_session("review_session_missing")
    with pytest.raises(TypeError, match="MeaningReviewRequest"):
        fixture.meaning.create_review_decision(object())  # type: ignore[arg-type]
    with pytest.raises(MeaningNotFoundError, match="ReviewDecision"):
        fixture.meaning.get_review_decision("review_missing")

    other_library = replace(
        fixture.request,
        scope=replace(fixture.request.scope, library_id="library_other"),
    )
    with pytest.raises(MeaningContractError, match="another Library"):
        fixture.meaning.import_proposal(request=other_library, proposal=fixture.proposal)
    mismatched_snapshot = replace(
        fixture.request,
        scope=replace(fixture.request.scope, snapshot_hash="a" * 64),
    )
    with pytest.raises(MeaningContractError, match="snapshot ID/hash"):
        fixture.meaning.import_proposal(request=mismatched_snapshot, proposal=fixture.proposal)

    fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    assert len(fixture.meaning.list_review_queue(limit=1)) == 1


def test_generator_failures_invalid_output_and_success_are_idempotently_receipted(
    semantic_fixture: SemanticFixture,
) -> None:
    fixture = semantic_fixture
    failed_request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="throws-v1"),
    )

    def raises_generator(
        _inputs: tuple[MeaningInputFragment, ...],
        _request: MeaningRunRequest,
    ) -> MeaningProposal:
        raise RuntimeError("provider failed")

    first_failed = fixture.meaning.extract(
        request=failed_request,
        generator=raises_generator,
    )
    second_failed = fixture.meaning.extract(
        request=failed_request,
        generator=raises_generator,
    )
    assert first_failed.run.status is MeaningRunStatus.FAILED
    assert first_failed.run.error_code == "generator_failed"
    assert second_failed.run.processing_run_id == first_failed.run.processing_run_id
    assert second_failed.deduplicated is True

    invalid_request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="invalid-output-v1"),
    )

    def invalid_generator(
        _inputs: tuple[MeaningInputFragment, ...],
        _request: MeaningRunRequest,
    ) -> MeaningProposal:
        return cast(MeaningProposal, object())

    invalid = fixture.meaning.extract(request=invalid_request, generator=invalid_generator)
    invalid_again = fixture.meaning.extract(
        request=invalid_request,
        generator=invalid_generator,
    )
    assert invalid.run.error_code == "invalid_generator_output"
    assert invalid_again.run.processing_run_id == invalid.run.processing_run_id
    assert invalid_again.deduplicated is True

    success_request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="extract-success-v1"),
    )

    def successful_generator(
        _inputs: tuple[MeaningInputFragment, ...],
        _request: MeaningRunRequest,
    ) -> MeaningProposal:
        return fixture.proposal

    success = fixture.meaning.extract(request=success_request, generator=successful_generator)
    success_again = fixture.meaning.extract(
        request=success_request,
        generator=successful_generator,
    )
    assert success.run.status is MeaningRunStatus.SUCCEEDED
    assert success_again.run.processing_run_id == success.run.processing_run_id
    assert success_again.deduplicated is True
    assert fixture.meaning.get_run(success.run.processing_run_id) == success


def test_preflight_provider_read_and_backlog_budgets_refuse_before_text_or_generation(
    semantic_fixture: SemanticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = semantic_fixture
    provider_request = replace(
        fixture.request,
        generator=replace(
            fixture.request.generator,
            prompt_version="external-v1",
            external_provider=True,
        ),
    )
    provider_result = fixture.meaning.import_proposal(
        request=provider_request,
        proposal=fixture.proposal,
    )
    assert provider_result.run.error_code == "provider_egress_denied"
    assert provider_result.read_receipt.items == ()

    called = False

    def must_not_run(
        _inputs: tuple[MeaningInputFragment, ...],
        _request: MeaningRunRequest,
    ) -> MeaningProposal:
        nonlocal called
        called = True
        return fixture.proposal

    extracted_provider = fixture.meaning.extract(
        request=provider_request,
        generator=must_not_run,
    )
    assert extracted_provider.run.error_code == "provider_egress_denied"
    assert called is False

    authorization = fixture.meaning._authorize(fixture.request)
    read_limited = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="read-limit-v1"),
        review_budget=ReviewBudget(max_read_fragments=1),
    )
    fake_authorization = _authorization_with_extra_manifest_item(authorization)
    with monkeypatch.context() as context:
        context.setattr(
            SQLiteMeaningRepository,
            "_authorize",
            lambda _self, _request: fake_authorization,
        )
        read_refused = fixture.meaning.extract(request=read_limited, generator=must_not_run)
    assert read_refused.run.error_code == "read_fragment_limit"
    assert read_refused.read_receipt.items == ()
    assert called is False

    fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    backlog_request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="explicit-backlog-v1"),
        review_budget=ReviewBudget(max_unresolved_candidates=1),
    )
    backlog = fixture.meaning.import_proposal(
        request=backlog_request,
        proposal=fixture.proposal,
    )
    assert backlog.run.error_code == "review_backlog_limit"
    assert backlog.read_receipt.items == ()


@pytest.mark.parametrize("state", [CoverageState.FAILED, CoverageState.REFUSED])
def test_noncomplete_terminal_proposals_never_create_candidates(
    semantic_fixture: SemanticFixture,
    state: CoverageState,
) -> None:
    fixture = semantic_fixture
    request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version=f"{state.value}-v1"),
    )
    proposal = MeaningProposal(
        state=state,
        omissions=(MeaningOmission(OmissionCategory.FAILURE, f"{state.value}_failure", 1),),
        failure_code=f"{state.value}_failure",
    )
    result = fixture.meaning.import_proposal(request=request, proposal=proposal)
    assert result.coverage_report.state is state
    assert result.run.output == MeaningOutputManifest()
    assert _candidate_counts(fixture.repository._store.connection) == (0,) * 9


def test_candidate_budget_refusal_is_complete_and_receipted(
    semantic_fixture: SemanticFixture,
) -> None:
    fixture = semantic_fixture
    request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="candidate-limit-v1"),
        review_budget=ReviewBudget(max_candidates=1),
    )
    result = fixture.meaning.import_proposal(request=request, proposal=fixture.proposal)
    assert result.run.status is MeaningRunStatus.REFUSED
    assert result.run.error_code == "candidate_budget_limit"
    assert result.read_receipt.items[0].source_fragment_id == fixture.public_fragment_id
    assert _candidate_counts(fixture.repository._store.connection) == (0,) * 9


def test_permitted_manifest_must_resolve_exactly(
    semantic_fixture: SemanticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = semantic_fixture
    authorization = fixture.meaning._authorize(fixture.request)
    permitted = fixture.repository.read_permitted_fragments(authorization)
    assert len(permitted) == 1

    with monkeypatch.context() as context:
        context.setattr(
            LibraryRepository,
            "read_permitted_fragments",
            lambda _self, _authorization: (),
        )
        with pytest.raises(MeaningPersistenceError, match="resolve exactly"):
            fixture.meaning._read_inputs(authorization)

    extra = replace(permitted[0], source_fragment_id="fragment_extra")
    with monkeypatch.context() as context:
        context.setattr(
            LibraryRepository,
            "read_permitted_fragments",
            lambda _self, _authorization: (*permitted, extra),
        )
        with pytest.raises(MeaningPersistenceError, match="outside manifest"):
            fixture.meaning._read_inputs(authorization)


def test_review_validation_supersede_and_insert_conflict_edges(
    semantic_fixture: SemanticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = semantic_fixture
    fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    queue = fixture.meaning.list_review_queue()
    statement = next(
        item for item in queue if item.target_type is MeaningReviewTargetType.STATEMENT
    )
    other = next(item for item in queue if item.target_id != statement.target_id)
    scope = MeaningReviewScope((fixture.collection_id,), "research")
    session = fixture.meaning.start_review_session(ReviewBudget(review_decision_limit=20))

    def review_request(
        target: object,
        *,
        action: MeaningReviewAction = MeaningReviewAction.DEFER,
        replacement_target_id: str | None = None,
        supersedes_review_decision_id: str | None = None,
        review_scope: MeaningReviewScope = scope,
    ) -> MeaningReviewRequest:
        item = cast(MeaningReviewQueueItem, target)
        return MeaningReviewRequest(
            review_session_id=session.review_session_id,
            target_type=item.target_type,
            target_id=item.target_id,
            target_hash=item.target_hash,
            action=action,
            reason="bounded review",
            authority="reviewer",
            scope=review_scope,
            replacement_target_id=replacement_target_id,
            supersedes_review_decision_id=supersedes_review_decision_id,
        )

    with pytest.raises(MeaningReviewError, match="hash is stale"):
        fixture.meaning.create_review_decision(
            replace(review_request(statement), target_hash="a" * 64)
        )
    with pytest.raises(MeaningReviewError, match="exceeds target"):
        fixture.meaning.create_review_decision(
            replace(
                review_request(statement),
                scope=MeaningReviewScope(("collection_other",), "research"),
            )
        )
    with pytest.raises(MeaningNotFoundError, match="review target"):
        fixture.meaning.create_review_decision(
            replace(review_request(statement), target_id="statement_missing")
        )
    with pytest.raises(MeaningReviewError, match="distinct replacement"):
        fixture.meaning.create_review_decision(
            review_request(
                statement,
                action=MeaningReviewAction.REVISE,
                replacement_target_id=statement.target_id,
            )
        )

    prior = fixture.meaning.create_review_decision(review_request(other))
    with pytest.raises(MeaningReviewError, match="retain target"):
        fixture.meaning.create_review_decision(
            review_request(
                statement,
                action=MeaningReviewAction.SUPERSEDE,
                supersedes_review_decision_id=prior.review_decision_id,
            )
        )

    target_prior = fixture.meaning.create_review_decision(review_request(statement))
    supersede_request = review_request(
        statement,
        action=MeaningReviewAction.SUPERSEDE,
        supersedes_review_decision_id=target_prior.review_decision_id,
    )
    superseded = fixture.meaning.create_review_decision(supersede_request)
    assert superseded.supersedes_review_decision_id == target_prior.review_decision_id
    with pytest.raises(MeaningReviewError, match="already superseded"):
        fixture.meaning.create_review_decision(supersede_request)

    with monkeypatch.context() as context:
        context.setattr(
            meaning_persistence,
            "new_id",
            lambda _prefix: superseded.review_decision_id,
        )
        with pytest.raises(MeaningPersistenceError, match="insert conflicted"):
            fixture.meaning.create_review_decision(review_request(statement))


def test_review_replacement_scope_session_integrity_and_orphan_target_edges(
    semantic_fixture: SemanticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = semantic_fixture
    original = fixture.meaning.import_proposal(
        request=fixture.request,
        proposal=fixture.proposal,
    )
    revision_request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version="scope-revision-v1"),
    )
    revision = fixture.meaning.import_proposal(
        request=revision_request,
        proposal=_deep_proposal(fixture.collection_id, statement_suffix=" — revision"),
    )
    target_id = original.run.output.statement_ids[0]
    replacement_id = next(
        item
        for item in revision.run.output.statement_ids
        if item not in original.run.output.statement_ids
    )
    target = fixture.meaning._target_info(MeaningReviewTargetType.STATEMENT, target_id)
    session = fixture.meaning.start_review_session()
    request = MeaningReviewRequest(
        review_session_id=session.review_session_id,
        target_type=MeaningReviewTargetType.STATEMENT,
        target_id=target.target_id,
        target_hash=target.target_hash,
        action=MeaningReviewAction.REVISE,
        reason="replacement scope",
        authority="reviewer",
        scope=MeaningReviewScope((fixture.collection_id,), "research"),
        replacement_target_id=replacement_id,
    )
    original_target_info = SQLiteMeaningRepository._target_info

    def replacement_outside_scope(
        repository: SQLiteMeaningRepository,
        target_type: MeaningReviewTargetType,
        candidate_id: str,
    ) -> object:
        info = original_target_info(repository, target_type, candidate_id)
        if candidate_id == replacement_id:
            return replace(info, collection_ids=("collection_other",))
        return info

    with monkeypatch.context() as context:
        context.setattr(
            SQLiteMeaningRepository,
            "_target_info",
            replacement_outside_scope,
        )
        with pytest.raises(MeaningReviewError, match="outside ReviewScope"):
            fixture.meaning.create_review_decision(request)

    with monkeypatch.context() as context:
        context.setattr(
            meaning_persistence,
            "new_id",
            lambda _prefix: session.review_session_id,
        )
        with pytest.raises(MeaningPersistenceError, match="session insert conflicted"):
            fixture.meaning.start_review_session()

    connection = fixture.repository._store.connection
    connection.execute("DROP TRIGGER meaning_review_sessions_no_update")
    connection.execute(
        "UPDATE meaning_review_sessions SET decision_limit = 999 WHERE review_session_id = ?",
        (session.review_session_id,),
    )
    with pytest.raises(MeaningPersistenceError, match="ReviewBudget is inconsistent"):
        fixture.meaning.get_review_session(session.review_session_id)

    connection.execute(
        """
        INSERT INTO entities(
            entity_id, library_id, kind, label, status, content_hash,
            first_processing_run_id, created_at
        ) VALUES ('entity_orphan', ?, 'person', 'Orphan', 'candidate', ?, ?, ?)
        """,
        (fixture.meaning.library_id, "a" * 64, original.run.processing_run_id, NOW_TEXT),
    )
    with pytest.raises(MeaningPersistenceError, match="no Collection closure"):
        fixture.meaning._target_info(MeaningReviewTargetType.ENTITY, "entity_orphan")


def test_statement_evidence_closure_and_empty_candidate_count_helpers(
    semantic_fixture: SemanticFixture,
) -> None:
    fixture = semantic_fixture
    result = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    connection = fixture.repository._store.connection
    fixture.meaning._verify_statement_evidence_closure(connection, MeaningOutputManifest())
    assert fixture.meaning._new_review_candidate_count(MeaningOutputManifest()) == 0

    voice_id = result.run.output.voice_ids[0]
    connection.execute(
        """
        INSERT INTO statements(
            statement_id, library_id, collection_id, kind, statement_text,
            voice_id, time_context_id, status, lifecycle, content_hash,
            first_processing_run_id, created_at
        ) VALUES (
            'statement_orphan', ?, ?, 'source_claim', 'Orphan', ?, NULL,
            'candidate', 'active', ?, ?, ?
        )
        """,
        (
            fixture.meaning.library_id,
            fixture.collection_id,
            voice_id,
            "a" * 64,
            result.run.processing_run_id,
            NOW_TEXT,
        ),
    )
    with pytest.raises(MeaningPersistenceError, match="lacks an EvidenceLink"):
        fixture.meaning._verify_statement_evidence_closure(
            connection,
            MeaningOutputManifest(statement_ids=("statement_orphan",)),
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("generator_hash", "generator receipt hash"),
        ("review_budget_hash", "ReviewBudget hash"),
        ("output_hash", "output manifest"),
        ("coverage_hash", "CoverageReport hash"),
        ("receipt_hash", "ReadReceipt hash"),
        ("coverage_count", "CoverageReport identity"),
        ("receipt_input", "does not reconstruct input hash"),
        ("missing_coverage", "lacks CoverageReport"),
        ("missing_receipt", "lacks ReadReceipt"),
    ],
)
def test_run_loader_rejects_persisted_closure_corruption(
    semantic_fixture: SemanticFixture,
    corruption: str,
    message: str,
) -> None:
    fixture = semantic_fixture
    result = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    connection = fixture.repository._store.connection
    run_id = result.run.processing_run_id
    if corruption in {"generator_hash", "review_budget_hash", "output_hash"}:
        connection.execute("DROP TRIGGER meaning_processing_runs_no_update")
        connection.execute(
            f'UPDATE meaning_processing_runs SET "{corruption}" = ? WHERE processing_run_id = ?',
            ("a" * 64, run_id),
        )
    elif corruption == "coverage_hash":
        connection.execute("DROP TRIGGER meaning_coverage_reports_no_update")
        connection.execute(
            "UPDATE meaning_coverage_reports SET report_hash = ? WHERE processing_run_id = ?",
            ("a" * 64, run_id),
        )
    elif corruption == "receipt_hash":
        connection.execute("DROP TRIGGER meaning_read_receipts_no_update")
        connection.execute(
            "UPDATE meaning_read_receipts SET receipt_hash = ? WHERE processing_run_id = ?",
            ("a" * 64, run_id),
        )
    elif corruption == "coverage_count":
        connection.execute("DROP TRIGGER meaning_coverage_reports_no_update")
        row = connection.execute(
            """
            SELECT extraction_profile, state, read_fragment_count,
                   candidate_count, omissions_json
            FROM meaning_coverage_reports WHERE processing_run_id = ?
            """,
            (run_id,),
        ).fetchone()
        new_count = int(row[3]) + 1
        payload: dict[str, object] = {
            "schema": "dithyramba.meaning_coverage_report/2.0",
            "processing_run_id": run_id,
            "profile": str(row[0]),
            "state": str(row[1]),
            "read_fragment_count": int(row[2]),
            "candidate_count": new_count,
            "omissions": json.loads(str(row[4])),
        }
        connection.execute(
            """
            UPDATE meaning_coverage_reports
            SET candidate_count = ?, report_hash = ? WHERE processing_run_id = ?
            """,
            (new_count, canonical_sha256_hex(payload), run_id),
        )
    elif corruption == "receipt_input":
        connection.execute("DROP TRIGGER meaning_read_receipts_no_update")
        receipt = connection.execute(
            "SELECT read_receipt_id FROM meaning_read_receipts WHERE processing_run_id = ?",
            (run_id,),
        ).fetchone()
        items = connection.execute(
            """
            SELECT source_fragment_id, source_version_id, source_id,
                   collection_ids_json, read_order, text_sha256
            FROM meaning_read_receipt_items
            WHERE read_receipt_id = ? ORDER BY read_order
            """,
            (str(receipt[0]),),
        ).fetchall()
        new_input_hash = "a" * 64
        payload = {
            "schema": "dithyramba.meaning_read_receipt/2.0",
            "processing_run_id": run_id,
            "input_hash": new_input_hash,
            "items": [
                {
                    "source_fragment_id": str(item[0]),
                    "source_version_id": str(item[1]),
                    "source_id": str(item[2]),
                    "collection_ids": json.loads(str(item[3])),
                    "read_order": int(item[4]),
                    "text_sha256": str(item[5]),
                }
                for item in items
            ],
        }
        connection.execute(
            """
            UPDATE meaning_read_receipts
            SET input_hash = ?, receipt_hash = ? WHERE processing_run_id = ?
            """,
            (new_input_hash, canonical_sha256_hex(payload), run_id),
        )
    elif corruption == "missing_coverage":
        connection.execute("DROP TRIGGER meaning_coverage_reports_no_delete")
        connection.execute(
            "DELETE FROM meaning_coverage_reports WHERE processing_run_id = ?",
            (run_id,),
        )
    else:
        connection.execute("DROP TRIGGER meaning_read_receipt_items_no_delete")
        connection.execute("DROP TRIGGER meaning_read_receipts_no_delete")
        receipt = connection.execute(
            "SELECT read_receipt_id FROM meaning_read_receipts WHERE processing_run_id = ?",
            (run_id,),
        ).fetchone()
        connection.execute(
            "DELETE FROM meaning_read_receipt_items WHERE read_receipt_id = ?",
            (str(receipt[0]),),
        )
        connection.execute(
            "DELETE FROM meaning_read_receipts WHERE processing_run_id = ?",
            (run_id,),
        )
    with pytest.raises(MeaningPersistenceError, match=message):
        fixture.meaning.get_run(run_id)


def test_review_decision_loader_rejects_stale_persisted_hash(
    semantic_fixture: SemanticFixture,
) -> None:
    fixture = semantic_fixture
    fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    target = fixture.meaning.list_review_queue(limit=1)[0]
    session = fixture.meaning.start_review_session()
    decision = fixture.meaning.create_review_decision(
        MeaningReviewRequest(
            review_session_id=session.review_session_id,
            target_type=target.target_type,
            target_id=target.target_id,
            target_hash=target.target_hash,
            action=MeaningReviewAction.DEFER,
            reason="later",
            authority="reviewer",
            scope=MeaningReviewScope((fixture.collection_id,), "research"),
        )
    )
    connection = fixture.repository._store.connection
    connection.execute("DROP TRIGGER meaning_review_decisions_no_update")
    connection.execute(
        "UPDATE meaning_review_decisions SET target_hash = ? WHERE review_decision_id = ?",
        ("a" * 64, decision.review_decision_id),
    )
    with pytest.raises(MeaningPersistenceError, match="hash is stale"):
        fixture.meaning.get_review_decision(decision.review_decision_id)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("scope_hash", "ProcessingRun receipt hash"),
        ("permitted_set_hash", "ProcessingRun receipt hash"),
        ("receipt_hash", "ProcessingRun receipt hash"),
        ("run_identity", "ProcessingRun identity"),
        ("invalid_clock", "canonical UTC"),
        ("reverse_clock", "finishes before"),
    ],
)
def test_processing_run_receipt_identity_and_clocks_fail_closed(
    semantic_fixture: SemanticFixture,
    corruption: str,
    message: str,
) -> None:
    fixture = semantic_fixture
    result = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    connection = fixture.repository._store.connection
    run_id = result.run.processing_run_id
    connection.execute("DROP TRIGGER meaning_processing_runs_no_update")
    if corruption in {"scope_hash", "permitted_set_hash", "receipt_hash"}:
        connection.execute(
            f'UPDATE meaning_processing_runs SET "{corruption}" = ? WHERE processing_run_id = ?',
            ("a" * 64, run_id),
        )
    elif corruption == "run_identity":
        connection.execute(
            "UPDATE meaning_processing_runs SET import_key = ? WHERE processing_run_id = ?",
            ("b" * 64, run_id),
        )
        _rewrite_run_receipt_hash(connection, run_id)
    elif corruption == "invalid_clock":
        connection.execute(
            "UPDATE meaning_processing_runs SET started_at = ? WHERE processing_run_id = ?",
            ("2026-07-21T12:00:00Z", run_id),
        )
        _rewrite_run_receipt_hash(connection, run_id)
    else:
        connection.execute(
            "UPDATE meaning_processing_runs SET started_at = ? WHERE processing_run_id = ?",
            ("2026-07-21T13:00:00.000000Z", run_id),
        )
        _rewrite_run_receipt_hash(connection, run_id)
    with pytest.raises(MeaningPersistenceError, match=message):
        fixture.meaning.get_run(run_id)


@pytest.mark.parametrize(
    ("field", "new_value", "message"),
    [
        ("state", "partial", "run and CoverageReport states"),
        ("candidate_count", None, "CoverageReport candidate count"),
        ("read_fragment_count", None, "CoverageReport read count"),
    ],
)
def test_coverage_coherence_checks_survive_valid_rehashing(
    semantic_fixture: SemanticFixture,
    field: str,
    new_value: object,
    message: str,
) -> None:
    fixture = semantic_fixture
    result = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    connection = fixture.repository._store.connection
    run_id = result.run.processing_run_id
    connection.execute("DROP TRIGGER meaning_coverage_reports_no_update")
    row = connection.execute(
        """
        SELECT extraction_profile, state, read_fragment_count,
               candidate_count, omissions_json
        FROM meaning_coverage_reports WHERE processing_run_id = ?
        """,
        (run_id,),
    ).fetchone()
    values: dict[str, object] = {
        "profile": str(row[0]),
        "state": str(row[1]),
        "read_fragment_count": int(row[2]),
        "candidate_count": int(row[3]),
        "omissions": json.loads(str(row[4])),
    }
    values[field] = new_value if new_value is not None else cast(int, values[field]) + 1
    payload: dict[str, object] = {
        "schema": "dithyramba.meaning_coverage_report/2.0",
        "processing_run_id": run_id,
        **values,
    }
    connection.execute(
        """
        UPDATE meaning_coverage_reports
        SET extraction_profile = ?, state = ?, read_fragment_count = ?,
            candidate_count = ?, omissions_json = ?, report_hash = ?,
            coverage_report_id = ?
        WHERE processing_run_id = ?
        """,
        (
            values["profile"],
            values["state"],
            values["read_fragment_count"],
            values["candidate_count"],
            canonical_json_bytes(values["omissions"]).decode("utf-8"),
            canonical_sha256_hex(payload),
            canonical_content_id("coverage", payload),
            run_id,
        ),
    )
    with pytest.raises(MeaningPersistenceError, match=message):
        fixture.meaning.get_run(run_id)


@pytest.mark.parametrize("corruption", ["input", "identity"])
def test_read_receipt_cross_closure_and_identity_fail_closed(
    semantic_fixture: SemanticFixture,
    corruption: str,
) -> None:
    fixture = semantic_fixture
    result = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    connection = fixture.repository._store.connection
    run_id = result.run.processing_run_id
    receipt = connection.execute(
        "SELECT read_receipt_id, input_hash FROM meaning_read_receipts WHERE processing_run_id = ?",
        (run_id,),
    ).fetchone()
    old_id = str(receipt[0])
    items = connection.execute(
        """
        SELECT source_fragment_id, source_version_id, source_id,
               collection_ids_json, read_order, text_sha256
        FROM meaning_read_receipt_items WHERE read_receipt_id = ? ORDER BY read_order
        """,
        (old_id,),
    ).fetchall()
    new_input_hash = "a" * 64 if corruption == "input" else str(receipt[1])
    payload: dict[str, object] = {
        "schema": "dithyramba.meaning_read_receipt/2.0",
        "processing_run_id": run_id,
        "input_hash": new_input_hash,
        "items": [
            {
                "source_fragment_id": str(item[0]),
                "source_version_id": str(item[1]),
                "source_id": str(item[2]),
                "collection_ids": json.loads(str(item[3])),
                "read_order": int(item[4]),
                "text_sha256": str(item[5]),
            }
            for item in items
        ],
    }
    canonical_id = canonical_content_id("read", payload)
    new_id = "read_tampered" if corruption == "identity" else canonical_id
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("DROP TRIGGER meaning_read_receipts_no_update")
    connection.execute("DROP TRIGGER meaning_read_receipt_items_no_update")
    connection.execute(
        """
        UPDATE meaning_read_receipts
        SET read_receipt_id = ?, input_hash = ?, receipt_hash = ?
        WHERE processing_run_id = ?
        """,
        (new_id, new_input_hash, canonical_sha256_hex(payload), run_id),
    )
    connection.execute(
        "UPDATE meaning_read_receipt_items SET read_receipt_id = ? WHERE read_receipt_id = ?",
        (new_id, old_id),
    )
    connection.execute("PRAGMA foreign_keys = ON")
    expected = "does not reconstruct input hash" if corruption == "input" else "identity"
    with pytest.raises(MeaningPersistenceError, match=expected):
        fixture.meaning.get_run(run_id)


@pytest.mark.parametrize("raced", [True, False])
def test_run_transaction_integrity_race_handling(
    semantic_fixture: SemanticFixture,
    monkeypatch: pytest.MonkeyPatch,
    raced: bool,
) -> None:
    fixture = semantic_fixture
    seed = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    request = replace(
        fixture.request,
        generator=replace(fixture.request.generator, prompt_version=f"race-{raced}"),
    )
    proposal = _deep_proposal(fixture.collection_id, statement_suffix=f" — {raced}")
    calls = 0

    def find_race(_repository: SQLiteMeaningRepository, _import_key: str) -> str | None:
        nonlocal calls
        calls += 1
        if raced and calls >= 3:
            return seed.run.processing_run_id
        return None

    def force_integrity_error(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.IntegrityError("forced race")

    with monkeypatch.context() as context:
        context.setattr(SQLiteMeaningRepository, "_find_run_by_import_key", find_race)
        context.setattr(SQLiteMeaningRepository, "_insert_graph", force_integrity_error)
        if raced:
            result = fixture.meaning.import_proposal(request=request, proposal=proposal)
            assert result.run.processing_run_id == seed.run.processing_run_id
            assert result.deduplicated is True
        else:
            with pytest.raises(MeaningPersistenceError, match="transaction conflicted"):
                fixture.meaning.import_proposal(request=request, proposal=proposal)


def test_scoped_resolution_requires_full_collection_coverage_and_exact_supersede(
    semantic_fixture: SemanticFixture,
    tmp_path: Path,
) -> None:
    fixture = semantic_fixture
    first = fixture.meaning.import_proposal(request=fixture.request, proposal=fixture.proposal)
    second_root = tmp_path / "second"
    second_root.mkdir()
    data_root = fixture.repository.paths.root.parents[1]
    second = fixture.repository.create_collection(
        CollectionConfig(
            library_id=fixture.meaning.library_id,
            name="Second",
            kind=CollectionKind.CORPUS,
            roots=(build_collection_root(second_root, data_root=data_root),),
        )
    )
    second_collection_id = second.config.collection_id
    _insert_source_fixture(
        fixture.repository,
        suffix="second",
        collection_id=second_collection_id,
        text=SECOND_TEXT,
        membership_state="active",
    )
    snapshot = fixture.repository.freeze_snapshot((fixture.collection_id, second_collection_id))
    policy = AccessPolicySnapshot(
        access_policy_id="policy_semantic_multi",
        library_id=fixture.meaning.library_id,
        allowed_purposes=("research",),
        collection_rules=(
            CollectionRule(fixture.collection_id, PolicyEffect.ALLOW),
            CollectionRule(second_collection_id, PolicyEffect.ALLOW),
        ),
    )
    fixture.repository.persist_access_policy(name="Semantic multi", snapshot=policy)
    request = MeaningRunRequest(
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        access_policy_id=policy.access_policy_id,
        scope=RequestScope(
            library_id=fixture.meaning.library_id,
            snapshot_hash=snapshot.manifest_hash,
            purpose="research",
            collection_ids=(fixture.collection_id, second_collection_id),
        ),
        profile=ExtractionProfile.LIGHT,
        generator=GeneratorProfile("fixture", "1", "internal", "multi-v1"),
        review_budget=ReviewBudget(),
        code_version="test",
    )
    second_result = fixture.meaning.import_proposal(
        request=request,
        proposal=_light_proposal(second_collection_id),
    )
    voice_id = next(
        voice_id
        for voice_id in second_result.run.output.voice_ids
        if voice_id in first.run.output.voice_ids
    )
    target = next(
        item for item in fixture.meaning.list_review_queue() if item.target_id == voice_id
    )
    assert set(target.collection_ids) == {fixture.collection_id, second_collection_id}
    session = fixture.meaning.start_review_session(ReviewBudget(review_decision_limit=10))

    def decide(
        collection_id: str,
        action: MeaningReviewAction,
        *,
        prior_id: str | None = None,
    ) -> MeaningReviewRequest:
        return MeaningReviewRequest(
            review_session_id=session.review_session_id,
            target_type=target.target_type,
            target_id=target.target_id,
            target_hash=target.target_hash,
            action=action,
            reason="scoped coverage",
            authority="reviewer",
            scope=MeaningReviewScope((collection_id,), "research"),
            supersedes_review_decision_id=prior_id,
        )

    fixture.meaning.create_review_decision(
        decide(fixture.collection_id, MeaningReviewAction.ACCEPT)
    )
    assert voice_id in {item.target_id for item in fixture.meaning.list_review_queue()}
    with pytest.raises(MeaningReviewError, match="supersede it explicitly"):
        fixture.meaning.create_review_decision(
            decide(fixture.collection_id, MeaningReviewAction.REJECT)
        )

    second_decision = fixture.meaning.create_review_decision(
        decide(second_collection_id, MeaningReviewAction.ACCEPT)
    )
    assert voice_id not in {item.target_id for item in fixture.meaning.list_review_queue()}
    fixture.meaning.create_review_decision(
        decide(
            second_collection_id,
            MeaningReviewAction.SUPERSEDE,
            prior_id=second_decision.review_decision_id,
        )
    )
    assert voice_id in {item.target_id for item in fixture.meaning.list_review_queue()}


def test_atomic_backlog_recheck_serializes_concurrent_writers(
    semantic_fixture: SemanticFixture,
) -> None:
    fixture = semantic_fixture
    data_root = fixture.repository.paths.root.parents[1]
    barrier = Barrier(2)
    requests = tuple(
        replace(
            fixture.request,
            generator=replace(
                fixture.request.generator,
                prompt_version=f"concurrent-{index}",
            ),
            review_budget=ReviewBudget(max_unresolved_candidates=12),
        )
        for index in range(2)
    )
    proposals = tuple(
        _deep_proposal(fixture.collection_id, statement_suffix=f" — concurrent {index}")
        for index in range(2)
    )

    def worker(index: int) -> object:
        with open_library(
            fixture.meaning.library_id,
            data_root=data_root,
            clock=_clock,
        ) as repository:
            meaning = SQLiteMeaningRepository(repository)

            def generator(
                _inputs: tuple[MeaningInputFragment, ...],
                _request: MeaningRunRequest,
            ) -> MeaningProposal:
                barrier.wait(timeout=5)
                return proposals[index]

            return meaning.extract(request=requests[index], generator=generator)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(worker, range(2)))
    typed_results = tuple(cast(MeaningRunResult, item) for item in results)
    assert sorted(result.run.status.value for result in typed_results) == [
        MeaningRunStatus.REFUSED.value,
        MeaningRunStatus.SUCCEEDED.value,
    ]
    refused = next(
        result for result in typed_results if result.run.status is MeaningRunStatus.REFUSED
    )
    assert refused.run.error_code == "review_backlog_limit"
    assert refused.read_receipt.items[0].source_fragment_id == fixture.public_fragment_id
    assert fixture.meaning.unresolved_candidate_count() <= 12


def _authorization_with_extra_manifest_item(authorization: AuthorizedRead) -> AuthorizedRead:
    original = authorization.compiled.manifest.items[0]
    extra = PermittedManifestItem(
        source_fragment_id="fragment_extra",
        source_version_id="source_version_extra",
        source_id="source_extra",
        source_family_id=None,
        collection_ids=original.collection_ids,
    )
    manifest = PermittedManifest(
        library_id=authorization.compiled.manifest.library_id,
        snapshot_hash=authorization.compiled.manifest.snapshot_hash,
        items=(original, extra),
    )
    old_token = authorization.compiled.token
    token = PermittedToken(
        library_id=old_token.library_id,
        snapshot_hash=old_token.snapshot_hash,
        policy_hash=old_token.policy_hash,
        exclusion_hash=old_token.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
    )
    return replace(
        authorization,
        compiled=CompiledAccess(manifest, token, authorization.compiled.public_result),
    )


def _rewrite_run_receipt_hash(connection: sqlite3.Connection, run_id: str) -> None:
    row = connection.execute(
        "SELECT * FROM meaning_processing_runs WHERE processing_run_id = ?",
        (run_id,),
    ).fetchone()
    payload = meaning_persistence._run_receipt_payload(
        processing_run_id=str(row["processing_run_id"]),
        kind=str(row["kind"]),
        corpus_snapshot_id=str(row["corpus_snapshot_id"]),
        access_policy_id=str(row["access_policy_id"]),
        scope_hash=str(row["scope_hash"]),
        permitted_set_hash=str(row["permitted_set_hash"]),
        extraction_profile=str(row["extraction_profile"]),
        code_version=str(row["code_version"]),
        generator_hash=str(row["generator_hash"]),
        review_budget_hash=str(row["review_budget_hash"]),
        input_hash=str(row["input_hash"]),
        proposal_hash=(None if row["proposal_hash"] is None else str(row["proposal_hash"])),
        import_key=str(row["import_key"]),
        status=str(row["status"]),
        candidate_count=int(row["candidate_count"]),
        unresolved_before=int(row["unresolved_before"]),
        backlog_warning=bool(row["backlog_warning"]),
        error_code=None if row["error_code"] is None else str(row["error_code"]),
        output_hash=str(row["output_hash"]),
        started_at=str(row["started_at"]),
        finished_at=str(row["finished_at"]),
    )
    connection.execute(
        "UPDATE meaning_processing_runs SET receipt_hash = ? WHERE processing_run_id = ?",
        (canonical_sha256_hex(payload), run_id),
    )


def _deep_proposal(collection_id: str, *, statement_suffix: str = "") -> MeaningProposal:
    author_quote = "пам'ять є практикою"
    narrator_quote = "пам'ять є монтажем"
    author_start = PUBLIC_TEXT.index(author_quote)
    narrator_start = PUBLIC_TEXT.index(narrator_quote)
    author_label = "Автор"
    author_mention_start = PUBLIC_TEXT.index(author_label)
    concept_label = "пам'ять"
    concept_mention_start = PUBLIC_TEXT.index(concept_label)
    return MeaningProposal(
        voices=(
            VoiceProposal("author", VoiceKind.AUTHOR, "Автор"),
            VoiceProposal("narrator", VoiceKind.NARRATOR, "Оповідач"),
        ),
        entities=(EntityProposal("author_entity", EntityKind.PERSON, "Автор", ("автор",)),),
        entity_mentions=(
            ExactMentionProposal(
                "author_entity",
                collection_id,
                "fragment_public",
                author_mention_start,
                author_mention_start + len(author_label),
                author_label,
                _digest(author_label),
            ),
        ),
        concepts=(ConceptProposal("memory", "пам'ять", ("memory",)),),
        concept_mentions=(
            ExactMentionProposal(
                "memory",
                collection_id,
                "fragment_public",
                concept_mention_start,
                concept_mention_start + len(concept_label),
                concept_label,
                _digest(concept_label),
            ),
        ),
        time_contexts=(TimeContextProposal("time", recorded_at="2026-07-21T12:00:00.000000Z"),),
        statements=(
            StatementProposal(
                "author_statement",
                StatementKind.SOURCE_CLAIM,
                f"Пам'ять є практикою{statement_suffix}",
                "author",
                collection_id,
                (
                    EvidenceLinkProposal(
                        "fragment_public",
                        author_start,
                        author_start + len(author_quote),
                        author_quote,
                        _digest(author_quote),
                        "author",
                    ),
                ),
                "time",
            ),
            StatementProposal(
                "narrator_statement",
                StatementKind.INTERPRETATION,
                f"Пам'ять є монтажем{statement_suffix}",
                "narrator",
                collection_id,
                (
                    EvidenceLinkProposal(
                        "fragment_public",
                        narrator_start,
                        narrator_start + len(narrator_quote),
                        narrator_quote,
                        _digest(narrator_quote),
                        "narrator",
                    ),
                ),
                "time",
            ),
        ),
        concept_meanings=(
            ConceptMeaningProposal(
                "author_meaning",
                "memory",
                "author",
                MEANING_TEXT,
                ("author_statement",),
                "time",
            ),
            ConceptMeaningProposal(
                "narrator_meaning",
                "memory",
                "narrator",
                MEANING_TEXT,
                ("narrator_statement",),
                "time",
            ),
        ),
    )


def _light_proposal(collection_id: str) -> MeaningProposal:
    quote = "пам'ять є повторенням"
    start = SECOND_TEXT.index(quote)
    return MeaningProposal(
        voices=(VoiceProposal("author", VoiceKind.AUTHOR, "Автор"),),
        statements=(
            StatementProposal(
                "second_statement",
                StatementKind.SOURCE_CLAIM,
                "Пам'ять є повторенням",
                "author",
                collection_id,
                (
                    EvidenceLinkProposal(
                        "fragment_second",
                        start,
                        start + len(quote),
                        quote,
                        _digest(quote),
                        "author",
                    ),
                ),
            ),
        ),
    )


def _insert_source_fixture(
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


def _candidate_counts(connection: sqlite3.Connection) -> tuple[int, ...]:
    return tuple(
        int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
        for table in (
            "voices",
            "entities",
            "entity_mentions",
            "concepts",
            "concept_mentions",
            "concept_meanings",
            "time_contexts",
            "statements",
            "evidence_links",
        )
    )


def _semantic_values(connection: sqlite3.Connection) -> str:
    values: list[str] = []
    for table in (
        "meaning_processing_runs",
        "meaning_coverage_reports",
        "meaning_read_receipts",
        "meaning_read_receipt_items",
        "voices",
        "entities",
        "entity_mentions",
        "concepts",
        "concept_mentions",
        "concept_meanings",
        "time_contexts",
        "statements",
        "evidence_links",
    ):
        for row in connection.execute(f'SELECT * FROM "{table}"').fetchall():
            values.extend(str(value) for value in row if value is not None)
    return "\n".join(values)


def _source_texts(repository: LibraryRepository) -> tuple[str, ...]:
    with repository._permit_fragment_text_read():
        rows = repository._store.connection.execute(
            "SELECT text FROM source_fragments ORDER BY source_fragment_id"
        ).fetchall()
    return tuple(str(row[0]) for row in rows)
