from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from dithyramba.access import RequestScope
from dithyramba.contracts import sha256_hex
from dithyramba.meaning import (
    ConceptMeaningProposal,
    ConceptProposal,
    CoverageState,
    EntityKind,
    EntityProposal,
    EvidenceLinkProposal,
    EvidencePolarity,
    ExactMentionProposal,
    ExtractionProfile,
    GeneratorProfile,
    MeaningContractError,
    MeaningEvidenceError,
    MeaningInputFragment,
    MeaningOmission,
    MeaningOutputManifest,
    MeaningProfileError,
    MeaningProposal,
    MeaningReviewAction,
    MeaningReviewRequest,
    MeaningReviewScope,
    MeaningReviewTargetType,
    MeaningRunRequest,
    OmissionCategory,
    ReviewBudget,
    SemanticAlignment,
    StatementKind,
    StatementProposal,
    TimeContextProposal,
    VoiceKind,
    VoiceProposal,
)
from dithyramba.meaning.validation import prepare_meaning_graph

HASH = "a" * 64
TEXT = "alpha beta"


def _digest(value: str) -> str:
    return sha256_hex(value.encode("utf-8"))


def _input(*, collection_id: str = "collection_one") -> MeaningInputFragment:
    return MeaningInputFragment(
        "fragment_one",
        "source_version_one",
        "source_one",
        (collection_id,),
        TEXT,
        _digest(TEXT),
    )


def _evidence(
    *, voice: str = "voice", collection_fragment: str = "fragment_one"
) -> EvidenceLinkProposal:
    return EvidenceLinkProposal(
        collection_fragment,
        0,
        5,
        "alpha",
        _digest("alpha"),
        voice,
        limits_text="bounded",
    )


def _statement(*, voice: str = "voice", time: str | None = None) -> StatementProposal:
    return StatementProposal(
        "statement",
        StatementKind.SOURCE_CLAIM,
        "Alpha",
        voice,
        "collection_one",
        (_evidence(voice=voice),),
        time,
    )


def _request(**changes: object) -> MeaningRunRequest:
    values: dict[str, object] = {
        "corpus_snapshot_id": "snapshot_one",
        "access_policy_id": "policy_one",
        "scope": RequestScope("library_one", HASH, "research", ("collection_one",)),
        "profile": ExtractionProfile.DEEP,
        "generator": GeneratorProfile("local", "1", "internal", "prompt", dimensions=8),
        "review_budget": ReviewBudget(),
        "code_version": "test",
    }
    values.update(changes)
    return MeaningRunRequest(
        corpus_snapshot_id=cast(str, values["corpus_snapshot_id"]),
        access_policy_id=cast(str, values["access_policy_id"]),
        scope=cast(RequestScope, values["scope"]),
        profile=cast(ExtractionProfile, values["profile"]),
        generator=cast(GeneratorProfile, values["generator"]),
        review_budget=cast(ReviewBudget, values["review_budget"]),
        code_version=cast(str, values["code_version"]),
    )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ReviewBudget(max_candidates=0), "max_candidates"),
        (lambda: ReviewBudget(max_read_fragments=2_001), "max_read_fragments"),
        (lambda: ReviewBudget(max_unresolved_candidates=0), "max_unresolved"),
        (lambda: ReviewBudget(review_decision_limit=0), "review_decision"),
        (
            lambda: GeneratorProfile(
                "local",
                "1",
                "internal",
                "prompt",
                external_provider=cast(bool, 1),
            ),
            "external_provider",
        ),
        (lambda: _request(scope=object()), "scope"),
        (lambda: _request(profile="deep"), "profile"),
        (lambda: _request(generator=object()), "generator"),
        (lambda: _request(review_budget=object()), "review_budget"),
        (
            lambda: MeaningInputFragment(
                "fragment_one",
                "source_version_one",
                "source_one",
                (),
                TEXT,
                _digest(TEXT),
            ),
            "requires a Collection",
        ),
        (
            lambda: MeaningInputFragment(
                "fragment_one",
                "source_version_one",
                "source_one",
                ("collection_one",),
                TEXT,
                HASH,
            ),
            "text hash",
        ),
        (lambda: VoiceProposal("voice", cast(VoiceKind, "author"), "A"), "Voice kind"),
        (
            lambda: EntityProposal("entity", cast(EntityKind, "person"), "A"),
            "Entity kind",
        ),
        (lambda: TimeContextProposal("time"), "at least one clock"),
        (
            lambda: TimeContextProposal("time", recorded_at="2026-02-30T00:00:00.000000Z"),
            "real UTC",
        ),
        (
            lambda: EvidenceLinkProposal(
                "fragment_one",
                0,
                5,
                "alpha",
                _digest("alpha"),
                "voice",
                polarity=cast(EvidencePolarity, "supports"),
            ),
            "polarity",
        ),
        (
            lambda: EvidenceLinkProposal(
                "fragment_one",
                0,
                5,
                "alpha",
                _digest("alpha"),
                "voice",
                alignment=cast(SemanticAlignment, "exact"),
            ),
            "alignment",
        ),
        (
            lambda: StatementProposal(
                "statement",
                cast(StatementKind, "source_claim"),
                "Alpha",
                "voice",
                "collection_one",
                (_evidence(),),
            ),
            "Statement kind",
        ),
        (
            lambda: StatementProposal(
                "statement",
                StatementKind.SOURCE_CLAIM,
                "Alpha",
                "voice",
                "collection_one",
                (),
            ),
            "requires typed",
        ),
        (
            lambda: StatementProposal(
                "statement",
                StatementKind.SOURCE_CLAIM,
                "Alpha",
                "voice",
                "collection_one",
                (_evidence(), _evidence()),
            ),
            "must be unique",
        ),
        (
            lambda: ConceptMeaningProposal("meaning", "concept", "voice", "Meaning", ()),
            "source-grounded",
        ),
        (
            lambda: MeaningOmission(cast(OmissionCategory, "failure"), "failed", 1),
            "OmissionCategory",
        ),
        (lambda: MeaningProposal(state=cast(CoverageState, "complete")), "CoverageState"),
        (
            lambda: MeaningProposal(voices=cast(tuple[VoiceProposal, ...], [])),
            "typed tuples",
        ),
        (
            lambda: MeaningProposal(failure_code="failed"),
            "complete proposal",
        ),
        (
            lambda: MeaningProposal(state=CoverageState.FAILED),
            "requires failure_code",
        ),
        (
            lambda: MeaningProposal(
                voices=(
                    VoiceProposal("duplicate", VoiceKind.AUTHOR, "A"),
                    VoiceProposal("duplicate", VoiceKind.NARRATOR, "B"),
                )
            ),
            "keys must be unique",
        ),
        (lambda: MeaningReviewScope((), "research"), "requires at least"),
        (lambda: VoiceProposal("Bad Key", VoiceKind.AUTHOR, "A"), "semantic key"),
        (lambda: VoiceProposal("voice", VoiceKind.AUTHOR, " "), "non-empty"),
        (
            lambda: ExactMentionProposal(
                "entity",
                "collection_one",
                "fragment_one",
                1,
                1,
                "a",
                _digest("a"),
            ),
            "offsets",
        ),
        (
            lambda: ExactMentionProposal(
                "entity",
                "collection_one",
                "fragment_one",
                0,
                1,
                "a",
                HASH,
            ),
            "quote hash",
        ),
        (
            lambda: ConceptProposal("concept", "C", cast(tuple[str, ...], ["alias"])),
            "must be a tuple",
        ),
        (
            lambda: ConceptProposal("concept", "C", ("alias", "alias")),
            "must be unique",
        ),
        (
            lambda: ConceptMeaningProposal(
                "meaning",
                "concept",
                "voice",
                "Meaning",
                cast(tuple[str, ...], ["statement"]),
            ),
            "must be a tuple",
        ),
        (
            lambda: ConceptMeaningProposal(
                "meaning",
                "concept",
                "voice",
                "Meaning",
                ("statement", "statement"),
            ),
            "must be unique",
        ),
        (
            lambda: MeaningOutputManifest(voice_ids=cast(tuple[str, ...], ["voice_one"])),
            "must be a tuple",
        ),
        (
            lambda: MeaningOutputManifest(voice_ids=("voice_one", "voice_one")),
            "must be unique",
        ),
        (
            lambda: MeaningOutputManifest(voice_ids=("bad",)),
            "voice_ identifier",
        ),
    ],
)
def test_model_contract_rejects_invalid_boundaries(factory: object, match: str) -> None:
    callable_factory = cast(Callable[[], object], factory)
    with pytest.raises(MeaningContractError, match=match):
        callable_factory()


def test_budget_profiles_and_request_payload_positive_branches() -> None:
    assert ReviewBudget(max_unresolved_candidates=1_001).has_explicit_backlog_override is True
    assert ReviewBudget().has_explicit_backlog_override is False
    assert ExtractionProfile.LIGHT.source_version_candidate_cap == 24
    request = _request()
    assert request.payload()["profile"] == "deep"
    assert request.generator.dimensions == 8
    assert _input().receipt_payload()["text_sha256"] == _digest(TEXT)


def test_review_request_action_boundaries() -> None:
    scope = MeaningReviewScope(("collection_one",), "research")
    base = MeaningReviewRequest(
        review_session_id="review_session_one",
        target_type=MeaningReviewTargetType.STATEMENT,
        target_id="statement_one",
        target_hash=HASH,
        action=MeaningReviewAction.ACCEPT,
        reason="reason",
        authority="reviewer",
        scope=scope,
    )
    with pytest.raises(MeaningContractError, match="target_type"):
        replace(base, target_type=cast(MeaningReviewTargetType, "statement"))
    with pytest.raises(MeaningContractError, match="action"):
        replace(base, action=cast(MeaningReviewAction, "accept"))
    with pytest.raises(MeaningContractError, match="scope"):
        replace(base, scope=cast(MeaningReviewScope, object()))
    with pytest.raises(MeaningContractError, match="replacement"):
        replace(base, action=MeaningReviewAction.REVISE)
    with pytest.raises(MeaningContractError, match="only revise"):
        replace(base, replacement_target_id="statement_two")
    with pytest.raises(MeaningContractError, match="prior"):
        replace(base, action=MeaningReviewAction.SUPERSEDE)
    with pytest.raises(MeaningContractError, match="only supersede"):
        replace(base, supersedes_review_decision_id="review_one")
    revised = replace(
        base,
        action=MeaningReviewAction.REVISE,
        replacement_target_id="statement_two",
    )
    superseded = replace(
        base,
        action=MeaningReviewAction.SUPERSEDE,
        supersedes_review_decision_id="review_one",
    )
    assert revised.replacement_target_id == "statement_two"
    assert superseded.supersedes_review_decision_id == "review_one"


def test_validator_profile_reference_grounding_collection_and_hash_edges() -> None:
    fragment = _input()
    statement = _statement()
    base = MeaningProposal(
        voices=(VoiceProposal("voice", VoiceKind.AUTHOR, "A"),),
        statements=(statement,),
    )
    with pytest.raises(MeaningProfileError, match="index"):
        prepare_meaning_graph(
            library_id="library_one",
            profile=ExtractionProfile.INDEX,
            review_budget=ReviewBudget(),
            proposal=base,
            fragments=(fragment,),
        )
    deep = MeaningProposal(
        voices=base.voices,
        concepts=(ConceptProposal("concept", "C"),),
        statements=base.statements,
        concept_meanings=(
            ConceptMeaningProposal("meaning", "concept", "voice", "M", ("statement",)),
        ),
    )
    with pytest.raises(MeaningProfileError, match="light"):
        prepare_meaning_graph(
            library_id="library_one",
            profile=ExtractionProfile.LIGHT,
            review_budget=ReviewBudget(),
            proposal=deep,
            fragments=(fragment,),
        )
    light_with_limits = MeaningProposal(
        voices=base.voices,
        statements=base.statements,
    )
    with pytest.raises(MeaningProfileError, match="limits"):
        prepare_meaning_graph(
            library_id="library_one",
            profile=ExtractionProfile.LIGHT,
            review_budget=ReviewBudget(),
            proposal=light_with_limits,
            fragments=(fragment,),
        )
    light_statement = replace(
        statement,
        evidence=(replace(statement.evidence[0], limits_text=None),),
    )
    light_graph = prepare_meaning_graph(
        library_id="library_one",
        profile=ExtractionProfile.LIGHT,
        review_budget=ReviewBudget(),
        proposal=MeaningProposal(voices=base.voices, statements=(light_statement,)),
        fragments=(fragment,),
    )
    assert light_graph.evidence_links[0].limits_text is None
    with pytest.raises(MeaningProfileError, match="typed"):
        prepare_meaning_graph(
            library_id="library_one",
            profile=cast(ExtractionProfile, "deep"),
            review_budget=ReviewBudget(),
            proposal=MeaningProposal(),
            fragments=(),
        )
    with pytest.raises(MeaningEvidenceError, match="duplicate"):
        prepare_meaning_graph(
            library_id="library_one",
            profile=ExtractionProfile.DEEP,
            review_budget=ReviewBudget(),
            proposal=base,
            fragments=(fragment, fragment),
        )

    orphan_cases = (
        MeaningProposal(voices=(VoiceProposal("voice", VoiceKind.AUTHOR, "A"),)),
        MeaningProposal(entities=(EntityProposal("entity", EntityKind.PERSON, "E"),)),
        MeaningProposal(concepts=(ConceptProposal("concept", "C"),)),
        MeaningProposal(
            time_contexts=(TimeContextProposal("time", recorded_at="2026-07-21T00:00:00.000000Z"),)
        ),
    )
    for proposal in orphan_cases:
        with pytest.raises(MeaningEvidenceError, match="every"):
            prepare_meaning_graph(
                library_id="library_one",
                profile=ExtractionProfile.DEEP,
                review_budget=ReviewBudget(),
                proposal=proposal,
                fragments=(fragment,),
            )

    unknown_voice = MeaningProposal(statements=(_statement(voice="missing"),))
    with pytest.raises(MeaningEvidenceError, match="unknown Voice"):
        prepare_meaning_graph(
            library_id="library_one",
            profile=ExtractionProfile.DEEP,
            review_budget=ReviewBudget(),
            proposal=unknown_voice,
            fragments=(fragment,),
        )
    unknown_mention = MeaningProposal(
        entity_mentions=(
            ExactMentionProposal(
                "missing", "collection_one", "fragment_one", 0, 5, "alpha", _digest("alpha")
            ),
        )
    )
    with pytest.raises(MeaningEvidenceError, match="unknown Entity"):
        prepare_meaning_graph(
            library_id="library_one",
            profile=ExtractionProfile.INDEX,
            review_budget=ReviewBudget(),
            proposal=unknown_mention,
            fragments=(fragment,),
        )

    other_collection = _input(collection_id="collection_other")
    with pytest.raises(MeaningEvidenceError, match="declared Collection"):
        prepare_meaning_graph(
            library_id="library_one",
            profile=ExtractionProfile.DEEP,
            review_budget=ReviewBudget(),
            proposal=base,
            fragments=(other_collection,),
        )
    mutated = _evidence()
    object.__setattr__(mutated, "quote_sha256", HASH)
    bad_hash = MeaningProposal(
        voices=base.voices,
        statements=(replace(statement, evidence=(mutated,)),),
    )
    with pytest.raises(MeaningEvidenceError, match="SHA-256"):
        prepare_meaning_graph(
            library_id="library_one",
            profile=ExtractionProfile.DEEP,
            review_budget=ReviewBudget(),
            proposal=bad_hash,
            fragments=(fragment,),
        )
