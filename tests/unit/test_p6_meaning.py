from __future__ import annotations

import pytest

from dithyramba.contracts import sha256_hex
from dithyramba.meaning import (
    ConceptMeaningProposal,
    ConceptProposal,
    CoverageState,
    EntityKind,
    EntityProposal,
    EvidenceLinkProposal,
    ExactMentionProposal,
    ExtractionProfile,
    MeaningBudgetError,
    MeaningContractError,
    MeaningEvidenceError,
    MeaningInputFragment,
    MeaningOmission,
    MeaningProposal,
    OmissionCategory,
    ReviewBudget,
    StatementKind,
    StatementProposal,
    TimeContextProposal,
    VoiceKind,
    VoiceProposal,
)
from dithyramba.meaning.validation import prepare_meaning_graph

LIBRARY_ID = "library_semantic"
COLLECTION_ID = "collection_public"


def _digest(value: str) -> str:
    return sha256_hex(value.encode("utf-8"))


def _fragment(
    text: str,
    *,
    suffix: str = "public",
    source_version_suffix: str | None = None,
) -> MeaningInputFragment:
    version_suffix = source_version_suffix or suffix
    return MeaningInputFragment(
        source_fragment_id=f"fragment_{suffix}",
        source_version_id=f"source_version_{version_suffix}",
        source_id=f"source_{version_suffix}",
        collection_ids=(COLLECTION_ID,),
        text=text,
        text_sha256=_digest(text),
    )


def _two_voice_proposal(text: str) -> MeaningProposal:
    first_quote = "пам'ять"
    first_start = text.index(first_quote)
    second_start = text.index(first_quote, first_start + 1)
    return MeaningProposal(
        voices=(
            VoiceProposal("author", VoiceKind.AUTHOR, "Автор"),
            VoiceProposal("narrator", VoiceKind.NARRATOR, "Оповідач"),
        ),
        concepts=(ConceptProposal("memory", "пам'ять"),),
        statements=(
            StatementProposal(
                key="author_statement",
                kind=StatementKind.SOURCE_CLAIM,
                statement_text="Автор говорить про пам'ять.",
                voice_key="author",
                collection_id=COLLECTION_ID,
                evidence=(
                    EvidenceLinkProposal(
                        source_fragment_id="fragment_public",
                        quote_start=first_start,
                        quote_end=first_start + len(first_quote),
                        quote_text=first_quote,
                        quote_sha256=_digest(first_quote),
                        attributed_voice_key="author",
                    ),
                ),
            ),
            StatementProposal(
                key="narrator_statement",
                kind=StatementKind.INTERPRETATION,
                statement_text="Оповідач також говорить про пам'ять.",
                voice_key="narrator",
                collection_id=COLLECTION_ID,
                evidence=(
                    EvidenceLinkProposal(
                        source_fragment_id="fragment_public",
                        quote_start=second_start,
                        quote_end=second_start + len(first_quote),
                        quote_text=first_quote,
                        quote_sha256=_digest(first_quote),
                        attributed_voice_key="narrator",
                    ),
                ),
            ),
        ),
        concept_meanings=(
            ConceptMeaningProposal(
                "author_meaning",
                "memory",
                "author",
                "практика",
                ("author_statement",),
            ),
            ConceptMeaningProposal(
                "narrator_meaning",
                "memory",
                "narrator",
                "практика",
                ("narrator_statement",),
            ),
        ),
    )


def test_deep_graph_keeps_same_label_meanings_distinct_by_voice_and_exact_evidence() -> None:
    text = "Автор: пам'ять. Оповідач: пам'ять."
    graph = prepare_meaning_graph(
        library_id=LIBRARY_ID,
        profile=ExtractionProfile.DEEP,
        review_budget=ReviewBudget(),
        proposal=_two_voice_proposal(text),
        fragments=(_fragment(text),),
    )

    assert len(graph.concept_meanings) == 2
    assert len({item.concept_meaning_id for item in graph.concept_meanings}) == 2
    assert len({item.voice_id for item in graph.concept_meanings}) == 2
    statements = {item.statement_id: item for item in graph.statements}
    assert all(
        statements[evidence.statement_id].voice_id == evidence.attributed_voice_id
        for evidence in graph.evidence_links
    )
    assert graph.output.candidate_count == 9


def test_exact_quote_access_collection_and_attribution_fail_closed() -> None:
    text = "Автор: пам'ять. Оповідач: пам'ять."
    proposal = _two_voice_proposal(text)
    first = proposal.statements[0]
    evidence = first.evidence[0]
    bad_quote = EvidenceLinkProposal(
        source_fragment_id=evidence.source_fragment_id,
        quote_start=evidence.quote_start + 1,
        quote_end=evidence.quote_end + 1,
        quote_text=evidence.quote_text,
        quote_sha256=evidence.quote_sha256,
        attributed_voice_key=evidence.attributed_voice_key,
    )
    mismatched = MeaningProposal(
        voices=proposal.voices,
        concepts=proposal.concepts,
        statements=(
            StatementProposal(
                first.key,
                first.kind,
                first.statement_text,
                first.voice_key,
                first.collection_id,
                (bad_quote,),
            ),
        ),
        concept_meanings=(
            ConceptMeaningProposal(
                "meaning",
                "memory",
                first.voice_key,
                "практика",
                (first.key,),
            ),
        ),
    )
    with pytest.raises(MeaningEvidenceError, match="offsets"):
        prepare_meaning_graph(
            library_id=LIBRARY_ID,
            profile=ExtractionProfile.DEEP,
            review_budget=ReviewBudget(),
            proposal=mismatched,
            fragments=(_fragment(text),),
        )

    wrong_voice = EvidenceLinkProposal(
        source_fragment_id=evidence.source_fragment_id,
        quote_start=evidence.quote_start,
        quote_end=evidence.quote_end,
        quote_text=evidence.quote_text,
        quote_sha256=evidence.quote_sha256,
        attributed_voice_key="narrator",
    )
    attributed = MeaningProposal(
        voices=proposal.voices,
        concepts=proposal.concepts,
        statements=(
            StatementProposal(
                first.key,
                first.kind,
                first.statement_text,
                first.voice_key,
                first.collection_id,
                (wrong_voice,),
            ),
            proposal.statements[1],
        ),
        concept_meanings=proposal.concept_meanings,
    )
    with pytest.raises(MeaningEvidenceError, match="retain"):
        prepare_meaning_graph(
            library_id=LIBRARY_ID,
            profile=ExtractionProfile.DEEP,
            review_budget=ReviewBudget(),
            proposal=attributed,
            fragments=(_fragment(text),),
        )


def test_mentions_count_toward_source_and_run_candidate_caps() -> None:
    source_text = "x"
    entities = tuple(
        EntityProposal(f"entity-{index}", EntityKind.OBJECT, f"Object {index}")
        for index in range(7)
    )
    mentions = tuple(
        ExactMentionProposal(
            target_key=entity.key,
            collection_id=COLLECTION_ID,
            source_fragment_id="fragment_public",
            quote_start=0,
            quote_end=1,
            quote_text=source_text,
            quote_sha256=_digest(source_text),
        )
        for entity in entities
    )
    with pytest.raises(MeaningBudgetError, match="per-SourceVersion"):
        prepare_meaning_graph(
            library_id=LIBRARY_ID,
            profile=ExtractionProfile.INDEX,
            review_budget=ReviewBudget(),
            proposal=MeaningProposal(
                entities=entities,
                entity_mentions=mentions,
            ),
            fragments=(_fragment(source_text),),
        )

    many_concepts = tuple(ConceptProposal(f"concept-{index}", f"C{index}") for index in range(201))
    many_mentions = tuple(
        ExactMentionProposal(
            target_key=concept.key,
            collection_id=COLLECTION_ID,
            source_fragment_id=f"fragment_{index}",
            quote_start=0,
            quote_end=1,
            quote_text=source_text,
            quote_sha256=_digest(source_text),
        )
        for index, concept in enumerate(many_concepts)
    )
    fragments = tuple(
        _fragment(source_text, suffix=str(index), source_version_suffix=str(index))
        for index in range(201)
    )
    with pytest.raises(MeaningBudgetError, match="run candidate"):
        prepare_meaning_graph(
            library_id=LIBRARY_ID,
            profile=ExtractionProfile.INDEX,
            review_budget=ReviewBudget(),
            proposal=MeaningProposal(
                concepts=many_concepts,
                concept_mentions=many_mentions,
            ),
            fragments=fragments,
        )


def test_time_context_alias_and_noncomplete_proposal_boundaries() -> None:
    valid = TimeContextProposal(
        "time",
        event_valid_start="2026-07-21T12:00:00.000000Z",
        event_valid_end="2026-07-21T13:00:00.000000Z",
    )
    assert valid.event_valid_start is not None
    with pytest.raises(MeaningContractError, match="canonical UTC"):
        TimeContextProposal("bad", recorded_at="2026-07-21")
    with pytest.raises(MeaningContractError, match="reversed"):
        TimeContextProposal(
            "bad",
            event_valid_start="2026-07-21T13:00:00.000000Z",
            event_valid_end="2026-07-21T12:00:00.000000Z",
        )
    with pytest.raises(MeaningContractError, match="at most 32"):
        ConceptProposal("aliases", "Label", tuple(f"Alias {index}" for index in range(33)))

    partial = MeaningProposal(
        state=CoverageState.PARTIAL,
        omissions=(MeaningOmission(OmissionCategory.FAILURE, "parser_partial", 1),),
        failure_code="parser_partial",
    )
    assert partial.statements == ()
    with pytest.raises(MeaningContractError, match="cannot create"):
        MeaningProposal(
            state=CoverageState.PARTIAL,
            voices=(VoiceProposal("author", VoiceKind.AUTHOR, "Author"),),
            omissions=(MeaningOmission(OmissionCategory.FAILURE, "parser_partial", 1),),
            failure_code="parser_partial",
        )
