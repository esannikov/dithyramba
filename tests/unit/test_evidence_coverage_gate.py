"""Exact-fragment tests for the deterministic EvidenceCoverageGate."""

from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import ValidationError

from dithyramba.evidence import (
    EvidenceAnswerability,
    EvidenceCandidate,
    EvidenceCoverageGate,
    EvidenceGateContractError,
    EvidenceGateDecision,
    EvidenceGateSpec,
    EvidenceRequirement,
    RequirementCoverageStatus,
)


def _address(*, start: int = 10, end: int = 80) -> dict[str, object]:
    return {
        "schema": "dithyramba.source_address/1.0",
        "kind": "markdown",
        "heading_path": ["Evidence"],
        "line_start": 2,
        "line_end": 4,
        "char_start": start,
        "char_end": end,
    }


def _candidate(
    *,
    fragment: str,
    source: str,
    text: str,
    rank: int = 1,
    kind: str = "patent",
    family: str | None = None,
    authority: str = "primary_legal_record",
    group: str | None = None,
    tags: tuple[str, ...] = (),
) -> EvidenceCandidate:
    return EvidenceCandidate(
        rank=rank,
        source_fragment_id=fragment,
        source_id=source,
        text=text,
        source_address=_address(start=rank * 100, end=rank * 100 + len(text)),
        source_kind=kind,
        source_family=f"family_{source}" if family is None else family,
        authority=authority,
        independence_group=source if group is None else group,
        evidence_tags=tags,
    )


def _spec(
    *requirements: EvidenceRequirement,
    answerability: EvidenceAnswerability = EvidenceAnswerability.ANSWERABLE,
    min_groups: int = 1,
    matching_profile: Literal[
        "exact_fragment_unicode_v1", "exact_fragment_unicode_v2"
    ] = "exact_fragment_unicode_v2",
) -> EvidenceGateSpec:
    return EvidenceGateSpec(
        query_key="q_test",
        question="What exact evidence answers this question?",
        expected_answerability=answerability,
        requirements=tuple(requirements),
        min_total_independent_groups=min_groups,
        matching_profile=matching_profile,
    )


def test_exact_source_role_and_anchor_are_required_together() -> None:
    requirement = EvidenceRequirement(
        key="common_return_mechanism",
        label="The patent states the common-return simplification",
        allowed_source_ids=("patent_us390413",),
        source_kinds_any=("patent",),
        anchor_groups=(("single return path or wire in common", "common return-wire"),),
    )
    metadata_only = _candidate(
        fragment="fragment_metadata",
        source="patent_us390413",
        text="US390413A: System of electrical distribution",
    )
    exact = _candidate(
        fragment="fragment_exact",
        source="patent_us390413",
        text="The circuits use a SINGLE RETURN PATH OR WIRE IN COMMON.",
        rank=2,
    )

    failed = EvidenceCoverageGate(_spec(requirement)).evaluate((metadata_only,))
    passed = EvidenceCoverageGate(_spec(requirement)).evaluate((exact, metadata_only))

    assert failed.decision is EvidenceGateDecision.INSUFFICIENT
    assert failed.requirements[0].status is RequirementCoverageStatus.MISSING
    assert failed.requirements[0].closest_candidates[0].rejection_codes == (
        "missing_anchor_group_1",
    )
    assert passed.decision is EvidenceGateDecision.READY
    assert passed.requirements[0].matched_fragment_ids == ("fragment_exact",)


def test_multi_role_question_remains_partial_until_both_roles_are_covered() -> None:
    first_person = EvidenceRequirement(
        key="tesla_retrospective",
        label="Tesla's own technical account",
        source_kinds_any=("first-person technical retrospective",),
        anchor_groups=(("my system of concatenated tuned circuits",),),
    )
    court = EvidenceRequirement(
        key="court_holding",
        label="The court's actual holding",
        source_kinds_any=("court decision",),
        anchor_groups=(("marconi showed no invention over stone",),),
    )
    court_candidate = _candidate(
        fragment="fragment_court",
        source="marconi_1943_supreme_court",
        text="The court concluded that Marconi showed no invention over Stone.",
        kind="court decision",
        family="court_opinion",
        group="scotus_marconi_1943",
    )
    tesla_candidate = _candidate(
        fragment="fragment_tesla",
        source="tesla_1919_true_wireless",
        text="This was my system of concatenated tuned circuits.",
        rank=2,
        kind="first-person technical retrospective",
        family="published_article",
        authority="primary",
        group="tesla_first_person",
    )
    gate = EvidenceCoverageGate(_spec(court, first_person, min_groups=2))

    partial = gate.evaluate((court_candidate,))
    ready = gate.evaluate((tesla_candidate, court_candidate))

    assert partial.decision is EvidenceGateDecision.PARTIAL
    assert len(partial.covered_requirement_ids) == 1
    assert len(partial.missing_requirement_ids) == 1
    assert ready.decision is EvidenceGateDecision.READY
    assert ready.matched_independence_groups == (
        "scotus_marconi_1943",
        "tesla_first_person",
    )


def test_domain_and_role_anchors_must_cooccur_in_one_fragment() -> None:
    requirement = EvidenceRequirement(
        key="art_explainability_limit",
        label="An explainability limit stated inside art research",
        anchor_groups=(
            ("artwork", "painting"),
            ("explainability", "interpretability"),
            ("black box", "limitation"),
        ),
    )
    generic_transformer = _candidate(
        fragment="fragment_generic_transformer",
        source="source_transformer_handbook",
        text="Transformer explainability remains limited by the black box problem.",
    )
    art_only = _candidate(
        fragment="fragment_art_only",
        source="source_art_history",
        text="The artwork is interpreted through its historical context.",
        rank=2,
    )
    scoped = _candidate(
        fragment="fragment_scoped",
        source="source_art_ai_study",
        text=(
            "For painting attribution, interpretability remains a limitation because "
            "the classifier behaves as a black box."
        ),
        rank=3,
    )
    gate = EvidenceCoverageGate(_spec(requirement))

    split = gate.evaluate((generic_transformer, art_only))
    ready = gate.evaluate((generic_transformer, art_only, scoped))

    assert split.decision is EvidenceGateDecision.INSUFFICIENT
    assert split.requirements[0].status is RequirementCoverageStatus.MISSING
    assert ready.decision is EvidenceGateDecision.READY
    assert ready.requirements[0].matched_fragment_ids == ("fragment_scoped",)


def test_duplicate_reprints_do_not_satisfy_independence_requirement() -> None:
    requirement = EvidenceRequirement(
        key="corroboration",
        label="Two independent reports",
        source_kinds_any=("laboratory report",),
        anchor_groups=(("measured energy balance",),),
        min_independent_groups=2,
    )
    first = _candidate(
        fragment="fragment_lab_one",
        source="lab_report_one",
        text="Measured energy balance was neutral.",
        kind="laboratory report",
        family="report",
        authority="independent_lab",
        group="same_report",
    )
    reprint = _candidate(
        fragment="fragment_lab_reprint",
        source="lab_report_reprint",
        text="Measured energy balance was neutral.",
        rank=2,
        kind="laboratory report",
        family="reprint",
        authority="independent_lab",
        group="same_report",
    )
    result = EvidenceCoverageGate(_spec(requirement)).evaluate((first, reprint))

    assert result.decision is EvidenceGateDecision.INSUFFICIENT
    assert result.requirements[0].missing_conditions == ("insufficient_independent_groups",)
    assert result.requirements[0].matched_independence_groups == ("same_report",)


def test_gap_is_preserved_by_thematic_but_wrong_kind_candidate() -> None:
    requirement = EvidenceRequirement(
        key="independent_energy_test",
        label="Independent laboratory report with a measured energy balance",
        source_kinds_any=("independent laboratory report",),
        anchor_groups=(("energy balance", "input and output power"),),
        required_tags=("independent_test",),
    )
    patent = _candidate(
        fragment="fragment_radiant_patent",
        source="patent_us685957",
        text="Apparatus for the utilization of radiant energy.",
        tags=("patent_disclosure",),
    )
    result = EvidenceCoverageGate(
        _spec(requirement, answerability=EvidenceAnswerability.NOT_IN_CORPUS)
    ).evaluate((patent,))

    assert result.decision is EvidenceGateDecision.GAP_PRESERVED
    assert result.requirements[0].closest_candidates[0].rejection_codes == (
        "wrong_source_kind",
        "missing_anchor_group_1",
        "missing_tag_independent_test",
    )


def test_gap_challenge_requires_all_declared_conditions() -> None:
    requirement = EvidenceRequirement(
        key="official_laureate_record",
        label="Official record naming Tesla as a laureate",
        source_kinds_any=("official laureate record",),
        anchor_groups=(("nikola tesla",), ("laureate", "awarded the prize")),
        forbidden_anchors=("did not receive",),
    )
    official = _candidate(
        fragment="fragment_official",
        source="nobel_official_record",
        text="Official laureate: Nikola Tesla was awarded the prize.",
        kind="official laureate record",
        family="official_record",
        authority="issuing_institution",
    )
    result = EvidenceCoverageGate(
        _spec(requirement, answerability=EvidenceAnswerability.NOT_IN_CORPUS)
    ).evaluate((official,))

    assert result.decision is EvidenceGateDecision.GAP_CHALLENGED
    assert result.missing_requirement_ids == ()


def test_forbidden_anchor_blocks_directionally_wrong_fragment() -> None:
    requirement = EvidenceRequirement(
        key="award_outcome",
        label="Positive official award outcome",
        source_kinds_any=("official laureate record",),
        anchor_groups=(("nikola tesla",), ("received the nobel prize",)),
        forbidden_anchors=("did not receive the nobel prize",),
    )
    negative = _candidate(
        fragment="fragment_negative",
        source="nobel_official_record",
        text="Nikola Tesla did not receive the Nobel Prize.",
        kind="official laureate record",
        family="official_record",
        authority="issuing_institution",
    )
    result = EvidenceCoverageGate(_spec(requirement)).evaluate((negative,))

    assert result.decision is EvidenceGateDecision.INSUFFICIENT
    assert "forbidden_anchor_present" in (
        result.requirements[0].closest_candidates[0].rejection_codes
    )


def test_negative_only_requirement_is_preserved_in_v1_but_rejected_in_v2() -> None:
    requirement = EvidenceRequirement(
        key="negative_only",
        label="Legacy absence-only selector",
        forbidden_anchors=("forbidden phrase",),
    )
    candidate = _candidate(
        fragment="fragment_legacy_negative",
        source="source_legacy_negative",
        text="A different phrase is present.",
    )

    legacy = EvidenceCoverageGate(
        _spec(requirement, matching_profile="exact_fragment_unicode_v1")
    ).evaluate((candidate,))

    assert legacy.decision is EvidenceGateDecision.READY
    with pytest.raises((EvidenceGateContractError, ValidationError), match="positive condition"):
        _spec(requirement)


def test_anchor_matching_is_diacritic_insensitive_but_word_bounded() -> None:
    requirement = EvidenceRequirement(
        key="location",
        label="The exact place is present",
        anchor_groups=(("cote",),),
    )
    accented = _candidate(
        fragment="fragment_accented",
        source="source_accented",
        text="The event occurred on the côte.",
    )
    substring_only = _candidate(
        fragment="fragment_substring",
        source="source_substring",
        text="The coteau appears in a different account.",
        rank=2,
    )

    result = EvidenceCoverageGate(_spec(requirement)).evaluate((substring_only, accented))

    assert result.decision is EvidenceGateDecision.READY
    assert result.requirements[0].matched_fragment_ids == ("fragment_accented",)


def test_v1_substring_semantics_are_replayable_while_v2_is_word_bounded() -> None:
    requirement = EvidenceRequirement(
        key="short_anchor",
        label="A short literal anchor",
        anchor_groups=(("art",),),
    )
    candidate = _candidate(
        fragment="fragment_substring_only",
        source="source_substring_only",
        text="A Kantian artist appears in the index.",
    )

    legacy = EvidenceCoverageGate(
        _spec(requirement, matching_profile="exact_fragment_unicode_v1")
    ).evaluate((candidate,))
    current = EvidenceCoverageGate(_spec(requirement)).evaluate((candidate,))

    assert legacy.decision is EvidenceGateDecision.READY
    assert current.decision is EvidenceGateDecision.INSUFFICIENT


def test_v2_does_not_merge_ukrainian_letters_during_anchor_matching() -> None:
    requirement = EvidenceRequirement(
        key="kyiv",
        label="The Ukrainian place name is exact",
        anchor_groups=(("Київ",),),
    )
    misspelled = _candidate(
        fragment="fragment_kyiv_misspelled",
        source="source_kyiv_misspelled",
        text="Архівний запис: Киів.",
    )
    exact = _candidate(
        fragment="fragment_kyiv_exact",
        source="source_kyiv_exact",
        text="Архівний запис: Київ.",
        rank=2,
    )

    result = EvidenceCoverageGate(_spec(requirement)).evaluate((misspelled, exact))

    assert result.decision is EvidenceGateDecision.READY
    assert result.requirements[0].matched_fragment_ids == ("fragment_kyiv_exact",)


def test_conflicting_caller_asserted_lineage_is_rejected() -> None:
    requirement = EvidenceRequirement(
        key="corroboration",
        label="A supported claim",
        anchor_groups=(("supported claim",),),
    )
    first = _candidate(
        fragment="fragment_lineage_one",
        source="source_lineage",
        text="Supported claim.",
        family="family_lineage",
        group="root_one",
    )
    conflicting_source = _candidate(
        fragment="fragment_lineage_two",
        source="source_lineage",
        text="Supported claim.",
        rank=2,
        family="family_lineage",
        group="root_two",
    )
    conflicting_family = _candidate(
        fragment="fragment_lineage_three",
        source="source_other",
        text="Supported claim.",
        rank=3,
        family="family_lineage",
        group="root_two",
    )
    gate = EvidenceCoverageGate(_spec(requirement))

    with pytest.raises(EvidenceGateContractError, match="one source_id"):
        gate.evaluate((first, conflicting_source))
    with pytest.raises(EvidenceGateContractError, match="one source_family"):
        gate.evaluate((first, conflicting_family))

    legacy = EvidenceCoverageGate(
        _spec(requirement, matching_profile="exact_fragment_unicode_v1")
    ).evaluate((first, conflicting_source))
    assert legacy.decision is EvidenceGateDecision.READY


def test_selectors_and_tags_are_checked_and_receipt_is_deterministic() -> None:
    requirement = EvidenceRequirement(
        key="typed_observation",
        label="A typed observation",
        allowed_source_ids=("source_primary",),
        source_kinds_any=("diary",),
        source_families_any=("personal_archive",),
        authorities_any=("primary",),
        required_tags=("in_person",),
    )
    wrong = _candidate(
        fragment="fragment_wrong",
        source="source_secondary",
        text="A later account.",
        kind="article",
        family="scholarship",
        authority="secondary",
    )
    right = _candidate(
        fragment="fragment_right",
        source="source_primary",
        text="I met him in person.",
        rank=2,
        kind="diary",
        family="personal_archive",
        authority="primary",
        tags=("in_person",),
    )
    gate = EvidenceCoverageGate(_spec(requirement))
    first = gate.evaluate((right, wrong))
    second = gate.evaluate((wrong, right))

    assert first.canonical_bytes == second.canonical_bytes
    assert first.result_hash == second.result_hash
    assert first.result_id == second.result_id
    assert json.loads(first.canonical_bytes)["decision"] == "ready"


def test_total_independence_gate_can_block_otherwise_covered_requirements() -> None:
    requirement = EvidenceRequirement(
        key="one_role",
        label="One evidence role",
        source_kinds_any=("diary",),
    )
    candidate = _candidate(
        fragment="fragment_diary",
        source="source_diary",
        text="One diary entry.",
        kind="diary",
        family="archive",
        authority="primary",
    )
    result = EvidenceCoverageGate(_spec(requirement, min_groups=2)).evaluate((candidate,))

    assert result.decision is EvidenceGateDecision.PARTIAL
    assert result.gate_conditions == ("insufficient_total_independent_groups",)
    assert result.missing_requirement_ids == ()


def test_empty_candidate_set_is_a_valid_insufficient_result() -> None:
    requirement = EvidenceRequirement(
        key="required_document",
        label="Required document",
        source_kinds_any=("court decision",),
    )
    result = EvidenceCoverageGate(_spec(requirement)).evaluate(())

    assert result.decision is EvidenceGateDecision.INSUFFICIENT
    assert result.requirements[0].closest_candidates == ()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: EvidenceRequirement(key="Bad-Key", label="x", source_kinds_any=("x",)),
            "requirement key",
        ),
        (
            lambda: EvidenceRequirement(key="valid", label="x"),
            "at least one exact condition",
        ),
        (
            lambda: EvidenceRequirement(
                key="valid", label="x", source_kinds_any=("Patent", "patent")
            ),
            "must be unique",
        ),
        (
            lambda: EvidenceRequirement(key="valid", label="x", anchor_groups=((),)),
            "cannot be empty",
        ),
        (
            lambda: EvidenceGateSpec(
                query_key="bad-key",
                question="x",
                expected_answerability=EvidenceAnswerability.ANSWERABLE,
                requirements=(
                    EvidenceRequirement(key="valid", label="x", source_kinds_any=("patent",)),
                ),
            ),
            "query_key",
        ),
    ],
)
def test_invalid_requirement_and_spec_contracts_fail_closed(factory: object, message: str) -> None:
    with pytest.raises((EvidenceGateContractError, ValidationError), match=message):
        factory()  # type: ignore[operator]


def test_duplicate_requirements_candidates_and_wrong_runtime_types_fail_closed() -> None:
    requirement = EvidenceRequirement(
        key="document",
        label="Document",
        source_kinds_any=("patent",),
    )
    with pytest.raises((EvidenceGateContractError, ValidationError), match="keys must be unique"):
        _spec(requirement, requirement)
    with pytest.raises(EvidenceGateContractError, match="spec must be"):
        EvidenceCoverageGate(object())  # type: ignore[arg-type]

    candidate = _candidate(
        fragment="fragment_same",
        source="source_patent",
        text="Exact patent text.",
    )
    gate = EvidenceCoverageGate(_spec(requirement))
    with pytest.raises(EvidenceGateContractError, match="fragment IDs must be unique"):
        gate.evaluate((candidate, candidate))
    with pytest.raises(EvidenceGateContractError, match="exact EvidenceCandidate"):
        gate.evaluate((object(),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "address",
    [
        "not-an-object",
        {
            "schema": "wrong",
            "kind": "markdown",
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "unknown",
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "markdown",
            "heading_path": "not-an-array",
            "line_start": 1,
            "line_end": 1,
            "char_start": 0,
            "char_end": 5,
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "markdown",
            "heading_path": [],
            "line_start": 1,
            "line_end": 1,
            "char_start": 0,
            "char_end": 5,
            "extra": "forbidden",
        },
    ],
)
def test_invalid_source_addresses_fail_closed(address: object) -> None:
    with pytest.raises((EvidenceGateContractError, ValidationError), match="source_address"):
        EvidenceCandidate(
            rank=1,
            source_fragment_id="fragment_one",
            source_id="source_one",
            text="text",
            source_address=address,  # type: ignore[arg-type]
            source_kind="patent",
            source_family="patent",
            authority="primary",
            independence_group="one",
        )


def test_pdf_source_address_and_canonical_requirement_identity() -> None:
    candidate = EvidenceCandidate(
        rank=1,
        source_fragment_id="fragment_pdf",
        source_id="source_pdf",
        text="page text",
        source_address={
            "schema": "dithyramba.source_address/1.0",
            "kind": "pdf",
            "page": 2,
            "bbox": ["0.000", "0.000", "10.000", "20.000"],
            "char_start": 0,
            "char_end": 9,
        },
        source_kind="court decision",
        source_family="court",
        authority="primary",
        independence_group="court_one",
    )
    requirement = EvidenceRequirement(
        key="court_record",
        label="Court record",
        source_kinds_any=("court decision",),
    )

    result = EvidenceCoverageGate(_spec(requirement)).evaluate((candidate,))

    assert result.decision is EvidenceGateDecision.READY
    assert requirement.evidence_requirement_id.startswith("evidence_requirement_")
    assert len(requirement.requirement_hash) == 64
