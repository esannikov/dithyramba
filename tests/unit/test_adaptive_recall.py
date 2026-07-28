"""Bounded adaptive recall, QueryCloud, and q0-anchored Harrier tests."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from random import Random
from typing import Any, cast

import pytest
from pydantic import ValidationError

import dithyramba.recall.adaptive as adaptive
from dithyramba.contracts import sha256_hex
from dithyramba.evidence import (
    EvidenceAnswerability,
    EvidenceCandidate,
    EvidenceCoverageGate,
    EvidenceGateDecision,
    EvidenceGateSpec,
    EvidenceRequirement,
)
from dithyramba.recall.adaptive import (
    AdaptiveFragment,
    AdaptiveRecallError,
    AdaptiveRecallResult,
    AdaptiveRetrievalConfig,
    AliasEntry,
    AliasRegistry,
    CandidateOrigin,
    CandidateOriginKind,
    CandidateSelectionReason,
    CandidateText,
    DerivedQuery,
    HarrierScoreBatch,
    HarrierScoreItem,
    QueryRepairKind,
    QueryRepairPlan,
    SemanticHarrierCandidateScorer,
    execute_adaptive_retrieval,
)
from dithyramba.recall.hybrid_models import ModelRuntimeProfile
from dithyramba.recall.provisioning import (
    HARRIER_OSS_V1_270M_PROFILE,
    MULTILINGUAL_E5_SMALL_PROFILE,
)
from dithyramba.recall.vector import PackedVector, pack_normalized_vector

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
QUESTION = "Which exact record proves the decisive anchor?"


def _address(*, ordinal: int = 0) -> dict[str, object]:
    return {
        "schema": "dithyramba.source_address/1.0",
        "kind": "markdown",
        "heading_path": ["Evidence"],
        "line_start": ordinal + 1,
        "line_end": ordinal + 1,
        "char_start": ordinal * 100,
        "char_end": ordinal * 100 + 80,
    }


def _fragment(
    fragment_id: str,
    text: str,
    *,
    source: str | None = None,
    version: str | None = None,
    ordinal: int = 0,
    family: str = "archive",
    kind: str = "archival record",
    authority: str = "primary",
    group: str | None = None,
    tags: tuple[str, ...] = (),
    proof: bool = True,
) -> AdaptiveFragment:
    suffix = fragment_id.removeprefix("fragment_")
    source_id = source or f"source_{suffix}"
    return AdaptiveFragment(
        source_fragment_id=fragment_id,
        source_version_id=version or f"version_{suffix}",
        source_id=source_id,
        ordinal=ordinal,
        fragment_kind="paragraph",
        text=text,
        text_sha256=sha256_hex(text.encode("utf-8")),
        source_address=_address(ordinal=ordinal),
        source_kind=kind,
        source_family=family,
        authority=authority,
        independence_group=group or source_id,
        evidence_tags=tags,
        body_proof_eligible=proof,
    )


def _requirement(
    *,
    key: str = "decisive_record",
    anchor: str = "decisive anchor",
    kind: str = "archival record",
) -> EvidenceRequirement:
    return EvidenceRequirement(
        key=key,
        label=f"Exact fragment containing {anchor}",
        source_kinds_any=(kind,),
        anchor_groups=((anchor,),),
    )


def _spec(
    question: str = QUESTION,
    *requirements: EvidenceRequirement,
    answerability: EvidenceAnswerability = EvidenceAnswerability.ANSWERABLE,
) -> EvidenceGateSpec:
    return EvidenceGateSpec(
        query_key="adaptive_test",
        question=question,
        expected_answerability=answerability,
        requirements=requirements or (_requirement(),),
        min_total_independent_groups=1,
    )


@dataclass
class _StaticHarrier:
    priority: tuple[str, ...] = ()
    q0_override: str | None = None
    omit_last: bool = False
    return_object: bool = False
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def score(
        self,
        *,
        q0: str,
        candidates: tuple[CandidateText, ...],
    ) -> HarrierScoreBatch:
        self.calls.append((q0, tuple(item.source_fragment_id for item in candidates)))
        if self.return_object:
            return cast(HarrierScoreBatch, object())
        priority = {identifier: rank for rank, identifier in enumerate(self.priority)}
        ordered = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    priority.get(item.source_fragment_id, len(priority)),
                    item.source_fragment_id.encode("ascii"),
                ),
            )
        )
        if self.omit_last:
            ordered = ordered[:-1]
        return HarrierScoreBatch(
            original_question_sha256=sha256_hex((self.q0_override or q0).encode("utf-8")),
            model_profile_hash=HASH_A,
            runtime_profile_hash=HASH_B,
            provisioning_receipt_hash=HASH_C,
            items=tuple(
                HarrierScoreItem(
                    source_fragment_id=item.source_fragment_id,
                    text_sha256=item.text_sha256,
                    rank=rank,
                    score_hex=max(-1.0, 1.0 - rank / 100.0).hex(),
                    vector_sha256=sha256_hex(f"vector:{item.source_fragment_id}".encode()),
                )
                for rank, item in enumerate(ordered, start=1)
            ),
        )


@dataclass
class _PlanProvider:
    plan_value: object
    calls: list[tuple[str, tuple[str, ...], str]] = field(default_factory=list)

    def plan(
        self,
        *,
        question: str,
        missing_requirement_keys: tuple[str, ...],
        alias_registry: AliasRegistry,
    ) -> QueryRepairPlan:
        self.calls.append((question, missing_requirement_keys, alias_registry.registry_hash))
        return cast(QueryRepairPlan, self.plan_value)


def _query_plan(
    question: str,
    *,
    target: str = "repair_record",
    query_count: int = 1,
    bound_question: str | None = None,
) -> QueryRepairPlan:
    texts = (
        "xylophonic relay calibrated",
        "Alice relay laboratory record",
    )
    return QueryRepairPlan(
        original_question_sha256=sha256_hex((bound_question or question).encode("utf-8")),
        planner_id="static_test_planner",
        queries=tuple(
            DerivedQuery(
                query_key=f"q{index}",
                text=texts[index - 1],
                kind=(
                    QueryRepairKind.LEXICAL_BRIDGE
                    if index == 1
                    else QueryRepairKind.EVIDENCE_ROLE_SPLIT
                ),
                target_requirement_keys=(target,),
            )
            for index in range(1, query_count + 1)
        ),
    )


def _config(
    *,
    base: int = 1,
    rescue: int = 2,
    diversity: int = 20,
    family: int = 2,
) -> AdaptiveRetrievalConfig:
    return AdaptiveRetrievalConfig(
        base_fts_candidates=base,
        rescue_fts_candidates=rescue,
        max_union_candidates=40,
        family_diversity_window_candidates=diversity,
        max_per_family_before_diversity=family,
    )


def test_q0_fts_harrier_and_gate_finish_at_base_when_exact_proof_is_ready() -> None:
    fragment = _fragment(
        "fragment_exact_record",
        "The exact archival record proves the decisive anchor.",
    )
    harrier = _StaticHarrier()

    result = execute_adaptive_retrieval(
        question=QUESTION,
        fragments=(fragment,),
        gate_spec=_spec(),
        harrier=harrier,
        config=_config(base=1, rescue=2),
    )

    assert result.coverage.decision is EvidenceGateDecision.READY
    assert tuple(stage.stage for stage in result.stages) == ("base",)
    assert harrier.calls == [(QUESTION, ("fragment_exact_record",))]
    assert result.candidates[0].origins[0].kind is CandidateOriginKind.ORIGINAL_FTS
    assert result.original_question_sha256 == sha256_hex(QUESTION.encode("utf-8"))
    assert result.gate_spec_hash == _spec().spec_hash
    assert result.result_hash == result.result_hash


def test_incomplete_base_expands_from_one_to_two_fts_candidates() -> None:
    question = "Which wire record contains the proof?"
    bad = _fragment("fragment_a_bad", "wire record without the needed wording")
    good = _fragment("fragment_b_good", "wire record with the decisive anchor")
    harrier = _StaticHarrier(priority=(good.source_fragment_id, bad.source_fragment_id))

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(bad, good),
        gate_spec=_spec(question),
        harrier=harrier,
        config=_config(base=1, rescue=2),
    )

    assert tuple(stage.stage for stage in result.stages) == ("base", "rescue")
    assert result.stages[0].decision is EvidenceGateDecision.INSUFFICIENT
    assert result.stages[1].decision is EvidenceGateDecision.READY
    assert tuple(len(call[1]) for call in harrier.calls) == (1, 2)
    assert all(call[0] == question for call in harrier.calls)


def test_query_cloud_runs_only_after_rescue_and_keeps_all_harrier_calls_on_q0() -> None:
    question = "Who repaired the device?"
    requirement = _requirement(key="repair_record", anchor="Alice calibrated")
    distractor = _fragment(
        "fragment_a_distractor",
        "The device was repaired, but the report does not identify the operator.",
    )
    proof = _fragment(
        "fragment_z_proof",
        "Alice calibrated the xylophonic relay in the laboratory record.",
    )
    plan = _query_plan(question, query_count=2)
    provider = _PlanProvider(plan)
    harrier = _StaticHarrier(priority=(proof.source_fragment_id, distractor.source_fragment_id))

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(distractor, proof),
        gate_spec=_spec(question, requirement),
        harrier=harrier,
        query_repair=provider,
        config=_config(base=1, rescue=1),
    )

    assert tuple(stage.stage for stage in result.stages) == (
        "base",
        "rescue",
        "query_cloud",
    )
    assert result.coverage.decision is EvidenceGateDecision.READY
    assert result.query_repair_plan == plan
    assert provider.calls[0][:2] == (question, ("repair_record",))
    assert len(plan.queries) == 2
    assert all(call[0] == question for call in harrier.calls)
    proof_trace = next(
        item for item in result.candidates if item.source_fragment_id == proof.source_fragment_id
    )
    assert {
        (origin.kind, origin.query_key)
        for origin in proof_trace.origins
        if origin.kind is CandidateOriginKind.QUERY_CLOUD
    } == {
        (CandidateOriginKind.QUERY_CLOUD, "q1"),
        (CandidateOriginKind.QUERY_CLOUD, "q2"),
    }


def test_query_cloud_can_recover_after_both_q0_stages_have_an_empty_union() -> None:
    question = "Who handled unnamed apparatus?"
    requirement = _requirement(key="repair_record", anchor="Alice calibrated")
    proof = _fragment(
        "fragment_query_only",
        "Alice calibrated the xylophonic relay in the laboratory record.",
    )
    harrier = _StaticHarrier()

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(proof,),
        gate_spec=_spec(question, requirement),
        harrier=harrier,
        query_repair=_PlanProvider(_query_plan(question)),
        config=_config(base=1, rescue=1),
    )

    assert tuple(stage.candidate_count for stage in result.stages) == (0, 0, 1)
    assert result.coverage.decision is EvidenceGateDecision.READY
    assert len(harrier.calls) == 1
    assert harrier.calls[0][0] == question


def test_query_cloud_reserve_keeps_exact_lexical_hit_below_harrier_union() -> None:
    question = "Who repaired the device?"
    requirement = _requirement(key="repair_record", anchor="Alice calibrated")
    distractors = tuple(
        _fragment(
            f"fragment_distractor_{index:02d}",
            f"The device repair report omits the operator name, copy {index}.",
        )
        for index in range(1, 41)
    )
    proof = _fragment(
        "fragment_query_only_proof",
        "Alice calibrated the xylophonic relay in the laboratory record.",
    )
    priority = tuple(item.source_fragment_id for item in (*distractors, proof))

    without_reserve = execute_adaptive_retrieval(
        question=question,
        fragments=(*distractors, proof),
        gate_spec=_spec(question, requirement),
        harrier=_StaticHarrier(priority=priority),
        query_repair=_PlanProvider(_query_plan(question)),
        config=AdaptiveRetrievalConfig(
            base_fts_candidates=40,
            rescue_fts_candidates=40,
            max_union_candidates=40,
            family_diversity_window_candidates=30,
            query_cloud_reserve_per_query=0,
        ),
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(*distractors, proof),
        gate_spec=_spec(question, requirement),
        harrier=_StaticHarrier(priority=priority),
        query_repair=_PlanProvider(_query_plan(question)),
        config=AdaptiveRetrievalConfig(
            base_fts_candidates=40,
            rescue_fts_candidates=40,
            max_union_candidates=40,
            family_diversity_window_candidates=30,
            query_cloud_reserve_per_query=1,
        ),
    )

    assert without_reserve.coverage.decision is EvidenceGateDecision.INSUFFICIENT
    assert proof.source_fragment_id not in {
        item.source_fragment_id for item in without_reserve.candidates
    }
    assert result.coverage.decision is EvidenceGateDecision.READY
    assert result.matched_proof[0].source_fragment_id == proof.source_fragment_id
    assert result.matched_proof[0].rank == 40
    trace = next(
        item for item in result.candidates if item.source_fragment_id == proof.source_fragment_id
    )
    assert trace.query_cloud_reserved is True
    assert trace.origins[-1] == CandidateOrigin(
        kind=CandidateOriginKind.QUERY_CLOUD,
        query_key="q1",
        rank=1,
    )


def test_query_cloud_reserve_never_overrides_gate_forbidden_anchors() -> None:
    question = "Who repaired the device?"
    requirement = EvidenceRequirement(
        key="repair_record",
        label="Safe repair record",
        source_kinds_any=("archival record",),
        anchor_groups=(("Alice calibrated",),),
        forbidden_anchors=("fabricated appendix",),
    )
    negative = _fragment(
        "fragment_query_only_negative",
        "Alice calibrated the xylophonic relay in a fabricated appendix.",
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(negative,),
        gate_spec=_spec(question, requirement),
        harrier=_StaticHarrier(),
        query_repair=_PlanProvider(_query_plan(question)),
        config=_config(base=1, rescue=1),
    )

    assert result.coverage.decision is EvidenceGateDecision.INSUFFICIENT
    assert result.matched_proof == ()
    assert result.candidates[0].query_cloud_reserved is True


def test_query_cloud_reserve_preserves_prior_gate_match_and_completes_second_role() -> None:
    question = "Which control record proves the alpha role?"
    alpha = _fragment(
        "fragment_alpha_proof",
        "The control record proves the alpha evidence anchor.",
    )
    distractor = _fragment(
        "fragment_control_distractor",
        "A control record with no exact evidence anchor.",
    )
    beta = _fragment(
        "fragment_beta_proof",
        "The xylophonic relay calibrated report contains the beta evidence anchor.",
    )
    alpha_requirement = _requirement(
        key="alpha_record",
        anchor="alpha evidence anchor",
    )
    beta_requirement = _requirement(
        key="beta_record",
        anchor="beta evidence anchor",
    )
    plan = _query_plan(question, target="beta_record")
    harrier = _StaticHarrier(
        priority=(
            distractor.source_fragment_id,
            alpha.source_fragment_id,
            beta.source_fragment_id,
        )
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(distractor, beta, alpha),
        gate_spec=_spec(question, alpha_requirement, beta_requirement),
        harrier=harrier,
        query_repair=_PlanProvider(plan),
        config=AdaptiveRetrievalConfig(
            base_fts_candidates=2,
            rescue_fts_candidates=2,
            max_union_candidates=2,
            family_diversity_window_candidates=2,
            max_per_family_before_diversity=2,
            query_cloud_reserve_per_query=1,
        ),
    )

    assert tuple(stage.stage for stage in result.stages) == (
        "base",
        "rescue",
        "query_cloud",
    )
    prior_matches = result.stages[-2].gate_scan.matched_fragment_ids
    assert prior_matches == (alpha.source_fragment_id,)
    assert result.stages[-1].pool_selection.protected_fragment_ids == prior_matches
    assert result.stages[-1].union_selection.protected_fragment_ids == prior_matches
    assert set(prior_matches).issubset(result.stages[-1].gate_scan.matched_fragment_ids)
    assert result.coverage.decision is EvidenceGateDecision.READY
    assert {item.source_fragment_id for item in result.matched_proof} == {
        alpha.source_fragment_id,
        beta.source_fragment_id,
    }
    assert {item.source_fragment_id for item in result.candidates} == {
        alpha.source_fragment_id,
        beta.source_fragment_id,
    }
    reserve_receipt = result.stages[-1].union_selection
    assert reserve_receipt.initial_fragment_ids == (
        distractor.source_fragment_id,
        alpha.source_fragment_id,
    )
    assert reserve_receipt.protected_output_fragment_ids == (
        distractor.source_fragment_id,
        alpha.source_fragment_id,
    )
    assert reserve_receipt.forced_fragment_ids == (beta.source_fragment_id,)
    assert reserve_receipt.evicted_fragment_ids == (distractor.source_fragment_id,)
    assert reserve_receipt.output_fragment_ids == (
        alpha.source_fragment_id,
        beta.source_fragment_id,
    )


def test_saturated_discovery_over_500_keeps_q1_and_q2_evidence() -> None:
    question = "Where is controlzeta recorded?"
    alpha = _fragment(
        "fragment_q1_alpha_proof",
        "The xylophonic relay calibrated file contains the alpha evidence anchor.",
    )
    beta = _fragment(
        "fragment_q2_beta_proof",
        "The Alice relay laboratory record contains the beta evidence anchor.",
    )
    distractors = tuple(
        _fragment(
            f"fragment_control_{index:03d}",
            f"Controlzeta generic dossier item {index} without an exact anchor.",
        )
        for index in range(500)
    )
    alpha_requirement = _requirement(
        key="alpha_record",
        anchor="alpha evidence anchor",
    )
    beta_requirement = _requirement(
        key="beta_record",
        anchor="beta evidence anchor",
    )
    plan = QueryRepairPlan(
        original_question_sha256=sha256_hex(question.encode("utf-8")),
        planner_id="static_test_planner",
        queries=(
            DerivedQuery(
                query_key="q1",
                text="xylophonic relay calibrated",
                kind=QueryRepairKind.LEXICAL_BRIDGE,
                target_requirement_keys=("alpha_record",),
            ),
            DerivedQuery(
                query_key="q2",
                text="Alice relay laboratory record",
                kind=QueryRepairKind.EVIDENCE_ROLE_SPLIT,
                target_requirement_keys=("beta_record",),
            ),
        ),
    )
    priority = tuple(item.source_fragment_id for item in (*distractors, beta, alpha))

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(*distractors, beta, alpha),
        gate_spec=_spec(question, alpha_requirement, beta_requirement),
        harrier=_StaticHarrier(priority=priority),
        query_repair=_PlanProvider(plan),
        config=AdaptiveRetrievalConfig(
            base_fts_candidates=500,
            rescue_fts_candidates=500,
            max_union_candidates=40,
            family_diversity_window_candidates=20,
            max_per_family_before_diversity=2,
            query_cloud_reserve_per_query=1,
        ),
    )

    query_stage = result.stages[-1]
    assert query_stage.stage == "query_cloud"
    assert query_stage.discovered_candidate_count > 500
    assert query_stage.candidate_count == 500
    assert set(query_stage.pool_selection.forced_fragment_ids) == {
        alpha.source_fragment_id,
        beta.source_fragment_id,
    }
    assert len(query_stage.pool_selection.evicted_fragment_ids) == 2
    assert set(query_stage.union_selection.forced_fragment_ids) == {
        alpha.source_fragment_id,
        beta.source_fragment_id,
    }
    assert len(query_stage.union_selection.evicted_fragment_ids) == 2
    assert result.coverage.decision is EvidenceGateDecision.READY
    assert {item.source_fragment_id for item in result.matched_proof} == {
        alpha.source_fragment_id,
        beta.source_fragment_id,
    }
    origins_by_id = {item.source_fragment_id: item.origins for item in result.candidates}
    assert (
        CandidateOrigin(
            kind=CandidateOriginKind.QUERY_CLOUD,
            query_key="q1",
            rank=1,
        )
        in origins_by_id[alpha.source_fragment_id]
    )
    assert (
        CandidateOrigin(
            kind=CandidateOriginKind.QUERY_CLOUD,
            query_key="q2",
            rank=1,
        )
        in origins_by_id[beta.source_fragment_id]
    )
    final_rank = {item.source_fragment_id: item.final_rank for item in result.candidates}
    assert final_rank[beta.source_fragment_id] < final_rank[alpha.source_fragment_id]


def test_exact_phrase_and_neighbor_replay_is_invariant_to_fragment_shuffle() -> None:
    question = 'Where is "silent violet circuit" recorded?'
    proof = _fragment(
        "fragment_a_neighbor_proof",
        "The decisive anchor appears in the adjacent archival record.",
        source="source_shared",
        version="version_shared",
        ordinal=0,
    )
    phrase_seed = _fragment(
        "fragment_b_phrase_seed",
        "The silent violet circuit is recorded in the setup marker.",
        source="source_shared",
        version="version_shared",
        ordinal=1,
    )
    trailing = _fragment(
        "fragment_c_trailing_context",
        "The trailing paragraph contains context but no proof.",
        source="source_shared",
        version="version_shared",
        ordinal=2,
    )
    fragments = (proof, phrase_seed, trailing)
    config = AdaptiveRetrievalConfig(
        base_fts_candidates=1,
        rescue_fts_candidates=1,
        max_union_candidates=3,
        family_diversity_window_candidates=3,
        max_per_family_before_diversity=3,
        max_exact_phrase_candidates=1,
        max_neighbor_candidates=1,
        query_cloud_reserve_per_query=1,
    )
    payloads: list[dict[str, Any]] = []
    for seed in range(8):
        shuffled = list(fragments)
        Random(seed).shuffle(shuffled)
        result = execute_adaptive_retrieval(
            question=question,
            fragments=tuple(shuffled),
            gate_spec=_spec(question),
            harrier=_StaticHarrier(priority=(proof.source_fragment_id,)),
            config=config,
        )
        payloads.append(result.model_dump(mode="json"))
        by_id = {item.source_fragment_id: item for item in result.candidates}
        assert CandidateOriginKind.EXACT_PHRASE in {
            origin.kind for origin in by_id[phrase_seed.source_fragment_id].origins
        }
        assert CandidateOriginKind.NEIGHBOR in {
            origin.kind for origin in by_id[proof.source_fragment_id].origins
        }

    assert all(payload == payloads[0] for payload in payloads[1:])


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("final_stage_coverage", "stage coverage hash must match embedded coverage"),
        ("final_stage_decision", "stage decision must match embedded coverage"),
        ("candidate_text_hash", "Gate scan must match candidate rank and text hash"),
        ("selection_receipt", "selection output hash does not match output IDs"),
        ("gate_spec", "every stage must use the frozen Gate specification"),
    ),
)
def test_adaptive_result_rejects_tampered_final_closure(
    tamper: str,
    message: str,
) -> None:
    fragment = _fragment(
        "fragment_tamper_proof",
        "The archival record proves the decisive anchor.",
    )
    result = execute_adaptive_retrieval(
        question=QUESTION,
        fragments=(fragment,),
        gate_spec=_spec(),
        harrier=_StaticHarrier(),
        config=_config(base=1, rescue=1),
    )
    payload = deepcopy(result.model_dump(mode="json"))

    if tamper == "final_stage_coverage":
        payload["stages"][-1]["coverage_result_hash"] = "f" * 64
    elif tamper == "final_stage_decision":
        payload["stages"][-1]["decision"] = EvidenceGateDecision.INSUFFICIENT.value
    elif tamper == "candidate_text_hash":
        payload["candidates"][0]["text_sha256"] = "f" * 64
    elif tamper == "selection_receipt":
        payload["stages"][-1]["union_selection"]["output_hash"] = "f" * 64
    elif tamper == "gate_spec":
        payload["gate_spec_hash"] = "f" * 64
    else:  # pragma: no cover - the parameter list is the closed tamper set
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises(ValidationError, match=message):
        AdaptiveRecallResult.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("plan_value", "message"),
    (
        (object(), "invalid plan"),
        (
            _query_plan(
                "Who repaired the device?",
                bound_question="A different original question",
            ),
            "not bound to q0",
        ),
        (
            _query_plan("Who repaired the device?", target="already_covered_role"),
            "targets non-missing requirements",
        ),
    ),
)
def test_query_cloud_fails_closed_for_invalid_or_unbound_plans(
    plan_value: object,
    message: str,
) -> None:
    question = "Who repaired the device?"
    distractor = _fragment("fragment_distractor", "The device was repaired anonymously.")

    with pytest.raises(AdaptiveRecallError, match=message):
        execute_adaptive_retrieval(
            question=question,
            fragments=(distractor,),
            gate_spec=_spec(
                question,
                _requirement(key="repair_record", anchor="Alice calibrated"),
            ),
            harrier=_StaticHarrier(),
            query_repair=_PlanProvider(plan_value),
            config=_config(base=1, rescue=1),
        )


def test_alias_registry_adds_only_relevant_controlled_variants() -> None:
    question = "Did AC triumph?"
    proof = _fragment(
        "fragment_alternating_current",
        "The alternating current network supplied the decisive anchor.",
    )
    unrelated = AliasEntry(canonical="Nikola Tesla", alternatives=("Tesla",))
    registry = AliasRegistry(
        version="alias_v2",
        entries=(
            unrelated,
            AliasEntry(canonical="alternating current", alternatives=("AC",)),
        ),
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(proof,),
        gate_spec=_spec(question),
        harrier=_StaticHarrier(),
        alias_registry=registry,
        config=_config(base=1, rescue=1),
    )

    assert registry.expanded_question(question) == "Did AC triumph? alternating current"
    assert registry.expanded_question("A question without registry terms") is None
    assert result.coverage.decision is EvidenceGateDecision.READY
    assert CandidateOriginKind.ALIAS_FTS in {origin.kind for origin in result.candidates[0].origins}
    assert result.alias_registry_hash == registry.registry_hash


def test_quoted_phrase_is_a_traced_deterministic_candidate_origin() -> None:
    question = 'Where is "silent violet circuit" recorded?'
    proof = _fragment(
        "fragment_phrase",
        "The silent violet circuit is the decisive anchor in this record.",
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(proof,),
        gate_spec=_spec(question),
        harrier=_StaticHarrier(),
        config=_config(base=1, rescue=1),
    )

    assert CandidateOriginKind.EXACT_PHRASE in {
        origin.kind for origin in result.candidates[0].origins
    }


def test_same_version_neighbor_can_supply_exact_proof_without_query_overlap() -> None:
    question = "What followed the setup marker?"
    seed = _fragment(
        "fragment_setup",
        "The setup marker appears here.",
        source="source_shared",
        version="version_shared",
        ordinal=0,
    )
    proof = _fragment(
        "fragment_following",
        "The decisive anchor follows in the adjacent paragraph.",
        source="source_shared",
        version="version_shared",
        ordinal=1,
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(seed, proof),
        gate_spec=_spec(question),
        harrier=_StaticHarrier(priority=(proof.source_fragment_id, seed.source_fragment_id)),
        config=_config(base=1, rescue=1),
    )

    assert result.coverage.decision is EvidenceGateDecision.READY
    proof_trace = next(
        item for item in result.candidates if item.source_fragment_id == proof.source_fragment_id
    )
    assert len(proof_trace.origins) == 1
    assert proof_trace.origins[0].kind is CandidateOriginKind.NEIGHBOR


def test_family_diversity_defers_repeated_family_without_changing_scores() -> None:
    question = "Where does alpha appear?"
    first = _fragment("fragment_a_one", "alpha decisive anchor", family="archive")
    second = _fragment("fragment_a_two", "alpha secondary text", family="archive")
    other = _fragment("fragment_b_one", "alpha contextual text", family="monograph")
    priority = tuple(item.source_fragment_id for item in (first, second, other))

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(first, second, other),
        gate_spec=_spec(question),
        harrier=_StaticHarrier(priority=priority),
        config=_config(base=3, rescue=3, family=1),
    )

    assert tuple(item.source_fragment_id for item in result.candidates) == (
        first.source_fragment_id,
        other.source_fragment_id,
        second.source_fragment_id,
    )
    scores = tuple(float.fromhex(item.harrier_score_hex) for item in result.candidates)
    assert scores == pytest.approx((0.99, 0.97, 0.98))


def test_metadata_only_fragment_is_ranked_but_never_admitted_as_proof() -> None:
    question = "Where is the sealed evidence?"
    metadata = _fragment(
        "fragment_metadata",
        "The sealed evidence is the decisive anchor.",
        proof=False,
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(metadata,),
        gate_spec=_spec(question),
        harrier=_StaticHarrier(),
        config=_config(base=1, rescue=1),
    )

    assert len(result.candidates) == 1
    assert result.coverage.decision is EvidenceGateDecision.INSUFFICIENT
    assert result.coverage.requirements[0].closest_candidates == ()


def test_metadata_only_rank_does_not_hide_body_proof_from_wide_gate() -> None:
    question = "Which record contains the decisive anchor?"
    metadata = _fragment(
        "fragment_a_metadata",
        "A highly relevant record title without the required quotation.",
        proof=False,
    )
    body = _fragment(
        "fragment_b_body",
        "The record states the decisive anchor in its exact body text.",
    )
    harrier = _StaticHarrier(priority=(metadata.source_fragment_id, body.source_fragment_id))

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(metadata, body),
        gate_spec=_spec(question),
        harrier=harrier,
        config=_config(base=2, rescue=2, diversity=1),
    )

    assert tuple(item.source_fragment_id for item in result.candidates) == (
        metadata.source_fragment_id,
        body.source_fragment_id,
    )
    assert result.coverage.decision is EvidenceGateDecision.READY
    assert result.coverage.requirements[0].matched_fragment_ids == (body.source_fragment_id,)


def test_expected_corpus_gap_is_terminal_after_the_base_stage() -> None:
    question = "Is there an official impossible record?"
    candidate = _fragment("fragment_thematic", "A thematic but unrelated record.")

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(candidate,),
        gate_spec=_spec(
            question,
            _requirement(anchor="official impossible anchor"),
            answerability=EvidenceAnswerability.NOT_IN_CORPUS,
        ),
        harrier=_StaticHarrier(),
        config=_config(base=1, rescue=2),
    )

    assert result.coverage.decision is EvidenceGateDecision.GAP_PRESERVED
    assert tuple(stage.stage for stage in result.stages) == ("base",)


@dataclass
class _FakeSemanticProvider:
    model_profile: Any = HARRIER_OSS_V1_270M_PROFILE
    runtime_profile: ModelRuntimeProfile = field(
        default_factory=lambda: ModelRuntimeProfile(
            sentence_transformers_version="5.6.0",
            transformers_version="4.99.0",
            torch_version="2.9.0",
            batch_size=4,
        )
    )
    query_calls: list[str] = field(default_factory=list)
    passage_calls: list[tuple[str, ...]] = field(default_factory=list)
    wrong_count: bool = False

    def embed_query(self, question: str) -> PackedVector:
        self.query_calls.append(question)
        return pack_normalized_vector((1.0, 0.0))

    def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
        self.passage_calls.append(texts)
        result = tuple(
            pack_normalized_vector((1.0, 0.0) if "relevant" in text else (0.0, 1.0))
            for text in texts
        )
        return result[:-1] if self.wrong_count else result


def _candidate_text(identifier: str, text: str) -> CandidateText:
    return CandidateText(
        source_fragment_id=identifier,
        source_id=f"source_{identifier.removeprefix('fragment_')}",
        text=text,
        text_sha256=sha256_hex(text.encode("utf-8")),
    )


def test_semantic_harrier_scores_candidates_and_reuses_query_and_passage_vectors() -> None:
    provider = _FakeSemanticProvider()
    scorer = SemanticHarrierCandidateScorer(
        provider,
        provisioning_receipt_hash=HASH_A,
    )
    relevant = _candidate_text("fragment_relevant", "relevant exact passage")
    other = _candidate_text("fragment_other", "orthogonal passage")

    first = scorer.score(q0="original question", candidates=(other, relevant))
    second = scorer.score(q0="original question", candidates=(relevant, other))
    added = _candidate_text("fragment_added", "another relevant passage")
    third = scorer.score(q0="original question", candidates=(other, relevant, added))

    assert first.items[0].source_fragment_id == "fragment_relevant"
    assert first.original_question_sha256 == sha256_hex(b"original question")
    assert first.model_profile_hash == HARRIER_OSS_V1_270M_PROFILE.profile_hash
    assert first.runtime_profile_hash == provider.runtime_profile.profile_hash
    assert first.provisioning_receipt_hash == HASH_A
    assert first.batch_hash == second.batch_hash
    assert provider.query_calls == ["original question"]
    assert provider.passage_calls == [
        ("orthogonal passage", "relevant exact passage"),
        ("another relevant passage",),
    ]
    assert len(third.items) == 3


def test_semantic_harrier_uses_a_separate_query_cache_key_for_a_new_q0() -> None:
    provider = _FakeSemanticProvider()
    scorer = SemanticHarrierCandidateScorer(provider, provisioning_receipt_hash=HASH_A)
    candidate = _candidate_text("fragment_one", "relevant passage")

    scorer.score(q0="first question", candidates=(candidate,))
    scorer.score(q0="second question", candidates=(candidate,))

    assert provider.query_calls == ["first question", "second question"]
    assert provider.passage_calls == [("relevant passage",)]


def test_semantic_harrier_rejects_wrong_profile_invalid_candidates_and_bad_provider_count() -> None:
    with pytest.raises(AdaptiveRecallError, match="pinned Harrier profile"):
        SemanticHarrierCandidateScorer(
            _FakeSemanticProvider(model_profile=MULTILINGUAL_E5_SMALL_PROFILE),
            provisioning_receipt_hash=HASH_A,
        )
    with pytest.raises(ValueError, match="hash"):
        SemanticHarrierCandidateScorer(
            _FakeSemanticProvider(),
            provisioning_receipt_hash="bad",
        )

    scorer = SemanticHarrierCandidateScorer(
        _FakeSemanticProvider(), provisioning_receipt_hash=HASH_A
    )
    with pytest.raises(AdaptiveRecallError, match="1-500"):
        scorer.score(q0="question", candidates=())
    duplicate = _candidate_text("fragment_duplicate", "relevant text")
    with pytest.raises(AdaptiveRecallError, match="IDs must be unique"):
        scorer.score(q0="question", candidates=(duplicate, duplicate))

    broken = SemanticHarrierCandidateScorer(
        _FakeSemanticProvider(wrong_count=True),
        provisioning_receipt_hash=HASH_A,
    )
    with pytest.raises(AdaptiveRecallError, match="wrong vector count"):
        broken.score(q0="question", candidates=(duplicate,))


@pytest.mark.parametrize(
    ("harrier", "message"),
    (
        (_StaticHarrier(return_object=True), "invalid score batch"),
        (
            _StaticHarrier(q0_override="another question"),
            "not anchored to q0",
        ),
        (_StaticHarrier(omit_last=True), "does not close over"),
    ),
)
def test_route_fails_closed_for_malformed_harrier_batches(
    harrier: _StaticHarrier,
    message: str,
) -> None:
    fragments = (
        _fragment("fragment_one", "record decisive anchor"),
        _fragment("fragment_two", "record contextual material"),
    )

    with pytest.raises(AdaptiveRecallError, match=message):
        execute_adaptive_retrieval(
            question=QUESTION,
            fragments=fragments,
            gate_spec=_spec(),
            harrier=harrier,
            config=_config(base=2, rescue=2),
        )


def test_route_preserves_no_hit_gap_and_rejects_duplicate_or_unfrozen_inputs() -> None:
    fragment = _fragment("fragment_one", "unrelated vocabulary")
    no_hits = execute_adaptive_retrieval(
        question="tokens absent everywhere",
        fragments=(fragment,),
        gate_spec=_spec("tokens absent everywhere"),
        harrier=_StaticHarrier(),
        config=_config(base=1, rescue=1),
    )
    assert no_hits.coverage.decision is EvidenceGateDecision.INSUFFICIENT
    assert no_hits.harrier_batch_hash is None
    assert no_hits.candidates == ()
    assert tuple(stage.candidate_count for stage in no_hits.stages) == (0, 0)
    with pytest.raises(AdaptiveRecallError, match="fragment IDs must be unique"):
        execute_adaptive_retrieval(
            question=QUESTION,
            fragments=(fragment, fragment),
            gate_spec=_spec(),
            harrier=_StaticHarrier(),
        )
    with pytest.raises(AdaptiveRecallError, match="frozen for the exact original question"):
        execute_adaptive_retrieval(
            question=QUESTION,
            fragments=(fragment,),
            gate_spec=_spec("Different question"),
            harrier=_StaticHarrier(),
        )


def test_adaptive_contracts_reject_ambiguous_or_noncanonical_values() -> None:
    with pytest.raises(ValidationError, match="does not match fragment text"):
        AdaptiveFragment(
            **{
                **_fragment("fragment_valid", "valid text").model_dump(),
                "text_sha256": HASH_A,
            }
        )
    with pytest.raises(ValidationError, match="alternatives must be unique"):
        AliasEntry(canonical="Tesla", alternatives=("TESLA", "tesla"))
    with pytest.raises(ValidationError, match="canonicals must be unique"):
        AliasRegistry(
            entries=(
                AliasEntry(canonical="Tesla", alternatives=("Nikola",)),
                AliasEntry(canonical="tesla", alternatives=("N. Tesla",)),
            )
        )
    with pytest.raises(ValidationError, match="one or two"):
        QueryRepairPlan(
            original_question_sha256=HASH_A,
            planner_id="planner",
            queries=(),
        )
    with pytest.raises(ValidationError, match="rescue FTS budget"):
        AdaptiveRetrievalConfig(base_fts_candidates=2, rescue_fts_candidates=1)
    with pytest.raises(ValidationError, match="family diversity window"):
        AdaptiveRetrievalConfig(
            max_union_candidates=1,
            family_diversity_window_candidates=2,
        )
    with pytest.raises(ValidationError, match="QueryCloud reserve"):
        AdaptiveRetrievalConfig(
            max_union_candidates=10,
            family_diversity_window_candidates=10,
            query_cloud_reserve_per_query=6,
        )


def test_score_contracts_require_exact_ordered_finite_cosine_receipts() -> None:
    item = HarrierScoreItem(
        source_fragment_id="fragment_one",
        text_sha256=HASH_A,
        rank=1,
        score_hex=(0.5).hex(),
        vector_sha256=HASH_B,
    )
    assert item.score == 0.5
    with pytest.raises(ValidationError, match=r"exact float\.hex"):
        HarrierScoreItem(
            source_fragment_id="fragment_bad",
            text_sha256=HASH_A,
            rank=1,
            score_hex="not-a-float",
            vector_sha256=HASH_B,
        )
    with pytest.raises(ValidationError, match="outside cosine range"):
        HarrierScoreItem(
            source_fragment_id="fragment_bad",
            text_sha256=HASH_A,
            rank=1,
            score_hex=(2.0).hex(),
            vector_sha256=HASH_B,
        )
    second = HarrierScoreItem(
        source_fragment_id="fragment_two",
        text_sha256=HASH_B,
        rank=2,
        score_hex=(0.75).hex(),
        vector_sha256=HASH_C,
    )
    with pytest.raises(ValidationError, match="monotonic descending"):
        HarrierScoreBatch(
            original_question_sha256=HASH_A,
            model_profile_hash=HASH_A,
            runtime_profile_hash=HASH_B,
            provisioning_receipt_hash=HASH_C,
            items=(item, second),
        )


def test_query_plan_is_content_addressed_sorted_and_bounded_to_two_queries() -> None:
    question = "Who repaired the device?"
    plan = _query_plan(question, query_count=2)

    assert tuple(query.query_key for query in plan.queries) == ("q1", "q2")
    assert plan.plan_hash == plan.plan_hash
    with pytest.raises(ValidationError, match="q1 or q2"):
        DerivedQuery(
            query_key="q3",
            text="third recursive query",
            kind=QueryRepairKind.LEXICAL_BRIDGE,
            target_requirement_keys=("repair_record",),
        )
    with pytest.raises(ValidationError, match="at least one"):
        DerivedQuery(
            query_key="q1",
            text="untargeted query",
            kind=QueryRepairKind.LEXICAL_BRIDGE,
            target_requirement_keys=(),
        )


def _ranked_wide_gate_fixture(
    *,
    total: int,
    proof_rank: int,
) -> tuple[tuple[AdaptiveFragment, ...], _StaticHarrier]:
    fragments = tuple(
        _fragment(
            f"fragment_wide_{index:03d}",
            (
                "common record with the decisive anchor"
                if index == proof_rank
                else f"common record distractor {index}"
            ),
        )
        for index in range(1, total + 1)
    )
    return fragments, _StaticHarrier(priority=tuple(item.source_fragment_id for item in fragments))


def test_wide_gate_sees_exact_proof_below_the_old_rank_30_window() -> None:
    fragments, harrier = _ranked_wide_gate_fixture(total=40, proof_rank=37)
    provider = _PlanProvider(object())

    result = execute_adaptive_retrieval(
        question="Which common record contains the decisive anchor?",
        fragments=fragments,
        gate_spec=_spec("Which common record contains the decisive anchor?"),
        harrier=harrier,
        query_repair=provider,
        config=AdaptiveRetrievalConfig(
            base_fts_candidates=40,
            rescue_fts_candidates=40,
            max_union_candidates=40,
            family_diversity_window_candidates=30,
        ),
    )

    assert result.coverage.decision is EvidenceGateDecision.READY
    assert tuple(stage.stage for stage in result.stages) == ("base",)
    assert provider.calls == []
    assert result.matched_proof[0].rank == 37
    assert result.stages[0].gate_scan.candidate_count == 40


def test_wide_gate_includes_rank_160_but_never_rank_161() -> None:
    question = "Which common record contains the decisive anchor?"
    fragments, harrier = _ranked_wide_gate_fixture(total=161, proof_rank=160)
    config = AdaptiveRetrievalConfig(
        base_fts_candidates=161,
        rescue_fts_candidates=161,
        max_union_candidates=160,
        family_diversity_window_candidates=30,
    )

    at_limit = execute_adaptive_retrieval(
        question=question,
        fragments=fragments,
        gate_spec=_spec(question),
        harrier=harrier,
        config=config,
    )
    assert at_limit.coverage.decision is EvidenceGateDecision.READY
    assert at_limit.matched_proof[0].rank == 160

    beyond, beyond_harrier = _ranked_wide_gate_fixture(total=161, proof_rank=161)
    excluded = execute_adaptive_retrieval(
        question=question,
        fragments=beyond,
        gate_spec=_spec(question),
        harrier=beyond_harrier,
        config=config,
    )
    assert excluded.coverage.decision is EvidenceGateDecision.INSUFFICIENT
    assert excluded.stages[-1].gate_scan.candidate_count == 160
    assert excluded.matched_proof == ()


def test_matched_proof_contains_only_gate_matches_and_deduplicates_roles() -> None:
    question = "Which common record contains the decisive anchor?"
    distractor = _fragment("fragment_unmatched", "common record without the needed text")
    proof = _fragment("fragment_shared_proof", "common record decisive anchor")
    requirements = (
        _requirement(key="first_role"),
        _requirement(key="second_role"),
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(distractor, proof),
        gate_spec=_spec(question, *requirements),
        harrier=_StaticHarrier(priority=(distractor.source_fragment_id, proof.source_fragment_id)),
        config=_config(base=2, rescue=2),
    )

    assert result.coverage.decision is EvidenceGateDecision.READY
    assert tuple(item.source_fragment_id for item in result.matched_proof) == (
        proof.source_fragment_id,
    )
    assert result.stages[-1].gate_scan.matched_fragment_ids == (proof.source_fragment_id,)


def test_wide_gate_hard_negative_remains_unmatched() -> None:
    question = "Which record supplies safe clean-light guidance?"
    negative = _fragment(
        "fragment_hard_negative",
        "safe clean-light guidance decisive anchor but always add global haze",
    )
    requirement = EvidenceRequirement(
        key="clean_light",
        label="Safe clean-light guidance without global haze",
        source_kinds_any=("archival record",),
        anchor_groups=(("decisive anchor",),),
        forbidden_anchors=("global haze",),
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(negative,),
        gate_spec=_spec(question, requirement),
        harrier=_StaticHarrier(),
        config=_config(base=1, rescue=1),
    )

    assert result.coverage.decision is EvidenceGateDecision.INSUFFICIENT
    assert result.matched_proof == ()


def test_wide_gate_scan_budget_fails_closed_without_truncation() -> None:
    fragment = _fragment(
        "fragment_oversized",
        "common record decisive anchor with deliberately oversized body text",
    )

    with pytest.raises(AdaptiveRecallError, match="character budget exceeded"):
        execute_adaptive_retrieval(
            question="Which common record contains the decisive anchor?",
            fragments=(fragment,),
            gate_spec=_spec("Which common record contains the decisive anchor?"),
            harrier=_StaticHarrier(),
            config=AdaptiveRetrievalConfig(
                base_fts_candidates=1,
                rescue_fts_candidates=1,
                max_union_candidates=1,
                family_diversity_window_candidates=1,
                max_gate_scan_characters=10,
                query_cloud_reserve_per_query=0,
            ),
        )


def test_wide_gate_config_scan_projection_and_result_hashes_replay() -> None:
    question = "Which common record contains the decisive anchor?"
    fragments, harrier = _ranked_wide_gate_fixture(total=40, proof_rank=37)
    config = AdaptiveRetrievalConfig(
        base_fts_candidates=40,
        rescue_fts_candidates=40,
        max_union_candidates=40,
        family_diversity_window_candidates=30,
    )

    first = execute_adaptive_retrieval(
        question=question,
        fragments=fragments,
        gate_spec=_spec(question),
        harrier=harrier,
        config=config,
    )
    second = execute_adaptive_retrieval(
        question=question,
        fragments=fragments,
        gate_spec=_spec(question),
        harrier=_StaticHarrier(priority=harrier.priority),
        config=config,
    )

    assert first.config_hash == second.config_hash == config.config_hash
    assert first.stages[-1].gate_scan.receipt_hash == second.stages[-1].gate_scan.receipt_hash
    assert first.stages[-1].gate_scan.matched_proof_hash == (
        second.stages[-1].gate_scan.matched_proof_hash
    )
    assert first.result_hash == second.result_hash


def test_remaining_contract_edges_are_fail_closed() -> None:
    valid = _fragment("fragment_valid", "valid text")
    payload = valid.model_dump()
    with pytest.raises(ValidationError, match="fragment_<key>"):
        AdaptiveFragment(**{**payload, "source_fragment_id": "invalid"})
    with pytest.raises(ValidationError, match="stable lowercase identifiers"):
        AdaptiveFragment(**{**payload, "source_id": "Invalid Source"})
    with pytest.raises(ValidationError, match="evidence_tags must be unique"):
        AdaptiveFragment(**{**payload, "evidence_tags": ("tag", "tag")})
    with pytest.raises(ValidationError, match="1-32 alternatives"):
        AliasEntry(canonical="Tesla", alternatives=())
    with pytest.raises(ValidationError, match="stable snake_case"):
        AliasRegistry(version="Bad-Version")
    with pytest.raises(ValidationError, match="stable snake_case"):
        DerivedQuery(
            query_key="q1",
            text="valid query",
            kind=QueryRepairKind.LEXICAL_BRIDGE,
            target_requirement_keys=("Invalid Key",),
        )
    with pytest.raises(ValidationError, match="must be unique"):
        DerivedQuery(
            query_key="q1",
            text="valid query",
            kind=QueryRepairKind.LEXICAL_BRIDGE,
            target_requirement_keys=("same_role", "same_role"),
        )
    with pytest.raises(ValidationError, match="planner_id"):
        QueryRepairPlan(
            original_question_sha256=HASH_A,
            planner_id="Bad Planner",
            queries=(_query_plan("question").queries[0],),
        )

    first = DerivedQuery(
        query_key="q1",
        text="first query",
        kind=QueryRepairKind.LEXICAL_BRIDGE,
        target_requirement_keys=("role",),
    )
    duplicate_key = DerivedQuery(
        query_key="q1",
        text="second query",
        kind=QueryRepairKind.EVIDENCE_ROLE_SPLIT,
        target_requirement_keys=("role",),
    )
    duplicate_text = DerivedQuery(
        query_key="q2",
        text="first query",
        kind=QueryRepairKind.EVIDENCE_ROLE_SPLIT,
        target_requirement_keys=("role",),
    )
    with pytest.raises(ValidationError, match="query keys must be unique"):
        QueryRepairPlan(
            original_question_sha256=HASH_A,
            planner_id="planner",
            queries=(first, duplicate_key),
        )
    with pytest.raises(ValidationError, match="query texts must be unique"):
        QueryRepairPlan(
            original_question_sha256=HASH_A,
            planner_id="planner",
            queries=(first, duplicate_text),
        )


def test_harrier_batch_rejects_empty_gapped_and_duplicate_receipts() -> None:
    with pytest.raises(ValidationError, match="1-500 scores"):
        HarrierScoreBatch(
            original_question_sha256=HASH_A,
            model_profile_hash=HASH_A,
            runtime_profile_hash=HASH_B,
            provisioning_receipt_hash=HASH_C,
            items=(),
        )
    first = HarrierScoreItem(
        source_fragment_id="fragment_one",
        text_sha256=HASH_A,
        rank=1,
        score_hex=(0.9).hex(),
        vector_sha256=HASH_B,
    )
    gap = HarrierScoreItem(
        source_fragment_id="fragment_two",
        text_sha256=HASH_B,
        rank=3,
        score_hex=(0.8).hex(),
        vector_sha256=HASH_C,
    )
    with pytest.raises(ValidationError, match="contiguous"):
        HarrierScoreBatch(
            original_question_sha256=HASH_A,
            model_profile_hash=HASH_A,
            runtime_profile_hash=HASH_B,
            provisioning_receipt_hash=HASH_C,
            items=(first, gap),
        )
    duplicate = HarrierScoreItem(
        source_fragment_id="fragment_one",
        text_sha256=HASH_B,
        rank=2,
        score_hex=(0.8).hex(),
        vector_sha256=HASH_C,
    )
    with pytest.raises(ValidationError, match="fragment IDs must be unique"):
        HarrierScoreBatch(
            original_question_sha256=HASH_A,
            model_profile_hash=HASH_A,
            runtime_profile_hash=HASH_B,
            provisioning_receipt_hash=HASH_C,
            items=(first, duplicate),
        )


def test_candidate_pool_rejects_unknown_ids_and_bounds_local_rescue_helpers() -> None:
    left = _fragment(
        "fragment_left",
        "needle left",
        source="source_shared",
        version="version_shared",
        ordinal=0,
    )
    middle = _fragment(
        "fragment_middle",
        "needle middle",
        source="source_shared",
        version="version_shared",
        ordinal=1,
    )
    right = _fragment(
        "fragment_right",
        "needle right",
        source="source_shared",
        version="version_shared",
        ordinal=2,
    )
    fragments = {item.source_fragment_id: item for item in (left, middle, right)}
    origin = CandidateOrigin(kind=CandidateOriginKind.NEIGHBOR, query_key="q0")

    bounded = adaptive._CandidatePool(fragments)
    with pytest.raises(AdaptiveRecallError, match="outside the authorized set"):
        bounded.add("fragment_unknown", origin)
    bounded.add(middle.source_fragment_id, origin)
    bounded.add(left.source_fragment_id, origin)
    assert tuple(bounded.origins) == (
        middle.source_fragment_id,
        left.source_fragment_id,
    )

    phrase_pool = adaptive._CandidatePool(fragments)
    phrase_pool.add_exact_phrases(("needle",), max_new=0)
    assert phrase_pool.origins == {}
    phrase_pool.add_exact_phrases(("needle",), max_new=1)
    assert len(phrase_pool.origins) == 1

    neighbor_pool = adaptive._CandidatePool(fragments)
    neighbor_pool.add(middle.source_fragment_id, origin)
    neighbor_pool.add_neighbors(max_new=0)
    assert len(neighbor_pool.origins) == 1
    neighbor_pool.add_neighbors(max_new=1)
    assert tuple(neighbor_pool.origins) == (
        middle.source_fragment_id,
        left.source_fragment_id,
    )


def test_route_rejects_duplicate_source_version_positions() -> None:
    first = _fragment(
        "fragment_position_first",
        "The first exact record contains the decisive anchor.",
        source="source_shared",
        version="version_shared",
        ordinal=7,
    )
    second = _fragment(
        "fragment_position_second",
        "The second exact record contains supporting context.",
        source="source_shared",
        version="version_shared",
        ordinal=7,
    )

    with pytest.raises(AdaptiveRecallError, match="ordinal positions must be unique"):
        execute_adaptive_retrieval(
            question=QUESTION,
            fragments=(second, first),
            gate_spec=_spec(),
            harrier=_StaticHarrier(),
        )


def test_route_rejects_empty_or_oversized_input_and_contract_subclasses() -> None:
    with pytest.raises(AdaptiveRecallError, match="1-100000"):
        execute_adaptive_retrieval(
            question=QUESTION,
            fragments=(),
            gate_spec=_spec(),
            harrier=_StaticHarrier(),
        )
    fragment = _fragment("fragment_one", "record decisive anchor")
    with pytest.raises(AdaptiveRecallError, match="1-100000"):
        execute_adaptive_retrieval(
            question=QUESTION,
            fragments=(fragment,) * 100_001,
            gate_spec=_spec(),
            harrier=_StaticHarrier(),
        )

    class AliasRegistrySubclass(AliasRegistry):
        pass

    class ConfigSubclass(AdaptiveRetrievalConfig):
        pass

    with pytest.raises(AdaptiveRecallError, match="exact contracts"):
        execute_adaptive_retrieval(
            question=QUESTION,
            fragments=(fragment,),
            gate_spec=_spec(),
            harrier=_StaticHarrier(),
            alias_registry=AliasRegistrySubclass(),
        )
    with pytest.raises(AdaptiveRecallError, match="exact contracts"):
        execute_adaptive_retrieval(
            question=QUESTION,
            fragments=(fragment,),
            gate_spec=_spec(),
            harrier=_StaticHarrier(),
            config=ConfigSubclass(),
        )


def test_bounded_text_and_alias_projection_reject_unsafe_or_oversized_values() -> None:
    with pytest.raises(ValueError, match="text must be str"):
        adaptive._bounded_text(cast(str, 7), maximum=10)
    for value in (" padded", "padded ", "bad\x00text", ""):
        with pytest.raises(ValueError, match="bounded"):
            adaptive._bounded_text(value, maximum=10)
    with pytest.raises(ValueError, match="bounded"):
        adaptive._bounded_text("too long", maximum=3)

    registry = AliasRegistry(
        entries=(
            AliasEntry(
                canonical="alternating current",
                alternatives=("AC",),
            ),
        )
    )
    oversized_question = f"AC {'x' * 1_996}"
    assert registry.expanded_question(oversized_question) is None


def test_zero_exact_phrase_and_neighbor_budgets_are_valid_route_configuration() -> None:
    question = 'Where is "decisive anchor"?'
    fragment = _fragment("fragment_exact", "decisive anchor")
    config = _config(base=1, rescue=1).model_copy(
        update={"max_exact_phrase_candidates": 0, "max_neighbor_candidates": 0}
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(fragment,),
        gate_spec=_spec(question),
        harrier=_StaticHarrier(),
        config=config,
    )

    assert result.coverage.decision is EvidenceGateDecision.READY
    assert CandidateOriginKind.EXACT_PHRASE not in {
        origin.kind for origin in result.candidates[0].origins
    }


def _audit_base_case(
    *, two_candidates: bool = False
) -> tuple[AdaptiveRecallResult, tuple[AdaptiveFragment, ...]]:
    question = "Which audit record contains the decisive anchor?"
    proof = _fragment(
        "fragment_audit_proof",
        "The audit record contains the decisive anchor.",
    )
    fragments: tuple[AdaptiveFragment, ...] = (proof,)
    if two_candidates:
        fragments = (
            proof,
            _fragment(
                "fragment_audit_context",
                "The audit record contains contextual material only.",
            ),
        )
    result = execute_adaptive_retrieval(
        question=question,
        fragments=fragments,
        gate_spec=_spec(question),
        harrier=_StaticHarrier(priority=tuple(item.source_fragment_id for item in fragments)),
        config=_config(base=len(fragments), rescue=len(fragments)),
    )
    return result, fragments


def _audit_protected_result() -> AdaptiveRecallResult:
    question = "Which control record proves the alpha role?"
    alpha = _fragment(
        "fragment_audit_alpha",
        "The control record proves the alpha evidence anchor.",
    )
    distractor = _fragment(
        "fragment_audit_distractor",
        "A control record with no exact evidence anchor.",
    )
    beta = _fragment(
        "fragment_audit_beta",
        "The xylophonic relay calibrated report contains the beta evidence anchor.",
    )
    return execute_adaptive_retrieval(
        question=question,
        fragments=(distractor, beta, alpha),
        gate_spec=_spec(
            question,
            _requirement(key="alpha_record", anchor="alpha evidence anchor"),
            _requirement(key="beta_record", anchor="beta evidence anchor"),
        ),
        harrier=_StaticHarrier(
            priority=(
                distractor.source_fragment_id,
                alpha.source_fragment_id,
                beta.source_fragment_id,
            )
        ),
        query_repair=_PlanProvider(_query_plan(question, target="beta_record")),
        config=AdaptiveRetrievalConfig(
            base_fts_candidates=2,
            rescue_fts_candidates=2,
            max_union_candidates=2,
            family_diversity_window_candidates=2,
            max_per_family_before_diversity=2,
            query_cloud_reserve_per_query=1,
        ),
    )


def _selection_receipt_fixture() -> adaptive.AdaptiveCandidateSelectionReceipt:
    first = _fragment("fragment_select_first", "first selection record")
    query = _fragment("fragment_select_query", "query selection record")
    protected = _fragment("fragment_select_protected", "protected selection record")
    origins = {
        first.source_fragment_id: [
            CandidateOrigin(
                kind=CandidateOriginKind.ORIGINAL_FTS,
                query_key="q0",
                rank=1,
            )
        ],
        query.source_fragment_id: [
            CandidateOrigin(
                kind=CandidateOriginKind.QUERY_CLOUD,
                query_key="q1",
                rank=1,
            )
        ],
        protected.source_fragment_id: [
            CandidateOrigin(
                kind=CandidateOriginKind.ORIGINAL_FTS,
                query_key="q0",
                rank=2,
            )
        ],
    }
    _, receipt = adaptive._bounded_candidate_selection(
        (first, query, protected),
        origins=origins,
        protected_fragment_ids=(protected.source_fragment_id,),
        query_cloud_reserve_per_query=1,
        capacity=2,
        scope=adaptive.CandidateSelectionScope.FINAL_UNION,
    )
    return receipt


def _query_forced_selection_receipt() -> adaptive.AdaptiveCandidateSelectionReceipt:
    first = _fragment("fragment_forced_first", "first forced record")
    second = _fragment("fragment_forced_second", "second forced record")
    query = _fragment("fragment_forced_query", "query forced record")
    _, receipt = adaptive._bounded_candidate_selection(
        (first, second, query),
        origins={
            query.source_fragment_id: [
                CandidateOrigin(
                    kind=CandidateOriginKind.QUERY_CLOUD,
                    query_key="q1",
                    rank=1,
                )
            ]
        },
        protected_fragment_ids=(),
        query_cloud_reserve_per_query=1,
        capacity=2,
        scope=adaptive.CandidateSelectionScope.FINAL_UNION,
    )
    return receipt


def _two_forced_selection_receipt() -> adaptive.AdaptiveCandidateSelectionReceipt:
    fragments = tuple(
        _fragment(f"fragment_double_{name}", f"double forced record {name}")
        for name in ("first", "second", "third", "fourth")
    )
    _, receipt = adaptive._bounded_candidate_selection(
        fragments,
        origins={},
        protected_fragment_ids=(
            fragments[2].source_fragment_id,
            fragments[3].source_fragment_id,
        ),
        query_cloud_reserve_per_query=0,
        capacity=2,
        scope=adaptive.CandidateSelectionScope.FINAL_UNION,
    )
    return receipt


def _rehash_selection_payload(payload: dict[str, Any]) -> None:
    scope = adaptive.CandidateSelectionScope(payload["scope"])
    for values_key, hash_key in (
        ("input_fragment_ids", "input_hash"),
        ("initial_fragment_ids", "initial_hash"),
        ("protected_output_fragment_ids", "protected_output_hash"),
        ("output_fragment_ids", "output_hash"),
    ):
        payload[hash_key] = adaptive._candidate_sequence_hash(
            scope,
            tuple(payload[values_key]),
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("source_id", "source_fragment_id must use"),
        ("evicted_id", "evicted_fragment_id must use"),
        ("empty_reasons", "reasons must be non-empty and unique"),
        ("duplicate_reasons", "reasons must be non-empty and unique"),
        ("query_without_origins", "must close over query origins"),
        ("origins_without_query", "must close over query origins"),
        ("non_query_origin", "must be ranked QueryCloud origins"),
        ("unranked_query_origin", "must be ranked QueryCloud origins"),
        ("duplicate_query_origins", "query origins must be unique"),
        ("forced_without_eviction", "requires one evicted fragment"),
        ("visible_with_eviction", "only forced selection may name"),
    ),
)
def test_selection_decision_rejects_ambiguous_replay_state(
    tamper: str,
    message: str,
) -> None:
    payload: dict[str, Any] = {
        "source_fragment_id": "fragment_selected",
        "reasons": [CandidateSelectionReason.QUERY_CLOUD_RESERVE.value],
        "query_origins": [
            {
                "kind": CandidateOriginKind.QUERY_CLOUD.value,
                "query_key": "q1",
                "rank": 1,
            }
        ],
        "disposition": adaptive.CandidateSelectionDisposition.ALREADY_VISIBLE.value,
        "evicted_fragment_id": None,
    }
    if tamper == "source_id":
        payload["source_fragment_id"] = "invalid"
    elif tamper == "evicted_id":
        payload["disposition"] = adaptive.CandidateSelectionDisposition.FORCED.value
        payload["evicted_fragment_id"] = "invalid"
    elif tamper == "empty_reasons":
        payload["reasons"] = []
        payload["query_origins"] = []
    elif tamper == "duplicate_reasons":
        payload["reasons"] *= 2
    elif tamper == "query_without_origins":
        payload["query_origins"] = []
    elif tamper == "origins_without_query":
        payload["reasons"] = [CandidateSelectionReason.PRIOR_GATE_MATCH.value]
    elif tamper == "non_query_origin":
        payload["query_origins"][0]["kind"] = CandidateOriginKind.ORIGINAL_FTS.value
    elif tamper == "unranked_query_origin":
        payload["query_origins"][0]["rank"] = None
    elif tamper == "duplicate_query_origins":
        payload["query_origins"] *= 2
    elif tamper == "forced_without_eviction":
        payload["disposition"] = adaptive.CandidateSelectionDisposition.FORCED.value
    elif tamper == "visible_with_eviction":
        payload["evicted_fragment_id"] = "fragment_evicted"
    else:  # pragma: no cover - parameter list is the closed tamper set
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises(ValidationError, match=message):
        adaptive.AdaptiveCandidateSelectionDecision.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("duplicate_input", "selection input must be unique"),
        ("invalid_input", "selection input must use"),
        ("short_output", "output must fill the bounded capacity"),
        ("wrong_initial", "initial output must be the bounded input prefix"),
        (
            "short_protected_output",
            "protected selection output must preserve bounded size",
        ),
        ("outside_output", "selection output must be a subset"),
        ("protected_not_output", "prior Gate matches must remain selected"),
        ("protected_not_phase", "must survive the protected phase"),
        ("input_hash", "selection input hash does not match"),
        ("initial_hash", "selection initial hash does not match"),
        ("protected_hash", "selection protected-output hash does not match"),
        ("duplicate_decision", "selection decisions must name unique"),
        ("outside_decision", "selection decisions must refer to input"),
        ("wrong_prior_decisions", "prior-match decisions must exactly match"),
        ("visible_list", "already-visible list must match"),
        ("forced_list", "forced list must match"),
        ("skipped_list", "capacity-skipped list must match"),
        ("evicted_list", "evicted list must match"),
        ("prior_after_query", "prior-match decisions must precede"),
        ("protected_replay", "protected selection output does not replay"),
        ("output_replay", "selection output does not replay"),
        ("query_reserve", "exceeds the per-query QueryCloud reserve"),
    ),
)
def test_selection_receipt_rejects_non_replayable_tampering(
    tamper: str,
    message: str,
) -> None:
    payload = deepcopy(_selection_receipt_fixture().model_dump(mode="json"))
    first, query, protected = payload["input_fragment_ids"]
    if tamper == "duplicate_input":
        payload["input_fragment_ids"][1] = first
    elif tamper == "invalid_input":
        payload["input_fragment_ids"][0] = "invalid"
    elif tamper == "short_output":
        payload["output_fragment_ids"] = [query]
    elif tamper == "wrong_initial":
        payload["initial_fragment_ids"] = [query, first]
    elif tamper == "short_protected_output":
        payload["protected_output_fragment_ids"] = [protected]
    elif tamper == "outside_output":
        payload["output_fragment_ids"][0] = "fragment_outside"
    elif tamper == "protected_not_output":
        payload["protected_fragment_ids"] = [first]
    elif tamper == "protected_not_phase":
        payload["protected_output_fragment_ids"] = [first, query]
    elif tamper == "input_hash":
        payload["input_hash"] = HASH_A
    elif tamper == "initial_hash":
        payload["initial_hash"] = HASH_A
    elif tamper == "protected_hash":
        payload["protected_output_hash"] = HASH_A
    elif tamper == "duplicate_decision":
        payload["decisions"].append(deepcopy(payload["decisions"][0]))
    elif tamper == "outside_decision":
        payload["decisions"][1]["source_fragment_id"] = "fragment_outside"
    elif tamper == "wrong_prior_decisions":
        payload["protected_fragment_ids"] = []
    elif tamper == "visible_list":
        payload["already_visible_fragment_ids"] = []
    elif tamper == "forced_list":
        payload["forced_fragment_ids"] = []
    elif tamper == "skipped_list":
        payload["skipped_capacity_fragment_ids"] = [first]
    elif tamper == "evicted_list":
        payload["evicted_fragment_ids"] = []
    elif tamper == "prior_after_query":
        payload["decisions"].reverse()
    elif tamper == "protected_replay":
        payload["protected_output_fragment_ids"] = [protected, query]
        _rehash_selection_payload(payload)
    elif tamper == "output_replay":
        payload["output_fragment_ids"] = [protected, query]
        _rehash_selection_payload(payload)
    elif tamper == "query_reserve":
        payload["decisions"][0]["reasons"].append(
            CandidateSelectionReason.QUERY_CLOUD_RESERVE.value
        )
        payload["decisions"][0]["query_origins"] = [
            {
                "kind": CandidateOriginKind.QUERY_CLOUD.value,
                "query_key": "q1",
                "rank": 2,
            }
        ]
    else:  # pragma: no cover - parameter list is the closed tamper set
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises(ValidationError, match=message):
        adaptive.AdaptiveCandidateSelectionReceipt.model_validate_json(json.dumps(payload))


def test_selection_receipt_rejects_invalid_dispositions_and_evictions() -> None:
    payload = deepcopy(_selection_receipt_fixture().model_dump(mode="json"))
    first = payload["input_fragment_ids"][0]
    payload["decisions"][1]["source_fragment_id"] = first
    payload["already_visible_fragment_ids"] = [first]
    with pytest.raises(ValidationError, match="already-visible selection must remain"):
        adaptive.AdaptiveCandidateSelectionReceipt.model_validate_json(json.dumps(payload))

    payload = deepcopy(_query_forced_selection_receipt().model_dump(mode="json"))
    first = payload["input_fragment_ids"][0]
    payload["decisions"][0]["source_fragment_id"] = first
    payload["forced_fragment_ids"] = [first]
    with pytest.raises(ValidationError, match="forced selection must enter from outside"):
        adaptive.AdaptiveCandidateSelectionReceipt.model_validate_json(json.dumps(payload))

    payload = deepcopy(_query_forced_selection_receipt().model_dump(mode="json"))
    query = payload["decisions"][0]["source_fragment_id"]
    payload["decisions"][0]["disposition"] = (
        adaptive.CandidateSelectionDisposition.SKIPPED_CAPACITY.value
    )
    payload["decisions"][0]["evicted_fragment_id"] = None
    payload["forced_fragment_ids"] = []
    payload["skipped_capacity_fragment_ids"] = [query]
    payload["evicted_fragment_ids"] = []
    with pytest.raises(ValidationError, match="capacity-skipped selection must not enter"):
        adaptive.AdaptiveCandidateSelectionReceipt.model_validate_json(json.dumps(payload))

    payload = deepcopy(_two_forced_selection_receipt().model_dump(mode="json"))
    payload["decisions"][1]["evicted_fragment_id"] = payload["decisions"][0]["evicted_fragment_id"]
    with pytest.raises(ValidationError, match="must evict unique candidates"):
        adaptive.AdaptiveCandidateSelectionReceipt.model_validate_json(json.dumps(payload))

    payload = deepcopy(_two_forced_selection_receipt().model_dump(mode="json"))
    payload["decisions"][0]["evicted_fragment_id"] = "fragment_outside"
    with pytest.raises(ValidationError, match="may evict only initial candidates"):
        adaptive.AdaptiveCandidateSelectionReceipt.model_validate_json(json.dumps(payload))


def test_selection_receipt_hash_is_content_addressed() -> None:
    receipt = _selection_receipt_fixture()

    assert receipt.receipt_hash == receipt.receipt_hash


def test_candidate_trace_rejects_missing_duplicate_or_noncanonical_origins() -> None:
    payload: dict[str, Any] = {
        "source_fragment_id": "fragment_trace",
        "final_rank": 1,
        "harrier_score_hex": (0.5).hex(),
        "text_sha256": HASH_A,
        "source_family": "archive",
        "body_proof_eligible": True,
        "origins": [],
        "query_cloud_reserved": False,
    }
    with pytest.raises(ValidationError, match="origins must be non-empty and unique"):
        adaptive.AdaptiveCandidateTrace.model_validate_json(json.dumps(payload))

    original = {
        "kind": CandidateOriginKind.ORIGINAL_FTS.value,
        "query_key": "q0",
        "rank": 1,
    }
    payload["origins"] = [original, deepcopy(original)]
    with pytest.raises(ValidationError, match="origins must be non-empty and unique"):
        adaptive.AdaptiveCandidateTrace.model_validate_json(json.dumps(payload))

    payload["origins"] = [
        original,
        {
            "kind": CandidateOriginKind.EXACT_PHRASE.value,
            "query_key": "q0",
            "rank": None,
        },
    ]
    with pytest.raises(ValidationError, match="origins must use canonical order"):
        adaptive.AdaptiveCandidateTrace.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("item_id", "source_fragment_id must use"),
        ("duplicate_items", "Gate scan fragment IDs must be unique"),
        ("candidate_count", "candidate_count must equal item count"),
        ("character_count", "character_count must equal item characters"),
        ("duplicate_matches", "matched proof fragment IDs must be unique"),
        ("outside_match", "matched proof must be a subset"),
        ("matched_characters", "proof characters must equal"),
    ),
)
def test_gate_scan_receipt_rejects_non_replayable_counts_and_membership(
    tamper: str,
    message: str,
) -> None:
    result, _ = _audit_base_case()
    payload = deepcopy(result.stages[0].gate_scan.model_dump(mode="json"))
    if tamper == "item_id":
        payload["items"][0]["source_fragment_id"] = "invalid"
    elif tamper == "duplicate_items":
        payload["items"].append(deepcopy(payload["items"][0]))
        payload["candidate_count"] = 2
        payload["character_count"] *= 2
    elif tamper == "candidate_count":
        payload["candidate_count"] += 1
    elif tamper == "character_count":
        payload["character_count"] += 1
    elif tamper == "duplicate_matches":
        payload["matched_fragment_ids"] *= 2
    elif tamper == "outside_match":
        payload["matched_fragment_ids"] = ["fragment_outside"]
    elif tamper == "matched_characters":
        payload["matched_character_count"] += 1
    else:  # pragma: no cover - parameter list is the closed tamper set
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises(ValidationError, match=message):
        adaptive.AdaptiveGateScanReceipt.model_validate_json(json.dumps(payload))


def _selection_for_ids(
    fragment_ids: tuple[str, ...],
    *,
    capacity: int,
    scope: adaptive.CandidateSelectionScope,
    protected_fragment_ids: tuple[str, ...] = (),
) -> adaptive.AdaptiveCandidateSelectionReceipt:
    fragments = tuple(
        _fragment(fragment_id, f"selection replay text {index}")
        for index, fragment_id in enumerate(fragment_ids)
    )
    _, receipt = adaptive._bounded_candidate_selection(
        fragments,
        origins={},
        protected_fragment_ids=protected_fragment_ids,
        query_cloud_reserve_per_query=0,
        capacity=capacity,
        scope=scope,
    )
    return receipt


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("stage", "must be base, rescue, or query_cloud"),
        ("pool_scope", "pool selection must use Harrier-pool scope"),
        ("union_scope", "union selection must use final-union scope"),
        ("discovered_count", "discovered candidate count must match"),
        ("candidate_count", "candidate count must match scored Harrier pool"),
        ("union_input", "final-union input must close over"),
        ("scan_outside", "Gate scan must be a subset"),
        ("scan_rank", "Gate scan rank must match"),
        ("candidate_set_hash", "Gate scan must match embedded coverage"),
        ("matched_projection", "matched proof must equal all Gate-matched"),
    ),
)
def test_stage_receipt_rejects_cross_layer_replay_breaks(
    tamper: str,
    message: str,
) -> None:
    result, _ = _audit_base_case(two_candidates=tamper == "scan_rank")
    payload = deepcopy(result.stages[0].model_dump(mode="json"))
    if tamper == "stage":
        payload["stage"] = "unbounded"
    elif tamper == "pool_scope":
        payload["pool_selection"]["scope"] = adaptive.CandidateSelectionScope.FINAL_UNION.value
        _rehash_selection_payload(payload["pool_selection"])
    elif tamper == "union_scope":
        payload["union_selection"]["scope"] = adaptive.CandidateSelectionScope.HARRIER_POOL.value
        _rehash_selection_payload(payload["union_selection"])
    elif tamper == "discovered_count":
        payload["discovered_candidate_count"] += 1
    elif tamper == "candidate_count":
        payload["candidate_count"] += 1
    elif tamper == "union_input":
        payload["union_selection"] = _selection_for_ids(
            (),
            capacity=payload["union_selection"]["capacity"],
            scope=adaptive.CandidateSelectionScope.FINAL_UNION,
        ).model_dump(mode="json")
    elif tamper == "scan_outside":
        payload["gate_scan"]["items"][0]["source_fragment_id"] = "fragment_outside"
        payload["gate_scan"]["matched_fragment_ids"] = ["fragment_outside"]
    elif tamper == "scan_rank":
        payload["gate_scan"]["items"][0]["rank"] = 2
    elif tamper == "candidate_set_hash":
        payload["gate_scan"]["candidate_set_hash"] = HASH_A
    elif tamper == "matched_projection":
        payload["gate_scan"]["matched_fragment_ids"] = []
        payload["gate_scan"]["matched_character_count"] = 0
    else:  # pragma: no cover - parameter list is the closed tamper set
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises(ValidationError, match=message):
        adaptive.AdaptiveStageReceipt.model_validate_json(json.dumps(payload))


def test_stage_receipt_preserves_prior_gate_matches_across_bounded_layers() -> None:
    result = _audit_protected_result()
    payload = deepcopy(result.stages[-1].model_dump(mode="json"))
    protected_id = payload["pool_selection"]["protected_fragment_ids"][0]
    payload["gate_scan"]["matched_fragment_ids"] = [
        fragment_id
        for fragment_id in payload["gate_scan"]["matched_fragment_ids"]
        if fragment_id != protected_id
    ]
    payload["gate_scan"]["matched_character_count"] = sum(
        item["character_count"]
        for item in payload["gate_scan"]["items"]
        if item["source_fragment_id"] in payload["gate_scan"]["matched_fragment_ids"]
    )
    with pytest.raises(ValidationError, match="prior Gate matches must remain matched"):
        adaptive.AdaptiveStageReceipt.model_validate_json(json.dumps(payload))

    base_result, fragments = _audit_base_case()
    fragment = fragments[0]
    payload = deepcopy(base_result.stages[0].model_dump(mode="json"))
    payload["union_selection"] = _selection_for_ids(
        (fragment.source_fragment_id,),
        capacity=payload["union_selection"]["capacity"],
        scope=adaptive.CandidateSelectionScope.FINAL_UNION,
        protected_fragment_ids=(fragment.source_fragment_id,),
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="pool and union must protect the same"):
        adaptive.AdaptiveStageReceipt.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("no_stages", "requires at least one stage"),
        ("stage_order", "must follow base, rescue, QueryCloud order"),
        ("plan_without_stage", "stage and query-repair plan must exist together"),
        ("base_protected", "base stage cannot protect prior Gate matches"),
        ("final_coverage", "final stage coverage hash must match final coverage"),
        ("candidate_ids", "candidate traces must match final-union output"),
        ("rank_gap", "final candidate ranks must be contiguous"),
        ("proof_eligibility", "Gate scan must contain every proof-eligible"),
        ("reserve_flag", "reserve flags must match final-union receipt"),
        ("matched_ids", "final matched proof must match the Gate scan receipt"),
        ("matched_hash", "final matched proof hash does not match"),
        ("matched_rank", "matched proof rank must fit final candidate traces"),
        ("matched_trace", "matched proof must close over final candidate traces"),
    ),
)
def test_adaptive_result_rejects_forged_final_projection(
    tamper: str,
    message: str,
) -> None:
    result, fragments = _audit_base_case(two_candidates=tamper == "rank_gap")
    payload = deepcopy(result.model_dump(mode="json"))
    question = "Which audit record contains the decisive anchor?"
    if tamper == "no_stages":
        payload["stages"] = []
    elif tamper == "stage_order":
        payload["stages"].append(deepcopy(payload["stages"][0]))
    elif tamper == "plan_without_stage":
        payload["query_repair_plan"] = _query_plan(question).model_dump(mode="json")
    elif tamper == "base_protected":
        fragment_id = fragments[0].source_fragment_id
        for key, scope in (
            ("pool_selection", adaptive.CandidateSelectionScope.HARRIER_POOL),
            ("union_selection", adaptive.CandidateSelectionScope.FINAL_UNION),
        ):
            payload["stages"][0][key] = _selection_for_ids(
                (fragment_id,),
                capacity=payload["stages"][0][key]["capacity"],
                scope=scope,
                protected_fragment_ids=(fragment_id,),
            ).model_dump(mode="json")
    elif tamper == "final_coverage":
        payload["coverage"] = (
            EvidenceCoverageGate(_spec(question)).evaluate(()).model_dump(mode="json")
        )
    elif tamper == "candidate_ids":
        payload["candidates"] = []
    elif tamper == "rank_gap":
        payload["candidates"][1]["final_rank"] = 3
    elif tamper == "proof_eligibility":
        payload["candidates"][0]["body_proof_eligible"] = False
    elif tamper == "reserve_flag":
        payload["candidates"][0]["query_cloud_reserved"] = True
    elif tamper == "matched_ids":
        payload["matched_proof"] = []
    elif tamper == "matched_hash":
        payload["matched_proof"][0]["authority"] = "secondary"
    elif tamper in {"matched_rank", "matched_trace"}:
        if tamper == "matched_rank":
            payload["matched_proof"][0]["rank"] = 2
        else:
            text = payload["matched_proof"][0]["text"]
            payload["matched_proof"][0]["text"] = f"A{text[1:]}"
        proof = EvidenceCandidate.model_validate_json(json.dumps(payload["matched_proof"][0]))
        payload["stages"][-1]["gate_scan"]["matched_proof_hash"] = adaptive._matched_proof_hash(
            (proof,)
        )
    else:  # pragma: no cover - parameter list is the closed tamper set
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises(ValidationError, match=message):
        AdaptiveRecallResult.model_validate_json(json.dumps(payload))


def test_adaptive_result_rejects_dropped_protection_between_stages() -> None:
    result = _audit_protected_result()
    payload = deepcopy(result.model_dump(mode="json"))
    final_stage = payload["stages"][-1]
    pool_input_ids = tuple(final_stage["pool_selection"]["input_fragment_ids"])
    final_stage["pool_selection"] = _selection_for_ids(
        pool_input_ids,
        capacity=final_stage["pool_selection"]["capacity"],
        scope=adaptive.CandidateSelectionScope.HARRIER_POOL,
    ).model_dump(mode="json")
    desired_output_ids = tuple(final_stage["union_selection"]["output_fragment_ids"])
    union_input_ids = (
        *desired_output_ids,
        *(fragment_id for fragment_id in pool_input_ids if fragment_id not in desired_output_ids),
    )
    final_stage["union_selection"] = _selection_for_ids(
        union_input_ids,
        capacity=final_stage["union_selection"]["capacity"],
        scope=adaptive.CandidateSelectionScope.FINAL_UNION,
    ).model_dump(mode="json")

    with pytest.raises(ValidationError, match="stage pool must protect prior Gate matches"):
        AdaptiveRecallResult.model_validate_json(json.dumps(payload))


def test_existing_neighbor_origins_do_not_consume_the_novel_neighbor_budget() -> None:
    left = _fragment(
        "fragment_budget_left",
        "left budget record",
        version="version_budget",
        ordinal=0,
    )
    middle = _fragment(
        "fragment_budget_middle",
        "middle budget record",
        version="version_budget",
        ordinal=1,
    )
    right = _fragment(
        "fragment_budget_right",
        "right budget record",
        version="version_budget",
        ordinal=2,
    )
    pool = adaptive._CandidatePool(
        {item.source_fragment_id: item for item in (left, middle, right)}
    )
    original = CandidateOrigin(
        kind=CandidateOriginKind.ORIGINAL_FTS,
        query_key="q0",
        rank=1,
    )
    pool.add(left.source_fragment_id, original)
    pool.add(middle.source_fragment_id, original)

    pool.add_neighbors(max_new=1)

    assert tuple(pool.origins) == (
        left.source_fragment_id,
        middle.source_fragment_id,
        right.source_fragment_id,
    )
    assert (
        CandidateOrigin(
            kind=CandidateOriginKind.NEIGHBOR,
            query_key="q0",
        )
        in pool.origins[left.source_fragment_id]
    )


def test_boundary_projection_discovers_compound_identifiers_with_trace() -> None:
    question = "Where is Alpha2Beta?"
    proof = _fragment(
        "fragment_boundary_proof",
        "Alpha 2 Beta marks the decisive anchor.",
    )

    result = execute_adaptive_retrieval(
        question=question,
        fragments=(proof,),
        gate_spec=_spec(question),
        harrier=_StaticHarrier(),
        config=_config(base=1, rescue=1),
    )

    assert result.coverage.decision is EvidenceGateDecision.READY
    assert CandidateOriginKind.BOUNDARY_FTS in {
        origin.kind for origin in result.candidates[0].origins
    }
    assert len(result.stages[0].fts_result_hashes) == 2


def test_candidate_pool_fails_closed_at_the_discovery_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _fragment("fragment_discovery_first", "first discovery record")
    second = _fragment("fragment_discovery_second", "second discovery record")
    pool = adaptive._CandidatePool({item.source_fragment_id: item for item in (first, second)})
    origin = CandidateOrigin(
        kind=CandidateOriginKind.ORIGINAL_FTS,
        query_key="q0",
        rank=1,
    )
    monkeypatch.setattr(adaptive, "_MAX_DISCOVERED_CANDIDATES", 1)
    pool.add(first.source_fragment_id, origin)

    with pytest.raises(AdaptiveRecallError, match="discovery candidate budget exceeded"):
        pool.add(second.source_fragment_id, origin)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("duplicate_input", "input fragment IDs must be unique"),
        ("missing_protected", "prior Gate matches disappeared"),
        ("duplicate_protected", "prior Gate matches must be unique"),
        ("mandatory_over_capacity", "protected proof plus QueryCloud reserve exceed"),
    ),
)
def test_bounded_selection_rejects_lossy_or_ambiguous_inputs(
    tamper: str,
    message: str,
) -> None:
    fragments = tuple(
        _fragment(f"fragment_bound_{index}", f"bounded record {index}") for index in range(3)
    )
    ranked = fragments
    protected: tuple[str, ...] = ()
    origins: dict[str, list[CandidateOrigin]] = {}
    if tamper == "duplicate_input":
        ranked = (fragments[0], fragments[0])
    elif tamper == "missing_protected":
        protected = ("fragment_outside",)
    elif tamper == "duplicate_protected":
        protected = (fragments[0].source_fragment_id,) * 2
    elif tamper == "mandatory_over_capacity":
        protected = (fragments[0].source_fragment_id,)
        origins = {
            fragments[1].source_fragment_id: [
                CandidateOrigin(
                    kind=CandidateOriginKind.QUERY_CLOUD,
                    query_key="q1",
                    rank=1,
                )
            ],
            fragments[2].source_fragment_id: [
                CandidateOrigin(
                    kind=CandidateOriginKind.QUERY_CLOUD,
                    query_key="q2",
                    rank=1,
                )
            ],
        }
    else:  # pragma: no cover - parameter list is the closed tamper set
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises(AdaptiveRecallError, match=message):
        adaptive._bounded_candidate_selection(
            ranked,
            origins=origins,
            protected_fragment_ids=protected,
            query_cloud_reserve_per_query=1,
            capacity=2,
            scope=adaptive.CandidateSelectionScope.FINAL_UNION,
        )
