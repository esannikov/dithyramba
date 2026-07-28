"""Pure quality-first expansion contracts over frozen P7 artifacts."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.recall import (
    CorpusLayer,
    ExpandedCandidateKind,
    ExpandedCandidateTrace,
    ExpandedContractError,
    ExpandedEvidencePacket,
    ExpandedPacketItem,
    ExpandedQueryRequest,
    ExpansionOmission,
    ExpansionOmissionReason,
    ExpansionPath,
    ExpansionReceipt,
    ExpansionSeedRank,
    GraphExpansionConfig,
    RelationPathReceiptRef,
)
from dithyramba.relations import RelationNodeType, RelationType

H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64
H5 = "5" * 64
H6 = "6" * 64
H7 = "7" * 64
H8 = "8" * 64
STRUCTURE_READ_ID = f"structure_read_{H7[:32]}"
RELATION_PATH_ID = f"relation_path_{H8[:32]}"


def _request(
    *,
    structure: bool = False,
    graph: GraphExpansionConfig | None = None,
) -> ExpandedQueryRequest:
    return ExpandedQueryRequest(
        base_request_id="hybrid_query_base",
        base_request_hash=H1,
        structure_generation_id="structure_generation_one" if structure else None,
        structure_generation_hash=H2 if structure else None,
        reranker_profile_hash=H3,
        graph=graph,
        max_final=20,
    )


def _fragment_trace(
    *,
    rank: int = 1,
    score: float = 0.25,
    fragment_id: str = "fragment_one",
    anchor_ranks: tuple[int, ...] = (1,),
) -> ExpandedCandidateTrace:
    return ExpandedCandidateTrace(
        rank=rank,
        candidate_kind=ExpandedCandidateKind.SOURCE_FRAGMENT,
        candidate_id=fragment_id,
        source_fragment_ids=(fragment_id,),
        source_ids=("source_one",),
        source_family_ids=("family_one",),
        corpus_layers=(CorpusLayer.PRIMARY,),
        anchor_atomic_ranks=anchor_ranks,
        text_sha256=H4,
        pair_token_count=42,
        score_hex=score.hex(),
    )


def _structure_trace(
    *,
    rank: int = 2,
    score: float = 0.5,
    anchor_ranks: tuple[int, ...] = (2,),
) -> ExpandedCandidateTrace:
    return ExpandedCandidateTrace(
        rank=rank,
        candidate_kind=ExpandedCandidateKind.STRUCTURE_UNIT,
        candidate_id="structure_unit_one",
        source_fragment_ids=("fragment_three", "fragment_two"),
        source_ids=("source_two",),
        source_family_ids=("family_two",),
        corpus_layers=(CorpusLayer.DERIVED,),
        anchor_atomic_ranks=anchor_ranks,
        text_sha256=H5,
        pair_token_count=8_192,
        score_hex=score.hex(),
    )


def _relation_trace(*, rank: int = 1) -> ExpandedCandidateTrace:
    return ExpandedCandidateTrace(
        rank=rank,
        candidate_kind=ExpandedCandidateKind.RELATION_NODE,
        candidate_id="concept_one",
        relation_node_type=RelationNodeType.CONCEPT,
        source_fragment_ids=("fragment_two", "fragment_one"),
        source_ids=("source_two", "source_one"),
        source_family_ids=("family_two", "family_one"),
        corpus_layers=(CorpusLayer.SYNTHESIS, CorpusLayer.PRIMARY),
        anchor_atomic_ranks=(2, 1),
        text_sha256=H6,
        pair_token_count=100,
        score_hex=(-0.125).hex(),
    )


def _fragment_path() -> ExpansionPath:
    trace = _fragment_trace()
    return ExpansionPath(
        candidate_kind=trace.candidate_kind,
        candidate_id=trace.candidate_id,
        relation_node_type=trace.relation_node_type,
        source_fragment_ids=trace.source_fragment_ids,
        source_ids=trace.source_ids,
        source_family_ids=trace.source_family_ids,
        corpus_layers=trace.corpus_layers,
        anchor_atomic_ranks=trace.anchor_atomic_ranks,
        text_sha256=trace.text_sha256,
    )


def _structure_path() -> ExpansionPath:
    trace = _structure_trace()
    return ExpansionPath(
        candidate_kind=trace.candidate_kind,
        candidate_id=trace.candidate_id,
        relation_node_type=trace.relation_node_type,
        source_fragment_ids=trace.source_fragment_ids,
        source_ids=trace.source_ids,
        source_family_ids=trace.source_family_ids,
        corpus_layers=trace.corpus_layers,
        anchor_atomic_ranks=trace.anchor_atomic_ranks,
        text_sha256=trace.text_sha256,
        structure_read_receipt_id=STRUCTURE_READ_ID,
        structure_read_receipt_hash=H7,
    )


def _relation_path() -> ExpansionPath:
    trace = _relation_trace()
    return ExpansionPath(
        candidate_kind=trace.candidate_kind,
        candidate_id=trace.candidate_id,
        relation_node_type=trace.relation_node_type,
        source_fragment_ids=trace.source_fragment_ids,
        source_ids=trace.source_ids,
        source_family_ids=trace.source_family_ids,
        corpus_layers=trace.corpus_layers,
        anchor_atomic_ranks=trace.anchor_atomic_ranks,
        text_sha256=trace.text_sha256,
        relation_path_receipts=(
            RelationPathReceiptRef(
                receipt_id=RELATION_PATH_ID,
                receipt_hash=H8,
            ),
        ),
    )


def _receipt() -> ExpansionReceipt:
    request = _request(structure=True)
    eligible = (_fragment_trace(), _structure_trace())
    reranked = (
        _structure_trace(rank=1, score=0.9),
        _fragment_trace(rank=2, score=0.4),
    )
    paths = (_fragment_path(), _structure_path())
    return ExpansionReceipt(
        expanded_request_id=request.request_id,
        expanded_request_hash=request.request_hash,
        base_request_id=request.base_request_id,
        base_request_hash=request.base_request_hash,
        base_retrieval_receipt_hash=H4,
        reranker_profile_hash=request.reranker_profile_hash,
        seed_ranks=(
            ExpansionSeedRank(source_fragment_id="fragment_two", atomic_rank=2),
            ExpansionSeedRank(source_fragment_id="fragment_one", atomic_rank=1),
        ),
        eligible=eligible,
        reranked=reranked,
        omissions=(
            ExpansionOmission(
                candidate_kind=ExpandedCandidateKind.RELATION_NODE,
                candidate_id="concept_missing",
                relation_node_type=RelationNodeType.CONCEPT,
                reason=ExpansionOmissionReason.UNGROUNDED,
            ),
        ),
        expansion_path_hashes=tuple(item.path_hash for item in paths),
        expansion_paths=paths,
        structure_read_receipt_hashes=(H7,),
        relation_path_receipt_hashes=(),
    )


def _packet_item(trace: ExpandedCandidateTrace) -> ExpandedPacketItem:
    return ExpandedPacketItem(**trace.model_dump())


def test_request_supports_no_expansion_and_is_content_addressed() -> None:
    request = _request()
    replay = _request()

    assert request.structure_enabled is False
    assert request.graph_enabled is False
    assert request.request_id.startswith("expanded_query_")
    assert request.request_id == replay.request_id
    assert request.request_hash == replay.request_hash
    assert request.canonical_bytes == replay.canonical_bytes
    assert request.semantic_payload() == {
        "schema": "dithyramba.expanded_query_request/1.0",
        "base_request_id": "hybrid_query_base",
        "base_request_hash": H1,
        "structure_generation_id": None,
        "structure_generation_hash": None,
        "reranker_profile_hash": H3,
        "graph": None,
        "max_final": 20,
        "result_format": "expanded_evidence_packet",
    }


def test_request_enables_only_explicit_complete_structure_or_graph_modes() -> None:
    graph = GraphExpansionConfig(
        relation_types=(RelationType.SUPPORTS, RelationType.CONTRADICTS),
        max_hops=3,
    )
    request = _request(structure=True, graph=graph)

    assert request.structure_enabled is True
    assert request.graph_enabled is True
    assert graph.relation_types == (RelationType.CONTRADICTS, RelationType.SUPPORTS)
    assert cast(dict[str, object], request.semantic_payload()["graph"])["max_hops"] == 3

    with pytest.raises((ValidationError, ExpandedContractError), match="supplied together"):
        ExpandedQueryRequest(
            **{
                **_request().model_dump(),
                "structure_generation_id": "structure_generation_one",
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="at least one"):
        GraphExpansionConfig(relation_types=(), max_hops=1)
    with pytest.raises((ValidationError, ExpandedContractError), match="unique"):
        GraphExpansionConfig(
            relation_types=(RelationType.SUPPORTS, RelationType.SUPPORTS),
            max_hops=1,
        )
    with pytest.raises(ValidationError):
        GraphExpansionConfig(relation_types=(RelationType.SUPPORTS,), max_hops=4)


@pytest.mark.parametrize(
    "update",
    [
        {"base_request_id": "wrong"},
        {"base_request_hash": "bad"},
        {"structure_generation_id": "structure_wrong"},
        {"reranker_profile_hash": "ABC"},
        {"max_final": 0},
        {"max_final": 101},
        {"graph": cast(Any, {})},
        {"unknown": True},
    ],
)
def test_request_rejects_malformed_or_ambient_identity(update: dict[str, object]) -> None:
    with pytest.raises((ValidationError, ExpandedContractError)):
        ExpandedQueryRequest(**{**_request().model_dump(), **update})


def test_candidate_kinds_enforce_exact_identity_prefixes() -> None:
    fragment = _fragment_trace()
    structure = _structure_trace()
    relation = _relation_trace()

    assert fragment.candidate_key == ("source_fragment", "", "fragment_one")
    assert structure.candidate_key == ("structure_unit", "", "structure_unit_one")
    assert relation.candidate_key == ("relation_node", "concept", "concept_one")

    with pytest.raises((ValidationError, ExpandedContractError), match="fragment_"):
        ExpandedCandidateTrace(**{**fragment.model_dump(), "candidate_id": "source_one"})
    with pytest.raises((ValidationError, ExpandedContractError), match="node type"):
        ExpandedCandidateTrace(
            **{
                **relation.model_dump(),
                "relation_node_type": None,
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="concept_meaning_"):
        ExpandedCandidateTrace(
            **{
                **relation.model_dump(),
                "candidate_id": "concept_one",
                "relation_node_type": RelationNodeType.CONCEPT_MEANING,
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="cannot declare"):
        ExpandedCandidateTrace(
            **{
                **fragment.model_dump(),
                "relation_node_type": RelationNodeType.CONCEPT,
            }
        )


def test_candidate_trace_is_canonical_text_free_and_exact() -> None:
    relation = _relation_trace()

    assert relation.source_fragment_ids == ("fragment_one", "fragment_two")
    assert relation.source_ids == ("source_one", "source_two")
    assert relation.source_family_ids == ("family_one", "family_two")
    assert relation.corpus_layers == (CorpusLayer.PRIMARY, CorpusLayer.SYNTHESIS)
    assert relation.anchor_atomic_ranks == (1, 2)
    assert relation.pair_token_count == 100
    assert relation.score_hex == (-0.125).hex()
    assert "text" not in relation.payload()
    assert relation.payload()["text_sha256"] == H6

    for forbidden in ("text", "label", "relation_reason", "reason_text"):
        with pytest.raises(ValidationError, match=forbidden):
            ExpandedCandidateTrace(
                **{
                    **relation.model_dump(),
                    forbidden: "forbidden",
                }
            )


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"source_fragment_ids": ("fragment_other",)}, "ground itself"),
        ({"source_ids": ("source_one", "source_two")}, "one Source"),
        ({"anchor_atomic_ranks": ()}, "anchor_atomic_ranks"),
        ({"anchor_atomic_ranks": (1, 1)}, "unique"),
        ({"pair_token_count": 0}, "greater than or equal"),
        ({"pair_token_count": 32_769}, "less than or equal"),
        ({"score_hex": "0.25"}, "float.hex"),
        ({"score_hex": "nan"}, "finite"),
        ({"text_sha256": "BAD"}, "hash"),
        ({"source_fragment_ids": ("fragment_one", "fragment_one")}, "unique"),
    ],
)
def test_candidate_trace_rejects_ambiguous_grounding_or_score(
    update: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((ValidationError, ExpandedContractError), match=match):
        ExpandedCandidateTrace(**{**_fragment_trace().model_dump(), **update})


def test_candidate_trace_rejects_invalid_score_syntax_layers_and_structure_scope() -> None:
    with pytest.raises((ValidationError, ExpandedContractError), match=r"float\.hex"):
        ExpandedCandidateTrace(
            **{
                **_fragment_trace().model_dump(),
                "score_hex": "not-a-hex-float",
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="corpus_layers"):
        ExpandedCandidateTrace(
            **{
                **_fragment_trace().model_dump(),
                "corpus_layers": (),
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="unique"):
        ExpandedCandidateTrace(
            **{
                **_fragment_trace().model_dump(),
                "corpus_layers": (CorpusLayer.PRIMARY, CorpusLayer.PRIMARY),
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="one Source"):
        ExpandedCandidateTrace(
            **{
                **_structure_trace().model_dump(),
                "source_ids": ("source_one", "source_two"),
            }
        )


def test_structure_candidate_enforces_structure_member_bound() -> None:
    fragments = tuple(f"fragment_{index}" for index in range(129))
    with pytest.raises((ValidationError, ExpandedContractError), match="at most 128"):
        ExpandedCandidateTrace(
            **{
                **_structure_trace().model_dump(),
                "source_fragment_ids": fragments,
            }
        )


def test_expansion_paths_are_text_free_content_addressed_and_kind_specific() -> None:
    fragment = _fragment_path()
    structure = _structure_path()
    relation = _relation_path()

    assert fragment.SCHEMA == "dithyramba.expansion_path/1.0"
    assert fragment.path_id.startswith("expansion_path_")
    assert len(fragment.path_hash) == 64
    assert fragment.canonical_bytes.startswith(b'{"anchor_atomic_ranks"')
    assert fragment.structure_read_receipt_id is None
    assert fragment.relation_path_receipts == ()
    assert structure.structure_read_receipt_id == STRUCTURE_READ_ID
    assert structure.structure_read_receipt_hash == H7
    assert relation.relation_path_receipts[0].receipt_id == RELATION_PATH_ID
    assert relation.relation_path_receipts[0].receipt_hash == H8
    assert "text" not in fragment.semantic_payload()
    assert "label" not in structure.semantic_payload()
    assert "reason_text" not in relation.semantic_payload()

    replay = ExpansionPath(**relation.model_dump())
    assert replay.path_hash == relation.path_hash
    assert replay.path_id == relation.path_id


def test_seed_and_anchor_ranks_cover_the_full_fused_candidate_budget() -> None:
    seed = ExpansionSeedRank(source_fragment_id="fragment_tail", atomic_rank=500)
    trace = _fragment_trace(
        rank=1,
        fragment_id="fragment_tail",
        anchor_ranks=(500,),
    )

    assert seed.atomic_rank == 500
    assert trace.anchor_atomic_ranks == (500,)
    with pytest.raises(ValidationError, match="less than or equal to 500"):
        ExpansionSeedRank(source_fragment_id="fragment_tail", atomic_rank=501)
    with pytest.raises((ValidationError, ExpandedContractError), match="1-500"):
        _fragment_trace(fragment_id="fragment_tail", anchor_ranks=(501,))


def test_source_fragment_path_cannot_claim_structure_or_relation_lineage() -> None:
    fragment = _fragment_path()
    with pytest.raises((ValidationError, ExpandedContractError), match="cannot claim"):
        ExpansionPath(
            **{
                **fragment.model_dump(),
                "structure_read_receipt_id": STRUCTURE_READ_ID,
                "structure_read_receipt_hash": H7,
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="cannot claim"):
        ExpansionPath(
            **{
                **fragment.model_dump(),
                "relation_path_receipts": _relation_path().relation_path_receipts,
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="ground itself"):
        ExpansionPath(
            **{
                **fragment.model_dump(),
                "source_fragment_ids": ("fragment_other",),
            }
        )


def test_structure_path_requires_one_exact_paired_read_receipt() -> None:
    structure = _structure_path()
    without_receipt = {
        **structure.model_dump(),
        "structure_read_receipt_id": None,
        "structure_read_receipt_hash": None,
    }
    with pytest.raises((ValidationError, ExpandedContractError), match="requires an exact"):
        ExpansionPath(**without_receipt)
    with pytest.raises((ValidationError, ExpandedContractError), match="supplied together"):
        ExpansionPath(
            **{
                **structure.model_dump(),
                "structure_read_receipt_hash": None,
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="cannot claim"):
        ExpansionPath(
            **{
                **structure.model_dump(),
                "relation_path_receipts": _relation_path().relation_path_receipts,
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="structure_read_"):
        ExpansionPath(
            **{
                **structure.model_dump(),
                "structure_read_receipt_id": RELATION_PATH_ID,
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="derived from"):
        ExpansionPath(
            **{
                **structure.model_dump(),
                "structure_read_receipt_id": "structure_read_wrong",
            }
        )


def test_relation_node_path_requires_exact_relation_receipt_references() -> None:
    relation = _relation_path()
    with pytest.raises((ValidationError, ExpandedContractError), match="requires exact"):
        ExpansionPath(**{**relation.model_dump(), "relation_path_receipts": ()})
    with pytest.raises((ValidationError, ExpandedContractError), match="cannot claim"):
        ExpansionPath(
            **{
                **relation.model_dump(),
                "structure_read_receipt_id": STRUCTURE_READ_ID,
                "structure_read_receipt_hash": H7,
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="relation_path_"):
        RelationPathReceiptRef(receipt_id="relation_one", receipt_hash=H8)
    with pytest.raises((ValidationError, ExpandedContractError), match="hash"):
        RelationPathReceiptRef(receipt_id=RELATION_PATH_ID, receipt_hash="bad")
    with pytest.raises((ValidationError, ExpandedContractError), match="derived from"):
        RelationPathReceiptRef(receipt_id="relation_path_wrong", receipt_hash=H8)
    reference = relation.relation_path_receipts[0]
    with pytest.raises((ValidationError, ExpandedContractError), match="IDs must be unique"):
        ExpansionPath(
            **{
                **relation.model_dump(),
                "relation_path_receipts": (reference, reference),
            }
        )


def test_expansion_path_forbids_free_form_explanations() -> None:
    for forbidden in ("text", "label", "reason", "reason_text", "relation_reason"):
        with pytest.raises(ValidationError, match=forbidden):
            ExpansionPath(
                **{
                    **_fragment_path().model_dump(),
                    forbidden: "forbidden",
                }
            )


def test_omissions_are_closed_and_carry_no_free_form_reason() -> None:
    path = _structure_path()
    omission = ExpansionOmission(
        candidate_kind=ExpandedCandidateKind.STRUCTURE_UNIT,
        candidate_id="structure_unit_large",
        reason=ExpansionOmissionReason.TOKEN_BUDGET_EXCEEDED,
        observed_token_count=8_193,
        expansion_path_hash=path.path_hash,
    )
    assert omission.payload() == {
        "candidate_kind": "structure_unit",
        "candidate_id": "structure_unit_large",
        "relation_node_type": None,
        "reason": "token_budget_exceeded",
        "observed_token_count": 8_193,
        "expansion_path_hash": path.path_hash,
    }
    with pytest.raises(ValidationError, match="relation_reason"):
        ExpansionOmission(
            candidate_kind=ExpandedCandidateKind.STRUCTURE_UNIT,
            candidate_id="structure_unit_large",
            reason=ExpansionOmissionReason.TOKEN_BUDGET_EXCEEDED,
            observed_token_count=8_193,
            expansion_path_hash=path.path_hash,
            relation_reason="too long",  # type: ignore[call-arg]
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="observed token"):
        ExpansionOmission(
            candidate_kind=ExpandedCandidateKind.STRUCTURE_UNIT,
            candidate_id="structure_unit_large",
            reason=ExpansionOmissionReason.TOKEN_BUDGET_EXCEEDED,
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="reserved"):
        ExpansionOmission(
            candidate_kind=ExpandedCandidateKind.STRUCTURE_UNIT,
            candidate_id="structure_unit_large",
            reason=ExpansionOmissionReason.UNSUPPORTED,
            observed_token_count=100,
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="ExpansionPath hash"):
        ExpansionOmission(
            candidate_kind=ExpandedCandidateKind.STRUCTURE_UNIT,
            candidate_id="structure_unit_large",
            reason=ExpansionOmissionReason.LIMIT_EXCEEDED,
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="cannot claim"):
        ExpansionOmission(
            candidate_kind=ExpandedCandidateKind.STRUCTURE_UNIT,
            candidate_id="structure_unit_large",
            reason=ExpansionOmissionReason.UNGROUNDED,
            expansion_path_hash=path.path_hash,
        )


def test_receipt_binds_exact_candidate_closure_and_provenance_hashes() -> None:
    receipt = _receipt()

    assert tuple(item.atomic_rank for item in receipt.seed_ranks) == (1, 2)
    assert receipt.expansion_path_hashes == tuple(
        item.path_hash for item in receipt.expansion_paths
    )
    assert receipt.receipt_id.startswith("expansion_receipt_")
    assert len(receipt.receipt_hash) == 64
    assert len(receipt.eligible_set_hash) == 64
    assert receipt.semantic_payload()["base_retrieval_receipt_hash"] == H4
    assert {item.candidate_id for item in receipt.eligible} == {
        item.candidate_id for item in receipt.reranked
    }
    assert "relation_reason" not in receipt.semantic_payload()


def test_receipt_closes_relation_candidate_to_relation_path_receipt() -> None:
    request = _request(
        graph=GraphExpansionConfig(
            relation_types=(RelationType.SUPPORTS,),
            max_hops=2,
        )
    )
    candidate = _relation_trace()
    path = _relation_path()
    receipt = ExpansionReceipt(
        expanded_request_id=request.request_id,
        expanded_request_hash=request.request_hash,
        base_request_id=request.base_request_id,
        base_request_hash=request.base_request_hash,
        base_retrieval_receipt_hash=H4,
        reranker_profile_hash=request.reranker_profile_hash,
        seed_ranks=(
            ExpansionSeedRank(source_fragment_id="fragment_one", atomic_rank=1),
            ExpansionSeedRank(source_fragment_id="fragment_two", atomic_rank=2),
        ),
        eligible=(candidate,),
        reranked=(candidate,),
        expansion_path_hashes=(path.path_hash,),
        expansion_paths=(path,),
        relation_path_receipt_hashes=(H8,),
    )

    assert receipt.expansion_paths == (path,)
    assert receipt.expansion_path_hashes == (path.path_hash,)
    assert receipt.relation_path_receipt_hashes == (H8,)
    assert receipt.structure_read_receipt_hashes == ()


def test_receipt_allows_empty_candidate_closure() -> None:
    request = _request()
    receipt = ExpansionReceipt(
        expanded_request_id=request.request_id,
        expanded_request_hash=request.request_hash,
        base_request_id=request.base_request_id,
        base_request_hash=request.base_request_hash,
        base_retrieval_receipt_hash=H4,
        reranker_profile_hash=request.reranker_profile_hash,
    )

    assert receipt.eligible == ()
    assert receipt.reranked == ()
    assert receipt.expansion_path_hashes == ()


def test_receipt_preserves_lineage_for_grounded_token_omissions() -> None:
    request = _request(structure=True)
    path = _structure_path()
    omission = ExpansionOmission(
        candidate_kind=ExpandedCandidateKind.STRUCTURE_UNIT,
        candidate_id=path.candidate_id,
        reason=ExpansionOmissionReason.TOKEN_BUDGET_EXCEEDED,
        observed_token_count=8_193,
        expansion_path_hash=path.path_hash,
    )
    receipt = ExpansionReceipt(
        expanded_request_id=request.request_id,
        expanded_request_hash=request.request_hash,
        base_request_id=request.base_request_id,
        base_request_hash=request.base_request_hash,
        base_retrieval_receipt_hash=H4,
        reranker_profile_hash=request.reranker_profile_hash,
        seed_ranks=(ExpansionSeedRank(source_fragment_id="fragment_two", atomic_rank=2),),
        omissions=(omission,),
        expansion_path_hashes=(path.path_hash,),
        expansion_paths=(path,),
        structure_read_receipt_hashes=(H7,),
    )

    assert receipt.eligible == ()
    assert receipt.omissions == (omission,)
    with pytest.raises((ValidationError, ExpandedContractError), match="exact ExpansionPath"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "omissions": (omission.model_copy(update={"expansion_path_hash": H1}),),
            }
        )


def test_receipt_requires_real_paths_for_every_eligible_candidate() -> None:
    receipt = _receipt()
    with pytest.raises((ValidationError, ExpandedContractError), match="exact eligible"):
        ExpansionReceipt(**{**receipt.model_dump(), "expansion_paths": ()})

    forged = "f" * 64
    with pytest.raises((ValidationError, ExpandedContractError), match="exact ExpansionPath"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "expansion_path_hashes": (
                    forged,
                    receipt.expansion_paths[1].path_hash,
                ),
            }
        )

    original = receipt.expansion_paths[0]
    changed = ExpansionPath(**{**original.model_dump(), "text_sha256": H1})
    replacement_paths = (changed, receipt.expansion_paths[1])
    with pytest.raises((ValidationError, ExpandedContractError), match="grounding"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "expansion_paths": replacement_paths,
                "expansion_path_hashes": tuple(item.path_hash for item in replacement_paths),
            }
        )


def test_receipt_requires_path_receipt_hash_unions_to_close_exactly() -> None:
    receipt = _receipt()
    with pytest.raises((ValidationError, ExpandedContractError), match="structure receipt"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "structure_read_receipt_hashes": (),
            }
        )

    request = _request()
    candidate = _relation_trace()
    path = _relation_path()
    relation_receipt = {
        "expanded_request_id": request.request_id,
        "expanded_request_hash": request.request_hash,
        "base_request_id": request.base_request_id,
        "base_request_hash": request.base_request_hash,
        "base_retrieval_receipt_hash": H4,
        "reranker_profile_hash": request.reranker_profile_hash,
        "seed_ranks": (
            ExpansionSeedRank(source_fragment_id="fragment_one", atomic_rank=1),
            ExpansionSeedRank(source_fragment_id="fragment_two", atomic_rank=2),
        ),
        "eligible": (candidate,),
        "reranked": (candidate,),
        "expansion_path_hashes": (path.path_hash,),
        "expansion_paths": (path,),
        "relation_path_receipt_hashes": (),
    }
    with pytest.raises((ValidationError, ExpandedContractError), match="relation receipt"):
        ExpansionReceipt.model_validate(relation_receipt)


def test_receipt_rejects_candidate_add_drop_or_provenance_drift() -> None:
    receipt = _receipt()
    with pytest.raises((ValidationError, ExpandedContractError), match="exact eligible"):
        ExpansionReceipt(**{**receipt.model_dump(), "reranked": receipt.reranked[:1]})

    changed = receipt.reranked[0].model_copy(update={"text_sha256": H8})
    with pytest.raises((ValidationError, ExpandedContractError), match="provenance"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "reranked": (changed, receipt.reranked[1]),
            }
        )

    noncontiguous = receipt.reranked[0].model_copy(update={"rank": 2})
    with pytest.raises((ValidationError, ExpandedContractError), match="contiguous"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "reranked": (noncontiguous, receipt.reranked[1]),
            }
        )


def test_receipt_rejects_unknown_anchors_duplicate_hashes_and_omission_overlap() -> None:
    receipt = _receipt()
    with pytest.raises((ValidationError, ExpandedContractError), match="anchor ranks"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "seed_ranks": receipt.seed_ranks[:1],
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="unique"):
        path_hash = receipt.expansion_paths[0].path_hash
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "expansion_path_hashes": (path_hash, path_hash),
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="also be omissions"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "omissions": (
                    ExpansionOmission(
                        candidate_kind=ExpandedCandidateKind.SOURCE_FRAGMENT,
                        candidate_id="fragment_one",
                        reason=ExpansionOmissionReason.UNSUPPORTED,
                    ),
                ),
            }
        )


def test_receipt_rejects_missing_seed_closure_and_duplicate_seed_or_omission_ids() -> None:
    receipt = _receipt()
    with pytest.raises((ValidationError, ExpandedContractError), match="require exact atomic"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "seed_ranks": (),
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="SourceFragment IDs"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "seed_ranks": (
                    ExpansionSeedRank(source_fragment_id="fragment_one", atomic_rank=1),
                    ExpansionSeedRank(source_fragment_id="fragment_one", atomic_rank=2),
                ),
            }
        )
    omission = ExpansionOmission(
        candidate_kind=ExpandedCandidateKind.RELATION_NODE,
        candidate_id="concept_missing",
        relation_node_type=RelationNodeType.CONCEPT,
        reason=ExpansionOmissionReason.UNGROUNDED,
    )
    with pytest.raises((ValidationError, ExpandedContractError), match="unique candidates"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "omissions": (omission, omission),
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="atomic ranks"):
        ExpansionReceipt(
            **{
                **receipt.model_dump(),
                "seed_ranks": (
                    ExpansionSeedRank(source_fragment_id="fragment_one", atomic_rank=1),
                    ExpansionSeedRank(source_fragment_id="fragment_two", atomic_rank=1),
                ),
            }
        )


def test_expanded_packet_is_v2_text_free_ranked_and_content_addressed() -> None:
    request = _request(structure=True)
    receipt = _receipt()
    items = tuple(_packet_item(item) for item in receipt.reranked)
    packet = ExpandedEvidencePacket(
        expanded_request_id=request.request_id,
        expanded_request_hash=request.request_hash,
        base_packet_id="hybrid_packet_base",
        base_packet_hash=H4,
        expansion_receipt_hash=receipt.receipt_hash,
        result_status="evidence_found",
        items=items,
    )

    assert packet.SCHEMA == "dithyramba.expanded_evidence_packet/2.0"
    assert packet.packet_id.startswith("expanded_packet_")
    assert len(packet.packet_hash) == 64
    payload_items = cast(list[dict[str, object]], packet.semantic_payload()["items"])
    assert [item["rank"] for item in payload_items] == [1, 2]
    assert all("text" not in item for item in payload_items)
    assert payload_items[0]["source_fragment_ids"] == ["fragment_three", "fragment_two"]

    replay = ExpandedEvidencePacket(**packet.model_dump())
    assert replay.packet_hash == packet.packet_hash
    assert replay.packet_id == packet.packet_id


def test_expanded_packet_closes_status_rank_identity_and_free_text() -> None:
    receipt = _receipt()
    item = _packet_item(receipt.reranked[0])
    base: dict[str, object] = {
        "expanded_request_id": _request().request_id,
        "expanded_request_hash": _request().request_hash,
        "base_packet_id": "hybrid_packet_base",
        "base_packet_hash": H4,
        "expansion_receipt_hash": receipt.receipt_hash,
        "result_status": "evidence_found",
        "items": (item,),
    }
    with pytest.raises((ValidationError, ExpandedContractError), match="result_status"):
        ExpandedEvidencePacket.model_validate({**base, "result_status": "no_evidence"})
    with pytest.raises((ValidationError, ExpandedContractError), match="contiguous"):
        ExpandedEvidencePacket.model_validate(
            {
                **base,
                "items": (item.model_copy(update={"rank": 2}),),
            }
        )
    with pytest.raises((ValidationError, ExpandedContractError), match="duplicate"):
        ExpandedEvidencePacket.model_validate(
            {
                **base,
                "items": (item, item.model_copy(update={"rank": 2})),
            }
        )
    with pytest.raises(ValidationError, match="text"):
        ExpandedEvidencePacket.model_validate({**base, "text": "forbidden"})

    empty = ExpandedEvidencePacket(
        expanded_request_id=_request().request_id,
        expanded_request_hash=_request().request_hash,
        base_packet_id="hybrid_packet_base",
        base_packet_hash=H4,
        expansion_receipt_hash=receipt.receipt_hash,
        result_status="no_evidence",
    )
    assert empty.items == ()
