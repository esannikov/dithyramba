"""Bounded caller-assigned projections from CompactMemoryPacket 1.0/1.1."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest
from pydantic import ValidationError

from dithyramba.atlas import CompactMemoryPacket, CompactMemoryPacketV1_1
from dithyramba.connectors import (
    CONNECTOR_PACKET_PROFILE,
    CONNECTOR_PACKET_SCHEMA,
    CompactConnectorPacket,
    ConnectorPacketBudget,
    ConnectorPacketError,
    ConnectorPacketRequest,
    ConnectorPacketStatus,
    ConnectorRole,
    ConnectorRoleAssignment,
    project_compact_memory_packet,
)
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

_EXCERPTS = {
    "evidence_recipe": "Perform the documented procedure in three exact stages.",
    "evidence_precondition": "Begin only after the workspace is isolated.",
    "evidence_validation": "Validate the result against the frozen reference.",
    "evidence_contraindication": "Do not proceed when provenance is incomplete.",
    "evidence_extra": "Record a replay receipt after the final check.",
}


def _source_packet(schema_id: str = CompactMemoryPacket.SCHEMA) -> CompactMemoryPacket:
    sources = [
        {
            "source_id": "source_manual",
            "title": "Procedure manual",
            "creator": "Local editors",
            "source_kind": "institutional_record",
            "voice_kind": "institutional",
            "date_label": "2026",
            "publisher_or_archive": "Local archive",
            "independence_group": "family_manual",
            "verification_state": "verified",
            "original_url": None,
            "artifact_path": "artifacts/manual.md",
            "rights_note": "Local test fixture.",
        },
        {
            "source_id": "source_review",
            "title": "Independent review",
            "creator": "Review team",
            "source_kind": "scholarly_article",
            "voice_kind": "scholarly",
            "date_label": "2026",
            "publisher_or_archive": "Local archive",
            "independence_group": "family_review",
            "verification_state": "verified",
            "original_url": "https://example.test/review",
            "artifact_path": None,
            "rights_note": "Local test fixture.",
        },
    ]
    evidence = []
    for rank, (evidence_id, excerpt) in enumerate(_EXCERPTS.items(), start=1):
        source_id = "source_manual" if rank <= 2 else "source_review"
        evidence.append(
            {
                "evidence_id": evidence_id,
                "source_id": source_id,
                "locator": f"line {rank}",
                "excerpt": excerpt,
                "role": "supports" if evidence_id != "evidence_contraindication" else "refutes",
                "asserting_voice": "Test source",
                "limitation": f"Fixture limitation {rank}.",
                "source_fragment_id": f"fragment_{evidence_id.removeprefix('evidence_')}",
                "fragment_text_sha256": sha256(excerpt.encode()).hexdigest(),
                "source_address": {
                    "kind": "markdown",
                    "heading_path": ["Procedure"],
                    "line_start": rank,
                    "line_end": rank,
                    "char_start": 0,
                    "char_end": len(excerpt),
                },
            }
        )
    gaps = [
        {
            "gap_id": "gap_external_replication",
            "label": "External replication",
            "why_it_matters": "The procedure has only local validation.",
            "next_evidence": "An independent replay receipt.",
            "related_question_ids": ["question_procedure"],
            "related_hypothesis_ids": [],
        },
        {
            "gap_id": "gap_longitudinal_test",
            "label": "Longitudinal test",
            "why_it_matters": "Durability remains unmeasured.",
            "next_evidence": "A dated follow-up evaluation.",
            "related_question_ids": ["question_procedure"],
            "related_hypothesis_ids": [],
        },
    ]
    common: dict[str, object] = {
        "schema_id": schema_id,
        "router_profile": "test_connector_route_v1",
        "route_status": "routed",
        "query": "How should the procedure be executed and checked?",
        "query_terms": ["procedure", "checked"],
        "atlas_id": "atlas_connector_test",
        "case_id": "case_connector_test",
        "manifest_hash": "a" * 64,
        "sources": sources,
        "evidence": evidence,
        "questions": [
            {
                "question_id": "question_procedure",
                "prompt": "How should the procedure be executed and checked?",
                "short_answer": "Use the exact local procedure and validation evidence.",
                "state": "confirmed",
                "evidence_ids": list(_EXCERPTS),
                "gap": "External replication remains open.",
                "tags": ["procedure"],
            }
        ],
        "hypotheses": [],
        "relations": [],
        "timeline": [],
        "gaps": gaps,
    }
    if schema_id == CompactMemoryPacket.SCHEMA:
        common["route_matches"] = [
            {
                "object_kind": "question",
                "object_id": "question_procedure",
                "score": 10,
                "matched_terms": ["procedure"],
            }
        ]
        model: type[CompactMemoryPacket] = CompactMemoryPacket
    else:
        assert schema_id == CompactMemoryPacketV1_1.SCHEMA
        receipt_hash = "b" * 64
        common.update(
            {
                "route_matches": [],
                "route_receipt_id": f"route_candidate_receipt_{receipt_hash[:32]}",
                "route_receipt_hash": receipt_hash,
                "route_budget": {
                    "max_questions": 1,
                    "max_linked_hypotheses": 0,
                    "max_relations": 0,
                    "max_timeline_events": 0,
                    "max_gaps": 2,
                    "max_evidence": 5,
                    "max_sources": 2,
                },
                "route_trace": [
                    {
                        "stage": "seed",
                        "object_kind": "question",
                        "object_id": "question_procedure",
                        "channel": "lexical",
                        "channel_rank": 1,
                        "semantic_score_hex": None,
                        "lexical_score": 10,
                        "parent_object_kind": None,
                        "parent_object_id": None,
                    },
                    {
                        "stage": "graph_expansion",
                        "object_kind": "gap",
                        "object_id": "gap_external_replication",
                        "channel": None,
                        "channel_rank": None,
                        "semantic_score_hex": None,
                        "lexical_score": None,
                        "parent_object_kind": "question",
                        "parent_object_id": "question_procedure",
                    },
                    {
                        "stage": "graph_expansion",
                        "object_kind": "gap",
                        "object_id": "gap_longitudinal_test",
                        "channel": None,
                        "channel_rank": None,
                        "semantic_score_hex": None,
                        "lexical_score": None,
                        "parent_object_kind": "question",
                        "parent_object_id": "question_procedure",
                    },
                ],
                "omissions": [],
            }
        )
        model = CompactMemoryPacketV1_1
    raw = {
        **common,
        "packet_id": canonical_content_id("memory_packet", common),
        "packet_hash": canonical_sha256_hex(common),
    }
    return model.model_validate_json(json.dumps(raw))


def _complete_assignments(*, include_extra: bool = False) -> tuple[ConnectorRoleAssignment, ...]:
    recipe_ids = ("evidence_recipe", "evidence_extra") if include_extra else ("evidence_recipe",)
    return (
        ConnectorRoleAssignment(role=ConnectorRole.RECIPE, evidence_ids=recipe_ids),
        ConnectorRoleAssignment(
            role=ConnectorRole.PRECONDITION,
            evidence_ids=("evidence_precondition",),
        ),
        ConnectorRoleAssignment(
            role=ConnectorRole.VALIDATION,
            evidence_ids=("evidence_validation",),
        ),
        ConnectorRoleAssignment(
            role=ConnectorRole.CONTRAINDICATION,
            evidence_ids=("evidence_contraindication",),
        ),
        ConnectorRoleAssignment(
            role=ConnectorRole.GAP,
            gap_ids=("gap_external_replication",),
        ),
    )


def _request(
    *,
    assignments: tuple[ConnectorRoleAssignment, ...] | None = None,
    required_roles: tuple[ConnectorRole, ...] = tuple(ConnectorRole),
    budget: ConnectorPacketBudget | None = None,
) -> ConnectorPacketRequest:
    return ConnectorPacketRequest(
        consumer_id="maulstick",
        task_id="task_connector",
        required_roles=required_roles,
        assignments=_complete_assignments() if assignments is None else assignments,
        budget=budget or ConnectorPacketBudget(),
    )


@pytest.mark.parametrize(
    "source_schema",
    [CompactMemoryPacket.SCHEMA, CompactMemoryPacketV1_1.SCHEMA],
)
def test_complete_packet_preserves_source_closed_exact_provenance(source_schema: str) -> None:
    source = _source_packet(source_schema)

    packet = project_compact_memory_packet(source, _request())

    assert packet.schema_id == CONNECTOR_PACKET_SCHEMA
    assert packet.projection_profile == CONNECTOR_PACKET_PROFILE
    assert packet.status is ConnectorPacketStatus.COMPLETE
    assert packet.missing_roles == ()
    assert packet.source_packet_schema_id == source_schema
    assert (packet.source_packet_id, packet.source_packet_hash) == (
        source.packet_id,
        source.packet_hash,
    )
    assert packet.consumer_id == "maulstick"
    assert packet.task_id == "task_connector"
    assert [item.source_id for item in packet.sources] == ["source_manual", "source_review"]
    assert [item.connector_role for item in packet.evidence] == list(ConnectorRole)[:-1]
    assert packet.evidence[0].source_address is not None
    assert packet.evidence[0].source_fragment_id == "fragment_recipe"
    assert packet.evidence[0].limitation == "Fixture limitation 1."
    assert [item.gap_id for item in packet.gaps] == ["gap_external_replication"]
    assert packet.connector_packet_id == canonical_content_id(
        "connector_packet", packet.semantic_payload()
    )
    assert packet.connector_packet_hash == canonical_sha256_hex(packet.semantic_payload())


def test_legacy_compact_packet_refuses_unsigned_atlas_trace_spans() -> None:
    raw = _source_packet().model_dump(mode="json")
    raw["questions"][0]["trace_spans"] = [
        {
            "span_id": "trace_use",
            "start": 0,
            "end": 3,
            "text": "Use",
            "kind": "fact",
            "evidence_ids": ["evidence_recipe"],
        }
    ]

    with pytest.raises(ValidationError, match="cannot carry Atlas trace spans"):
        CompactMemoryPacket.model_validate_json(json.dumps(raw))


def test_partial_packet_reports_every_missing_required_role() -> None:
    request = _request(
        assignments=(
            ConnectorRoleAssignment(
                role=ConnectorRole.RECIPE,
                evidence_ids=("evidence_recipe",),
            ),
        )
    )

    packet = project_compact_memory_packet(_source_packet(), request)

    assert packet.status is ConnectorPacketStatus.PARTIAL
    assert packet.missing_roles == (
        ConnectorRole.PRECONDITION,
        ConnectorRole.VALIDATION,
        ConnectorRole.CONTRAINDICATION,
        ConnectorRole.GAP,
    )
    assert [item.source_id for item in packet.sources] == ["source_manual"]


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        (
            ConnectorRoleAssignment(
                role=ConnectorRole.RECIPE,
                evidence_ids=("evidence_absent",),
            ),
            "assigned evidence IDs are absent",
        ),
        (
            ConnectorRoleAssignment(role=ConnectorRole.GAP, gap_ids=("gap_absent",)),
            "assigned gap IDs are absent",
        ),
    ],
)
def test_projection_rejects_ids_absent_from_source_packet(
    assignment: ConnectorRoleAssignment,
    message: str,
) -> None:
    with pytest.raises(ConnectorPacketError, match=message):
        project_compact_memory_packet(
            _source_packet(),
            _request(assignments=(assignment,), required_roles=(assignment.role,)),
        )


def test_duplicate_and_conflicting_assignments_fail_closed() -> None:
    with pytest.raises(ValidationError, match="assignment evidence_ids must be unique"):
        ConnectorRoleAssignment(
            role=ConnectorRole.RECIPE,
            evidence_ids=("evidence_recipe", "evidence_recipe"),
        )

    duplicate_role = (
        ConnectorRoleAssignment(
            role=ConnectorRole.RECIPE,
            evidence_ids=("evidence_recipe",),
        ),
        ConnectorRoleAssignment(
            role=ConnectorRole.RECIPE,
            evidence_ids=("evidence_extra",),
        ),
    )
    with pytest.raises(ValidationError, match="assignment roles must be unique"):
        _request(assignments=duplicate_role)

    conflicting = (
        ConnectorRoleAssignment(
            role=ConnectorRole.RECIPE,
            evidence_ids=("evidence_recipe",),
        ),
        ConnectorRoleAssignment(
            role=ConnectorRole.VALIDATION,
            evidence_ids=("evidence_recipe",),
        ),
    )
    with pytest.raises(ValidationError, match="conflicting evidence assignment"):
        _request(assignments=conflicting)


def test_source_and_connector_hash_tampering_fail_closed() -> None:
    source = _source_packet()
    tampered_source = source.model_copy(update={"packet_hash": "0" * 64})
    with pytest.raises(ConnectorPacketError, match="content-address"):
        project_compact_memory_packet(tampered_source, _request())

    packet = project_compact_memory_packet(source, _request())
    raw = packet.model_dump(mode="json")
    raw["consumer_id"] = "different_consumer"
    with pytest.raises(ValidationError, match="connector_packet_id"):
        CompactConnectorPacket.model_validate_json(json.dumps(raw))

    raw = packet.model_dump(mode="json")
    raw["connector_packet_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="connector_packet_hash"):
        CompactConnectorPacket.model_validate_json(json.dumps(raw))


def test_projection_order_and_replay_are_deterministic() -> None:
    source = _source_packet(CompactMemoryPacketV1_1.SCHEMA)
    first_assignments = list(_complete_assignments(include_extra=True))
    first_gap = first_assignments[-1].model_copy(
        update={
            "gap_ids": ("gap_longitudinal_test", "gap_external_replication"),
        }
    )
    first_assignments[-1] = first_gap
    second_assignments = tuple(reversed(first_assignments))
    second_recipe = second_assignments[-1].model_copy(
        update={"evidence_ids": tuple(reversed(second_assignments[-1].evidence_ids))}
    )
    second_assignments = (*second_assignments[:-1], second_recipe)

    first = project_compact_memory_packet(
        source,
        _request(assignments=tuple(first_assignments)),
    )
    second = project_compact_memory_packet(
        source,
        _request(
            assignments=second_assignments,
            required_roles=tuple(reversed(tuple(ConnectorRole))),
        ),
    )

    assert first == second
    assert first.required_roles == tuple(ConnectorRole)
    assert [item.evidence_id for item in first.evidence[:2]] == [
        "evidence_extra",
        "evidence_recipe",
    ]
    assert [item.gap_id for item in first.gaps] == [
        "gap_external_replication",
        "gap_longitudinal_test",
    ]
    assert CompactConnectorPacket.model_validate_json(first.model_dump_json()) == first


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        (
            ConnectorPacketBudget(
                max_evidence=4,
                max_gaps=2,
                max_sources=2,
                max_excerpt_chars=10_000,
            ),
            "evidence count",
        ),
        (
            ConnectorPacketBudget(
                max_evidence=5,
                max_gaps=1,
                max_sources=2,
                max_excerpt_chars=10_000,
            ),
            "gap count",
        ),
        (
            ConnectorPacketBudget(
                max_evidence=5,
                max_gaps=2,
                max_sources=1,
                max_excerpt_chars=10_000,
            ),
            "source count",
        ),
        (
            ConnectorPacketBudget(
                max_evidence=5,
                max_gaps=2,
                max_sources=2,
                max_excerpt_chars=1,
            ),
            "character budget",
        ),
    ],
)
def test_projection_rejects_budget_overflow(
    budget: ConnectorPacketBudget,
    message: str,
) -> None:
    assignments = list(_complete_assignments(include_extra=True))
    assignments[-1] = assignments[-1].model_copy(
        update={
            "gap_ids": ("gap_external_replication", "gap_longitudinal_test"),
        }
    )

    with pytest.raises(ConnectorPacketError, match=message):
        project_compact_memory_packet(
            _source_packet(),
            _request(assignments=tuple(assignments), budget=budget),
        )


def test_contracts_are_strict_and_frozen() -> None:
    with pytest.raises(ValidationError):
        ConnectorPacketBudget(max_evidence="4")  # type: ignore[arg-type]

    packet = project_compact_memory_packet(_source_packet(), _request())
    with pytest.raises(ValidationError, match="frozen"):
        packet.status = ConnectorPacketStatus.PARTIAL


def test_assignment_shape_and_required_role_duplicates_are_rejected() -> None:
    with pytest.raises(ValidationError, match="gap assignments require"):
        ConnectorRoleAssignment(
            role=ConnectorRole.GAP,
            evidence_ids=("evidence_recipe",),
        )
    with pytest.raises(ValidationError, match="non-gap assignments require"):
        ConnectorRoleAssignment(role=ConnectorRole.RECIPE)
    with pytest.raises(ValidationError, match="assignment gap_ids must be unique"):
        ConnectorRoleAssignment(
            role=ConnectorRole.GAP,
            gap_ids=("gap_external_replication", "gap_external_replication"),
        )
    with pytest.raises(ValidationError, match="required_roles must be unique"):
        _request(required_roles=(ConnectorRole.RECIPE, ConnectorRole.RECIPE))


def _maximal_packet() -> CompactConnectorPacket:
    assignments = list(_complete_assignments(include_extra=True))
    assignments[-1] = assignments[-1].model_copy(
        update={
            "gap_ids": ("gap_longitudinal_test", "gap_external_replication"),
        }
    )
    return project_compact_memory_packet(
        _source_packet(),
        _request(assignments=tuple(assignments)),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema_id must be"),
        ("profile", "projection_profile must be"),
        ("source_schema", "unsupported source"),
        ("required_order", "canonical role order"),
        ("source_order", "canonical source_id order"),
        ("evidence_order", "canonical role and evidence_id order"),
        ("gap_order", "canonical gap_id order"),
        ("missing_roles", "exactly cover unassigned"),
        ("status", "required-role completeness"),
        ("source_closure", "exactly close selected evidence"),
        ("evidence_budget", "evidence count exceeds"),
        ("excerpt_budget", "excerpts exceed"),
        ("duplicate_source", "source IDs must be unique"),
        ("duplicate_evidence", "evidence IDs must be unique"),
        ("duplicate_gap", "gap IDs must be unique"),
    ],
)
def test_serialized_packet_contract_rejects_noncanonical_or_open_shapes(
    mutation: str,
    message: str,
) -> None:
    raw = _maximal_packet().model_dump(mode="json")
    if mutation == "schema":
        raw["schema_id"] = "dithyramba.compact_connector_packet/9.9"
    elif mutation == "profile":
        raw["projection_profile"] = "inferred_roles"
    elif mutation == "source_schema":
        raw["source_packet_schema_id"] = "dithyramba.compact_memory_packet/9.9"
    elif mutation == "required_order":
        raw["required_roles"] = list(reversed(raw["required_roles"]))
    elif mutation == "source_order":
        raw["sources"] = list(reversed(raw["sources"]))
    elif mutation == "evidence_order":
        raw["evidence"][0], raw["evidence"][1] = raw["evidence"][1], raw["evidence"][0]
    elif mutation == "gap_order":
        raw["gaps"] = list(reversed(raw["gaps"]))
    elif mutation == "missing_roles":
        raw["missing_roles"] = ["gap"]
    elif mutation == "status":
        raw["status"] = "partial"
    elif mutation == "source_closure":
        raw["sources"] = raw["sources"][:1]
    elif mutation == "evidence_budget":
        raw["budget"]["max_evidence"] = 4
    elif mutation == "excerpt_budget":
        raw["budget"]["max_excerpt_chars"] = 1
    elif mutation == "duplicate_source":
        raw["sources"].append(raw["sources"][0])
    elif mutation == "duplicate_evidence":
        raw["evidence"].append(raw["evidence"][0])
    else:
        assert mutation == "duplicate_gap"
        raw["gaps"].append(raw["gaps"][0])

    with pytest.raises(ValidationError, match=message):
        CompactConnectorPacket.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("gap_evidence", "gap is not an evidence"),
        ("partial_binding", "exact evidence binding must be complete"),
        ("wrong_gap_role", "selected gaps must use"),
        ("line_range", "line range is reversed"),
        ("char_range", "character range is reversed"),
        ("source_url", "absolute HTTP"),
        ("artifact_path", "case-relative path"),
    ],
)
def test_nested_projection_contracts_reject_invalid_provenance(
    mutation: str,
    message: str,
) -> None:
    raw = _maximal_packet().model_dump(mode="json")
    if mutation == "gap_evidence":
        raw["evidence"][0]["connector_role"] = "gap"
    elif mutation == "partial_binding":
        raw["evidence"][0]["fragment_text_sha256"] = None
    elif mutation == "wrong_gap_role":
        raw["gaps"][0]["connector_role"] = "recipe"
    elif mutation == "line_range":
        raw["evidence"][0]["source_address"]["line_start"] = 2
        raw["evidence"][0]["source_address"]["line_end"] = 1
    elif mutation == "char_range":
        raw["evidence"][0]["source_address"]["char_start"] = 2
        raw["evidence"][0]["source_address"]["char_end"] = 1
    elif mutation == "source_url":
        raw["sources"][1]["original_url"] = "ftp://example.test/review"
    else:
        assert mutation == "artifact_path"
        raw["sources"][0]["artifact_path"] = "../manual.md"

    with pytest.raises(ValidationError, match=message):
        CompactConnectorPacket.model_validate_json(json.dumps(raw))


def test_projection_revalidates_boundary_objects_before_use() -> None:
    with pytest.raises(ConnectorPacketError, match="source_packet must be"):
        project_compact_memory_packet(object(), _request())  # type: ignore[arg-type]

    unsupported = _source_packet().model_copy(update={"schema_id": "unsupported"})
    with pytest.raises(ConnectorPacketError, match="unsupported schema"):
        project_compact_memory_packet(unsupported, _request())

    with pytest.raises(ConnectorPacketError, match="request must be"):
        project_compact_memory_packet(_source_packet(), object())  # type: ignore[arg-type]

    assignments = _complete_assignments()
    tampered_request = _request().model_copy(update={"assignments": (*assignments, assignments[0])})
    with pytest.raises(ConnectorPacketError, match="assignment validation"):
        project_compact_memory_packet(_source_packet(), tampered_request)


def test_schema_1_0_legacy_evidence_without_exact_binding_is_preserved() -> None:
    raw = _source_packet().model_dump(mode="json")
    raw["evidence"][0]["source_fragment_id"] = None
    raw["evidence"][0]["fragment_text_sha256"] = None
    raw["evidence"][0]["source_address"] = None
    semantic = {key: value for key, value in raw.items() if key not in {"packet_id", "packet_hash"}}
    raw["packet_id"] = canonical_content_id("memory_packet", semantic)
    raw["packet_hash"] = canonical_sha256_hex(semantic)
    source = CompactMemoryPacket.model_validate_json(json.dumps(raw))
    request = _request(
        assignments=(
            ConnectorRoleAssignment(
                role=ConnectorRole.RECIPE,
                evidence_ids=("evidence_recipe",),
            ),
        ),
        required_roles=(ConnectorRole.RECIPE,),
    )

    packet = project_compact_memory_packet(source, request)

    assert packet.status is ConnectorPacketStatus.COMPLETE
    assert packet.evidence[0].source_fragment_id is None
    assert packet.evidence[0].fragment_text_sha256 is None
    assert packet.evidence[0].source_address is None
