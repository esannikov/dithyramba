"""Typed Relation and path-receipt contract tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from dithyramba.access import QueryExclusions, RequestScope
from dithyramba.relations import (
    RelationContractError,
    RelationDirection,
    RelationNodeRef,
    RelationNodeType,
    RelationPath,
    RelationPathRequest,
    RelationPathStep,
    RelationProposal,
    RelationType,
)

HASH = "a" * 64


def _scope() -> RequestScope:
    return RequestScope(
        library_id="library_one",
        snapshot_hash=HASH,
        purpose="research",
        collection_ids=("collection_one",),
        exclusions=QueryExclusions(),
    )


def _nodes() -> tuple[RelationNodeRef, RelationNodeRef, RelationNodeRef]:
    return (
        RelationNodeRef(RelationNodeType.STATEMENT, "statement_one"),
        RelationNodeRef(RelationNodeType.CONCEPT, "concept_one"),
        RelationNodeRef(RelationNodeType.STATEMENT, "statement_two"),
    )


def test_relation_proposal_canonicalizes_evidence_without_flattening_type() -> None:
    first, second, _third = _nodes()
    proposal = RelationProposal(
        RelationType.DEFINES,
        first,
        second,
        ("evidence_two", "evidence_one"),
        "The statement defines the concept.",
        "human_import/1.0",
    )

    assert proposal.evidence_link_ids == ("evidence_one", "evidence_two")
    assert proposal.payload()["relation_type"] == "defines"
    assert proposal.content_hash == proposal.content_hash


def test_relation_grounding_projection_excludes_untrusted_free_text() -> None:
    first, second, _third = _nodes()
    first_proposal = RelationProposal(
        RelationType.DEFINES,
        first,
        second,
        ("evidence_two", "evidence_one"),
        "A reviewer-only annotation that is not policy-certified.",
        "human_import/1.0",
    )
    second_proposal = RelationProposal(
        RelationType.DEFINES,
        first,
        second,
        ("evidence_one", "evidence_two"),
        "Different arbitrary private-looking annotation.",
        "another_extractor/2.0",
    )

    first_projection = first_proposal.grounding_projection()
    second_projection = second_proposal.grounding_projection()

    assert first_proposal.content_hash != second_proposal.content_hash
    assert first_projection == second_projection
    assert first_projection.projection_hash == second_projection.projection_hash
    assert "reason_text" not in first_projection.payload()
    assert "extraction_method" not in first_projection.payload()
    assert first_proposal.reason_text not in str(first_projection.payload())


def test_path_request_and_path_are_deterministic_and_cycle_free() -> None:
    first, second, third = _nodes()
    request = RelationPathRequest(
        "snapshot_one",
        "policy_one",
        _scope(),
        (third, first),
        (RelationType.SUPPORTS, RelationType.DEFINES, RelationType.SUPPORTS),
        RelationDirection.BOTH,
        2,
        10,
    )
    step_one = RelationPathStep(
        "relation_one",
        RelationType.DEFINES,
        first,
        second,
        ("evidence_one",),
    )
    step_two = RelationPathStep(
        "relation_two",
        RelationType.SUPPORTS,
        second,
        third,
        ("evidence_two",),
    )
    path = RelationPath(0, request.seeds[0], (step_one, step_two))

    assert request.relation_types == (RelationType.DEFINES, RelationType.SUPPORTS)
    assert len(request.seeds) == 2
    assert path.terminal == third
    assert len(path.path_hash) == 64

    cycle = RelationPathStep(
        "relation_three",
        RelationType.SUPPORTS,
        third,
        first,
        ("evidence_three",),
    )
    with pytest.raises(RelationContractError, match="cycle"):
        RelationPath(0, first, (step_one, step_two, cycle))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RelationNodeRef(RelationNodeType.STATEMENT, "concept_wrong"),
        lambda: RelationProposal(
            RelationType.SUPPORTS,
            _nodes()[0],
            _nodes()[0],
            ("evidence_one",),
            "Reason",
            "human",
        ),
        lambda: RelationProposal(
            RelationType.SUPPORTS,
            _nodes()[0],
            _nodes()[1],
            (),
            "Reason",
            "human",
        ),
        lambda: RelationPathRequest(
            "snapshot_one",
            "policy_one",
            _scope(),
            (_nodes()[0],),
            (RelationType.SUPPORTS,),
            max_hops=4,
        ),
        lambda: replace(
            RelationPathStep(
                "relation_one",
                RelationType.SUPPORTS,
                _nodes()[0],
                _nodes()[1],
                ("evidence_one",),
            ),
            evidence_link_ids=(),
        ),
    ],
)
def test_malformed_relation_contracts_fail(factory: Callable[[], object]) -> None:
    with pytest.raises(RelationContractError):
        factory()
