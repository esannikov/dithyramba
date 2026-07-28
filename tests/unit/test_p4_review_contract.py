"""Frozen VS0 ReviewDecision contract tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dithyramba.review import (
    ReviewAction,
    ReviewDecision,
    ReviewDecisionRequest,
    ReviewScope,
    ReviewTargetType,
)

HASH = "a" * 64


def _request(**updates: object) -> ReviewDecisionRequest:
    values: dict[str, object] = {
        "target_type": ReviewTargetType.EVIDENCE_PACKET,
        "target_id": "packet_alpha",
        "target_hash": HASH,
        "action": ReviewAction.ACCEPT,
        "reason": "The cited fragment supports the bounded interpretation.",
        "authority": "researcher:eugene",
        "scope": ReviewScope(collection_ids=("collection_phd",), use="research"),
    }
    values.update(updates)
    return ReviewDecisionRequest(**values)  # type: ignore[arg-type]


def test_scope_is_sorted_canonical_and_hashable() -> None:
    scope = ReviewScope(
        collection_ids=("collection_zeta", "collection_alpha"),
        use="research",
    )
    assert scope.collection_ids == ("collection_alpha", "collection_zeta")
    assert json.loads(scope.canonical_bytes) == {
        "schema": "dithyramba.review_scope/1.0",
        "collection_ids": ["collection_alpha", "collection_zeta"],
        "use": "research",
    }
    assert len(scope.scope_hash) == 64


@pytest.mark.parametrize(
    ("target_type", "target_id"),
    [
        (ReviewTargetType.EVIDENCE_PACKET, "packet_alpha"),
        (ReviewTargetType.PACKET_ITEM, "packet_item_alpha"),
        (ReviewTargetType.SOURCE_FRAGMENT, "fragment_alpha"),
    ],
)
def test_target_prefix_matches_target_type(target_type: ReviewTargetType, target_id: str) -> None:
    assert _request(target_type=target_type, target_id=target_id).target_id == target_id


def test_target_type_prefix_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="packet_"):
        _request(target_id="fragment_alpha")


def test_supersede_requires_prior_decision_and_only_supersede_can_reference_one() -> None:
    with pytest.raises(ValidationError, match="requires a prior"):
        _request(action=ReviewAction.SUPERSEDE)
    with pytest.raises(ValidationError, match="only supersede"):
        _request(supersedes_review_decision_id="review_prior")

    request = _request(
        action=ReviewAction.SUPERSEDE,
        supersedes_review_decision_id="review_prior",
    )
    assert request.supersedes_review_decision_id == "review_prior"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason", " padded"),
        ("reason", ""),
        ("authority", "authority\x00injection"),
        ("target_hash", "A" * 64),
    ],
)
def test_untrusted_review_fields_fail_closed(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        _request(**{field: value})


def test_review_decision_has_one_canonical_public_shape() -> None:
    request = _request()
    decision = ReviewDecision(
        review_decision_id="review_alpha",
        library_id="library_alpha",
        **request.model_dump(),
        created_at="2026-07-20T12:34:56.000000Z",
    )
    payload = json.loads(decision.canonical_bytes)
    assert payload["schema"] == "dithyramba.review_decision/1.0"
    assert payload["scope"]["schema"] == "dithyramba.review_scope/1.0"
    assert payload["target_hash"] == HASH


def test_contracts_are_strict_and_forbid_extension_fields() -> None:
    with pytest.raises(ValidationError):
        ReviewScope.model_validate({"collection_ids": ["collection_phd"], "use": "research"})
    with pytest.raises(ValidationError):
        ReviewDecisionRequest.model_validate({**_request().model_dump(), "global_validated": True})
