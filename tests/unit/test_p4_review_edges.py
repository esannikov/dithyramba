"""Adversarial reconstruction coverage for ReviewDecision storage contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest
from pydantic import ValidationError

from dithyramba.contracts import canonical_json_bytes
from dithyramba.persistence.review import (
    ReviewTarget,
    _query_from_storage,
    _strict_int,
    _strict_str,
    _validated_source_address_payload,
)
from dithyramba.recall import QueryRequest
from dithyramba.review import (
    ReviewAction,
    ReviewDecision,
    ReviewDecisionRequest,
    ReviewIntegrityError,
    ReviewScope,
    ReviewTargetType,
)

HASH = "a" * 64


def _query(*, collections: tuple[str, ...] = ("collection_alpha",)) -> QueryRequest:
    return QueryRequest(
        question="Which evidence is exact?",
        library_id="library_alpha",
        collection_ids=collections,
        corpus_snapshot_id="snapshot_alpha",
        access_policy_id="policy_alpha",
        purpose="research",
    )


def _stored_query(payload: dict[str, object], query: QueryRequest) -> QueryRequest:
    return _query_from_storage(
        query_request_id=query.query_request_id,
        request_json=canonical_json_bytes(payload).decode("utf-8"),
        request_hash=query.request_hash,
    )


def test_query_storage_reconstruction_accepts_only_the_exact_canonical_contract() -> None:
    query = _query()
    assert (
        _query_from_storage(
            query_request_id=query.query_request_id,
            request_json=query.canonical_bytes.decode("utf-8"),
            request_hash=query.request_hash,
        )
        == query
    )

    malformed: list[dict[str, object]] = []
    base = cast(dict[str, object], json.loads(query.canonical_bytes))
    for key, value in (
        ("schema", "dithyramba.query_request/9.9"),
        ("collection_ids", "collection_alpha"),
        ("collection_ids", ["collection_alpha", 7]),
        ("exclusions", []),
        ("exclusions", {"source_ids": [], "source_family_ids": []}),
        (
            "exclusions",
            {
                "source_ids": "source_alpha",
                "source_family_ids": [],
                "source_fragment_ids": [],
            },
        ),
        (
            "exclusions",
            {
                "source_ids": [7],
                "source_family_ids": [],
                "source_fragment_ids": [],
            },
        ),
        ("retrieval", []),
        ("result_contract", []),
        ("question", 7),
    ):
        candidate = dict(base)
        candidate[key] = value
        malformed.append(candidate)
    missing = dict(base)
    missing.pop("purpose")
    malformed.append(missing)

    for payload in malformed:
        with pytest.raises(ReviewIntegrityError, match="QueryRequest"):
            _stored_query(payload, query)


def test_query_storage_rejects_noncanonical_order_and_identity_or_hash_drift() -> None:
    query = _query(collections=("collection_alpha", "collection_zeta"))
    payload = cast(dict[str, object], json.loads(query.canonical_bytes))
    payload["collection_ids"] = ["collection_zeta", "collection_alpha"]
    with pytest.raises(ReviewIntegrityError, match="typed reconstruction"):
        _stored_query(payload, query)

    with pytest.raises(ReviewIntegrityError, match="identity/hash"):
        _query_from_storage(
            query_request_id="query_wrong",
            request_json=query.canonical_bytes.decode("utf-8"),
            request_hash=query.request_hash,
        )
    with pytest.raises(ReviewIntegrityError, match="identity/hash"):
        _query_from_storage(
            query_request_id=query.query_request_id,
            request_json=query.canonical_bytes.decode("utf-8"),
            request_hash="f" * 64,
        )


def test_strict_storage_scalars_reject_bool_bounds_and_non_text() -> None:
    assert _strict_int(0, "ordinal", minimum=0) == 0
    assert _strict_str("research", "use") == "research"
    for value in (True, -1, "1"):
        with pytest.raises(ReviewIntegrityError, match="ordinal"):
            _strict_int(value, "ordinal", minimum=0)
    with pytest.raises(ReviewIntegrityError, match="use"):
        _strict_str(7, "use")


def test_source_address_reconstruction_covers_markdown_and_pdf() -> None:
    markdown: dict[str, object] = {
        "schema": "dithyramba.source_address/1.0",
        "kind": "markdown",
        "heading_path": ["Evidence"],
        "line_start": 2,
        "line_end": 3,
        "char_start": 10,
        "char_end": 20,
    }
    pdf: dict[str, object] = {
        "schema": "dithyramba.source_address/1.0",
        "kind": "pdf",
        "page": 2,
        "bbox": ["0.000", "1.000", "20.000", "30.000"],
        "char_start": 0,
        "char_end": 10,
    }
    assert _validated_source_address_payload(markdown) == markdown
    assert _validated_source_address_payload(pdf) == pdf


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "markdown",
            "heading_path": [],
            "line_start": 1,
            "line_end": 1,
            "char_start": 0,
            "char_end": 1,
            "extra": True,
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "markdown",
            "heading_path": "Evidence",
            "line_start": 1,
            "line_end": 1,
            "char_start": 0,
            "char_end": 1,
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "markdown",
            "heading_path": [7],
            "line_start": 1,
            "line_end": 1,
            "char_start": 0,
            "char_end": 1,
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "markdown",
            "heading_path": [],
            "line_start": 0,
            "line_end": 1,
            "char_start": 0,
            "char_end": 1,
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "pdf",
            "page": 1,
            "bbox": ["0.000", "1.000", "2.000"],
            "char_start": 0,
            "char_end": 1,
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "pdf",
            "page": 1,
            "bbox": ["0.000", "1.000", "2.000", 3],
            "char_start": 0,
            "char_end": 1,
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "pdf",
            "page": 0,
            "bbox": ["0.000", "1.000", "2.000", "3.000"],
            "char_start": 0,
            "char_end": 1,
        },
        {"schema": "dithyramba.source_address/1.0", "kind": "video"},
    ],
)
def test_source_address_reconstruction_rejects_malformed_shapes(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ReviewIntegrityError, match="address"):
        _validated_source_address_payload(payload)


def test_source_address_reconstruction_rejects_schema_and_canonicalization_drift() -> None:
    payload: dict[str, object] = {
        "schema": "dithyramba.source_address/9.9",
        "kind": "markdown",
        "heading_path": [],
        "line_start": 1,
        "line_end": 1,
        "char_start": 0,
        "char_end": 1,
    }
    with pytest.raises(ReviewIntegrityError, match="typed reconstruction"):
        _validated_source_address_payload(payload)

    non_nfc = dict(payload)
    non_nfc["schema"] = "dithyramba.source_address/1.0"
    non_nfc["heading_path"] = ["Cafe\u0301"]
    with pytest.raises(ReviewIntegrityError, match="typed reconstruction"):
        _validated_source_address_payload(non_nfc)


def test_review_models_reject_scope_text_id_and_timestamp_edges() -> None:
    for ids in ((), tuple(f"collection_{index}" for index in range(17))):
        with pytest.raises(ValidationError, match="1-16"):
            ReviewScope(collection_ids=ids, use="research")
    with pytest.raises(ValidationError, match="unique"):
        ReviewScope(
            collection_ids=("collection_alpha", "collection_alpha"),
            use="research",
        )
    for use in ("Research", "", "a" * 65):
        with pytest.raises(ValidationError, match="use"):
            ReviewScope(collection_ids=("collection_alpha",), use=use)

    scope = ReviewScope(collection_ids=("collection_alpha",), use="research")
    base: dict[str, object] = {
        "target_type": ReviewTargetType.EVIDENCE_PACKET,
        "target_id": "packet_alpha",
        "target_hash": HASH,
        "action": ReviewAction.ACCEPT,
        "reason": "Exact evidence.",
        "authority": "researcher:eugene",
        "scope": scope,
    }
    for update in (
        {"target_id": ""},
        {"reason": "x" * 4_001},
        {"authority": "x" * 201},
        {"supersedes_review_decision_id": "invalid"},
    ):
        with pytest.raises(ValidationError):
            ReviewDecisionRequest(**{**base, **update})  # type: ignore[arg-type]

    request = ReviewDecisionRequest(**base)  # type: ignore[arg-type]
    decision_payload: dict[str, object] = {
        "review_decision_id": "review_alpha",
        "library_id": "library_alpha",
        **request.model_dump(),
        "created_at": "2026-07-20T12:34:56.000000Z",
    }
    with pytest.raises(ValidationError, match="created_at"):
        ReviewDecision(**{**decision_payload, "created_at": "2026-07-20"})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate("not-a-mapping")


def test_review_target_payload_and_immutability_are_explicit() -> None:
    target = ReviewTarget(
        target_type=ReviewTargetType.SOURCE_FRAGMENT,
        target_id="fragment_alpha",
        target_hash=HASH,
        collection_ids=("collection_alpha",),
    )
    assert target.payload() == {
        "target_type": "source_fragment",
        "target_id": "fragment_alpha",
        "target_hash": HASH,
        "collection_ids": ["collection_alpha"],
    }
    with pytest.raises(AttributeError):
        replace(target, target_hash=HASH).target_hash = "f" * 64  # type: ignore[misc]
