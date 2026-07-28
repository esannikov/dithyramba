from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pydantic import ValidationError

from dithyramba.access import QueryExclusions
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex
from dithyramba.recall import (
    EvidencePacketResultContract,
    QueryContractError,
    QueryRequest,
    RetrievalBudget,
)


def _request(**changes: Any) -> QueryRequest:
    values: dict[str, Any] = {
        "question": "How is the central term framed?",
        "library_id": "library_main",
        "collection_ids": ("collection_research",),
        "corpus_snapshot_id": "snapshot_abc123",
        "access_policy_id": "policy_research",
        "purpose": "research",
        "exclusions": QueryExclusions(),
        "retrieval": RetrievalBudget(),
        "result_contract": EvidencePacketResultContract(),
        **changes,
    }
    return QueryRequest.model_validate(values)


def test_query_request_normalizes_question_and_canonicalizes_collections() -> None:
    request = _request(
        question="How is Cafe\u0301 framed?",
        collection_ids=("collection_z", "collection_a"),
        exclusions=QueryExclusions(
            source_ids=("source_z", "source_a"),
            source_family_ids=("family_z", "family_a"),
            source_fragment_ids=("fragment_z", "fragment_a"),
        ),
    )
    assert request.question == "How is Café framed?"
    assert request.collection_ids == ("collection_a", "collection_z")
    assert request.exclusions.source_ids == ("source_a", "source_z")
    assert request.request_hash == canonical_sha256_hex(request.semantic_payload())
    assert request.query_request_id == canonical_content_id("query", request.semantic_payload())


def test_semantically_identical_requests_are_byte_identical() -> None:
    first = _request(collection_ids=("collection_b", "collection_a"))
    second = _request(collection_ids=("collection_a", "collection_b"))
    assert first.canonical_bytes == second.canonical_bytes
    assert first.request_hash == second.request_hash
    assert first.query_request_id == second.query_request_id


@pytest.mark.parametrize(
    "question",
    ["", "   ", " padded", "padded ", "bad\x00question", "x" * 2_001],
)
def test_query_question_is_nonblank_unpadded_nfc_and_bounded(question: str) -> None:
    with pytest.raises((ValidationError, QueryContractError)):
        _request(question=question)


@pytest.mark.parametrize("purpose", ["", "Research", "bad purpose", "x" * 65])
def test_query_purpose_is_a_stable_slug(purpose: str) -> None:
    with pytest.raises((ValidationError, QueryContractError)):
        _request(purpose=purpose)


def test_query_collection_scope_is_sorted_unique_and_bounded() -> None:
    with pytest.raises(ValidationError, match="1-16"):
        _request(collection_ids=())
    with pytest.raises(ValidationError, match="unique"):
        _request(collection_ids=("collection_a", "collection_a"))
    with pytest.raises(ValidationError, match="1-16"):
        _request(collection_ids=tuple(f"collection_{index}" for index in range(17)))
    with pytest.raises(ValidationError, match="collection_"):
        _request(collection_ids=("wrong",))


def test_query_has_exactly_one_library_snapshot_policy_and_no_extension_fields() -> None:
    base = _request().model_dump()
    for extra in (
        {"library_ids": ("library_main", "library_other")},
        {"cross_library_grant_id": "grant_forbidden"},
        {"provider": "openai"},
        {"as_of": "2026-07-20T12:00:00Z"},
        {"embedding_model": "forbidden"},
    ):
        with pytest.raises(ValidationError, match="Extra inputs"):
            QueryRequest.model_validate({**base, **extra})
    with pytest.raises(ValidationError, match="library_"):
        _request(library_id="wrong")
    with pytest.raises(ValidationError, match="snapshot_"):
        _request(corpus_snapshot_id="wrong")
    with pytest.raises(ValidationError, match="policy_"):
        _request(access_policy_id="wrong")


@pytest.mark.parametrize(
    "changes",
    [
        {"library_id": "library_"},
        {"corpus_snapshot_id": "snapshot_"},
        {"access_policy_id": "policy_"},
        {"collection_ids": ("collection_",)},
        {"library_id": "library_a_"},
        {"library_id": "library_a__b"},
        {"exclusions": QueryExclusions(source_ids=("source_a__b",))},
        {"exclusions": QueryExclusions(source_family_ids=("family_a__b",))},
        {"exclusions": QueryExclusions(source_fragment_ids=("fragment_a_",))},
    ],
)
def test_query_prefixed_ids_require_canonical_nonempty_suffixes(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        _request(**changes)


def test_query_rejects_nested_contract_subclasses_with_overridable_payloads() -> None:
    class DerivedExclusions(QueryExclusions):
        pass

    class DerivedBudget(RetrievalBudget):
        pass

    class DerivedResultContract(EvidencePacketResultContract):
        pass

    for changes in (
        {"exclusions": DerivedExclusions()},
        {"retrieval": DerivedBudget()},
        {"result_contract": DerivedResultContract()},
    ):
        with pytest.raises(ValidationError, match=r"exact|explicit"):
            _request(**changes)


@pytest.mark.parametrize(
    "values",
    [
        {"max_candidates": 0},
        {"max_candidates": 501},
        {"max_source_fragments": 0},
        {"max_source_fragments": 101},
        {"max_candidates": 10, "max_source_fragments": 11},
        {"max_candidates": True},
        {"max_candidates": 10.0},
        {"profile": "hybrid_v1"},
    ],
)
def test_retrieval_budget_is_strict_bounded_and_fts_only(values: dict[str, Any]) -> None:
    with pytest.raises((ValidationError, QueryContractError)):
        RetrievalBudget.model_validate(values)


def test_result_contract_is_fixed_and_rejects_format_extensions() -> None:
    with pytest.raises(ValidationError):
        EvidencePacketResultContract.model_validate({"format": "answer"})
    with pytest.raises(ValidationError):
        EvidencePacketResultContract.model_validate({"require_source_addresses": False})
    with pytest.raises(ValidationError, match="Extra inputs"):
        EvidencePacketResultContract.model_validate({"include_urls": True})


def test_query_hash_domain_covers_every_semantic_request_field() -> None:
    base = _request()
    variants = (
        _request(question="A different question"),
        _request(collection_ids=("collection_other",)),
        _request(corpus_snapshot_id="snapshot_other"),
        _request(access_policy_id="policy_other"),
        _request(purpose="analysis"),
        _request(exclusions=QueryExclusions(source_ids=("source_excluded",))),
        _request(retrieval=RetrievalBudget(max_candidates=50, max_source_fragments=20)),
    )
    assert len({base.request_hash, *(item.request_hash for item in variants)}) == 8


def test_query_contract_is_frozen_and_run_time_provider_fields_are_absent() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="frozen"):
        request.question = "Changed"
    payload = request.semantic_payload()
    assert not {
        "processing_run_id",
        "created_at",
        "started_at",
        "provider",
        "model",
        "as_of",
    }.intersection(payload)
    assert b"processing_run" not in request.canonical_bytes
    assert all(field.init for field in dataclasses.fields(QueryExclusions))
