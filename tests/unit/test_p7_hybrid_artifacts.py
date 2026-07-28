"""P7 text-free receipt contracts and negative closure tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.contracts import canonical_json_bytes
from dithyramba.recall.hybrid_artifacts import (
    HybridAccessReceipt,
    HybridArtifactError,
    HybridCoverageReport,
    HybridReadReceipt,
    HybridReadReceiptItem,
)

HASH = "a" * 64
SNAPSHOT_ID = f"snapshot_{HASH[:32]}"


def _item(*, fragment: str = "fragment_one", order: int = 0) -> HybridReadReceiptItem:
    return HybridReadReceiptItem(
        source_fragment_id=fragment,
        source_version_id="source_version_one",
        read_order=order,
        text_sha256=HASH,
    )


def _read(*, items: tuple[HybridReadReceiptItem, ...] = ()) -> HybridReadReceipt:
    return HybridReadReceipt(
        request_id="hybrid_query_one",
        request_hash=HASH,
        retrieval_corpus_hash=HASH,
        items=items,
    )


def _access(**updates: object) -> HybridAccessReceipt:
    payload: dict[str, object] = {
        "library_id": "library_one",
        "request_id": "hybrid_query_one",
        "request_hash": HASH,
        "access_policy_id": "policy_one",
        "policy_hash": HASH,
        "corpus_snapshot_id": SNAPSHOT_ID,
        "snapshot_hash": HASH,
        "exclusion_hash": HASH,
        "permitted_set_hash": HASH,
        "retrieval_corpus_hash": HASH,
        "policy_omission_present": False,
    }
    payload.update(updates)
    return HybridAccessReceipt.model_validate(payload)


def test_read_receipt_is_sorted_content_addressed_and_empty_safe() -> None:
    receipt = _read(items=(_item(fragment="fragment_two", order=1), _item()))

    assert tuple(item.source_fragment_id for item in receipt.items) == (
        "fragment_one",
        "fragment_two",
    )
    assert receipt.receipt_id.startswith("hybrid_read_")
    assert len(receipt.receipt_hash) == 64
    assert receipt.canonical_bytes == canonical_json_bytes(receipt.semantic_payload())
    assert _read().items == ()


@pytest.mark.parametrize(
    "items",
    [
        (_item(), _item()),
        (_item(order=1),),
        (_item(order=0), _item(fragment="fragment_two", order=2)),
        cast(Any, (object(),)),
    ],
)
def test_read_receipt_rejects_duplicate_gap_and_invalid_items(
    items: tuple[HybridReadReceiptItem, ...],
) -> None:
    with pytest.raises((ValidationError, HybridArtifactError)):
        _read(items=items)


def test_access_receipt_binds_the_complete_policy_tuple() -> None:
    receipt = _access(policy_omission_present=True)

    assert receipt.receipt_id.startswith("hybrid_access_")
    assert receipt.semantic_payload()["policy_omission_present"] is True
    assert receipt.canonical_bytes == canonical_json_bytes(receipt.semantic_payload())


def test_access_receipt_rejects_snapshot_mismatch() -> None:
    with pytest.raises((ValidationError, HybridArtifactError), match="snapshot ID/hash"):
        _access(corpus_snapshot_id="snapshot_wrong")


def test_coverage_is_text_free_bounded_and_content_addressed() -> None:
    report = HybridCoverageReport(
        request_id="hybrid_query_one",
        request_hash=HASH,
        processed_count=2,
        policy_omission_present=True,
    )

    payload = report.semantic_payload()
    assert payload["policy_omission"] == {"present": True, "count": None}
    assert report.report_id.startswith("hybrid_coverage_")
    assert report.canonical_bytes == canonical_json_bytes(payload)
    assert "text" not in report.canonical_bytes.decode("utf-8")


@pytest.mark.parametrize(("skipped", "failed"), [(1, 0), (0, 1), (1, 1)])
def test_successful_coverage_rejects_hidden_failures(skipped: int, failed: int) -> None:
    with pytest.raises((ValidationError, HybridArtifactError), match="cannot hide"):
        HybridCoverageReport(
            request_id="hybrid_query_one",
            request_hash=HASH,
            processed_count=1,
            skipped_count=skipped,
            failed_count=failed,
        )


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (_read, "request_id", "query_one"),
        (_read, "request_hash", "A" * 64),
        (_access, "library_id", "wrong"),
        (_access, "access_policy_id", "wrong"),
        (_access, "policy_hash", "a" * 63),
    ],
)
def test_identifiers_and_hashes_fail_closed(
    factory: Any,
    field: str,
    value: str,
) -> None:
    if factory is _read:
        payload = {
            "request_id": "hybrid_query_one",
            "request_hash": HASH,
            "retrieval_corpus_hash": HASH,
        }
        payload[field] = value
        with pytest.raises((ValidationError, HybridArtifactError)):
            HybridReadReceipt.model_validate(payload)
    else:
        with pytest.raises((ValidationError, HybridArtifactError)):
            _access(**{field: value})


@pytest.mark.parametrize(
    "payload",
    [
        {"source_fragment_id": "wrong"},
        {"source_version_id": "wrong"},
        {"text_sha256": "A" * 64},
    ],
)
def test_read_item_rejects_invalid_identity(payload: dict[str, object]) -> None:
    values: dict[str, object] = {
        "source_fragment_id": "fragment_one",
        "source_version_id": "source_version_one",
        "read_order": 0,
        "text_sha256": HASH,
    }
    values.update(payload)
    with pytest.raises((ValidationError, HybridArtifactError)):
        HybridReadReceiptItem.model_validate(values)


def test_contracts_are_frozen_and_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HybridCoverageReport.model_validate(
            {
                "request_id": "hybrid_query_one",
                "request_hash": HASH,
                "processed_count": 0,
                "denied_ids": ["fragment_secret"],
            }
        )
    report = HybridCoverageReport(
        request_id="hybrid_query_one",
        request_hash=HASH,
        processed_count=0,
    )
    with pytest.raises(ValidationError):
        report.processed_count = 1
