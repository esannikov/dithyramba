from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)
from dithyramba.ingest import MarkdownSourceAddress, PdfSourceAddress
from dithyramba.recall import (
    AccessReceipt,
    ArtifactContractError,
    EvidenceFragment,
    EvidencePacket,
    OmissionCategory,
    OmissionDisclosure,
    PacketResultStatus,
    ReadReceipt,
    ReadReceiptItem,
    RecallCoverageReport,
    RecallOmission,
    RetrievalReceipt,
    RetrievalTraceItem,
)

QUERY_HASH = "1" * 64
SNAPSHOT_HASH = "2" * 64
QUERY_ID = f"query_{QUERY_HASH[:32]}"
SNAPSHOT_ID = f"snapshot_{SNAPSHOT_HASH[:32]}"
POLICY_HASH = "3" * 64
EXCLUSION_HASH = "4" * 64
PERMITTED_HASH = "5" * 64
CORPUS_HASH = "6" * 64
TEXT_A = "Exact source text"
TEXT_B = "Second source"


def _read_receipt(*, query_hash: str = QUERY_HASH, corpus_hash: str = CORPUS_HASH) -> ReadReceipt:
    return ReadReceipt(
        query_request_hash=query_hash,
        retrieval_corpus_hash=corpus_hash,
        items=(
            ReadReceiptItem(
                source_fragment_id="fragment_b",
                source_version_id="source_version_b",
                read_order=1,
                text_sha256=sha256_hex(TEXT_B.encode()),
            ),
            ReadReceiptItem(
                source_fragment_id="fragment_a",
                source_version_id="source_version_a",
                read_order=0,
                text_sha256=sha256_hex(TEXT_A.encode()),
            ),
        ),
    )


def _access_receipt(
    *,
    query_hash: str = QUERY_HASH,
    corpus_hash: str = CORPUS_HASH,
    snapshot_hash: str = SNAPSHOT_HASH,
    snapshot_id: str = SNAPSHOT_ID,
    library_id: str = "library_main",
    access_policy_id: str = "policy_research",
    policy_omission_present: bool = False,
) -> AccessReceipt:
    return AccessReceipt(
        library_id=library_id,
        query_request_hash=query_hash,
        access_policy_id=access_policy_id,
        policy_hash=POLICY_HASH,
        corpus_snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        exclusion_hash=EXCLUSION_HASH,
        permitted_set_hash=PERMITTED_HASH,
        retrieval_corpus_hash=corpus_hash,
        policy_omission_present=policy_omission_present,
    )


def _retrieval_receipt(
    *,
    query_hash: str = QUERY_HASH,
    corpus_hash: str = CORPUS_HASH,
    candidate_count: int = 5,
    selected_count: int = 1,
    max_candidates: int = 10,
    max_source_fragments: int = 2,
) -> RetrievalReceipt:
    return RetrievalReceipt(
        query_request_hash=query_hash,
        profile="fts_v1",
        profile_version="fts_v1.0",
        code_version="0.1.0.dev2",
        retrieval_corpus_hash=corpus_hash,
        max_candidates=max_candidates,
        max_source_fragments=max_source_fragments,
        candidate_count=candidate_count,
        selected_count=selected_count,
        trace=(
            RetrievalTraceItem(
                source_fragment_id="fragment_b",
                rank=2,
                score="-1.000000",
            ),
            RetrievalTraceItem(
                source_fragment_id="fragment_a",
                rank=1,
                score="-2.000000",
            ),
        ),
    )


def _coverage(
    *,
    query_hash: str = QUERY_HASH,
    processed_count: int = 2,
    policy_omission_present: bool = False,
    omissions: tuple[RecallOmission, ...] | None = None,
) -> RecallCoverageReport:
    canonical_omissions = (
        (
            RecallOmission(
                category=OmissionCategory.BUDGET,
                disclosure=OmissionDisclosure.COUNTED,
                count=3,
                reason_code="candidate_budget",
            ),
        )
        if omissions is None
        else omissions
    )
    return RecallCoverageReport(
        query_request_hash=query_hash,
        processed_count=processed_count,
        skipped_count=0,
        failed_count=0,
        policy_omission_present=policy_omission_present,
        omissions=canonical_omissions,
    )


def _markdown_address(text: str = TEXT_A) -> dict[str, object]:
    return {
        "schema": "dithyramba.source_address/1.0",
        "kind": "markdown",
        "heading_path": ["Section"],
        "line_start": 10,
        "line_end": 10,
        "char_start": 100,
        "char_end": 100 + len(text),
    }


def _evidence_fragment(**changes: Any) -> EvidenceFragment:
    values: dict[str, Any] = {
        "source_fragment_id": "fragment_a",
        "source_version_id": "source_version_a",
        "source_family_id": "family_a",
        "rank": 1,
        "score": "-2.000000",
        "text": TEXT_A,
        "text_sha256": sha256_hex(TEXT_A.encode()),
        "source_address": _markdown_address(),
        **changes,
    }
    return EvidenceFragment.model_validate(values)


def _packet(**changes: Any) -> EvidencePacket:
    values: dict[str, Any] = {
        "query_request_id": QUERY_ID,
        "query_request_hash": QUERY_HASH,
        "corpus_snapshot_id": SNAPSHOT_ID,
        "corpus_snapshot_hash": SNAPSHOT_HASH,
        "result_status": PacketResultStatus.EVIDENCE_FOUND,
        "source_fragments": (_evidence_fragment(),),
        "coverage_report": _coverage(),
        "read_receipt": _read_receipt(),
        "access_receipt": _access_receipt(),
        "retrieval_receipt": _retrieval_receipt(),
        **changes,
    }
    return EvidencePacket.model_validate(values)


def test_receipts_are_content_addressed_and_read_items_are_order_canonical() -> None:
    read = _read_receipt()
    assert [item.source_fragment_id for item in read.items] == ["fragment_a", "fragment_b"]
    assert read.receipt_hash == canonical_sha256_hex(read.semantic_payload())
    assert read.read_receipt_id == canonical_content_id("read", read.semantic_payload())
    assert read.canonical_bytes == canonical_json_bytes(read.semantic_payload())

    access = _access_receipt()
    assert access.receipt_hash == canonical_sha256_hex(access.semantic_payload())
    assert access.access_receipt_id == canonical_content_id("access", access.semantic_payload())
    assert access.canonical_bytes == canonical_json_bytes(access.semantic_payload())

    retrieval = _retrieval_receipt()
    assert [item.source_fragment_id for item in retrieval.trace] == [
        "fragment_a",
        "fragment_b",
    ]
    assert retrieval.receipt_hash == canonical_sha256_hex(retrieval.semantic_payload())
    assert retrieval.retrieval_receipt_id == canonical_content_id(
        "retrieval", retrieval.semantic_payload()
    )
    assert retrieval.canonical_bytes == canonical_json_bytes(retrieval.semantic_payload())


def test_read_receipt_rejects_duplicate_fragments_and_noncontiguous_orders() -> None:
    item = ReadReceiptItem(
        source_fragment_id="fragment_a",
        source_version_id="source_version_a",
        read_order=0,
        text_sha256=sha256_hex(TEXT_A.encode()),
    )
    duplicate = item.model_copy(update={"read_order": 1})
    with pytest.raises(ValidationError, match="unique"):
        ReadReceipt(
            query_request_hash=QUERY_HASH,
            retrieval_corpus_hash=CORPUS_HASH,
            items=(item, duplicate),
        )
    with pytest.raises(ValidationError, match="contiguous"):
        ReadReceipt(
            query_request_hash=QUERY_HASH,
            retrieval_corpus_hash=CORPUS_HASH,
            items=(item.model_copy(update={"read_order": 1}),),
        )


def test_artifact_prefixed_ids_require_canonical_nonempty_suffixes() -> None:
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        ReadReceiptItem(
            source_fragment_id="fragment_",
            source_version_id="source_version_a",
            read_order=0,
            text_sha256=sha256_hex(TEXT_A.encode()),
        )
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        ReadReceiptItem(
            source_fragment_id="fragment_a",
            source_version_id="source_version_",
            read_order=0,
            text_sha256=sha256_hex(TEXT_A.encode()),
        )
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        _access_receipt(library_id="library_")
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        _access_receipt(access_policy_id="policy_")
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        _access_receipt(snapshot_id="snapshot_")
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        RetrievalTraceItem(source_fragment_id="fragment_", rank=1, score="-1.000000")
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        _evidence_fragment(source_family_id="family_")
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        _packet(query_request_id="query_")
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        _packet(corpus_snapshot_id="snapshot_")
    for source_fragment_id in ("fragment_a_", "fragment_a__b"):
        with pytest.raises(ValidationError, match="canonical non-empty suffix"):
            _evidence_fragment(source_fragment_id=source_fragment_id)


def test_receipt_hash_domains_bind_query_corpus_and_exact_retrieval_budget() -> None:
    read = _read_receipt()
    assert read.receipt_hash != _read_receipt(query_hash="7" * 64).receipt_hash
    assert read.receipt_hash != _read_receipt(corpus_hash="8" * 64).receipt_hash

    access = _access_receipt()
    assert access.receipt_hash != _access_receipt(query_hash="7" * 64).receipt_hash
    assert access.receipt_hash != _access_receipt(corpus_hash="8" * 64).receipt_hash

    retrieval = _retrieval_receipt()
    assert retrieval.receipt_hash != _retrieval_receipt(query_hash="7" * 64).receipt_hash
    assert retrieval.receipt_hash != _retrieval_receipt(corpus_hash="8" * 64).receipt_hash
    assert retrieval.receipt_hash != _retrieval_receipt(max_candidates=11).receipt_hash
    assert retrieval.receipt_hash != _retrieval_receipt(max_source_fragments=3).receipt_hash


def test_query_and_snapshot_content_ids_must_match_their_full_hashes() -> None:
    with pytest.raises(ValidationError, match="snapshot ID/hash tuple"):
        _access_receipt(snapshot_id="snapshot_other")
    with pytest.raises(ValidationError, match="QueryRequest ID/hash tuple"):
        _packet(query_request_id="query_other")
    with pytest.raises(ValidationError, match="CorpusSnapshot ID/hash tuple"):
        _packet(corpus_snapshot_id="snapshot_other")


def test_retrieval_candidate_count_may_exceed_bounded_trace() -> None:
    receipt = _retrieval_receipt(candidate_count=100, max_candidates=10)
    assert len(receipt.trace) == 2
    assert receipt.candidate_count == 100


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_count": 1},
        {"selected_count": 3},
        {"max_candidates": 1},
        {"max_candidates": 1, "max_source_fragments": 2},
        {"max_source_fragments": 1, "selected_count": 2},
    ],
)
def test_retrieval_counts_obey_selected_trace_candidate_and_budget_bounds(
    changes: dict[str, Any],
) -> None:
    with pytest.raises((ValidationError, ArtifactContractError)):
        _retrieval_receipt(**changes)


def test_retrieval_rejects_noncanonical_rank_score_or_score_order() -> None:
    base: dict[str, Any] = {
        "query_request_hash": QUERY_HASH,
        "profile_version": "fts_v1.0",
        "code_version": "0.1.0",
        "retrieval_corpus_hash": CORPUS_HASH,
        "max_candidates": 10,
        "max_source_fragments": 2,
        "candidate_count": 2,
        "selected_count": 1,
    }
    with pytest.raises(ValidationError, match="contiguous"):
        RetrievalReceipt(
            **base,
            trace=(RetrievalTraceItem(source_fragment_id="fragment_a", rank=2, score="-1.000000"),),
        )
    with pytest.raises(ValidationError, match="bm25"):
        RetrievalReceipt(
            **base,
            trace=(
                RetrievalTraceItem(source_fragment_id="fragment_a", rank=1, score="-1.000000"),
                RetrievalTraceItem(source_fragment_id="fragment_b", rank=2, score="-2.000000"),
            ),
        )
    # Equal serialized scores can represent distinct raw bm25 values. Their raw
    # order is enforced before quantization by the FTS implementation.
    equal_serialized_scores = RetrievalReceipt(
        **base,
        trace=(
            RetrievalTraceItem(source_fragment_id="fragment_b", rank=1, score="-1.000000"),
            RetrievalTraceItem(source_fragment_id="fragment_a", rank=2, score="-1.000000"),
        ),
    )
    assert [item.source_fragment_id for item in equal_serialized_scores.trace] == [
        "fragment_b",
        "fragment_a",
    ]
    for score in ("1.0", "1.00000", "-0.000000", "nan", 1.0):
        with pytest.raises((ValidationError, ArtifactContractError)):
            RetrievalTraceItem.model_validate(
                {"source_fragment_id": "fragment_a", "rank": 1, "score": score}
            )


def test_retrieval_rejects_bad_versions_duplicate_trace_and_trace_over_budget() -> None:
    with pytest.raises(ValidationError, match="version"):
        RetrievalReceipt(
            query_request_hash=QUERY_HASH,
            profile_version="bad version",
            code_version="0.1.0",
            retrieval_corpus_hash=CORPUS_HASH,
            max_candidates=10,
            max_source_fragments=2,
            candidate_count=0,
            selected_count=0,
        )

    duplicate_trace = (
        RetrievalTraceItem(source_fragment_id="fragment_a", rank=1, score="-2.000000"),
        RetrievalTraceItem(source_fragment_id="fragment_a", rank=2, score="-1.000000"),
    )
    with pytest.raises(ValidationError, match="unique"):
        RetrievalReceipt(
            query_request_hash=QUERY_HASH,
            profile_version="fts_v1.0",
            code_version="0.1.0",
            retrieval_corpus_hash=CORPUS_HASH,
            max_candidates=10,
            max_source_fragments=2,
            candidate_count=2,
            selected_count=1,
            trace=duplicate_trace,
        )
    with pytest.raises(ValidationError, match="exceeds max_candidates"):
        RetrievalReceipt(
            query_request_hash=QUERY_HASH,
            profile_version="fts_v1.0",
            code_version="0.1.0",
            retrieval_corpus_hash=CORPUS_HASH,
            max_candidates=1,
            max_source_fragments=1,
            candidate_count=2,
            selected_count=1,
            trace=_retrieval_receipt().trace,
        )


def test_recall_coverage_is_content_addressed_sorted_and_has_no_run_or_time() -> None:
    policy = RecallOmission(
        category=OmissionCategory.POLICY,
        disclosure=OmissionDisclosure.REDACTED,
        count=None,
        reason_code="policy_omission",
    )
    budget = RecallOmission(
        category=OmissionCategory.BUDGET,
        disclosure=OmissionDisclosure.COUNTED,
        count=2,
        reason_code="candidate_budget",
    )
    coverage = RecallCoverageReport(
        query_request_hash=QUERY_HASH,
        processed_count=2,
        skipped_count=0,
        failed_count=0,
        policy_omission_present=True,
        omissions=(policy, budget),
    )
    assert [item.category for item in coverage.omissions] == [
        OmissionCategory.BUDGET,
        OmissionCategory.POLICY,
    ]
    assert coverage.report_hash == canonical_sha256_hex(coverage.semantic_payload())
    assert coverage.coverage_report_id == canonical_content_id(
        "coverage", coverage.semantic_payload()
    )
    assert coverage.canonical_bytes == canonical_json_bytes(coverage.semantic_payload())
    assert coverage.report_hash != _coverage(query_hash="7" * 64).report_hash
    assert coverage.semantic_payload()["query_request_hash"] == QUERY_HASH
    assert "processing_run_id" not in coverage.semantic_payload()
    assert "created_at" not in coverage.semantic_payload()


def test_vs0_recall_coverage_forbids_skipped_failed_and_invalid_policy_disclosure() -> None:
    for changes in ({"skipped_count": 1}, {"failed_count": 1}):
        with pytest.raises(ValidationError, match="requires"):
            RecallCoverageReport(
                query_request_hash=QUERY_HASH,
                processed_count=0,
                skipped_count=changes.get("skipped_count", 0),
                failed_count=changes.get("failed_count", 0),
            )
    with pytest.raises(ValidationError, match="redacted"):
        RecallOmission(
            category=OmissionCategory.POLICY,
            disclosure=OmissionDisclosure.COUNTED,
            count=1,
            reason_code="policy_omission",
        )
    with pytest.raises(ValidationError, match="counted"):
        RecallOmission(
            category=OmissionCategory.BUDGET,
            disclosure=OmissionDisclosure.REDACTED,
            count=None,
            reason_code="candidate_budget",
        )
    with pytest.raises(ValidationError, match="snake_case"):
        RecallOmission(
            category=OmissionCategory.BUDGET,
            disclosure=OmissionDisclosure.COUNTED,
            count=1,
            reason_code="Bad reason",
        )


def test_coverage_rejects_duplicate_omissions_and_policy_presence_mismatch() -> None:
    budget = RecallOmission(
        category=OmissionCategory.BUDGET,
        disclosure=OmissionDisclosure.COUNTED,
        count=1,
        reason_code="candidate_budget",
    )
    with pytest.raises(ValidationError, match="unique"):
        RecallCoverageReport(
            query_request_hash=QUERY_HASH,
            processed_count=0,
            skipped_count=0,
            failed_count=0,
            omissions=(budget, budget),
        )
    with pytest.raises(ValidationError, match="presence"):
        RecallCoverageReport(
            query_request_hash=QUERY_HASH,
            processed_count=0,
            skipped_count=0,
            failed_count=0,
            policy_omission_present=True,
            omissions=(budget,),
        )


def test_evidence_fragment_strictly_validates_markdown_and_pdf_addresses() -> None:
    markdown = _evidence_fragment()
    assert isinstance(markdown.source_address, MarkdownSourceAddress)

    pdf_text = "PDF text"
    pdf_fragment = _evidence_fragment(
        text=pdf_text,
        text_sha256=sha256_hex(pdf_text.encode()),
        source_address={
            "schema": "dithyramba.source_address/1.0",
            "kind": "pdf",
            "page": 3,
            "bbox": ["72.000", "100.000", "520.000", "180.000"],
            "char_start": 0,
            "char_end": len(pdf_text),
        },
    )
    assert isinstance(pdf_fragment.source_address, PdfSourceAddress)

    direct = _evidence_fragment(
        source_address=MarkdownSourceAddress(
            heading_path=("Section",),
            line_start=10,
            line_end=10,
            char_start=100,
            char_end=100 + len(TEXT_A),
        )
    )
    assert isinstance(direct.source_address, MarkdownSourceAddress)


@pytest.mark.parametrize(
    "address",
    [
        {**_markdown_address(), "schema": "wrong"},
        {**_markdown_address(), "extra": True},
        {key: value for key, value in _markdown_address().items() if key != "line_end"},
        {**_markdown_address(), "kind": "unknown"},
        {**_markdown_address(), "char_end": 101},
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "pdf",
            "page": 1,
            "bbox": [72.0, "100.000", "520.000", "180.000"],
            "char_start": 0,
            "char_end": len(TEXT_A),
        },
        {
            "schema": "dithyramba.source_address/1.0",
            "kind": "pdf",
            "page": 1,
            "bbox": ["72.000", "100.000", "520.000"],
            "char_start": 0,
            "char_end": len(TEXT_A),
        },
    ],
)
def test_evidence_fragment_rejects_address_tampering(address: dict[str, object]) -> None:
    with pytest.raises((ValidationError, ArtifactContractError)):
        _evidence_fragment(source_address=address)


def test_evidence_fragment_rejects_text_hash_and_span_mismatch() -> None:
    with pytest.raises(ValidationError, match="does not match text"):
        _evidence_fragment(text_sha256="f" * 64)
    with pytest.raises(ValidationError, match="span length"):
        _evidence_fragment(source_address={**_markdown_address(), "char_end": 101})
    for text in ("   ", "bad\x00text"):
        with pytest.raises(ValidationError, match="nonblank NFC"):
            _evidence_fragment(text=text)
    with pytest.raises(ValidationError, match="SourceAddress payload"):
        _evidence_fragment(source_address="not-an-address")
    with pytest.raises(ValidationError, match="fragment_"):
        _evidence_fragment(source_fragment_id="wrong")
    with pytest.raises(ValidationError, match="SHA-256"):
        _evidence_fragment(text_sha256="A" * 64)


def test_evidence_fragment_nfc_inputs_have_identical_canonical_payloads() -> None:
    composed = "Café"
    text_hash = sha256_hex(composed.encode())
    first = _evidence_fragment(
        text="Cafe\u0301",
        text_sha256=text_hash,
        source_address=_markdown_address(composed),
    )
    second = _evidence_fragment(
        text=composed,
        text_sha256=text_hash,
        source_address=_markdown_address(composed),
    )
    assert first.payload() == second.payload()
    assert canonical_json_bytes(first.payload()) == canonical_json_bytes(second.payload())


def test_artifacts_reject_subclasses_that_can_override_canonical_payloads() -> None:
    class DerivedReadReceiptItem(ReadReceiptItem):
        pass

    derived_item = DerivedReadReceiptItem.model_validate(_read_receipt().items[0].model_dump())
    with pytest.raises(ValidationError, match="ReadReceiptItem"):
        ReadReceipt(
            query_request_hash=QUERY_HASH,
            retrieval_corpus_hash=CORPUS_HASH,
            items=(derived_item,),
        )

    class DerivedEvidenceFragment(EvidenceFragment):
        pass

    derived_values = _evidence_fragment().model_dump(exclude={"source_address"})
    derived_values["source_address"] = _markdown_address()
    derived_fragment = DerivedEvidenceFragment.model_validate(derived_values)
    with pytest.raises(ValidationError, match="EvidenceFragment"):
        _packet(source_fragments=(derived_fragment,))

    @dataclass(frozen=True, slots=True)
    class DerivedMarkdownAddress(MarkdownSourceAddress):
        pass

    with pytest.raises(ValidationError, match="exact SourceAddress"):
        _evidence_fragment(
            source_address=DerivedMarkdownAddress(
                heading_path=("Section",),
                line_start=10,
                line_end=10,
                char_start=100,
                char_end=100 + len(TEXT_A),
            )
        )


def test_packet_is_reproducible_content_addressed_and_contains_only_frozen_vs0_surfaces() -> None:
    packet = _packet()
    same = _packet()
    assert packet.semantic_payload() == same.semantic_payload()
    assert packet.canonical_bytes == same.canonical_bytes
    assert packet.packet_hash == canonical_sha256_hex(packet.semantic_payload())
    assert packet.evidence_packet_id == canonical_content_id("packet", packet.semantic_payload())
    decoded = json.loads(packet.canonical_bytes)
    assert decoded["packet_hash"] == packet.packet_hash
    assert decoded["counterevidence"] == {"status": "not_evaluated", "items": []}
    assert decoded["evidence_gaps"] == {"status": "not_evaluated", "items": []}
    assert "processing_run_id" not in packet.canonical_bytes.decode()
    assert "created_at" not in packet.canonical_bytes.decode()
    assert isinstance(decoded["source_fragments"][0]["score"], str)


def test_empty_packet_returns_no_evidence_without_invented_text() -> None:
    retrieval = _retrieval_receipt(selected_count=0)
    packet = _packet(
        result_status=PacketResultStatus.NO_EVIDENCE,
        source_fragments=(),
        retrieval_receipt=retrieval,
    )
    assert packet.source_fragments == ()
    assert packet.semantic_payload()["result_status"] == "no_evidence"


def test_packet_and_receipt_validators_reject_noncanonical_dependency_objects() -> None:
    with pytest.raises(ArtifactContractError, match="RetrievalTraceItem"):
        RetrievalReceipt._trace(cast(Any, (object(),)))
    with pytest.raises(ArtifactContractError, match="RecallOmission"):
        RecallCoverageReport._omissions(cast(Any, (object(),)))
    with pytest.raises(ArtifactContractError, match="RecallCoverageReport"):
        EvidencePacket._coverage_report(cast(Any, object()))
    with pytest.raises(ArtifactContractError, match="ReadReceipt"):
        EvidencePacket._read_receipt(cast(Any, object()))
    with pytest.raises(ArtifactContractError, match="AccessReceipt"):
        EvidencePacket._access_receipt(cast(Any, object()))
    with pytest.raises(ArtifactContractError, match="RetrievalReceipt"):
        EvidencePacket._retrieval_receipt(cast(Any, object()))


def test_empty_corpus_no_evidence_packet_has_closed_zero_counts_and_canonical_bytes() -> None:
    read = ReadReceipt(
        query_request_hash=QUERY_HASH,
        retrieval_corpus_hash=CORPUS_HASH,
        items=(),
    )
    retrieval = RetrievalReceipt(
        query_request_hash=QUERY_HASH,
        profile_version="fts_v1.0",
        code_version="0.1.0.dev2",
        retrieval_corpus_hash=CORPUS_HASH,
        max_candidates=10,
        max_source_fragments=2,
        candidate_count=0,
        selected_count=0,
        trace=(),
    )
    packet = _packet(
        result_status=PacketResultStatus.NO_EVIDENCE,
        source_fragments=(),
        read_receipt=read,
        retrieval_receipt=retrieval,
        coverage_report=_coverage(processed_count=0, omissions=()),
    )
    decoded = json.loads(packet.canonical_bytes)
    assert decoded["source_fragments"] == []
    assert decoded["coverage_report"]["processed_count"] == 0
    assert read.semantic_payload()["items"] == []
    assert retrieval.semantic_payload()["trace"] == []


def test_packet_coverage_counts_unique_read_source_versions() -> None:
    shared_version = "source_version_shared"
    read = ReadReceipt(
        query_request_hash=QUERY_HASH,
        retrieval_corpus_hash=CORPUS_HASH,
        items=(
            ReadReceiptItem(
                source_fragment_id="fragment_a",
                source_version_id=shared_version,
                read_order=0,
                text_sha256=sha256_hex(TEXT_A.encode()),
            ),
            ReadReceiptItem(
                source_fragment_id="fragment_b",
                source_version_id=shared_version,
                read_order=1,
                text_sha256=sha256_hex(TEXT_B.encode()),
            ),
        ),
    )
    packet = _packet(
        source_fragments=(_evidence_fragment(source_version_id=shared_version),),
        read_receipt=read,
        coverage_report=_coverage(processed_count=1),
    )
    assert packet.coverage_report.processed_count == 1

    with pytest.raises(ValidationError, match="unique ReadReceipt SourceVersions"):
        _packet(coverage_report=_coverage(processed_count=1))


def test_packet_selected_evidence_must_match_read_version_and_text_hash_exactly() -> None:
    def receipt_with_a(*, source_version_id: str, text_hash: str) -> ReadReceipt:
        return ReadReceipt(
            query_request_hash=QUERY_HASH,
            retrieval_corpus_hash=CORPUS_HASH,
            items=(
                ReadReceiptItem(
                    source_fragment_id="fragment_a",
                    source_version_id=source_version_id,
                    read_order=0,
                    text_sha256=text_hash,
                ),
                ReadReceiptItem(
                    source_fragment_id="fragment_b",
                    source_version_id="source_version_b",
                    read_order=1,
                    text_sha256=sha256_hex(TEXT_B.encode()),
                ),
            ),
        )

    mismatched_receipts = (
        receipt_with_a(
            source_version_id="source_version_other",
            text_hash=sha256_hex(TEXT_A.encode()),
        ),
        receipt_with_a(source_version_id="source_version_a", text_hash="f" * 64),
    )
    for read_receipt in mismatched_receipts:
        with pytest.raises(ValidationError, match="version/text hash"):
            _packet(read_receipt=read_receipt)


def test_packet_policy_omission_disclosure_closes_access_and_coverage() -> None:
    policy = RecallOmission(
        category=OmissionCategory.POLICY,
        disclosure=OmissionDisclosure.REDACTED,
        count=None,
        reason_code="policy_omission",
    )
    coverage = _coverage(policy_omission_present=True, omissions=(policy,))
    access = _access_receipt(policy_omission_present=True)
    packet = _packet(coverage_report=coverage, access_receipt=access)
    assert packet.access_receipt.policy_omission_present is True
    assert packet.coverage_report.policy_omission_present is True

    with pytest.raises(ValidationError, match="policy omission disclosures differ"):
        _packet(access_receipt=access)
    with pytest.raises(ValidationError, match="policy omission disclosures differ"):
        _packet(coverage_report=coverage)


@pytest.mark.parametrize(
    "changes",
    [
        {"result_status": PacketResultStatus.NO_EVIDENCE},
        {"query_request_hash": "7" * 64},
        {"corpus_snapshot_hash": "7" * 64},
        {"access_receipt": _access_receipt(corpus_hash="7" * 64)},
        {"read_receipt": _read_receipt(corpus_hash="7" * 64)},
        {"retrieval_receipt": _retrieval_receipt(selected_count=0)},
        {"coverage_report": _coverage(query_hash="7" * 64)},
    ],
)
def test_packet_rejects_inconsistent_dependency_receipts(changes: dict[str, Any]) -> None:
    with pytest.raises((ValidationError, ArtifactContractError)):
        _packet(**changes)


def test_packet_rejects_each_query_snapshot_trace_and_read_dependency_mismatch() -> None:
    wrong_selected = _evidence_fragment(source_fragment_id="fragment_other")
    only_a_read = ReadReceipt(
        query_request_hash=QUERY_HASH,
        retrieval_corpus_hash=CORPUS_HASH,
        items=(
            ReadReceiptItem(
                source_fragment_id="fragment_a",
                source_version_id="source_version_a",
                read_order=0,
                text_sha256=sha256_hex(TEXT_A.encode()),
            ),
        ),
    )
    mismatches = (
        {"access_receipt": _access_receipt(query_hash="7" * 64)},
        {"retrieval_receipt": _retrieval_receipt(query_hash="7" * 64)},
        {"coverage_report": _coverage(query_hash="7" * 64)},
        {
            "access_receipt": _access_receipt(
                snapshot_id=f"snapshot_{'7' * 32}",
                snapshot_hash="7" * 64,
            )
        },
        {"source_fragments": (wrong_selected,)},
        {"read_receipt": only_a_read},
    )
    for changes in mismatches:
        with pytest.raises((ValidationError, ArtifactContractError)):
            _packet(**changes)


def test_packet_rejects_duplicate_and_noncontiguous_selected_fragments() -> None:
    fragment = _evidence_fragment()
    with pytest.raises(ValidationError, match="unique"):
        _packet(source_fragments=(fragment, fragment))
    with pytest.raises(ValidationError, match="contiguous"):
        _packet(source_fragments=(_evidence_fragment(rank=2),))


def test_artifact_models_are_frozen_and_forbid_run_time_and_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ReadReceipt.model_validate(
            {
                "query_request_hash": QUERY_HASH,
                "retrieval_corpus_hash": CORPUS_HASH,
                "items": (),
                "processing_run_id": "run_forbidden",
                "created_at": "2026-07-20T12:00:00.000000Z",
            }
        )
    packet = _packet()
    with pytest.raises(ValidationError, match="frozen"):
        packet.query_request_id = "query_changed"
