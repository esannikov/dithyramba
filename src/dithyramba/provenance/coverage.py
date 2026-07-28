"""Canonical CoverageReport construction for deterministic ingest outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from dithyramba.contracts import canonical_content_id, canonical_json_bytes, canonical_sha256_hex

from .models import IngestInputOutcome, TerminalInputOutcome

_UNSUPPORTED_CODES = frozenset(
    {
        "no_extractable_text",
        "unsupported_encrypted_pdf",
        "unsupported_format",
    }
)


@dataclass(frozen=True, slots=True)
class OmissionDraft:
    """Canonical omission row ready for one CoverageReport transaction."""

    omission_id: str
    category: str
    disclosure: str
    count: int | None
    reason_code: str


@dataclass(frozen=True, slots=True)
class CoverageDraft:
    """Canonical payload and relational projections for one ingest batch."""

    coverage_report_id: str
    processing_run_id: str
    collection_id: str
    processed_count: int
    skipped_count: int
    failed_count: int
    policy_omission_present: bool
    report_json: str
    report_hash: str
    omissions: tuple[OmissionDraft, ...]


def build_ingest_coverage(
    *,
    processing_run_id: str,
    collection_id: str,
    outcomes: tuple[IngestInputOutcome, ...],
) -> CoverageDraft:
    """Build the sole canonical VS0 ingest CoverageReport for a run."""

    processed = sum(
        outcome.terminal_outcome is TerminalInputOutcome.PROCESSED for outcome in outcomes
    )
    skipped = sum(outcome.terminal_outcome is TerminalInputOutcome.SKIPPED for outcome in outcomes)
    failed = sum(outcome.terminal_outcome is TerminalInputOutcome.FAILED for outcome in outcomes)

    omission_counts: dict[tuple[str, str], int] = {}
    for outcome in outcomes:
        if outcome.terminal_outcome is TerminalInputOutcome.PROCESSED:
            continue
        if outcome.failure_code is None:  # guarded by IngestInputOutcome
            raise ValueError("non-processed outcome must have a failure code")
        category = "unsupported" if outcome.failure_code in _UNSUPPORTED_CODES else "parser_failure"
        key = (category, outcome.failure_code)
        omission_counts[key] = omission_counts.get(key, 0) + 1

    omission_payloads: list[dict[str, object]] = [
        {
            "category": category,
            "disclosure": "counted",
            "count": count,
            "reason_code": reason_code,
        }
        for (category, reason_code), count in sorted(omission_counts.items())
    ]
    payload: dict[str, object] = {
        "schema": "dithyramba.coverage_report/1.0",
        "stage": "ingest",
        "processing_run_id": processing_run_id,
        "collection_id": collection_id,
        "processed_count": processed,
        "skipped_count": skipped,
        "failed_count": failed,
        "policy_omission": {"present": False, "count": None},
        "outcomes": [outcome.payload() for outcome in outcomes],
        "omissions": omission_payloads,
    }
    report_json = canonical_json_bytes(payload).decode("utf-8")
    report_hash = canonical_sha256_hex(payload)
    coverage_report_id = canonical_content_id("coverage", payload)
    omission_records: list[OmissionDraft] = []
    for item in omission_payloads:
        count = item["count"]
        if type(count) is not int:
            raise TypeError("counted ingest omission must have an integer count")
        omission_records.append(
            OmissionDraft(
                omission_id=canonical_content_id(
                    "omission",
                    {
                        "schema": "dithyramba.omission/1.0",
                        "coverage_report_id": coverage_report_id,
                        **item,
                    },
                ),
                category=str(item["category"]),
                disclosure=str(item["disclosure"]),
                count=count,
                reason_code=str(item["reason_code"]),
            )
        )
    omissions = tuple(omission_records)
    return CoverageDraft(
        coverage_report_id=coverage_report_id,
        processing_run_id=processing_run_id,
        collection_id=collection_id,
        processed_count=processed,
        skipped_count=skipped,
        failed_count=failed,
        policy_omission_present=False,
        report_json=report_json,
        report_hash=report_hash,
        omissions=omissions,
    )
