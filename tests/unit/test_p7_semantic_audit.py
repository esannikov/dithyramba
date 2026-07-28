"""Pure tests for exact, audit-first P7 semantic coverage."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from dithyramba.contracts import sha256_hex
from dithyramba.persistence.models import SourceFragmentText
from dithyramba.recall.semantic_audit import (
    PreparedTokenCounter,
    PreparedTokenCountError,
    SemanticAuditBinding,
    SemanticAuditContractError,
    SemanticAuditFailureReason,
    SemanticAuditItem,
    SemanticAuditItemStatus,
    SemanticAuditStatus,
    SemanticCoverageAudit,
    audit_semantic_coverage,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
SNAPSHOT_HASH = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64
HASH_0 = "0" * 64


def _binding(*, max_tokens: int = 512) -> SemanticAuditBinding:
    return SemanticAuditBinding(
        library_id="library_test",
        corpus_snapshot_id=f"snapshot_{SNAPSHOT_HASH[:32]}",
        snapshot_hash=SNAPSHOT_HASH,
        access_policy_id="policy_research",
        policy_hash=HASH_A,
        exclusion_hash=HASH_B,
        permitted_set_hash=HASH_D,
        model_profile_hash=HASH_E,
        runtime_profile_hash=HASH_F,
        provisioning_receipt_hash=HASH_0,
        max_tokens=max_tokens,
        passage_prefix="passage: ",
    )


def _fragment(
    fragment_id: str,
    text: str,
    *,
    source_version_id: str = "source_version_one",
    source_id: str = "source_one",
    ordinal: int = 0,
) -> SourceFragmentText:
    return SourceFragmentText(
        source_fragment_id=fragment_id,
        source_version_id=source_version_id,
        source_id=source_id,
        ordinal=ordinal,
        fragment_kind="paragraph",
        text=text,
        text_sha256=sha256_hex(text.encode("utf-8")),
        source_address_json="{}",
    )


class MappingCounter:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.calls: list[str] = []

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
        self.calls.append(prepared_text)
        return self.counts[prepared_text]


class OutcomeCounter:
    def __init__(self, outcomes: tuple[object, ...]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
        self.calls.append(prepared_text)
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return cast(int, outcome)


class RecordingCounter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
        self.calls.append(prepared_text)
        return 1


def test_audit_counts_exact_prefixed_inputs_and_never_serializes_source_text() -> None:
    alpha = _fragment("fragment_alpha", "classified alpha", ordinal=0)
    beta = _fragment("fragment_beta", "classified beta", ordinal=1)
    counter = MappingCounter(
        {
            "passage: classified alpha": 512,
            "passage: classified beta": 513,
        }
    )

    audit = audit_semantic_coverage((alpha, beta), binding=_binding(), token_counter=counter)

    assert counter.calls == ["passage: classified alpha", "passage: classified beta"]
    assert audit.status is SemanticAuditStatus.TRUNCATION_RISK
    assert tuple(item.status for item in audit.items) == (
        SemanticAuditItemStatus.FIT,
        SemanticAuditItemStatus.WINDOW_REQUIRED,
    )
    assert tuple(item.prepared_token_count for item in audit.items) == (512, 513)
    assert audit.payload()["counts"] == {
        "total": 2,
        "fit": 1,
        "window_required": 1,
        "failed": 0,
    }
    assert b"classified alpha" not in audit.canonical_bytes
    assert b"classified beta" not in audit.canonical_bytes
    assert audit.audit_id.startswith("semantic_audit_")
    assert len(audit.audit_hash) == 64
    assert len(audit.fragment_manifest_hash) == 64


def test_audit_is_deterministic_and_binds_complete_fragment_identity() -> None:
    fragment = _fragment("fragment_alpha", "same source text")
    first = audit_semantic_coverage(
        (fragment,),
        binding=_binding(),
        token_counter=MappingCounter({"passage: same source text": 12}),
    )
    second = audit_semantic_coverage(
        (fragment,),
        binding=_binding(),
        token_counter=MappingCounter({"passage: same source text": 12}),
    )

    assert first == second
    assert first.canonical_bytes == second.canonical_bytes
    assert first.audit_hash == second.audit_hash
    assert first.audit_id == second.audit_id
    assert first.items[0].identity_payload() == {
        "source_fragment_id": "fragment_alpha",
        "source_version_id": "source_version_one",
        "source_id": "source_one",
        "ordinal": 0,
        "text_sha256": fragment.text_sha256,
    }


def test_classified_and_unknown_tokenizer_failures_fail_closed_without_details() -> None:
    fragments = (
        _fragment("fragment_alpha", "secret alpha", ordinal=0),
        _fragment("fragment_beta", "secret beta", ordinal=1),
        _fragment("fragment_gamma", "secret gamma", ordinal=2),
        _fragment("fragment_zeta", "secret zeta", ordinal=3),
    )
    counter = OutcomeCounter(
        (
            PreparedTokenCountError(SemanticAuditFailureReason.TOKENIZER_REJECTED_INPUT),
            RuntimeError("must not enter the audit"),
            False,
            "12",
        )
    )

    audit = audit_semantic_coverage(fragments, binding=_binding(), token_counter=counter)

    assert audit.status is SemanticAuditStatus.FAILED
    assert tuple(item.failure_reason for item in audit.items) == (
        SemanticAuditFailureReason.TOKENIZER_REJECTED_INPUT,
        SemanticAuditFailureReason.TOKENIZER_ERROR,
        SemanticAuditFailureReason.INVALID_TOKEN_COUNT,
        SemanticAuditFailureReason.INVALID_TOKEN_COUNT,
    )
    assert all(item.prepared_token_count is None for item in audit.items)
    assert b"must not enter" not in audit.canonical_bytes
    assert b"secret alpha" not in audit.canonical_bytes


def test_missing_counter_is_an_explicit_failure_for_every_fragment() -> None:
    fragments = (
        _fragment("fragment_alpha", "alpha", ordinal=0),
        _fragment("fragment_beta", "beta", ordinal=1),
    )

    audit = audit_semantic_coverage(
        fragments,
        binding=_binding(),
        token_counter=cast(PreparedTokenCounter, object()),
    )

    assert audit.status is SemanticAuditStatus.FAILED
    assert {item.failure_reason for item in audit.items} == {
        SemanticAuditFailureReason.TOKENIZER_UNAVAILABLE
    }


def test_empty_authorized_closure_is_complete_and_content_addressed() -> None:
    counter = RecordingCounter()

    audit = audit_semantic_coverage((), binding=_binding(), token_counter=counter)

    assert audit.status is SemanticAuditStatus.COMPLETE_FIT
    assert audit.items == ()
    assert audit.payload()["counts"] == {
        "total": 0,
        "fit": 0,
        "window_required": 0,
        "failed": 0,
    }
    assert counter.calls == []


@pytest.mark.parametrize(
    "fragments",
    [
        [_fragment("fragment_alpha", "alpha")],
        (cast(SourceFragmentText, object()),),
        (
            _fragment("fragment_beta", "beta", ordinal=1),
            _fragment("fragment_alpha", "alpha", ordinal=0),
        ),
        (
            _fragment("fragment_alpha", "alpha", ordinal=0),
            _fragment("fragment_alpha", "other", ordinal=1),
        ),
        (
            _fragment("fragment_alpha", "alpha", ordinal=0),
            _fragment("fragment_beta", "beta", ordinal=0),
        ),
        (
            _fragment("fragment_alpha", "alpha", source_id="source_one", ordinal=0),
            _fragment("fragment_beta", "beta", source_id="source_two", ordinal=1),
        ),
        (replace(_fragment("fragment_alpha", "alpha"), source_fragment_id="bad"),),
        (replace(_fragment("fragment_alpha", "alpha"), ordinal=-1),),
        (replace(_fragment("fragment_alpha", "alpha"), text=""),),
        (replace(_fragment("fragment_alpha", "alpha"), text_sha256=HASH_A),),
    ],
)
def test_invalid_fragment_closure_fails_before_any_tokenizer_call(fragments: object) -> None:
    counter = RecordingCounter()

    with pytest.raises(SemanticAuditContractError):
        audit_semantic_coverage(
            cast(tuple[SourceFragmentText, ...], fragments),
            binding=_binding(),
            token_counter=counter,
        )

    assert counter.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("library_id", "bad"),
        ("access_policy_id", "access_research"),
        ("corpus_snapshot_id", "snapshot_wrong"),
        ("policy_hash", "A" * 64),
        ("max_tokens", True),
        ("max_tokens", 0),
        ("max_tokens", 32_769),
        ("passage_prefix", "e\u0301"),
        ("passage_prefix", "bad\x00prefix"),
        ("passage_prefix", "x" * 129),
    ],
)
def test_binding_validation_is_strict(field: str, value: object) -> None:
    with pytest.raises(SemanticAuditContractError):
        replace(_binding(), **cast(Any, {field: value}))


def test_item_and_audit_invariants_cannot_be_forged() -> None:
    fit = SemanticAuditItem(
        source_fragment_id="fragment_alpha",
        source_version_id="source_version_one",
        source_id="source_one",
        ordinal=0,
        text_sha256=HASH_A,
        status=SemanticAuditItemStatus.FIT,
        prepared_token_count=512,
    )
    required = replace(
        fit,
        source_fragment_id="fragment_beta",
        ordinal=1,
        status=SemanticAuditItemStatus.WINDOW_REQUIRED,
        prepared_token_count=513,
    )
    audit = SemanticCoverageAudit(binding=_binding(), items=(fit, required))
    assert audit.status is SemanticAuditStatus.TRUNCATION_RISK

    with pytest.raises(SemanticAuditContractError, match="fit item count"):
        SemanticCoverageAudit(binding=_binding(), items=(replace(fit, prepared_token_count=513),))
    with pytest.raises(SemanticAuditContractError, match="window-required"):
        SemanticCoverageAudit(
            binding=_binding(),
            items=(replace(required, prepared_token_count=512),),
        )
    with pytest.raises(SemanticAuditContractError, match="failure reason"):
        replace(
            fit,
            status=SemanticAuditItemStatus.FAILED,
            prepared_token_count=None,
        )
    with pytest.raises(SemanticAuditContractError, match="positive token"):
        replace(fit, prepared_token_count=cast(int, False))
    with pytest.raises(SemanticAuditContractError, match="non-negative"):
        replace(fit, ordinal=-1)
    with pytest.raises(SemanticAuditContractError, match="status"):
        replace(fit, status=cast(SemanticAuditItemStatus, "fit"))
    with pytest.raises(SemanticAuditContractError, match="cannot contain a token"):
        replace(
            fit,
            status=SemanticAuditItemStatus.FAILED,
            failure_reason=SemanticAuditFailureReason.TOKENIZER_ERROR,
        )
    with pytest.raises(SemanticAuditContractError, match="cannot contain a failure"):
        replace(fit, failure_reason=SemanticAuditFailureReason.TOKENIZER_ERROR)
    with pytest.raises(SemanticAuditContractError, match="canonical fragment_"):
        replace(fit, source_fragment_id="fragment_bad-suffix")
    with pytest.raises(SemanticAuditContractError, match="exact tuple"):
        SemanticCoverageAudit(
            binding=_binding(),
            items=cast(tuple[SemanticAuditItem, ...], [fit]),
        )

    with pytest.raises(SemanticAuditContractError, match="exact SemanticAuditBinding"):
        SemanticCoverageAudit(
            binding=cast(SemanticAuditBinding, object()),
            items=(),
        )
    with pytest.raises(SemanticAuditContractError, match="exact SemanticAuditItem"):
        SemanticCoverageAudit(
            binding=_binding(),
            items=(cast(SemanticAuditItem, object()),),
        )
    with pytest.raises(SemanticAuditContractError, match="fragment IDs must be unique"):
        SemanticCoverageAudit(binding=_binding(), items=(fit, replace(fit, ordinal=1)))
    with pytest.raises(SemanticAuditContractError, match="fragment-ID order"):
        SemanticCoverageAudit(binding=_binding(), items=(required, fit))
    with pytest.raises(SemanticAuditContractError, match="ordinal pairs must be unique"):
        SemanticCoverageAudit(
            binding=_binding(),
            items=(fit, replace(fit, source_fragment_id="fragment_beta")),
        )
    with pytest.raises(SemanticAuditContractError, match="multiple Sources"):
        SemanticCoverageAudit(
            binding=_binding(),
            items=(
                fit,
                replace(
                    fit,
                    source_fragment_id="fragment_beta",
                    source_id="source_two",
                    ordinal=1,
                ),
            ),
        )


def test_audit_helper_requires_an_exact_binding_before_tokenization() -> None:
    counter = RecordingCounter()
    with pytest.raises(SemanticAuditContractError, match="exact SemanticAuditBinding"):
        audit_semantic_coverage(
            (),
            binding=cast(SemanticAuditBinding, object()),
            token_counter=counter,
        )
    assert counter.calls == []


def test_prepared_token_error_requires_a_stable_reason_enum() -> None:
    with pytest.raises(SemanticAuditContractError, match="failure reason"):
        PreparedTokenCountError(cast(SemanticAuditFailureReason, "tokenizer_error"))
