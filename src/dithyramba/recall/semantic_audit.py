"""Exact, source-text-free coverage audit for semantic embedding inputs.

The audit is deliberately upstream of vector generation.  It asks the exact
inference tokenizer for a no-truncation prepared-input count and records
whether every authorized SourceFragment fits the model profile.  It never
persists source text and it does not claim that semantic windows exist.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)
from dithyramba.persistence.models import SourceFragmentText
from dithyramba.recall.hybrid_models import HybridContractError

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class SemanticAuditContractError(HybridContractError):
    """A semantic coverage audit violates its immutable public contract."""


class SemanticAuditItemStatus(StrEnum):
    """Outcome of exact prepared-token counting for one SourceFragment."""

    FIT = "fit"
    WINDOW_REQUIRED = "window_required"
    FAILED = "failed"


class SemanticAuditStatus(StrEnum):
    """Corpus-level semantic coverage status."""

    COMPLETE_FIT = "complete_fit"
    TRUNCATION_RISK = "truncation_risk"
    FAILED = "failed"


class SemanticAuditFailureReason(StrEnum):
    """Stable, text-free reasons why an exact token count was unavailable."""

    TOKENIZER_UNAVAILABLE = "tokenizer_unavailable"
    TOKENIZER_REJECTED_INPUT = "tokenizer_rejected_input"
    TOKENIZER_ERROR = "tokenizer_error"
    INVALID_TOKEN_COUNT = "invalid_token_count"


class PreparedTokenCountError(RuntimeError):
    """A token counter's classified, source-text-free failure."""

    def __init__(self, reason: SemanticAuditFailureReason) -> None:
        if type(reason) is not SemanticAuditFailureReason:
            raise SemanticAuditContractError(
                "prepared token count failure reason must use SemanticAuditFailureReason"
            )
        self.reason = reason
        super().__init__(reason.value)


class PreparedTokenCounter(Protocol):
    """Exact inference-tokenizer adapter used by the coverage audit.

    Implementations must count ``prepared_text`` with the tokenizer used for
    inference, including special tokens and with ``truncation=False``.  The
    caller supplies the already-prefixed passage text.
    """

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
        """Return the exact no-truncation prepared-input token count."""


@dataclass(frozen=True, slots=True)
class SemanticAuditBinding:
    """All immutable identities needed to interpret one coverage audit."""

    SCHEMA = "dithyramba.semantic_audit_binding/1.0"

    library_id: str
    corpus_snapshot_id: str
    snapshot_hash: str
    access_policy_id: str
    policy_hash: str
    exclusion_hash: str
    permitted_set_hash: str
    model_profile_hash: str
    runtime_profile_hash: str
    provisioning_receipt_hash: str
    max_tokens: int
    passage_prefix: str

    def __post_init__(self) -> None:
        _require_id(self.library_id, "library")
        _require_id(self.corpus_snapshot_id, "snapshot")
        _require_id(self.access_policy_id, "policy")
        for value in (
            self.snapshot_hash,
            self.policy_hash,
            self.exclusion_hash,
            self.permitted_set_hash,
            self.model_profile_hash,
            self.runtime_profile_hash,
            self.provisioning_receipt_hash,
        ):
            _require_hash(value)
        expected_snapshot_id = f"snapshot_{self.snapshot_hash[:32]}"
        if self.corpus_snapshot_id != expected_snapshot_id:
            raise SemanticAuditContractError(
                "corpus_snapshot_id must be derived from the bound snapshot_hash"
            )
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 32_768:
            raise SemanticAuditContractError("max_tokens must be an integer from 1 to 32768")
        _require_prefix(self.passage_prefix)

    def payload(self) -> dict[str, object]:
        """Return the canonical semantic identity payload."""

        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "access_policy_id": self.access_policy_id,
            "policy_hash": self.policy_hash,
            "exclusion_hash": self.exclusion_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "model_profile_hash": self.model_profile_hash,
            "runtime_profile_hash": self.runtime_profile_hash,
            "provisioning_receipt_hash": self.provisioning_receipt_hash,
            "max_tokens": self.max_tokens,
            "passage_prefix": self.passage_prefix,
        }

    @property
    def binding_hash(self) -> str:
        """Return the canonical hash of the complete audit binding."""

        return canonical_sha256_hex(self.payload())


@dataclass(frozen=True, slots=True)
class SemanticAuditItem:
    """Immutable, source-text-free audit result for one SourceFragment."""

    source_fragment_id: str
    source_version_id: str
    source_id: str
    ordinal: int
    text_sha256: str
    status: SemanticAuditItemStatus
    prepared_token_count: int | None
    failure_reason: SemanticAuditFailureReason | None = None

    def __post_init__(self) -> None:
        _require_id(self.source_fragment_id, "fragment")
        _require_id(self.source_version_id, "source_version")
        _require_id(self.source_id, "source")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise SemanticAuditContractError("fragment ordinal must be a non-negative integer")
        _require_hash(self.text_sha256)
        if type(self.status) is not SemanticAuditItemStatus:
            raise SemanticAuditContractError("audit item status must use SemanticAuditItemStatus")
        if self.status is SemanticAuditItemStatus.FAILED:
            if self.prepared_token_count is not None:
                raise SemanticAuditContractError("failed audit item cannot contain a token count")
            if type(self.failure_reason) is not SemanticAuditFailureReason:
                raise SemanticAuditContractError("failed audit item requires a failure reason")
            return
        if type(self.prepared_token_count) is not int or self.prepared_token_count < 1:
            raise SemanticAuditContractError(
                "successful audit item requires a positive token count"
            )
        if self.failure_reason is not None:
            raise SemanticAuditContractError(
                "successful audit item cannot contain a failure reason"
            )

    def identity_payload(self) -> dict[str, object]:
        """Return the exact fragment-manifest entry, without source text."""

        return {
            "source_fragment_id": self.source_fragment_id,
            "source_version_id": self.source_version_id,
            "source_id": self.source_id,
            "ordinal": self.ordinal,
            "text_sha256": self.text_sha256,
        }

    def payload(self) -> dict[str, object]:
        """Return the canonical result entry, without source text."""

        return {
            **self.identity_payload(),
            "status": self.status.value,
            "prepared_token_count": self.prepared_token_count,
            "failure_reason": (None if self.failure_reason is None else self.failure_reason.value),
        }


@dataclass(frozen=True, slots=True)
class SemanticCoverageAudit:
    """Canonical proof of semantic coverage for an exact fragment closure."""

    SCHEMA = "dithyramba.semantic_coverage_audit/1.0"
    MANIFEST_SCHEMA = "dithyramba.semantic_fragment_manifest/1.0"

    binding: SemanticAuditBinding
    items: tuple[SemanticAuditItem, ...]

    def __post_init__(self) -> None:
        if type(self.binding) is not SemanticAuditBinding:
            raise SemanticAuditContractError("audit binding must be an exact SemanticAuditBinding")
        if type(self.items) is not tuple:
            raise SemanticAuditContractError("audit items must be an exact tuple")
        for item in self.items:
            if type(item) is not SemanticAuditItem:
                raise SemanticAuditContractError(
                    "audit items must be exact SemanticAuditItem values"
                )
        _validate_item_closure(self.items, max_tokens=self.binding.max_tokens)

    @property
    def status(self) -> SemanticAuditStatus:
        """Return the fail-closed corpus-level status."""

        if any(item.status is SemanticAuditItemStatus.FAILED for item in self.items):
            return SemanticAuditStatus.FAILED
        if any(item.status is SemanticAuditItemStatus.WINDOW_REQUIRED for item in self.items):
            return SemanticAuditStatus.TRUNCATION_RISK
        return SemanticAuditStatus.COMPLETE_FIT

    @property
    def fragment_manifest_hash(self) -> str:
        """Hash the exact ordered SourceFragment closure independently of results."""

        return canonical_sha256_hex(
            {
                "schema": self.MANIFEST_SCHEMA,
                "items": [item.identity_payload() for item in self.items],
            }
        )

    def payload(self) -> dict[str, object]:
        """Return the canonical source-text-free audit payload."""

        counts = {
            status.value: sum(item.status is status for item in self.items)
            for status in SemanticAuditItemStatus
        }
        return {
            "schema": self.SCHEMA,
            "status": self.status.value,
            "binding": self.binding.payload(),
            "binding_hash": self.binding.binding_hash,
            "fragment_manifest_hash": self.fragment_manifest_hash,
            "counts": {"total": len(self.items), **counts},
            "items": [item.payload() for item in self.items],
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Return the exact canonical JSON representation of this audit."""

        return canonical_json_bytes(self.payload())

    @property
    def audit_hash(self) -> str:
        """Return the canonical SHA-256 audit hash."""

        return canonical_sha256_hex(self.payload())

    @property
    def audit_id(self) -> str:
        """Return the content-addressed audit identifier."""

        return canonical_content_id("semantic_audit", self.payload())


def audit_semantic_coverage(
    fragments: tuple[SourceFragmentText, ...],
    *,
    binding: SemanticAuditBinding,
    token_counter: PreparedTokenCounter,
) -> SemanticCoverageAudit:
    """Audit every already-authorized SourceFragment without truncation.

    The entire fragment tuple is validated before the tokenizer is called, so
    malformed input cannot produce a partial result that looks authoritative.
    Tokenizer failures are retained as classified per-fragment failures; raw
    exception text and source text never enter the canonical artifact.
    """

    if type(binding) is not SemanticAuditBinding:
        raise SemanticAuditContractError("binding must be an exact SemanticAuditBinding")
    validated = _validate_fragments(fragments)
    counter = getattr(token_counter, "count_prepared_tokens_no_truncation", None)
    if not callable(counter):
        return SemanticCoverageAudit(
            binding=binding,
            items=tuple(
                _failed_item(fragment, SemanticAuditFailureReason.TOKENIZER_UNAVAILABLE)
                for fragment in validated
            ),
        )

    items: list[SemanticAuditItem] = []
    for fragment in validated:
        try:
            token_count = counter(f"{binding.passage_prefix}{fragment.text}")
        except PreparedTokenCountError as exc:
            items.append(_failed_item(fragment, exc.reason))
            continue
        except Exception:
            items.append(_failed_item(fragment, SemanticAuditFailureReason.TOKENIZER_ERROR))
            continue
        if type(token_count) is not int or token_count < 1:
            items.append(_failed_item(fragment, SemanticAuditFailureReason.INVALID_TOKEN_COUNT))
            continue
        status = (
            SemanticAuditItemStatus.FIT
            if token_count <= binding.max_tokens
            else SemanticAuditItemStatus.WINDOW_REQUIRED
        )
        items.append(
            SemanticAuditItem(
                source_fragment_id=fragment.source_fragment_id,
                source_version_id=fragment.source_version_id,
                source_id=fragment.source_id,
                ordinal=fragment.ordinal,
                text_sha256=fragment.text_sha256,
                status=status,
                prepared_token_count=token_count,
            )
        )
    return SemanticCoverageAudit(binding=binding, items=tuple(items))


def _failed_item(
    fragment: SourceFragmentText,
    reason: SemanticAuditFailureReason,
) -> SemanticAuditItem:
    return SemanticAuditItem(
        source_fragment_id=fragment.source_fragment_id,
        source_version_id=fragment.source_version_id,
        source_id=fragment.source_id,
        ordinal=fragment.ordinal,
        text_sha256=fragment.text_sha256,
        status=SemanticAuditItemStatus.FAILED,
        prepared_token_count=None,
        failure_reason=reason,
    )


def _validate_fragments(
    fragments: tuple[SourceFragmentText, ...],
) -> tuple[SourceFragmentText, ...]:
    if type(fragments) is not tuple:
        raise SemanticAuditContractError("fragments must be an exact tuple")
    fragment_ids: list[str] = []
    version_ordinals: set[tuple[str, int]] = set()
    version_sources: dict[str, str] = {}
    for fragment in fragments:
        if type(fragment) is not SourceFragmentText:
            raise SemanticAuditContractError(
                "fragments must contain exact SourceFragmentText values"
            )
        _require_id(fragment.source_fragment_id, "fragment")
        _require_id(fragment.source_version_id, "source_version")
        _require_id(fragment.source_id, "source")
        if type(fragment.ordinal) is not int or fragment.ordinal < 0:
            raise SemanticAuditContractError("fragment ordinal must be a non-negative integer")
        if type(fragment.text) is not str or not fragment.text:
            raise SemanticAuditContractError("fragment text must be a non-empty exact string")
        _require_hash(fragment.text_sha256)
        actual_text_hash = sha256_hex(fragment.text.encode("utf-8"))
        if actual_text_hash != fragment.text_sha256:
            raise SemanticAuditContractError("fragment text does not match text_sha256")
        fragment_ids.append(fragment.source_fragment_id)
        version_ordinal = (fragment.source_version_id, fragment.ordinal)
        if version_ordinal in version_ordinals:
            raise SemanticAuditContractError("fragment SourceVersion ordinal pairs must be unique")
        version_ordinals.add(version_ordinal)
        existing_source = version_sources.setdefault(
            fragment.source_version_id,
            fragment.source_id,
        )
        if existing_source != fragment.source_id:
            raise SemanticAuditContractError("one SourceVersion cannot belong to multiple Sources")
    if len(set(fragment_ids)) != len(fragment_ids):
        raise SemanticAuditContractError("fragment IDs must be unique")
    expected_ids = tuple(sorted(fragment_ids, key=lambda value: value.encode("ascii")))
    if tuple(fragment_ids) != expected_ids:
        raise SemanticAuditContractError("fragments must use canonical fragment-ID order")
    return fragments


def _validate_item_closure(
    items: tuple[SemanticAuditItem, ...],
    *,
    max_tokens: int,
) -> None:
    fragment_ids = tuple(item.source_fragment_id for item in items)
    if len(set(fragment_ids)) != len(fragment_ids):
        raise SemanticAuditContractError("audit item fragment IDs must be unique")
    expected_ids = tuple(sorted(fragment_ids, key=lambda value: value.encode("ascii")))
    if fragment_ids != expected_ids:
        raise SemanticAuditContractError("audit items must use canonical fragment-ID order")
    version_ordinals: set[tuple[str, int]] = set()
    version_sources: dict[str, str] = {}
    for item in items:
        version_ordinal = (item.source_version_id, item.ordinal)
        if version_ordinal in version_ordinals:
            raise SemanticAuditContractError(
                "audit item SourceVersion ordinal pairs must be unique"
            )
        version_ordinals.add(version_ordinal)
        existing_source = version_sources.setdefault(item.source_version_id, item.source_id)
        if existing_source != item.source_id:
            raise SemanticAuditContractError("one SourceVersion cannot belong to multiple Sources")
        if item.status is SemanticAuditItemStatus.FIT:
            if item.prepared_token_count is None or item.prepared_token_count > max_tokens:
                raise SemanticAuditContractError("fit item count must not exceed max_tokens")
        elif item.status is SemanticAuditItemStatus.WINDOW_REQUIRED and (
            item.prepared_token_count is None or item.prepared_token_count <= max_tokens
        ):
            raise SemanticAuditContractError("window-required item count must exceed max_tokens")


def _require_hash(value: object) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise SemanticAuditContractError(
            "artifact hash must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_id(value: object, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise SemanticAuditContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise SemanticAuditContractError(f"identifier must use a canonical {expected} suffix")
    return value


def _require_prefix(value: object) -> str:
    if type(value) is not str or len(value) > 128 or "\x00" in value:
        raise SemanticAuditContractError("passage_prefix must be bounded text without NUL")
    if unicodedata.normalize("NFC", value) != value:
        raise SemanticAuditContractError("passage_prefix must already be NFC")
    return value
