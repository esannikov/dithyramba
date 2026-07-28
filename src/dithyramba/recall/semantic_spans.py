"""Lossless, text-free semantic span planning over exact parent token offsets.

``content_token_start`` and ``content_token_end`` are half-open coordinates in
the parent fragment's one no-special-token offset mapping. They are planning
coordinates, not a claim that independently tokenizing the substring produces
the same token IDs. Every final substring is therefore preflighted again with
the exact prepared-input tokenizer before a span is admitted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, cast

from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)
from dithyramba.persistence.models import SourceFragmentText

from .hybrid_models import HybridContractError

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_TOKENIZER_IGNORABLE_GAP_CHARACTERS = frozenset({"\u200b", "\ufffd"})


class SemanticSpanContractError(HybridContractError):
    """A semantic span or span plan violates its immutable contract."""


class SemanticSpanPlanningError(SemanticSpanContractError):
    """The exact tokenizer could not produce a lossless bounded span plan."""


@dataclass(frozen=True, slots=True, order=True)
class ContentTokenOffset:
    """One Python-code-point range from the parent no-special-token mapping."""

    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if type(self.char_start) is not int or self.char_start < 0:
            raise SemanticSpanContractError("token char_start must be a non-negative integer")
        if type(self.char_end) is not int or self.char_end <= self.char_start:
            raise SemanticSpanContractError("token char_end must be greater than char_start")

    def payload(self) -> list[int]:
        return [self.char_start, self.char_end]


class _ProfileHashBinding(Protocol):
    @property
    def profile_hash(self) -> str: ...


class _ReceiptHashBinding(Protocol):
    @property
    def receipt_hash(self) -> str: ...


class SemanticSpanTokenizer(Protocol):
    """Protocol seam for the exact inference tokenizer, without an ML import.

    Offset mappings must use Python string code-point indices for ``text``,
    include content tokens only (``add_special_tokens=False``), and disable
    truncation. Prepared counts must use the same tokenizer with special tokens,
    the already-prefixed input, and ``truncation=False``.
    """

    @property
    def model_profile(self) -> _ProfileHashBinding: ...

    @property
    def runtime_profile(self) -> _ProfileHashBinding: ...

    @property
    def provisioning_receipt(self) -> _ReceiptHashBinding: ...

    def content_token_offsets_no_special_tokens(self, text: str) -> tuple[ContentTokenOffset, ...]:
        """Return the exact parent content-token offsets without truncation."""

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
        """Return the exact prepared count including special tokens."""


@dataclass(frozen=True, slots=True)
class SemanticSpanProfile:
    """Frozen v1 model and overlap policy for lossless parent spans."""

    SCHEMA: ClassVar[str] = "dithyramba.semantic_span_profile/1.0"

    model_profile_hash: str
    runtime_profile_hash: str
    provisioning_receipt_hash: str
    profile_version: Literal["1.0"] = "1.0"
    max_prepared_tokens: int = 512
    overlap_content_tokens: int = 64
    passage_prefix: str = "passage: "

    def __post_init__(self) -> None:
        if self.profile_version != "1.0":
            raise SemanticSpanContractError("unsupported SemanticSpanProfile version")
        for value in (
            self.model_profile_hash,
            self.runtime_profile_hash,
            self.provisioning_receipt_hash,
        ):
            _require_hash(value)
        if type(self.max_prepared_tokens) is not int or not 1 <= self.max_prepared_tokens <= 32_768:
            raise SemanticSpanContractError(
                "max_prepared_tokens must be an integer from 1 to 32768"
            )
        if (
            type(self.overlap_content_tokens) is not int
            or not 0 <= self.overlap_content_tokens < self.max_prepared_tokens
        ):
            raise SemanticSpanContractError(
                "overlap_content_tokens must be smaller than max_prepared_tokens"
            )
        _require_prefix(self.passage_prefix)

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "profile_version": self.profile_version,
            "model_profile_hash": self.model_profile_hash,
            "runtime_profile_hash": self.runtime_profile_hash,
            "provisioning_receipt_hash": self.provisioning_receipt_hash,
            "max_prepared_tokens": self.max_prepared_tokens,
            "overlap_content_tokens": self.overlap_content_tokens,
            "passage_prefix": self.passage_prefix,
        }

    @property
    def profile_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    @property
    def profile_id(self) -> str:
        return canonical_content_id("semantic_span_profile", self.payload())


@dataclass(frozen=True, slots=True)
class SemanticSpan:
    """Content-addressed, source-text-free window over one parent fragment."""

    SCHEMA: ClassVar[str] = "dithyramba.semantic_span/1.0"

    parent_fragment_id: str
    parent_text_sha256: str
    parent_ordinal: int
    span_ordinal: int
    char_start: int
    char_end: int
    content_token_start: int
    content_token_end: int
    span_text_sha256: str
    prepared_token_count: int
    profile_hash: str
    model_profile_hash: str

    def __post_init__(self) -> None:
        _require_id(self.parent_fragment_id, "fragment")
        _require_hash(self.parent_text_sha256)
        _require_hash(self.span_text_sha256)
        _require_hash(self.profile_hash)
        _require_hash(self.model_profile_hash)
        for value, label in (
            (self.parent_ordinal, "parent_ordinal"),
            (self.span_ordinal, "span_ordinal"),
            (self.char_start, "char_start"),
            (self.content_token_start, "content_token_start"),
        ):
            if type(value) is not int or value < 0:
                raise SemanticSpanContractError(f"{label} must be a non-negative integer")
        if type(self.char_end) is not int or self.char_end <= self.char_start:
            raise SemanticSpanContractError("char_end must be greater than char_start")
        if (
            type(self.content_token_end) is not int
            or self.content_token_end < self.content_token_start
        ):
            raise SemanticSpanContractError(
                "content_token_end must be at least content_token_start"
            )
        if type(self.prepared_token_count) is not int or self.prepared_token_count < 1:
            raise SemanticSpanContractError("prepared_token_count must be a positive integer")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "parent_fragment_id": self.parent_fragment_id,
            "parent_text_sha256": self.parent_text_sha256,
            "parent_ordinal": self.parent_ordinal,
            "span_ordinal": self.span_ordinal,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "content_token_start": self.content_token_start,
            "content_token_end": self.content_token_end,
            "span_text_sha256": self.span_text_sha256,
            "prepared_token_count": self.prepared_token_count,
            "profile_hash": self.profile_hash,
            "model_profile_hash": self.model_profile_hash,
        }

    @property
    def span_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    @property
    def span_id(self) -> str:
        return canonical_content_id("semantic_span", self.payload())


@dataclass(frozen=True, slots=True)
class SemanticSpanParentPlan:
    """Exact text-free span closure for one parent SourceFragment."""

    SCHEMA: ClassVar[str] = "dithyramba.semantic_span_parent_plan/1.0"
    OFFSET_SCHEMA: ClassVar[str] = "dithyramba.semantic_content_offsets/1.0"

    source_fragment_id: str
    source_version_id: str
    source_id: str
    ordinal: int
    text_sha256: str
    text_char_count: int
    content_token_count: int
    full_prepared_token_count: int
    content_offsets_hash: str
    spans: tuple[SemanticSpan, ...]

    def __post_init__(self) -> None:
        _require_id(self.source_fragment_id, "fragment")
        _require_id(self.source_version_id, "source_version")
        _require_id(self.source_id, "source")
        _require_hash(self.text_sha256)
        _require_hash(self.content_offsets_hash)
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise SemanticSpanContractError("parent ordinal must be a non-negative integer")
        if type(self.text_char_count) is not int or self.text_char_count < 1:
            raise SemanticSpanContractError("parent text_char_count must be positive")
        if type(self.content_token_count) is not int or self.content_token_count < 0:
            raise SemanticSpanContractError("parent content_token_count must be non-negative")
        if type(self.full_prepared_token_count) is not int or self.full_prepared_token_count < 1:
            raise SemanticSpanContractError("parent full_prepared_token_count must be positive")
        if type(self.spans) is not tuple or not self.spans:
            raise SemanticSpanContractError("every parent requires at least one semantic span")
        if any(type(span) is not SemanticSpan for span in self.spans):
            raise SemanticSpanContractError("parent spans must be exact SemanticSpan values")
        if tuple(span.span_ordinal for span in self.spans) != tuple(range(len(self.spans))):
            raise SemanticSpanContractError("span ordinals must be contiguous from zero")
        if any(
            span.parent_fragment_id != self.source_fragment_id
            or span.parent_text_sha256 != self.text_sha256
            or span.parent_ordinal != self.ordinal
            for span in self.spans
        ):
            raise SemanticSpanContractError("span parent binding differs from its closure")
        _validate_char_closure(self.spans, self.text_char_count)
        _validate_token_closure(self.spans, self.content_token_count)

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_version_id": self.source_version_id,
            "source_id": self.source_id,
            "ordinal": self.ordinal,
            "text_sha256": self.text_sha256,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            **self.identity_payload(),
            "text_char_count": self.text_char_count,
            "content_token_count": self.content_token_count,
            "full_prepared_token_count": self.full_prepared_token_count,
            "content_offsets_hash": self.content_offsets_hash,
            "spans": [
                span.payload() | {"span_id": span.span_id, "span_hash": span.span_hash}
                for span in self.spans
            ],
        }


@dataclass(frozen=True, slots=True)
class SemanticSpanPlan:
    """Canonical exact closure of parent fragments and their lossless spans."""

    SCHEMA: ClassVar[str] = "dithyramba.semantic_span_plan/1.0"
    MANIFEST_SCHEMA: ClassVar[str] = "dithyramba.semantic_span_parent_manifest/1.0"
    CLOSURE_SCHEMA: ClassVar[str] = "dithyramba.semantic_span_closure/1.0"

    profile: SemanticSpanProfile
    parents: tuple[SemanticSpanParentPlan, ...]

    def __post_init__(self) -> None:
        if type(self.profile) is not SemanticSpanProfile:
            raise SemanticSpanContractError("profile must be an exact SemanticSpanProfile")
        if type(self.parents) is not tuple:
            raise SemanticSpanContractError("span plan parents must be an exact tuple")
        if any(type(parent) is not SemanticSpanParentPlan for parent in self.parents):
            raise SemanticSpanContractError(
                "span plan parents must be exact SemanticSpanParentPlan values"
            )
        fragment_ids = tuple(parent.source_fragment_id for parent in self.parents)
        if len(set(fragment_ids)) != len(fragment_ids):
            raise SemanticSpanContractError("span plan parent fragment IDs must be unique")
        if fragment_ids != tuple(sorted(fragment_ids, key=lambda value: value.encode("ascii"))):
            raise SemanticSpanContractError(
                "span plan parents must use canonical fragment-ID order"
            )
        version_ordinals: set[tuple[str, int]] = set()
        version_sources: dict[str, str] = {}
        for parent in self.parents:
            version_ordinal = (parent.source_version_id, parent.ordinal)
            if version_ordinal in version_ordinals:
                raise SemanticSpanContractError(
                    "span plan SourceVersion ordinal pairs must be unique"
                )
            version_ordinals.add(version_ordinal)
            prior_source = version_sources.setdefault(parent.source_version_id, parent.source_id)
            if prior_source != parent.source_id:
                raise SemanticSpanContractError(
                    "one SourceVersion cannot belong to multiple Sources"
                )
            _validate_parent_against_profile(parent, self.profile)

    @property
    def spans(self) -> tuple[SemanticSpan, ...]:
        return tuple(span for parent in self.parents for span in parent.spans)

    @property
    def parent_manifest_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": self.MANIFEST_SCHEMA,
                "parents": [parent.identity_payload() for parent in self.parents],
            }
        )

    @property
    def span_closure_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": self.CLOSURE_SCHEMA,
                "span_ids": [span.span_id for span in self.spans],
            }
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "profile": self.profile.payload(),
            "profile_hash": self.profile.profile_hash,
            "parent_manifest_hash": self.parent_manifest_hash,
            "span_closure_hash": self.span_closure_hash,
            "counts": {"parents": len(self.parents), "spans": len(self.spans)},
            "parents": [parent.payload() for parent in self.parents],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def plan_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    @property
    def plan_id(self) -> str:
        return canonical_content_id("semantic_span_plan", self.payload())


def plan_semantic_spans(
    fragments: tuple[SourceFragmentText, ...],
    *,
    profile: SemanticSpanProfile,
    tokenizer: SemanticSpanTokenizer,
) -> SemanticSpanPlan:
    """Plan lossless parent windows and preflight every exact substring.

    The returned artifact contains hashes, offsets, counts and identities only.
    Source text exists solely inside this pure planning call.
    """

    if type(profile) is not SemanticSpanProfile:
        raise SemanticSpanContractError("profile must be an exact SemanticSpanProfile")
    validated = _validate_fragments(fragments)
    if not validated:
        return SemanticSpanPlan(profile=profile, parents=())
    offsetter = getattr(tokenizer, "content_token_offsets_no_special_tokens", None)
    counter = getattr(tokenizer, "count_prepared_tokens_no_truncation", None)
    if not callable(offsetter) or not callable(counter):
        raise SemanticSpanPlanningError("exact tokenizer offset and preflight methods are required")
    _validate_tokenizer_binding(tokenizer, profile)

    parents = tuple(
        _plan_parent(
            fragment,
            profile=profile,
            offsetter=cast("_Offsetter", offsetter),
            counter=cast("_Counter", counter),
        )
        for fragment in validated
    )
    return SemanticSpanPlan(profile=profile, parents=parents)


class _Offsetter(Protocol):
    def __call__(self, text: str) -> tuple[ContentTokenOffset, ...]: ...


class _Counter(Protocol):
    def __call__(self, prepared_text: str) -> int: ...


def _plan_parent(
    fragment: SourceFragmentText,
    *,
    profile: SemanticSpanProfile,
    offsetter: _Offsetter,
    counter: _Counter,
) -> SemanticSpanParentPlan:
    try:
        raw_offsets = offsetter(fragment.text)
    except Exception as exc:
        raise SemanticSpanPlanningError("content token offset mapping failed") from exc
    offsets = _validate_offsets(raw_offsets, fragment.text)
    offsets_hash = canonical_sha256_hex(
        {
            "schema": SemanticSpanParentPlan.OFFSET_SCHEMA,
            "source_fragment_id": fragment.source_fragment_id,
            "text_sha256": fragment.text_sha256,
            "offsets": [offset.payload() for offset in offsets],
        }
    )
    full_count = _count_prepared(
        counter,
        f"{profile.passage_prefix}{fragment.text}",
    )
    if full_count <= profile.max_prepared_tokens:
        spans: tuple[SemanticSpan, ...] = (
            _build_span(
                fragment,
                profile=profile,
                span_ordinal=0,
                char_start=0,
                char_end=len(fragment.text),
                token_start=0,
                token_end=len(offsets),
                prepared_token_count=full_count,
            ),
        )
    else:
        spans = _window_parent(
            fragment,
            offsets=offsets,
            profile=profile,
            counter=counter,
        )
    return SemanticSpanParentPlan(
        source_fragment_id=fragment.source_fragment_id,
        source_version_id=fragment.source_version_id,
        source_id=fragment.source_id,
        ordinal=fragment.ordinal,
        text_sha256=fragment.text_sha256,
        text_char_count=len(fragment.text),
        content_token_count=len(offsets),
        full_prepared_token_count=full_count,
        content_offsets_hash=offsets_hash,
        spans=spans,
    )


def _window_parent(
    fragment: SourceFragmentText,
    *,
    offsets: tuple[ContentTokenOffset, ...],
    profile: SemanticSpanProfile,
    counter: _Counter,
) -> tuple[SemanticSpan, ...]:
    spans: list[SemanticSpan] = []
    start = 0
    while start < len(offsets):
        end = min(len(offsets), start + profile.max_prepared_tokens)
        accepted: tuple[int, int, int, int] | None = None
        while end > start:
            char_start = 0 if start == 0 else offsets[start].char_start
            char_end = (
                len(fragment.text)
                if end == len(offsets)
                else max(offsets[end - 1].char_end, offsets[end].char_start)
            )
            if char_end <= char_start:
                raise SemanticSpanPlanningError("token offsets produce a zero-length window")
            span_text = fragment.text[char_start:char_end]
            prepared_count = _count_prepared(
                counter,
                f"{profile.passage_prefix}{span_text}",
            )
            if prepared_count <= profile.max_prepared_tokens:
                accepted = (end, char_start, char_end, prepared_count)
                break
            end -= 1
        if accepted is None:
            raise SemanticSpanPlanningError(
                "no non-empty content-token window fits max_prepared_tokens"
            )
        end, char_start, char_end, accepted_prepared_count = accepted
        if end < len(offsets) and end - start <= profile.overlap_content_tokens:
            raise SemanticSpanPlanningError(
                "overlap leaves no forward progress for a bounded semantic window"
            )
        prepared_count = _count_prepared(
            counter,
            f"{profile.passage_prefix}{fragment.text[char_start:char_end]}",
        )
        if prepared_count > profile.max_prepared_tokens:
            raise SemanticSpanPlanningError("repeated exact span preflight exceeds profile budget")
        if prepared_count != accepted_prepared_count:
            raise SemanticSpanPlanningError("repeated exact span preflight count drifted")
        spans.append(
            _build_span(
                fragment,
                profile=profile,
                span_ordinal=len(spans),
                char_start=char_start,
                char_end=char_end,
                token_start=start,
                token_end=end,
                prepared_token_count=prepared_count,
            )
        )
        if end == len(offsets):
            break
        start = end - profile.overlap_content_tokens
    if not spans:
        raise SemanticSpanPlanningError(
            "an over-budget parent has no content-token window anchors; "
            "planner produced zero windows"
        )
    return tuple(spans)


def _build_span(
    fragment: SourceFragmentText,
    *,
    profile: SemanticSpanProfile,
    span_ordinal: int,
    char_start: int,
    char_end: int,
    token_start: int,
    token_end: int,
    prepared_token_count: int,
) -> SemanticSpan:
    span_text = fragment.text[char_start:char_end]
    if not span_text:
        raise SemanticSpanPlanningError("semantic span text cannot be empty")
    return SemanticSpan(
        parent_fragment_id=fragment.source_fragment_id,
        parent_text_sha256=fragment.text_sha256,
        parent_ordinal=fragment.ordinal,
        span_ordinal=span_ordinal,
        char_start=char_start,
        char_end=char_end,
        content_token_start=token_start,
        content_token_end=token_end,
        span_text_sha256=sha256_hex(span_text.encode("utf-8")),
        prepared_token_count=prepared_token_count,
        profile_hash=profile.profile_hash,
        model_profile_hash=profile.model_profile_hash,
    )


def _validate_fragments(
    fragments: tuple[SourceFragmentText, ...],
) -> tuple[SourceFragmentText, ...]:
    if type(fragments) is not tuple:
        raise SemanticSpanContractError("fragments must be an exact tuple")
    if any(type(fragment) is not SourceFragmentText for fragment in fragments):
        raise SemanticSpanContractError("fragments must contain exact SourceFragmentText values")
    for fragment in fragments:
        _require_id(fragment.source_fragment_id, "fragment")
        _require_id(fragment.source_version_id, "source_version")
        _require_id(fragment.source_id, "source")
        if type(fragment.ordinal) is not int or fragment.ordinal < 0:
            raise SemanticSpanContractError("fragment ordinal must be a non-negative integer")
        if (
            type(fragment.text) is not str
            or not fragment.text
            or "\x00" in fragment.text
            or unicodedata.normalize("NFC", fragment.text) != fragment.text
        ):
            raise SemanticSpanContractError("fragment text must be non-empty exact NFC text")
        try:
            text_bytes = fragment.text.encode("utf-8")
        except UnicodeEncodeError:
            raise SemanticSpanContractError("fragment text must be valid UTF-8 text") from None
        _require_hash(fragment.text_sha256)
        if sha256_hex(text_bytes) != fragment.text_sha256:
            raise SemanticSpanContractError("fragment text does not match text_sha256")

    ordered = tuple(sorted(fragments, key=lambda item: item.source_fragment_id.encode("ascii")))
    fragment_ids: set[str] = set()
    version_ordinals: set[tuple[str, int]] = set()
    version_sources: dict[str, str] = {}
    for fragment in ordered:
        if fragment.source_fragment_id in fragment_ids:
            raise SemanticSpanContractError("fragment IDs must be unique")
        fragment_ids.add(fragment.source_fragment_id)
        version_ordinal = (fragment.source_version_id, fragment.ordinal)
        if version_ordinal in version_ordinals:
            raise SemanticSpanContractError("fragment SourceVersion ordinal pairs must be unique")
        version_ordinals.add(version_ordinal)
        prior_source = version_sources.setdefault(fragment.source_version_id, fragment.source_id)
        if prior_source != fragment.source_id:
            raise SemanticSpanContractError("one SourceVersion cannot belong to multiple Sources")
    return ordered


def _validate_offsets(
    offsets: object,
    text: str,
) -> tuple[ContentTokenOffset, ...]:
    if type(offsets) is not tuple:
        raise SemanticSpanPlanningError("content token offsets must be an exact tuple")
    values = cast(tuple[object, ...], offsets)
    if any(type(offset) is not ContentTokenOffset for offset in values):
        raise SemanticSpanPlanningError(
            "content token offsets must contain exact ContentTokenOffset values"
        )
    typed = cast(tuple[ContentTokenOffset, ...], values)
    previous_start = -1
    union_end = 0
    for offset in typed:
        if offset.char_end > len(text):
            raise SemanticSpanPlanningError("content token offset exceeds parent text")
        if offset.char_start < previous_start or offset.char_end < union_end:
            raise SemanticSpanPlanningError("content token offsets must be nondecreasing")
        uncovered_position = _first_uncovered_non_ignorable_position(
            text,
            union_end,
            offset.char_start,
        )
        if uncovered_position is not None:
            raise SemanticSpanPlanningError(
                "content token offsets leave an uncovered non-whitespace gap "
                f"at position {uncovered_position} "
                f"(U+{ord(text[uncovered_position]):04X})"
            )
        previous_start = offset.char_start
        union_end = offset.char_end
    uncovered_position = _first_uncovered_non_ignorable_position(
        text,
        union_end,
        len(text),
    )
    if uncovered_position is not None:
        raise SemanticSpanPlanningError(
            "content token offsets leave an uncovered trailing non-whitespace gap "
            f"at position {uncovered_position} "
            f"(U+{ord(text[uncovered_position]):04X})"
        )
    return typed


def _first_uncovered_non_ignorable_position(
    text: str,
    start: int,
    end: int,
) -> int | None:
    for position in range(start, end):
        character = text[position]
        if (
            character.isspace()
            or (position == 0 and character == "\ufeff")
            or character in _TOKENIZER_IGNORABLE_GAP_CHARACTERS
        ):
            continue
        return position
    return None


def _count_prepared(counter: _Counter, prepared_text: str) -> int:
    try:
        count = counter(prepared_text)
    except Exception as exc:
        raise SemanticSpanPlanningError("exact prepared-token preflight failed") from exc
    if type(count) is not int or count < 1:
        raise SemanticSpanPlanningError("exact prepared-token preflight returned an invalid count")
    return count


def _validate_tokenizer_binding(
    tokenizer: object,
    profile: SemanticSpanProfile,
) -> None:
    try:
        bound = cast(SemanticSpanTokenizer, tokenizer)
        observed = (
            bound.model_profile.profile_hash,
            bound.runtime_profile.profile_hash,
            bound.provisioning_receipt.receipt_hash,
        )
    except Exception as exc:
        raise SemanticSpanPlanningError("exact tokenizer profile binding is unavailable") from exc
    expected = (
        profile.model_profile_hash,
        profile.runtime_profile_hash,
        profile.provisioning_receipt_hash,
    )
    if any(type(value) is not str for value in observed) or observed != expected:
        raise SemanticSpanPlanningError("exact tokenizer profile binding differs")


def _validate_char_closure(spans: tuple[SemanticSpan, ...], text_char_count: int) -> None:
    if spans[0].char_start != 0 or spans[-1].char_end != text_char_count:
        raise SemanticSpanContractError("semantic spans must cover both parent char boundaries")
    previous_end = 0
    for span in spans:
        if span.char_start > previous_end:
            raise SemanticSpanContractError("semantic span char closure contains a gap")
        if span.char_end <= previous_end:
            raise SemanticSpanContractError("semantic span char closure makes no progress")
        if span.char_end > text_char_count:
            raise SemanticSpanContractError("semantic span exceeds parent char boundary")
        previous_end = span.char_end


def _validate_token_closure(spans: tuple[SemanticSpan, ...], content_token_count: int) -> None:
    if spans[0].content_token_start != 0 or spans[-1].content_token_end != content_token_count:
        raise SemanticSpanContractError("semantic spans must cover both parent token boundaries")
    previous_end = 0
    for span in spans:
        if span.content_token_start > previous_end:
            raise SemanticSpanContractError("semantic span token closure contains a gap")
        if content_token_count and span.content_token_end <= previous_end:
            raise SemanticSpanContractError("semantic span token closure makes no progress")
        if span.content_token_end > content_token_count:
            raise SemanticSpanContractError("semantic span exceeds parent token boundary")
        previous_end = span.content_token_end


def _validate_parent_against_profile(
    parent: SemanticSpanParentPlan,
    profile: SemanticSpanProfile,
) -> None:
    for span in parent.spans:
        if (
            span.profile_hash != profile.profile_hash
            or span.model_profile_hash != profile.model_profile_hash
        ):
            raise SemanticSpanContractError("semantic span profile/model binding differs")
        if span.prepared_token_count > profile.max_prepared_tokens:
            raise SemanticSpanContractError("semantic span exceeds max_prepared_tokens")
    if parent.full_prepared_token_count <= profile.max_prepared_tokens:
        if len(parent.spans) != 1:
            raise SemanticSpanContractError("a fitting parent must have exactly one span")
        span = parent.spans[0]
        if (
            span.char_start != 0
            or span.char_end != parent.text_char_count
            or span.content_token_start != 0
            or span.content_token_end != parent.content_token_count
            or span.prepared_token_count != parent.full_prepared_token_count
        ):
            raise SemanticSpanContractError("a fitting parent span must equal the full text")
        return
    if parent.content_token_count == 0:
        raise SemanticSpanContractError(
            "an over-budget parent requires content-token window anchors"
        )
    if len(parent.spans) < 2:
        raise SemanticSpanContractError("an over-budget parent requires multiple spans")
    for previous, current in zip(parent.spans, parent.spans[1:], strict=False):
        overlap = previous.content_token_end - current.content_token_start
        if overlap != profile.overlap_content_tokens:
            raise SemanticSpanContractError("semantic span token overlap differs from profile")
        if (
            previous.content_token_end - previous.content_token_start
            <= profile.overlap_content_tokens
        ):
            raise SemanticSpanContractError(
                "a non-final semantic span must advance beyond the configured overlap"
            )


def _require_hash(value: object) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise SemanticSpanContractError("artifact hash must be 64 lowercase hexadecimal characters")
    return value


def _require_id(value: object, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise SemanticSpanContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise SemanticSpanContractError(f"identifier must use a canonical {expected} suffix")
    return value


def _require_prefix(value: object) -> str:
    if type(value) is not str or len(value) > 128 or "\x00" in value:
        raise SemanticSpanContractError("passage_prefix must be bounded text without NUL")
    if unicodedata.normalize("NFC", value) != value:
        raise SemanticSpanContractError("passage_prefix must already be NFC")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise SemanticSpanContractError("passage_prefix must be valid UTF-8 text") from None
    return value
