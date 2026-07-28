"""Pure lossless semantic-span planning and contract tests."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

import dithyramba.recall.semantic_spans as semantic_spans
from dithyramba.contracts import sha256_hex
from dithyramba.persistence.models import SourceFragmentText
from dithyramba.recall.semantic_spans import (
    ContentTokenOffset,
    SemanticSpan,
    SemanticSpanContractError,
    SemanticSpanParentPlan,
    SemanticSpanPlan,
    SemanticSpanPlanningError,
    SemanticSpanProfile,
    SemanticSpanTokenizer,
    plan_semantic_spans,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class _ProfileBinding:
    def __init__(self, profile_hash: str) -> None:
        self.profile_hash = profile_hash


class _ReceiptBinding:
    def __init__(self, receipt_hash: str) -> None:
        self.receipt_hash = receipt_hash


def _profile(
    *,
    max_tokens: int = 5,
    overlap: int = 1,
    prefix: str = "passage: ",
) -> SemanticSpanProfile:
    return SemanticSpanProfile(
        model_profile_hash=HASH_A,
        runtime_profile_hash=HASH_B,
        provisioning_receipt_hash=HASH_C,
        max_prepared_tokens=max_tokens,
        overlap_content_tokens=overlap,
        passage_prefix=prefix,
    )


def _fragment(
    fragment_id: str,
    text: str,
    *,
    ordinal: int = 0,
    version: str = "source_version_one",
    source: str = "source_one",
) -> SourceFragmentText:
    return SourceFragmentText(
        source_fragment_id=fragment_id,
        source_version_id=version,
        source_id=source,
        ordinal=ordinal,
        fragment_kind="paragraph",
        text=text,
        text_sha256=sha256_hex(text.encode("utf-8")),
        source_address_json="{}",
    )


class WordTokenizer:
    """Unicode-aware test adapter with ignored whitespace offsets."""

    def __init__(
        self,
        *,
        special_tokens: int = 2,
        count_overrides: dict[str, int] | None = None,
        model_profile_hash: str = HASH_A,
        runtime_profile_hash: str = HASH_B,
        provisioning_receipt_hash: str = HASH_C,
    ) -> None:
        self.special_tokens = special_tokens
        self.count_overrides = count_overrides or {}
        self.offset_calls: list[str] = []
        self.count_calls: list[str] = []
        self.model_profile = _ProfileBinding(model_profile_hash)
        self.runtime_profile = _ProfileBinding(runtime_profile_hash)
        self.provisioning_receipt = _ReceiptBinding(provisioning_receipt_hash)

    def content_token_offsets_no_special_tokens(self, text: str) -> tuple[ContentTokenOffset, ...]:
        self.offset_calls.append(text)
        return tuple(
            ContentTokenOffset(match.start(), match.end()) for match in re.finditer(r"\S+", text)
        )

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
        self.count_calls.append(prepared_text)
        return self.count_overrides.get(
            prepared_text,
            len(tuple(re.finditer(r"\S+", prepared_text))) + self.special_tokens,
        )


class StaticTokenizer:
    def __init__(
        self,
        offsets: object,
        counts: tuple[object, ...] = (1,),
        *,
        model_profile_hash: str = HASH_A,
        runtime_profile_hash: str = HASH_B,
        provisioning_receipt_hash: str = HASH_C,
    ) -> None:
        self.offsets = offsets
        self.counts = counts
        self.offset_calls = 0
        self.count_calls = 0
        self.model_profile = _ProfileBinding(model_profile_hash)
        self.runtime_profile = _ProfileBinding(runtime_profile_hash)
        self.provisioning_receipt = _ReceiptBinding(provisioning_receipt_hash)

    def content_token_offsets_no_special_tokens(self, text: str) -> object:
        del text
        self.offset_calls += 1
        if isinstance(self.offsets, BaseException):
            raise self.offsets
        return self.offsets

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> object:
        del prepared_text
        result = self.counts[min(self.count_calls, len(self.counts) - 1)]
        self.count_calls += 1
        if isinstance(result, BaseException):
            raise result
        return result


def test_fitting_multilingual_fragment_is_one_exact_full_text_span() -> None:
    text = "Привіт 🌍"
    fragment = _fragment("fragment_fit", text)
    tokenizer = WordTokenizer()

    plan = plan_semantic_spans((fragment,), profile=_profile(), tokenizer=tokenizer)

    assert len(plan.parents) == 1
    assert len(plan.spans) == 1
    span = plan.spans[0]
    assert (span.char_start, span.char_end) == (0, len(text))
    assert (span.content_token_start, span.content_token_end) == (0, 2)
    assert span.span_text_sha256 == sha256_hex(text.encode("utf-8"))
    assert span.prepared_token_count == 5
    assert span.parent_text_sha256 == fragment.text_sha256
    assert span.parent_ordinal == 0
    assert span.profile_hash == plan.profile.profile_hash
    assert span.model_profile_hash == HASH_A
    assert span.span_id.startswith("semantic_span_")
    assert len(span.span_hash) == 64
    assert tokenizer.offset_calls == [text]
    assert tokenizer.count_calls == [f"passage: {text}"]


def test_oversized_unicode_parent_has_exact_overlap_and_lossless_char_closure() -> None:
    text = "Привіт світе 你好 café"
    fragment = _fragment("fragment_long", text)
    tokenizer = WordTokenizer()

    plan = plan_semantic_spans((fragment,), profile=_profile(), tokenizer=tokenizer)

    parent = plan.parents[0]
    assert parent.full_prepared_token_count == 7
    assert parent.content_token_count == 4
    assert len(parent.spans) == 3
    assert [(span.content_token_start, span.content_token_end) for span in parent.spans] == [
        (0, 2),
        (1, 3),
        (2, 4),
    ]
    assert all(
        previous.content_token_end - current.content_token_start == 1
        for previous, current in zip(parent.spans, parent.spans[1:], strict=False)
    )
    covered = [False] * len(text)
    for span in parent.spans:
        assert span.prepared_token_count <= 5
        for index in range(span.char_start, span.char_end):
            covered[index] = True
    assert all(covered)
    assert parent.spans[0].char_start == 0
    assert parent.spans[-1].char_end == len(text)
    assert len(parent.content_offsets_hash) == 64


def test_every_candidate_and_final_substring_is_exactly_preflighted_again() -> None:
    text = "alpha beta gamma delta"
    fragment = _fragment("fragment_boundary", text)
    tokenizer = WordTokenizer()

    plan = plan_semantic_spans((fragment,), profile=_profile(), tokenizer=tokenizer)

    assert len(plan.spans) == 3
    final_inputs = {f"passage: {text[span.char_start : span.char_end]}" for span in plan.spans}
    assert all(tokenizer.count_calls.count(value) >= 2 for value in final_inputs)
    assert tokenizer.count_calls.count(f"passage: {text}") >= 2


def test_plan_is_deterministic_canonical_and_contains_no_source_text() -> None:
    alpha = _fragment("fragment_alpha", "classified один два три чотири", ordinal=0)
    beta = _fragment("fragment_beta", "секретний текст", ordinal=1)

    first = plan_semantic_spans((beta, alpha), profile=_profile(), tokenizer=WordTokenizer())
    second = plan_semantic_spans((alpha, beta), profile=_profile(), tokenizer=WordTokenizer())

    assert first == second
    assert tuple(parent.source_fragment_id for parent in first.parents) == (
        "fragment_alpha",
        "fragment_beta",
    )
    assert first.canonical_bytes == second.canonical_bytes
    assert b"classified" not in first.canonical_bytes
    assert "секретний".encode() not in first.canonical_bytes
    assert first.plan_id.startswith("semantic_span_plan_")
    assert len(first.plan_hash) == 64
    assert len(first.parent_manifest_hash) == 64
    assert len(first.span_closure_hash) == 64
    assert first.payload()["counts"] == {"parents": 2, "spans": 5}


def test_whitespace_only_fragment_can_fit_without_fabricating_content_tokens() -> None:
    fragment = _fragment("fragment_spaces", "   \t")

    plan = plan_semantic_spans((fragment,), profile=_profile(), tokenizer=WordTokenizer())

    parent = plan.parents[0]
    assert parent.content_token_count == 0
    assert len(parent.spans) == 1
    assert parent.spans[0].content_token_start == 0
    assert parent.spans[0].content_token_end == 0


def test_empty_authorized_closure_produces_a_canonical_empty_plan_without_tokenizer() -> None:
    first = plan_semantic_spans((), profile=_profile(), tokenizer=cast(Any, object()))
    second = plan_semantic_spans((), profile=_profile(), tokenizer=cast(Any, object()))

    assert first == second
    assert first.parents == ()
    assert first.spans == ()
    assert first.payload()["counts"] == {"parents": 0, "spans": 0}
    assert first.plan_id.startswith("semantic_span_plan_")
    assert len(first.parent_manifest_hash) == 64
    assert len(first.span_closure_hash) == 64


@pytest.mark.parametrize(
    ("offsets", "text"),
    [
        ([ContentTokenOffset(0, 1)], "a"),
        ((object(),), "a"),
        ((ContentTokenOffset(0, 4),), "abc"),
        ((ContentTokenOffset(1, 3), ContentTokenOffset(0, 5)), " abcd"),
        ((ContentTokenOffset(0, 3), ContentTokenOffset(1, 2)), "abc"),
        ((ContentTokenOffset(0, 1), ContentTokenOffset(2, 3)), "abc"),
        ((ContentTokenOffset(1, 3),), "abc"),
        ((ContentTokenOffset(0, 2),), "abc"),
    ],
)
def test_invalid_or_lossy_offset_mappings_fail_closed(offsets: object, text: str) -> None:
    tokenizer = StaticTokenizer(offsets)

    with pytest.raises(SemanticSpanPlanningError):
        plan_semantic_spans(
            (_fragment("fragment_bad_offsets", text),),
            profile=_profile(),
            tokenizer=cast(SemanticSpanTokenizer, tokenizer),
        )


def test_whitespace_gaps_in_offsets_are_allowed_but_non_whitespace_gaps_are_not() -> None:
    text = " alpha  beta "
    offsets = (ContentTokenOffset(1, 6), ContentTokenOffset(8, 12))
    tokenizer = StaticTokenizer(offsets, (5,))

    plan = plan_semantic_spans(
        (_fragment("fragment_whitespace", text),),
        profile=_profile(),
        tokenizer=cast(SemanticSpanTokenizer, tokenizer),
    )

    assert plan.spans[0].char_start == 0
    assert plan.spans[0].char_end == len(text)


def test_tokenizer_ignored_leading_bom_remains_in_exact_span_closure_and_hash() -> None:
    text = "\ufeffalpha beta"
    offsets = (ContentTokenOffset(1, 6), ContentTokenOffset(7, 11))

    first = plan_semantic_spans(
        (_fragment("fragment_leading_bom", text),),
        profile=_profile(),
        tokenizer=cast(SemanticSpanTokenizer, StaticTokenizer(offsets, (5,))),
    )
    second = plan_semantic_spans(
        (_fragment("fragment_leading_bom", text),),
        profile=_profile(),
        tokenizer=cast(SemanticSpanTokenizer, StaticTokenizer(offsets, (5,))),
    )

    span = first.spans[0]
    assert first == second
    assert first.canonical_bytes == second.canonical_bytes
    assert (span.char_start, span.char_end) == (0, len(text))
    assert span.span_text_sha256 == sha256_hex(text.encode("utf-8"))
    assert span.span_text_sha256 != sha256_hex(text[1:].encode("utf-8"))
    assert "\ufeff".encode("utf-8") not in first.canonical_bytes


@pytest.mark.parametrize("ignored_character", ["\u200b", "\ufffd"])
def test_tokenizer_ignored_character_remains_in_exact_span_closure_and_hash(
    ignored_character: str,
) -> None:
    text = f"alpha{ignored_character}beta"
    offsets = (ContentTokenOffset(0, 5), ContentTokenOffset(6, 10))

    plan = plan_semantic_spans(
        (_fragment("fragment_replacement_character", text),),
        profile=_profile(),
        tokenizer=cast(SemanticSpanTokenizer, StaticTokenizer(offsets, (5,))),
    )

    span = plan.spans[0]
    assert (span.char_start, span.char_end) == (0, len(text))
    assert span.span_text_sha256 == sha256_hex(text.encode("utf-8"))
    assert span.span_text_sha256 != sha256_hex(text.replace(ignored_character, "").encode("utf-8"))


@pytest.mark.parametrize(
    ("text", "offsets", "diagnostic"),
    [
        (
            "a\ufeffb",
            (ContentTokenOffset(0, 1), ContentTokenOffset(2, 3)),
            "position 1 (U+FEFF)",
        ),
        ("xa", (ContentTokenOffset(1, 2),), "position 0 (U+0078)"),
    ],
)
def test_only_a_bom_at_parent_position_zero_may_be_uncovered(
    text: str,
    offsets: tuple[ContentTokenOffset, ...],
    diagnostic: str,
) -> None:
    with pytest.raises(SemanticSpanPlanningError, match=re.escape(diagnostic)):
        plan_semantic_spans(
            (_fragment("fragment_disallowed_gap", text),),
            profile=_profile(),
            tokenizer=cast(SemanticSpanTokenizer, StaticTokenizer(offsets, (5,))),
        )


@pytest.mark.parametrize(
    ("text", "offsets"),
    [
        (
            " " * 165 + "abcdefghi",
            (ContentTokenOffset(165, 166), ContentTokenOffset(165, 174)),
        ),
        ("🌍", (ContentTokenOffset(0, 1), ContentTokenOffset(0, 1))),
        ("abc", (ContentTokenOffset(0, 2), ContentTokenOffset(1, 3))),
    ],
)
def test_nondecreasing_overlapping_and_duplicate_unicode_offsets_are_lossless(
    text: str,
    offsets: tuple[ContentTokenOffset, ...],
) -> None:
    plan = plan_semantic_spans(
        (_fragment("fragment_overlap_shape", text),),
        profile=_profile(),
        tokenizer=cast(SemanticSpanTokenizer, StaticTokenizer(offsets, (5,))),
    )

    parent = plan.parents[0]
    assert parent.content_token_count == len(offsets)
    assert parent.spans[0].char_start == 0
    assert parent.spans[0].char_end == len(text)


def test_overlapping_offsets_keep_full_admitted_tokens_and_exact_char_union() -> None:
    text = "abc def"
    offsets = (
        ContentTokenOffset(0, 1),
        ContentTokenOffset(0, 3),
        ContentTokenOffset(4, 7),
    )
    tokenizer = StaticTokenizer(offsets, (10, 10, 2, 2, 10, 2, 2, 2, 2))

    plan = plan_semantic_spans(
        (_fragment("fragment_overlap_windows", text),),
        profile=_profile(max_tokens=2, overlap=0),
        tokenizer=cast(SemanticSpanTokenizer, tokenizer),
    )

    assert [
        (span.content_token_start, span.content_token_end, span.char_start, span.char_end)
        for span in plan.spans
    ] == [
        (0, 1, 0, 1),
        (1, 2, 0, 4),
        (2, 3, 4, 7),
    ]
    covered = [False] * len(text)
    for span in plan.spans:
        for index in range(span.char_start, span.char_end):
            covered[index] = True
    assert all(covered)


@pytest.mark.parametrize(
    "tokenizer",
    [
        StaticTokenizer(RuntimeError("offset secret")),
        StaticTokenizer((ContentTokenOffset(0, 1),), (RuntimeError("count secret"),)),
        StaticTokenizer((ContentTokenOffset(0, 1),), (False,)),
        StaticTokenizer((ContentTokenOffset(0, 1),), (0,)),
    ],
)
def test_tokenizer_failures_and_invalid_counts_fail_closed_without_details(
    tokenizer: StaticTokenizer,
) -> None:
    with pytest.raises(SemanticSpanPlanningError) as caught:
        plan_semantic_spans(
            (_fragment("fragment_tokenizer", "a"),),
            profile=_profile(),
            tokenizer=cast(SemanticSpanTokenizer, tokenizer),
        )

    assert "secret" not in str(caught.value)


def test_missing_tokenizer_protocol_fails_closed() -> None:
    with pytest.raises(SemanticSpanPlanningError, match="methods"):
        plan_semantic_spans(
            (_fragment("fragment_missing", "a"),),
            profile=_profile(),
            tokenizer=cast(SemanticSpanTokenizer, object()),
        )


@pytest.mark.parametrize(
    "tokenizer",
    [
        WordTokenizer(model_profile_hash=HASH_B),
        WordTokenizer(runtime_profile_hash=HASH_C),
        WordTokenizer(provisioning_receipt_hash=HASH_A),
    ],
)
def test_tokenizer_profile_binding_must_match_the_requested_profile(
    tokenizer: WordTokenizer,
) -> None:
    with pytest.raises(SemanticSpanPlanningError, match="profile binding differs"):
        plan_semantic_spans(
            (_fragment("fragment_binding_mismatch", "a"),),
            profile=_profile(),
            tokenizer=tokenizer,
        )


def test_tokenizer_profile_binding_must_be_available_and_exact_text() -> None:
    unavailable = WordTokenizer()
    del cast(Any, unavailable).model_profile
    with pytest.raises(SemanticSpanPlanningError, match="binding is unavailable"):
        plan_semantic_spans(
            (_fragment("fragment_binding_missing", "a"),),
            profile=_profile(),
            tokenizer=unavailable,
        )

    invalid_type = WordTokenizer()
    cast(Any, invalid_type).model_profile.profile_hash = 7
    with pytest.raises(SemanticSpanPlanningError, match="profile binding differs"):
        plan_semantic_spans(
            (_fragment("fragment_binding_type", "a"),),
            profile=_profile(),
            tokenizer=invalid_type,
        )


def test_overbudget_parent_fails_when_no_lossless_window_can_progress() -> None:
    empty_offsets = StaticTokenizer((), (10,))
    with pytest.raises(SemanticSpanPlanningError, match="no content-token"):
        plan_semantic_spans(
            (_fragment("fragment_empty_offsets", "   "),),
            profile=_profile(),
            tokenizer=cast(SemanticSpanTokenizer, empty_offsets),
        )

    never_fits = StaticTokenizer((ContentTokenOffset(0, 1),), (10,))
    with pytest.raises(SemanticSpanPlanningError, match="no non-empty"):
        plan_semantic_spans(
            (_fragment("fragment_never_fits", "a"),),
            profile=_profile(),
            tokenizer=cast(SemanticSpanTokenizer, never_fits),
        )

    no_progress = WordTokenizer(special_tokens=1)
    with pytest.raises(SemanticSpanPlanningError, match="no forward progress"):
        plan_semantic_spans(
            (_fragment("fragment_no_progress", "a b c d"),),
            profile=_profile(max_tokens=3, overlap=2),
            tokenizer=no_progress,
        )


def test_repeated_final_preflight_cannot_be_bypassed() -> None:
    tokenizer = StaticTokenizer(
        (ContentTokenOffset(0, 1), ContentTokenOffset(2, 3)),
        (10, 5, 6),
    )

    with pytest.raises(SemanticSpanPlanningError, match="repeated"):
        plan_semantic_spans(
            (_fragment("fragment_changed_count", "a b"),),
            profile=_profile(),
            tokenizer=cast(SemanticSpanTokenizer, tokenizer),
        )


def test_repeated_final_preflight_fails_closed_on_in_budget_count_drift() -> None:
    tokenizer = StaticTokenizer(
        (ContentTokenOffset(0, 1), ContentTokenOffset(2, 3)),
        (10, 5, 4),
    )

    with pytest.raises(SemanticSpanPlanningError, match="count drifted"):
        plan_semantic_spans(
            (_fragment("fragment_nondeterministic_count", "a b"),),
            profile=_profile(),
            tokenizer=cast(SemanticSpanTokenizer, tokenizer),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile_version", "2.0"),
        ("model_profile_hash", "A" * 64),
        ("max_prepared_tokens", True),
        ("max_prepared_tokens", 0),
        ("max_prepared_tokens", 32_769),
        ("overlap_content_tokens", True),
        ("overlap_content_tokens", -1),
        ("overlap_content_tokens", 5),
        ("passage_prefix", "e\u0301"),
        ("passage_prefix", "bad\x00"),
        ("passage_prefix", "x" * 129),
        ("passage_prefix", "\ud800"),
    ],
)
def test_profile_contract_is_strict(field: str, value: object) -> None:
    with pytest.raises(SemanticSpanContractError):
        replace(_profile(), **cast(Any, {field: value}))


def test_profile_is_versioned_and_content_addressed() -> None:
    profile = _profile()

    assert profile.payload()["schema"] == "dithyramba.semantic_span_profile/1.0"
    assert profile.profile_id.startswith("semantic_span_profile_")
    assert len(profile.profile_hash) == 64


def test_profile_defaults_to_512_prepared_tokens_and_64_parent_token_overlap() -> None:
    profile = SemanticSpanProfile(
        model_profile_hash=HASH_A,
        runtime_profile_hash=HASH_B,
        provisioning_receipt_hash=HASH_C,
    )

    assert profile.max_prepared_tokens == 512
    assert profile.overlap_content_tokens == 64


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ContentTokenOffset(-1, 1),
        lambda: ContentTokenOffset(1, 1),
    ],
)
def test_content_token_offset_contract_is_strict(factory: Callable[[], object]) -> None:
    with pytest.raises(SemanticSpanContractError):
        factory()


def _span(**changes: object) -> SemanticSpan:
    values: dict[str, object] = {
        "parent_fragment_id": "fragment_parent",
        "parent_text_sha256": HASH_A,
        "parent_ordinal": 0,
        "span_ordinal": 0,
        "char_start": 0,
        "char_end": 3,
        "content_token_start": 0,
        "content_token_end": 1,
        "span_text_sha256": HASH_B,
        "prepared_token_count": 3,
        "profile_hash": _profile().profile_hash,
        "model_profile_hash": HASH_A,
    }
    values.update(changes)
    return SemanticSpan(**cast(Any, values))


@pytest.mark.parametrize(
    "changes",
    [
        {"parent_fragment_id": "bad"},
        {"parent_text_sha256": "bad"},
        {"parent_ordinal": -1},
        {"span_ordinal": True},
        {"char_start": -1},
        {"char_end": 0},
        {"content_token_start": -1},
        {"content_token_end": -1},
        {"prepared_token_count": False},
    ],
)
def test_semantic_span_contract_rejects_invalid_coordinates(changes: dict[str, object]) -> None:
    with pytest.raises(SemanticSpanContractError):
        _span(**changes)


def _parent(
    spans: tuple[SemanticSpan, ...] | None = None,
    **changes: object,
) -> SemanticSpanParentPlan:
    values: dict[str, object] = {
        "source_fragment_id": "fragment_parent",
        "source_version_id": "source_version_one",
        "source_id": "source_one",
        "ordinal": 0,
        "text_sha256": HASH_A,
        "text_char_count": 3,
        "content_token_count": 1,
        "full_prepared_token_count": 3,
        "content_offsets_hash": HASH_C,
        "spans": (_span(),) if spans is None else spans,
    }
    values.update(changes)
    return SemanticSpanParentPlan(**cast(Any, values))


def test_parent_and_plan_closure_contracts_fail_closed_on_forgery() -> None:
    with pytest.raises(SemanticSpanContractError, match="at least one"):
        _parent(())
    with pytest.raises(SemanticSpanContractError, match="exact SemanticSpan"):
        _parent((cast(SemanticSpan, object()),))
    with pytest.raises(SemanticSpanContractError, match="span ordinals"):
        _parent((_span(span_ordinal=1),))
    with pytest.raises(SemanticSpanContractError, match="parent binding"):
        _parent((_span(parent_fragment_id="fragment_other"),))
    with pytest.raises(SemanticSpanContractError, match="char boundaries"):
        _parent((_span(char_start=1),))
    with pytest.raises(SemanticSpanContractError, match="char closure contains a gap"):
        _parent(
            (
                _span(char_end=1, content_token_end=1),
                _span(span_ordinal=1, char_start=2, content_token_start=0),
            )
        )
    with pytest.raises(SemanticSpanContractError, match="token boundaries"):
        _parent((_span(content_token_start=1),))

    parent = _parent()
    with pytest.raises(SemanticSpanContractError, match="exact SemanticSpanProfile"):
        SemanticSpanPlan(cast(SemanticSpanProfile, object()), (parent,))
    with pytest.raises(SemanticSpanContractError, match="exact tuple"):
        SemanticSpanPlan(
            _profile(),
            cast(tuple[SemanticSpanParentPlan, ...], []),
        )
    with pytest.raises(SemanticSpanContractError, match="exact SemanticSpanParentPlan"):
        SemanticSpanPlan(_profile(), (cast(SemanticSpanParentPlan, object()),))
    with pytest.raises(SemanticSpanContractError, match="fragment IDs must be unique"):
        SemanticSpanPlan(_profile(), (parent, parent))


@pytest.mark.parametrize(
    "changes",
    [
        {"ordinal": -1},
        {"text_char_count": 0},
        {"content_token_count": -1},
        {"full_prepared_token_count": 0},
    ],
)
def test_parent_scalar_contract_rejects_invalid_counts(changes: dict[str, object]) -> None:
    with pytest.raises(SemanticSpanContractError):
        _parent(**cast(Any, changes))


def test_parent_char_closure_rejects_nonprogress_and_parent_overrun() -> None:
    with pytest.raises(SemanticSpanContractError, match="char closure makes no progress"):
        _parent(
            (
                _span(char_end=3),
                _span(span_ordinal=1, char_start=1, char_end=3),
            )
        )

    with pytest.raises(SemanticSpanContractError, match="exceeds parent char"):
        _parent(
            (
                _span(char_end=4),
                _span(span_ordinal=1, char_start=2, char_end=3),
            )
        )


def test_parent_token_closure_rejects_gaps_nonprogress_and_parent_overrun() -> None:
    with pytest.raises(SemanticSpanContractError, match="token closure contains a gap"):
        _parent(
            (
                _span(char_end=2, content_token_end=1),
                _span(
                    span_ordinal=1,
                    char_start=1,
                    content_token_start=2,
                    content_token_end=3,
                ),
            ),
            content_token_count=3,
        )

    with pytest.raises(SemanticSpanContractError, match="token closure makes no progress"):
        _parent(
            (
                _span(char_end=2, content_token_end=2),
                _span(
                    span_ordinal=1,
                    char_start=1,
                    content_token_start=1,
                    content_token_end=2,
                ),
            ),
            content_token_count=2,
        )

    with pytest.raises(SemanticSpanContractError, match="exceeds parent token"):
        _parent(
            (
                _span(char_end=2, content_token_end=3),
                _span(
                    span_ordinal=1,
                    char_start=1,
                    content_token_start=1,
                    content_token_end=2,
                ),
            ),
            content_token_count=2,
        )


def _bound_parent(
    fragment_id: str,
    *,
    version: str,
    source: str,
    ordinal: int,
) -> SemanticSpanParentPlan:
    return _parent(
        (_span(parent_fragment_id=fragment_id, parent_ordinal=ordinal),),
        source_fragment_id=fragment_id,
        source_version_id=version,
        source_id=source,
        ordinal=ordinal,
    )


def test_plan_rejects_noncanonical_and_ambiguous_parent_closures() -> None:
    alpha = _bound_parent(
        "fragment_alpha",
        version="source_version_shared",
        source="source_alpha",
        ordinal=0,
    )
    beta = _bound_parent(
        "fragment_beta",
        version="source_version_other",
        source="source_beta",
        ordinal=1,
    )
    with pytest.raises(SemanticSpanContractError, match="canonical fragment-ID order"):
        SemanticSpanPlan(_profile(), (beta, alpha))

    duplicate_ordinal = _bound_parent(
        "fragment_beta",
        version="source_version_shared",
        source="source_alpha",
        ordinal=0,
    )
    with pytest.raises(SemanticSpanContractError, match="ordinal pairs must be unique"):
        SemanticSpanPlan(_profile(), (alpha, duplicate_ordinal))

    conflicting_source = _bound_parent(
        "fragment_beta",
        version="source_version_shared",
        source="source_beta",
        ordinal=1,
    )
    with pytest.raises(SemanticSpanContractError, match="multiple Sources"):
        SemanticSpanPlan(_profile(), (alpha, conflicting_source))


def test_plan_enforces_profile_budget_fit_and_overlap_contracts() -> None:
    with pytest.raises(SemanticSpanContractError, match="profile/model binding"):
        SemanticSpanPlan(_profile(), (_parent((_span(profile_hash=HASH_C),)),))

    with pytest.raises(SemanticSpanContractError, match="exceeds max_prepared_tokens"):
        SemanticSpanPlan(
            _profile(),
            (_parent((_span(prepared_token_count=6),), full_prepared_token_count=6),),
        )

    fitting_two_spans = _parent(
        (
            _span(char_end=2, content_token_end=0),
            _span(
                span_ordinal=1,
                char_start=1,
                content_token_start=0,
                content_token_end=0,
            ),
        ),
        content_token_count=0,
    )
    with pytest.raises(SemanticSpanContractError, match="fitting parent"):
        SemanticSpanPlan(_profile(), (fitting_two_spans,))

    with pytest.raises(SemanticSpanContractError, match="equal the full text"):
        SemanticSpanPlan(_profile(), (_parent((_span(prepared_token_count=2),)),))

    with pytest.raises(SemanticSpanContractError, match="requires multiple spans"):
        SemanticSpanPlan(
            _profile(),
            (_parent((_span(prepared_token_count=5),), full_prepared_token_count=6),),
        )

    zero_overlap_profile = _profile(overlap=0)
    overbudget_without_anchors = _parent(
        (
            _span(
                char_end=2,
                content_token_end=0,
                prepared_token_count=5,
                profile_hash=zero_overlap_profile.profile_hash,
            ),
            _span(
                span_ordinal=1,
                char_start=1,
                content_token_start=0,
                content_token_end=0,
                prepared_token_count=5,
                profile_hash=zero_overlap_profile.profile_hash,
            ),
        ),
        content_token_count=0,
        full_prepared_token_count=6,
    )
    with pytest.raises(SemanticSpanContractError, match="window anchors"):
        SemanticSpanPlan(zero_overlap_profile, (overbudget_without_anchors,))

    incorrect_overlap = _parent(
        (
            _span(char_end=2, content_token_end=2, prepared_token_count=5),
            _span(
                span_ordinal=1,
                char_start=1,
                content_token_start=0,
                content_token_end=3,
                prepared_token_count=5,
            ),
        ),
        content_token_count=3,
        full_prepared_token_count=6,
    )
    with pytest.raises(SemanticSpanContractError, match="overlap differs"):
        SemanticSpanPlan(_profile(), (incorrect_overlap,))

    nonprogressing_window = _parent(
        (
            _span(char_end=2, content_token_end=1, prepared_token_count=5),
            _span(
                span_ordinal=1,
                char_start=1,
                content_token_start=0,
                content_token_end=2,
                prepared_token_count=5,
            ),
        ),
        content_token_count=2,
        full_prepared_token_count=6,
    )
    with pytest.raises(SemanticSpanContractError, match="advance beyond"):
        SemanticSpanPlan(_profile(), (nonprogressing_window,))


def test_canonical_identifier_suffix_is_enforced() -> None:
    with pytest.raises(SemanticSpanContractError, match="canonical fragment_ suffix"):
        _span(parent_fragment_id="fragment_NotCanonical")


def test_defensive_window_and_span_invariants_fail_closed() -> None:
    fragment = _fragment("fragment_internal_guard", "a b")
    malformed_offsets = (
        ContentTokenOffset(0, 1),
        ContentTokenOffset(10, 11),
        ContentTokenOffset(12, 13),
    )
    tokenizer = StaticTokenizer(malformed_offsets, (2,))

    with pytest.raises(SemanticSpanPlanningError, match="zero-length window"):
        semantic_spans._window_parent(
            fragment,
            offsets=malformed_offsets,
            profile=_profile(max_tokens=2, overlap=1),
            counter=cast(Any, tokenizer.count_prepared_tokens_no_truncation),
        )

    with pytest.raises(SemanticSpanPlanningError, match="text cannot be empty"):
        semantic_spans._build_span(
            fragment,
            profile=_profile(),
            span_ordinal=0,
            char_start=0,
            char_end=0,
            token_start=0,
            token_end=0,
            prepared_token_count=1,
        )


@pytest.mark.parametrize(
    "fragments",
    [
        [cast(SourceFragmentText, object())],
        (cast(SourceFragmentText, object()),),
        (
            _fragment("fragment_a", "a", ordinal=0),
            _fragment("fragment_a", "b", ordinal=1),
        ),
        (
            _fragment("fragment_a", "a", ordinal=0),
            _fragment("fragment_b", "b", ordinal=0),
        ),
        (
            _fragment("fragment_a", "a", ordinal=0, source="source_one"),
            _fragment("fragment_b", "b", ordinal=1, source="source_two"),
        ),
        (replace(_fragment("fragment_a", "a"), ordinal=-1),),
        (replace(_fragment("fragment_a", "a"), text=""),),
        (replace(_fragment("fragment_a", "a"), text="e\u0301"),),
        (replace(_fragment("fragment_a", "a"), text="a\x00"),),
        (replace(_fragment("fragment_a", "a"), text="\ud800"),),
        (replace(_fragment("fragment_a", "a"), text_sha256=HASH_A),),
        (replace(_fragment("fragment_a", "a"), source_fragment_id="fragment_é"),),
        (replace(_fragment("fragment_a", "a"), source_fragment_id=cast(Any, 7)),),
        (replace(_fragment("fragment_a", "a"), source_version_id="bad"),),
        (replace(_fragment("fragment_a", "a"), source_id="bad"),),
    ],
)
def test_invalid_parent_fragment_closure_fails_before_tokenizer_calls(
    fragments: object,
) -> None:
    tokenizer = WordTokenizer()

    with pytest.raises(SemanticSpanContractError):
        plan_semantic_spans(
            cast(tuple[SourceFragmentText, ...], fragments),
            profile=_profile(),
            tokenizer=tokenizer,
        )

    assert tokenizer.offset_calls == []
    assert tokenizer.count_calls == []


def test_planner_requires_exact_profile_before_tokenizer_calls() -> None:
    tokenizer = WordTokenizer()
    with pytest.raises(SemanticSpanContractError, match="exact SemanticSpanProfile"):
        plan_semantic_spans(
            (_fragment("fragment_a", "a"),),
            profile=cast(SemanticSpanProfile, object()),
            tokenizer=tokenizer,
        )
    assert tokenizer.offset_calls == []
