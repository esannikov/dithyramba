from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import FrozenInstanceError
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.ingest.models import (
    LARGE_DOCUMENT_PARSER_PROFILE,
    PARSER_PROFILE,
    SOURCE_ADDRESS_SCHEMA,
    FileIdentity,
    FragmentKind,
    MarkdownSourceAddress,
    MediaType,
    ParsedFragment,
    ParseResult,
    ParserLimits,
    ParseStatus,
    PdfSourceAddress,
    SourceBytes,
    format_pdf_point,
    parser_profile,
)

_CANONICAL_POINT = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{3}$")


def _replace(instance: Any, field: str, value: object) -> Any:
    return dataclasses.replace(instance, **{field: value})


def _identity(*, size: int = 3) -> FileIdentity:
    return FileIdentity(device=1, inode=2, size=size, modified_ns=3)


def _source_bytes(
    *,
    data: bytes = b"abc",
    identity: FileIdentity | None = None,
    **overrides: object,
) -> SourceBytes:
    values: dict[str, object] = {
        "relative_path": "notes/example.md",
        "canonical_uri": "file:///vault/notes/example.md",
        "media_type": MediaType.MARKDOWN,
        "data": data,
        "content_sha256": sha256_hex(data),
        "source_modified_at": "2026-07-20T12:00:00.123456Z",
        "identity": identity or _identity(size=len(data)),
    }
    values.update(overrides)
    return SourceBytes(**values)  # type: ignore[arg-type]


def _markdown_address(
    text: str = "text",
    *,
    heading_path: tuple[str, ...] = ("Section",),
    line_start: int = 1,
    line_end: int = 1,
    char_start: int = 0,
) -> MarkdownSourceAddress:
    return MarkdownSourceAddress(
        heading_path=heading_path,
        line_start=line_start,
        line_end=line_end,
        char_start=char_start,
        char_end=char_start + len(text),
    )


def _pdf_address(
    text: str = "text",
    *,
    page: int = 1,
    bbox: tuple[str, str, str, str] = ("0.000", "1.000", "10.000", "20.000"),
    char_start: int = 0,
) -> PdfSourceAddress:
    return PdfSourceAddress(
        page=page,
        bbox=bbox,
        char_start=char_start,
        char_end=char_start + len(text),
    )


def _fragment(
    ordinal: int = 0,
    *,
    kind: FragmentKind = FragmentKind.PARAGRAPH,
    text: str = "text",
) -> ParsedFragment:
    address = _pdf_address(text) if kind is FragmentKind.PAGE_TEXT else _markdown_address(text)
    return ParsedFragment(ordinal=ordinal, kind=kind, text=text, address=address)


def test_parser_limits_defaults_are_the_frozen_vs0_profile() -> None:
    limits = ParserLimits()

    assert limits == ParserLimits(
        max_file_bytes=25 * 1024 * 1024,
        max_pdf_pages=500,
        max_extracted_codepoints=2_000_000,
        timeout_seconds=30,
        max_rss_mib=512,
    )


def test_large_document_profile_is_versioned_and_explicitly_bounded() -> None:
    profile, limits = parser_profile("large-document")

    assert profile == LARGE_DOCUMENT_PARSER_PROFILE
    assert limits == ParserLimits(
        max_file_bytes=512 * 1024 * 1024,
        max_pdf_pages=2_000,
        max_extracted_codepoints=20_000_000,
        timeout_seconds=180,
        max_rss_mib=2_560,
    )


@pytest.mark.parametrize(
    "field",
    [
        "max_file_bytes",
        "max_pdf_pages",
        "max_extracted_codepoints",
        "timeout_seconds",
        "max_rss_mib",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1"])
def test_parser_limits_reject_non_positive_or_non_integer_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a positive integer"):
        _replace(ParserLimits(), field, value)


def test_parser_limits_are_frozen() -> None:
    limits = ParserLimits()

    with pytest.raises(FrozenInstanceError):
        limits.max_file_bytes = 1  # type: ignore[misc]


def test_file_identity_accepts_zero_and_positive_descriptor_values() -> None:
    assert FileIdentity(device=0, inode=1, size=0, modified_ns=2) == FileIdentity(0, 1, 0, 2)


@pytest.mark.parametrize("field", ["device", "inode", "size", "modified_ns"])
@pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
def test_file_identity_rejects_negative_or_non_integer_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a non-negative integer"):
        _replace(_identity(), field, value)


def test_source_bytes_preserves_exact_bytes_and_reports_byte_size() -> None:
    source = _source_bytes(data=b"\x00\xffmarkdown")

    assert source.data == b"\x00\xffmarkdown"
    assert source.byte_size == len(source.data)
    assert source.content_sha256 == sha256_hex(source.data)


@pytest.mark.parametrize("media_type", list(MediaType))
def test_source_bytes_accepts_every_frozen_media_type(media_type: MediaType) -> None:
    assert _source_bytes(media_type=media_type).media_type is media_type


def test_source_bytes_requires_string_relative_path() -> None:
    with pytest.raises(TypeError, match="relative_path must be str"):
        _source_bytes(relative_path=1)


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "   ",
        "notes/\x00secret.md",
        "notes\\secret.md",
        "/absolute/note.md",
        "notes//secret.md",
        "notes/./secret.md",
        "notes/../secret.md",
        "notes/",
        ".",
        "..",
    ],
)
def test_source_bytes_requires_unambiguous_relative_posix_path(relative_path: str) -> None:
    with pytest.raises(ValueError, match="unambiguous relative POSIX path"):
        _source_bytes(relative_path=relative_path)


def test_source_bytes_accepts_localhost_absolute_file_uri() -> None:
    source = _source_bytes(canonical_uri="file://localhost/vault/notes/example.md")

    assert source.canonical_uri == "file://localhost/vault/notes/example.md"


def test_source_bytes_requires_string_canonical_uri() -> None:
    with pytest.raises(TypeError, match="canonical_uri must be str"):
        _source_bytes(canonical_uri=1)


@pytest.mark.parametrize(
    "canonical_uri",
    [
        "https://example.test/note.md",
        "file://remotehost/vault/note.md",
        "file:relative/note.md",
        "file:///",
        "file:///vault/note.md?revision=1",
        "file:///vault/note.md#fragment",
    ],
)
def test_source_bytes_requires_absolute_local_file_uri(canonical_uri: str) -> None:
    with pytest.raises(ValueError, match="absolute file URI"):
        _source_bytes(canonical_uri=canonical_uri)


def test_source_bytes_rejects_untyped_media_type() -> None:
    with pytest.raises(TypeError, match="media_type must be MediaType"):
        _source_bytes(media_type="text/markdown")


def test_source_bytes_rejects_mutable_bytearray() -> None:
    with pytest.raises(TypeError, match="immutable bytes"):
        _source_bytes(data=bytearray(b"abc"))  # type: ignore[arg-type]


def test_source_bytes_requires_file_identity_contract() -> None:
    with pytest.raises(TypeError, match="identity must be FileIdentity"):
        _source_bytes(identity="descriptor")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
    ],
)
def test_source_bytes_rejects_noncanonical_sha256(digest: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256 hex"):
        _source_bytes(content_sha256=digest)


def test_source_bytes_rejects_digest_that_does_not_match_data() -> None:
    with pytest.raises(ValueError, match="does not match data"):
        _source_bytes(content_sha256="0" * 64)


def test_source_bytes_rejects_descriptor_size_that_does_not_match_data() -> None:
    with pytest.raises(ValueError, match="identity size"):
        _source_bytes(identity=_identity(size=2))


@pytest.mark.parametrize(
    "source_modified_at",
    [
        "",
        None,
        "2026-07-20T12:00:00Z",
        "2026-07-20T12:00:00.123Z",
        "2026-07-20 12:00:00.123456Z",
        "2026-07-20T12:00:00.123456+00:00",
        "2026-07-20T12:00:00.123456z",
        "2026-00-20T12:00:00.123456Z",
        "2026-02-30T12:00:00.123456Z",
        "2026-07-20T24:00:00.123456Z",
        "0000-01-01T00:00:00.000000Z",
    ],
)
def test_source_bytes_requires_canonical_utc_microsecond_timestamp(
    source_modified_at: object,
) -> None:
    with pytest.raises(ValueError, match="canonical UTC with microseconds"):
        _source_bytes(source_modified_at=source_modified_at)


@pytest.mark.parametrize(
    ("contract", "field", "replacement"),
    [
        (_identity(), "size", 99),
        (_source_bytes(), "relative_path", "changed.md"),
        (_markdown_address(), "line_start", 2),
        (_pdf_address(), "page", 2),
    ],
    ids=["file-identity", "source-bytes", "markdown-address", "pdf-address"],
)
def test_shared_source_contracts_are_frozen(
    contract: object, field: str, replacement: object
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(contract, field, replacement)


def test_markdown_address_normalizes_heading_path_and_has_canonical_payload() -> None:
    address = MarkdownSourceAddress(
        heading_path=("Cafe\u0301", "Nested"),
        line_start=2,
        line_end=4,
        char_start=10,
        char_end=14,
    )

    assert address.heading_path == ("Caf\u00e9", "Nested")
    assert address.payload() == {
        "schema": SOURCE_ADDRESS_SCHEMA,
        "kind": "markdown",
        "heading_path": ["Caf\u00e9", "Nested"],
        "line_start": 2,
        "line_end": 4,
        "char_start": 10,
        "char_end": 14,
    }


def test_markdown_address_allows_empty_root_heading_path() -> None:
    assert _markdown_address(heading_path=()).heading_path == ()


@pytest.mark.parametrize("heading_path", [["Section"], "Section"])
def test_markdown_address_requires_tuple_heading_path(heading_path: object) -> None:
    with pytest.raises(TypeError, match="tuple of strings"):
        MarkdownSourceAddress(heading_path, 1, 1, 0, 1)  # type: ignore[arg-type]


def test_markdown_address_requires_string_heading_entries() -> None:
    with pytest.raises(TypeError, match="tuple of strings"):
        MarkdownSourceAddress(("Section", 1), 1, 1, 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("heading_path", [("",), ("   ",), ("Valid", "\t")])
def test_markdown_address_rejects_blank_heading_entries(heading_path: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="heading_path entries"):
        _markdown_address(heading_path=heading_path)


@pytest.mark.parametrize("line_start", [0, -1, True, 1.0])
def test_markdown_address_requires_positive_integer_line_start(line_start: object) -> None:
    with pytest.raises(ValueError, match="line_start"):
        MarkdownSourceAddress(("Section",), line_start, 1, 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("line_end", [0, True, 1.0])
def test_markdown_address_requires_integer_line_end_not_before_start(line_end: object) -> None:
    with pytest.raises(ValueError, match="line_end"):
        MarkdownSourceAddress(("Section",), 1, line_end, 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("char_start", [-1, True, 1.0])
def test_source_address_requires_non_negative_integer_char_start(char_start: object) -> None:
    with pytest.raises(ValueError, match="char_start"):
        MarkdownSourceAddress(("Section",), 1, 1, char_start, 2)  # type: ignore[arg-type]


@pytest.mark.parametrize("char_end", [0, -1, True, 1.0])
def test_source_address_requires_integer_char_end_greater_than_start(char_end: object) -> None:
    with pytest.raises(ValueError, match="char_end"):
        MarkdownSourceAddress(("Section",), 1, 1, 0, char_end)  # type: ignore[arg-type]


def test_pdf_address_accepts_negative_coordinates_and_has_canonical_payload() -> None:
    address = PdfSourceAddress(
        page=3,
        bbox=("-10.250", "-2.000", "0.000", "18.125"),
        char_start=7,
        char_end=11,
    )

    assert address.payload() == {
        "schema": SOURCE_ADDRESS_SCHEMA,
        "kind": "pdf",
        "page": 3,
        "bbox": ["-10.250", "-2.000", "0.000", "18.125"],
        "char_start": 7,
        "char_end": 11,
    }


@pytest.mark.parametrize("page", [0, -1, True, 1.0])
def test_pdf_address_requires_positive_integer_page(page: object) -> None:
    with pytest.raises(ValueError, match="page"):
        PdfSourceAddress(page, ("0.000", "0.000", "1.000", "1.000"), 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bbox",
    [
        ["0.000", "0.000", "1.000", "1.000"],
        ("0.000", "0.000", "1.000"),
        ("0.000", "0.000", "1.000", "1.000", "2.000"),
    ],
)
def test_pdf_address_requires_a_four_item_tuple(bbox: object) -> None:
    with pytest.raises(ValueError, match="exactly four"):
        PdfSourceAddress(1, bbox, 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "point",
    [0, "0", "0.0", "+0.000", "00.000", ".000", "NaN", "Infinity", "1.0000"],
)
def test_pdf_address_rejects_noncanonical_point_strings(point: object) -> None:
    with pytest.raises(ValueError, match="fixed-point strings"):
        PdfSourceAddress(1, (point, "0.000", "1.000", "1.000"), 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bbox",
    [
        ("1.000", "0.000", "1.000", "2.000"),
        ("2.000", "0.000", "1.000", "2.000"),
        ("0.000", "2.000", "1.000", "2.000"),
        ("0.000", "3.000", "1.000", "2.000"),
    ],
)
def test_pdf_address_requires_strictly_increasing_bbox_axes(
    bbox: tuple[str, str, str, str],
) -> None:
    with pytest.raises(ValueError, match="x0 < x1 and top < bottom"):
        PdfSourceAddress(1, bbox, 0, 1)


def test_pdf_address_accepts_supported_coordinate_boundaries() -> None:
    address = PdfSourceAddress(
        1,
        (
            "-1000000000.000",
            "-1000000000.000",
            "1000000000.000",
            "1000000000.000",
        ),
        0,
        1,
    )

    assert address.bbox[0] == "-1000000000.000"
    assert address.bbox[2] == "1000000000.000"


@pytest.mark.parametrize(
    "bbox",
    [
        ("-0.000", "0.000", "1.000", "1.000"),
        ("0.000", "0.000", "1000000000.001", "1.000"),
        ("-1000000000.001", "0.000", "1.000", "1.000"),
    ],
)
def test_pdf_address_rejects_negative_zero_or_out_of_range_points(
    bbox: tuple[str, str, str, str],
) -> None:
    with pytest.raises(ValueError, match="canonical and within the supported range"):
        PdfSourceAddress(1, bbox, 0, 1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0.000"),
        (-0.0, "0.000"),
        (Decimal("-0.0004"), "0.000"),
        (1, "1.000"),
        ("-12.25", "-12.250"),
        (Decimal("1.2345"), "1.234"),
        (Decimal("1.2355"), "1.236"),
        (Decimal("-1.2345"), "-1.234"),
        (Decimal("-1.2355"), "-1.236"),
        (Decimal("1000000000"), "1000000000.000"),
        (Decimal("-1000000000"), "-1000000000.000"),
    ],
)
def test_format_pdf_point_uses_fixed_three_decimal_half_even_format(
    value: object, expected: str
) -> None:
    assert format_pdf_point(value) == expected


@pytest.mark.parametrize(
    "value",
    [True, False, "not-a-number", object()],
)
def test_format_pdf_point_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        format_pdf_point(value)


@pytest.mark.parametrize(
    "value",
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("1000000000.001"),
        Decimal("-1000000000.001"),
    ],
)
def test_format_pdf_point_rejects_nonfinite_or_out_of_range_values(value: Decimal) -> None:
    with pytest.raises(ValueError, match="finite and within"):
        format_pdf_point(value)


@given(
    st.decimals(
        min_value=Decimal("-1000000000"),
        max_value=Decimal("1000000000"),
        allow_nan=False,
        allow_infinity=False,
        places=6,
    )
)
def test_format_pdf_point_is_canonical_idempotent_and_half_even(value: Decimal) -> None:
    formatted = format_pdf_point(value)
    expected = value.quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN)

    assert _CANONICAL_POINT.fullmatch(formatted)
    assert formatted != "-0.000"
    assert Decimal(formatted) == expected
    assert format_pdf_point(formatted) == formatted


@pytest.mark.parametrize("kind", [FragmentKind.HEADING, FragmentKind.PARAGRAPH])
def test_markdown_fragment_normalizes_text_and_derives_canonical_hashes(
    kind: FragmentKind,
) -> None:
    normalized = "Caf\u00e9"
    address = _markdown_address(normalized, heading_path=("Cafe\u0301",), char_start=4)
    fragment = ParsedFragment(ordinal=0, kind=kind, text="Cafe\u0301", address=address)

    expected_payload = address.payload()
    assert fragment.text == normalized
    assert fragment.text_sha256 == sha256_hex(normalized.encode("utf-8"))
    assert fragment.address_payload == expected_payload
    assert fragment.address_json == canonical_json_bytes(expected_payload).decode("utf-8")
    assert json.loads(fragment.address_json) == expected_payload
    assert fragment.address_hash == canonical_sha256_hex(expected_payload)


def test_pdf_page_text_fragment_accepts_matching_address_and_span() -> None:
    fragment = _fragment(kind=FragmentKind.PAGE_TEXT, text="page")

    assert fragment.kind is FragmentKind.PAGE_TEXT
    assert isinstance(fragment.address, PdfSourceAddress)


@pytest.mark.parametrize("ordinal", [-1, True, 1.0])
def test_parsed_fragment_requires_non_negative_integer_ordinal(ordinal: object) -> None:
    with pytest.raises(ValueError, match="ordinal"):
        ParsedFragment(ordinal, FragmentKind.PARAGRAPH, "x", _markdown_address("x"))  # type: ignore[arg-type]


def test_parsed_fragment_requires_fragment_kind_enum() -> None:
    with pytest.raises(TypeError, match="kind must be FragmentKind"):
        ParsedFragment(0, "paragraph", "x", _markdown_address("x"))  # type: ignore[arg-type]


def test_parsed_fragment_requires_string_text() -> None:
    with pytest.raises(TypeError, match="text must be str"):
        ParsedFragment(0, FragmentKind.PARAGRAPH, b"x", _markdown_address("x"))  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", " \t\n", "prefix\x00suffix"])
def test_parsed_fragment_rejects_blank_or_nul_text(text: str) -> None:
    with pytest.raises(ValueError, match="non-empty and contain no NUL"):
        ParsedFragment(0, FragmentKind.PARAGRAPH, text, _markdown_address("x"))


@pytest.mark.parametrize("kind", [FragmentKind.HEADING, FragmentKind.PARAGRAPH])
def test_markdown_fragment_kinds_reject_pdf_address(kind: FragmentKind) -> None:
    with pytest.raises(ValueError, match="require MarkdownSourceAddress"):
        ParsedFragment(0, kind, "x", _pdf_address("x"))


def test_page_text_fragment_rejects_markdown_address() -> None:
    with pytest.raises(ValueError, match="page_text requires PdfSourceAddress"):
        ParsedFragment(0, FragmentKind.PAGE_TEXT, "x", _markdown_address("x"))


@pytest.mark.parametrize("char_end", [1, 5])
def test_parsed_fragment_requires_exact_character_span(char_end: int) -> None:
    address = MarkdownSourceAddress(("Section",), 1, 1, 0, char_end)

    with pytest.raises(ValueError, match="span must equal"):
        ParsedFragment(0, FragmentKind.PARAGRAPH, "text", address)


def test_parsed_fragment_is_frozen() -> None:
    fragment = _fragment()

    with pytest.raises(FrozenInstanceError):
        fragment.text = "changed"  # type: ignore[misc]


def test_parse_result_processed_factory_preserves_contiguous_fragments() -> None:
    fragments = (_fragment(0), _fragment(1, kind=FragmentKind.HEADING, text="heading"))

    result = ParseResult.processed(
        fragments,
        parser_revision="markdown/1.0",
        normalized_codepoints=11,
    )

    assert result.status is ParseStatus.PROCESSED
    assert result.fragments == fragments
    assert result.failure_code is None
    assert result.parser_profile == PARSER_PROFILE
    assert result.normalized_codepoints == 11


@pytest.mark.parametrize(
    ("factory", "expected_status"),
    [(ParseResult.skipped, ParseStatus.SKIPPED), (ParseResult.failed, ParseStatus.FAILED)],
)
def test_parse_result_nonprocessed_factories_create_coded_empty_results(
    factory: Any, expected_status: ParseStatus
) -> None:
    result = factory("file_too_large", parser_revision="reader/1.0")

    assert result.status is expected_status
    assert result.fragments == ()
    assert result.failure_code == "file_too_large"
    assert result.normalized_codepoints == 0


def test_parse_result_requires_parse_status_enum() -> None:
    with pytest.raises(TypeError, match="status must be ParseStatus"):
        ParseResult("processed", (_fragment(),), "parser/1", 4)  # type: ignore[arg-type]


def test_parse_result_requires_fragment_tuple() -> None:
    with pytest.raises(TypeError, match="tuple of ParsedFragment"):
        ParseResult(ParseStatus.PROCESSED, [_fragment()], "parser/1", 4)  # type: ignore[arg-type]


def test_parse_result_rejects_non_fragment_tuple_member() -> None:
    with pytest.raises(TypeError, match="tuple of ParsedFragment"):
        ParseResult(ParseStatus.PROCESSED, ("fragment",), "parser/1", 4)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("parser_revision", "parser_profile"),
    [("", PARSER_PROFILE), (" \t", PARSER_PROFILE), ("parser/1", ""), ("parser/1", " \n")],
)
def test_parse_result_requires_nonblank_parser_revision_and_profile(
    parser_revision: str, parser_profile: str
) -> None:
    with pytest.raises(ValueError, match="revision/profile"):
        ParseResult(
            ParseStatus.PROCESSED,
            (_fragment(),),
            parser_revision,
            4,
            parser_profile=parser_profile,
        )


@pytest.mark.parametrize("normalized_codepoints", [-1, True, 1.0])
def test_parse_result_requires_non_negative_integer_codepoint_count(
    normalized_codepoints: object,
) -> None:
    with pytest.raises(ValueError, match="normalized_codepoints"):
        ParseResult(
            ParseStatus.PROCESSED,
            (_fragment(),),
            "parser/1",
            normalized_codepoints,  # type: ignore[arg-type]
        )


def test_processed_result_requires_at_least_one_fragment() -> None:
    with pytest.raises(ValueError, match="requires fragments and no failure code"):
        ParseResult(ParseStatus.PROCESSED, (), "parser/1", 0)


def test_processed_result_rejects_failure_code() -> None:
    with pytest.raises(ValueError, match="requires fragments and no failure code"):
        ParseResult(ParseStatus.PROCESSED, (_fragment(),), "parser/1", 4, "unexpected")


@pytest.mark.parametrize("ordinals", [(1,), (0, 2), (0, 0)])
def test_processed_result_requires_ordinals_contiguous_from_zero(ordinals: tuple[int, ...]) -> None:
    fragments = tuple(_fragment(ordinal) for ordinal in ordinals)

    with pytest.raises(ValueError, match="ordinals must be contiguous"):
        ParseResult.processed(fragments, parser_revision="parser/1", normalized_codepoints=4)


@pytest.mark.parametrize("status", [ParseStatus.SKIPPED, ParseStatus.FAILED])
def test_nonprocessed_result_rejects_fragments(status: ParseStatus) -> None:
    with pytest.raises(ValueError, match="requires no fragments and a failure code"):
        ParseResult(status, (_fragment(),), "parser/1", 4, "parse_failed")


@pytest.mark.parametrize("status", [ParseStatus.SKIPPED, ParseStatus.FAILED])
def test_nonprocessed_result_requires_failure_code(status: ParseStatus) -> None:
    with pytest.raises(ValueError, match="requires no fragments and a failure code"):
        ParseResult(status, (), "parser/1", 0)


@pytest.mark.parametrize(
    "failure_code",
    [
        "",
        "a",
        "1invalid",
        "Invalid",
        "invalid-code",
        "invalid.code",
        "\u043f\u043e\u043c\u0438\u043b\u043a\u0430",
        "a" * 65,
        "valid_code\n",
    ],
)
def test_nonprocessed_result_rejects_unstable_failure_code_grammar(
    failure_code: str,
) -> None:
    with pytest.raises(ValueError, match="stable snake_case grammar"):
        ParseResult(ParseStatus.FAILED, (), "parser/1", 0, failure_code)


@pytest.mark.parametrize("failure_code", ["aa", "a_", "parse_failed", "a" + "b" * 63])
def test_nonprocessed_result_accepts_failure_code_grammar_boundaries(
    failure_code: str,
) -> None:
    result = ParseResult.failed(failure_code, parser_revision="parser/1")

    assert result.failure_code == failure_code


def test_parse_result_is_frozen() -> None:
    result = ParseResult.processed(
        (_fragment(),), parser_revision="parser/1", normalized_codepoints=4
    )

    with pytest.raises(FrozenInstanceError):
        result.status = ParseStatus.FAILED  # type: ignore[misc]
