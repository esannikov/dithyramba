"""Bounded parser dispatch for immutable captured source bytes."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    ExtractedTextLimitExceededError,
    IngestError,
    NoExtractableTextError,
    ParserOutputInvalidError,
)
from .models import (
    FragmentKind,
    MarkdownSourceAddress,
    MediaType,
    ParsedFragment,
    ParseResult,
    ParserLimits,
    SourceBytes,
)
from .pdf import parse_pdf_source

_MARKDOWN_REVISION = "markdown_atx/1.0"
_PLAIN_TEXT_REVISION = "plain_text_atx/1.0"
_ATX_HEADING = re.compile(r"^(?P<indent> {0,3})(?P<marks>#{1,6})(?:[ \t]+(?P<body>.*)|[ \t]*)$")
_CLOSING_MARKS = re.compile(r"[ \t]+#+$")
_OPEN_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?:[^`]*)$")


@dataclass(frozen=True, slots=True)
class _Line:
    number: int
    start: int
    end: int
    content: str


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int
    text: str
    start: int
    end: int


def parse_source(
    source: SourceBytes,
    limits: ParserLimits,
    *,
    pdf_temp_root: Path | None = None,
) -> ParseResult:
    """Parse captured bytes and convert typed content failures to outcomes."""

    revisions = {
        MediaType.MARKDOWN: _MARKDOWN_REVISION,
        MediaType.PLAIN_TEXT: _PLAIN_TEXT_REVISION,
        MediaType.PDF: "pdfplumber/0.11.10+page_text/1.0",
    }
    revision = revisions[source.media_type]

    try:
        if source.media_type in (MediaType.MARKDOWN, MediaType.PLAIN_TEXT):
            return _parse_text(source, limits, parser_revision=revision)
        return parse_pdf_source(source, limits, temp_root=pdf_temp_root)
    except IngestError as error:
        if error.outcome == "skipped":
            return ParseResult.skipped(error.code, parser_revision=revision)
        return ParseResult.failed(error.code, parser_revision=revision)


def _parse_text(
    source: SourceBytes,
    limits: ParserLimits,
    *,
    parser_revision: str,
) -> ParseResult:
    try:
        decoded = source.data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        from .errors import InvalidUtf8Error

        raise InvalidUtf8Error("source is not strict UTF-8") from error
    normalized = unicodedata.normalize("NFC", decoded.replace("\r\n", "\n").replace("\r", "\n"))
    if "\x00" in normalized:
        raise ParserOutputInvalidError("text source contains NUL")
    if len(normalized) > limits.max_extracted_codepoints:
        raise ExtractedTextLimitExceededError(
            f"normalized text exceeds {limits.max_extracted_codepoints} code points"
        )

    lines = _lines(normalized)
    fragments: list[ParsedFragment] = []
    heading_stack: list[tuple[int, str]] = []
    paragraph_lines: list[_Line] = []
    open_fence: tuple[str, int] | None = None

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        first = paragraph_lines[0]
        last = paragraph_lines[-1]
        text = normalized[first.start : last.end]
        if not text.strip():
            paragraph_lines.clear()
            return
        fragments.append(
            ParsedFragment(
                ordinal=len(fragments),
                kind=FragmentKind.PARAGRAPH,
                text=text,
                address=MarkdownSourceAddress(
                    heading_path=tuple(title for _, title in heading_stack),
                    line_start=first.number,
                    line_end=last.number,
                    char_start=first.start,
                    char_end=last.end,
                ),
            )
        )
        paragraph_lines.clear()

    for line in lines:
        if open_fence is not None:
            paragraph_lines.append(line)
            if _is_closing_fence(line.content, open_fence):
                open_fence = None
            continue

        heading = _parse_atx_heading(line)
        if heading is not None:
            flush_paragraph()
            while heading_stack and heading_stack[-1][0] >= heading.level:
                heading_stack.pop()
            heading_stack.append((heading.level, heading.text))
            fragments.append(
                ParsedFragment(
                    ordinal=len(fragments),
                    kind=FragmentKind.HEADING,
                    text=heading.text,
                    address=MarkdownSourceAddress(
                        heading_path=tuple(title for _, title in heading_stack),
                        line_start=line.number,
                        line_end=line.number,
                        char_start=heading.start,
                        char_end=heading.end,
                    ),
                )
            )
            continue

        if not line.content.strip():
            flush_paragraph()
            continue

        paragraph_lines.append(line)
        opening = _opening_fence(line.content)
        if opening is not None:
            open_fence = opening

    flush_paragraph()
    if not fragments:
        raise NoExtractableTextError("source contains no extractable text")
    return ParseResult.processed(
        tuple(fragments),
        parser_revision=parser_revision,
        normalized_codepoints=len(normalized),
    )


def _lines(text: str) -> tuple[_Line, ...]:
    records: list[_Line] = []
    offset = 0
    for number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
        content = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        records.append(_Line(number, offset, offset + len(content), content))
        offset += len(raw_line)
    return tuple(records)


def _parse_atx_heading(line: _Line) -> _Heading | None:
    match = _ATX_HEADING.fullmatch(line.content)
    if match is None:
        return None
    body = match.group("body")
    if body is None:
        return None
    body_start = match.start("body")
    content_end = len(line.content.rstrip(" \t"))
    closing = _CLOSING_MARKS.search(line.content[body_start:content_end])
    if closing is not None:
        content_end = body_start + closing.start()
    if content_end <= body_start:
        return None
    text = line.content[body_start:content_end]
    if not text.strip():
        return None
    return _Heading(
        level=len(match.group("marks")),
        text=text,
        start=line.start + body_start,
        end=line.start + content_end,
    )


def _opening_fence(line: str) -> tuple[str, int] | None:
    match = _OPEN_FENCE.match(line)
    if match is None:
        return None
    fence = match.group("fence")
    return fence[0], len(fence)


def _is_closing_fence(line: str, opening: tuple[str, int]) -> bool:
    marker, minimum = opening
    return re.fullmatch(rf" {{0,3}}{re.escape(marker)}{{{minimum},}}[ \t]*", line) is not None
