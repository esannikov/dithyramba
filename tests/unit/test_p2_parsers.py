from __future__ import annotations

import importlib.metadata
import io
import json
import os
import resource
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dithyramba.contracts import sha256_hex
from dithyramba.ingest import parsers, pdf, pdf_worker
from dithyramba.ingest.errors import (
    ExtractedTextLimitExceededError,
    NoExtractableTextError,
    ParserInvalidPdfError,
    ParserOutputInvalidError,
    ParserProcessError,
    ParserResourceLimitError,
    ParserTimeoutError,
    PdfPageLimitExceededError,
    UnsupportedEncryptedPdfError,
)
from dithyramba.ingest.models import (
    FileIdentity,
    FragmentKind,
    MarkdownSourceAddress,
    MediaType,
    ParserLimits,
    ParseStatus,
    PdfSourceAddress,
    SourceBytes,
)


def _source(data: bytes, media_type: MediaType, *, name: str | None = None) -> SourceBytes:
    default_names = {
        MediaType.MARKDOWN: "source.md",
        MediaType.PLAIN_TEXT: "source.txt",
        MediaType.PDF: "source.pdf",
    }
    relative_path = name or default_names[media_type]
    return SourceBytes(
        relative_path=relative_path,
        canonical_uri=f"file:///sources/{relative_path}",
        media_type=media_type,
        data=data,
        content_sha256=sha256_hex(data),
        source_modified_at="2026-07-20T12:00:00.000000Z",
        identity=FileIdentity(device=1, inode=2, size=len(data), modified_ns=3),
    )


def _pdf_bytes(page_texts: tuple[str | None, ...]) -> bytes:
    page_count = len(page_texts)
    page_ids = tuple(range(3, 3 + page_count))
    content_ids = tuple(range(3 + page_count, 3 + 2 * page_count))
    font_id = 3 + 2 * page_count
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
    ]
    for content_id in content_ids:
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode()
        )
    for page_text in page_texts:
        stream = (
            b""
            if page_text is None
            else f"BT /F1 12 Tf 72 720 Td ({page_text}) Tj ET".encode("ascii")
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)


def test_markdown_normalizes_unicode_newlines_and_exact_spans() -> None:
    source = _source(
        "# Cafe\u0301\r\n\r\nFirst line\rSecond line\n".encode(),
        MediaType.MARKDOWN,
    )

    result = parsers.parse_source(source, ParserLimits())

    assert result.status is ParseStatus.PROCESSED
    assert result.normalized_codepoints == len("# Café\n\nFirst line\nSecond line\n")
    heading, paragraph = result.fragments
    assert heading.kind is FragmentKind.HEADING
    assert heading.text == "Café"
    assert heading.address == MarkdownSourceAddress(("Café",), 1, 1, 2, 6)
    assert paragraph.text == "First line\nSecond line"
    assert paragraph.address == MarkdownSourceAddress(("Café",), 3, 4, 8, 30)


def test_markdown_heading_hierarchy_closing_marks_and_paragraph_context() -> None:
    normalized = "# Root #\nIntro\n\n### Deep ###\nBody\n\n## Sibling\nMore"
    result = parsers.parse_source(_source(normalized.encode(), MediaType.MARKDOWN), ParserLimits())

    assert [fragment.text for fragment in result.fragments] == [
        "Root",
        "Intro",
        "Deep",
        "Body",
        "Sibling",
        "More",
    ]
    addresses = [cast(MarkdownSourceAddress, fragment.address) for fragment in result.fragments]
    assert addresses[2].heading_path == ("Root", "Deep")
    assert addresses[4].heading_path == ("Root", "Sibling")
    assert normalized[addresses[2].char_start : addresses[2].char_end] == "Deep"


def test_markdown_does_not_treat_invalid_or_fenced_hashes_as_headings() -> None:
    text = "#No heading\n\n```python\n# inside fence\n```\n\n    # indented"
    result = parsers.parse_source(_source(text.encode(), MediaType.MARKDOWN), ParserLimits())

    assert all(fragment.kind is FragmentKind.PARAGRAPH for fragment in result.fragments)
    assert [fragment.text for fragment in result.fragments] == [
        "#No heading",
        "```python\n# inside fence\n```",
        "    # indented",
    ]


def test_empty_atx_headings_do_not_create_invalid_heading_fragments() -> None:
    assert parsers._parse_atx_heading(parsers._Line(1, 0, 1, "#")) is None
    assert parsers._parse_atx_heading(parsers._Line(1, 0, 4, "#   ")) is None


def test_plain_text_uses_same_exact_fragment_contract() -> None:
    result = parsers.parse_source(
        _source(b"# Section\n\nPlain text", MediaType.PLAIN_TEXT),
        ParserLimits(),
    )
    assert result.parser_revision == "plain_text_atx/1.0"
    assert [fragment.text for fragment in result.fragments] == ["Section", "Plain text"]


@pytest.mark.parametrize("data", [b"", b" \n\t\n"])
def test_empty_text_is_an_explicit_skip(data: bytes) -> None:
    result = parsers.parse_source(_source(data, MediaType.PLAIN_TEXT), ParserLimits())
    assert result.status is ParseStatus.SKIPPED
    assert result.failure_code == "no_extractable_text"


def test_invalid_utf8_nul_and_codepoint_limit_are_explicit_failures() -> None:
    invalid = parsers.parse_source(_source(b"\xff", MediaType.PLAIN_TEXT), ParserLimits())
    nul = parsers.parse_source(_source(b"hello\x00world", MediaType.PLAIN_TEXT), ParserLimits())
    limited = parsers.parse_source(
        _source(b"four", MediaType.PLAIN_TEXT),
        ParserLimits(max_extracted_codepoints=3),
    )
    assert invalid.failure_code == "invalid_utf8"
    assert nul.failure_code == "parser_output_invalid"
    assert limited.failure_code == "extracted_text_limit_exceeded"


def test_pdf_is_parsed_out_of_process_with_word_union_bbox(tmp_path: Path) -> None:
    result = parsers.parse_source(
        _source(_pdf_bytes(("Hello Dithyramba",)), MediaType.PDF),
        ParserLimits(),
        pdf_temp_root=tmp_path,
    )

    assert result.status is ParseStatus.PROCESSED
    assert result.parser_revision == pdf.PDF_PARSER_REVISION
    assert result.normalized_codepoints == 16
    fragment = result.fragments[0]
    assert fragment.text == "Hello Dithyramba"
    assert fragment.kind is FragmentKind.PAGE_TEXT
    address = cast(PdfSourceAddress, fragment.address)
    assert address.page == 1
    assert address.bbox == ("72.000", "62.484", "164.016", "74.484")


def test_pdf_signature_corruption_and_no_text_have_distinct_outcomes(tmp_path: Path) -> None:
    signature = parsers.parse_source(_source(b"not-a-pdf", MediaType.PDF), ParserLimits())
    corrupt = parsers.parse_source(
        _source(b"%PDF-corrupt", MediaType.PDF),
        ParserLimits(),
        pdf_temp_root=tmp_path,
    )
    no_text = parsers.parse_source(
        _source(_pdf_bytes((None,)), MediaType.PDF),
        ParserLimits(),
        pdf_temp_root=tmp_path,
    )
    assert signature.failure_code == "invalid_pdf_signature"
    assert corrupt.failure_code == "parser_invalid_pdf"
    assert no_text.status is ParseStatus.SKIPPED
    assert no_text.failure_code == "no_extractable_text"


def test_pdf_worker_enforces_page_and_text_limits(tmp_path: Path) -> None:
    page_limited = parsers.parse_source(
        _source(_pdf_bytes(("one", "two")), MediaType.PDF),
        ParserLimits(max_pdf_pages=1),
        pdf_temp_root=tmp_path,
    )
    text_limited = parsers.parse_source(
        _source(_pdf_bytes(("long text",)), MediaType.PDF),
        ParserLimits(max_extracted_codepoints=4),
        pdf_temp_root=tmp_path,
    )
    assert page_limited.failure_code == "page_limit_exceeded"
    assert text_limited.failure_code == "extracted_text_limit_exceeded"


def test_pdf_temp_root_and_exact_dependency_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(_pdf_bytes(("text",)), MediaType.PDF)
    relative_root = parsers.parse_source(source, ParserLimits(), pdf_temp_root=Path("relative"))
    missing_root = parsers.parse_source(
        source,
        ParserLimits(),
        pdf_temp_root=tmp_path / "missing",
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.11.9")
    wrong_version = parsers.parse_source(source, ParserLimits(), pdf_temp_root=tmp_path)
    assert relative_root.failure_code == "parser_process_failed"
    assert missing_root.failure_code == "parser_process_failed"
    assert wrong_version.failure_code == "parser_process_failed"

    def package_missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", package_missing)
    missing_dependency = parsers.parse_source(source, ParserLimits(), pdf_temp_root=tmp_path)
    assert missing_dependency.failure_code == "parser_process_failed"


def test_pdf_temp_root_rejects_symlink_and_none_uses_system_default(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    assert pdf._validated_temp_root(None) is None
    with pytest.raises(ParserProcessError, match="real directory"):
        pdf._validated_temp_root(linked)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "skipped", "code": "no_extractable_text"}, NoExtractableTextError),
        (
            {"status": "skipped", "code": "unsupported_encrypted_pdf"},
            UnsupportedEncryptedPdfError,
        ),
        (
            {"status": "failed", "code": "extracted_text_limit_exceeded"},
            ExtractedTextLimitExceededError,
        ),
        ({"status": "failed", "code": "page_limit_exceeded"}, PdfPageLimitExceededError),
        ({"status": "failed", "code": "parser_invalid_pdf"}, ParserInvalidPdfError),
        ({"status": "failed", "code": "parser_output_invalid"}, ParserOutputInvalidError),
        ({"status": "failed", "code": "parser_process_failed"}, ParserProcessError),
        ({"status": "failed", "code": "parser_resource_limit"}, ParserResourceLimitError),
        ({"status": "failed", "code": "parser_timeout"}, ParserTimeoutError),
    ],
)
def test_parent_maps_worker_outcome_codes_to_typed_errors(
    payload: dict[str, str],
    expected: type[Exception],
) -> None:
    with pytest.raises(expected):
        pdf._parse_worker_payload(payload, ParserLimits())


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"status": "unknown"},
        {"status": "failed", "code": "unknown"},
        {"status": "failed", "code": 1},
        {"status": "processed", "pages": []},
        {"status": "processed", "pages": "not-list"},
        {"status": "processed", "pages": [{"page": 1}]},
        {
            "status": "processed",
            "pages": [{"page": 0, "text": "text", "bbox": [0, 0, 1, 1]}],
        },
        {
            "status": "processed",
            "pages": [{"page": 1, "text": "", "bbox": [0, 0, 1, 1]}],
        },
        {
            "status": "processed",
            "pages": [{"page": 1, "text": "bad\rtext", "bbox": [0, 0, 1, 1]}],
        },
        {
            "status": "processed",
            "pages": [{"page": 1, "text": "text", "bbox": [0, 0, 0, 1]}],
        },
        {
            "status": "processed",
            "pages": [{"page": 1, "text": "text", "bbox": [0, 1, 2]}],
        },
    ],
)
def test_parent_rejects_invalid_worker_payloads(payload: Any) -> None:
    with pytest.raises(ParserOutputInvalidError):
        pdf._parse_worker_payload(payload, ParserLimits())


def test_parent_accepts_sparse_increasing_page_numbers_and_rounds_bbox() -> None:
    result = pdf._parse_worker_payload(
        {
            "status": "processed",
            "pages": [
                {"page": 1, "text": "one", "bbox": [0, 1, 2.3456, 4]},
                {"page": 3, "text": "two", "bbox": [1, 2, 3, 4]},
            ],
        },
        ParserLimits(max_pdf_pages=3),
    )
    assert [cast(PdfSourceAddress, item.address).page for item in result.fragments] == [1, 3]
    assert cast(PdfSourceAddress, result.fragments[0].address).bbox == (
        "0.000",
        "1.000",
        "2.346",
        "4.000",
    )


def test_parent_revalidates_worker_page_and_text_limits() -> None:
    pages = [
        {"page": 1, "text": "one", "bbox": [0, 0, 1, 1]},
        {"page": 2, "text": "two", "bbox": [0, 0, 1, 1]},
    ]
    with pytest.raises(ParserOutputInvalidError, match="page limit"):
        pdf._parse_worker_payload(
            {"status": "processed", "pages": pages},
            ParserLimits(max_pdf_pages=1),
        )
    with pytest.raises(ParserOutputInvalidError, match="text limit"):
        pdf._parse_worker_payload(
            {"status": "processed", "pages": pages[:1]},
            ParserLimits(max_extracted_codepoints=2),
        )


class _FakeProcess:
    def __init__(
        self,
        stdout: bytes = b"{}",
        *,
        returncode: int | None = 0,
        timeout_once: bool = False,
        pid: int = 123,
    ) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.timeout_once = timeout_once
        self.pid = pid
        self.killed = False

    def communicate(
        self,
        input: bytes | None = None,
        timeout: int | None = None,
    ) -> tuple[bytes, bytes]:
        if self.timeout_once:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("worker", float(timeout or 0))
        return self.stdout, b"stderr"

    def kill(self) -> None:
        self.killed = True

    def poll(self) -> int | None:
        return self.returncode


def _patch_process(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess,
) -> None:
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)


def test_worker_runner_rejects_invalid_json_duplicate_keys_and_large_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"%PDF-")
    _patch_process(monkeypatch, _FakeProcess(b"{"))
    with pytest.raises(ParserOutputInvalidError):
        pdf._run_worker(input_path, ParserLimits(), tmp_path)
    _patch_process(monkeypatch, _FakeProcess(b'{"a":1,"a":2}'))
    with pytest.raises(ParserOutputInvalidError):
        pdf._run_worker(input_path, ParserLimits(), tmp_path)
    limits = ParserLimits(max_extracted_codepoints=1, max_pdf_pages=1)
    _patch_process(monkeypatch, _FakeProcess(b"x" * 70_000))
    with pytest.raises(ParserOutputInvalidError, match="bound"):
        pdf._run_worker(input_path, limits, tmp_path)


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [(-9, ParserResourceLimitError), (1, ParserProcessError)],
)
def test_worker_runner_maps_process_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    expected: type[Exception],
) -> None:
    _patch_process(monkeypatch, _FakeProcess(returncode=returncode))
    with pytest.raises(expected):
        pdf._run_worker(tmp_path / "source.pdf", ParserLimits(), tmp_path)


def test_worker_runner_maps_spawn_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn")),
    )
    with pytest.raises(ParserProcessError, match="start"):
        pdf._run_worker(tmp_path / "source.pdf", ParserLimits(), tmp_path)

    process = _FakeProcess(timeout_once=True)
    _patch_process(monkeypatch, process)
    monkeypatch.setattr(pdf, "_kill_process_group", lambda _process: None)
    with pytest.raises(ParserTimeoutError):
        pdf._run_worker(tmp_path / "source.pdf", ParserLimits(timeout_seconds=1), tmp_path)


def test_process_group_kill_handles_no_pid_and_missing_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_pid = _FakeProcess(pid=0)
    pdf._kill_process_group(cast(Any, no_pid))
    assert no_pid.killed is True

    def missing(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", missing)
    pdf._kill_process_group(cast(Any, _FakeProcess()))


def test_darwin_rss_watchdog_kills_process_above_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=None)
    stop = threading.Event()
    exceeded = threading.Event()
    monitor_failed = threading.Event()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(pdf, "_darwin_rss_bytes", lambda _pid: 2 * 1024 * 1024)

    def kill(target: _FakeProcess) -> None:
        target.returncode = -9

    monkeypatch.setattr(pdf, "_kill_process_group", kill)
    monitor = pdf._start_rss_monitor(cast(Any, process), 1, stop, exceeded, monitor_failed)
    assert monitor is not None
    monitor.join(timeout=1)
    assert exceeded.is_set()
    assert not monitor_failed.is_set()
    assert process.returncode == -9


def test_darwin_rss_watchdog_fails_closed_and_kills_when_observation_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=None)
    stop = threading.Event()
    exceeded = threading.Event()
    monitor_failed = threading.Event()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(pdf, "_darwin_rss_bytes", lambda _pid: None)

    def kill(target: _FakeProcess) -> None:
        target.killed = True
        target.returncode = -9

    monkeypatch.setattr(pdf, "_kill_process_group", kill)
    monitor = pdf._start_rss_monitor(cast(Any, process), 1, stop, exceeded, monitor_failed)
    assert monitor is not None
    monitor.join(timeout=1)
    assert monitor_failed.is_set()
    assert not exceeded.is_set()
    assert process.killed is True


def test_darwin_rss_watchdog_accepts_exit_between_poll_and_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=None)
    stop = threading.Event()
    exceeded = threading.Event()
    monitor_failed = threading.Event()
    monkeypatch.setattr(sys, "platform", "darwin")

    def observe_after_exit(_pid: int) -> None:
        process.returncode = 0
        return None

    monkeypatch.setattr(pdf, "_darwin_rss_bytes", observe_after_exit)
    monitor = pdf._start_rss_monitor(cast(Any, process), 1, stop, exceeded, monitor_failed)
    assert monitor is not None
    monitor.join(timeout=1)
    assert not monitor.is_alive()
    assert not monitor_failed.is_set()
    assert not exceeded.is_set()
    assert process.killed is False


def test_worker_runner_maps_rss_monitor_failure_to_parser_process_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=0)
    _patch_process(monkeypatch, process)

    def fail_monitor(
        _process: object,
        _max_rss_mib: int,
        _stop: threading.Event,
        _exceeded: threading.Event,
        monitor_failed: threading.Event,
    ) -> None:
        monitor_failed.set()

    monkeypatch.setattr(pdf, "_start_rss_monitor", fail_monitor)
    with pytest.raises(ParserProcessError, match="monitor failed"):
        pdf._run_worker(tmp_path / "source.pdf", ParserLimits(), tmp_path)


def test_worker_runner_maps_rss_exceeded_event_to_resource_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_process(monkeypatch, _FakeProcess(returncode=0))

    def exceed_monitor(
        _process: object,
        _max_rss_mib: int,
        _stop: threading.Event,
        exceeded: threading.Event,
        _monitor_failed: threading.Event,
    ) -> None:
        exceeded.set()

    monkeypatch.setattr(pdf, "_start_rss_monitor", exceed_monitor)
    with pytest.raises(ParserResourceLimitError, match="resident-memory limit"):
        pdf._run_worker(tmp_path / "source.pdf", ParserLimits(), tmp_path)


def test_rss_monitor_is_disabled_off_macos_and_observer_exceptions_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=None)
    stop = threading.Event()
    exceeded = threading.Event()
    monitor_failed = threading.Event()
    monkeypatch.setattr(sys, "platform", "linux")
    assert pdf._start_rss_monitor(cast(Any, process), 1, stop, exceeded, monitor_failed) is None

    monkeypatch.setattr(sys, "platform", "darwin")

    def fail_observation(_pid: int) -> int:
        raise RuntimeError("observer failed")

    monkeypatch.setattr(pdf, "_darwin_rss_bytes", fail_observation)

    def kill(target: _FakeProcess) -> None:
        target.killed = True
        target.returncode = -9

    monkeypatch.setattr(pdf, "_kill_process_group", kill)
    monitor = pdf._start_rss_monitor(cast(Any, process), 1, stop, exceeded, monitor_failed)
    assert monitor is not None
    monitor.join(timeout=1)
    assert monitor_failed.is_set()
    assert process.killed is True


def test_darwin_rss_reader_parses_ps_and_fails_closed_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout=b"2048\n"),
    )
    assert pdf._darwin_rss_bytes(123) == 2 * 1024 * 1024
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout=b""),
    )
    assert pdf._darwin_rss_bytes(123) is None
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout=b"invalid"),
    )
    assert pdf._darwin_rss_bytes(123) is None


def test_private_pdf_write_wraps_os_failure(tmp_path: Path) -> None:
    with pytest.raises(ParserProcessError, match="materialize"):
        pdf._write_private_file(tmp_path / "missing" / "source.pdf", b"data")


def test_worker_word_union_validates_geometry() -> None:
    assert pdf_worker._word_union(
        [
            {"x0": 1, "top": 2, "x1": 3, "bottom": 4},
            {"x0": 0, "top": 3, "x1": 5, "bottom": 6},
        ]
    ) == [0.0, 2.0, 5.0, 6.0]
    with pytest.raises(ValueError):
        pdf_worker._word_union([{"x0": 1, "top": 2, "x1": 1, "bottom": 4}])
    with pytest.raises(ValueError):
        pdf_worker._word_union([{"x0": float("nan"), "top": 2, "x1": 3, "bottom": 4}])


def test_worker_extract_directly_processes_and_skips_pdf(tmp_path: Path) -> None:
    processed_path = tmp_path / "processed.pdf"
    processed_path.write_bytes(_pdf_bytes(("Direct",)))
    empty_path = tmp_path / "empty.pdf"
    empty_path.write_bytes(_pdf_bytes((None,)))
    base = {
        "max_pdf_pages": 10,
        "max_extracted_codepoints": 100,
        "max_rss_mib": 512,
        "timeout_seconds": 30,
    }
    assert pdf_worker._extract({**base, "input_path": str(processed_path)})["status"] == "processed"
    assert pdf_worker._extract({**base, "input_path": str(empty_path)}) == {
        "status": "skipped",
        "code": "no_extractable_text",
    }
    assert pdf_worker._extract({**base, "input_path": str(processed_path), "max_pdf_pages": 0}) == {
        "status": "failed",
        "code": "page_limit_exceeded",
    }
    assert pdf_worker._extract(
        {**base, "input_path": str(processed_path), "max_extracted_codepoints": 1}
    ) == {"status": "failed", "code": "extracted_text_limit_exceeded"}


def test_worker_extract_classifies_encrypted_wordless_and_parser_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pdfplumber
    from pdfminer.pdfdocument import PDFPasswordIncorrect
    from pdfminer.pdfparser import PDFSyntaxError

    config = {
        "input_path": "/tmp/source.pdf",
        "max_pdf_pages": 10,
        "max_extracted_codepoints": 100,
        "max_rss_mib": 512,
        "timeout_seconds": 30,
    }

    class FakePdf:
        def __init__(self, *, encrypted: bool = False, pages: list[Any] | None = None) -> None:
            self.doc = SimpleNamespace(encryption=("encrypted" if encrypted else None))
            self.pages = pages or []

        def __enter__(self) -> FakePdf:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class WordlessPage:
        def __init__(self) -> None:
            self.closed = False

        def extract_text(self) -> str:
            return "text"

        def extract_words(self) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(pdfplumber, "open", lambda _path: FakePdf(encrypted=True))
    assert pdf_worker._extract(config)["code"] == "unsupported_encrypted_pdf"
    wordless_page = WordlessPage()
    monkeypatch.setattr(pdfplumber, "open", lambda _path: FakePdf(pages=[wordless_page]))
    assert pdf_worker._extract(config)["code"] == "no_extractable_text"
    assert wordless_page.closed

    for exception, expected in [
        (PDFPasswordIncorrect(), "unsupported_encrypted_pdf"),
        (PDFSyntaxError("bad"), "parser_invalid_pdf"),
        (MemoryError(), "parser_resource_limit"),
        (RuntimeError("bad"), "parser_invalid_pdf"),
    ]:
        monkeypatch.setattr(
            pdfplumber,
            "open",
            lambda _path, exception=exception: (_ for _ in ()).throw(exception),
        )
        assert pdf_worker._extract(config)["code"] == expected


def test_worker_closes_page_caches_on_continue_success_and_early_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pdfplumber

    config = {
        "input_path": "/tmp/source.pdf",
        "max_pdf_pages": 10,
        "max_extracted_codepoints": 100,
        "max_rss_mib": 512,
        "timeout_seconds": 30,
    }

    class FakePage:
        def __init__(self, text: str, *, words: bool = True) -> None:
            self.text = text
            self.words = words
            self.close_count = 0

        def extract_text(self) -> str:
            return self.text

        def extract_words(self) -> list[dict[str, Any]]:
            if not self.words:
                return []
            return [{"x0": 1, "top": 2, "x1": 3, "bottom": 4}]

        def close(self) -> None:
            self.close_count += 1

    class FakePdf:
        def __init__(self, pages: list[FakePage]) -> None:
            self.doc = SimpleNamespace(encryption=None)
            self.pages = pages

        def __enter__(self) -> FakePdf:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    pages = [FakePage(""), FakePage("wordless", words=False), FakePage("kept")]
    monkeypatch.setattr(pdfplumber, "open", lambda _path: FakePdf(pages))
    assert pdf_worker._extract(config)["status"] == "processed"
    assert [page.close_count for page in pages] == [1, 1, 1]

    limited = FakePage("too long")
    monkeypatch.setattr(pdfplumber, "open", lambda _path: FakePdf([limited]))
    assert pdf_worker._extract({**config, "max_extracted_codepoints": 1}) == {
        "status": "failed",
        "code": "extracted_text_limit_exceeded",
    }
    assert limited.close_count == 1


def test_worker_config_validation_and_emission(monkeypatch: pytest.MonkeyPatch) -> None:
    valid = {
        "input_path": "/tmp/source.pdf",
        "max_pdf_pages": 1,
        "max_extracted_codepoints": 2,
        "max_rss_mib": 3,
        "timeout_seconds": 4,
    }
    fake_stdin = SimpleNamespace(buffer=io.BytesIO(json.dumps(valid).encode()))
    monkeypatch.setattr(sys, "stdin", cast(Any, fake_stdin))
    assert pdf_worker._read_config() == valid

    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    assert pdf_worker._emit("failed", "code") == 0
    assert json.loads(output.getvalue()) == {"status": "failed", "code": "code"}


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"{}",
        b'{"input_path":"relative"}',
        b"x" * (pdf_worker._MAX_CONFIG_BYTES + 1),
    ],
)
def test_worker_rejects_invalid_config(raw: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "stdin",
        cast(Any, SimpleNamespace(buffer=io.BytesIO(raw))),
    )
    with pytest.raises((ValueError, json.JSONDecodeError)):
        pdf_worker._read_config()


def test_worker_resource_limit_helper_handles_optional_and_required_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (100, 1_000))
    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda kind, value: calls.append((kind, value)),
    )
    pdf_worker._lower_soft_limit(7, 50)
    assert calls == [(7, (50, 1_000))]

    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda _kind, _value: (_ for _ in ()).throw(ValueError("unsupported")),
    )
    pdf_worker._lower_soft_limit(7, 50, optional=True)
    with pytest.raises(ValueError):
        pdf_worker._lower_soft_limit(7, 50)


def test_worker_soft_limit_clamps_infinite_soft_to_finite_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resource,
        "getrlimit",
        lambda _kind: (resource.RLIM_INFINITY, 40),
    )
    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda kind, value: calls.append((kind, value)),
    )
    pdf_worker._lower_soft_limit(7, 50)
    assert calls == [(7, (40, 40))]


def test_worker_main_emits_processed_and_resource_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "input_path": "/tmp/source.pdf",
        "max_pdf_pages": 1,
        "max_extracted_codepoints": 2,
        "max_rss_mib": 3,
        "timeout_seconds": 4,
    }
    monkeypatch.setattr(pdf_worker, "_read_config", lambda: config)
    monkeypatch.setattr(pdf_worker, "_apply_resource_limits", lambda _config: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.11.10")
    monkeypatch.setattr(
        pdf_worker,
        "_extract",
        lambda _config: {"status": "processed", "pages": []},
    )
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    assert pdf_worker.main() == 0
    assert json.loads(output.getvalue())["status"] == "processed"

    output.seek(0)
    output.truncate(0)
    monkeypatch.setattr(
        pdf_worker,
        "_extract",
        lambda _config: (_ for _ in ()).throw(MemoryError),
    )
    assert pdf_worker.main() == 0
    assert json.loads(output.getvalue())["code"] == "parser_resource_limit"
