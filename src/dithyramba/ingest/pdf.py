"""Parent-side PDF subprocess orchestration and output validation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import unicodedata
from pathlib import Path
from typing import Any

from .errors import (
    ExtractedTextLimitExceededError,
    InvalidPdfSignatureError,
    NoExtractableTextError,
    ParserInvalidPdfError,
    ParserOutputInvalidError,
    ParserProcessError,
    ParserResourceLimitError,
    ParserTimeoutError,
    PdfPageLimitExceededError,
    UnsupportedEncryptedPdfError,
)
from .models import (
    FragmentKind,
    ParsedFragment,
    ParseResult,
    ParserLimits,
    PdfSourceAddress,
    SourceBytes,
    format_pdf_point,
)

PDF_PARSER_REVISION = "pdfplumber/0.11.10+page_text_page_close/1.1"
_EXPECTED_PDFPLUMBER_VERSION = "0.11.10"
_WORKER = Path(__file__).with_name("pdf_worker.py")
_SKIP_CODES = {
    "no_extractable_text": NoExtractableTextError,
    "unsupported_encrypted_pdf": UnsupportedEncryptedPdfError,
}
_FAILURE_CODES = {
    "extracted_text_limit_exceeded": ExtractedTextLimitExceededError,
    "page_limit_exceeded": PdfPageLimitExceededError,
    "parser_invalid_pdf": ParserInvalidPdfError,
    "parser_output_invalid": ParserOutputInvalidError,
    "parser_process_failed": ParserProcessError,
    "parser_resource_limit": ParserResourceLimitError,
    "parser_timeout": ParserTimeoutError,
}


def parse_pdf_source(
    source: SourceBytes,
    limits: ParserLimits,
    *,
    temp_root: Path | None = None,
) -> ParseResult:
    """Parse a captured PDF in a resource-limited isolated subprocess."""

    if not source.data.startswith(b"%PDF-"):
        raise InvalidPdfSignatureError("PDF bytes must begin with %PDF-")
    try:
        installed_version = importlib.metadata.version("pdfplumber")
    except importlib.metadata.PackageNotFoundError as error:
        raise ParserProcessError("pdfplumber is not installed") from error
    if installed_version != _EXPECTED_PDFPLUMBER_VERSION:
        raise ParserProcessError(f"pdfplumber {_EXPECTED_PDFPLUMBER_VERSION} is required exactly")

    parent = _validated_temp_root(temp_root)
    work_directory = Path(tempfile.mkdtemp(prefix="dithyramba-pdf-", dir=parent))
    os.chmod(work_directory, 0o700)
    try:
        input_path = work_directory / "source.pdf"
        _write_private_file(input_path, source.data)
        payload = _run_worker(input_path, limits, work_directory)
        return _parse_worker_payload(payload, limits)
    finally:
        shutil.rmtree(work_directory, ignore_errors=True)


def _validated_temp_root(temp_root: Path | None) -> str | None:
    if temp_root is None:
        return None
    if not temp_root.is_absolute() or ".." in temp_root.parts:
        raise ParserProcessError("PDF temporary root must be an absolute path")
    try:
        root_lstat = temp_root.lstat()
        resolved = temp_root.resolve(strict=True)
    except OSError as error:
        raise ParserProcessError("PDF temporary root is unavailable") from error
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise ParserProcessError("PDF temporary root must be a real directory")
    return str(resolved)


def _write_private_file(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(data)
            written = 0
            while written < len(view):
                written += os.write(descriptor, view[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ParserProcessError("could not materialize captured PDF") from error


def _run_worker(input_path: Path, limits: ParserLimits, work_directory: Path) -> Any:
    config = json.dumps(
        {
            "input_path": str(input_path),
            "max_pdf_pages": limits.max_pdf_pages,
            "max_extracted_codepoints": limits.max_extracted_codepoints,
            "max_rss_mib": limits.max_rss_mib,
            "timeout_seconds": limits.timeout_seconds,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    command = [sys.executable, "-I", str(_WORKER)]
    environment = {"PYTHONHASHSEED": "0", "PYTHONUTF8": "1"}
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_directory,
            env=environment,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as error:
        raise ParserProcessError("could not start PDF parser process") from error
    monitor_stop = threading.Event()
    rss_exceeded = threading.Event()
    monitor_failed = threading.Event()
    monitor = _start_rss_monitor(
        process,
        limits.max_rss_mib,
        monitor_stop,
        rss_exceeded,
        monitor_failed,
    )
    try:
        stdout, _stderr = process.communicate(input=config, timeout=limits.timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _kill_process_group(process)
        process.communicate()
        raise ParserTimeoutError("PDF parser exceeded its timeout") from error
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=1)
            if monitor.is_alive():
                monitor_failed.set()
                _kill_process_group(process)
    if monitor_failed.is_set():
        raise ParserProcessError("PDF parser resident-memory monitor failed")
    if rss_exceeded.is_set():
        raise ParserResourceLimitError("PDF parser exceeded its resident-memory limit")
    if process.returncode != 0:
        if process.returncode is not None and process.returncode < 0:
            raise ParserResourceLimitError("PDF parser was terminated by a resource limit")
        raise ParserProcessError("PDF parser process failed")
    maximum_output = limits.max_extracted_codepoints * 8 + limits.max_pdf_pages * 512 + 65_536
    if len(stdout) > maximum_output:
        raise ParserOutputInvalidError("PDF parser output exceeded its validated bound")
    try:
        return json.loads(stdout, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ParserOutputInvalidError("PDF parser returned invalid JSON") from error


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.pid <= 0:
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _start_rss_monitor(
    process: subprocess.Popen[bytes],
    max_rss_mib: int,
    stop: threading.Event,
    exceeded: threading.Event,
    monitor_failed: threading.Event,
) -> threading.Thread | None:
    """Enforce RSS on macOS, where ``RLIMIT_RSS`` is only advisory/unsupported."""

    if not _is_darwin_platform():
        return None

    def monitor() -> None:
        while not stop.is_set():
            if process.poll() is not None:
                return
            try:
                rss_bytes = _darwin_rss_bytes(process.pid)
            except Exception:
                rss_bytes = None
            if rss_bytes is None:
                # A short-lived worker can exit after the poll above but before
                # ``ps`` observes it.  Treat that completed-process race as a
                # normal monitor shutdown; an unobservable process that is
                # still running remains a fail-closed condition.
                if process.poll() is not None:
                    return
                monitor_failed.set()
                _kill_process_group(process)
                return
            if rss_bytes > max_rss_mib * 1024 * 1024:
                exceeded.set()
                _kill_process_group(process)
                return
            if stop.wait(0.1):
                return

    thread = threading.Thread(target=monitor, name="dithyramba-pdf-rss", daemon=True)
    thread.start()
    return thread


def _is_darwin_platform() -> bool:
    """Keep runtime platform selection testable without static-platform dead code."""

    return sys.platform == "darwin"


def _darwin_rss_bytes(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=1,
            env={"LC_ALL": "C"},
            close_fds=True,
        )
        if completed.returncode != 0:
            return None
        rss_kib = int(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return rss_kib * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _parse_worker_payload(payload: Any, limits: ParserLimits) -> ParseResult:
    if type(payload) is not dict:
        raise ParserOutputInvalidError("PDF parser result must be an object")
    status = payload.get("status")
    if status in ("skipped", "failed"):
        if set(payload) != {"status", "code"} or type(payload.get("code")) is not str:
            raise ParserOutputInvalidError("PDF parser failure payload is invalid")
        code = payload["code"]
        error_type = _SKIP_CODES.get(code) if status == "skipped" else _FAILURE_CODES.get(code)
        if error_type is None:
            raise ParserOutputInvalidError("PDF parser returned an unknown failure code")
        raise error_type(f"PDF parser outcome: {code}")

    if status != "processed" or set(payload) != {"status", "pages"}:
        raise ParserOutputInvalidError("PDF parser result shape is invalid")
    pages = payload["pages"]
    if type(pages) is not list or not pages:
        raise ParserOutputInvalidError("processed PDF result requires pages")
    if len(pages) > limits.max_pdf_pages:
        raise ParserOutputInvalidError("PDF parser output exceeds the page limit")

    fragments: list[ParsedFragment] = []
    codepoints = 0
    previous_page = 0
    for raw_page in pages:
        if type(raw_page) is not dict or set(raw_page) != {"page", "text", "bbox"}:
            raise ParserOutputInvalidError("PDF page result shape is invalid")
        page = raw_page["page"]
        text = raw_page["text"]
        bbox = raw_page["bbox"]
        if type(page) is not int or page <= previous_page or page > limits.max_pdf_pages:
            raise ParserOutputInvalidError("PDF page numbers must be increasing and in range")
        if type(text) is not str or not text.strip() or "\x00" in text:
            raise ParserOutputInvalidError("PDF page text must be non-empty text without NUL")
        if unicodedata.normalize("NFC", text) != text or "\r" in text:
            raise ParserOutputInvalidError("PDF page text is not NFC/LF normalized")
        if type(bbox) is not list or len(bbox) != 4:
            raise ParserOutputInvalidError("PDF page bbox must contain four points")
        try:
            formatted_bbox = tuple(format_pdf_point(point) for point in bbox)
            address = PdfSourceAddress(
                page=page,
                bbox=(formatted_bbox[0], formatted_bbox[1], formatted_bbox[2], formatted_bbox[3]),
                char_start=0,
                char_end=len(text),
            )
        except (TypeError, ValueError) as error:
            raise ParserOutputInvalidError("PDF page bbox is invalid") from error
        codepoints += len(text)
        if codepoints > limits.max_extracted_codepoints:
            raise ParserOutputInvalidError("PDF parser output exceeds the text limit")
        fragments.append(
            ParsedFragment(
                ordinal=len(fragments),
                kind=FragmentKind.PAGE_TEXT,
                text=text,
                address=address,
            )
        )
        previous_page = page
    return ParseResult.processed(
        tuple(fragments),
        parser_revision=PDF_PARSER_REVISION,
        normalized_codepoints=codepoints,
    )
