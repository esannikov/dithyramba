"""Isolated pdfplumber worker.

This module is executed as a script by :mod:`dithyramba.ingest.pdf`; it is not
an in-process parsing API. It emits one small JSON document and never receives
source bytes through a shell or template channel.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import resource
import sys
import unicodedata
from pathlib import Path
from typing import Any

_EXPECTED_PDFPLUMBER_VERSION = "0.11.10"
_MAX_CONFIG_BYTES = 64 * 1024


def main() -> int:
    """Apply limits, extract page text, and write a validated-shape payload."""

    try:
        config = _read_config()
        _apply_resource_limits(config)
        if importlib.metadata.version("pdfplumber") != _EXPECTED_PDFPLUMBER_VERSION:
            return _emit("failed", "parser_process_failed")
        result = _extract(config)
    except MemoryError:
        return _emit("failed", "parser_resource_limit")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return _emit("failed", "parser_process_failed")
    sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return 0


def _read_config() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(_MAX_CONFIG_BYTES + 1)
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError("worker config is too large")
    decoded = json.loads(raw)
    if type(decoded) is not dict:
        raise ValueError("worker config must be an object")
    expected = {
        "input_path",
        "max_pdf_pages",
        "max_extracted_codepoints",
        "max_rss_mib",
        "timeout_seconds",
    }
    if set(decoded) != expected:
        raise ValueError("worker config keys are invalid")
    input_path = decoded["input_path"]
    if type(input_path) is not str or not Path(input_path).is_absolute():
        raise ValueError("input_path must be absolute")
    for key in (
        "max_pdf_pages",
        "max_extracted_codepoints",
        "max_rss_mib",
        "timeout_seconds",
    ):
        if type(decoded[key]) is not int or decoded[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return decoded


def _apply_resource_limits(config: dict[str, Any]) -> None:
    rss_bytes = int(config["max_rss_mib"]) * 1024 * 1024
    _lower_soft_limit(resource.RLIMIT_RSS, rss_bytes, optional=True)
    if sys.platform.startswith("linux"):
        _lower_soft_limit(resource.RLIMIT_AS, rss_bytes)
    _lower_soft_limit(resource.RLIMIT_CPU, int(config["timeout_seconds"]))
    _lower_soft_limit(resource.RLIMIT_NOFILE, 64)


def _lower_soft_limit(kind: int, requested: int, *, optional: bool = False) -> None:
    soft, hard = resource.getrlimit(kind)
    hard_cap = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    finite_soft = hard_cap if soft == resource.RLIM_INFINITY else min(hard_cap, soft)
    try:
        resource.setrlimit(kind, (finite_soft, hard))
    except (OSError, ValueError):
        if not optional:
            raise


def _extract(config: dict[str, Any]) -> dict[str, object]:
    import pdfplumber
    from pdfminer.pdfdocument import PDFPasswordIncorrect
    from pdfminer.pdfparser import PDFSyntaxError
    from pdfminer.psparser import PSEOF

    input_path = str(config["input_path"])
    max_pages = int(config["max_pdf_pages"])
    max_codepoints = int(config["max_extracted_codepoints"])
    try:
        with pdfplumber.open(input_path) as pdf:
            if pdf.doc.encryption is not None:
                return {"status": "skipped", "code": "unsupported_encrypted_pdf"}
            if len(pdf.pages) > max_pages:
                return {"status": "failed", "code": "page_limit_exceeded"}
            pages: list[dict[str, object]] = []
            total = 0
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    text = unicodedata.normalize(
                        "NFC", text.replace("\r\n", "\n").replace("\r", "\n")
                    )
                    if not text.strip():
                        continue
                    words = page.extract_words()
                    if not words:
                        continue
                    bbox = _word_union(words)
                    total += len(text)
                    if total > max_codepoints:
                        return {"status": "failed", "code": "extracted_text_limit_exceeded"}
                    pages.append({"page": page_number, "text": text, "bbox": bbox})
                finally:
                    # pdfplumber caches layout, object and text maps on each Page.
                    # The document retains every Page instance in ``pdf.pages``, so
                    # explicit close is required to keep long-book RSS bounded.
                    page.close()
    except PDFPasswordIncorrect:
        return {"status": "skipped", "code": "unsupported_encrypted_pdf"}
    except (PDFSyntaxError, PSEOF, ValueError, TypeError):
        return {"status": "failed", "code": "parser_invalid_pdf"}
    except MemoryError:
        return {"status": "failed", "code": "parser_resource_limit"}
    except Exception:  # hostile parser inputs can surface library-specific exceptions
        return {"status": "failed", "code": "parser_invalid_pdf"}
    if not pages:
        return {"status": "skipped", "code": "no_extractable_text"}
    return {"status": "processed", "pages": pages}


def _word_union(words: list[dict[str, Any]]) -> list[float]:
    coordinates: list[tuple[float, float, float, float]] = []
    for word in words:
        values = tuple(float(word[key]) for key in ("x0", "top", "x1", "bottom"))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("word bbox is non-finite")
        x0, top, x1, bottom = values
        if x0 >= x1 or top >= bottom:
            raise ValueError("word bbox is invalid")
        coordinates.append((x0, top, x1, bottom))
    return [
        min(item[0] for item in coordinates),
        min(item[1] for item in coordinates),
        max(item[2] for item in coordinates),
        max(item[3] for item in coordinates),
    ]


def _emit(status: str, code: str) -> int:
    sys.stdout.write(json.dumps({"status": status, "code": code}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
