"""Typed failures for source discovery, safe reads, parsing, and ingest."""

from __future__ import annotations

from typing import ClassVar, Literal

FailureOutcome = Literal["failed", "skipped"]


class IngestError(Exception):
    """Base failure carrying a stable machine code and retry contract."""

    code: ClassVar[str] = "ingest_error"
    retryable: ClassVar[bool] = False
    outcome: ClassVar[FailureOutcome] = "failed"


class InvalidSourcePathError(IngestError):
    code = "invalid_source_path"


class SourceIdentityManifestInvalidError(IngestError):
    code = "source_identity_manifest_invalid"


class SourceUnavailableError(IngestError):
    code = "source_unavailable"
    retryable = True


class SourceSymlinkError(IngestError):
    code = "source_symlink_rejected"


class SourceNotRegularError(IngestError):
    code = "source_not_regular"


class SourceChangedDuringIngestError(IngestError):
    code = "source_changed_during_ingest"
    retryable = True


class FileSizeLimitExceededError(IngestError):
    code = "file_size_limit_exceeded"


class UnsupportedFormatError(IngestError):
    code = "unsupported_format"


class InvalidPdfSignatureError(IngestError):
    code = "invalid_pdf_signature"


class InvalidUtf8Error(IngestError):
    code = "invalid_utf8"


class ExtractedTextLimitExceededError(IngestError):
    code = "extracted_text_limit_exceeded"


class NoExtractableTextError(IngestError):
    code = "no_extractable_text"
    outcome = "skipped"


class UnsupportedEncryptedPdfError(IngestError):
    code = "unsupported_encrypted_pdf"
    outcome = "skipped"


class PdfPageLimitExceededError(IngestError):
    code = "page_limit_exceeded"


class ParserInvalidPdfError(IngestError):
    code = "parser_invalid_pdf"


class ParserTimeoutError(IngestError):
    code = "parser_timeout"
    retryable = True


class ParserResourceLimitError(IngestError):
    code = "parser_resource_limit"
    retryable = True


class ParserOutputInvalidError(IngestError):
    code = "parser_output_invalid"


class ParserProcessError(IngestError):
    code = "parser_process_failed"
    retryable = True


class IngestPersistenceError(IngestError):
    code = "ingest_persistence_failed"
    retryable = True
