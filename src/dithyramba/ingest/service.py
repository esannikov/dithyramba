"""Library-scoped synchronous ingest orchestration and outcome accounting."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path, PurePosixPath

from dithyramba._version import __version__
from dithyramba.collections import (
    CollectionError,
    CollectionRoot,
    load_source_identity_manifest,
)
from dithyramba.ingest.errors import (
    IngestError,
    IngestPersistenceError,
    InvalidSourcePathError,
    SourceIdentityManifestInvalidError,
)
from dithyramba.ingest.models import PARSER_PROFILE, ParserLimits, ParseStatus
from dithyramba.ingest.parsers import parse_source
from dithyramba.ingest.reader import (
    discover_source_paths,
    read_source_bytes,
    validate_source_unchanged,
)
from dithyramba.persistence.errors import PersistenceError
from dithyramba.persistence.repository import LibraryRepository
from dithyramba.provenance import (
    BlobStore,
    IngestBatchResult,
    IngestInputOutcome,
    ProcessingRunRecord,
    ProcessingRunStatus,
    ProvenanceError,
    TerminalInputOutcome,
    build_ingest_coverage,
)
from dithyramba.provenance.blobs import PrivateLockStore


class IngestService:
    """Turn bounded Collection inputs into immutable, auditable provenance."""

    def __init__(
        self,
        repository: LibraryRepository,
        *,
        limits: ParserLimits | None = None,
        code_version: str = __version__,
        profile_version: str = PARSER_PROFILE,
        pdf_temp_root: Path | None = None,
    ) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("IngestService requires a LibraryRepository")
        self._repository = repository
        self._limits = limits or ParserLimits()
        self._code_version = code_version
        self._profile_version = profile_version
        self._pdf_temp_root = pdf_temp_root or repository.paths.cache
        self._blobs = BlobStore(repository.paths)
        self._locks = PrivateLockStore(repository.paths)

    @property
    def repository(self) -> LibraryRepository:
        """Return the single Library repository owned by this service."""

        return self._repository

    def ingest_collection(self, collection_id: str) -> IngestBatchResult:
        """Discover and ingest every allowed path under all Collection roots."""

        collection = self._repository.get_collection(collection_id)
        with self._repository._store.defer_wal_autocheckpoints():
            run = self._begin_run()
            outcomes: list[IngestInputOutcome] = []
            current_root_id = (
                collection.collection_root_ids[0] if collection.collection_root_ids else "root_none"
            )
            current_relative_path: str | None = None
            try:
                for root_id, root in zip(
                    collection.collection_root_ids,
                    collection.config.roots,
                    strict=True,
                ):
                    current_root_id = root_id
                    current_relative_path = None
                    try:
                        relative_paths = discover_source_paths(root)
                    except IngestError as exc:
                        outcomes.append(_input_error(root_id, None, exc))
                        continue
                    for relative_path in relative_paths:
                        current_relative_path = relative_path
                        outcomes.append(
                            self._ingest_one(
                                collection_id=collection_id,
                                collection_root_id=root_id,
                                root=root,
                                relative_path=relative_path,
                            )
                        )
                return self._finish_run(
                    run.processing_run_id,
                    collection_id,
                    tuple(outcomes),
                    ProcessingRunStatus.SUCCEEDED,
                )
            except (ProvenanceError, PersistenceError, sqlite3.Error, OSError) as exc:
                _replace_or_append_outcome(
                    outcomes,
                    _infrastructure_error(current_root_id, current_relative_path, exc),
                )
                return self._finish_run(
                    run.processing_run_id,
                    collection_id,
                    tuple(outcomes),
                    ProcessingRunStatus.FAILED,
                    error_code="ingest_persistence_failed",
                )
            except Exception:
                _replace_or_append_outcome(
                    outcomes,
                    IngestInputOutcome(
                        collection_root_id=current_root_id,
                        relative_path=current_relative_path,
                        terminal_outcome=TerminalInputOutcome.FAILED,
                        failure_code="internal_ingest_error",
                        retryable=False,
                        infrastructure_failure=True,
                    ),
                )
                try:
                    self._finish_run(
                        run.processing_run_id,
                        collection_id,
                        tuple(outcomes),
                        ProcessingRunStatus.FAILED,
                        error_code="internal_ingest_error",
                    )
                except Exception as completion_error:
                    raise IngestPersistenceError(
                        "ingest failed and its failure receipt could not be persisted"
                    ) from completion_error
                raise

    def ingest_path(
        self,
        collection_id: str,
        relative_path: str,
        *,
        collection_root_id: str | None = None,
    ) -> IngestBatchResult:
        """Ingest one explicit relative path and still emit a full run/report."""

        collection = self._repository.get_collection(collection_id)
        selected_root_id, selected_root = _select_root(
            collection.collection_root_ids,
            collection.config.roots,
            requested_id=collection_root_id,
        )
        run = self._begin_run()
        try:
            outcome = self._ingest_one(
                collection_id=collection_id,
                collection_root_id=selected_root_id,
                root=selected_root,
                relative_path=relative_path,
            )
            return self._finish_run(
                run.processing_run_id,
                collection_id,
                (outcome,),
                ProcessingRunStatus.SUCCEEDED,
            )
        except (ProvenanceError, PersistenceError, sqlite3.Error, OSError) as exc:
            outcome = _infrastructure_error(selected_root_id, relative_path, exc)
            return self._finish_run(
                run.processing_run_id,
                collection_id,
                (outcome,),
                ProcessingRunStatus.FAILED,
                error_code="ingest_persistence_failed",
            )
        except Exception:
            outcome = IngestInputOutcome(
                collection_root_id=selected_root_id,
                relative_path=relative_path,
                terminal_outcome=TerminalInputOutcome.FAILED,
                failure_code="internal_ingest_error",
                infrastructure_failure=True,
            )
            try:
                self._finish_run(
                    run.processing_run_id,
                    collection_id,
                    (outcome,),
                    ProcessingRunStatus.FAILED,
                    error_code="internal_ingest_error",
                )
            except Exception as completion_error:
                raise IngestPersistenceError(
                    "ingest failed and its failure receipt could not be persisted"
                ) from completion_error
            raise

    def _begin_run(self) -> ProcessingRunRecord:
        try:
            return self._repository.begin_ingest_run(
                code_version=self._code_version,
                profile_version=self._profile_version,
            )
        except Exception as exc:
            raise IngestPersistenceError("ingest ProcessingRun could not be started") from exc

    def _ingest_one(
        self,
        *,
        collection_id: str,
        collection_root_id: str,
        root: CollectionRoot,
        relative_path: str,
    ) -> IngestInputOutcome:
        try:
            lock_key = _canonical_source_lock_key(root, relative_path)
        except IngestError as exc:
            return _input_error(collection_root_id, relative_path, exc)
        with self._locks.hold("sources", lock_key):
            try:
                manifest = load_source_identity_manifest(root)
                declaration = (
                    manifest.declaration_for(relative_path) if manifest is not None else None
                )
            except CollectionError as exc:
                return _input_error(
                    collection_root_id,
                    relative_path,
                    SourceIdentityManifestInvalidError(str(exc)),
                )
            try:
                source = read_source_bytes(root, relative_path, self._limits)
            except IngestError as exc:
                return _input_error(collection_root_id, relative_path, exc)
            reusable = self._repository.find_reusable_unchanged_source(
                collection_id=collection_id,
                collection_root_id=collection_root_id,
                source=source,
                parser_profile=self._profile_version,
                identity_declaration=declaration,
            )
            if reusable is not None:
                try:
                    validate_source_unchanged(root, source, self._limits)
                except IngestError as exc:
                    return _input_error(collection_root_id, relative_path, exc)
                return IngestInputOutcome(
                    collection_root_id=collection_root_id,
                    relative_path=relative_path,
                    terminal_outcome=TerminalInputOutcome.PROCESSED,
                    disposition=reusable.disposition,
                    source_id=reusable.source_id,
                    source_version_id=reusable.source_version_id,
                    source_family_id=reusable.source_family_id,
                    root_source_id=reusable.root_source_id,
                    fragment_count=reusable.fragment_count,
                )
            parsed = replace(
                parse_source(source, self._limits, pdf_temp_root=self._pdf_temp_root),
                parser_profile=self._profile_version,
            )
            try:
                validate_source_unchanged(root, source, self._limits)
            except IngestError as exc:
                return _input_error(collection_root_id, relative_path, exc)
            if parsed.status is ParseStatus.SKIPPED:
                return IngestInputOutcome(
                    collection_root_id=collection_root_id,
                    relative_path=relative_path,
                    terminal_outcome=TerminalInputOutcome.SKIPPED,
                    failure_code=parsed.failure_code,
                )
            if parsed.status is ParseStatus.FAILED:
                return IngestInputOutcome(
                    collection_root_id=collection_root_id,
                    relative_path=relative_path,
                    terminal_outcome=TerminalInputOutcome.FAILED,
                    failure_code=parsed.failure_code,
                    retryable=parsed.failure_code
                    in {
                        "parser_process_failed",
                        "parser_resource_limit",
                        "parser_timeout",
                    },
                )
            blob = self._blobs.promote(source.data, source.content_sha256)
            persisted = self._repository.record_processed_source(
                collection_id=collection_id,
                collection_root_id=collection_root_id,
                source=source,
                parsed=parsed,
                blob=blob,
                identity_declaration=declaration,
            )
        return IngestInputOutcome(
            collection_root_id=collection_root_id,
            relative_path=relative_path,
            terminal_outcome=TerminalInputOutcome.PROCESSED,
            disposition=persisted.disposition,
            source_id=persisted.source_id,
            source_version_id=persisted.source_version_id,
            source_family_id=persisted.source_family_id,
            root_source_id=persisted.root_source_id,
            fragment_count=persisted.fragment_count,
        )

    def _finish_run(
        self,
        processing_run_id: str,
        collection_id: str,
        outcomes: tuple[IngestInputOutcome, ...],
        status: ProcessingRunStatus,
        *,
        error_code: str | None = None,
    ) -> IngestBatchResult:
        coverage = build_ingest_coverage(
            processing_run_id=processing_run_id,
            collection_id=collection_id,
            outcomes=outcomes,
        )
        run_record, coverage_record = self._repository.complete_ingest_run(
            coverage=coverage,
            status=status,
            error_code=error_code,
        )
        return IngestBatchResult(run_record, coverage_record, outcomes)


def _input_error(
    collection_root_id: str,
    relative_path: str | None,
    error: IngestError,
) -> IngestInputOutcome:
    terminal = (
        TerminalInputOutcome.SKIPPED
        if error.outcome == TerminalInputOutcome.SKIPPED.value
        else TerminalInputOutcome.FAILED
    )
    return IngestInputOutcome(
        collection_root_id=collection_root_id,
        relative_path=relative_path,
        terminal_outcome=terminal,
        failure_code=error.code,
        retryable=error.retryable,
    )


def _infrastructure_error(
    collection_root_id: str,
    relative_path: str | None,
    error: BaseException,
) -> IngestInputOutcome:
    del error
    return IngestInputOutcome(
        collection_root_id=collection_root_id,
        relative_path=relative_path,
        terminal_outcome=TerminalInputOutcome.FAILED,
        failure_code="ingest_persistence_failed",
        retryable=True,
        infrastructure_failure=True,
    )


def _replace_or_append_outcome(
    outcomes: list[IngestInputOutcome],
    terminal: IngestInputOutcome,
) -> None:
    """Keep one terminal classification per exact Collection input."""

    if outcomes and (
        outcomes[-1].collection_root_id,
        outcomes[-1].relative_path,
    ) == (terminal.collection_root_id, terminal.relative_path):
        outcomes[-1] = terminal
    else:
        outcomes.append(terminal)


def _select_root(
    root_ids: tuple[str, ...],
    roots: tuple[CollectionRoot, ...],
    *,
    requested_id: str | None,
) -> tuple[str, CollectionRoot]:
    if not root_ids:
        raise ValueError("Collection has no roots")
    if requested_id is None:
        if len(root_ids) != 1:
            raise ValueError("collection_root_id is required for a multi-root Collection")
        return root_ids[0], roots[0]
    try:
        index = root_ids.index(requested_id)
    except ValueError as exc:
        raise ValueError("collection_root_id is absent from the Collection") from exc
    return root_ids[index], roots[index]


def _canonical_source_lock_key(root: CollectionRoot, relative_path: str) -> str:
    """Derive the exact canonical URI before the first descriptor capture."""

    if type(relative_path) is not str or not relative_path or "\x00" in relative_path:
        raise InvalidSourcePathError("source path must be non-empty text without NUL")
    if "\\" in relative_path:
        raise InvalidSourcePathError("source path must use POSIX separators")
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != relative_path
    ):
        raise InvalidSourcePathError("source path must be normalized and relative")
    return (root.path / Path(*relative.parts)).as_uri()
