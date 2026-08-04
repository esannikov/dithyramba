"""Library-scoped repositories, outbox delivery, and online backup."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import cast

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    CompiledAccess,
    MembershipState,
    PermittedManifestItem,
    PolicyEffect,
    RequestScope,
    SnapshotCandidate,
    SnapshotMembership,
    SourceRule,
    compile_access,
)
from dithyramba.collections import (
    CollectionConfig,
    CollectionError,
    CollectionKind,
    CollectionRoot,
    SourceIdentityDeclaration,
    build_collection_root,
    load_source_identity_manifest,
)
from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    new_id,
    sha256_hex,
)
from dithyramba.ingest.models import ParseResult, ParseStatus, SourceBytes
from dithyramba.library import (
    LibraryConfig,
    LibraryPaths,
    application_data_root,
    create_library_layout,
    library_paths,
    validate_library_id,
    validate_library_layout,
)
from dithyramba.provenance import (
    BlobRecord,
    BlobStore,
    CollectionMembershipRecord,
    CoverageDraft,
    CoverageReportRecord,
    IngestDisposition,
    IngestInputOutcome,
    OmissionRecord,
    PersistedSourceOutcome,
    ProcessingRunRecord,
    ProcessingRunStatus,
    PromotedBlob,
    SourceFamilyRole,
    SourceFragmentRecord,
    SourceLineageIntegrityError,
    SourceRecord,
    SourceVersionRecord,
    build_ingest_coverage,
)
from dithyramba.snapshots import CorpusSnapshot, SnapshotMember, build_corpus_snapshot
from dithyramba.store import MigrationRunner, Store, schema_fingerprint, verify_integrity

from .errors import (
    AccessPolicyNotFoundError,
    AuthorizationError,
    BackupError,
    CollectionNotFoundError,
    CorpusSnapshotNotFoundError,
    CoverageReportNotFoundError,
    LibraryNotFoundError,
    OutboxExportError,
    PersistenceConflictError,
    PersistenceIntegrityError,
    ProcessingRunNotFoundError,
    SourceNotFoundError,
    SourceVersionNotFoundError,
)
from .models import (
    AccessPolicyRecord,
    AuthorizedRead,
    BackupRecord,
    CollectionRecord,
    CorpusSnapshotSummary,
    LibraryCounts,
    LibraryDescription,
    LibraryRecord,
    OutboxEvent,
    OutboxExportResult,
    SourceFragmentText,
)

Clock = Callable[[], datetime]
EventIdFactory = Callable[[], str]
AfterAppendHook = Callable[[OutboxEvent], None]

_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_LABEL_LENGTH = 200
_LARGE_FRAGMENT_READ_THRESHOLD = 300
_FRAGMENT_READ_PERMIT_TABLE = "dithyramba_fragment_read_permit"


@dataclass(slots=True)
class _CandidateParts:
    source_version_id: str
    source_id: str
    memberships: list[SnapshotMembership]


@dataclass(frozen=True, slots=True)
class _AuthorizedReadContext:
    """Independent issuance-time inputs used after capability revalidation."""

    access_policy_id: str
    scope: RequestScope
    compiled: CompiledAccess


@dataclass(frozen=True, slots=True)
class _IssuedAuthorization:
    """Repository identity plus an immutable issuance-time context and fingerprint."""

    capability: AuthorizedRead
    context: _AuthorizedReadContext
    fingerprint: str


def _authorization_fingerprint(authorization: AuthorizedRead) -> str:
    """Hash every public field that defined a capability when it was issued."""

    if type(authorization) is not AuthorizedRead:
        raise TypeError("authorization fingerprint requires an exact AuthorizedRead")
    if type(authorization.scope) is not RequestScope:
        raise TypeError("authorization scope must use the exact RequestScope")
    if type(authorization.compiled) is not CompiledAccess:
        raise TypeError("authorization compiler result must use the exact CompiledAccess")
    scope = authorization.scope
    compiled = authorization.compiled
    return canonical_sha256_hex(
        {
            "schema": "dithyramba.authorized_read_issuance/1.0",
            "authorization_id": authorization.authorization_id,
            "repository_instance_id": authorization.repository_instance_id,
            "access_policy_id": authorization.access_policy_id,
            "scope": {
                "library_id": scope.library_id,
                "snapshot_hash": scope.snapshot_hash,
                "purpose": scope.purpose,
                "collection_ids": list(scope.collection_ids),
                "exclusions": scope.exclusions.payload(),
                "exclusion_hash": scope.exclusion_hash,
            },
            "compiled": {
                "manifest": compiled.manifest.payload(),
                "permitted_set_hash": compiled.manifest.permitted_set_hash,
                "token": compiled.token.public_payload(),
                "public_result": compiled.public_result.payload(),
            },
        }
    )


def _snapshot_authorization_context(
    authorization: AuthorizedRead,
) -> _AuthorizedReadContext:
    """Copy validated issuance fields so later public-object changes cannot affect use."""

    scope = deepcopy(authorization.scope)
    compiled = deepcopy(authorization.compiled)
    if scope != authorization.scope or compiled != authorization.compiled:
        raise AuthorizationError("AuthorizedRead could not be snapshotted exactly")
    return _AuthorizedReadContext(
        access_policy_id=authorization.access_policy_id,
        scope=scope,
        compiled=compiled,
    )


def library_logical_identity_hash(config: LibraryConfig) -> str:
    """Return the path-independent semantic identity hash for a Library."""

    return canonical_sha256_hex(
        {
            "schema": "dithyramba.library_identity/1.0",
            "library_id": config.library_id,
        }
    )


def initialize_library(
    config: LibraryConfig,
    *,
    data_root: str | os.PathLike[str] | None = None,
    declared_source_roots: tuple[str | os.PathLike[str], ...] = (),
    declared_sync_roots: tuple[str | os.PathLike[str], ...] = (),
    clock: Clock | None = None,
    event_id_factory: EventIdFactory | None = None,
) -> LibraryRepository:
    """Create, migrate, and atomically identify one new physical Library.

    The filesystem boundary is created first, but any later failure closes the
    database and removes only the newly created Library directory. The Library
    row and its creation event always commit or roll back together.
    """

    paths = create_library_layout(
        config,
        data_root=data_root,
        declared_source_roots=declared_source_roots,
        declared_sync_roots=declared_sync_roots,
    )
    store: Store | None = None
    selected_clock = clock or _system_clock
    selected_event_factory = event_id_factory or _new_event_id
    try:
        store = Store.open_library(paths)
        created_at = _timestamp(selected_clock)
        identity_hash = library_logical_identity_hash(config)
        payload: dict[str, object] = {
            "schema": "dithyramba.library_created/1.0",
            "library_id": config.library_id,
            "name": config.name,
            "logical_identity_hash": identity_hash,
        }
        with store.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO libraries(library_id, name, logical_identity_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (config.library_id, config.name, identity_hash, created_at),
            )
            _insert_outbox_event(
                connection,
                event_id=selected_event_factory(),
                event_type="library.created",
                aggregate_type="library",
                aggregate_id=config.library_id,
                payload=payload,
                occurred_at=created_at,
            )
        record = LibraryRecord(config, identity_hash, created_at, paths)
        repository = LibraryRepository(
            paths=paths,
            store=store,
            library=record,
            clock=selected_clock,
            event_id_factory=selected_event_factory,
        )
        repository.verify()
        return repository
    except Exception:
        if store is not None:
            store.close()
        shutil.rmtree(paths.root, ignore_errors=True)
        raise


def open_library(
    library_id: str,
    *,
    data_root: str | os.PathLike[str] | None = None,
    clock: Clock | None = None,
    event_id_factory: EventIdFactory | None = None,
) -> LibraryRepository:
    """Open an existing Library without creating or registering global state."""

    paths = library_paths(validate_library_id(library_id), data_root=data_root)
    if not paths.root.exists():
        raise LibraryNotFoundError(f"Library does not exist: {paths.root}")
    validate_library_layout(paths)
    store = Store.open_library(paths, apply_migrations=False)
    try:
        _require_latest_schema(store)
        record = _load_library_record(store.connection, paths, expected_id=library_id)
        return LibraryRepository(
            paths=paths,
            store=store,
            library=record,
            clock=clock or _system_clock,
            event_id_factory=event_id_factory or _new_event_id,
        )
    except Exception:
        store.close()
        raise


def list_libraries(*, data_root: str | os.PathLike[str] | None = None) -> tuple[LibraryRecord, ...]:
    """Safely scan immediate Library directories under the application root."""

    app_root = application_data_root(data_root)
    libraries_root = app_root / "libraries"
    if not libraries_root.exists():
        return ()
    if libraries_root.is_symlink() or not libraries_root.is_dir():
        raise PersistenceIntegrityError("libraries root must be a real directory")

    records: list[LibraryRecord] = []
    for candidate in sorted(libraries_root.iterdir(), key=lambda path: path.name):
        if candidate.is_symlink():
            if candidate.name.startswith("library_"):
                raise PersistenceIntegrityError(
                    f"Library installation may not be a symlink: {candidate}"
                )
            continue
        if not candidate.is_dir():
            continue
        try:
            library_id = validate_library_id(candidate.name)
        except Exception:
            continue
        with open_library(library_id, data_root=app_root) as repository:
            records.append(repository.library)
    return tuple(records)


class LibraryRepository(AbstractContextManager["LibraryRepository"]):
    """Narrow persistence/service boundary for exactly one physical Library."""

    def __init__(
        self,
        *,
        paths: LibraryPaths,
        store: Store,
        library: LibraryRecord,
        clock: Clock,
        event_id_factory: EventIdFactory,
    ) -> None:
        self.paths = paths
        self._store = store
        self.library = library
        self._clock = clock
        self._event_id_factory = event_id_factory
        self._instance_id = new_id("repository")
        self._authorizations: dict[str, _IssuedAuthorization] = {}
        self._fragment_text_read_depth = 0
        self._vector_blob_read_depth = 0
        self._store.connection.set_authorizer(self._authorize_sql_access)

    @property
    def library_id(self) -> str:
        """Return the only logical Library ID this repository may access."""

        return self.library.config.library_id

    @property
    def schema_version(self) -> int:
        """Return the verified migration head without exposing SQLite."""

        return self._store.schema_version

    @property
    def schema_fingerprint(self) -> str:
        """Return the verified actual-schema fingerprint without exposing SQLite."""

        return schema_fingerprint(self._store.connection)

    def verify(self) -> None:
        """Verify layout, schema, integrity, and the semantic Library identity."""

        validate_library_layout(self.paths)
        self._store.verify()
        _require_latest_schema(self._store)
        current = _load_library_record(
            self._store.connection,
            self.paths,
            expected_id=self.library_id,
        )
        if current != self.library:
            raise PersistenceIntegrityError("Library identity changed after repository open")

    def describe(self, *, recent_run_limit: int = 10) -> LibraryDescription:
        """Return one verified, source-text-free overview for operators and agents."""

        if type(recent_run_limit) is not int or not 1 <= recent_run_limit <= 100:
            raise ValueError("recent_run_limit must be an integer from 1 through 100")
        self.verify()
        connection = self._store.connection
        counts = LibraryCounts(
            collections=_scoped_count(
                connection,
                "SELECT COUNT(*) FROM collections WHERE library_id = ?",
                self.library_id,
            ),
            access_policies=_scoped_count(
                connection,
                "SELECT COUNT(*) FROM access_policies WHERE library_id = ?",
                self.library_id,
            ),
            sources=_scoped_count(
                connection,
                "SELECT COUNT(*) FROM sources WHERE library_id = ?",
                self.library_id,
            ),
            source_versions=_scoped_count(
                connection,
                """
                SELECT COUNT(*) FROM source_versions AS sv
                JOIN sources AS s ON s.source_id = sv.source_id
                WHERE s.library_id = ?
                """,
                self.library_id,
            ),
            source_fragments=_scoped_count(
                connection,
                """
                SELECT COUNT(*) FROM source_fragments AS sf
                JOIN source_versions AS sv ON sv.source_version_id = sf.source_version_id
                JOIN sources AS s ON s.source_id = sv.source_id
                WHERE s.library_id = ?
                """,
                self.library_id,
            ),
            collection_memberships=_scoped_count(
                connection,
                """
                SELECT COUNT(*) FROM collection_memberships AS cm
                JOIN collections AS c ON c.collection_id = cm.collection_id
                JOIN sources AS s ON s.source_id = cm.source_id
                WHERE c.library_id = ? AND s.library_id = ?
                """,
                self.library_id,
                self.library_id,
            ),
            corpus_snapshots=_scoped_count(
                connection,
                "SELECT COUNT(*) FROM corpus_snapshots WHERE library_id = ?",
                self.library_id,
            ),
            snapshot_members=_scoped_count(
                connection,
                """
                SELECT COUNT(*) FROM snapshot_members AS sm
                JOIN corpus_snapshots AS cs
                  ON cs.corpus_snapshot_id = sm.corpus_snapshot_id
                WHERE cs.library_id = ?
                """,
                self.library_id,
            ),
            evidence_packets=_scoped_count(
                connection,
                """
                SELECT COUNT(*) FROM evidence_packets AS ep
                JOIN query_requests AS qr ON qr.query_request_id = ep.query_request_id
                WHERE qr.library_id = ?
                """,
                self.library_id,
            ),
            research_sessions=_scoped_count(
                connection,
                "SELECT COUNT(*) FROM research_sessions WHERE library_id = ?",
                self.library_id,
            ),
            processing_runs=_scoped_count(connection, "SELECT COUNT(*) FROM processing_runs"),
        )
        snapshot_rows = connection.execute(
            """
            SELECT corpus_snapshot_id, created_at
            FROM corpus_snapshots
            WHERE library_id = ?
            ORDER BY created_at DESC, corpus_snapshot_id
            """,
            (self.library_id,),
        ).fetchall()
        snapshots = []
        for row in snapshot_rows:
            snapshot = self.get_corpus_snapshot(str(row[0]))
            snapshots.append(
                CorpusSnapshotSummary(
                    corpus_snapshot_id=snapshot.corpus_snapshot_id,
                    manifest_hash=snapshot.manifest_hash,
                    collection_ids=snapshot.collection_ids,
                    member_count=len(snapshot.members),
                    created_at=_validated_timestamp(str(row[1])),
                )
            )
        run_rows = connection.execute(
            """
            SELECT processing_run_id FROM processing_runs
            ORDER BY started_at DESC, processing_run_id
            LIMIT ?
            """,
            (recent_run_limit,),
        ).fetchall()
        return LibraryDescription(
            library=self.library,
            counts=counts,
            collections=self.list_collections(),
            access_policies=self.list_access_policies(),
            snapshots=tuple(snapshots),
            recent_processing_runs=tuple(self.get_processing_run(str(row[0])) for row in run_rows),
        )

    def close(self) -> None:
        """Invalidate ephemeral read capabilities and close SQLite."""

        self._authorizations.clear()
        with suppress(Exception):
            self._store.connection.set_authorizer(None)
        self._store.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _authorize_sql_access(
        self,
        action_code: int,
        table_name: str | None,
        column_name: str | None,
        database_name: str | None,
        trigger_name: str | None,
    ) -> int:
        """Deny direct protected-content reads below their separate gates."""

        del database_name
        if (
            action_code == sqlite3.SQLITE_READ
            and table_name == "source_fragments"
            and column_name == "text"
            and trigger_name != "source_fragments_capture_text_metrics"
            and self._fragment_text_read_depth == 0
        ):
            return sqlite3.SQLITE_DENY
        if (
            action_code == sqlite3.SQLITE_READ
            and table_name in {"corpus_vector_items", "semantic_span_vector_items"}
            and column_name == "vector_blob"
            and self._vector_blob_read_depth == 0
        ):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    @contextmanager
    def _permit_fragment_text_read(self) -> Iterator[None]:
        self._fragment_text_read_depth += 1
        try:
            yield
        finally:
            self._fragment_text_read_depth -= 1

    @contextmanager
    def _permit_vector_blob_read(self) -> Iterator[None]:
        """Permit one adapter-owned vector materialization without opening text reads."""

        self._vector_blob_read_depth += 1
        try:
            yield
        finally:
            self._vector_blob_read_depth -= 1

    def create_collection(self, config: CollectionConfig) -> CollectionRecord:
        """Persist one validated logical Collection and all of its physical roots."""

        if config.library_id != self.library_id:
            raise PersistenceIntegrityError("Collection belongs to a different Library")
        validated_roots = tuple(
            build_collection_root(
                root.path,
                data_root=self.paths.application_data_root,
                include_globs=root.include_globs,
                exclude_globs=root.exclude_globs,
                identity_manifest_path=root.identity_manifest_path,
                identity_manifest_sha256=root.identity_manifest_sha256,
            )
            for root in config.roots
        )
        for root in validated_roots:
            load_source_identity_manifest(root)
        validated_config = config.model_copy(update={"roots": validated_roots})
        root_ids = tuple(new_id("root") for _ in validated_roots)
        created_at = _timestamp(self._clock)
        payload = _collection_event_payload(validated_config, root_ids)
        try:
            with self._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO collections(collection_id, library_id, name, kind, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        validated_config.collection_id,
                        self.library_id,
                        validated_config.name,
                        validated_config.kind.value,
                        created_at,
                    ),
                )
                for root_id, root in zip(root_ids, validated_roots, strict=True):
                    connection.execute(
                        """
                        INSERT INTO collection_roots(
                            collection_root_id, collection_id, resolved_path,
                            include_globs_json, exclude_globs_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            root_id,
                            validated_config.collection_id,
                            str(root.path),
                            _canonical_text(list(root.include_globs)),
                            _canonical_text(list(root.exclude_globs)),
                            created_at,
                        ),
                    )
                    if root.identity_manifest_path is not None:
                        if root.identity_manifest_sha256 is None:
                            raise PersistenceIntegrityError("identity manifest path has no pin")
                        connection.execute(
                            """
                            INSERT INTO collection_root_identity_manifests(
                                collection_root_id, manifest_path, manifest_sha256, created_at
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (
                                root_id,
                                str(root.identity_manifest_path),
                                root.identity_manifest_sha256,
                                created_at,
                            ),
                        )
                _insert_outbox_event(
                    connection,
                    event_id=self._event_id_factory(),
                    event_type="collection.created",
                    aggregate_type="collection",
                    aggregate_id=validated_config.collection_id,
                    payload=payload,
                    occurred_at=created_at,
                )
        except sqlite3.IntegrityError as exc:
            raise PersistenceConflictError(
                "Collection insert conflicted with existing state"
            ) from exc
        return self.get_collection(validated_config.collection_id)

    def get_collection(self, collection_id: str) -> CollectionRecord:
        """Load one Collection, revalidating every persisted physical root."""

        row = self._store.connection.execute(
            """
            SELECT collection_id, library_id, name, kind, created_at
            FROM collections
            WHERE collection_id = ? AND library_id = ?
            """,
            (collection_id, self.library_id),
        ).fetchone()
        if row is None:
            raise CollectionNotFoundError(
                f"Collection does not exist in Library {self.library_id}: {collection_id}"
            )
        root_rows = self._store.connection.execute(
            """
            SELECT cr.collection_root_id, cr.resolved_path,
                   cr.include_globs_json, cr.exclude_globs_json,
                   manifest.manifest_path, manifest.manifest_sha256
            FROM collection_roots AS cr
            JOIN collections AS c ON c.collection_id = cr.collection_id
            LEFT JOIN collection_root_identity_manifests AS manifest
              ON manifest.collection_root_id = cr.collection_root_id
            WHERE cr.collection_id = ? AND c.library_id = ?
            ORDER BY cr.rowid
            """,
            (collection_id, self.library_id),
        ).fetchall()
        roots: list[CollectionRoot] = []
        root_ids: list[str] = []
        for root_row in root_rows:
            include = _load_canonical_string_tuple(str(root_row[2]), "include globs")
            exclude = _load_canonical_string_tuple(str(root_row[3]), "exclude globs")
            persisted_path = Path(str(root_row[1]))
            try:
                validated_root = build_collection_root(
                    persisted_path,
                    data_root=self.paths.application_data_root,
                    include_globs=include,
                    exclude_globs=exclude,
                    identity_manifest_path=(
                        Path(str(root_row[4])) if root_row[4] is not None else None
                    ),
                    identity_manifest_sha256=(
                        str(root_row[5]) if root_row[5] is not None else None
                    ),
                )
            except CollectionError as exc:
                raise PersistenceIntegrityError(
                    "persisted Collection root no longer satisfies its boundary"
                ) from exc
            if validated_root.path != persisted_path:
                raise PersistenceIntegrityError(
                    "persisted Collection root resolves to a different path"
                )
            try:
                load_source_identity_manifest(validated_root)
            except CollectionError as exc:
                raise PersistenceIntegrityError(
                    "persisted Collection identity manifest is invalid"
                ) from exc
            roots.append(validated_root)
            root_ids.append(str(root_row[0]))
        try:
            config = CollectionConfig(
                collection_id=str(row[0]),
                library_id=str(row[1]),
                name=str(row[2]),
                kind=CollectionKind(str(row[3])),
                roots=tuple(roots),
            )
        except (TypeError, ValueError) as exc:
            raise PersistenceIntegrityError("persisted Collection is invalid") from exc
        created_at = _validated_timestamp(str(row[4]))
        return CollectionRecord(config, tuple(root_ids), created_at)

    def list_collections(self) -> tuple[CollectionRecord, ...]:
        """List Collections scoped to this Library in stable ID order."""

        rows = self._store.connection.execute(
            """
            SELECT collection_id FROM collections
            WHERE library_id = ? ORDER BY collection_id
            """,
            (self.library_id,),
        ).fetchall()
        return tuple(self.get_collection(str(row[0])) for row in rows)

    def persist_access_policy(
        self,
        *,
        name: str,
        snapshot: AccessPolicySnapshot,
    ) -> AccessPolicyRecord:
        """Insert one immutable policy and its normalized allow/deny rules."""

        policy_name = _validated_label(name, "AccessPolicy name")
        if snapshot.library_id != self.library_id:
            raise PersistenceIntegrityError("AccessPolicy belongs to a different Library")
        created_at = _timestamp(self._clock)
        payload: dict[str, object] = {
            "schema": "dithyramba.access_policy_created/1.0",
            "access_policy_id": snapshot.access_policy_id,
            "name": policy_name,
            "policy_hash": snapshot.policy_hash,
            "policy": snapshot.semantic_payload(),
        }
        try:
            with self._store.transaction(immediate=True) as connection:
                self._require_collections(
                    connection,
                    tuple(rule.collection_id for rule in snapshot.collection_rules),
                )
                self._require_sources(
                    connection,
                    tuple(rule.source_id for rule in snapshot.source_rules),
                )
                connection.execute(
                    """
                    INSERT INTO access_policies(
                        access_policy_id, library_id, name, allowed_purposes_json,
                        allow_export, allow_external_provider, policy_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.access_policy_id,
                        self.library_id,
                        policy_name,
                        _canonical_text(list(snapshot.allowed_purposes)),
                        int(snapshot.allow_export),
                        int(snapshot.allow_external_provider),
                        snapshot.policy_hash,
                        created_at,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO access_policy_collection_rules(
                        access_policy_id, collection_id, effect
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        (snapshot.access_policy_id, rule.collection_id, rule.effect.value)
                        for rule in snapshot.collection_rules
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO access_policy_source_rules(access_policy_id, source_id, effect)
                    VALUES (?, ?, ?)
                    """,
                    (
                        (snapshot.access_policy_id, rule.source_id, rule.effect.value)
                        for rule in snapshot.source_rules
                    ),
                )
                _insert_outbox_event(
                    connection,
                    event_id=self._event_id_factory(),
                    event_type="access_policy.created",
                    aggregate_type="access_policy",
                    aggregate_id=snapshot.access_policy_id,
                    payload=payload,
                    occurred_at=created_at,
                )
        except sqlite3.IntegrityError as exc:
            raise PersistenceConflictError(
                "AccessPolicy insert conflicted with existing state"
            ) from exc
        return self.get_access_policy(snapshot.access_policy_id)

    def get_access_policy(self, access_policy_id: str) -> AccessPolicyRecord:
        """Load and independently re-hash one persisted AccessPolicy."""

        row = self._store.connection.execute(
            """
            SELECT access_policy_id, library_id, name, allowed_purposes_json,
                   allow_export, allow_external_provider, policy_hash, created_at
            FROM access_policies
            WHERE access_policy_id = ? AND library_id = ?
            """,
            (access_policy_id, self.library_id),
        ).fetchone()
        if row is None:
            raise AccessPolicyNotFoundError(
                f"AccessPolicy does not exist in Library {self.library_id}: {access_policy_id}"
            )
        collection_rows = self._store.connection.execute(
            """
            SELECT r.collection_id, r.effect
            FROM access_policy_collection_rules AS r
            JOIN access_policies AS p ON p.access_policy_id = r.access_policy_id
            JOIN collections AS c ON c.collection_id = r.collection_id
            WHERE r.access_policy_id = ? AND p.library_id = ? AND c.library_id = ?
            ORDER BY r.collection_id
            """,
            (access_policy_id, self.library_id, self.library_id),
        ).fetchall()
        source_rows = self._store.connection.execute(
            """
            SELECT r.source_id, r.effect
            FROM access_policy_source_rules AS r
            JOIN access_policies AS p ON p.access_policy_id = r.access_policy_id
            JOIN sources AS s ON s.source_id = r.source_id
            WHERE r.access_policy_id = ? AND p.library_id = ? AND s.library_id = ?
            ORDER BY r.source_id
            """,
            (access_policy_id, self.library_id, self.library_id),
        ).fetchall()
        purposes = _load_canonical_string_tuple(str(row[3]), "allowed purposes")
        try:
            snapshot = AccessPolicySnapshot(
                access_policy_id=str(row[0]),
                library_id=str(row[1]),
                allowed_purposes=purposes,
                collection_rules=tuple(
                    CollectionRule(str(item[0]), PolicyEffect(str(item[1])))
                    for item in collection_rows
                ),
                source_rules=tuple(
                    SourceRule(str(item[0]), PolicyEffect(str(item[1]))) for item in source_rows
                ),
                allow_export=_strict_bool(row[4], "allow_export"),
                allow_external_provider=_strict_bool(row[5], "allow_external_provider"),
            )
        except (TypeError, ValueError) as exc:
            raise PersistenceIntegrityError("persisted AccessPolicy is invalid") from exc
        if snapshot.policy_hash != str(row[6]):
            raise PersistenceIntegrityError("persisted AccessPolicy hash mismatch")
        return AccessPolicyRecord(
            name=_validated_label(str(row[2]), "AccessPolicy name"),
            snapshot=snapshot,
            created_at=_validated_timestamp(str(row[7])),
        )

    def list_access_policies(self) -> tuple[AccessPolicyRecord, ...]:
        """List immutable policy snapshots scoped to this Library."""

        rows = self._store.connection.execute(
            """
            SELECT access_policy_id FROM access_policies
            WHERE library_id = ? ORDER BY access_policy_id
            """,
            (self.library_id,),
        ).fetchall()
        return tuple(self.get_access_policy(str(row[0])) for row in rows)

    def begin_ingest_run(
        self,
        *,
        code_version: str,
        profile_version: str,
    ) -> ProcessingRunRecord:
        """Start one auditable synchronous ingest batch."""

        run_id = new_id("run")
        started_at = _timestamp(self._clock)
        code = _validated_label(code_version, "code version")
        profile = _validated_label(profile_version, "profile version")
        try:
            with self._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO processing_runs(
                        processing_run_id, kind, query_request_id, status,
                        code_version, profile_version, started_at, finished_at,
                        error_code, output_hash
                    ) VALUES (?, 'ingest', NULL, 'running', ?, ?, ?, NULL, NULL, NULL)
                    """,
                    (run_id, code, profile, started_at),
                )
        except sqlite3.IntegrityError as exc:
            raise PersistenceConflictError("ProcessingRun insert conflicted") from exc
        return self.get_processing_run(run_id)

    def get_processing_run(self, processing_run_id: str) -> ProcessingRunRecord:
        """Load one ProcessingRun and validate its lifecycle fields."""

        row = self._store.connection.execute(
            """
            SELECT processing_run_id, kind, status, code_version, profile_version,
                   started_at, finished_at, error_code, output_hash
            FROM processing_runs WHERE processing_run_id = ?
            """,
            (processing_run_id,),
        ).fetchone()
        if row is None:
            raise ProcessingRunNotFoundError(f"ProcessingRun does not exist: {processing_run_id}")
        try:
            status = ProcessingRunStatus(str(row[2]))
        except ValueError as exc:
            raise PersistenceIntegrityError("persisted ProcessingRun status is invalid") from exc
        finished_at = None if row[6] is None else _validated_timestamp(str(row[6]))
        error_code = None if row[7] is None else _validated_label(str(row[7]), "error code")
        output_hash = None if row[8] is None else _validated_sha256(str(row[8]), "output hash")
        if status is ProcessingRunStatus.RUNNING:
            if finished_at is not None or error_code is not None or output_hash is not None:
                raise PersistenceIntegrityError("running ProcessingRun has terminal fields")
        elif finished_at is None or output_hash is None:
            raise PersistenceIntegrityError("terminal ProcessingRun lacks completion fields")
        if status is ProcessingRunStatus.FAILED and error_code is None:
            raise PersistenceIntegrityError("failed ProcessingRun lacks an error code")
        if status is ProcessingRunStatus.SUCCEEDED and error_code is not None:
            raise PersistenceIntegrityError("successful ProcessingRun has an error code")
        return ProcessingRunRecord(
            processing_run_id=str(row[0]),
            kind=_validated_label(str(row[1]), "run kind"),
            status=status,
            code_version=_validated_label(str(row[3]), "code version"),
            profile_version=_validated_label(str(row[4]), "profile version"),
            started_at=_validated_timestamp(str(row[5])),
            finished_at=finished_at,
            error_code=error_code,
            output_hash=output_hash,
        )

    def find_reusable_unchanged_source(
        self,
        *,
        collection_id: str,
        collection_root_id: str,
        source: SourceBytes,
        parser_profile: str,
        identity_declaration: SourceIdentityDeclaration | None = None,
    ) -> PersistedSourceOutcome | None:
        """Return an exact current Source without invoking its parser again.

        This is deliberately a read-only fast path.  It applies only when the
        Source is already a member of the requested Collection, its current
        bytes and parser profile match, its lineage remains valid, and the
        referenced Blob still verifies physically.  Any incomplete match falls
        back to the ordinary parse-and-persist route.
        """

        if not isinstance(source, SourceBytes):
            raise TypeError("find_reusable_unchanged_source requires SourceBytes")
        profile = _validated_label(parser_profile, "parser profile")
        root_id = _validated_label(collection_root_id, "Collection root ID")
        if identity_declaration is not None and not isinstance(
            identity_declaration, SourceIdentityDeclaration
        ):
            raise TypeError("identity_declaration must be SourceIdentityDeclaration or None")

        connection = self._store.connection
        collection_row = connection.execute(
            """
            SELECT c.kind, cr.resolved_path
            FROM collections AS c
            LEFT JOIN collection_roots AS cr
              ON cr.collection_id = c.collection_id
             AND cr.collection_root_id = ?
            WHERE c.collection_id = ? AND c.library_id = ?
            """,
            (root_id, collection_id, self.library_id),
        ).fetchone()
        if collection_row is None:
            raise CollectionNotFoundError(
                f"Collection does not exist in Library {self.library_id}: {collection_id}"
            )
        if collection_row[1] is None:
            raise PersistenceIntegrityError(
                "Collection root does not belong to the ingest Collection"
            )
        persisted_root = Path(str(collection_row[1]))
        if not persisted_root.is_absolute():
            raise PersistenceIntegrityError("persisted Collection root is not absolute")
        expected_uri = (persisted_root / Path(*PurePosixPath(source.relative_path).parts)).as_uri()
        if source.canonical_uri != expected_uri:
            raise PersistenceIntegrityError(
                "captured Source URI does not match its exact Collection root and path"
            )
        membership_state = "holdout" if str(collection_row[0]) == "holdout" else "active"

        row = connection.execute(
            """
            SELECT s.source_id, s.media_type, sh.source_version_id,
                   sv.content_sha256, sv.parser_profile, sv.parse_status,
                   membership.state
            FROM sources AS s
            JOIN source_heads AS sh ON sh.source_id = s.source_id
            JOIN source_versions AS sv
              ON sv.source_id = s.source_id
             AND sv.source_version_id = sh.source_version_id
            LEFT JOIN collection_memberships AS membership
              ON membership.source_id = s.source_id
             AND membership.collection_id = ?
            WHERE s.library_id = ? AND s.canonical_uri = ?
            """,
            (collection_id, self.library_id, source.canonical_uri),
        ).fetchone()
        if row is None:
            return None
        if str(row[1]) != source.media_type.value:
            raise PersistenceIntegrityError(
                "persisted Source media type changed for the same canonical URI"
            )
        if (
            str(row[3]) != source.content_sha256
            or str(row[4]) != profile
            or str(row[5]) != ParseStatus.PROCESSED.value
            or row[6] is None
        ):
            return None
        if str(row[6]) != membership_state:
            raise PersistenceIntegrityError(
                "existing Collection membership state differs from ingest scope"
            )

        source_id = str(row[0])
        source_version_id = str(row[2])
        if identity_declaration is None:
            source_family_id, root_source_id, family_role = self._lineage_for_source(
                connection, source_id
            )
        else:
            source_family_id, root_source_id, family_role = (
                self._declared_lineage_for_existing_source(
                    connection,
                    source_id=source_id,
                    declaration=identity_declaration,
                )
            )
            self._require_identity_binding(
                connection,
                source_id=source_id,
                source_version_id=source_version_id,
                declaration=identity_declaration,
            )

        fragment_count = int(
            connection.execute(
                """
                SELECT count(*) FROM source_fragments
                WHERE source_version_id = ?
                """,
                (source_version_id,),
            ).fetchone()[0]
        )
        if fragment_count < 1:
            raise PersistenceIntegrityError("processed SourceVersion has no SourceFragments")
        self.get_blob(source.content_sha256)
        return PersistedSourceOutcome(
            IngestDisposition.UNCHANGED,
            source_id,
            source_version_id,
            source_family_id,
            root_source_id,
            family_role,
            fragment_count,
        )

    def record_processed_source(
        self,
        *,
        collection_id: str,
        collection_root_id: str,
        source: SourceBytes,
        parsed: ParseResult,
        blob: PromotedBlob,
        identity_declaration: SourceIdentityDeclaration | None = None,
    ) -> PersistedSourceOutcome:
        """Serialize one processed input into an all-or-nothing provenance graph.

        Blob promotion occurs before this method. The promoted bytes are
        independently verified before ``BEGIN IMMEDIATE`` and every logical row
        plus its outbox event commits in one writer transaction.
        """

        if not isinstance(source, SourceBytes):
            raise TypeError("record_processed_source requires SourceBytes")
        if not isinstance(parsed, ParseResult) or parsed.status is not ParseStatus.PROCESSED:
            raise ValueError("only a processed ParseResult may create provenance rows")
        if not isinstance(blob, PromotedBlob):
            raise TypeError("record_processed_source requires a PromotedBlob")
        if identity_declaration is not None and not isinstance(
            identity_declaration, SourceIdentityDeclaration
        ):
            raise TypeError("identity_declaration must be SourceIdentityDeclaration or None")
        if blob.content_sha256 != source.content_sha256 or blob.byte_size != source.byte_size:
            raise PersistenceIntegrityError("promoted Blob differs from captured Source bytes")
        source_modified_at = _validated_timestamp(source.source_modified_at)
        parser_profile = _validated_label(parsed.parser_profile, "parser profile")
        root_id = _validated_label(collection_root_id, "Collection root ID")
        BlobStore(self.paths).verify(blob)

        observed_at = _timestamp(self._clock)
        try:
            with self._store.transaction(immediate=True) as connection:
                collection_row = connection.execute(
                    """
                    SELECT c.kind, cr.resolved_path
                    FROM collections AS c
                    LEFT JOIN collection_roots AS cr
                      ON cr.collection_id = c.collection_id
                     AND cr.collection_root_id = ?
                    WHERE c.collection_id = ? AND c.library_id = ?
                    """,
                    (root_id, collection_id, self.library_id),
                ).fetchone()
                if collection_row is None:
                    raise CollectionNotFoundError(
                        f"Collection does not exist in Library {self.library_id}: {collection_id}"
                    )
                if collection_row[1] is None:
                    raise PersistenceIntegrityError(
                        "Collection root does not belong to the ingest Collection"
                    )
                persisted_root = Path(str(collection_row[1]))
                if not persisted_root.is_absolute():
                    raise PersistenceIntegrityError("persisted Collection root is not absolute")
                expected_uri = (
                    persisted_root / Path(*PurePosixPath(source.relative_path).parts)
                ).as_uri()
                if source.canonical_uri != expected_uri:
                    raise PersistenceIntegrityError(
                        "captured Source URI does not match its exact Collection root and path"
                    )
                membership_state = "holdout" if str(collection_row[0]) == "holdout" else "active"
                self._ensure_blob_row(connection, blob, observed_at)

                source_row = connection.execute(
                    """
                    SELECT source_id, media_type FROM sources
                    WHERE library_id = ? AND canonical_uri = ?
                    """,
                    (self.library_id, source.canonical_uri),
                ).fetchone()
                created_source = source_row is None
                if created_source:
                    source_id = new_id("source")
                    connection.execute(
                        """
                        INSERT INTO sources(
                            source_id, library_id, canonical_uri, media_type, title, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            self.library_id,
                            source.canonical_uri,
                            source.media_type.value,
                            Path(source.relative_path).name,
                            observed_at,
                        ),
                    )
                    if identity_declaration is None:
                        lineage = self._lineage_for_content(
                            connection,
                            source.content_sha256,
                            excluding_source_id=None,
                        )
                    else:
                        lineage = self._declared_lineage_for_new_source(
                            connection,
                            declaration=identity_declaration,
                            content_sha256=source.content_sha256,
                        )
                    if lineage is None:
                        source_family_id = new_id("family")
                        root_source_id = source_id
                        family_role = SourceFamilyRole.ROOT
                        connection.execute(
                            """
                            INSERT INTO source_families(
                                source_family_id, library_id, label, created_at
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (source_family_id, self.library_id, source_id, observed_at),
                        )
                        if identity_declaration is not None:
                            connection.execute(
                                """
                                INSERT INTO source_family_identities(
                                    source_family_id, library_id, logical_source_uri, created_at
                                ) VALUES (?, ?, ?, ?)
                                """,
                                (
                                    source_family_id,
                                    self.library_id,
                                    identity_declaration.logical_source_uri,
                                    observed_at,
                                ),
                            )
                        root_value: str | None = None
                    else:
                        source_family_id, root_source_id = lineage
                        family_role = (
                            SourceFamilyRole.DUPLICATE
                            if identity_declaration is None
                            or self._declared_content_exists(
                                connection,
                                source_family_id=source_family_id,
                                content_sha256=source.content_sha256,
                            )
                            else SourceFamilyRole.DERIVATIVE
                        )
                        root_value = root_source_id
                    connection.execute(
                        """
                        INSERT INTO source_family_members(
                            source_family_id, source_id, role, root_source_id, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            source_family_id,
                            source_id,
                            family_role.value,
                            root_value,
                            observed_at,
                        ),
                    )
                    disposition = IngestDisposition.ADDED
                    version_number = 1
                else:
                    source_id = str(source_row[0])
                    if str(source_row[1]) != source.media_type.value:
                        raise PersistenceIntegrityError(
                            "persisted Source media type changed for the same canonical URI"
                        )
                    if identity_declaration is None:
                        source_family_id, root_source_id, family_role = self._lineage_for_source(
                            connection,
                            source_id,
                        )
                    else:
                        source_family_id, root_source_id, family_role = (
                            self._declared_lineage_for_existing_source(
                                connection,
                                source_id=source_id,
                                declaration=identity_declaration,
                            )
                        )
                    version_rows = connection.execute(
                        """
                        SELECT source_version_id, version_number, content_sha256,
                               parser_profile
                        FROM source_versions
                        WHERE source_id = ? ORDER BY version_number
                        """,
                        (source_id,),
                    ).fetchall()
                    if not version_rows:
                        raise PersistenceIntegrityError("persisted Source has no SourceVersion")
                    head = connection.execute(
                        """
                        SELECT sh.source_version_id, sv.content_sha256,
                               sv.parser_profile
                        FROM source_heads AS sh
                        JOIN source_versions AS sv
                          ON sv.source_version_id = sh.source_version_id
                        WHERE sh.source_id = ?
                        """,
                        (source_id,),
                    ).fetchone()
                    if head is None:
                        raise PersistenceIntegrityError("persisted Source has no source_heads row")
                    if str(head[1]) == source.content_sha256 and str(head[2]) == parser_profile:
                        if identity_declaration is not None:
                            self._require_identity_binding(
                                connection,
                                source_id=source_id,
                                source_version_id=str(head[0]),
                                declaration=identity_declaration,
                            )
                        self._ensure_membership(
                            connection,
                            collection_id=collection_id,
                            source_id=source_id,
                            state=membership_state,
                            created_at=observed_at,
                        )
                        fragment_count = int(
                            connection.execute(
                                """
                                SELECT count(*) FROM source_fragments
                                WHERE source_version_id = ?
                                """,
                                (str(head[0]),),
                            ).fetchone()[0]
                        )
                        return PersistedSourceOutcome(
                            IngestDisposition.UNCHANGED,
                            source_id,
                            str(head[0]),
                            source_family_id,
                            root_source_id,
                            family_role,
                            fragment_count,
                        )
                    historical = next(
                        (
                            row
                            for row in version_rows
                            if str(row[2]) == source.content_sha256
                            and str(row[3]) == parser_profile
                        ),
                        None,
                    )
                    if historical is not None:
                        historical_version_id = str(historical[0])
                        if identity_declaration is not None:
                            self._require_identity_binding(
                                connection,
                                source_id=source_id,
                                source_version_id=historical_version_id,
                                declaration=identity_declaration,
                            )
                        connection.execute(
                            """
                            UPDATE source_heads
                            SET source_version_id = ?, observed_at = ?, source_modified_at = ?
                            WHERE source_id = ?
                            """,
                            (
                                historical_version_id,
                                observed_at,
                                source_modified_at,
                                source_id,
                            ),
                        )
                        self._ensure_membership(
                            connection,
                            collection_id=collection_id,
                            source_id=source_id,
                            state=membership_state,
                            created_at=observed_at,
                        )
                        fragment_count = int(
                            connection.execute(
                                """
                                SELECT count(*) FROM source_fragments
                                WHERE source_version_id = ?
                                """,
                                (historical_version_id,),
                            ).fetchone()[0]
                        )
                        _insert_outbox_event(
                            connection,
                            event_id=self._event_id_factory(),
                            event_type="source_head.reverted",
                            aggregate_type="source",
                            aggregate_id=source_id,
                            payload={
                                "schema": "dithyramba.source_head_reverted/1.0",
                                "library_id": self.library_id,
                                "collection_id": collection_id,
                                "collection_root_id": root_id,
                                "source_id": source_id,
                                "previous_source_version_id": str(head[0]),
                                "source_version_id": historical_version_id,
                                "content_sha256": source.content_sha256,
                                "disposition": IngestDisposition.REVERTED.value,
                            },
                            occurred_at=observed_at,
                        )
                        return PersistedSourceOutcome(
                            IngestDisposition.REVERTED,
                            source_id,
                            historical_version_id,
                            source_family_id,
                            root_source_id,
                            family_role,
                            fragment_count,
                        )
                    if identity_declaration is None:
                        matching_lineage = self._lineage_for_content(
                            connection,
                            source.content_sha256,
                            excluding_source_id=source_id,
                        )
                        if matching_lineage is not None and matching_lineage != (
                            source_family_id,
                            root_source_id,
                        ):
                            raise SourceLineageIntegrityError(
                                "changed Source bytes collide with an incompatible SourceFamily"
                            )
                    disposition = IngestDisposition.CHANGED
                    version_number = max(int(row[1]) for row in version_rows) + 1

                source_version_id = new_id("source_version")
                connection.execute(
                    """
                    INSERT INTO source_versions(
                        source_version_id, source_id, version_number, content_sha256,
                        byte_size, observed_at, source_modified_at, parser_profile,
                        parse_status, failure_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'processed', NULL)
                    """,
                    (
                        source_version_id,
                        source_id,
                        version_number,
                        source.content_sha256,
                        source.byte_size,
                        observed_at,
                        source_modified_at,
                        parser_profile,
                    ),
                )
                fragment_ids: list[str] = []
                for fragment in parsed.fragments:
                    fragment_id = canonical_content_id(
                        "fragment",
                        {
                            "schema": "dithyramba.source_fragment_identity/1.0",
                            "source_version_id": source_version_id,
                            "ordinal": fragment.ordinal,
                            "fragment_kind": fragment.kind.value,
                            "text_sha256": fragment.text_sha256,
                            "address_hash": fragment.address_hash,
                        },
                    )
                    fragment_ids.append(fragment_id)
                    connection.execute(
                        """
                        INSERT INTO source_fragments(
                            source_fragment_id, source_version_id, ordinal, fragment_kind,
                            text, text_sha256, source_address_json, address_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fragment_id,
                            source_version_id,
                            fragment.ordinal,
                            fragment.kind.value,
                            fragment.text,
                            fragment.text_sha256,
                            fragment.address_json,
                            fragment.address_hash,
                        ),
                    )
                if identity_declaration is not None:
                    binding_id = self._insert_identity_binding(
                        connection,
                        source_id=source_id,
                        source_version_id=source_version_id,
                        source_family_id=source_family_id,
                        declaration=identity_declaration,
                        content_sha256=source.content_sha256,
                        created_at=observed_at,
                    )
                    _insert_outbox_event(
                        connection,
                        event_id=self._event_id_factory(),
                        event_type="source_identity.binding_recorded",
                        aggregate_type="source_identity_binding",
                        aggregate_id=binding_id,
                        payload={
                            "schema": "dithyramba.source_identity_binding/1.0",
                            "library_id": self.library_id,
                            "source_identity_binding_id": binding_id,
                            "source_id": source_id,
                            "source_version_id": source_version_id,
                            "source_family_id": source_family_id,
                            "logical_source_uri": identity_declaration.logical_source_uri,
                            "connector": identity_declaration.connector,
                            "connector_revision": identity_declaration.connector_revision,
                            "representation": identity_declaration.representation,
                            "part_number": identity_declaration.part_number,
                            "part_metadata_hash": identity_declaration.part_metadata_hash,
                            "content_sha256": source.content_sha256,
                            "manifest_sha256": identity_declaration.manifest_sha256,
                        },
                        occurred_at=observed_at,
                    )
                if created_source:
                    connection.execute(
                        """
                        INSERT INTO source_heads(
                            source_id, source_version_id, observed_at, source_modified_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            source_version_id,
                            observed_at,
                            source_modified_at,
                        ),
                    )
                else:
                    cursor = connection.execute(
                        """
                        UPDATE source_heads
                        SET source_version_id = ?, observed_at = ?, source_modified_at = ?
                        WHERE source_id = ?
                        """,
                        (
                            source_version_id,
                            observed_at,
                            source_modified_at,
                            source_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise PersistenceIntegrityError("Source head changed during ingest")
                self._ensure_membership(
                    connection,
                    collection_id=collection_id,
                    source_id=source_id,
                    state=membership_state,
                    created_at=observed_at,
                )
                event_payload: dict[str, object] = {
                    "schema": "dithyramba.source_version_recorded/1.0",
                    "library_id": self.library_id,
                    "collection_id": collection_id,
                    "collection_root_id": root_id,
                    "source_id": source_id,
                    "source_version_id": source_version_id,
                    "source_family_id": source_family_id,
                    "root_source_id": root_source_id,
                    "family_role": family_role.value,
                    "version_number": version_number,
                    "content_sha256": source.content_sha256,
                    "byte_size": source.byte_size,
                    "parser_profile": parser_profile,
                    "parser_revision": parsed.parser_revision,
                    "disposition": disposition.value,
                    "source_fragment_ids": fragment_ids,
                }
                _insert_outbox_event(
                    connection,
                    event_id=self._event_id_factory(),
                    event_type="source_version.recorded",
                    aggregate_type="source_version",
                    aggregate_id=source_version_id,
                    payload=event_payload,
                    occurred_at=observed_at,
                )
        except sqlite3.IntegrityError as exc:
            raise PersistenceConflictError(
                "processed Source transaction conflicted with existing state"
            ) from exc
        return PersistedSourceOutcome(
            disposition,
            source_id,
            source_version_id,
            source_family_id,
            root_source_id,
            family_role,
            len(parsed.fragments),
        )

    def complete_ingest_run(
        self,
        *,
        coverage: CoverageDraft,
        status: ProcessingRunStatus,
        error_code: str | None = None,
    ) -> tuple[ProcessingRunRecord, CoverageReportRecord]:
        """Atomically persist CoverageReport and finish its ingest run."""

        if not isinstance(coverage, CoverageDraft):
            raise TypeError("complete_ingest_run requires CoverageDraft")
        if status not in (ProcessingRunStatus.SUCCEEDED, ProcessingRunStatus.FAILED):
            raise ValueError("ingest completion status must be succeeded or failed")
        if status is ProcessingRunStatus.FAILED:
            if error_code is None:
                raise ValueError("failed ingest run requires error_code")
            terminal_error = _validated_label(error_code, "error code")
        elif error_code is not None:
            raise ValueError("successful ingest run cannot have error_code")
        else:
            terminal_error = None
        payload = _load_canonical_object(coverage.report_json, "CoverageReport payload")
        rebuilt, outcomes = _canonical_ingest_coverage_from_payload(payload)
        if rebuilt.report_hash != coverage.report_hash:
            raise PersistenceIntegrityError("CoverageReport draft hash mismatch")
        if rebuilt.coverage_report_id != coverage.coverage_report_id:
            raise PersistenceIntegrityError("CoverageReport draft ID mismatch")
        if tuple(item.omission_id for item in coverage.omissions) != tuple(
            item.omission_id for item in rebuilt.omissions
        ):
            raise PersistenceIntegrityError("CoverageReport omission ID mismatch")
        if coverage != rebuilt:
            raise PersistenceIntegrityError(
                "CoverageReport draft differs from its typed canonical reconstruction"
            )
        _validate_ingest_run_outcomes(status, terminal_error, outcomes)
        finished_at = _timestamp(self._clock)
        try:
            with self._store.transaction(immediate=True) as connection:
                run_row = connection.execute(
                    """
                    SELECT status FROM processing_runs
                    WHERE processing_run_id = ? AND kind = 'ingest'
                    """,
                    (coverage.processing_run_id,),
                ).fetchone()
                if run_row is None:
                    raise ProcessingRunNotFoundError(
                        f"ProcessingRun does not exist: {coverage.processing_run_id}"
                    )
                if str(run_row[0]) != ProcessingRunStatus.RUNNING.value:
                    raise PersistenceConflictError("ProcessingRun is already terminal")
                connection.execute(
                    """
                    INSERT INTO coverage_reports(
                        coverage_report_id, processing_run_id, stage,
                        processed_count, skipped_count, failed_count,
                        policy_omission_present, report_json, report_hash
                    ) VALUES (?, ?, 'ingest', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        coverage.coverage_report_id,
                        coverage.processing_run_id,
                        coverage.processed_count,
                        coverage.skipped_count,
                        coverage.failed_count,
                        int(coverage.policy_omission_present),
                        coverage.report_json,
                        coverage.report_hash,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO omissions(
                        omission_id, coverage_report_id, category,
                        disclosure, count, reason_code
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            omission.omission_id,
                            coverage.coverage_report_id,
                            omission.category,
                            omission.disclosure,
                            omission.count,
                            omission.reason_code,
                        )
                        for omission in coverage.omissions
                    ),
                )
                cursor = connection.execute(
                    """
                    UPDATE processing_runs
                    SET status = ?, finished_at = ?, error_code = ?, output_hash = ?
                    WHERE processing_run_id = ? AND status = 'running'
                    """,
                    (
                        status.value,
                        finished_at,
                        terminal_error,
                        coverage.report_hash,
                        coverage.processing_run_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise PersistenceConflictError("ProcessingRun changed during completion")
                event_payload: dict[str, object] = {
                    "schema": "dithyramba.ingest_run_completed/1.0",
                    "processing_run_id": coverage.processing_run_id,
                    "coverage_report_id": coverage.coverage_report_id,
                    "coverage_report_hash": coverage.report_hash,
                    "status": status.value,
                    "error_code": terminal_error,
                }
                _insert_outbox_event(
                    connection,
                    event_id=self._event_id_factory(),
                    event_type="ingest_run.completed",
                    aggregate_type="processing_run",
                    aggregate_id=coverage.processing_run_id,
                    payload=event_payload,
                    occurred_at=finished_at,
                )
        except sqlite3.IntegrityError as exc:
            raise PersistenceConflictError("ingest completion conflicted") from exc
        return (
            self.get_processing_run(coverage.processing_run_id),
            self.get_coverage_report(coverage.coverage_report_id),
        )

    def get_coverage_report(self, coverage_report_id: str) -> CoverageReportRecord:
        """Load and independently verify one canonical CoverageReport."""

        row = self._store.connection.execute(
            """
            SELECT coverage_report_id, processing_run_id, stage,
                   processed_count, skipped_count, failed_count,
                   policy_omission_present, report_json, report_hash
            FROM coverage_reports WHERE coverage_report_id = ?
            """,
            (coverage_report_id,),
        ).fetchone()
        if row is None:
            raise CoverageReportNotFoundError(
                f"CoverageReport does not exist: {coverage_report_id}"
            )
        payload = _load_canonical_object(str(row[7]), "CoverageReport payload")
        _rebuilt, typed_outcomes = _canonical_ingest_coverage_from_payload(payload)
        report_hash = canonical_sha256_hex(payload)
        if report_hash != str(row[8]):
            raise PersistenceIntegrityError("persisted CoverageReport hash mismatch")
        if canonical_content_id("coverage", payload) != str(row[0]):
            raise PersistenceIntegrityError("persisted CoverageReport ID mismatch")
        if payload.get("schema") != "dithyramba.coverage_report/1.0":
            raise PersistenceIntegrityError("persisted CoverageReport schema is invalid")
        policy_present = _strict_bool(row[6], "policy_omission_present")
        expected_scalars: dict[str, object] = {
            "stage": str(row[2]),
            "processing_run_id": str(row[1]),
            "processed_count": int(row[3]),
            "skipped_count": int(row[4]),
            "failed_count": int(row[5]),
        }
        if any(payload.get(key) != value for key, value in expected_scalars.items()):
            raise PersistenceIntegrityError("CoverageReport JSON differs from relational fields")
        policy = payload.get("policy_omission")
        if type(policy) is not dict or policy != {"count": None, "present": policy_present}:
            raise PersistenceIntegrityError("CoverageReport policy omission projection mismatch")
        omission_rows = self._store.connection.execute(
            """
            SELECT omission_id, coverage_report_id, category,
                   disclosure, count, reason_code
            FROM omissions WHERE coverage_report_id = ?
            ORDER BY category, reason_code, omission_id
            """,
            (coverage_report_id,),
        ).fetchall()
        omissions = tuple(
            OmissionRecord(
                omission_id=str(item[0]),
                coverage_report_id=str(item[1]),
                category=str(item[2]),
                disclosure=str(item[3]),
                count=None if item[4] is None else int(item[4]),
                reason_code=str(item[5]),
            )
            for item in omission_rows
        )
        omission_payloads = payload.get("omissions")
        expected_omission_payloads = [
            {
                "category": omission.category,
                "disclosure": omission.disclosure,
                "count": omission.count,
                "reason_code": omission.reason_code,
            }
            for omission in omissions
        ]
        if omission_payloads != expected_omission_payloads:
            raise PersistenceIntegrityError("CoverageReport omissions projection mismatch")
        for omission, omission_payload in zip(
            omissions,
            expected_omission_payloads,
            strict=True,
        ):
            expected_id = canonical_content_id(
                "omission",
                {
                    "schema": "dithyramba.omission/1.0",
                    "coverage_report_id": str(row[0]),
                    **omission_payload,
                },
            )
            if omission.omission_id != expected_id:
                raise PersistenceIntegrityError("persisted Omission ID mismatch")
        raw_outcomes = payload.get("outcomes")
        if type(raw_outcomes) is not list or any(type(item) is not dict for item in raw_outcomes):
            raise PersistenceIntegrityError("CoverageReport outcomes must be a JSON object array")
        outcome_counts = {
            status: sum(item.get("terminal_outcome") == status for item in raw_outcomes)
            for status in ("processed", "skipped", "failed")
        }
        if outcome_counts != {
            "processed": int(row[3]),
            "skipped": int(row[4]),
            "failed": int(row[5]),
        }:
            raise PersistenceIntegrityError("CoverageReport outcome counts mismatch")
        run = self.get_processing_run(str(row[1]))
        if run.kind != "ingest" or run.output_hash != report_hash:
            raise PersistenceIntegrityError(
                "CoverageReport does not match its terminal ingest ProcessingRun"
            )
        _validate_ingest_run_outcomes(run.status, run.error_code, typed_outcomes)
        return CoverageReportRecord(
            coverage_report_id=str(row[0]),
            processing_run_id=str(row[1]),
            stage=str(row[2]),
            processed_count=int(row[3]),
            skipped_count=int(row[4]),
            failed_count=int(row[5]),
            policy_omission_present=policy_present,
            report_json=str(row[7]),
            report_hash=report_hash,
            omissions=omissions,
        )

    def get_blob(self, content_sha256: str) -> BlobRecord:
        """Load and physically verify one referenced Blob."""

        digest = _validated_sha256(content_sha256, "Blob content hash")
        row = self._store.connection.execute(
            """
            SELECT content_sha256, byte_size, relative_path, created_at
            FROM blobs WHERE content_sha256 = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            raise SourceNotFoundError(f"Blob does not exist: {digest}")
        promoted = PromotedBlob(
            content_sha256=str(row[0]),
            byte_size=int(row[1]),
            relative_path=str(row[2]),
            path=self.paths.blobs / str(row[2]),
            created=False,
        )
        BlobStore(self.paths).verify(promoted)
        return BlobRecord(
            promoted.content_sha256,
            promoted.byte_size,
            promoted.relative_path,
            _validated_timestamp(str(row[3])),
            promoted.path,
        )

    def get_source(self, source_id: str) -> SourceRecord:
        """Load one Source with exactly one verified root-source lineage."""

        row = self._store.connection.execute(
            """
            SELECT source_id, library_id, canonical_uri, media_type, title, created_at
            FROM sources WHERE source_id = ? AND library_id = ?
            """,
            (source_id, self.library_id),
        ).fetchone()
        if row is None:
            raise SourceNotFoundError(
                f"Source does not exist in Library {self.library_id}: {source_id}"
            )
        source_family_id, root_source_id, family_role = self._lineage_for_source(
            self._store.connection,
            source_id,
        )
        membership_rows = self._store.connection.execute(
            """
            SELECT cm.collection_id, cm.state, cm.created_at
            FROM collection_memberships AS cm
            JOIN collections AS c ON c.collection_id = cm.collection_id
            WHERE cm.source_id = ? AND c.library_id = ?
            ORDER BY cm.collection_id
            """,
            (source_id, self.library_id),
        ).fetchall()
        memberships = tuple(
            CollectionMembershipRecord(
                collection_id=str(item[0]),
                state=str(item[1]),
                created_at=_validated_timestamp(str(item[2])),
            )
            for item in membership_rows
        )
        canonical_uri = str(row[2])
        if not canonical_uri.startswith("file://"):
            raise PersistenceIntegrityError("persisted Source URI is not a file URI")
        media_type = str(row[3])
        if media_type not in {"text/markdown", "text/plain", "application/pdf"}:
            raise PersistenceIntegrityError("persisted Source media type is invalid")
        head = self._store.connection.execute(
            """
            SELECT source_version_id FROM source_heads WHERE source_id = ?
            """,
            (source_id,),
        ).fetchone()
        if head is None:
            raise PersistenceIntegrityError("persisted Source has no current SourceVersion")
        return SourceRecord(
            source_id=str(row[0]),
            library_id=str(row[1]),
            canonical_uri=canonical_uri,
            media_type=media_type,
            title=None if row[4] is None else str(row[4]),
            source_family_id=source_family_id,
            family_role=family_role,
            root_source_id=root_source_id,
            current_source_version_id=str(head[0]),
            created_at=_validated_timestamp(str(row[5])),
            memberships=memberships,
        )

    def list_sources(self, *, collection_id: str | None = None) -> tuple[SourceRecord, ...]:
        """List Sources in stable ID order, optionally scoped to one Collection."""

        if collection_id is None:
            rows = self._store.connection.execute(
                """
                SELECT source_id FROM sources
                WHERE library_id = ? ORDER BY source_id
                """,
                (self.library_id,),
            ).fetchall()
        else:
            self._require_collections(self._store.connection, (collection_id,))
            rows = self._store.connection.execute(
                """
                SELECT s.source_id
                FROM sources AS s
                JOIN collection_memberships AS cm ON cm.source_id = s.source_id
                WHERE s.library_id = ? AND cm.collection_id = ?
                ORDER BY s.source_id
                """,
                (self.library_id, collection_id),
            ).fetchall()
        return tuple(self.get_source(str(row[0])) for row in rows)

    def get_source_version(self, source_version_id: str) -> SourceVersionRecord:
        """Load one immutable SourceVersion without materializing fragment text."""

        row = self._store.connection.execute(
            """
            SELECT sv.source_version_id, sv.source_id, sv.version_number,
                   sv.content_sha256, sv.byte_size, sv.observed_at,
                   sv.source_modified_at, sv.parser_profile, sv.parse_status,
                   sv.failure_code,
                   CASE WHEN sh.source_version_id IS NULL THEN 0 ELSE 1 END
            FROM source_versions AS sv
            JOIN sources AS s ON s.source_id = sv.source_id
            LEFT JOIN source_heads AS sh
              ON sh.source_id = sv.source_id
             AND sh.source_version_id = sv.source_version_id
            WHERE sv.source_version_id = ? AND s.library_id = ?
            """,
            (source_version_id, self.library_id),
        ).fetchone()
        if row is None:
            raise SourceVersionNotFoundError(
                f"SourceVersion does not exist in Library {self.library_id}: {source_version_id}"
            )
        parse_status = str(row[8])
        if parse_status not in {"processed", "skipped", "failed"}:
            raise PersistenceIntegrityError("persisted SourceVersion parse status is invalid")
        failure_code = None if row[9] is None else str(row[9])
        if (parse_status == "processed") != (failure_code is None):
            raise PersistenceIntegrityError("SourceVersion failure projection is invalid")
        return SourceVersionRecord(
            source_version_id=str(row[0]),
            source_id=str(row[1]),
            version_number=int(row[2]),
            content_sha256=_validated_sha256(str(row[3]), "SourceVersion content hash"),
            byte_size=int(row[4]),
            observed_at=_validated_timestamp(str(row[5])),
            source_modified_at=(None if row[6] is None else _validated_timestamp(str(row[6]))),
            parser_profile=_validated_label(str(row[7]), "parser profile"),
            parse_status=parse_status,
            failure_code=failure_code,
            is_current=_strict_bool(row[10], "SourceVersion current flag"),
        )

    def list_source_versions(self, source_id: str) -> tuple[SourceVersionRecord, ...]:
        """List all immutable versions of one Source in version order."""

        self.get_source(source_id)
        rows = self._store.connection.execute(
            """
            SELECT source_version_id FROM source_versions
            WHERE source_id = ? ORDER BY version_number
            """,
            (source_id,),
        ).fetchall()
        return tuple(self.get_source_version(str(row[0])) for row in rows)

    def list_source_fragments(
        self,
        source_version_id: str,
    ) -> tuple[SourceFragmentRecord, ...]:
        """List exact fragment metadata while preserving the P1 text-read gate."""

        self.get_source_version(source_version_id)
        rows = self._store.connection.execute(
            """
            SELECT source_fragment_id, source_version_id, ordinal,
                   fragment_kind, text_sha256, source_address_json, address_hash
            FROM source_fragments
            WHERE source_version_id = ? ORDER BY ordinal
            """,
            (source_version_id,),
        ).fetchall()
        records: list[SourceFragmentRecord] = []
        for row in rows:
            address_json = str(row[5])
            address = _load_canonical_object(address_json, "SourceAddress")
            address_hash = _validated_sha256(str(row[6]), "SourceAddress hash")
            if canonical_sha256_hex(address) != address_hash:
                raise PersistenceIntegrityError("persisted SourceAddress hash mismatch")
            records.append(
                SourceFragmentRecord(
                    source_fragment_id=str(row[0]),
                    source_version_id=str(row[1]),
                    ordinal=int(row[2]),
                    fragment_kind=str(row[3]),
                    text_sha256=_validated_sha256(str(row[4]), "fragment text hash"),
                    source_address_json=address_json,
                    address_hash=address_hash,
                )
            )
        if tuple(record.ordinal for record in records) != tuple(range(len(records))):
            raise PersistenceIntegrityError("SourceFragment ordinals are not contiguous")
        return tuple(records)

    def get_source_fragment(self, source_fragment_id: str) -> SourceFragmentRecord:
        """Load exact fragment metadata without materializing its source version.

        Fragment text remains behind the authorized read gate. This method is
        intentionally metadata-only and keeps packet/view projections O(1) per
        selected fragment instead of scanning every fragment in a large source.
        """

        row = self._store.connection.execute(
            """
            SELECT sf.source_fragment_id, sf.source_version_id, sf.ordinal,
                   sf.fragment_kind, sf.text_sha256, sf.source_address_json,
                   sf.address_hash
            FROM source_fragments AS sf
            JOIN source_versions AS sv
              ON sv.source_version_id = sf.source_version_id
            JOIN sources AS s ON s.source_id = sv.source_id
            WHERE sf.source_fragment_id = ? AND s.library_id = ?
            """,
            (source_fragment_id, self.library_id),
        ).fetchone()
        if row is None:
            raise SourceNotFoundError(
                f"SourceFragment does not exist in Library {self.library_id}: {source_fragment_id}"
            )
        address_json = str(row[5])
        address = _load_canonical_object(address_json, "SourceAddress")
        address_hash = _validated_sha256(str(row[6]), "SourceAddress hash")
        if canonical_sha256_hex(address) != address_hash:
            raise PersistenceIntegrityError("persisted SourceAddress hash mismatch")
        return SourceFragmentRecord(
            source_fragment_id=str(row[0]),
            source_version_id=str(row[1]),
            ordinal=int(row[2]),
            fragment_kind=str(row[3]),
            text_sha256=_validated_sha256(str(row[4]), "fragment text hash"),
            source_address_json=address_json,
            address_hash=address_hash,
        )

    def freeze_snapshot(self, collection_ids: Iterable[str]) -> CorpusSnapshot:
        """Freeze the exact current heads for a validated Collection scope.

        Empty Collections remain explicit in ``scope_json`` even though they
        contribute no ``snapshot_members`` rows. The semantic manifest is
        content-addressed; timestamps, event IDs, and run IDs are deliberately
        outside its hash domain.
        """

        try:
            scope = build_corpus_snapshot(
                library_id=self.library_id,
                collection_ids=collection_ids,
            )
        except (TypeError, ValueError) as exc:
            raise PersistenceIntegrityError("CorpusSnapshot scope is invalid") from exc

        try:
            with self._store.transaction(immediate=True) as connection:
                self._require_collections(connection, scope.collection_ids)
                members = self._current_snapshot_members(connection, scope.collection_ids)
                snapshot = build_corpus_snapshot(
                    library_id=self.library_id,
                    collection_ids=scope.collection_ids,
                    members=members,
                )
                existing_rows = connection.execute(
                    """
                    SELECT corpus_snapshot_id
                    FROM corpus_snapshots
                    WHERE corpus_snapshot_id = ? OR manifest_hash = ?
                    ORDER BY corpus_snapshot_id
                    """,
                    (snapshot.corpus_snapshot_id, snapshot.manifest_hash),
                ).fetchall()
                if existing_rows:
                    if len(existing_rows) != 1:
                        raise PersistenceIntegrityError(
                            "CorpusSnapshot ID/hash resolve to conflicting persisted rows"
                        )
                    existing = self._reconstruct_corpus_snapshot(
                        connection,
                        str(existing_rows[0][0]),
                    )
                    if existing.canonical_bytes != snapshot.canonical_bytes:
                        raise PersistenceIntegrityError(
                            "persisted CorpusSnapshot conflicts with its content address"
                        )
                    return existing

                created_at = _timestamp(self._clock)
                scope_json = _canonical_text(snapshot.semantic_payload()["scope"])
                connection.execute(
                    """
                    INSERT INTO corpus_snapshots(
                        corpus_snapshot_id, library_id, scope_json,
                        manifest_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.corpus_snapshot_id,
                        self.library_id,
                        scope_json,
                        snapshot.manifest_hash,
                        created_at,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO snapshot_members(
                        corpus_snapshot_id, collection_id,
                        source_version_id, membership_state
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        (
                            snapshot.corpus_snapshot_id,
                            member.collection_id,
                            member.source_version_id,
                            member.membership_state.value,
                        )
                        for member in snapshot.members
                    ),
                )
                _insert_outbox_event(
                    connection,
                    event_id=self._event_id_factory(),
                    event_type="corpus_snapshot.frozen",
                    aggregate_type="corpus_snapshot",
                    aggregate_id=snapshot.corpus_snapshot_id,
                    payload={
                        "schema": "dithyramba.corpus_snapshot_frozen/1.0",
                        "library_id": self.library_id,
                        "corpus_snapshot_id": snapshot.corpus_snapshot_id,
                        "manifest_hash": snapshot.manifest_hash,
                        "collection_ids": list(snapshot.collection_ids),
                        "member_count": len(snapshot.members),
                    },
                    occurred_at=created_at,
                )
        except sqlite3.IntegrityError as exc:
            raise PersistenceConflictError(
                "CorpusSnapshot transaction conflicted with existing state"
            ) from exc
        return self.get_corpus_snapshot(snapshot.corpus_snapshot_id)

    def get_corpus_snapshot(self, corpus_snapshot_id: str) -> CorpusSnapshot:
        """Load and independently rehash one complete snapshot manifest."""

        return self._reconstruct_corpus_snapshot(
            self._store.connection,
            corpus_snapshot_id,
        )

    def authorize_read(
        self,
        *,
        access_policy_id: str,
        scope: RequestScope,
    ) -> AuthorizedRead:
        """Compile a DB-derived permitted set and issue an ephemeral capability."""

        with self._store.transaction() as _connection:
            compiled = self._compile_db_access(access_policy_id=access_policy_id, scope=scope)
        authorization = AuthorizedRead(
            authorization_id=new_id("authorization"),
            repository_instance_id=self._instance_id,
            access_policy_id=access_policy_id,
            scope=scope,
            compiled=compiled,
        )
        self._authorizations[authorization.authorization_id] = _IssuedAuthorization(
            capability=authorization,
            context=_snapshot_authorization_context(authorization),
            fingerprint=_authorization_fingerprint(authorization),
        )
        return authorization

    def read_permitted_fragments(
        self,
        authorization: AuthorizedRead,
    ) -> tuple[SourceFragmentText, ...]:
        """Read exactly one DB-derived manifest, never a caller-forged token set."""

        if not isinstance(authorization, AuthorizedRead):
            raise AuthorizationError("fragment reads require an AuthorizedRead")
        return self.read_permitted_fragment_subset(
            authorization,
            tuple(item.source_fragment_id for item in authorization.compiled.manifest.items),
        )

    def revalidate_authorized_read(
        self,
        authorization: AuthorizedRead,
    ) -> CompiledAccess:
        """Recompile one exact repository-issued capability without reading content.

        The method is safe both as a standalone metadata check and from inside
        an adapter-owned transaction.  Object identity is intentional: copied
        dataclass values and capabilities issued by another repository instance
        are rejected even when all their public fields happen to match.
        """

        connection = self._store.connection
        if connection.in_transaction:
            return self._revalidate_authorized_read(authorization)
        with self._store.transaction() as _connection:
            return self._revalidate_authorized_read(authorization)

    def read_permitted_fragment_subset(
        self,
        authorization: AuthorizedRead,
        source_fragment_ids: tuple[str, ...],
    ) -> tuple[SourceFragmentText, ...]:
        """Read an ordered subset of one repository-issued permitted manifest.

        This is the exact-content boundary used by StructureUnit rendering. It
        prevents a broader policy scope from becoming a claim that every
        permitted fragment was actually read.
        """

        if type(authorization) is not AuthorizedRead:
            raise AuthorizationError("fragment reads require an AuthorizedRead")
        if (
            type(source_fragment_ids) is not tuple
            or any(
                type(item) is not str or not item.startswith("fragment_")
                for item in source_fragment_ids
            )
            or len(set(source_fragment_ids)) != len(source_fragment_ids)
        ):
            raise AuthorizationError("fragment subset must be a unique tuple of SourceFragment IDs")
        with self._store.transaction() as connection:
            current = self.revalidate_authorized_read(authorization)
            manifest = current.manifest
            if not source_fragment_ids:
                return ()

            expected = {item.source_fragment_id: item for item in manifest.items}
            if not set(source_fragment_ids).issubset(expected):
                raise AuthorizationError(
                    "fragment subset contains an ID outside the permitted manifest"
                )
            if len(source_fragment_ids) > _LARGE_FRAGMENT_READ_THRESHOLD:
                rows = self._read_large_permitted_fragment_subset(
                    connection,
                    source_fragment_ids,
                    expected,
                )
            else:
                rows = []
                with self._permit_fragment_text_read():
                    for chunk in _chunks(source_fragment_ids, 500):
                        placeholders = ", ".join("?" for _ in chunk)
                        query = f"""
                            SELECT sf.source_fragment_id, sf.source_version_id, sv.source_id,
                                   sf.ordinal, sf.fragment_kind, sf.text, sf.text_sha256,
                                   sf.source_address_json
                            FROM source_fragments AS sf
                            JOIN source_versions AS sv
                              ON sv.source_version_id = sf.source_version_id
                            JOIN sources AS s ON s.source_id = sv.source_id
                            WHERE s.library_id = ?
                              AND sf.source_fragment_id IN ({placeholders})
                        """
                        rows.extend(connection.execute(query, (self.library_id, *chunk)).fetchall())

            if {str(row[0]) for row in rows} != set(source_fragment_ids):
                raise PersistenceIntegrityError(
                    "permitted fragment subset does not resolve to the exact persisted fragment set"
                )
            fragments: dict[str, SourceFragmentText] = {}
            for row in rows:
                fragment_id = str(row[0])
                item = expected[fragment_id]
                if str(row[1]) != item.source_version_id or str(row[2]) != item.source_id:
                    raise PersistenceIntegrityError(
                        "permitted manifest lineage differs from persisted fragment lineage"
                    )
                source_address_json = str(row[7])
                _load_canonical_object(source_address_json, "SourceAddress")
                text = str(row[5])
                text_hash = str(row[6])
                if sha256_hex(text.encode("utf-8")) != text_hash:
                    raise PersistenceIntegrityError("persisted fragment text hash mismatch")
                fragments[fragment_id] = SourceFragmentText(
                    source_fragment_id=fragment_id,
                    source_version_id=str(row[1]),
                    source_id=str(row[2]),
                    ordinal=int(row[3]),
                    fragment_kind=str(row[4]),
                    text=text,
                    text_sha256=text_hash,
                    source_address_json=source_address_json,
                )
            return tuple(fragments[item] for item in source_fragment_ids)

    def _read_large_permitted_fragment_subset(
        self,
        connection: sqlite3.Connection,
        source_fragment_ids: tuple[str, ...],
        expected: dict[str, PermittedManifestItem],
    ) -> list[sqlite3.Row]:
        """Read a large exact permit without a planner-sensitive text-key scan.

        The temporary table is metadata-only.  It pins caller order and the
        freshly recompiled manifest lineage before the protected text gate is
        opened.  Text is then materialized by monotonically increasing SQLite
        rowid, avoiding the unstable large ``IN (TEXT...)`` plan while the
        public result is still reconstructed in the caller-requested order.
        """

        table = _FRAGMENT_READ_PERMIT_TABLE
        connection.execute(f"DROP TABLE IF EXISTS temp.{table}")
        connection.execute(
            f"""
            CREATE TEMP TABLE {table}(
                source_fragment_id TEXT PRIMARY KEY,
                read_order INTEGER NOT NULL UNIQUE,
                expected_source_version_id TEXT NOT NULL,
                expected_source_id TEXT NOT NULL,
                expected_source_family_id TEXT NOT NULL,
                source_rowid INTEGER UNIQUE
            ) WITHOUT ROWID
            """
        )
        try:
            permit_rows: list[tuple[str, int, str, str, str]] = []
            for read_order, fragment_id in enumerate(source_fragment_ids):
                item = expected[fragment_id]
                if item.source_family_id is None:
                    raise PersistenceIntegrityError(
                        "permitted manifest fragment lacks SourceFamily lineage"
                    )
                permit_rows.append(
                    (
                        fragment_id,
                        read_order,
                        item.source_version_id,
                        item.source_id,
                        item.source_family_id,
                    )
                )
            connection.executemany(
                f"""
                INSERT INTO {table}(
                    source_fragment_id, read_order,
                    expected_source_version_id, expected_source_id,
                    expected_source_family_id, source_rowid
                ) VALUES (?, ?, ?, ?, ?, NULL)
                """,
                permit_rows,
            )
            connection.execute(
                f"""
                UPDATE {table} AS permit
                SET source_rowid = (
                    SELECT sf.rowid
                    FROM source_fragments AS sf
                         INDEXED BY sqlite_autoindex_source_fragments_1
                    WHERE sf.source_fragment_id = permit.source_fragment_id
                )
                """
            )
            lineage_rows = connection.execute(
                f"""
                SELECT permit.source_fragment_id, permit.read_order,
                       permit.expected_source_version_id, permit.expected_source_id,
                       permit.expected_source_family_id, permit.source_rowid,
                       sf.source_version_id, sv.source_id, s.library_id,
                       sfm.source_family_id, family.library_id
                FROM {table} AS permit
                LEFT JOIN source_fragments AS sf ON sf.rowid = permit.source_rowid
                LEFT JOIN source_versions AS sv
                  ON sv.source_version_id = sf.source_version_id
                LEFT JOIN sources AS s ON s.source_id = sv.source_id
                LEFT JOIN source_family_members AS sfm ON sfm.source_id = s.source_id
                LEFT JOIN source_families AS family
                  ON family.source_family_id = sfm.source_family_id
                """
            ).fetchall()
            if len(lineage_rows) != len(source_fragment_ids):
                raise PersistenceIntegrityError(
                    "permitted fragment subset does not resolve to the exact persisted fragment set"
                )
            rowids: list[int] = []
            for row in lineage_rows:
                if (
                    row[5] is None
                    or str(row[2]) != str(row[6])
                    or str(row[3]) != str(row[7])
                    or str(row[4]) != str(row[9])
                    or str(row[8]) != self.library_id
                    or str(row[10]) != self.library_id
                ):
                    raise PersistenceIntegrityError(
                        "permitted manifest lineage differs from persisted fragment lineage"
                    )
                rowids.append(int(row[5]))
            if len(set(rowids)) != len(source_fragment_ids):
                raise PersistenceIntegrityError(
                    "permitted fragment subset does not resolve to unique persisted rows"
                )

            rows: list[sqlite3.Row] = []
            with self._permit_fragment_text_read():
                for rowid_chunk in _chunks(tuple(str(value) for value in sorted(rowids)), 500):
                    placeholders = ", ".join("?" for _ in rowid_chunk)
                    rows.extend(
                        connection.execute(
                            f"""
                            SELECT sf.source_fragment_id, sf.source_version_id, sv.source_id,
                                   sf.ordinal, sf.fragment_kind, sf.text, sf.text_sha256,
                                   sf.source_address_json
                            FROM source_fragments AS sf
                            JOIN source_versions AS sv
                              ON sv.source_version_id = sf.source_version_id
                            JOIN sources AS s ON s.source_id = sv.source_id
                            WHERE s.library_id = ? AND sf.rowid IN ({placeholders})
                            ORDER BY sf.rowid
                            """,
                            (self.library_id, *rowid_chunk),
                        ).fetchall()
                    )
            return rows
        finally:
            connection.execute(f"DROP TABLE IF EXISTS temp.{table}")

    def _revalidate_authorized_read(
        self,
        authorization: AuthorizedRead,
    ) -> CompiledAccess:
        """Internal non-transaction-opening implementation for adapter use."""

        return self._revalidate_authorized_read_context(authorization).compiled

    def _revalidate_authorized_read_context(
        self,
        authorization: AuthorizedRead,
    ) -> _AuthorizedReadContext:
        """Return fresh data derived only from the immutable issuance snapshot."""

        if type(authorization) is not AuthorizedRead:
            raise AuthorizationError("repository access requires an exact AuthorizedRead")
        issued = self._authorizations.get(authorization.authorization_id)
        if (
            issued is None
            or issued.capability is not authorization
            or authorization.repository_instance_id != self._instance_id
        ):
            raise AuthorizationError("AuthorizedRead was not issued by this repository instance")
        try:
            current_fingerprint = _authorization_fingerprint(authorization)
        except Exception as error:
            raise AuthorizationError("AuthorizedRead changed after issuance") from error
        if current_fingerprint != issued.fingerprint:
            raise AuthorizationError("AuthorizedRead changed after issuance")
        current = self._compile_db_access(
            access_policy_id=issued.context.access_policy_id,
            scope=issued.context.scope,
        )
        if current != issued.context.compiled:
            raise AuthorizationError("authorized policy or snapshot metadata changed")
        try:
            final_fingerprint = _authorization_fingerprint(authorization)
        except Exception as error:
            raise AuthorizationError("AuthorizedRead changed during revalidation") from error
        if final_fingerprint != issued.fingerprint:
            raise AuthorizationError("AuthorizedRead changed during revalidation")
        return _AuthorizedReadContext(
            access_policy_id=issued.context.access_policy_id,
            scope=deepcopy(issued.context.scope),
            compiled=current,
        )

    def pending_outbox_events(self) -> tuple[OutboxEvent, ...]:
        """Return committed, undelivered events in deterministic delivery order."""

        rows = self._store.connection.execute(
            """
            SELECT event_id, event_type, aggregate_type, aggregate_id,
                   payload_json, payload_hash, occurred_at, delivered_at
            FROM event_outbox
            WHERE delivered_at IS NULL
            ORDER BY occurred_at, event_id
            """
        ).fetchall()
        return tuple(_outbox_event_from_row(row) for row in rows)

    def export_outbox(
        self,
        *,
        after_append: AfterAppendHook | None = None,
    ) -> OutboxExportResult:
        """Append committed events once by ID, then mark them delivered.

        A crash after the fsynced append but before ``delivered_at`` leaves the
        row pending. On retry the existing canonical line is verified and the
        row is marked without appending a duplicate.
        """

        self.verify()
        appended = 0
        deduplicated = 0
        delivered = 0
        last_event_id: str | None = None
        with _exclusive_event_file(self.paths.events) as event_descriptor:
            # Pending rows are read only after the interprocess lock is held.
            # The lock remains held through append/fsync and delivered_at.
            existing = _read_event_descriptor(event_descriptor)
            for event in self.pending_outbox_events():
                line = canonical_json_bytes(event.envelope()) + b"\n"
                previous = existing.get(event.event_id)
                if previous is None:
                    _append_and_fsync_descriptor(event_descriptor, line)
                    existing[event.event_id] = line
                    appended += 1
                    if after_append is not None:
                        after_append(event)
                elif previous != line:
                    raise OutboxExportError(
                        f"events.jsonl contains conflicting event ID: {event.event_id}"
                    )
                else:
                    deduplicated += 1

                delivered_at = _timestamp(self._clock)
                with self._store.transaction(immediate=True) as connection:
                    cursor = connection.execute(
                        """
                        UPDATE event_outbox SET delivered_at = ?
                        WHERE event_id = ? AND delivered_at IS NULL
                        """,
                        (delivered_at, event.event_id),
                    )
                    if cursor.rowcount != 1:
                        raise OutboxExportError(
                            f"outbox row changed during delivery: {event.event_id}"
                        )
                delivered += 1
                last_event_id = event.event_id
        return OutboxExportResult(appended, deduplicated, delivered, last_event_id)

    def create_online_backup(self) -> BackupRecord:
        """Create, verify, fsync, and atomically promote an online DB backup."""

        self.verify()
        if self._store.connection.in_transaction:
            raise BackupError("cannot start an online backup inside a transaction")
        created_at = _timestamp(self._clock)
        backup_id = new_id("backup")
        destination = self.paths.backups / f"{backup_id}.sqlite3"
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".online-",
                suffix=".sqlite3.tmp",
                dir=self.paths.backups,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            destination_connection = sqlite3.connect(temporary, isolation_level=None)
            try:
                self._store.connection.backup(destination_connection)
                runner = MigrationRunner(destination_connection)
                runner.verify()
                if runner.current_version() != runner.latest_version:
                    raise BackupError("backup schema is not current")
                verify_integrity(destination_connection)
            finally:
                destination_connection.close()
            os.chmod(temporary, 0o600)
            _fsync_file(temporary)
            os.replace(temporary, destination)
            _fsync_directory(self.paths.backups)
            return BackupRecord(
                path=destination,
                created_at=created_at,
                sha256=_file_sha256(destination),
                schema_version=self._store.schema_version,
            )
        except Exception as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            if isinstance(exc, BackupError):
                raise
            raise BackupError("online SQLite backup failed verification") from exc

    def _ensure_blob_row(
        self,
        connection: sqlite3.Connection,
        blob: PromotedBlob,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO blobs(
                content_sha256, byte_size, relative_path, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (blob.content_sha256, blob.byte_size, blob.relative_path, created_at),
        )
        row = connection.execute(
            """
            SELECT byte_size, relative_path FROM blobs WHERE content_sha256 = ?
            """,
            (blob.content_sha256,),
        ).fetchone()
        if row is None or int(row[0]) != blob.byte_size or str(row[1]) != blob.relative_path:
            raise PersistenceIntegrityError("Blob row conflicts with verified physical bytes")

    def _ensure_membership(
        self,
        connection: sqlite3.Connection,
        *,
        collection_id: str,
        source_id: str,
        state: str,
        created_at: str,
    ) -> None:
        row = connection.execute(
            """
            SELECT state FROM collection_memberships
            WHERE collection_id = ? AND source_id = ?
            """,
            (collection_id, source_id),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO collection_memberships(
                    collection_id, source_id, state, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (collection_id, source_id, state, created_at),
            )
        elif str(row[0]) != state:
            raise PersistenceIntegrityError(
                "existing Collection membership state differs from ingest scope"
            )

    def _declared_lineage_for_new_source(
        self,
        connection: sqlite3.Connection,
        *,
        declaration: SourceIdentityDeclaration,
        content_sha256: str,
    ) -> tuple[str, str] | None:
        """Resolve a declared logical identity without cross-identity byte merging."""

        del content_sha256
        row = connection.execute(
            """
            SELECT source_family_id FROM source_family_identities
            WHERE library_id = ? AND logical_source_uri = ?
            """,
            (self.library_id, declaration.logical_source_uri),
        ).fetchone()
        if row is None:
            return None
        family_id = str(row[0])
        roots = connection.execute(
            """
            SELECT source_id FROM source_family_members
            WHERE source_family_id = ? AND role = 'root' AND root_source_id IS NULL
            """,
            (family_id,),
        ).fetchall()
        if len(roots) != 1:
            raise SourceLineageIntegrityError(
                "declared SourceFamily must retain exactly one technical root Source"
            )
        return family_id, str(roots[0][0])

    def _declared_content_exists(
        self,
        connection: sqlite3.Connection,
        *,
        source_family_id: str,
        content_sha256: str,
    ) -> bool:
        return (
            connection.execute(
                """
                SELECT 1 FROM source_identity_bindings
                WHERE source_family_id = ? AND content_sha256 = ?
                LIMIT 1
                """,
                (source_family_id, content_sha256),
            ).fetchone()
            is not None
        )

    def _declared_lineage_for_existing_source(
        self,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        declaration: SourceIdentityDeclaration,
    ) -> tuple[str, str, SourceFamilyRole]:
        """Require an existing physical Source to retain its original declaration."""

        rows = connection.execute(
            """
            SELECT DISTINCT binding.source_family_id, identity.logical_source_uri
            FROM source_identity_bindings AS binding
            JOIN source_family_identities AS identity
              ON identity.source_family_id = binding.source_family_id
            WHERE binding.source_id = ? AND identity.library_id = ?
            """,
            (source_id, self.library_id),
        ).fetchall()
        if len(rows) != 1 or str(rows[0][1]) != declaration.logical_source_uri:
            raise SourceLineageIntegrityError(
                "declared logical identity conflicts with existing physical Source; "
                "rebinding is forbidden"
            )
        family_id, root_source_id, role = self._lineage_for_source(connection, source_id)
        if family_id != str(rows[0][0]):
            raise SourceLineageIntegrityError(
                "declared Source identity binding disagrees with lineage"
            )
        return family_id, root_source_id, role

    def _require_identity_binding(
        self,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        source_version_id: str,
        declaration: SourceIdentityDeclaration,
    ) -> None:
        row = connection.execute(
            """
            SELECT binding.connector, binding.connector_revision, binding.representation,
                   binding.part_number, binding.part_metadata_hash, binding.manifest_sha256,
                   identity.logical_source_uri
            FROM source_identity_bindings AS binding
            JOIN source_family_identities AS identity
              ON identity.source_family_id = binding.source_family_id
            WHERE binding.source_id = ? AND binding.source_version_id = ?
              AND identity.library_id = ?
            """,
            (source_id, source_version_id, self.library_id),
        ).fetchone()
        expected = (
            declaration.connector,
            declaration.connector_revision,
            declaration.representation,
            declaration.part_number,
            declaration.part_metadata_hash,
            declaration.manifest_sha256,
            declaration.logical_source_uri,
        )
        if row is None or tuple(row) != expected:
            raise SourceLineageIntegrityError(
                "declared identity does not match the immutable SourceVersion binding"
            )

    def _insert_identity_binding(
        self,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        source_version_id: str,
        source_family_id: str,
        declaration: SourceIdentityDeclaration,
        content_sha256: str,
        created_at: str,
    ) -> str:
        binding_id = new_id("source_identity")
        connection.execute(
            """
            INSERT INTO source_identity_bindings(
                source_identity_binding_id, source_id, source_version_id, source_family_id,
                connector, connector_revision, representation, part_number,
                part_metadata_hash, content_sha256, manifest_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                binding_id,
                source_id,
                source_version_id,
                source_family_id,
                declaration.connector,
                declaration.connector_revision,
                declaration.representation,
                declaration.part_number,
                declaration.part_metadata_hash,
                content_sha256,
                declaration.manifest_sha256,
                created_at,
            ),
        )
        return binding_id

    def _lineage_for_source(
        self,
        connection: sqlite3.Connection,
        source_id: str,
    ) -> tuple[str, str, SourceFamilyRole]:
        rows = connection.execute(
            """
            SELECT sfm.source_family_id, sfm.role, sfm.root_source_id
            FROM source_family_members AS sfm
            JOIN source_families AS sf
              ON sf.source_family_id = sfm.source_family_id
            WHERE sfm.source_id = ? AND sf.library_id = ?
            ORDER BY sfm.source_family_id
            """,
            (source_id, self.library_id),
        ).fetchall()
        if len(rows) != 1:
            raise SourceLineageIntegrityError(
                "every VS0 Source must belong to exactly one SourceFamily"
            )
        row = rows[0]
        try:
            role = SourceFamilyRole(str(row[1]))
        except ValueError as exc:
            raise SourceLineageIntegrityError("SourceFamily role is invalid") from exc
        family_id = str(row[0])
        if role is SourceFamilyRole.ROOT:
            if row[2] is not None:
                raise SourceLineageIntegrityError("root Source has a root_source_id")
            root_source_id = source_id
        else:
            if row[2] is None:
                raise SourceLineageIntegrityError("non-root Source lacks root_source_id")
            root_source_id = str(row[2])
            root = connection.execute(
                """
                SELECT 1 FROM source_family_members
                WHERE source_family_id = ? AND source_id = ?
                  AND role = 'root' AND root_source_id IS NULL
                """,
                (family_id, root_source_id),
            ).fetchone()
            if root is None:
                raise SourceLineageIntegrityError(
                    "SourceFamily root_source_id does not resolve to its root member"
                )
        return family_id, root_source_id, role

    def _lineage_for_content(
        self,
        connection: sqlite3.Connection,
        content_sha256: str,
        *,
        excluding_source_id: str | None,
    ) -> tuple[str, str] | None:
        if excluding_source_id is None:
            rows = connection.execute(
                """
                SELECT DISTINCT s.source_id
                FROM source_versions AS sv
                JOIN sources AS s ON s.source_id = sv.source_id
                WHERE s.library_id = ? AND sv.content_sha256 = ?
                ORDER BY s.source_id
                """,
                (self.library_id, content_sha256),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT DISTINCT s.source_id
                FROM source_versions AS sv
                JOIN sources AS s ON s.source_id = sv.source_id
                WHERE s.library_id = ? AND sv.content_sha256 = ?
                  AND s.source_id <> ?
                ORDER BY s.source_id
                """,
                (self.library_id, content_sha256, excluding_source_id),
            ).fetchall()
        if not rows:
            return None
        lineages = {
            lineage[:2]
            for lineage in (self._lineage_for_source(connection, str(row[0])) for row in rows)
        }
        if len(lineages) != 1:
            raise SourceLineageIntegrityError(
                "identical bytes resolve to incompatible SourceFamily roots"
            )
        return next(iter(lineages))

    def _current_snapshot_members(
        self,
        connection: sqlite3.Connection,
        collection_ids: tuple[str, ...],
    ) -> tuple[SnapshotMember, ...]:
        placeholders = ", ".join("?" for _ in collection_ids)
        raw_count = int(
            connection.execute(
                f"""
                SELECT COUNT(*) FROM collection_memberships
                WHERE collection_id IN ({placeholders})
                """,
                collection_ids,
            ).fetchone()[0]
        )
        rows = connection.execute(
            f"""
            SELECT cm.collection_id, cm.source_id, cm.state,
                   sh.source_version_id, sv.content_sha256
            FROM collection_memberships AS cm
            JOIN collections AS c
              ON c.collection_id = cm.collection_id AND c.library_id = ?
            JOIN sources AS s
              ON s.source_id = cm.source_id AND s.library_id = ?
            LEFT JOIN source_heads AS sh ON sh.source_id = cm.source_id
            LEFT JOIN source_versions AS sv
              ON sv.source_version_id = sh.source_version_id
             AND sv.source_id = cm.source_id
            WHERE cm.collection_id IN ({placeholders})
            ORDER BY cm.collection_id, sh.source_version_id, cm.source_id
            """,
            (self.library_id, self.library_id, *collection_ids),
        ).fetchall()
        if len(rows) != raw_count:
            raise PersistenceIntegrityError(
                "Collection memberships cross the current Library boundary"
            )

        members: list[SnapshotMember] = []
        try:
            for row in rows:
                if row[3] is None or row[4] is None:
                    raise PersistenceIntegrityError(
                        "Collection membership has no valid current SourceVersion head"
                    )
                source_id = str(row[1])
                family_id, root_source_id, family_role = self._lineage_for_source(
                    connection,
                    source_id,
                )
                members.append(
                    SnapshotMember(
                        collection_id=str(row[0]),
                        source_id=source_id,
                        source_version_id=str(row[3]),
                        content_sha256=str(row[4]),
                        membership_state=MembershipState(str(row[2])),
                        source_family_id=family_id,
                        root_source_id=root_source_id,
                        family_role=family_role,
                    )
                )
        except (SourceLineageIntegrityError, TypeError, ValueError) as exc:
            raise PersistenceIntegrityError(
                "current Collection state cannot form a valid CorpusSnapshot"
            ) from exc
        return tuple(members)

    def _reconstruct_corpus_snapshot(
        self,
        connection: sqlite3.Connection,
        corpus_snapshot_id: str,
    ) -> CorpusSnapshot:
        row = connection.execute(
            """
            SELECT corpus_snapshot_id, library_id, scope_json,
                   manifest_hash, created_at
            FROM corpus_snapshots
            WHERE corpus_snapshot_id = ?
            """,
            (corpus_snapshot_id,),
        ).fetchone()
        if row is None or str(row[1]) != self.library_id:
            raise CorpusSnapshotNotFoundError(
                f"CorpusSnapshot does not exist in Library {self.library_id}: {corpus_snapshot_id}"
            )

        scope_json = str(row[2])
        scope_payload = _load_canonical_object(scope_json, "CorpusSnapshot scope")
        scope_library_id = scope_payload.get("library_id")
        raw_collection_ids = scope_payload.get("collection_ids")
        if type(scope_library_id) is not str or type(raw_collection_ids) is not list:
            raise PersistenceIntegrityError("persisted CorpusSnapshot scope is invalid")
        if any(type(item) is not str for item in raw_collection_ids):
            raise PersistenceIntegrityError("persisted CorpusSnapshot scope is invalid")
        collection_ids = tuple(cast(list[str], raw_collection_ids))
        if scope_library_id != self.library_id:
            raise PersistenceIntegrityError(
                "persisted CorpusSnapshot scope crosses the Library boundary"
            )
        try:
            self._require_collections(connection, collection_ids)
        except CollectionNotFoundError as exc:
            raise PersistenceIntegrityError(
                "persisted CorpusSnapshot scope references an absent Collection"
            ) from exc

        raw_member_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM snapshot_members
                WHERE corpus_snapshot_id = ?
                """,
                (corpus_snapshot_id,),
            ).fetchone()[0]
        )
        member_rows = connection.execute(
            """
            SELECT sm.collection_id, sv.source_id, sm.source_version_id,
                   sv.content_sha256, sm.membership_state
            FROM snapshot_members AS sm
            JOIN collections AS c
              ON c.collection_id = sm.collection_id AND c.library_id = ?
            JOIN source_versions AS sv
              ON sv.source_version_id = sm.source_version_id
            JOIN sources AS s
              ON s.source_id = sv.source_id AND s.library_id = ?
            WHERE sm.corpus_snapshot_id = ?
            ORDER BY sm.collection_id, sm.source_version_id
            """,
            (self.library_id, self.library_id, corpus_snapshot_id),
        ).fetchall()
        if len(member_rows) != raw_member_count:
            raise PersistenceIntegrityError(
                "persisted CorpusSnapshot members cross the Library boundary"
            )

        try:
            members: list[SnapshotMember] = []
            for member_row in member_rows:
                source_id = str(member_row[1])
                family_id, root_source_id, family_role = self._lineage_for_source(
                    connection,
                    source_id,
                )
                members.append(
                    SnapshotMember(
                        collection_id=str(member_row[0]),
                        source_id=source_id,
                        source_version_id=str(member_row[2]),
                        content_sha256=str(member_row[3]),
                        membership_state=MembershipState(str(member_row[4])),
                        source_family_id=family_id,
                        root_source_id=root_source_id,
                        family_role=family_role,
                    )
                )
            snapshot = build_corpus_snapshot(
                library_id=scope_library_id,
                collection_ids=collection_ids,
                members=members,
            )
        except (SourceLineageIntegrityError, TypeError, ValueError) as exc:
            raise PersistenceIntegrityError("persisted CorpusSnapshot manifest is invalid") from exc

        persisted_scope = snapshot.semantic_payload()["scope"]
        if _canonical_text(persisted_scope) != scope_json:
            raise PersistenceIntegrityError(
                "persisted CorpusSnapshot scope differs from canonical reconstruction"
            )
        if str(row[0]) != snapshot.corpus_snapshot_id:
            raise PersistenceIntegrityError("persisted CorpusSnapshot ID mismatch")
        manifest_hash = _validated_sha256(str(row[3]), "CorpusSnapshot manifest hash")
        if manifest_hash != snapshot.manifest_hash:
            raise PersistenceIntegrityError("persisted CorpusSnapshot manifest hash mismatch")
        _validated_timestamp(str(row[4]))
        return snapshot

    def _compile_db_access(
        self,
        *,
        access_policy_id: str,
        scope: RequestScope,
    ) -> CompiledAccess:
        if not isinstance(scope, RequestScope):
            raise AuthorizationError("authorization requires a RequestScope")
        if scope.library_id != self.library_id:
            raise AuthorizationError("request scope belongs to a different Library")
        self._require_collections(self._store.connection, scope.collection_ids)
        policy = self.get_access_policy(access_policy_id).snapshot
        snapshot_row = self._store.connection.execute(
            """
            SELECT corpus_snapshot_id
            FROM corpus_snapshots
            WHERE library_id = ? AND manifest_hash = ?
            """,
            (self.library_id, scope.snapshot_hash),
        ).fetchone()
        if snapshot_row is None:
            raise AuthorizationError("request snapshot is not present in this Library")
        snapshot = self._reconstruct_corpus_snapshot(
            self._store.connection,
            str(snapshot_row[0]),
        )
        if not set(scope.collection_ids).issubset(snapshot.collection_ids):
            raise AuthorizationError("request references a Collection absent from the snapshot")
        candidates = self._snapshot_candidates(str(snapshot_row[0]))
        return compile_access(policy=policy, scope=scope, candidates=candidates)

    def _snapshot_candidates(self, corpus_snapshot_id: str) -> tuple[SnapshotCandidate, ...]:
        rows = self._store.connection.execute(
            """
            SELECT sf.source_fragment_id, sf.source_version_id, sv.source_id,
                   sm.collection_id, sm.membership_state
            FROM snapshot_members AS sm
            JOIN corpus_snapshots AS cs
              ON cs.corpus_snapshot_id = sm.corpus_snapshot_id
            JOIN collections AS c ON c.collection_id = sm.collection_id
            JOIN source_versions AS sv ON sv.source_version_id = sm.source_version_id
            JOIN sources AS s ON s.source_id = sv.source_id
            JOIN source_fragments AS sf ON sf.source_version_id = sv.source_version_id
            WHERE cs.corpus_snapshot_id = ?
              AND cs.library_id = ? AND c.library_id = ? AND s.library_id = ?
            """,
            (corpus_snapshot_id, self.library_id, self.library_id, self.library_id),
        ).fetchall()
        parts: dict[str, _CandidateParts] = {}
        source_ids: set[str] = set()
        for row in rows:
            fragment_id = str(row[0])
            source_version_id = str(row[1])
            source_id = str(row[2])
            try:
                membership = SnapshotMembership(
                    collection_id=str(row[3]),
                    state=MembershipState(str(row[4])),
                )
            except (TypeError, ValueError) as exc:
                raise PersistenceIntegrityError(
                    "snapshot contains an invalid Collection membership"
                ) from exc
            current = parts.get(fragment_id)
            if current is None:
                current = _CandidateParts(source_version_id, source_id, [])
                parts[fragment_id] = current
            elif current.source_version_id != source_version_id or current.source_id != source_id:
                raise PersistenceIntegrityError("snapshot fragment lineage is inconsistent")
            current.memberships.append(membership)
            source_ids.add(source_id)
        family_by_source = self._source_families(source_ids)
        return tuple(
            SnapshotCandidate(
                source_fragment_id=fragment_id,
                source_version_id=part.source_version_id,
                source_id=part.source_id,
                source_family_id=family_by_source.get(part.source_id),
                memberships=tuple(part.memberships),
            )
            for fragment_id, part in sorted(parts.items())
        )

    def _source_families(self, source_ids: set[str]) -> dict[str, str]:
        if not source_ids:
            return {}
        result: dict[str, str] = {}
        for chunk in _chunks(tuple(sorted(source_ids)), 500):
            placeholders = ", ".join("?" for _ in chunk)
            query = f"""
                SELECT sfm.source_id, sfm.source_family_id
                FROM source_family_members AS sfm
                JOIN source_families AS sf
                  ON sf.source_family_id = sfm.source_family_id
                WHERE sf.library_id = ? AND sfm.source_id IN ({placeholders})
                ORDER BY sfm.source_id, sfm.source_family_id
            """
            for row in self._store.connection.execute(query, (self.library_id, *chunk)):
                source_id = str(row[0])
                family_id = str(row[1])
                previous = result.setdefault(source_id, family_id)
                if previous != family_id:
                    raise PersistenceIntegrityError(
                        "VS0 authorization requires at most one SourceFamily per Source"
                    )
        return result

    def _require_collections(
        self,
        connection: sqlite3.Connection,
        collection_ids: Iterable[str],
    ) -> None:
        requested = set(collection_ids)
        if not requested:
            return
        found: set[str] = set()
        for chunk in _chunks(tuple(sorted(requested)), 500):
            placeholders = ", ".join("?" for _ in chunk)
            query = f"""
                SELECT collection_id FROM collections
                WHERE library_id = ? AND collection_id IN ({placeholders})
            """
            found.update(
                str(row[0])
                for row in connection.execute(query, (self.library_id, *chunk)).fetchall()
            )
        if found != requested:
            raise CollectionNotFoundError("one or more Collections are absent from this Library")

    def _require_sources(
        self,
        connection: sqlite3.Connection,
        source_ids: Iterable[str],
    ) -> None:
        requested = set(source_ids)
        if not requested:
            return
        found: set[str] = set()
        for chunk in _chunks(tuple(sorted(requested)), 500):
            placeholders = ", ".join("?" for _ in chunk)
            query = f"""
                SELECT source_id FROM sources
                WHERE library_id = ? AND source_id IN ({placeholders})
            """
            found.update(
                str(row[0])
                for row in connection.execute(query, (self.library_id, *chunk)).fetchall()
            )
        if found != requested:
            raise PersistenceIntegrityError(
                "AccessPolicy Source rules must reference Sources in this Library"
            )


def _load_library_record(
    connection: sqlite3.Connection,
    paths: LibraryPaths,
    *,
    expected_id: str,
) -> LibraryRecord:
    rows = connection.execute(
        """
        SELECT library_id, name, logical_identity_hash, created_at
        FROM libraries ORDER BY library_id
        """
    ).fetchall()
    if len(rows) != 1:
        raise PersistenceIntegrityError("a Library database must contain exactly one Library row")
    row = rows[0]
    try:
        config = LibraryConfig(library_id=str(row[0]), name=str(row[1]))
    except (TypeError, ValueError) as exc:
        raise PersistenceIntegrityError("persisted Library identity is invalid") from exc
    if config.library_id != expected_id:
        raise PersistenceIntegrityError("physical and logical Library IDs differ")
    expected_hash = library_logical_identity_hash(config)
    if str(row[2]) != expected_hash:
        raise PersistenceIntegrityError("persisted Library logical identity hash mismatch")
    return LibraryRecord(config, expected_hash, _validated_timestamp(str(row[3])), paths)


def _insert_outbox_event(
    connection: sqlite3.Connection,
    *,
    event_id: str,
    event_type: str,
    aggregate_type: str,
    aggregate_id: str,
    payload: dict[str, object],
    occurred_at: str,
) -> None:
    payload_json = _canonical_text(payload)
    connection.execute(
        """
        INSERT INTO event_outbox(
            event_id, event_type, aggregate_type, aggregate_id,
            payload_json, payload_hash, occurred_at, delivered_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            event_id,
            _validated_label(event_type, "event type"),
            _validated_label(aggregate_type, "aggregate type"),
            _validated_label(aggregate_id, "aggregate ID"),
            payload_json,
            canonical_sha256_hex(payload),
            _validated_timestamp(occurred_at),
        ),
    )


def _outbox_event_from_row(row: sqlite3.Row) -> OutboxEvent:
    payload = _load_canonical_object(str(row[4]), "outbox payload")
    payload_hash = canonical_sha256_hex(payload)
    if payload_hash != str(row[5]):
        raise PersistenceIntegrityError("outbox payload hash mismatch")
    delivered_at = None if row[7] is None else _validated_timestamp(str(row[7]))
    return OutboxEvent(
        event_id=str(row[0]),
        event_type=_validated_label(str(row[1]), "event type"),
        aggregate_type=_validated_label(str(row[2]), "aggregate type"),
        aggregate_id=_validated_label(str(row[3]), "aggregate ID"),
        payload=payload,
        payload_hash=payload_hash,
        occurred_at=_validated_timestamp(str(row[6])),
        delivered_at=delivered_at,
    )


def _collection_event_payload(
    config: CollectionConfig,
    root_ids: tuple[str, ...],
) -> dict[str, object]:
    roots: list[object] = []
    for root_id, root in zip(root_ids, config.roots, strict=True):
        roots.append(
            {
                "collection_root_id": root_id,
                "resolved_path": str(root.path),
                "include_globs": list(root.include_globs),
                "exclude_globs": list(root.exclude_globs),
                "identity_manifest_path": (
                    str(root.identity_manifest_path)
                    if root.identity_manifest_path is not None
                    else None
                ),
                "identity_manifest_sha256": root.identity_manifest_sha256,
            }
        )
    return {
        "schema": "dithyramba.collection_created/1.0",
        "collection_id": config.collection_id,
        "library_id": config.library_id,
        "name": config.name,
        "kind": config.kind.value,
        "roots": roots,
    }


def _read_event_file(path: Path) -> dict[str, bytes]:
    try:
        with _exclusive_event_file(path) as descriptor:
            return _read_event_descriptor(descriptor)
    except OSError as exc:
        raise OutboxExportError(f"events file cannot be read: {path}") from exc


def _read_event_descriptor(descriptor: int) -> dict[str, bytes]:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    except OSError as exc:
        raise OutboxExportError("events file read failed") from exc
    payload = b"".join(chunks)
    if payload and not payload.endswith(b"\n"):
        raise OutboxExportError("events.jsonl ends with a partial line")
    result: dict[str, bytes] = {}
    for line in payload.splitlines(keepends=True):
        try:
            decoded_text = line[:-1].decode("utf-8")
            event = _load_canonical_object(decoded_text, "events.jsonl line")
        except (UnicodeDecodeError, PersistenceIntegrityError) as exc:
            raise OutboxExportError("events.jsonl contains an invalid canonical line") from exc
        event_id = event.get("event_id")
        if type(event_id) is not str or not event_id.startswith("event_"):
            raise OutboxExportError("events.jsonl line has an invalid event_id")
        if event_id in result:
            raise OutboxExportError(f"events.jsonl contains duplicate event ID: {event_id}")
        result[event_id] = line
    return result


def _append_and_fsync(path: Path, payload: bytes) -> None:
    try:
        with _exclusive_event_file(path) as descriptor:
            _append_and_fsync_descriptor(descriptor, payload)
    except (OSError, OutboxExportError) as exc:
        raise OutboxExportError(f"event append failed: {path}") from exc


def _append_and_fsync_descriptor(descriptor: int, payload: bytes) -> None:
    try:
        os.lseek(descriptor, 0, os.SEEK_END)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
    except OSError as exc:
        raise OutboxExportError("event append/fsync failed") from exc


@contextmanager
def _exclusive_event_file(path: Path) -> Iterator[int]:
    """Hold a POSIX advisory lock for one complete outbox export decision."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - VS0 validation target is macOS
        raise OutboxExportError("interprocess outbox locking is unavailable") from exc

    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                path,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise OutboxExportError(f"events file cannot be opened: {path}") from exc
        _require_private_regular_descriptor(descriptor, path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise OutboxExportError(f"events file cannot be locked: {path}") from exc
        yield descriptor
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _canonical_text(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _scoped_count(
    connection: sqlite3.Connection,
    statement: str,
    *parameters: str,
) -> int:
    """Run one fixed aggregate query and validate its scalar result."""

    row = connection.execute(statement, parameters).fetchone()
    if row is None or len(row) != 1 or type(row[0]) is not int or int(row[0]) < 0:
        raise PersistenceIntegrityError("Library description count is invalid")
    return int(row[0])


def _canonical_ingest_coverage_from_payload(
    payload: dict[str, object],
) -> tuple[CoverageDraft, tuple[IngestInputOutcome, ...]]:
    """Rebuild an ingest report from exact typed outcomes and reject ambiguity."""

    processing_run_id = payload.get("processing_run_id")
    collection_id = payload.get("collection_id")
    if type(processing_run_id) is not str or not processing_run_id:
        raise PersistenceIntegrityError("CoverageReport processing_run_id is invalid")
    if type(collection_id) is not str or not collection_id:
        raise PersistenceIntegrityError("CoverageReport collection_id is invalid")
    raw_outcomes = payload.get("outcomes")
    if type(raw_outcomes) is not list:
        raise PersistenceIntegrityError("CoverageReport outcomes must be an array")
    outcomes: list[IngestInputOutcome] = []
    seen_inputs: set[tuple[str, str | None]] = set()
    for raw_outcome in raw_outcomes:
        try:
            outcome = IngestInputOutcome.from_payload(raw_outcome)
        except (TypeError, ValueError) as exc:
            raise PersistenceIntegrityError("CoverageReport outcome payload is invalid") from exc
        input_identity = (outcome.collection_root_id, outcome.relative_path)
        if input_identity in seen_inputs:
            raise PersistenceIntegrityError("CoverageReport contains a duplicate input outcome")
        seen_inputs.add(input_identity)
        outcomes.append(outcome)
    rebuilt = build_ingest_coverage(
        processing_run_id=processing_run_id,
        collection_id=collection_id,
        outcomes=tuple(outcomes),
    )
    if rebuilt.report_json != _canonical_text(payload):
        raise PersistenceIntegrityError(
            "CoverageReport differs from its typed canonical reconstruction"
        )
    return rebuilt, tuple(outcomes)


def _validate_ingest_run_outcomes(
    status: ProcessingRunStatus,
    error_code: str | None,
    outcomes: tuple[IngestInputOutcome, ...],
) -> None:
    infrastructure = tuple(item for item in outcomes if item.infrastructure_failure)
    if status is ProcessingRunStatus.FAILED:
        if not infrastructure:
            raise PersistenceIntegrityError(
                "failed ingest ProcessingRun requires an infrastructure-failure outcome"
            )
        if error_code not in {item.failure_code for item in infrastructure}:
            raise PersistenceIntegrityError(
                "failed ingest error_code must match an infrastructure-failure outcome"
            )
    elif status is ProcessingRunStatus.SUCCEEDED:
        if infrastructure:
            raise PersistenceIntegrityError(
                "successful ingest ProcessingRun cannot contain infrastructure failure"
            )
    else:
        raise PersistenceIntegrityError("CoverageReport requires a terminal ingest run")


def _load_canonical_object(value: str, label: str) -> dict[str, object]:
    try:
        decoded: object = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PersistenceIntegrityError(f"{label} is not valid JSON") from exc
    if type(decoded) is not dict:
        raise PersistenceIntegrityError(f"{label} must be a JSON object")
    result = cast(dict[str, object], decoded)
    if _canonical_text(result) != value:
        raise PersistenceIntegrityError(f"{label} is not canonical JSON")
    return result


def _load_canonical_string_tuple(value: str, label: str) -> tuple[str, ...]:
    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PersistenceIntegrityError(f"{label} is not valid JSON") from exc
    if type(decoded) is not list or any(type(item) is not str for item in decoded):
        raise PersistenceIntegrityError(f"{label} must be a JSON string array")
    values = cast(list[str], decoded)
    if _canonical_text(values) != value:
        raise PersistenceIntegrityError(f"{label} is not canonical JSON")
    return tuple(values)


def _strict_bool(value: object, label: str) -> bool:
    if type(value) is not int or value not in (0, 1):
        raise PersistenceIntegrityError(f"{label} must be stored as 0 or 1")
    return bool(value)


def _validated_label(value: str, label: str) -> str:
    if not value or value != value.strip() or len(value) > _MAX_LABEL_LENGTH:
        raise PersistenceIntegrityError(f"{label} must be non-empty, unpadded short text")
    return value


def _validated_sha256(value: str, label: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise PersistenceIntegrityError(f"{label} must be lowercase SHA-256 hex")
    return value


def _system_clock() -> datetime:
    return datetime.now(UTC)


def _timestamp(clock: Clock) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PersistenceIntegrityError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _validated_timestamp(value: str) -> str:
    if _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise PersistenceIntegrityError(
            "operational timestamp must be UTC RFC3339 with microseconds"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise PersistenceIntegrityError("operational timestamp is not a real date-time") from exc
    return value


def _new_event_id() -> str:
    return new_id("event")


def _require_latest_schema(store: Store) -> None:
    runner = MigrationRunner(store.connection)
    if runner.current_version() != runner.latest_version:
        raise PersistenceIntegrityError("Library database has pending migrations")


def _chunks(values: tuple[str, ...], size: int) -> Iterator[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        _require_private_regular_descriptor(descriptor, path)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_private_regular_descriptor(descriptor: int, path: Path) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise OutboxExportError(f"expected one private regular file: {path}")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise OutboxExportError(f"file descriptor is not private: {path}")
