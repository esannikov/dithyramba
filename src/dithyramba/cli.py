"""Dithyramba command-line entrypoint."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, ParamSpec, TypeVar

import typer
import uvicorn
from pydantic import ValidationError

from dithyramba import __version__
from dithyramba.access import (
    AccessContractError,
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    QueryExclusions,
    RequestScope,
    SourceRule,
)
from dithyramba.answers import ClaimEvidenceCaseSet, ClaimEvidenceEntailmentResult
from dithyramba.api import (
    ConceptLensWebConfig,
    EvidenceBoardWebConfig,
    LensMode,
    bearer_token_for,
    create_app,
    create_lens_app,
)
from dithyramba.atlas import migrate_legacy_research_atlas
from dithyramba.backup import (
    BackupBundleError,
    create_backup_bundle,
    restore_backup_bundle,
    verify_backup_bundle,
)
from dithyramba.collections import CollectionConfig, CollectionKind, CollectionRoot
from dithyramba.contracts import ContractError, canonical_json_bytes, new_id
from dithyramba.ingest.errors import IngestError
from dithyramba.ingest.models import parser_profile
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.mcp_stdio import run_stdio_mcp
from dithyramba.persistence import (
    AccessPolicyRecord,
    CollectionRecord,
    LibraryDescription,
    LibraryRecord,
    LibraryRepository,
    PersistenceError,
    SQLiteRecallBackend,
    SQLiteResearchSessionRepository,
    initialize_library,
    list_libraries,
    open_library,
)
from dithyramba.provenance import (
    IngestBatchResult,
    ProcessingRunRecord,
    ProvenanceError,
    SourceRecord,
    SourceVersionRecord,
)
from dithyramba.reading_room import ReadingRoomError, ReadingRoomLimits
from dithyramba.reasoning import (
    IdeaTrace,
    ReasoningClosureDecision,
    ReasoningClosureGate,
)
from dithyramba.recall import (
    FtsError,
    QueryRequest,
    RecallError,
    RecallResult,
    RecallService,
    RetrievalBudget,
    current_fts_runtime_profile,
)
from dithyramba.review import (
    ReviewAction,
    ReviewDecisionRequest,
    ReviewError,
    ReviewScope,
    ReviewTargetType,
)
from dithyramba.review.service import ReviewService
from dithyramba.store import StoreError

_P = ParamSpec("_P")
_R = TypeVar("_R")
_PURPOSE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

app = typer.Typer(
    name="dithyramba",
    help="Local, source-grounded interpretive memory for replayable evidence packets.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
library_app = typer.Typer(help="Create and inspect explicit physical Libraries.")
collection_app = typer.Typer(help="Manage logical Collections inside one Library.")
access_policy_app = typer.Typer(help="Create and inspect immutable access policies.")
source_app = typer.Typer(help="Ingest and inspect source provenance without source text.")
packet_app = typer.Typer(help="Inspect and deterministically replay EvidencePackets.")
review_app = typer.Typer(help="Record and inspect scoped, append-only human reviews.")
lens_app = typer.Typer(
    help="Open one read-only human view of a Library, session, atlas, concept map, or flow."
)
app.add_typer(library_app, name="library")
app.add_typer(collection_app, name="collection")
app.add_typer(access_policy_app, name="access-policy")
app.add_typer(source_app, name="source")
app.add_typer(packet_app, name="packet")
app.add_typer(review_app, name="review")
app.add_typer(lens_app, name="lens")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def _guard(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Render expected operator failures without leaking a traceback."""

    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except (
            AccessContractError,
            ContractError,
            BackupBundleError,
            FtsError,
            IngestError,
            PersistenceError,
            ProvenanceError,
            RecallError,
            ReadingRoomError,
            ReviewError,
            StoreError,
            ValidationError,
            OSError,
            sqlite3.Error,
        ) as exc:
            typer.echo(f"error: {_human_text(str(exc))}", err=True)
            raise typer.Exit(code=1) from None

    return guarded


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed package version and exit.",
        ),
    ] = False,
) -> None:
    """Run a Dithyramba command."""


@app.command()
def about() -> None:
    """Describe the current pre-alpha source preview and its limits."""

    typer.echo(
        "Dithyramba 0.1.0rc1 is a pre-alpha source preview with a persisted, "
        "replayable local FTS evidence route. It is not yet a validated memory system; "
        "graph recall and synthesis remain experimental or incomplete."
    )


@app.command("atlas-migrate")
@_guard
def atlas_migrate(
    source: Annotated[
        Path,
        typer.Argument(help="Historical Research Atlas manifest."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New manifest path; parent must already exist."),
    ],
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help="Existing case-data root used to rebuild exact line bindings.",
        ),
    ] = None,
    trace_overrides: Annotated[
        Path | None,
        typer.Option(
            "--trace-overrides",
            help="Optional JSON map of exact visible spans to evidence IDs.",
        ),
    ] = None,
    receipt_output: Annotated[
        Path | None,
        typer.Option(
            "--receipt-output",
            help="Optional migration receipt path; defaults beside the output manifest.",
        ),
    ] = None,
) -> None:
    """Migrate one historical Atlas without modifying its frozen input."""

    if not source.is_file():
        raise ContractError("Atlas source must be an existing file")
    if not output.is_absolute() or not output.parent.is_dir():
        raise ContractError("Atlas output must be absolute and its parent must exist")
    if artifact_root is not None and (
        not artifact_root.is_absolute() or not artifact_root.is_dir()
    ):
        raise ContractError("artifact root must be an existing absolute directory")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        overrides = (
            {}
            if trace_overrides is None
            else json.loads(trace_overrides.read_text(encoding="utf-8"))
        )
        if not isinstance(payload, dict) or not isinstance(overrides, dict):
            raise ValueError("Atlas input and trace overrides must be JSON objects")
        migrated = migrate_legacy_research_atlas(
            payload,
            artifact_root=artifact_root,
            trace_overrides=overrides,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc
    receipt_path = receipt_output or output.with_name("MIGRATION_RECEIPT.json")
    if not receipt_path.is_absolute() or not receipt_path.parent.is_dir():
        raise ContractError("receipt output must be absolute and its parent must exist")
    output.write_bytes(
        canonical_json_bytes(migrated.manifest.model_dump(mode="json", exclude_none=True)) + b"\n"
    )
    receipt_path.write_bytes(canonical_json_bytes(migrated.receipt.model_dump(mode="json")) + b"\n")
    typer.echo(f"manifest: {output}")
    typer.echo(f"receipt: {receipt_path}")
    typer.echo(f"sources: {migrated.receipt.source_count}")
    typer.echo(f"evidence: {migrated.receipt.evidence_count}")
    typer.echo(f"exact_bindings: {migrated.receipt.exact_binding_count}")
    typer.echo(f"trace_spans: {migrated.receipt.trace_span_count}")


@app.command("index")
@_guard
def index_collection(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    collection: Annotated[str, typer.Option("--collection", help="Explicit Collection ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    parser_profile_name: Annotated[
        str,
        typer.Option(
            "--parser-profile",
            help="Bounded ingest profile: default or large-document.",
        ),
    ] = "default",
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Synchronously ingest every discovered input in one Collection."""

    try:
        profile_version, limits = parser_profile(parser_profile_name)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    with open_library(library, data_root=data_home) as repository:
        result = IngestService(
            repository, limits=limits, profile_version=profile_version
        ).ingest_collection(collection)
    _emit_ingest_result(
        result,
        library_id=library,
        collection_id=collection,
        json_output=json_output,
    )


@library_app.command("init")
@_guard
def library_init(
    name: Annotated[str, typer.Argument(help="Human-facing Library name.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    source_root: Annotated[
        list[Path] | None,
        typer.Option("--source-root", help="Declared source root; repeat as needed."),
    ] = None,
    sync_root: Annotated[
        list[Path] | None,
        typer.Option("--sync-root", help="Declared synchronization root; repeat as needed."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Create one private, physically isolated Library."""

    repository = initialize_library(
        LibraryConfig(name=name),
        data_root=data_home,
        declared_source_roots=tuple(source_root or ()),
        declared_sync_roots=tuple(sync_root or ()),
    )
    with repository:
        payload = _library_payload(repository)
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"library_id: {payload['library_id']}",
            f"name: {payload['name']}",
            f"path: {payload['path']}",
        ),
    )


@library_app.command("list")
@_guard
def library_list(
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """List verified Libraries below one explicit application-data root."""

    libraries = tuple(
        _library_record_payload(record) for record in list_libraries(data_root=data_home)
    )
    payload: dict[str, object] = {"libraries": list(libraries)}
    _emit(
        payload,
        json_output=json_output,
        human_lines=tuple(
            f"{record['library_id']}\t{record['name']}\t{record['path']}" for record in libraries
        )
        or ("No Libraries.",),
    )


@library_app.command("describe")
@_guard
def library_describe(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    recent_runs: Annotated[
        int,
        typer.Option(
            "--recent-runs",
            min=1,
            max=100,
            help="Number of recent ProcessingRuns to include.",
        ),
    ] = 10,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Describe one verified Library without reading source text."""

    with open_library(library, data_root=data_home) as repository:
        description = repository.describe(recent_run_limit=recent_runs)
        payload = _library_description_payload(
            description,
            schema_version=repository.schema_version,
            schema_fingerprint=repository.schema_fingerprint,
        )
    counts = description.counts
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"library_id: {library}",
            f"name: {description.library.config.name}",
            "status: ok",
            f"collections: {counts.collections}",
            f"sources: {counts.sources}",
            f"source_fragments: {counts.source_fragments}",
            f"snapshots: {counts.corpus_snapshots}",
            f"research_sessions: {counts.research_sessions}",
            "external_services_contacted: false",
            "backup_tracking: external_bundle",
        ),
    )


@library_app.command("doctor")
@_guard
def library_doctor(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Verify layout, schema, integrity, and local SQLite capabilities."""

    payload = _doctor_payload(library, data_home)
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"library_id: {payload['library_id']}",
            f"status: {payload['status']}",
            f"schema_version: {payload['schema_version']}",
            f"schema_fingerprint: {payload['schema_fingerprint']}",
            f"fts5_available: {str(payload['fts5_available']).lower()}",
        ),
    )


@collection_app.command("add")
@_guard
def collection_add(
    name: Annotated[str, typer.Argument(help="Human-facing Collection name.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    root: Annotated[
        list[Path],
        typer.Option("--root", help="Absolute corpus root; repeat as needed."),
    ],
    kind: Annotated[
        CollectionKind,
        typer.Option("--kind", help="Logical role of this Collection."),
    ],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    include: Annotated[
        list[str] | None,
        typer.Option("--include", help="Relative include glob; repeat as needed."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Relative exclude glob; repeat as needed."),
    ] = None,
    identity_manifest: Annotated[
        list[Path] | None,
        typer.Option(
            "--identity-manifest",
            help="Pinned absolute identity manifest; repeat once for each --root.",
        ),
    ] = None,
    identity_manifest_sha256: Annotated[
        list[str] | None,
        typer.Option(
            "--identity-manifest-sha256",
            help="SHA-256 matching each --identity-manifest; repeat in root order.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Create one logical Collection pointing at validated external roots."""

    includes = (
        tuple(include)
        if include is not None
        else ("**/*.md", "**/*.markdown", "**/*.txt", "**/*.pdf")
    )
    excludes = tuple(exclude or ())
    manifests = identity_manifest or []
    manifest_hashes = identity_manifest_sha256 or []
    if bool(manifests) != bool(manifest_hashes) or (
        manifests and (len(manifests) != len(root) or len(manifest_hashes) != len(root))
    ):
        raise ContractError(
            "identity manifest paths and SHA-256 pins must both occur once for each --root"
        )
    config = CollectionConfig(
        library_id=library,
        name=name,
        kind=kind,
        roots=tuple(
            CollectionRoot(
                path=item,
                include_globs=includes,
                exclude_globs=excludes,
                identity_manifest_path=(manifests[index] if manifests else None),
                identity_manifest_sha256=(manifest_hashes[index] if manifest_hashes else None),
            )
            for index, item in enumerate(root)
        ),
    )
    with open_library(library, data_root=data_home) as repository:
        record = repository.create_collection(config)
        payload = _collection_payload(record)
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"collection_id: {payload['collection_id']}",
            f"name: {payload['name']}",
            f"kind: {payload['kind']}",
        ),
    )


@collection_app.command("list")
@_guard
def collection_list(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """List Collections inside one explicit Library."""

    with open_library(library, data_root=data_home) as repository:
        collections = tuple(_collection_payload(record) for record in repository.list_collections())
    payload: dict[str, object] = {"collections": list(collections), "library_id": library}
    _emit(
        payload,
        json_output=json_output,
        human_lines=tuple(
            f"{record['collection_id']}\t{record['kind']}\t{record['name']}"
            for record in collections
        )
        or ("No Collections.",),
    )


@collection_app.command("freeze")
@_guard
def collection_freeze(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    collection: Annotated[
        list[str],
        typer.Option("--collection", help="Collection ID; repeat for an exact scope."),
    ],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Freeze current Source heads into one immutable CorpusSnapshot."""

    if not collection or len(collection) != len(set(collection)):
        raise ContractError("Collection scope must be non-empty and contain unique IDs")
    with open_library(library, data_root=data_home) as repository:
        snapshot = repository.freeze_snapshot(tuple(collection))
    payload = {
        "schema": snapshot.SCHEMA,
        "corpus_snapshot_id": snapshot.corpus_snapshot_id,
        "manifest_hash": snapshot.manifest_hash,
        **snapshot.semantic_payload(),
    }
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"corpus_snapshot_id: {snapshot.corpus_snapshot_id}",
            f"manifest_hash: {snapshot.manifest_hash}",
            f"members: {len(snapshot.members)}",
        ),
    )


@source_app.command("add")
@_guard
def source_add(
    relative_path: Annotated[
        str,
        typer.Argument(help="Relative POSIX path below the selected Collection root."),
    ],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    collection: Annotated[str, typer.Option("--collection", help="Explicit Collection ID.")],
    collection_root: Annotated[
        str | None,
        typer.Option(
            "--collection-root",
            help="Collection root ID; required when the Collection has multiple roots.",
        ),
    ] = None,
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    parser_profile_name: Annotated[
        str,
        typer.Option(
            "--parser-profile",
            help="Bounded ingest profile: default or large-document.",
        ),
    ] = "default",
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Ingest one explicit source path and emit its complete outcome receipt."""

    try:
        profile_version, limits = parser_profile(parser_profile_name)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    with open_library(library, data_root=data_home) as repository:
        record = repository.get_collection(collection)
        _validate_collection_root_selection(record, collection_root)
        service = IngestService(repository, limits=limits, profile_version=profile_version)
        result = service.ingest_path(
            collection,
            relative_path,
            collection_root_id=collection_root,
        )
    _emit_ingest_result(
        result,
        library_id=library,
        collection_id=collection,
        json_output=json_output,
    )


@source_app.command("list")
@_guard
def source_list(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    collection: Annotated[
        str | None,
        typer.Option("--collection", help="Optional explicit Collection ID filter."),
    ] = None,
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """List source metadata in one Library without paths or source text."""

    with open_library(library, data_root=data_home) as repository:
        sources = tuple(
            _source_payload(record) for record in repository.list_sources(collection_id=collection)
        )
    payload: dict[str, object] = {
        "collection_id": collection,
        "library_id": library,
        "sources": list(sources),
    }
    _emit(
        payload,
        json_output=json_output,
        human_lines=tuple(
            f"{record['source_id']}\t{record['media_type']}\t"
            f"{_human_text(str(record['title'])) if record['title'] else '-'}"
            for record in sources
        )
        or ("No Sources.",),
    )


@source_app.command("versions")
@_guard
def source_versions(
    source_id: Annotated[str, typer.Argument(help="Source ID.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """List immutable versions of one Library-scoped Source."""

    with open_library(library, data_root=data_home) as repository:
        source = repository.get_source(source_id)
        versions = tuple(
            _source_version_payload(record) for record in repository.list_source_versions(source_id)
        )
    payload: dict[str, object] = {
        "library_id": library,
        "source": _source_payload(source),
        "versions": list(versions),
    }
    _emit(
        payload,
        json_output=json_output,
        human_lines=tuple(
            f"{record['version_number']}\t{record['source_version_id']}\t"
            f"{record['parse_status']}\tcurrent={str(record['is_current']).lower()}"
            for record in versions
        ),
    )


@access_policy_app.command("create")
@_guard
def access_policy_create(
    name: Annotated[str, typer.Argument(help="Human-facing AccessPolicy name.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    purpose: Annotated[
        list[str],
        typer.Option("--purpose", help="Allowed purpose; repeat as needed."),
    ],
    allow_collection: Annotated[
        list[str],
        typer.Option("--allow-collection", help="Allowed Collection ID; repeat as needed."),
    ],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    deny_collection: Annotated[
        list[str] | None,
        typer.Option("--deny-collection", help="Denied Collection ID; repeat as needed."),
    ] = None,
    allow_source: Annotated[
        list[str] | None,
        typer.Option("--allow-source", help="Allowed Source ID; repeat as needed."),
    ] = None,
    deny_source: Annotated[
        list[str] | None,
        typer.Option("--deny-source", help="Denied Source ID; repeat as needed."),
    ] = None,
    allow_export: Annotated[
        bool,
        typer.Option("--allow-export", help="Allow derived artifact export."),
    ] = False,
    allow_external_provider: Annotated[
        bool,
        typer.Option(
            "--allow-external-provider",
            help="Allow content disclosure to an external model provider.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Persist an immutable, default-deny AccessPolicy snapshot."""

    snapshot = AccessPolicySnapshot(
        access_policy_id=new_id("policy"),
        library_id=library,
        allowed_purposes=tuple(purpose),
        collection_rules=tuple(
            [CollectionRule(item, PolicyEffect.ALLOW) for item in allow_collection]
            + [CollectionRule(item, PolicyEffect.DENY) for item in deny_collection or ()]
        ),
        source_rules=tuple(
            [SourceRule(item, PolicyEffect.ALLOW) for item in allow_source or ()]
            + [SourceRule(item, PolicyEffect.DENY) for item in deny_source or ()]
        ),
        allow_export=allow_export,
        allow_external_provider=allow_external_provider,
    )
    with open_library(library, data_root=data_home) as repository:
        record = repository.persist_access_policy(name=name, snapshot=snapshot)
        payload = _policy_payload(record)
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"access_policy_id: {payload['access_policy_id']}",
            f"name: {payload['name']}",
            f"policy_hash: {payload['policy_hash']}",
        ),
    )


@access_policy_app.command("show")
@_guard
def access_policy_show(
    policy_id: Annotated[str, typer.Argument(help="AccessPolicy ID.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Show one immutable AccessPolicy without corpus content."""

    with open_library(library, data_root=data_home) as repository:
        payload = _policy_payload(repository.get_access_policy(policy_id))
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"access_policy_id: {payload['access_policy_id']}",
            f"name: {payload['name']}",
            f"policy_hash: {payload['policy_hash']}",
        ),
    )


@access_policy_app.command("check")
@_guard
def access_policy_check(
    policy_id: Annotated[str, typer.Argument(help="AccessPolicy ID.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    purpose: Annotated[str, typer.Option("--purpose", help="Requested purpose.")],
    collection: Annotated[
        list[str],
        typer.Option("--collection", help="Requested Collection ID; repeat as needed."),
    ],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Evaluate purpose and Collection rules without reading corpus content."""

    if _PURPOSE_PATTERN.fullmatch(purpose) is None:
        raise ContractError("purpose must match [a-z][a-z0-9_-]{0,63}")
    requested = tuple(sorted(set(collection)))
    if not requested or len(requested) != len(collection):
        raise ContractError("Collection scope must be non-empty and contain unique IDs")
    with open_library(library, data_root=data_home) as repository:
        policy = repository.get_access_policy(policy_id)
        for collection_id in requested:
            repository.get_collection(collection_id)
        payload = _policy_check_payload(policy.snapshot, purpose, requested)
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"access_policy_id: {payload['access_policy_id']}",
            f"purpose_allowed: {str(payload['purpose_allowed']).lower()}",
            f"request_allowed: {str(payload['request_allowed']).lower()}",
            "default_deny: true",
        ),
    )


@app.command("recall")
@_guard
def recall(
    question: Annotated[str, typer.Argument(help="Exact question stored in QueryRequest.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    collection: Annotated[
        list[str],
        typer.Option("--collection", help="Collection ID; repeat for the exact scope."),
    ],
    snapshot: Annotated[
        str,
        typer.Option("--snapshot", help="Frozen CorpusSnapshot ID."),
    ],
    access_policy: Annotated[
        str,
        typer.Option("--access-policy", help="Immutable AccessPolicy ID."),
    ],
    purpose: Annotated[str, typer.Option("--purpose", help="Canonical policy purpose.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    max_candidates: Annotated[
        int,
        typer.Option("--max-candidates", min=1, max=500, help="Bounded FTS trace size."),
    ] = 100,
    max_source_fragments: Annotated[
        int,
        typer.Option(
            "--max-source-fragments",
            min=1,
            max=100,
            help="Bounded EvidencePacket selection size.",
        ),
    ] = 30,
    exclude_source: Annotated[
        list[str] | None,
        typer.Option("--exclude-source", help="Explicit Source exclusion; repeat as needed."),
    ] = None,
    exclude_family: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-family",
            help="Explicit SourceFamily exclusion; repeat as needed.",
        ),
    ] = None,
    exclude_fragment: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-fragment",
            help="Explicit SourceFragment exclusion; repeat as needed.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Run permitted-only local FTS and persist one reproducible EvidencePacket."""

    request = QueryRequest(
        question=question,
        library_id=library,
        collection_ids=tuple(collection),
        corpus_snapshot_id=snapshot,
        access_policy_id=access_policy,
        purpose=purpose,
        exclusions=QueryExclusions(
            source_ids=tuple(exclude_source or ()),
            source_family_ids=tuple(exclude_family or ()),
            source_fragment_ids=tuple(exclude_fragment or ()),
        ),
        retrieval=RetrievalBudget(
            max_candidates=max_candidates,
            max_source_fragments=max_source_fragments,
        ),
    )
    with open_library(library, data_root=data_home) as repository:
        result = _recall_service(repository).recall(request)
    payload = _recall_result_payload(result)
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"processing_run_id: {result.run.processing_run_id}",
            f"evidence_packet_id: {result.packet.evidence_packet_id}",
            f"packet_hash: {result.packet.packet_hash}",
            f"result_status: {result.packet.result_status.value}",
            f"source_fragments: {len(result.packet.source_fragments)}",
        ),
    )


@packet_app.command("inspect")
@_guard
def packet_inspect(
    evidence_packet_id: Annotated[str, typer.Argument(help="EvidencePacket ID.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Load and fully revalidate one persisted EvidencePacket."""

    with open_library(library, data_root=data_home) as repository:
        packet = SQLiteRecallBackend(repository).load_evidence_packet(evidence_packet_id)
    _emit(
        packet.payload(),
        json_output=json_output,
        human_lines=(
            f"evidence_packet_id: {packet.evidence_packet_id}",
            f"packet_hash: {packet.packet_hash}",
            f"result_status: {packet.result_status.value}",
            f"source_fragments: {len(packet.source_fragments)}",
        ),
    )


@packet_app.command("replay")
@_guard
def packet_replay(
    evidence_packet_id: Annotated[str, typer.Argument(help="EvidencePacket ID.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Re-run a packet with the exact recorded code and local FTS profile."""

    with open_library(library, data_root=data_home) as repository:
        result = _recall_service(repository).replay(evidence_packet_id)
    _emit(
        _recall_result_payload(result),
        json_output=json_output,
        human_lines=(
            f"processing_run_id: {result.run.processing_run_id}",
            f"evidence_packet_id: {result.packet.evidence_packet_id}",
            f"packet_hash: {result.packet.packet_hash}",
            "replay_match: true",
        ),
    )


@packet_app.command("read-receipt")
@_guard
def packet_read_receipt(
    evidence_packet_id: Annotated[str, typer.Argument(help="EvidencePacket ID.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Show the exact protected SourceFragments materialized for a packet."""

    with open_library(library, data_root=data_home) as repository:
        receipt = (
            SQLiteRecallBackend(repository).load_evidence_packet(evidence_packet_id).read_receipt
        )
    payload = {
        **receipt.semantic_payload(),
        "read_receipt_id": receipt.read_receipt_id,
        "receipt_hash": receipt.receipt_hash,
    }
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"read_receipt_id: {receipt.read_receipt_id}",
            f"receipt_hash: {receipt.receipt_hash}",
            f"items: {len(receipt.items)}",
        ),
    )


@review_app.command("decide")
@_guard
def review_decide(
    target_id: Annotated[str, typer.Argument(help="Exact review target ID.")],
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    target_type: Annotated[
        ReviewTargetType,
        typer.Option("--target-type", help="EvidencePacket, packet item, or SourceFragment."),
    ],
    target_hash: Annotated[
        str,
        typer.Option("--target-hash", help="Current canonical hash for optimistic concurrency."),
    ],
    action: Annotated[ReviewAction, typer.Option("--action", help="Human review action.")],
    reason: Annotated[str, typer.Option("--reason", help="Bounded human rationale.")],
    authority: Annotated[
        str,
        typer.Option("--authority", help="Human or institutional review authority."),
    ],
    collection: Annotated[
        list[str],
        typer.Option("--collection", help="ReviewScope Collection ID; repeat as needed."),
    ],
    use: Annotated[str, typer.Option("--use", help="Canonical intended-use label.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    supersedes: Annotated[
        str | None,
        typer.Option("--supersedes", help="Prior ReviewDecision ID for a supersede action."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Append one hash-pinned ReviewDecision without mutating its target."""

    request = ReviewDecisionRequest(
        target_type=target_type,
        target_id=target_id,
        target_hash=target_hash,
        action=action,
        reason=reason,
        authority=authority,
        scope=ReviewScope(collection_ids=tuple(collection), use=use),
        supersedes_review_decision_id=supersedes,
    )
    with open_library(library, data_root=data_home) as repository:
        decision = ReviewService(repository).decide(request)
    payload = decision.payload()
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"review_decision_id: {decision.review_decision_id}",
            f"target: {decision.target_type.value}:{decision.target_id}",
            f"action: {decision.action.value}",
            f"scope_hash: {decision.scope.scope_hash}",
        ),
    )


@review_app.command("queue")
@_guard
def review_queue(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    target_type: Annotated[
        ReviewTargetType | None,
        typer.Option("--target-type", help="Optional exact target-type filter."),
    ] = None,
    target_id: Annotated[
        str | None,
        typer.Option("--target-id", help="Optional exact target ID; requires --target-type."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=500, help="Maximum decisions to return."),
    ] = 100,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """List ReviewDecisions in deterministic append order."""

    with open_library(library, data_root=data_home) as repository:
        decisions = ReviewService(repository).list(
            target_type=target_type,
            target_id=target_id,
            limit=limit,
        )
    payload: dict[str, object] = {
        "schema": "dithyramba.review_decision_list/1.0",
        "library_id": library,
        "items": [item.payload() for item in decisions],
    }
    _emit(
        payload,
        json_output=json_output,
        human_lines=tuple(
            f"{item.review_decision_id}\t{item.action.value}\t"
            f"{item.target_type.value}:{item.target_id}"
            for item in decisions
        )
        or ("No ReviewDecisions.",),
    )


@app.command("backup")
@_guard
def backup(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path | None,
        typer.Option(
            "--data-home",
            help="Absolute application-data root; defaults to the OS Dithyramba directory.",
        ),
    ] = None,
    output_directory: Annotated[
        Path | None,
        typer.Option(
            "--output-directory",
            help="Existing private directory for the complete BackupBundle.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Create and verify a complete portable single-Library BackupBundle."""

    with open_library(library, data_root=data_home) as repository:
        record = create_backup_bundle(repository, output_directory=output_directory)
    payload: dict[str, object] = {
        **record.manifest.payload(),
        "path": str(record.path),
        "manifest_hash": record.manifest_hash,
    }
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"library_id: {library}",
            f"path: {record.path}",
            f"manifest_hash: {record.manifest_hash}",
            f"files: {len(record.manifest.files)}",
        ),
    )


@app.command("restore")
@_guard
def restore(
    bundle: Annotated[Path, typer.Argument(help="Absolute verified BackupBundle path.")],
    data_home: Annotated[
        Path,
        typer.Option("--data-home", help="New isolated application-data root."),
    ],
    source_root: Annotated[
        list[Path] | None,
        typer.Option("--source-root", help="Declared source root; repeat as needed."),
    ] = None,
    sync_root: Annotated[
        list[Path] | None,
        typer.Option("--sync-root", help="Declared synchronization root; repeat as needed."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Verify and restore one BackupBundle without overwriting an existing Library."""

    verified = verify_backup_bundle(bundle)
    with restore_backup_bundle(
        bundle,
        data_root=data_home,
        declared_source_roots=tuple(source_root or ()),
        declared_sync_roots=tuple(sync_root or ()),
    ) as repository:
        repository.verify()
        payload = {
            **_library_payload(repository),
            "schema": "dithyramba.restore_receipt/1.0",
            "backup_bundle_id": verified.manifest.backup_bundle_id,
            "backup_manifest_hash": verified.manifest_hash,
            "schema_version": repository.schema_version,
            "status": "verified",
        }
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"library_id: {payload['library_id']}",
            f"path: {payload['path']}",
            f"status: {payload['status']}",
            f"backup_manifest_hash: {payload['backup_manifest_hash']}",
        ),
    )


@app.command("serve")
@_guard
def serve(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path,
        typer.Option("--data-home", help="Absolute Dithyramba application-data root."),
    ],
    port: Annotated[
        int,
        typer.Option("--port", min=1_024, max=65_535, help="Loopback TCP port."),
    ] = 8347,
) -> None:
    """Serve one Library on loopback with a fresh mutation bearer token."""

    with open_library(library, data_root=data_home) as repository:
        repository.verify()
    origin = f"http://127.0.0.1:{port}"
    application = create_app(
        library_id=library,
        data_home=data_home,
        allowed_origin=origin,
    )
    token = bearer_token_for(application)
    typer.echo(f"viewer: {origin}")
    typer.echo(f"mutation_bearer_token: {token}", err=True)
    uvicorn.run(
        application,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        server_header=False,
        date_header=False,
    )


@app.command("mcp")
@_guard
def mcp_stdio(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path,
        typer.Option("--data-home", help="Absolute Dithyramba application-data root."),
    ],
    agent_id: Annotated[
        str,
        typer.Option("--agent-id", help="Stable model actor ID for the session journal."),
    ] = "agent:mcp",
) -> None:
    """Expose the bounded research-session facade as a local MCP stdio server."""

    with open_library(library, data_root=data_home) as repository:
        repository.verify()
        run_stdio_mcp(
            repository,
            agent_id=agent_id,
            profile_version=current_fts_runtime_profile().profile_version,
        )


@lens_app.command("library")
@_guard
def reading_room(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    snapshot: Annotated[
        str,
        typer.Option("--snapshot", help="Immutable CorpusSnapshot ID."),
    ],
    access_policy: Annotated[
        str,
        typer.Option("--access-policy", help="Immutable AccessPolicy ID."),
    ],
    collection: Annotated[
        list[str],
        typer.Option(
            "--collection",
            help="Collection in the requested scope; repeat for each Collection.",
        ),
    ],
    purpose: Annotated[
        str,
        typer.Option("--purpose", help="Policy purpose, such as research."),
    ],
    data_home: Annotated[
        Path,
        typer.Option("--data-home", help="Absolute Dithyramba application-data root."),
    ],
    board_primary_packet: Annotated[
        str | None,
        typer.Option(
            "--board-primary-packet",
            help="Primary-source EvidencePacket for the optional comparison board.",
        ),
    ] = None,
    board_research_packet: Annotated[
        str | None,
        typer.Option(
            "--board-research-packet",
            help="Research-layer EvidencePacket for the optional comparison board.",
        ),
    ] = None,
    board_primary_collection: Annotated[
        list[str] | None,
        typer.Option(
            "--board-primary-collection",
            help="Primary Collection represented by the board packet; repeat as needed.",
        ),
    ] = None,
    board_research_collection: Annotated[
        list[str] | None,
        typer.Option(
            "--board-research-collection",
            help="Research Collection represented by the board packet; repeat as needed.",
        ),
    ] = None,
    board_question: Annotated[
        str | None,
        typer.Option(
            "--board-question",
            help="Human-facing research question shown above the evidence lanes.",
        ),
    ] = None,
    board_gap: Annotated[
        str | None,
        typer.Option(
            "--board-gap",
            help="Optional unresolved question shown without promoting it to evidence.",
        ),
    ] = None,
    exclude_source: Annotated[
        list[str] | None,
        typer.Option("--exclude-source", help="Excluded Source ID; repeat as needed."),
    ] = None,
    exclude_family: Annotated[
        list[str] | None,
        typer.Option("--exclude-family", help="Excluded SourceFamily ID; repeat as needed."),
    ] = None,
    exclude_fragment: Annotated[
        list[str] | None,
        typer.Option("--exclude-fragment", help="Excluded SourceFragment ID; repeat as needed."),
    ] = None,
    node_limit: Annotated[
        int,
        typer.Option("--node-limit", min=1, max=2_000, help="Maximum visible graph nodes."),
    ] = 500,
    edge_limit: Annotated[
        int,
        typer.Option("--edge-limit", min=1, max=5_000, help="Maximum visible graph edges."),
    ] = 1_000,
    corpus_fragment_limit: Annotated[
        int,
        typer.Option(
            "--corpus-fragment-limit",
            min=1,
            max=2_000,
            help="Maximum text-free corpus inventory rows.",
        ),
    ] = 200,
    port: Annotated[
        int,
        typer.Option("--port", min=1_024, max=65_535, help="Loopback TCP port."),
    ] = 8348,
) -> None:
    """Open a GET-only visual overview of one policy-scoped memory."""

    if _PURPOSE_PATTERN.fullmatch(purpose) is None:
        raise typer.BadParameter("purpose must match [a-z][a-z0-9_-]{0,63}")
    board_required = (
        board_primary_packet,
        board_research_packet,
        board_primary_collection,
        board_research_collection,
        board_question,
    )
    if any(item is not None for item in board_required) and not all(
        item is not None for item in board_required
    ):
        raise typer.BadParameter(
            "evidence board requires both packets, both Collection lanes, and a question"
        )
    board = (
        None
        if board_primary_packet is None
        else EvidenceBoardWebConfig(
            primary_packet_id=board_primary_packet,
            research_packet_id=board_research_packet or "",
            primary_collection_ids=tuple(board_primary_collection or ()),
            research_collection_ids=tuple(board_research_collection or ()),
            display_question=board_question or "",
            open_gap=board_gap,
        )
    )
    with open_library(library, data_root=data_home) as repository:
        repository.verify()
        corpus_snapshot = repository.get_corpus_snapshot(snapshot)
        scope = RequestScope(
            library_id=library,
            snapshot_hash=corpus_snapshot.manifest_hash,
            purpose=purpose,
            collection_ids=tuple(collection),
            exclusions=QueryExclusions(
                source_ids=tuple(exclude_source or ()),
                source_family_ids=tuple(exclude_family or ()),
                source_fragment_ids=tuple(exclude_fragment or ()),
            ),
        )
    limits = ReadingRoomLimits(
        node_limit=node_limit,
        edge_limit=edge_limit,
        corpus_fragment_limit=corpus_fragment_limit,
    )
    origin = f"http://127.0.0.1:{port}"
    application = create_lens_app(
        LensMode.LIBRARY,
        library_id=library,
        data_home=data_home,
        corpus_snapshot_id=snapshot,
        access_policy_id=access_policy,
        scope=scope,
        allowed_origin=origin,
        limits=limits,
        evidence_board=board,
    )
    typer.echo(f"lens: {origin}")
    typer.echo("lens_mode: library")
    typer.echo(f"projection_json: {origin}/projection.json")
    if board is not None:
        typer.echo(f"evidence_board: {origin}/evidence-board")
    typer.echo("mode: read-only")
    uvicorn.run(
        application,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        server_header=False,
        date_header=False,
    )


@lens_app.command("atlas")
@_guard
def research_atlas(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Absolute path to a validated Research Atlas manifest.",
        ),
    ],
    projection: Annotated[
        Path | None,
        typer.Option(
            "--projection",
            help=(
                "Optional hash-bound research projection with visible "
                "candidate and contextual material."
            ),
        ),
    ] = None,
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help="Optional absolute case root that authorizes manifest-listed local copies.",
        ),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", min=1_024, max=65_535, help="Loopback TCP port."),
    ] = 8350,
) -> None:
    """Open one immutable, case-neutral Research Atlas."""

    if not manifest.is_absolute():
        raise typer.BadParameter("manifest must be an absolute path")
    if projection is not None and not projection.is_absolute():
        raise typer.BadParameter("projection must be an absolute path")
    if artifact_root is not None and not artifact_root.is_absolute():
        raise typer.BadParameter("artifact-root must be an absolute path")
    origin = f"http://127.0.0.1:{port}"
    application = create_lens_app(
        LensMode.ATLAS,
        manifest_path=manifest,
        allowed_origin=origin,
        projection_path=projection,
        artifact_root=artifact_root,
    )
    config = application.state.research_atlas_config
    typer.echo(f"lens: {origin}")
    typer.echo("lens_mode: atlas")
    typer.echo(f"manifest_hash: {config.atlas.manifest_hash}")
    if config.projection is not None:
        typer.echo(f"projection_hash: {config.projection.manifest_hash}")
    typer.echo("mode: read-only")
    uvicorn.run(
        application,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        server_header=False,
        date_header=False,
    )


@lens_app.command("flow")
@_guard
def flow_view(
    port: Annotated[
        int,
        typer.Option("--port", min=1_024, max=65_535, help="Loopback TCP port."),
    ] = 8351,
) -> None:
    """Open the small, read-only map of Dithyramba's main data flow."""

    origin = f"http://127.0.0.1:{port}"
    application = create_lens_app(LensMode.FLOW, allowed_origin=origin)
    typer.echo(f"lens: {origin}")
    typer.echo("lens_mode: flow")
    typer.echo("mode: read-only static process map")
    uvicorn.run(
        application,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        server_header=False,
        date_header=False,
    )


@lens_app.command("concepts")
@_guard
def concept_lens(
    projection: Annotated[
        Path,
        typer.Option(
            "--projection",
            help="Absolute path to one validated candidate-ontology projection.",
        ),
    ],
    presentation: Annotated[
        Path | None,
        typer.Option(
            "--presentation",
            help="Optional ontology-bound JSON with human-facing cluster labels.",
        ),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", min=1_024, max=65_535, help="Loopback TCP port."),
    ] = 8352,
) -> None:
    """Open one scoped, read-only candidate ontology in Dithyramba Lens."""

    if not projection.is_absolute():
        raise typer.BadParameter("projection must be an absolute path")
    if presentation is not None and not presentation.is_absolute():
        raise typer.BadParameter("presentation must be an absolute path")
    origin = f"http://127.0.0.1:{port}"
    application = create_lens_app(
        LensMode.CONCEPTS,
        projection_path=projection,
        presentation_path=presentation,
        allowed_origin=origin,
    )
    config: ConceptLensWebConfig = application.state.concept_lens_config
    typer.echo(f"lens: {origin}")
    typer.echo("lens_mode: concepts")
    typer.echo(f"ontology_id: {config.ontology.manifest.ontology_id}")
    typer.echo(f"manifest_hash: {config.ontology.manifest_hash}")
    typer.echo("mode: candidate-only read-only projection")
    uvicorn.run(
        application,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        server_header=False,
        date_header=False,
    )


@lens_app.command("session")
@_guard
def session_lens(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    session: Annotated[
        str,
        typer.Option("--session", help="Durable ResearchSession ID."),
    ],
    data_home: Annotated[
        Path,
        typer.Option("--data-home", help="Absolute Dithyramba application-data root."),
    ],
    port: Annotated[
        int,
        typer.Option("--port", min=1_024, max=65_535, help="Loopback TCP port."),
    ] = 8353,
) -> None:
    """Open a GET-only working journal for one interactive research session."""

    with open_library(library, data_root=data_home) as repository:
        repository.verify()
        SQLiteResearchSessionRepository(repository).get_session(session)
    origin = f"http://127.0.0.1:{port}"
    application = create_lens_app(
        LensMode.SESSION,
        library_id=library,
        data_home=data_home,
        session_id=session,
        allowed_origin=origin,
    )
    typer.echo(f"lens: {origin}")
    typer.echo("lens_mode: session")
    typer.echo(f"projection_json: {origin}/projection.json")
    typer.echo("mode: read-only session projection")
    uvicorn.run(
        application,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        server_header=False,
        date_header=False,
    )


@app.command("reasoning-check")
@_guard
def reasoning_check(
    trace_path: Annotated[
        Path,
        typer.Option("--trace", help="Absolute path to one IdeaTrace/1.0 JSON artifact."),
    ],
    case_set_path: Annotated[
        Path,
        typer.Option(
            "--case-set",
            help="Absolute path to its exact ClaimEvidenceCaseSet JSON artifact.",
        ),
    ],
    entailment_path: Annotated[
        Path,
        typer.Option(
            "--entailment",
            help="Absolute path to the deterministic claim-evidence result JSON.",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete canonical closure receipt."),
    ] = False,
) -> None:
    """Verify one public reasoning trace without a provider or model call."""

    for label, path in (
        ("trace", trace_path),
        ("case-set", case_set_path),
        ("entailment", entailment_path),
    ):
        if not path.is_absolute():
            raise typer.BadParameter(f"{label} path must be absolute")
    trace = IdeaTrace.model_validate_json(trace_path.read_bytes())
    case_set = ClaimEvidenceCaseSet.model_validate_json(case_set_path.read_bytes())
    entailment = ClaimEvidenceEntailmentResult.model_validate_json(entailment_path.read_bytes())
    result = ReasoningClosureGate().evaluate(
        trace,
        case_set=case_set,
        entailment=entailment,
    )
    _emit(
        result.model_dump(mode="json"),
        json_output=json_output,
        human_lines=(
            f"decision: {result.decision.value}",
            f"trace_id: {result.trace_id}",
            f"closed_steps: {sum(item.closed for item in result.steps)}/{len(result.steps)}",
            f"review_eligible: {str(result.review_eligible).lower()}",
            "note: closure is structural; human acceptance remains separate",
        ),
    )
    if result.decision is not ReasoningClosureDecision.PASSED:
        raise typer.Exit(code=2)


@app.command("doctor")
@_guard
def doctor(
    library: Annotated[str, typer.Option("--library", help="Explicit Library ID.")],
    data_home: Annotated[
        Path,
        typer.Option("--data-home", help="Absolute Dithyramba application-data root."),
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Emit canonical JSON.")] = False,
) -> None:
    """Alias for ``library doctor``."""

    payload = _doctor_payload(library, data_home)
    _emit(
        payload,
        json_output=json_output,
        human_lines=(
            f"library_id: {payload['library_id']}",
            f"status: {payload['status']}",
            f"schema_version: {payload['schema_version']}",
        ),
    )


def _library_payload(repository: LibraryRepository) -> dict[str, object]:
    return _library_record_payload(repository.library)


def _library_record_payload(record: LibraryRecord) -> dict[str, object]:
    return {
        "created_at": record.created_at,
        "library_id": record.config.library_id,
        "logical_identity_hash": record.logical_identity_hash,
        "name": record.config.name,
        "path": str(record.paths.root),
    }


def _library_description_payload(
    description: LibraryDescription,
    *,
    schema_version: int,
    schema_fingerprint: str,
) -> dict[str, object]:
    if type(description) is not LibraryDescription:
        raise TypeError("Library description requires an exact LibraryDescription")
    counts = description.counts
    return {
        "schema": "dithyramba.library_description/1.0",
        "library": _library_record_payload(description.library),
        "health": {
            "status": "ok",
            "schema_version": schema_version,
            "schema_fingerprint": schema_fingerprint,
            "fts5_available": _fts5_available(),
            "external_services_contacted": False,
        },
        "counts": {
            "collections": counts.collections,
            "access_policies": counts.access_policies,
            "sources": counts.sources,
            "source_versions": counts.source_versions,
            "source_fragments": counts.source_fragments,
            "collection_memberships": counts.collection_memberships,
            "corpus_snapshots": counts.corpus_snapshots,
            "snapshot_members": counts.snapshot_members,
            "evidence_packets": counts.evidence_packets,
            "research_sessions": counts.research_sessions,
            "processing_runs": counts.processing_runs,
        },
        "collections": [_collection_payload(record) for record in description.collections],
        "access_policies": [_policy_payload(record) for record in description.access_policies],
        "snapshots": [
            {
                "corpus_snapshot_id": record.corpus_snapshot_id,
                "manifest_hash": record.manifest_hash,
                "collection_ids": list(record.collection_ids),
                "member_count": record.member_count,
                "created_at": record.created_at,
            }
            for record in description.snapshots
        ],
        "recent_processing_runs": [
            _processing_run_payload(record) for record in description.recent_processing_runs
        ],
        "backup_tracking": {
            "mode": "external_bundle",
            "registered_in_library": False,
        },
    }


def _processing_run_payload(record: ProcessingRunRecord) -> dict[str, object]:
    return {
        "processing_run_id": record.processing_run_id,
        "kind": record.kind,
        "status": record.status.value,
        "code_version": record.code_version,
        "profile_version": record.profile_version,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error_code": record.error_code,
        "output_hash": record.output_hash,
    }


def _collection_payload(record: CollectionRecord) -> dict[str, object]:
    roots = [
        {
            "collection_root_id": root_id,
            "exclude_globs": list(root.exclude_globs),
            "identity_manifest_path": (
                str(root.identity_manifest_path)
                if root.identity_manifest_path is not None
                else None
            ),
            "identity_manifest_sha256": root.identity_manifest_sha256,
            "include_globs": list(root.include_globs),
            "path": str(root.path),
        }
        for root_id, root in zip(record.collection_root_ids, record.config.roots, strict=True)
    ]
    return {
        "collection_id": record.config.collection_id,
        "created_at": record.created_at,
        "kind": record.config.kind.value,
        "library_id": record.config.library_id,
        "name": record.config.name,
        "roots": roots,
    }


def _source_payload(record: SourceRecord) -> dict[str, object]:
    """Serialize Source metadata without disclosing its absolute file URI."""

    return {
        "created_at": record.created_at,
        "current_source_version_id": record.current_source_version_id,
        "family_role": record.family_role.value,
        "library_id": record.library_id,
        "media_type": record.media_type,
        "memberships": [
            {
                "collection_id": membership.collection_id,
                "created_at": membership.created_at,
                "state": membership.state,
            }
            for membership in record.memberships
        ],
        "root_source_id": record.root_source_id,
        "source_family_id": record.source_family_id,
        "source_id": record.source_id,
        "title": record.title,
    }


def _source_version_payload(record: SourceVersionRecord) -> dict[str, object]:
    return {
        "byte_size": record.byte_size,
        "content_sha256": record.content_sha256,
        "failure_code": record.failure_code,
        "is_current": record.is_current,
        "observed_at": record.observed_at,
        "parse_status": record.parse_status,
        "parser_profile": record.parser_profile,
        "source_id": record.source_id,
        "source_modified_at": record.source_modified_at,
        "source_version_id": record.source_version_id,
        "version_number": record.version_number,
    }


def _recall_service(repository: LibraryRepository) -> RecallService:
    profile = current_fts_runtime_profile()
    return RecallService(
        SQLiteRecallBackend(repository),
        profile_version=profile.profile_version,
    )


def _recall_result_payload(result: RecallResult) -> dict[str, object]:
    run = result.run
    return {
        "schema": "dithyramba.recall_result/1.0",
        "run": {
            "processing_run_id": run.processing_run_id,
            "kind": run.kind,
            "status": run.status.value,
            "code_version": run.code_version,
            "profile_version": run.profile_version,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "error_code": run.error_code,
            "output_hash": run.output_hash,
        },
        "packet": result.packet.payload(),
    }


def _ingest_result_payload(
    result: IngestBatchResult,
    *,
    library_id: str,
    collection_id: str,
) -> dict[str, object]:
    run = result.run
    coverage = result.coverage
    return {
        "collection_id": collection_id,
        "coverage": {
            "coverage_report_id": coverage.coverage_report_id,
            "failed_count": coverage.failed_count,
            "omissions": [
                {
                    "category": omission.category,
                    "count": omission.count,
                    "disclosure": omission.disclosure,
                    "omission_id": omission.omission_id,
                    "reason_code": omission.reason_code,
                }
                for omission in coverage.omissions
            ],
            "policy_omission_present": coverage.policy_omission_present,
            "processed_count": coverage.processed_count,
            "processing_run_id": coverage.processing_run_id,
            "report_hash": coverage.report_hash,
            "skipped_count": coverage.skipped_count,
            "stage": coverage.stage,
        },
        "library_id": library_id,
        "outcomes": [outcome.payload() for outcome in result.outcomes],
        "run": {
            "code_version": run.code_version,
            "error_code": run.error_code,
            "finished_at": run.finished_at,
            "kind": run.kind,
            "output_hash": run.output_hash,
            "processing_run_id": run.processing_run_id,
            "profile_version": run.profile_version,
            "started_at": run.started_at,
            "status": run.status.value,
        },
        "schema": "dithyramba.ingest_result/1.0",
    }


def _emit_ingest_result(
    result: IngestBatchResult,
    *,
    library_id: str,
    collection_id: str,
    json_output: bool,
) -> None:
    payload = _ingest_result_payload(
        result,
        library_id=library_id,
        collection_id=collection_id,
    )
    coverage = result.coverage
    lines = [
        f"processing_run_id: {result.run.processing_run_id}",
        f"status: {result.run.status.value}",
        (
            "coverage: "
            f"processed={coverage.processed_count} "
            f"skipped={coverage.skipped_count} "
            f"failed={coverage.failed_count}"
        ),
    ]
    for outcome in result.outcomes:
        path = outcome.relative_path or "<root>"
        detail = (
            outcome.disposition.value
            if outcome.disposition is not None
            else outcome.failure_code or "unknown"
        )
        lines.append(f"{_human_text(path)}\t{outcome.terminal_outcome.value}\t{detail}")
    _emit(payload, json_output=json_output, human_lines=tuple(lines))
    if not result.succeeded:
        raise typer.Exit(code=1)


def _validate_collection_root_selection(
    collection: CollectionRecord,
    requested_root_id: str | None,
) -> None:
    root_ids = collection.collection_root_ids
    if not root_ids:
        raise ContractError("Collection has no roots")
    if requested_root_id is None and len(root_ids) != 1:
        raise ContractError("--collection-root is required for a multi-root Collection")
    if requested_root_id is not None and requested_root_id not in root_ids:
        raise ContractError("--collection-root is absent from the Collection")


def _policy_payload(record: AccessPolicyRecord) -> dict[str, object]:
    snapshot = record.snapshot
    return {
        "access_policy_id": snapshot.access_policy_id,
        "allow_export": snapshot.allow_export,
        "allow_external_provider": snapshot.allow_external_provider,
        "allowed_purposes": list(snapshot.allowed_purposes),
        "collection_rules": [
            {"collection_id": rule.collection_id, "effect": rule.effect.value}
            for rule in snapshot.collection_rules
        ],
        "created_at": record.created_at,
        "library_id": snapshot.library_id,
        "name": record.name,
        "policy_hash": snapshot.policy_hash,
        "source_rules": [
            {"effect": rule.effect.value, "source_id": rule.source_id}
            for rule in snapshot.source_rules
        ],
    }


def _policy_check_payload(
    policy: AccessPolicySnapshot,
    purpose: str,
    collection_ids: tuple[str, ...],
) -> dict[str, object]:
    effects = {rule.collection_id: rule.effect for rule in policy.collection_rules}
    decisions: list[dict[str, str]] = []
    for collection_id in collection_ids:
        effect = effects.get(collection_id)
        if effect is PolicyEffect.ALLOW:
            basis = "explicit_allow"
            decision = "allow"
        elif effect is PolicyEffect.DENY:
            basis = "explicit_deny"
            decision = "deny"
        else:
            basis = "default_deny"
            decision = "deny"
        decisions.append({"basis": basis, "collection_id": collection_id, "decision": decision})
    purpose_allowed = purpose in policy.allowed_purposes
    request_allowed = purpose_allowed and all(item["decision"] == "allow" for item in decisions)
    return {
        "access_policy_id": policy.access_policy_id,
        "collection_decisions": decisions,
        "default_deny": True,
        "library_id": policy.library_id,
        "purpose": purpose,
        "purpose_allowed": purpose_allowed,
        "request_allowed": request_allowed,
    }


def _doctor_payload(library_id: str, data_home: Path | None) -> dict[str, object]:
    with open_library(library_id, data_root=data_home) as repository:
        repository.verify()
        return {
            "fts5_available": _fts5_available(),
            "library_id": repository.library_id,
            "external_services_contacted": False,
            "path": str(repository.paths.root),
            "schema_fingerprint": repository.schema_fingerprint,
            "schema_version": repository.schema_version,
            "status": "ok",
        }


def _fts5_available() -> bool:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE fts5_probe USING fts5(value)")
    except sqlite3.DatabaseError:
        return False
    finally:
        connection.close()
    return True


def _emit(
    payload: dict[str, object],
    *,
    json_output: bool,
    human_lines: tuple[str, ...],
) -> None:
    if json_output:
        typer.echo(canonical_json_bytes(payload).decode("utf-8"))
        return
    typer.echo("\n".join(human_lines))


def _human_text(value: str) -> str:
    """Render untrusted operator/source text without terminal control sequences."""

    return canonical_json_bytes(value).decode("utf-8")
