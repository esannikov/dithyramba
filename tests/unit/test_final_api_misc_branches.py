"""Final fail-closed branch coverage for API projection and store boundaries."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Message, Receive, Scope, Send

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.api.app import (
    _packet_markdown,
    _packet_presentation,
    _packet_review_decisions,
    _validate_collection_root,
)
from dithyramba.api.security import LoopbackSecurityMiddleware
from dithyramba.api.services import PacketProjection, PacketViewService, ViewerNotFoundError
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import LibraryRepository, SQLiteRecallBackend, initialize_library
from dithyramba.persistence.errors import SourceVersionNotFoundError
from dithyramba.provenance import (
    IngestInputOutcome,
    SourceFragmentRecord,
    SourceRecord,
    SourceVersionRecord,
    TerminalInputOutcome,
)
from dithyramba.recall import (
    EvidencePacket,
    QueryRequest,
    RecallService,
    current_fts_runtime_profile,
)
from dithyramba.review import (
    ReviewAction,
    ReviewDecision,
    ReviewScope,
    ReviewTargetType,
)
from dithyramba.store import Store
from dithyramba.store import database as database_module
from dithyramba.store.errors import StoreOpenError


def _failed_outcome_payload() -> dict[str, object]:
    return IngestInputOutcome(
        collection_root_id="root_alpha",
        relative_path="source.md",
        terminal_outcome=TerminalInputOutcome.FAILED,
        failure_code="parse_failed",
    ).payload()


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    [
        ({"terminal_outcome": "failed"}, TypeError, "TerminalInputOutcome"),
        ({"disposition": "added"}, TypeError, "IngestDisposition"),
        ({"source_id": ""}, ValueError, "source_id must be non-empty"),
    ],
)
def test_ingest_outcome_runtime_values_fail_closed_before_claiming_provenance(
    changes: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "collection_root_id": "root_alpha",
        "relative_path": "source.md",
        "terminal_outcome": TerminalInputOutcome.FAILED,
        "failure_code": "parse_failed",
        **changes,
    }

    with pytest.raises(error_type, match=message):
        IngestInputOutcome(**cast(Any, values))


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        ([], TypeError, "payload must be an object"),
        (
            {**_failed_outcome_payload(), "terminal_outcome": 1},
            TypeError,
            "terminal_outcome payload value must be text",
        ),
        (
            {**_failed_outcome_payload(), "terminal_outcome": "unknown"},
            ValueError,
            "terminal_outcome payload value is invalid",
        ),
        (
            {**_failed_outcome_payload(), "disposition": 1},
            TypeError,
            "disposition payload value must be text or null",
        ),
        (
            {**_failed_outcome_payload(), "disposition": "unknown"},
            ValueError,
            "disposition payload value is invalid",
        ),
    ],
)
def test_ingest_outcome_payload_rejects_untyped_and_unknown_enums(
    payload: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        IngestInputOutcome.from_payload(payload)


class _PacketRepositoryStub:
    def __init__(
        self,
        *,
        source_version: SourceVersionRecord | Exception,
        fragment_metadata: tuple[SourceFragmentRecord, ...],
        source: SourceRecord,
    ) -> None:
        self.source_version = source_version
        self.fragment_metadata = fragment_metadata
        self.source = source

    def get_source_version(self, _source_version_id: str) -> SourceVersionRecord:
        if isinstance(self.source_version, Exception):
            raise self.source_version
        return self.source_version

    def list_source_fragments(
        self,
        _source_version_id: str,
    ) -> tuple[SourceFragmentRecord, ...]:
        return self.fragment_metadata

    def get_source(self, _source_id: str) -> SourceRecord:
        return self.source


def _stub_packet_service(repository: _PacketRepositoryStub) -> PacketViewService:
    return PacketViewService(
        cast(LibraryRepository, repository),
        cast(SQLiteRecallBackend, object()),
    )


def test_packet_chip_projection_rejects_missing_metadata_hash_drift_and_lineage(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "evidence.md").write_text(
        "# Evidence\n\nThe oracular fragment is exact.\n",
        encoding="utf-8",
    )
    with initialize_library(
        LibraryConfig(name="Packet branch test"),
        data_root=tmp_path / "data",
    ) as repository:
        collection = repository.create_collection(
            CollectionConfig(
                library_id=repository.library_id,
                name="Evidence",
                kind=CollectionKind.CORPUS,
                roots=(build_collection_root(source_root, data_root=tmp_path / "data"),),
            )
        )
        IngestService(repository).ingest_path(
            collection.config.collection_id,
            "evidence.md",
        )
        snapshot = repository.freeze_snapshot((collection.config.collection_id,))
        policy = AccessPolicySnapshot(
            access_policy_id="policy_research",
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),),
        )
        repository.persist_access_policy(name="Research", snapshot=policy)
        request = QueryRequest(
            question="oracular",
            library_id=repository.library_id,
            collection_ids=(collection.config.collection_id,),
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            purpose="research",
        )
        packet = (
            RecallService(
                SQLiteRecallBackend(repository),
                profile_version=current_fts_runtime_profile().profile_version,
            )
            .recall(request)
            .packet
        )
        service = PacketViewService(repository, SQLiteRecallBackend(repository))
        projection = service.load_packet(packet.evidence_packet_id)
        fragment = projection.packet.source_fragments[0]
        source_version = repository.get_source_version(fragment.source_version_id)
        metadata = next(
            item
            for item in repository.list_source_fragments(fragment.source_version_id)
            if item.source_fragment_id == fragment.source_fragment_id
        )
        source = repository.get_source(source_version.source_id)

        missing_version = _PacketRepositoryStub(
            source_version=SourceVersionNotFoundError("missing"),
            fragment_metadata=(metadata,),
            source=source,
        )
        with pytest.raises(ViewerNotFoundError, match="does not resolve"):
            _stub_packet_service(missing_version)._chip_for_fragment(projection.packet, fragment)

        missing_metadata = _PacketRepositoryStub(
            source_version=source_version,
            fragment_metadata=(),
            source=source,
        )
        with pytest.raises(ViewerNotFoundError, match="does not resolve"):
            _stub_packet_service(missing_metadata)._chip_for_fragment(projection.packet, fragment)

        corrupt_metadata = _PacketRepositoryStub(
            source_version=source_version,
            fragment_metadata=(replace(metadata, text_sha256="0" * 64),),
            source=source,
        )
        with pytest.raises(ViewerNotFoundError, match="provenance is inconsistent"):
            _stub_packet_service(corrupt_metadata)._chip_for_fragment(projection.packet, fragment)

        wrong_lineage = _PacketRepositoryStub(
            source_version=source_version,
            fragment_metadata=(metadata,),
            source=replace(source, source_family_id="family_wrong"),
        )
        with pytest.raises(ViewerNotFoundError, match="lineage is inconsistent"):
            _stub_packet_service(wrong_lineage)._chip_for_fragment(projection.packet, fragment)


def test_collection_root_gate_rejects_absent_and_ambiguous_implicit_roots() -> None:
    with pytest.raises(StarletteHTTPException) as absent:
        _validate_collection_root((), None)
    with pytest.raises(StarletteHTTPException) as ambiguous:
        _validate_collection_root(("root_alpha", "root_beta"), None)

    assert absent.value.status_code == 422
    assert ambiguous.value.status_code == 422


def _review_decision() -> ReviewDecision:
    return ReviewDecision(
        review_decision_id="review_alpha",
        library_id="library_alpha",
        target_type=ReviewTargetType.EVIDENCE_PACKET,
        target_id="packet_alpha",
        target_hash="a" * 64,
        action=ReviewAction.ACCEPT,
        reason="The empty packet was reviewed in its exact scope.",
        authority="researcher:test",
        scope=ReviewScope(collection_ids=("collection_alpha",), use="research"),
        created_at="2026-07-21T00:00:00.000000Z",
    )


def test_reviewed_empty_packet_markdown_is_explicit_and_deterministic() -> None:
    decision = _review_decision()
    packet = SimpleNamespace(
        evidence_packet_id="packet_alpha",
        packet_hash="b" * 64,
        result_status=SimpleNamespace(value="no_evidence"),
        source_fragments=(),
    )
    projection = PacketProjection(
        packet=cast(EvidencePacket, packet),
        source_chips=(),
    )
    presentation = _packet_presentation((decision,))

    rendered = _packet_markdown(projection, (decision,), presentation)

    assert "Review status: `has_decisions`" in rendered
    assert "`review_alpha` — `accept`" in rendered
    assert "Collections `collection_alpha`" in rendered
    assert "No evidence fragments were selected." in rendered
    assert rendered.endswith("\n")


def test_packet_review_collection_deduplicates_decisions_across_exact_targets() -> None:
    decision = _review_decision()

    class Reviews:
        def list(self, **_kwargs: object) -> tuple[ReviewDecision, ...]:
            return (decision,)

        def packet_item_targets(self, _packet_id: str) -> tuple[SimpleNamespace, ...]:
            return (SimpleNamespace(target_id="packet_item_alpha"),)

    services = SimpleNamespace(reviews=Reviews())
    packet = SimpleNamespace(
        evidence_packet_id="packet_alpha",
        source_fragments=(SimpleNamespace(source_fragment_id="fragment_alpha"),),
    )
    projection = PacketProjection(packet=cast(EvidencePacket, packet), source_chips=())

    assert _packet_review_decisions(cast(Any, services), projection) == (decision,)


def test_store_open_preserves_existing_file_when_connect_never_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite_failure = tmp_path / "sqlite-failure.sqlite3"
    sqlite_failure.write_bytes(b"")
    sqlite_failure.chmod(0o600)

    def fail_sqlite_connect(*_args: object, **_kwargs: object) -> NoReturn:
        raise sqlite3.OperationalError("connect failed")

    monkeypatch.setattr("dithyramba.store.database.sqlite3.connect", fail_sqlite_connect)
    with pytest.raises(StoreOpenError, match="could not open"):
        Store.open(sqlite_failure)
    assert sqlite_failure.exists()

    generic_failure = tmp_path / "generic-failure.sqlite3"
    generic_failure.write_bytes(b"")
    generic_failure.chmod(0o600)

    def fail_generic_connect(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("unexpected connector failure")

    monkeypatch.setattr("dithyramba.store.database.sqlite3.connect", fail_generic_connect)
    with pytest.raises(RuntimeError, match="unexpected connector failure"):
        Store.open(generic_failure)
    assert generic_failure.exists()


def test_database_file_security_failures_remove_only_new_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_failure = tmp_path / "create-failure.sqlite3"
    with monkeypatch.context() as patch:
        patch.setattr(
            "dithyramba.store.database.os.open",
            lambda *_args, **_kwargs: _raise_os_error(),
        )
        with pytest.raises(StoreOpenError, match="could not securely create"):
            database_module._prepare_database_file(create_failure)
    assert not create_failure.exists()

    wrong_owner = tmp_path / "wrong-owner.sqlite3"
    actual_euid = os.geteuid()
    with monkeypatch.context() as patch:
        patch.setattr("dithyramba.store.database.os.geteuid", lambda: actual_euid + 1)
        with pytest.raises(StoreOpenError, match="different owner"):
            database_module._prepare_database_file(wrong_owner)
    assert not wrong_owner.exists()

    non_private = tmp_path / "non-private.sqlite3"
    non_private.write_bytes(b"")
    non_private.chmod(0o644)
    original_chmod = Path.chmod

    def ignore_target_chmod(
        path: Path,
        mode: int,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        if path != non_private:
            original_chmod(path, mode, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "chmod", ignore_target_chmod)
        with pytest.raises(StoreOpenError, match="permissions are not private"):
            database_module._prepare_database_file(non_private)
    assert non_private.exists()


def _raise_os_error() -> NoReturn:
    raise OSError("secure creation unavailable")


def _http_scope() -> Scope:
    return cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": [(b"host", b"localhost")],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 4312),
        },
    )


def _security_middleware(app: Any) -> LoopbackSecurityMiddleware:
    return LoopbackSecurityMiddleware(
        app,
        bearer_token="secret",
        allowed_origin="http://localhost:4312",
        trusted_hosts=("localhost",),
        max_request_body_bytes=16,
    )


def test_security_middleware_passes_disconnect_and_never_double_starts_errors() -> None:
    disconnect_seen = False
    ordinary_messages: list[Message] = []

    async def receive_disconnect() -> Message:
        return {"type": "http.disconnect"}

    async def capture_ordinary(message: Message) -> None:
        ordinary_messages.append(message)

    async def consume_disconnect(_scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal disconnect_seen
        disconnect_seen = (await receive())["type"] == "http.disconnect"
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    asyncio.run(
        _security_middleware(consume_disconnect)(
            _http_scope(),
            receive_disconnect,
            capture_ordinary,
        )
    )
    assert disconnect_seen is True
    assert ordinary_messages[0]["type"] == "http.response.start"

    started_messages: list[Message] = []

    async def capture_started(message: Message) -> None:
        started_messages.append(message)

    async def start_then_fail(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"set-cookie", b"forbidden=1"),
                    (b"access-control-allow-origin", b"*"),
                ],
            }
        )
        raise RuntimeError("stream failed after response start")

    with pytest.raises(RuntimeError, match="after response start"):
        asyncio.run(
            _security_middleware(start_then_fail)(
                _http_scope(),
                receive_disconnect,
                capture_started,
            )
        )

    assert [message["type"] for message in started_messages] == ["http.response.start"]
    headers = started_messages[0].get("headers", [])
    assert all(name != b"set-cookie" for name, _value in headers)
    assert all(not name.startswith(b"access-control-") for name, _value in headers)
