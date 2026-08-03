"""Focused contract and fail-closed edge coverage for the loopback API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from dithyramba.api import bearer_token_for
from dithyramba.api.config import LoopbackApiConfig, build_config
from dithyramba.api.models import address_response
from dithyramba.api.services import PacketProjection, PacketViewService, ViewerNotFoundError
from dithyramba.ingest import PdfSourceAddress, SourceAddress
from dithyramba.interactive import AgentEvidencePacket
from dithyramba.persistence import SQLiteRecallBackend
from dithyramba.persistence.repository import open_library

from .support import ApiWorld, make_test_app, mutation_headers


def _review_payload(world: ApiWorld) -> dict[str, object]:
    return {
        "schema": "dithyramba.review_decision_request/1.0",
        "target_type": "evidence_packet",
        "target_id": world.primary.evidence_packet_id,
        "target_hash": world.primary.packet_hash,
        "action": "accept",
        "reason": "Exact packet review.",
        "authority": "researcher:test",
        "scope": {
            "schema": "dithyramba.review_scope/1.0",
            "collection_ids": [world.primary.collection_id],
            "use": "research",
        },
        "supersedes_review_decision_id": None,
    }


def test_configuration_is_explicit_canonical_and_loopback_only(api_world: ApiWorld) -> None:
    library_id = api_world.primary.library_id
    production = build_config(
        library_id=library_id,
        data_home=str(api_world.data_home),
        allowed_origin="http://localhost:4312",
    )
    assert production.data_home.is_absolute()
    assert production.allowed_origin == "http://localhost:4312"
    assert production.trusted_hosts == ("127.0.0.1", "localhost")

    with pytest.raises(ValueError, match="absolute Path"):
        LoopbackApiConfig(
            library_id=library_id,
            data_home=Path("relative"),
            allowed_origin="http://localhost",
        )
    with pytest.raises(TypeError, match="must be a bool"):
        LoopbackApiConfig(
            library_id=library_id,
            data_home=api_world.data_home,
            allowed_origin="http://localhost",
            test_only_allow_testserver=cast(Any, 1),
        )
    for invalid_limit in (1_023, 1024 * 1024 + 1, True):
        with pytest.raises(ValueError, match="between 1 KiB and 1 MiB"):
            build_config(
                library_id=library_id,
                data_home=api_world.data_home,
                allowed_origin="http://localhost",
                max_request_body_bytes=cast(Any, invalid_limit),
            )

    invalid_origins = (
        "",
        "http://localhost:not-a-port",
        "http://attacker.example",
        "http://localhost/path",
        "HTTP://localhost",
        "http://localhost:4312/",
    )
    for invalid_origin in invalid_origins:
        with pytest.raises(ValueError):
            build_config(
                library_id=library_id,
                data_home=api_world.data_home,
                allowed_origin=invalid_origin,
            )


def test_address_projection_supports_pdf_and_rejects_unknown_types() -> None:
    pdf = address_response(
        PdfSourceAddress(
            page=2,
            bbox=("1.000", "2.000", "30.000", "40.000"),
            char_start=10,
            char_end=20,
        )
    )
    assert pdf.model_dump(mode="json", by_alias=True) == {
        "schema": "dithyramba.source_address/1.0",
        "kind": "pdf",
        "page": 2,
        "bbox": ["1.000", "2.000", "30.000", "40.000"],
        "char_start": 10,
        "char_end": 20,
    }
    with pytest.raises(TypeError, match="unsupported SourceAddress"):
        address_response(cast(SourceAddress, object()))


def test_invalid_domain_request_conflict_and_missing_token_are_generic(
    api_world: ApiWorld,
) -> None:
    app, token = make_test_app(api_world)
    payload = _review_payload(api_world)
    with TestClient(app) as client:
        invalid_domain = client.post(
            "/v1/review-decisions",
            json={**payload, "scope": {"collection_ids": [], "use": "research"}},
            headers=mutation_headers(token),
        )
        stale_hash = client.post(
            "/v1/review-decisions",
            json={**payload, "target_hash": "0" * 64},
            headers=mutation_headers(token),
        )
        assert invalid_domain.status_code == 422
        assert invalid_domain.json() == {"detail": "invalid request"}
        assert stale_hash.status_code == 409
        assert stale_hash.json() == {"detail": "conflict"}

    app.state.loopback_bearer_token = 7
    with pytest.raises(RuntimeError, match="token is unavailable"):
        bearer_token_for(app)


def test_inactive_lifespan_and_unexpected_viewer_failure_are_generic(
    api_world: ApiWorld,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inactive_app, _token = make_test_app(api_world)
    inactive_client = TestClient(inactive_app, raise_server_exceptions=False)
    inactive = inactive_client.get("/health")
    inactive_client.close()
    assert inactive.status_code == 500
    assert inactive.json() == {"detail": "internal error"}

    active_app, _token = make_test_app(api_world)

    def explode(_name: str) -> None:
        raise RuntimeError("sensitive implementation detail")

    monkeypatch.setattr("dithyramba.api.app._TEMPLATES.get_template", explode)
    source = api_world.primary
    with TestClient(active_app, raise_server_exceptions=False) as client:
        response = client.get(
            f"/sources/{source.source_id}/versions/{source.source_version_id}",
            params={
                "packet": source.evidence_packet_id,
                "fragment": source.source_fragment_id,
            },
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "internal error"}
    assert "sensitive" not in response.text
    assert response.headers["content-security-policy"] == (
        "default-src 'self'; script-src 'none'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'"
    )


def test_packet_view_service_rejects_every_inconsistent_projection(
    api_world: ApiWorld,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = api_world.primary
    with open_library(source.library_id, data_root=api_world.data_home) as repository:
        service = PacketViewService(repository, SQLiteRecallBackend(repository))
        projection = service.load_packet(source.evidence_packet_id)
        compact = AgentEvidencePacket.create(projection.packet)
        compact_projection = service.load_agent_packet(compact)
        assert compact_projection.packet == compact
        assert compact_projection.source_chips == projection.source_chips
        with pytest.raises(TypeError, match="exact AgentEvidencePacket"):
            service.load_agent_packet(cast(AgentEvidencePacket, object()))
        with pytest.raises(ViewerNotFoundError, match="compact binding"):
            service.load_agent_packet(compact.model_copy(update={"evidence_packet_hash": "b" * 64}))
        fragment = next(
            item
            for item in projection.packet.source_fragments
            if item.source_fragment_id == source.source_fragment_id
        )
        chip = next(
            item
            for item in projection.source_chips
            if item.source_fragment_id == source.source_fragment_id
        )

        absent_fragment = fragment.model_copy(update={"source_fragment_id": "fragment_absent"})
        with pytest.raises(ViewerNotFoundError, match="does not resolve"):
            service._chip_for_fragment(projection.packet, absent_fragment)

        mismatched_fragment = fragment.model_copy(
            update={"source_version_id": "source_version_absent"}
        )
        with pytest.raises(ViewerNotFoundError, match="inconsistent"):
            service._chip_for_fragment(projection.packet, mismatched_fragment)

        monkeypatch.setattr(
            service,
            "load_packet",
            lambda _packet_id: PacketProjection(
                packet=projection.packet,
                source_chips=(),
            ),
        )
        with pytest.raises(ViewerNotFoundError, match="does not exist"):
            service.load_viewer(
                source_id=source.source_id,
                source_version_id=source.source_version_id,
                source_fragment_id=source.source_fragment_id,
                evidence_packet_id=source.evidence_packet_id,
            )

        wrong_chip = chip.model_copy(update={"source_id": "source_absent"})
        monkeypatch.setattr(
            service,
            "load_packet",
            lambda _packet_id: PacketProjection(
                packet=projection.packet,
                source_chips=tuple(
                    wrong_chip if item.source_fragment_id == source.source_fragment_id else item
                    for item in projection.source_chips
                ),
            ),
        )
        with pytest.raises(ViewerNotFoundError, match="does not exist"):
            service.load_viewer(
                source_id=source.source_id,
                source_version_id=source.source_version_id,
                source_fragment_id=source.source_fragment_id,
                evidence_packet_id=source.evidence_packet_id,
            )
