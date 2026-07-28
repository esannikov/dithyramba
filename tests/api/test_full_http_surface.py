"""Frozen P4 HTTP surface, canonical parity, export, and read-only HTML views."""

from __future__ import annotations

import json
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from dithyramba.cli import app as cli_app
from dithyramba.persistence import AccessPolicyRecord, LibraryRepository

from .support import ApiWorld, make_test_app, mutation_headers


class _ExactHighlightParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.highlights: list[dict[str, str | None]] = []
        self.forbidden: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if attributes.get("data-source-highlight") == "true":
            self.highlights.append(attributes)
        if tag in {"script", "img", "form", "button", "input"}:
            self.forbidden.append(tag)


def _export_request(export_format: str) -> dict[str, str]:
    return {
        "schema": "dithyramba.packet_export_request/1.0",
        "format": export_format,
    }


def test_library_collection_source_snapshot_and_version_surface(
    api_world: ApiWorld,
    tmp_path: Path,
) -> None:
    app, token = make_test_app(api_world)
    root = tmp_path / "api-created-root"
    root.mkdir()
    (root / "created.md").write_text(
        "# Created through API\n\nExact API source.\n",
        encoding="utf-8",
    )
    collection_request: dict[str, object] = {
        "name": "API-created corpus",
        "kind": "corpus",
        "roots": [
            {
                "path": str(root),
                "include_globs": ["**/*.md"],
                "exclude_globs": [],
            }
        ],
    }

    with TestClient(app) as client:
        libraries = client.get("/v1/libraries")
        assert libraries.status_code == 200
        assert len(libraries.json()["libraries"]) == 1
        assert libraries.json()["libraries"][0]["library_id"] == (api_world.primary.library_id)
        assert api_world.foreign.library_id not in libraries.text

        collections = client.get(f"/v1/libraries/{api_world.primary.library_id}/collections")
        foreign_collections = client.get(
            f"/v1/libraries/{api_world.foreign.library_id}/collections"
        )
        assert collections.status_code == 200
        assert collections.json()["library_id"] == api_world.primary.library_id
        assert [item["collection_id"] for item in collections.json()["collections"]] == [
            api_world.primary.collection_id
        ]
        assert foreign_collections.status_code == 404
        assert foreign_collections.json() == {"detail": "not found"}

        no_bootstrap = client.post(
            "/v1/libraries",
            json={
                "library_id": api_world.primary.library_id,
                "name": api_world.primary.library_name,
            },
            headers=mutation_headers(token),
        )
        assert no_bootstrap.status_code == 405
        assert no_bootstrap.json() == {"detail": "method not allowed"}

        created = client.post(
            f"/v1/libraries/{api_world.primary.library_id}/collections",
            json=collection_request,
            headers=mutation_headers(token),
        )
        assert created.status_code == 201
        collection = created.json()
        assert collection["library_id"] == api_world.primary.library_id
        assert collection["name"] == "API-created corpus"
        assert collection["roots"] == [
            {
                "collection_root_id": collection["roots"][0]["collection_root_id"],
                "exclude_globs": [],
                "include_globs": ["**/*.md"],
                "path": str(root.resolve()),
            }
        ]

        ingest = client.post(
            "/v1/sources",
            json={
                "library_id": api_world.primary.library_id,
                "collection_id": collection["collection_id"],
                "relative_path": "created.md",
                "collection_root_id": collection["roots"][0]["collection_root_id"],
            },
            headers=mutation_headers(token),
        )
        assert ingest.status_code == 200
        result = ingest.json()
        assert result["schema"] == "dithyramba.ingest_result/1.0"
        assert result["run"]["status"] == "succeeded"
        assert result["coverage"]["processed_count"] == 1
        assert result["outcomes"][0]["terminal_outcome"] == "processed"
        source_id = result["outcomes"][0]["source_id"]
        assert isinstance(source_id, str)

        versions = client.get(f"/v1/sources/{source_id}/versions")
        assert versions.status_code == 200
        assert versions.json()["library_id"] == api_world.primary.library_id
        assert versions.json()["source"]["source_id"] == source_id
        assert versions.json()["versions"][0]["is_current"] is True
        assert "canonical_uri" not in versions.text

        snapshot = client.post(
            f"/v1/libraries/{api_world.primary.library_id}/collections/"
            f"{collection['collection_id']}/snapshots",
            headers=mutation_headers(token),
        )
        assert snapshot.status_code == 200
        assert snapshot.json()["schema"] == "dithyramba.corpus_snapshot_manifest/1.0"
        assert snapshot.json()["scope"] == {
            "schema": "dithyramba.corpus_snapshot_scope/1.0",
            "library_id": api_world.primary.library_id,
            "collection_ids": [collection["collection_id"]],
        }

        wrong_library = client.post(
            f"/v1/libraries/{api_world.foreign.library_id}/collections",
            json=collection_request,
            headers=mutation_headers(token),
        )
        wrong_root = client.post(
            "/v1/sources",
            json={
                "library_id": api_world.primary.library_id,
                "collection_id": api_world.primary.collection_id,
                "relative_path": "evidence.md",
                "collection_root_id": "root_absent",
            },
            headers=mutation_headers(token),
        )
        assert wrong_library.status_code == 404
        assert wrong_library.json() == {"detail": "not found"}
        assert wrong_root.status_code == 422
        assert wrong_root.json() == {"detail": "invalid request"}


def test_packet_receipt_recall_and_replay_are_canonical_cli_shapes(
    api_world: ApiWorld,
) -> None:
    app, token = make_test_app(api_world)
    primary = api_world.primary
    with TestClient(app) as client:
        packet = client.get(f"/v1/evidence-packets/{primary.evidence_packet_id}")
        receipt = client.get(f"/v1/evidence-packets/{primary.evidence_packet_id}/read-receipt")
        recall = client.post(
            "/v1/recall",
            json=primary.query_payload,
            headers=mutation_headers(token),
        )
        replay = client.post(
            f"/v1/evidence-packets/{primary.evidence_packet_id}/replay",
            headers=mutation_headers(token),
        )

        assert packet.json() == primary.packet_payload
        assert receipt.json() == primary.read_receipt_payload
        assert recall.status_code == 200
        assert recall.json()["schema"] == "dithyramba.recall_result/1.0"
        assert recall.json()["packet"] == primary.packet_payload
        assert replay.status_code == 200
        assert replay.json()["schema"] == "dithyramba.recall_result/1.0"
        assert replay.json()["packet"] == primary.packet_payload

        incomplete = client.post(
            "/v1/recall",
            json={
                "schema": "dithyramba.query_request/1.0",
                "question": "oracular",
                "library_id": primary.library_id,
            },
            headers=mutation_headers(token),
        )
        foreign = client.post(
            "/v1/recall",
            json={**primary.query_payload, "library_id": api_world.foreign.library_id},
            headers=mutation_headers(token),
        )
        assert incomplete.status_code == 422
        assert foreign.status_code == 404
        assert foreign.json() == {"detail": "not found"}

    cli = CliRunner().invoke(
        cli_app,
        [
            "packet",
            "inspect",
            primary.evidence_packet_id,
            "--library",
            primary.library_id,
            "--data-home",
            str(api_world.data_home),
            "--json",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.stdout) == primary.packet_payload


def test_every_emitted_source_chip_resolves_exact_packet_fragment(
    api_world: ApiWorld,
) -> None:
    app, _token = make_test_app(api_world)
    primary = api_world.primary
    with TestClient(app) as client:
        packet = client.get(f"/v1/evidence-packets/{primary.evidence_packet_id}").json()
        chips_response = client.get(
            f"/v1/evidence-packets/{primary.evidence_packet_id}/source-chips"
        )
        assert chips_response.status_code == 200
        chips = chips_response.json()["items"]
        assert chips
        assert len(chips) == len(packet["source_fragments"])

        fragments = {item["source_fragment_id"]: item for item in packet["source_fragments"]}
        for chip in chips:
            fragment_id = chip["source_fragment_id"]
            fragment = client.get(
                f"/v1/source-fragments/{fragment_id}",
                params={"packet": primary.evidence_packet_id},
            )
            assert fragment.status_code == 200
            assert fragment.json()["evidence_packet_id"] == primary.evidence_packet_id
            assert fragment.json()["packet_hash"] == primary.packet_hash
            assert fragment.json()["fragment"] == fragments[fragment_id]
            assert fragment.json()["source_chip"] == chip
            assert chip["source_family_id"]
            assert chip["root_source_id"]

            viewer = client.get(chip["viewer_url"])
            assert viewer.status_code == 200
            parser = _ExactHighlightParser()
            parser.feed(viewer.text)
            assert parser.forbidden == []
            assert len(parser.highlights) == 1
            attributes = parser.highlights[0]
            assert attributes["data-source-version-id"] == chip["source_version_id"]
            assert attributes["data-source-fragment-id"] == fragment_id
            assert attributes["data-text-sha256"] == chip["text_sha256"]
            assert (
                json.loads(attributes["data-source-address"] or "null") == (chip["source_address"])
            )

        same_fragment_other_packet = client.get(
            f"/v1/source-fragments/{primary.source_fragment_id}",
            params={"packet": primary.denied_evidence_packet_id},
        )
        assert same_fragment_other_packet.status_code == 200
        assert same_fragment_other_packet.json()["evidence_packet_id"] == (
            primary.denied_evidence_packet_id
        )
        assert (
            f"packet={primary.denied_evidence_packet_id}"
            in same_fragment_other_packet.json()["source_chip"]["viewer_url"]
        )

        missing_packet_gate = client.get(f"/v1/source-fragments/{primary.source_fragment_id}")
        foreign_packet = client.get(
            f"/v1/source-fragments/{primary.source_fragment_id}",
            params={"packet": api_world.foreign.evidence_packet_id},
        )
        assert missing_packet_gate.status_code == 422
        assert foreign_packet.status_code == 404
        assert foreign_packet.json() == {"detail": "not found"}


def test_packet_exports_require_exact_authorization_and_are_inert(
    api_world: ApiWorld,
) -> None:
    app, token = make_test_app(api_world)
    primary = api_world.primary
    allowed_path = f"/v1/evidence-packets/{primary.evidence_packet_id}/exports"
    denied_path = f"/v1/evidence-packets/{primary.denied_evidence_packet_id}/exports"
    with TestClient(app) as client:
        unauthenticated = client.post(
            allowed_path,
            json=_export_request("json"),
        )
        denied = client.post(
            denied_path,
            json=_export_request("json"),
            headers=mutation_headers(token),
        )
        exported_json = client.post(
            allowed_path,
            json=_export_request("json"),
            headers=mutation_headers(token),
        )
        exported_markdown = client.post(
            allowed_path,
            json=_export_request("markdown"),
            headers=mutation_headers(token),
        )
        read_only_get = client.get(allowed_path)

        assert unauthenticated.status_code == 401
        assert denied.status_code == 403
        assert denied.json() == {"detail": "request rejected"}
        assert read_only_get.status_code == 405

        assert exported_json.status_code == 200
        assert exported_json.headers["content-type"] == "application/json"
        assert exported_json.headers["content-disposition"] == (
            f'attachment; filename="evidence-packet-{primary.evidence_packet_id}.json"'
        )
        body = exported_json.json()
        assert body["schema"] == "dithyramba.evidence_packet_export/1.0"
        assert body["library_id"] == primary.library_id
        assert body["presentation"] == {
            "candidate_status": "candidate",
            "review_status": "unreviewed",
            "review_decision_ids": [],
        }
        assert body["export_authorization"] == {
            "access_policy_id": primary.access_policy_id,
            "policy_hash": primary.access_policy_hash,
            "allow_export": True,
        }
        assert body["evidence_packet"] == primary.packet_payload
        assert all(item["source_family_id"] for item in body["source_chips"])
        assert all(item["root_source_id"] for item in body["source_chips"])
        assert "canonical_uri" not in exported_json.text

        assert exported_markdown.status_code == 200
        assert exported_markdown.headers["content-type"].startswith("text/markdown")
        assert exported_markdown.headers["content-disposition"].endswith(
            f'{primary.evidence_packet_id}.md"'
        )
        assert "Candidate status: `candidate`" in exported_markdown.text
        assert "Review status: `unreviewed`" in exported_markdown.text
        assert "SourceFamily:" in exported_markdown.text
        assert "\n    The oracular fragment is packet-backed. <script>" in (exported_markdown.text)


def test_export_policy_hash_mismatch_fails_closed_without_detail(
    api_world: ApiWorld,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, token = make_test_app(api_world)
    original = LibraryRepository.get_access_policy

    def mismatched_policy(
        repository: LibraryRepository,
        access_policy_id: str,
    ) -> AccessPolicyRecord:
        record = original(repository, access_policy_id)
        return AccessPolicyRecord(
            name=record.name,
            snapshot=replace(record.snapshot, allow_export=False),
            created_at=record.created_at,
        )

    monkeypatch.setattr(LibraryRepository, "get_access_policy", mismatched_policy)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/v1/evidence-packets/{api_world.primary.evidence_packet_id}/exports",
            json=_export_request("json"),
            headers=mutation_headers(token),
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "internal error"}
    assert "policy" not in response.text


def test_library_and_packet_html_are_read_only_autoescaped_views(
    api_world: ApiWorld,
) -> None:
    app, _token = make_test_app(api_world)
    primary = api_world.primary
    with TestClient(app) as client:
        library = client.get(f"/libraries/{primary.library_id}")
        packet = client.get(f"/evidence-packets/{primary.evidence_packet_id}")
        foreign_library = client.get(f"/libraries/{api_world.foreign.library_id}")

        assert library.status_code == 200
        assert primary.library_name in library.text
        assert primary.collection_id in library.text
        assert str(api_world.data_home) not in library.text
        assert packet.status_code == 200
        assert primary.evidence_packet_id in packet.text
        assert 'id="candidate-status">candidate' in packet.text
        assert 'id="review-status">unreviewed' in packet.text
        assert "<script>" not in packet.text
        assert "<img " not in packet.text
        assert "&lt;script&gt;" in packet.text
        assert "&lt;img src=x onerror=window.pwned=true&gt;" in packet.text
        assert foreign_library.status_code == 404
        assert foreign_library.json() == {"detail": "not found"}

        parser = _ExactHighlightParser()
        parser.feed(packet.text)
        assert parser.forbidden == []


def test_canonical_export_policy_hash_is_distinct_from_receipt_hash(
    api_world: ApiWorld,
) -> None:
    """Guard the test oracle: policy hash is loaded from the canonical receipt."""

    packet = api_world.primary.packet_payload
    receipts = cast(dict[str, object], packet["receipts"])
    assert receipts["access_receipt_hash"] != api_world.primary.access_policy_hash
