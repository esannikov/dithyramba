"""Packet projection, review API, and exact read-only SourceChip viewer."""

from __future__ import annotations

import json
from html.parser import HTMLParser

from fastapi.testclient import TestClient

from .support import ApiWorld, make_test_app, mutation_headers


class _ViewerParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.highlight_attributes: list[dict[str, str | None]] = []
        self.forbidden_elements: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if attributes.get("data-source-highlight") == "true":
            self.highlight_attributes.append(attributes)
        if tag in {"script", "img", "form", "button", "input"}:
            self.forbidden_elements.append(tag)


def _review_payload(world: ApiWorld) -> dict[str, object]:
    return {
        "schema": "dithyramba.review_decision_request/1.0",
        "target_type": "evidence_packet",
        "target_id": world.primary.evidence_packet_id,
        "target_hash": world.primary.packet_hash,
        "action": "accept",
        "reason": "Reviewed against the exact packet-backed source fragment.",
        "authority": "researcher:test",
        "scope": {
            "schema": "dithyramba.review_scope/1.0",
            "collection_ids": [world.primary.collection_id],
            "use": "research",
        },
        "supersedes_review_decision_id": None,
    }


def test_health_and_packet_source_chip_navigate_to_exact_highlight(
    api_world: ApiWorld,
) -> None:
    app, _token = make_test_app(api_world)
    expected = api_world.primary
    with TestClient(app) as client:
        health = client.get("/health")
        packet = client.get(f"/v1/evidence-packets/{expected.evidence_packet_id}")
        chips = client.get(f"/v1/evidence-packets/{expected.evidence_packet_id}/source-chips")

        assert health.json()["library_id"] == expected.library_id
        assert packet.status_code == 200
        body = packet.json()
        assert body == expected.packet_payload
        assert body["schema"] == "dithyramba.evidence_packet/1.0"
        assert body["evidence_packet_id"] == expected.evidence_packet_id
        assert body["packet_hash"] == expected.packet_hash
        assert len(body["source_fragments"]) >= 1
        chip_items = chips.json()["items"]
        assert len(chip_items) == len(body["source_fragments"])
        chip = next(
            item for item in chip_items if item["source_fragment_id"] == expected.source_fragment_id
        )
        packet_fragment = next(
            item
            for item in body["source_fragments"]
            if item["source_fragment_id"] == expected.source_fragment_id
        )
        assert chip == {
            "schema": "dithyramba.source_chip/1.0",
            "evidence_packet_id": expected.evidence_packet_id,
            "packet_hash": expected.packet_hash,
            "source_id": expected.source_id,
            "source_family_id": expected.source_family_id,
            "root_source_id": expected.root_source_id,
            "family_role": expected.family_role,
            "source_version_id": expected.source_version_id,
            "source_fragment_id": expected.source_fragment_id,
            "text_sha256": expected.text_sha256,
            "source_address": packet_fragment["source_address"],
            "viewer_url": (
                f"/sources/{expected.source_id}/versions/{expected.source_version_id}"
                f"?packet={expected.evidence_packet_id}"
                f"&fragment={expected.source_fragment_id}"
            ),
        }

        viewer = client.get(chip["viewer_url"])
        parser = _ViewerParser()
        parser.feed(viewer.text)
        assert viewer.status_code == 200
        assert viewer.headers["content-type"].startswith("text/html")
        assert len(parser.highlight_attributes) == 1
        assert parser.forbidden_elements == []
        attributes = parser.highlight_attributes[0]
        assert attributes["data-source-id"] == expected.source_id
        assert attributes["data-source-version-id"] == expected.source_version_id
        assert attributes["data-source-fragment-id"] == expected.source_fragment_id
        assert attributes["data-text-sha256"] == expected.text_sha256
        assert json.loads(attributes["data-source-address"] or "null") == (expected.source_address)
        assert "source-address" in viewer.text
        assert expected.text in viewer.text.replace("&lt;", "<").replace("&gt;", ">")
        assert "<script>" not in viewer.text
        assert "<img " not in viewer.text
        assert "&lt;script&gt;" in viewer.text
        assert "&lt;img src=x onerror=window.pwned=true&gt;" in viewer.text


def test_viewer_route_is_get_only_and_requires_exact_provenance_tuple(
    api_world: ApiWorld,
) -> None:
    app, token = make_test_app(api_world)
    source = api_world.primary
    exact_path = (
        f"/sources/{source.source_id}/versions/{source.source_version_id}"
        f"?packet={source.evidence_packet_id}&fragment={source.source_fragment_id}"
    )
    with TestClient(app) as client:
        for method in ("post", "put", "patch", "delete"):
            response = client.request(
                method,
                exact_path,
                headers=mutation_headers(token),
            )
            assert response.status_code == 405
            assert response.json() == {"detail": "method not allowed"}

        mismatched_version = client.get(
            f"/sources/{source.source_id}/versions/{api_world.foreign.source_version_id}"
            f"?packet={source.evidence_packet_id}&fragment={source.source_fragment_id}"
        )
        mismatched_fragment = client.get(
            f"/sources/{source.source_id}/versions/{source.source_version_id}"
            f"?packet={source.evidence_packet_id}"
            f"&fragment={api_world.foreign.source_fragment_id}"
        )
        for response in (mismatched_version, mismatched_fragment):
            assert response.status_code == 404
            assert response.json() == {"detail": "not found"}


def test_review_create_get_and_list_are_typed_and_library_scoped(
    api_world: ApiWorld,
) -> None:
    app, token = make_test_app(api_world)
    with TestClient(app) as client:
        created = client.post(
            "/v1/review-decisions",
            json=_review_payload(api_world),
            headers=mutation_headers(token),
        )
        assert created.status_code == 201
        body = created.json()
        assert body["schema"] == "dithyramba.review_decision/1.0"
        assert body["library_id"] == api_world.primary.library_id
        assert body["target_id"] == api_world.primary.evidence_packet_id
        assert body["scope"] == {
            "schema": "dithyramba.review_scope/1.0",
            "collection_ids": [api_world.primary.collection_id],
            "use": "research",
        }

        loaded = client.get(f"/v1/review-decisions/{body['review_decision_id']}")
        listed = client.get(
            "/v1/review-decisions",
            params={
                "target_type": "evidence_packet",
                "target_id": api_world.primary.evidence_packet_id,
            },
        )
        assert loaded.status_code == 200
        assert loaded.json() == body
        assert listed.status_code == 200
        assert listed.json()["items"] == [body]


def test_foreign_library_ids_are_indistinguishable_from_absent_ids(
    api_world: ApiWorld,
) -> None:
    app, token = make_test_app(api_world)
    foreign = api_world.foreign
    with TestClient(app) as client:
        missing_packet = client.get("/v1/evidence-packets/packet_absent")
        foreign_packet = client.get(f"/v1/evidence-packets/{foreign.evidence_packet_id}")
        foreign_viewer = client.get(
            f"/sources/{foreign.source_id}/versions/{foreign.source_version_id}"
            f"?packet={foreign.evidence_packet_id}&fragment={foreign.source_fragment_id}"
        )
        foreign_review = client.post(
            "/v1/review-decisions",
            json={
                "schema": "dithyramba.review_decision_request/1.0",
                "target_type": "evidence_packet",
                "target_id": foreign.evidence_packet_id,
                "target_hash": foreign.packet_hash,
                "action": "accept",
                "reason": "This foreign target must not disclose whether it exists.",
                "authority": "researcher:test",
                "scope": {
                    "schema": "dithyramba.review_scope/1.0",
                    "collection_ids": [api_world.primary.collection_id],
                    "use": "research",
                },
                "supersedes_review_decision_id": None,
            },
            headers=mutation_headers(token),
        )

        assert missing_packet.status_code == 404
        for response in (foreign_packet, foreign_viewer, foreign_review):
            assert response.status_code == missing_packet.status_code
            assert response.json() == missing_packet.json() == {"detail": "not found"}
