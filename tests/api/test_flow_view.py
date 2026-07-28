"""Acceptance checks for the small, script-free memory flow view."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dithyramba.api import create_flow_view_app


def _client() -> TestClient:
    return TestClient(
        create_flow_view_app(
            allowed_origin="http://testserver",
            test_only_allow_testserver=True,
        )
    )


def test_flow_view_renders_main_process_without_live_claims() -> None:
    response = _client().get("/")

    assert response.status_code == 200
    assert "Від джерела до перевіреної памʼяті" in response.text
    assert "Пакет" in response.text
    assert "Перевірка" in response.text
    assert "схема · не live" in response.text
    assert "<script" not in response.text
    assert response.headers["content-security-policy"].startswith("default-src 'self'")


def test_flow_view_selects_one_node_and_keeps_details_short() -> None:
    response = _client().get("/?node=gate")

    assert response.status_code == 200
    assert "Брама перевіряє" in response.text
    assert "Отримує" in response.text
    assert "Передає" in response.text
    assert "відсоток істини" in response.text


def test_flow_view_rejects_unknown_node() -> None:
    response = _client().get("/?node=unknown-stage")

    assert response.status_code == 404


def test_flow_json_is_small_and_presentation_safe() -> None:
    response = _client().get("/flow.json")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema"] == "dithyramba.flow_view/1.0"
    assert payload["mode"] == "static_process_map_not_live_trace"
    assert len(payload["nodes"]) == 8
    assert len(payload["links"]) == 8
    assert "flow_hash" in payload
    assert "source_text" not in response.text


def test_flow_styles_are_local_and_responsive() -> None:
    response = _client().get("/flow-view.css")

    assert response.status_code == 200
    assert "@media (max-width: 48rem)" in response.text
    assert "prefers-reduced-motion" in response.text
    assert "linear-gradient" not in response.text
