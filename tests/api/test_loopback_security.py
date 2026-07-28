"""Fail-closed loopback boundary and generic error behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dithyramba.api import bearer_token_for, create_app

from .support import ApiWorld, make_test_app, mutation_headers

_CSP = (
    "default-src 'self'; script-src 'none'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'"
)


def _valid_review_payload(world: ApiWorld) -> dict[str, object]:
    return {
        "schema": "dithyramba.review_decision_request/1.0",
        "target_type": "evidence_packet",
        "target_id": world.primary.evidence_packet_id,
        "target_hash": world.primary.packet_hash,
        "action": "accept",
        "reason": "The persisted packet and its exact provenance were reviewed.",
        "authority": "researcher:test",
        "scope": {
            "schema": "dithyramba.review_scope/1.0",
            "collection_ids": [world.primary.collection_id],
            "use": "research",
        },
        "supersedes_review_decision_id": None,
    }


def test_app_tokens_are_random_per_factory_instance(api_world: ApiWorld) -> None:
    first, first_token = make_test_app(api_world)
    second, second_token = make_test_app(api_world)

    assert first is not second
    assert first_token != second_token
    assert len(first_token) >= 43
    assert bearer_token_for(first) == first_token


def test_production_factory_rejects_testserver_origin(api_world: ApiWorld) -> None:
    try:
        create_app(
            library_id=api_world.primary.library_id,
            data_home=api_world.data_home,
            allowed_origin="http://testserver",
        )
    except ValueError as error:
        assert "loopback Origin" in str(error)
    else:
        raise AssertionError("testserver must require the explicit test-only flag")


def test_docs_host_headers_cookies_and_cors_are_fail_closed(api_world: ApiWorld) -> None:
    app, _token = make_test_app(api_world)
    with TestClient(app) as client:
        health = client.get("/health")
        wrong_host = client.get("/health", headers={"Host": "attacker.example"})

        assert health.status_code == 200
        for path in ("/docs", "/redoc", "/openapi.json"):
            response = client.get(path)
            assert response.status_code == 404
            assert response.json() == {"detail": "not found"}

        assert wrong_host.status_code == 400
        assert wrong_host.json() == {"detail": "invalid request"}
        for response in (health, wrong_host):
            assert response.headers["content-security-policy"] == _CSP
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["referrer-policy"] == "no-referrer"
            assert "camera=()" in response.headers["permissions-policy"]
            assert response.headers["cache-control"] == "no-store"
            assert "set-cookie" not in response.headers
            assert "access-control-allow-origin" not in response.headers

        preflight = client.options(
            "/v1/review-decisions",
            headers={
                "Origin": "http://attacker.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 405
        assert "access-control-allow-origin" not in preflight.headers


def test_every_mutating_method_requires_token_and_exact_origin(api_world: ApiWorld) -> None:
    app, token = make_test_app(api_world)
    payload = _valid_review_payload(api_world)
    with TestClient(app) as client:
        for method in ("post", "put", "patch", "delete"):
            unauthenticated = client.request(method, "/v1/review-decisions", json=payload)
            assert unauthenticated.status_code == 401
            assert unauthenticated.json() == {"detail": "authentication required"}

        wrong_token = client.post(
            "/v1/review-decisions",
            json=payload,
            headers={
                "Authorization": "Bearer definitely-wrong",
                "Origin": "http://testserver",
            },
        )
        assert wrong_token.status_code == 401

        for origin in (None, "http://attacker.example", "http://testserver/"):
            headers = {"Authorization": f"Bearer {token}"}
            if origin is not None:
                headers["Origin"] = origin
            rejected = client.post("/v1/review-decisions", json=payload, headers=headers)
            assert rejected.status_code == 403
            assert rejected.json() == {"detail": "request rejected"}

        for method in ("put", "patch", "delete"):
            method_safe = client.request(
                method,
                "/v1/review-decisions",
                json=payload,
                headers=mutation_headers(token),
            )
            assert method_safe.status_code == 405
            assert method_safe.json() == {"detail": "method not allowed"}


def test_body_cap_and_validation_errors_are_generic(api_world: ApiWorld) -> None:
    app, token = make_test_app(api_world, max_request_body_bytes=1_024)
    with TestClient(app) as client:
        oversized = client.post(
            "/v1/review-decisions",
            content=b"x" * 1_025,
            headers={
                **mutation_headers(token),
                "Content-Type": "application/json",
            },
        )
        assert oversized.status_code == 413
        assert oversized.json() == {"detail": "request body too large"}

        invalid = client.post(
            "/v1/review-decisions",
            json={**_valid_review_payload(api_world), "unexpected": "rejected"},
            headers=mutation_headers(token),
        )
        assert invalid.status_code == 422
        assert invalid.json() == {"detail": "invalid request"}
