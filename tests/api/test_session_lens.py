"""GET-only Session Lens acceptance checks over a durable research journal."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
)
from dithyramba.api import create_session_lens_app
from dithyramba.api.models import SourceChipResponse
from dithyramba.api.session_lens import (
    SessionLensWebConfig,
    _address_label,
    _event_copy,
    _Runtime,
    _source_role,
)
from dithyramba.cli import app
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.ingest import MarkdownSourceAddress
from dithyramba.ingest.service import IngestService
from dithyramba.interactive import AgentResearchFacade
from dithyramba.library import LibraryConfig
from dithyramba.persistence import initialize_library
from dithyramba.recall import EvidenceFragment, current_fts_runtime_profile
from dithyramba.sessions import ResearchSessionBrief, SessionEventKind


def _session_world(tmp_path: Path) -> tuple[str, Path, str, str, str]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "craft.md").write_text(
        "# Dialogue direction\n\nBlocking changes the power relation inside a dialogue scene.\n",
        encoding="utf-8",
    )
    data_home = tmp_path / "data"
    with initialize_library(
        LibraryConfig(name="Directing craft memory"),
        data_root=data_home,
    ) as repository:
        collection = repository.create_collection(
            CollectionConfig(
                library_id=repository.library_id,
                name="Directing corpus",
                kind=CollectionKind.CORPUS,
                roots=(
                    build_collection_root(
                        corpus,
                        data_root=repository.paths.application_data_root,
                    ),
                ),
            )
        )
        outcome = (
            IngestService(repository)
            .ingest_path(
                collection.config.collection_id,
                "craft.md",
            )
            .outcomes[0]
        )
        assert outcome.source_id is not None
        snapshot = repository.freeze_snapshot((collection.config.collection_id,))
        policy = AccessPolicySnapshot(
            access_policy_id="policy_session_lens",
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),),
        )
        repository.persist_access_policy(name="Session Lens", snapshot=policy)
        facade = AgentResearchFacade(
            repository,
            agent_id="agent:session_lens_test",
            profile_version=current_fts_runtime_profile().profile_version,
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="How should a dialogue scene reveal changing power?",
                intended_use="directorial decision",
                success_criteria=("find exact source support",),
                boundaries=("do not treat a fragment as an accepted conclusion",),
            ),
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            purpose="research",
            collection_ids=(collection.config.collection_id,),
        )
        turn = facade.recall(
            opened.session_id,
            "blocking changes power relation dialogue scene",
            command_id="command_session_lens_001",
        )
        facade.record_draft(
            opened.session_id,
            "Movement can externalize a status change; retain this as a working synthesis.",
        )
        facade.record_gap(opened.session_id, "Need a static-blocking counterexample.")
        facade.reject_path(
            opened.session_id,
            "Do not generalize one craft sentence into a universal directing law.",
        )
        fragment = turn.evidence_packet.source_fragments[0]
        return (
            repository.library_id,
            data_home,
            opened.session_id,
            turn.evidence_packet.evidence_packet_id,
            fragment.source_fragment_id,
        )


def _client(tmp_path: Path) -> tuple[TestClient, str, str]:
    library_id, data_home, session_id, packet_id, fragment_id = _session_world(tmp_path)
    app = create_session_lens_app(
        library_id=library_id,
        data_home=data_home,
        session_id=session_id,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    return TestClient(app), packet_id, fragment_id


def test_session_lens_renders_human_journal_and_exact_source(tmp_path: Path) -> None:
    client, packet_id, fragment_id = _client(tmp_path)
    with client:
        response = client.get("/")
        selected = client.get(
            "/",
            params={"packet": packet_id, "fragment": fragment_id},
        )

        assert response.status_code == selected.status_code == 200
        assert "How should a dialogue scene reveal changing power?" in response.text
        assert "Directing craft memory" in response.text
        assert "Blocking changes the power relation" in response.text
        assert "Робочий синтез" in response.text
        assert "Прогалина" in response.text
        assert "Відхилений шлях" in response.text
        assert "не автоматично" in response.text
        assert "прийнятий висновок" in response.text
        assert 'aria-current="true"' in selected.text
        assert response.headers["content-security-policy"].startswith("default-src 'self'")


def test_session_lens_projection_is_stable_compact_and_get_only(tmp_path: Path) -> None:
    client, packet_id, fragment_id = _client(tmp_path)
    with client:
        first = client.get("/projection.json")
        second = client.get("/projection.json")
        styles = client.get("/session-lens.css")

        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert first.headers["etag"] == second.headers["etag"]
        assert first.json()["schema"] == "dithyramba.session_lens_projection/1.0"
        assert first.json()["projection_hash"] in first.headers["etag"]
        assert "Blocking changes the power relation" not in first.text
        assert styles.status_code == 200
        assert "@media (max-width: 55rem)" in styles.text
        assert client.get("/favicon.ico").status_code == 204

        for method in ("post", "put", "patch", "delete"):
            result = client.request(
                method,
                "/",
                headers={
                    "Authorization": "Bearer wrong",
                    "Origin": "http://testserver",
                },
            )
            assert result.status_code == 401

        assert client.get("/", params={"packet": packet_id}).status_code == 404
        assert (
            client.get(
                "/",
                params={
                    "packet": packet_id,
                    "fragment": "fragment_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                },
            ).status_code
            == 404
        )
        assert fragment_id.startswith("fragment_")


def test_session_lens_rejects_a_missing_session_at_startup(tmp_path: Path) -> None:
    library_id, data_home, _session_id, _packet_id, _fragment_id = _session_world(tmp_path)
    app = create_session_lens_app(
        library_id=library_id,
        data_home=data_home,
        session_id="research_session_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with (
        pytest.raises(Exception, match="ResearchSession does not exist"),
        TestClient(app),
    ):
        pass


def test_session_lens_copy_source_roles_and_addresses_are_complete() -> None:
    for kind in SessionEventKind:
        label, explanation, tone = _event_copy(kind)
        assert label and explanation and tone

    chip = SourceChipResponse.model_construct(family_role="root")
    assert _source_role(chip) == "оригінальна версія джерела"
    derivative = SourceChipResponse.model_construct(family_role="derivative")
    assert _source_role(derivative) == "похідна версія"
    duplicate = SourceChipResponse.model_construct(family_role="duplicate")
    assert _source_role(duplicate) == "дублікат перевіреного джерела"
    unknown = SourceChipResponse.model_construct(family_role="unknown")
    assert _source_role(unknown) == "unknown"

    addressed = EvidenceFragment.model_construct(
        source_address=MarkdownSourceAddress(
            heading_path=("Chapter", "Scene"),
            line_start=8,
            line_end=10,
            char_start=0,
            char_end=12,
        )
    )
    assert _address_label(addressed) == "Chapter / Scene · рядки 8-10"
    one_line = EvidenceFragment.model_construct(
        source_address=MarkdownSourceAddress(
            heading_path=(),
            line_start=4,
            line_end=4,
            char_start=0,
            char_end=5,
        )
    )
    assert _address_label(one_line) == "рядок 4"

    class _EmptyAddress:
        @staticmethod
        def payload() -> dict[str, object]:
            return {}

    no_address = EvidenceFragment.model_construct(source_address=_EmptyAddress())
    assert _address_label(no_address) == "точна адреса збережена в пакеті"

    runtime = _Runtime(cast(SessionLensWebConfig, object()))
    with pytest.raises(RuntimeError, match="lifespan"):
        runtime.require_repository()
    with pytest.raises(RuntimeError, match="lifespan"):
        runtime.require_sessions()
    with pytest.raises(RuntimeError, match="lifespan"):
        runtime.require_packets()


def test_session_lens_cli_opens_exact_session_and_starts_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_id, data_home, session_id, _packet_id, _fragment_id = _session_world(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(application: FastAPI, **kwargs: object) -> None:
        captured["application"] = application
        captured.update(kwargs)

    monkeypatch.setattr("dithyramba.cli.uvicorn.run", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "session-lens",
            "--library",
            library_id,
            "--session",
            session_id,
            "--data-home",
            str(data_home),
            "--port",
            "8354",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "session_lens: http://127.0.0.1:8354\n"
        "projection_json: http://127.0.0.1:8354/projection.json\n"
        "mode: read-only session projection\n"
    )
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8354
    assert captured["access_log"] is False
