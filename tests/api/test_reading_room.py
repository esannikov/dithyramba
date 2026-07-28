"""Policy-safe, script-free Reading Room HTTP acceptance."""

# ruff: noqa: RUF001

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    RequestScope,
)
from dithyramba.api import (
    EvidenceBoardWebConfig,
    ReadingRoomWebConfig,
    create_reading_room_app,
)
from dithyramba.api.reading_room import (
    _address_label,
    _board_lane,
    _evidence_board,
    _memory_state,
    _query_excerpt,
    _Runtime,
    _validate_board_requests,
)
from dithyramba.api.services import ViewerNotFoundError
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import sha256_hex
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.meaning import (
    EvidenceLinkProposal,
    ExtractionProfile,
    GeneratorProfile,
    MeaningProposal,
    MeaningRunRequest,
    ReviewBudget,
    StatementKind,
    StatementProposal,
    VoiceKind,
    VoiceProposal,
)
from dithyramba.persistence import (
    SQLiteMeaningRepository,
    SQLiteRecallBackend,
    initialize_library,
    open_library,
)
from dithyramba.reading_room import ReadingRoomProjection
from dithyramba.recall import (
    EvidenceFragment,
    EvidencePacket,
    QueryRequest,
    RecallService,
    search_ephemeral_fts,
)

from .support import ApiWorld

_CSP = (
    "default-src 'self'; script-src 'none'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'"
)


def _scope(world: ApiWorld) -> RequestScope:
    with open_library(world.primary.library_id, data_root=world.data_home) as repository:
        snapshot = repository.get_corpus_snapshot(world.primary.corpus_snapshot_id)
    return RequestScope(
        library_id=world.primary.library_id,
        snapshot_hash=snapshot.manifest_hash,
        purpose="research",
        collection_ids=(world.primary.collection_id,),
    )


def _app(world: ApiWorld) -> FastAPI:
    return create_reading_room_app(
        library_id=world.primary.library_id,
        data_home=world.data_home,
        corpus_snapshot_id=world.primary.corpus_snapshot_id,
        access_policy_id=world.primary.access_policy_id,
        scope=_scope(world),
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )


def _import_candidate(world: ApiWorld) -> None:
    text = world.primary.text
    quote = "oracular"
    start = text.index(quote)
    scope = _scope(world)
    proposal = MeaningProposal(
        voices=(VoiceProposal("author", VoiceKind.AUTHOR, "Author <img onerror=alert(1)>."),),
        statements=(
            StatementProposal(
                key="claim",
                kind=StatementKind.SOURCE_CLAIM,
                statement_text="Candidate <script>window.pwned=true</script> stays escaped.",
                voice_key="author",
                collection_id=world.primary.collection_id,
                evidence=(
                    EvidenceLinkProposal(
                        source_fragment_id=world.primary.source_fragment_id,
                        quote_start=start,
                        quote_end=start + len(quote),
                        quote_text=quote,
                        quote_sha256=sha256_hex(quote.encode("utf-8")),
                        attributed_voice_key="author",
                    ),
                ),
            ),
        ),
    )
    request = MeaningRunRequest(
        corpus_snapshot_id=world.primary.corpus_snapshot_id,
        access_policy_id=world.primary.access_policy_id,
        scope=scope,
        profile=ExtractionProfile.LIGHT,
        generator=GeneratorProfile(
            generator_id="test-import",
            revision="1",
            license="internal",
            prompt_version="reading-room-test-v1",
        ),
        review_budget=ReviewBudget(),
        code_version="test",
    )
    with open_library(world.primary.library_id, data_root=world.data_home) as repository:
        result = SQLiteMeaningRepository(repository).import_proposal(
            request=request,
            proposal=proposal,
        )
        assert result.run.status.value == "succeeded"


def test_empty_reading_room_is_real_get_only_scope_with_hardened_assets(
    api_world: ApiWorld,
) -> None:
    app = _app(api_world)
    with TestClient(app) as client:
        page = client.get("/")
        projection = client.get("/projection.json")
        stylesheet = client.get("/reading-room.css")
        absent_board = client.get("/evidence-board")
        mutation = client.post("/", content=b"ignored")

    assert page.status_code == 200
    assert page.headers["content-security-policy"] == _CSP
    assert page.headers["cache-control"] == "no-store"
    assert '<html lang="uk">' in page.text
    assert "корпус готовий" in page.text
    assert "Типізованих зв’язків у цьому зрізі ще немає" in page.text
    assert "<script" not in page.text.casefold()
    assert "<style" not in page.text.casefold()
    assert projection.status_code == 200
    assert projection.json()["schema"] == "dithyramba.reading_room_projection/1.0"
    assert projection.json()["projection_hash"] == projection.headers["etag"].strip('"')
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert "--teal: #0f5f5b" in stylesheet.text
    assert absent_board.status_code == 404
    assert mutation.status_code == 401
    assert mutation.json() == {"detail": "authentication required"}


def test_populated_reading_room_escapes_candidates_and_never_copies_source_text(
    api_world: ApiWorld,
) -> None:
    _import_candidate(api_world)
    app = _app(api_world)
    with TestClient(app) as client:
        page = client.get("/")
        projection = client.get("/projection.json")
        absent_selection = client.get("/?selected=statement_not_visible")
        source = client.get(
            f"/sources/{api_world.primary.source_id}/versions/"
            f"{api_world.primary.source_version_id}"
            f"?packet={api_world.primary.evidence_packet_id}"
            f"&fragment={api_world.primary.source_fragment_id}"
        )
        missing_source = client.get(
            f"/sources/{api_world.primary.source_id}/versions/"
            f"{api_world.primary.source_version_id}"
            f"?packet=packet_missing&fragment={api_world.primary.source_fragment_id}"
        )
        wrong_policy_source = client.get(
            f"/sources/{api_world.primary.source_id}/versions/"
            f"{api_world.primary.source_version_id}"
            f"?packet={api_world.primary.denied_evidence_packet_id}"
            f"&fragment={api_world.primary.source_fragment_id}"
        )

    assert page.status_code == 200
    assert "Candidate &lt;script&gt;window.pwned=true&lt;/script&gt; stays escaped." in page.text
    assert "<script>window.pwned=true</script>" not in page.text
    assert "<img onerror=alert(1)>" not in page.text
    assert api_world.primary.text not in page.text
    assert api_world.primary.text not in projection.text
    assert api_world.primary.source_fragment_id in projection.text
    assert "Відкрити точний фрагмент" in page.text
    assert f"packet={api_world.primary.evidence_packet_id}" in page.text
    assert source.status_code == 200
    assert "The oracular fragment is packet-backed." in source.text
    assert "&lt;script&gt;window.pwned = true&lt;/script&gt;" in source.text
    assert source.headers["content-security-policy"] == _CSP
    assert missing_source.status_code == 404
    assert missing_source.json() == {"detail": "not found"}
    assert wrong_policy_source.status_code == 404
    assert wrong_policy_source.json() == {"detail": "not found"}
    assert api_world.primary.text not in wrong_policy_source.text
    payload = projection.json()
    corpus_fragments = payload["corpus"]["fragments"]
    fragment = next(
        item
        for item in corpus_fragments
        if item["source_fragment_id"] == api_world.primary.source_fragment_id
    )
    assert fragment["source_version_id"] == api_world.primary.source_version_id
    assert fragment["source_id"] == api_world.primary.source_id
    assert fragment["source_family_id"] == api_world.primary.source_family_id
    assert fragment["collection_ids"] == [api_world.primary.collection_id]
    assert fragment["text_sha256"] == api_world.primary.text_sha256
    assert payload["graph"]["visible_edge_count"] >= 3
    assert {item["edge_type"] for item in payload["graph"]["edges"]} >= {
        "statement_voice",
        "statement_evidence_link",
        "evidence_link_fragment",
    }
    assert "relation-triple" in page.text
    assert "має доказ" in page.text
    assert page.text.count("↓") >= 2
    assert absent_selection.status_code == 200
    assert "Оберіть об’єкт" in absent_selection.text


def test_reading_room_rejects_wrong_host_without_disclosing_projection(
    api_world: ApiWorld,
) -> None:
    app = _app(api_world)
    with TestClient(app) as client:
        response = client.get("/projection.json", headers={"Host": "attacker.example"})

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid request"}
    assert api_world.primary.library_id not in response.text


def test_evidence_board_keeps_primary_and_research_packets_separate_and_authorized(
    tmp_path: Path,
) -> None:
    data_home = tmp_path / "data"
    primary_root = tmp_path / "primary"
    research_root = tmp_path / "research"
    primary_root.mkdir()
    research_root.mkdir()
    (primary_root / "catalogue.md").write_text(
        "# Catalogue\n\nGrothe, Heinrich — Singerstraße 849. <script>unsafe()</script>\n",
        encoding="utf-8",
    )
    (research_root / "notes.md").write_text(
        "# Working note\n\n"
        "Singerstraße 849 may share the Fähnrichhof address cluster; verify maps.\n",
        encoding="utf-8",
    )
    repository = initialize_library(LibraryConfig(name="Two-lane case"), data_root=data_home)
    try:
        primary = repository.create_collection(
            CollectionConfig(
                library_id=repository.library_id,
                name="Primary documents",
                kind=CollectionKind.CORPUS,
                roots=(build_collection_root(primary_root, data_root=data_home),),
            )
        )
        research = repository.create_collection(
            CollectionConfig(
                library_id=repository.library_id,
                name="Research notes",
                kind=CollectionKind.CORPUS,
                roots=(build_collection_root(research_root, data_root=data_home),),
            )
        )
        primary_outcome = (
            IngestService(repository)
            .ingest_path(primary.config.collection_id, "catalogue.md")
            .outcomes[0]
        )
        research_outcome = (
            IngestService(repository)
            .ingest_path(research.config.collection_id, "notes.md")
            .outcomes[0]
        )
        primary_snapshot = repository.freeze_snapshot((primary.config.collection_id,))
        research_snapshot = repository.freeze_snapshot((research.config.collection_id,))
        combined_snapshot = repository.freeze_snapshot(
            (primary.config.collection_id, research.config.collection_id)
        )
        policy = AccessPolicySnapshot(
            access_policy_id="policy_board",
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(
                CollectionRule(primary.config.collection_id, PolicyEffect.ALLOW),
                CollectionRule(research.config.collection_id, PolicyEffect.ALLOW),
            ),
            allow_export=True,
        )
        repository.persist_access_policy(name="Board policy", snapshot=policy)
        recall = RecallService(
            SQLiteRecallBackend(repository),
            profile_version=search_ephemeral_fts(
                question="profile", fragments=(), max_candidates=1
            ).profile_version,
        )
        question = "Singerstraße 849 Fähnrichhof"
        primary_packet = recall.recall(
            QueryRequest(
                question=question,
                library_id=repository.library_id,
                collection_ids=(primary.config.collection_id,),
                corpus_snapshot_id=primary_snapshot.corpus_snapshot_id,
                access_policy_id=policy.access_policy_id,
                purpose="research",
            )
        ).packet
        research_packet = recall.recall(
            QueryRequest(
                question=question,
                library_id=repository.library_id,
                collection_ids=(research.config.collection_id,),
                corpus_snapshot_id=research_snapshot.corpus_snapshot_id,
                access_policy_id=policy.access_policy_id,
                purpose="research",
            )
        ).packet
        mismatched_question_packet = recall.recall(
            QueryRequest(
                question="A different research question",
                library_id=repository.library_id,
                collection_ids=(research.config.collection_id,),
                corpus_snapshot_id=research_snapshot.corpus_snapshot_id,
                access_policy_id=policy.access_policy_id,
                purpose="research",
            )
        ).packet
        assert primary_packet.source_fragments and research_packet.source_fragments
        primary_fragment = primary_packet.source_fragments[0]
        assert primary_outcome.source_id is not None
        assert primary_outcome.source_version_id is not None
        assert research_outcome.source_id is not None

        scope = RequestScope(
            library_id=repository.library_id,
            snapshot_hash=combined_snapshot.manifest_hash,
            purpose="research",
            collection_ids=(primary.config.collection_id, research.config.collection_id),
        )
        app = create_reading_room_app(
            library_id=repository.library_id,
            data_home=data_home,
            corpus_snapshot_id=combined_snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            scope=scope,
            allowed_origin="http://testserver",
            evidence_board=EvidenceBoardWebConfig(
                primary_packet_id=primary_packet.evidence_packet_id,
                research_packet_id=research_packet.evidence_packet_id,
                primary_collection_ids=(primary.config.collection_id,),
                research_collection_ids=(research.config.collection_id,),
                display_question="Чи йдеться про одну адресу?",
                open_gap="Потрібна карта нумерації.",
            ),
            test_only_allow_testserver=True,
        )
        mismatch_app = create_reading_room_app(
            library_id=repository.library_id,
            data_home=data_home,
            corpus_snapshot_id=combined_snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            scope=scope,
            allowed_origin="http://testserver",
            evidence_board=EvidenceBoardWebConfig(
                primary_packet_id=primary_packet.evidence_packet_id,
                research_packet_id=mismatched_question_packet.evidence_packet_id,
                primary_collection_ids=(primary.config.collection_id,),
                research_collection_ids=(research.config.collection_id,),
                display_question="Чи йдеться про одну адресу?",
            ),
            test_only_allow_testserver=True,
        )
        swapped_scope_app = create_reading_room_app(
            library_id=repository.library_id,
            data_home=data_home,
            corpus_snapshot_id=combined_snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            scope=scope,
            allowed_origin="http://testserver",
            evidence_board=EvidenceBoardWebConfig(
                primary_packet_id=research_packet.evidence_packet_id,
                research_packet_id=primary_packet.evidence_packet_id,
                primary_collection_ids=(primary.config.collection_id,),
                research_collection_ids=(research.config.collection_id,),
                display_question="Чи йдеться про одну адресу?",
            ),
            test_only_allow_testserver=True,
        )
    finally:
        repository.close()

    with TestClient(app) as client:
        page = client.get("/evidence-board")
        selected = client.get(
            "/evidence-board",
            params={
                "selected_packet": primary_packet.evidence_packet_id,
                "selected_fragment": primary_fragment.source_fragment_id,
            },
        )
        source = client.get(
            f"/sources/{primary_outcome.source_id}/versions/{primary_outcome.source_version_id}",
            params={
                "packet": primary_packet.evidence_packet_id,
                "fragment": primary_fragment.source_fragment_id,
            },
        )
        unauthorized = client.get(
            f"/sources/{primary_outcome.source_id}/versions/{primary_outcome.source_version_id}",
            params={
                "packet": primary_packet.evidence_packet_id,
                "fragment": "fragment_missing",
            },
        )
        incomplete_selection = client.get(
            "/evidence-board",
            params={"selected_packet": primary_packet.evidence_packet_id},
        )
        missing_selection = client.get(
            "/evidence-board",
            params={
                "selected_packet": primary_packet.evidence_packet_id,
                "selected_fragment": "fragment_missing",
            },
        )
        research_selection = client.get(
            "/evidence-board",
            params={
                "selected_packet": research_packet.evidence_packet_id,
                "selected_fragment": research_packet.source_fragments[0].source_fragment_id,
            },
        )

    with TestClient(mismatch_app) as client:
        mismatch = client.get("/evidence-board")
    with TestClient(swapped_scope_app) as client:
        swapped_scope = client.get("/evidence-board")

    assert page.status_code == 200
    assert page.headers["content-security-policy"] == _CSP
    assert "Первинні джерела" in page.text
    assert "Дослідницькі тлумачення" in page.text
    assert primary_packet.evidence_packet_id in page.text
    assert research_packet.evidence_packet_id in page.text
    assert "Потрібна карта нумерації." in page.text
    assert "&lt;script&gt;unsafe()&lt;/script&gt;" in page.text
    assert "<script" not in page.text.casefold()
    assert "<form" not in page.text.casefold()
    assert selected.status_code == 200
    assert "Grothe, Heinrich" in selected.text
    assert source.status_code == 200
    assert "Незмінний фрагмент" in source.text
    assert unauthorized.status_code == 404
    assert incomplete_selection.status_code == 404
    assert missing_selection.status_code == 404
    assert research_selection.status_code == 200
    assert "Дослідницькі тлумачення" in research_selection.text
    assert mismatch.status_code == 409
    assert swapped_scope.status_code == 409


@pytest.mark.parametrize(
    "update",
    [
        {"primary_packet_id": "invalid"},
        {"research_packet_id": "packet_primary"},
        {"primary_collection_ids": ()},
        {"research_collection_ids": ("wrong",)},
        {"display_question": " padded "},
        {"open_gap": ""},
    ],
)
def test_evidence_board_config_rejects_ambiguous_inputs(update: dict[str, object]) -> None:
    values: dict[str, object] = {
        "primary_packet_id": "packet_primary",
        "research_packet_id": "packet_research",
        "primary_collection_ids": ("collection_primary",),
        "research_collection_ids": ("collection_research",),
        "display_question": "A bounded question",
        "open_gap": None,
    }
    values.update(update)

    with pytest.raises(ValueError):
        EvidenceBoardWebConfig(**values)  # type: ignore[arg-type]


def test_reading_room_config_and_inactive_runtime_fail_closed(api_world: ApiWorld) -> None:
    valid = _app(api_world).state.reading_room_config
    assert isinstance(valid, ReadingRoomWebConfig)
    wrong_library_scope = RequestScope(
        library_id=api_world.foreign.library_id,
        snapshot_hash=valid.scope.snapshot_hash,
        purpose=valid.scope.purpose,
        collection_ids=valid.scope.collection_ids,
    )
    overlap_board = EvidenceBoardWebConfig(
        primary_packet_id="packet_primary",
        research_packet_id="packet_research",
        primary_collection_ids=(api_world.primary.collection_id,),
        research_collection_ids=(api_world.primary.collection_id,),
        display_question="Question",
    )
    wider_board = EvidenceBoardWebConfig(
        primary_packet_id="packet_primary",
        research_packet_id="packet_research",
        primary_collection_ids=(api_world.primary.collection_id,),
        research_collection_ids=("collection_extra",),
        display_question="Question",
    )

    for update, error in (
        ({"scope": wrong_library_scope}, ValueError),
        ({"corpus_snapshot_id": "wrong"}, ValueError),
        ({"access_policy_id": "wrong"}, ValueError),
        ({"limits": object()}, TypeError),
        ({"evidence_board": overlap_board}, ValueError),
        ({"evidence_board": wider_board}, ValueError),
    ):
        with pytest.raises(error):
            replace(valid, **update)

    inactive = _Runtime(valid)
    with pytest.raises(RuntimeError):
        inactive.require_service()
    with pytest.raises(RuntimeError):
        inactive.require_packets()
    with pytest.raises(RuntimeError):
        inactive.require_recall_backend()
    with pytest.raises(ViewerNotFoundError):
        _evidence_board(inactive, selected_packet=None, selected_fragment=None)

    request = QueryRequest(
        question="oracular",
        library_id=api_world.primary.library_id,
        collection_ids=(api_world.primary.collection_id,),
        corpus_snapshot_id=api_world.primary.corpus_snapshot_id,
        access_policy_id=api_world.primary.access_policy_id,
        purpose="research",
    )
    with pytest.raises(ViewerNotFoundError):
        _validate_board_requests(
            inactive,
            primary_request=request,
            research_request=request,
        )
    with pytest.raises(RuntimeError):
        _board_lane(
            inactive,
            lane_id="primary",
            label="Primary",
            explanation="Literal evidence",
            packet=cast(EvidencePacket, object()),
            request=request,
            chips=(),
            collection_ids=(),
        )


def test_query_excerpt_is_bounded_at_middle_and_end() -> None:
    middle = _query_excerpt("x" * 200 + "needle" + "y" * 200, "needle", limit=100)
    tail = _query_excerpt("x" * 400 + "needle", "needle", limit=100)

    assert len(middle) == 102
    assert middle.startswith("…") and middle.endswith("…")
    assert len(tail) == 101
    assert tail.startswith("…") and tail.endswith("needle")


def test_address_label_uses_heading_and_single_line_when_available() -> None:
    address = SimpleNamespace(
        payload=lambda: {
            "heading_path": ["Chapter", "Address"],
            "line_start": 7,
            "line_end": 7,
        }
    )
    fragment = cast(
        EvidenceFragment,
        SimpleNamespace(source_address=address, source_fragment_id="fragment_example"),
    )
    fallback = cast(
        EvidenceFragment,
        SimpleNamespace(
            source_address=SimpleNamespace(payload=lambda: {}),
            source_fragment_id="fragment_fallback",
        ),
    )

    assert _address_label(fragment) == "Chapter › Address · рядок 7"
    assert _address_label(fallback) == "fragment_fallback"


@pytest.mark.parametrize(
    ("fragment_count", "run_states", "unresolved", "expected"),
    [
        (0, (), 0, "порожній дозволений зріз"),
        (1, ("failed",), 0, "потребує уваги"),
        (1, ("partial",), 0, "частково опрацьовано"),
        (1, ("complete",), 1, "кандидати очікують перевірки"),
        (1, ("complete",), 0, "опрацьовано в поточному scope"),
    ],
)
def test_memory_state_keeps_failure_and_review_states_distinct(
    fragment_count: int,
    run_states: tuple[str, ...],
    unresolved: int,
    expected: str,
) -> None:
    projection = cast(
        ReadingRoomProjection,
        SimpleNamespace(
            corpus=SimpleNamespace(source_fragment_count=fragment_count),
            recent_runs=tuple(SimpleNamespace(coverage_state=state) for state in run_states),
            review=SimpleNamespace(unresolved_count=unresolved),
        ),
    )

    assert _memory_state(projection)[0] == expected
