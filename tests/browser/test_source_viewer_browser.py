"""Real Chromium acceptance test for packet-backed SourceChip navigation."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Browser, Error, Playwright, sync_playwright

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.api import create_app
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import SQLiteRecallBackend, initialize_library
from dithyramba.recall import QueryRequest, RecallService, search_ephemeral_fts


@dataclass(frozen=True, slots=True)
class PreparedLibrary:
    library_id: str
    evidence_packet_id: str
    source_id: str
    source_version_id: str
    source_fragment_id: str
    text: str
    text_sha256: str


def build_prepared_library(
    *,
    data_home: Path,
    source_root: Path,
    name: str,
    text: str,
) -> PreparedLibrary:
    source_root.mkdir(parents=True)
    (source_root / "evidence.md").write_text(text, encoding="utf-8")
    repository = initialize_library(LibraryConfig(name=name), data_root=data_home)
    try:
        collection = repository.create_collection(
            CollectionConfig(
                library_id=repository.library_id,
                name="Browser corpus",
                kind=CollectionKind.CORPUS,
                roots=(build_collection_root(source_root, data_root=data_home),),
            )
        )
        outcome = (
            IngestService(repository)
            .ingest_path(
                collection.config.collection_id,
                "evidence.md",
            )
            .outcomes[0]
        )
        if outcome.source_id is None:
            raise AssertionError("browser test source did not ingest")
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
        profile_version = search_ephemeral_fts(
            question="profile",
            fragments=(),
            max_candidates=1,
        ).profile_version
        result = RecallService(
            SQLiteRecallBackend(repository),
            profile_version=profile_version,
            code_version="browser-test",
        ).recall(request)
        if not result.packet.source_fragments:
            raise AssertionError("browser test recall must return a source fragment")
        fragment = max(result.packet.source_fragments, key=lambda item: len(item.text))
        return PreparedLibrary(
            library_id=repository.library_id,
            evidence_packet_id=result.packet.evidence_packet_id,
            source_id=outcome.source_id,
            source_version_id=fragment.source_version_id,
            source_fragment_id=fragment.source_fragment_id,
            text=fragment.text,
            text_sha256=fragment.text_sha256,
        )
    finally:
        repository.close()


@contextmanager
def _running_server(data_home: Path, prepared: PreparedLibrary) -> Iterator[str]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    origin = f"http://127.0.0.1:{port}"
    app = create_app(
        library_id=prepared.library_id,
        data_home=data_home,
        allowed_origin=origin,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("loopback test server did not start")
    try:
        yield origin
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("loopback test server did not stop")


def _launch_browser() -> tuple[Playwright, Browser]:
    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.launch(headless=True)
    except Error as error:
        playwright.stop()
        message = str(error).casefold()
        if "executable doesn't exist" in message or "playwright install" in message:
            pytest.skip("Playwright Chromium executable is unavailable")
        raise
    return playwright, browser


def test_source_chip_opens_one_exact_inert_highlight_in_real_browser(tmp_path: Path) -> None:
    data_home = tmp_path / "data"
    prepared = build_prepared_library(
        data_home=data_home,
        source_root=tmp_path / "browser-source",
        name="Browser API library",
        text=(
            "# Browser evidence\n\n"
            "The oracular browser fragment stays inert. "
            "<script>window.pwned = true</script> "
            "<img src=x onerror=window.pwned=true>\n"
        ),
    )
    with _running_server(data_home, prepared) as origin:
        playwright, browser = _launch_browser()
        try:
            page = browser.new_page()
            packet_response = page.goto(
                origin + f"/evidence-packets/{prepared.evidence_packet_id}",
                wait_until="domcontentloaded",
            )
            assert packet_response is not None
            assert packet_response.status == 200
            assert page.locator('[data-source-chip="true"]').count() >= 1
            assert page.locator("script, img, form, button, input").count() == 0
            assert page.evaluate("() => window.pwned") is None
            with page.expect_navigation(wait_until="domcontentloaded") as navigation:
                page.locator(
                    f'[data-source-fragment-id="{prepared.source_fragment_id}"] '
                    '[data-source-chip="true"]'
                ).click()
            response = navigation.value
            assert response.status == 200
            assert response.headers["content-security-policy"] == (
                "default-src 'self'; script-src 'none'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'"
            )
            highlight = page.locator('[data-source-highlight="true"]')
            assert highlight.count() == 1
            assert highlight.get_attribute("data-source-id") == prepared.source_id
            assert highlight.get_attribute("data-source-version-id") == prepared.source_version_id
            assert highlight.get_attribute("data-source-fragment-id") == prepared.source_fragment_id
            assert highlight.get_attribute("data-text-sha256") == prepared.text_sha256
            assert highlight.inner_text() == prepared.text
            assert page.locator("script, img, form, button, input").count() == 0
            assert page.evaluate("() => window.pwned") is None
        finally:
            browser.close()
            playwright.stop()
