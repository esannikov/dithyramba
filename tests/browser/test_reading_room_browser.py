# ruff: noqa: RUF001
"""Real Chromium acceptance for the script-free Reading Room."""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Browser, Error, Page, Playwright, sync_playwright

from dithyramba.access import RequestScope
from dithyramba.api import create_reading_room_app
from dithyramba.contracts import sha256_hex
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
from dithyramba.persistence import SQLiteMeaningRepository, open_library
from tests.api.support import ApiWorld, build_api_world


def _scope(world: ApiWorld) -> RequestScope:
    with open_library(world.primary.library_id, data_root=world.data_home) as repository:
        snapshot = repository.get_corpus_snapshot(world.primary.corpus_snapshot_id)
    return RequestScope(
        library_id=world.primary.library_id,
        snapshot_hash=snapshot.manifest_hash,
        purpose="research",
        collection_ids=(world.primary.collection_id,),
    )


def _import_candidate(world: ApiWorld) -> None:
    quote = "oracular"
    start = world.primary.text.index(quote)
    scope = _scope(world)
    proposal = MeaningProposal(
        voices=(
            VoiceProposal(
                "author",
                VoiceKind.AUTHOR,
                "Author <img src=x onerror=window.pwned=true>.",
            ),
        ),
        statements=(
            StatementProposal(
                key="claim",
                kind=StatementKind.SOURCE_CLAIM,
                statement_text=(
                    "Candidate <script>window.pwned=true</script> remains source-grounded."
                ),
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
            generator_id="browser-test",
            revision="1",
            license="internal",
            prompt_version="reading-room-browser-v1",
        ),
        review_budget=ReviewBudget(),
        code_version="browser-test",
    )
    with open_library(world.primary.library_id, data_root=world.data_home) as repository:
        result = SQLiteMeaningRepository(repository).import_proposal(
            request=request,
            proposal=proposal,
        )
    assert result.run.status.value == "succeeded"


@contextmanager
def _running_server(world: ApiWorld) -> Iterator[str]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    origin = f"http://127.0.0.1:{port}"
    application = create_reading_room_app(
        library_id=world.primary.library_id,
        data_home=world.data_home,
        corpus_snapshot_id=world.primary.corpus_snapshot_id,
        access_policy_id=world.primary.access_policy_id,
        scope=_scope(world),
        allowed_origin=origin,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            application,
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
        raise RuntimeError("Reading Room test server did not start")
    try:
        yield origin
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("Reading Room test server did not stop")


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


def _document_fits_viewport(page: Page) -> bool:
    return bool(
        page.evaluate(
            "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )
    )


def test_reading_room_is_inert_traceable_and_responsive_in_real_chromium(
    tmp_path: Path,
) -> None:
    world = build_api_world(tmp_path)
    _import_candidate(world)
    with _running_server(world) as origin:
        playwright, browser = _launch_browser()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            console_errors: list[str] = []
            page.on(
                "console",
                lambda message: (
                    console_errors.append(message.text) if message.type == "error" else None
                ),
            )
            response = page.goto(origin, wait_until="networkidle")
            assert response is not None
            assert response.status == 200
            assert page.locator("h1").inner_text().startswith("Стан пам’яті:")
            assert page.locator(".connection-list li").count() >= 3
            assert page.locator("script, img, form, button, input").count() == 0
            assert page.evaluate("() => window.pwned") is None
            assert _document_fits_viewport(page)

            page.keyboard.press("Tab")
            assert page.locator(":focus").get_attribute("class") == "skip-link"
            page.locator(".corpus-inventory summary").click()
            assert (
                page.locator(
                    f".corpus-inventory code:text-is('{world.primary.source_fragment_id}')"
                ).count()
                == 1
            )
            assert world.primary.text not in page.content()
            assert _document_fits_viewport(page)
            relation_text = page.locator(".relation-triple").all_inner_texts()
            assert any(
                "Candidate" in text and "має доказ" in text and "evidence_" in text
                for text in relation_text
            )
            with page.expect_navigation(wait_until="domcontentloaded") as navigation:
                page.locator(".source-chip").first.click()
            assert navigation.value.status == 200
            assert page.locator('[data-source-highlight="true"]').inner_text() == world.primary.text
            assert page.evaluate("() => window.pwned") is None
            page.go_back(wait_until="networkidle")

            capture_root = os.environ.get("DITHYRAMBA_READING_ROOM_CAPTURE")
            if capture_root:
                target = Path(capture_root)
                target.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=target / "desktop.png", full_page=True)

            page.set_viewport_size({"width": 390, "height": 844})
            page.reload(wait_until="networkidle")
            assert _document_fits_viewport(page)
            assert page.locator(".question-nav").count() == 1
            assert page.locator(".inspector").count() == 1
            unresolved = page.locator('.relation-verb[data-review-state="unresolved"]').first
            assert (
                unresolved.evaluate(
                    "element => getComputedStyle(element, '::before').borderLeftStyle"
                )
                == "dashed"
            )
            assert (
                unresolved.evaluate(
                    "element => getComputedStyle(element, '::after').borderBottomStyle"
                )
                == "dashed"
            )
            if capture_root:
                page.screenshot(path=Path(capture_root) / "mobile.png", full_page=True)
            assert console_errors == []
        finally:
            browser.close()
            playwright.stop()
