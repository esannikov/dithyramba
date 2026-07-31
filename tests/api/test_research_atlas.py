"""Strict Research Atlas contract and GET-only renderer acceptance."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from dithyramba.api import create_research_atlas_app
from dithyramba.api.config import LoopbackApiConfig
from dithyramba.api.research_atlas import (
    ResearchAtlasWebConfig,
    _chip_attribution,
    _evidence_views,
    _first_evidence,
    _first_span_evidence,
    _object_id,
    _selected_object,
    _selected_trace_span,
    _source_binding_view,
    _trace_text_parts,
)
from dithyramba.atlas import load_research_atlas, load_research_projection
from dithyramba.cli import app

_CSP = (
    "default-src 'self'; script-src 'none'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'"
)


def _payload() -> dict[str, object]:
    return {
        "schema_id": "dithyramba.research_atlas/1.0",
        "atlas_id": "atlas_tesla",
        "case_id": "tesla_case",
        "title": "Нікола Тесла: життя, робота й міфи",
        "subject": "Нікола Тесла",
        "subtitle": "Доказовий стіл для документального дослідження.",
        "generated_at": "2026-07-23T12:00:00Z",
        "language": "uk",
        "read_only": True,
        "corpus": {
            "label": "Tesla research corpus",
            "source_count": 2,
            "source_family_count": 2,
            "scope_note": "Листи, патенти й академічна історіографія.",
            "cutoff_note": "Корпус зафіксовано 23 липня 2026 року.",
        },
        "sources": [
            {
                "source_id": "source_letter",
                "title": "My Inventions",
                "creator": "Nikola Tesla",
                "source_kind": "letter",
                "voice_kind": "first_person",
                "date_label": "1919",
                "publisher_or_archive": "Electrical Experimenter",
                "independence_group": "family_tesla",
                "verification_state": "verified",
                "original_url": "https://example.org/tesla",
                "artifact_path": "artifacts/tesla.md",
                "rights_note": "Public-domain text; local excerpt recorded.",
            },
            {
                "source_id": "source_study",
                "title": "Tesla in historiography",
                "creator": "Researcher <script>alert(1)</script>",
                "source_kind": "scholarly_article",
                "voice_kind": "scholarly",
                "date_label": "2020",
                "publisher_or_archive": "Journal",
                "independence_group": "family_study",
                "verification_state": "partial",
                "original_url": None,
                "artifact_path": None,
                "rights_note": "Citation metadata only.",
            },
        ],
        "evidence": [
            {
                "evidence_id": "evidence_letter",
                "source_id": "source_letter",
                "locator": "chapter 1, paragraph 3",
                "excerpt": "The first-person passage.",
                "role": "supports",
                "asserting_voice": "Nikola Tesla, retrospective first-person account",
                "limitation": "Retrospective memory is not an independent chronology.",
                "source_fragment_id": "fragment_letter",
                "fragment_text_sha256": "0" * 64,
                "source_address": {
                    "kind": "markdown",
                    "heading_path": ["Exact source"],
                    "line_start": 2,
                    "line_end": 2,
                    "char_start": 15,
                    "char_end": 40,
                },
            },
            {
                "evidence_id": "evidence_study",
                "source_id": "source_study",
                "locator": "p. 12",
                "excerpt": None,
                "role": "qualifies",
                "asserting_voice": "The article author",
                "limitation": None,
            },
        ],
        "questions": [
            {
                "question_id": "question_origin",
                "prompt": "Як Тесла сам описував походження винахідницького методу?",
                "short_answer": "Він пов’язував його з уявним випробуванням машин.",
                "state": "qualified",
                "evidence_ids": ["evidence_letter", "evidence_study"],
                "trace_spans": [
                    {
                        "span_id": "trace_question_method",
                        "start": 4,
                        "end": 48,
                        "text": "пов’язував його з уявним випробуванням машин",
                        "kind": "fact",
                        "evidence_ids": ["evidence_letter"],
                    }
                ],
                "gap": "Потрібна рання незалежна фіксація.",
                "tags": ["method"],
            },
            {
                "question_id": "question_open",
                "prompt": "Що лишається невідомим?",
                "short_answer": "Архівного доказу ще немає.",
                "state": "open",
                "evidence_ids": [],
                "gap": "Перевірити архів.",
                "tags": [],
            },
        ],
        "hypotheses": [
            {
                "hypothesis_id": "hypothesis_visual",
                "title": "Уявна симуляція була частиною методу",
                "synthesis": (
                    "Особистий опис і пізніший аналіз сходяться, але не є рівними голосами."
                ),
                "state": "working",
                "evidence_ids": ["evidence_letter"],
                "counterevidence_ids": ["evidence_study"],
                "trace_spans": [
                    {
                        "span_id": "trace_hypothesis_first_person",
                        "start": 0,
                        "end": 14,
                        "text": "Особистий опис",
                        "kind": "fact",
                        "evidence_ids": ["evidence_letter"],
                    },
                    {
                        "span_id": "trace_hypothesis_synthesis",
                        "start": 17,
                        "end": 42,
                        "text": "пізніший аналіз сходяться",
                        "kind": "synthesis",
                        "evidence_ids": ["evidence_study"],
                    },
                ],
                "gap": "Знайти сучасне Теслі свідчення.",
                "question_ids": ["question_origin"],
            },
            {
                "hypothesis_id": "hypothesis_context",
                "title": "Пізня автобіографія змінила акценти",
                "synthesis": "Це окрема інтерпретація.",
                "state": "open",
                "evidence_ids": [],
                "counterevidence_ids": [],
                "gap": "Порівняти редакції.",
                "question_ids": ["question_open"],
            },
        ],
        "relations": [
            {
                "relation_id": "relation_one",
                "source_hypothesis_id": "hypothesis_context",
                "target_hypothesis_id": "hypothesis_visual",
                "kind": "qualifies",
                "evidence_ids": ["evidence_study"],
                "explanation": "Пізній жанр обмежує буквальне читання.",
            }
        ],
        "timeline": [
            {
                "event_id": "event_publication",
                "date_start": "1919",
                "date_end": None,
                "date_label": "1919",
                "title": "Публікація автобіографічної серії",
                "summary": "Тесла оприлюднює ретроспективний опис методу.",
                "state": "confirmed",
                "evidence_ids": ["evidence_letter"],
                "question_ids": ["question_origin"],
                "hypothesis_ids": ["hypothesis_visual"],
            }
        ],
        "gaps": [
            {
                "gap_id": "gap_early_record",
                "label": "Ранній запис про метод",
                "why_it_matters": "Відділяє пізню пам’ять від практики часу винаходу.",
                "next_evidence": "Лист або лабораторний запис до 1900 року.",
                "related_question_ids": ["question_origin"],
                "related_hypothesis_ids": ["hypothesis_visual"],
            }
        ],
        "methodology_note": (
            "Система перевіряє ідентичність джерела, точний локатор, незалежність "
            "родини та відповідність уривка висновку."
        ),
    }


def _manifest(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "atlas.json"
    path.write_text(
        json.dumps(payload or _payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _projection_payload(base_manifest_hash: str, fragment_hash: str) -> dict[str, object]:
    return {
        "schema_id": "dithyramba.research_projection/1.0",
        "projection_id": "projection_tesla",
        "case_id": "tesla_case",
        "base_manifest_hash": base_manifest_hash,
        "title": "Ширша дослідницька памʼять",
        "summary": "Кандидатні матеріали залишаються видимими без promotion.",
        "generated_at": "2026-07-24T12:00:00Z",
        "collections": [
            {
                "collection_id": "collection_backbone",
                "label": "Біографічний хребет",
                "kind": "event",
                "record_count": 1,
                "description": "Повна часова поверхня кандидатних подій.",
                "status_note": "Кандидати потребують окремого review.",
            }
        ],
        "periods": [
            {
                "period_id": "period_early",
                "label": "Ранній період",
                "date_start": "1853",
                "date_end": "1880",
                "summary": "Походження, навчання і перехід до практики.",
            }
        ],
        "themes": [
            {
                "theme_id": "theme_method",
                "title": "Як формується метод",
                "summary": "Ширший матеріал навколо становлення практики.",
                "what_it_changes": "Відділяє перевірений висновок від поля кандидатів.",
                "hypothesis_ids": ["hypothesis_visual"],
                "question_ids": ["question_origin"],
                "gap_ids": ["gap_early_record"],
            }
        ],
        "materials": [
            {
                "material_id": "material_candidate",
                "kind": "event",
                "state": "candidate",
                "title": "Ранній кандидатний запис про метод",
                "date_start": "1880",
                "date_label": "August",
                "period_id": "period_early",
                "theme_ids": ["theme_method"],
                "source_id": "source_letter",
                "fragment_text_sha256": fragment_hash,
                "source_address": {
                    "kind": "markdown",
                    "heading_path": ["Exact source"],
                    "line_start": 2,
                    "line_end": 2,
                    "char_start": 15,
                    "char_end": 40,
                },
                "locator": "Exact source · line 2",
                "asserting_voice": "Editorial chronology",
                "limitation": "Кандидат ще не пройшов EvidenceCoverageGate.",
                "review_priority": True,
            }
        ],
    }


def _projection(
    tmp_path: Path,
    *,
    base_manifest_hash: str,
    fragment_hash: str,
) -> Path:
    path = tmp_path / "projection.json"
    path.write_text(
        json.dumps(
            _projection_payload(base_manifest_hash, fragment_hash),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_research_atlas_is_traceable_script_free_and_get_only(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    app = create_research_atlas_app(
        manifest_path=path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        overview = client.get("/")
        question = client.get(
            "/?view=questions&selected=question_origin"
            "&span=trace_question_method&evidence=evidence_letter"
        )
        hypothesis = client.get(
            "/?view=hypotheses&selected=hypothesis_visual"
            "&span=trace_hypothesis_synthesis&evidence=evidence_study"
        )
        timeline = client.get("/?view=timeline&selected=event_publication")
        gaps = client.get("/?view=gaps")
        sources = client.get("/?view=sources&selected=source_study")
        stylesheet = client.get("/research-atlas.css")
        manifest = client.get("/manifest.json")
        mutation = client.post("/", content=b"ignored")

    assert overview.status_code == 200
    assert overview.headers["content-security-policy"] == _CSP
    assert "<script" not in overview.text.casefold()
    assert "<form" not in overview.text.casefold()
    assert "Ключові питання" in overview.text
    assert "пов’язував його з уявним випробуванням машин" in question.text
    assert '<strong class="source-chip-title">My Inventions</strong>' in question.text
    assert '<small class="source-chip-attribution">Nikola Tesla · 1919</small>' in question.text
    assert 'class="traceable-span trace-fact is-selected"' in question.text
    assert "Підкреслений текст відкриває точний доказ." in question.text
    assert "перевірюване твердження" in question.text
    assert 'class="source-chip role-supports is-active"' in question.text
    assert "Nikola Tesla · My Inventions · 1919 · chapter 1, paragraph 3" not in question.text
    assert "The first-person passage." in question.text
    assert "Ретроспективна" not in question.text
    assert "Retrospective memory is not an independent chronology." in question.text
    assert "Як пов’язані інтерпретації" in hypothesis.text
    assert 'class="traceable-span trace-synthesis is-selected"' in hypothesis.text
    assert "синтез джерел" in hypothesis.text
    assert 'class="source-chip role-qualifies is-active"' in hypothesis.text
    assert "Пізній жанр обмежує буквальне читання." in hypothesis.text
    assert "Хронологія як маршрут до доказів" in timeline.text
    timeline_source_chip = timeline.text.split('class="source-chip ', maxsplit=1)[1].split(
        "</a>", maxsplit=1
    )[0]
    assert "chapter 1, paragraph 3" not in timeline_source_chip
    assert "Що ще не доведено" in gaps.text
    assert "Researcher &lt;script&gt;alert(1)&lt;/script&gt;" in sources.text
    assert "<script>alert(1)</script>" not in sources.text
    assert stylesheet.status_code == 200
    assert "--paper: #f3eee5" in stylesheet.text
    assert manifest.status_code == 200
    assert manifest.json()["schema_id"] == "dithyramba.research_atlas/1.0"
    assert manifest.json()["manifest_hash"] == manifest.headers["etag"].strip('"')
    assert mutation.status_code == 401


def test_projection_keeps_candidates_visible_but_separate_from_evidence(
    tmp_path: Path,
) -> None:
    payload = _payload()
    exact_line = "<script>alert(1)</script>"
    fragment_hash = sha256(exact_line.encode("utf-8")).hexdigest()
    payload["evidence"][0]["fragment_text_sha256"] = fragment_hash  # type: ignore[index]
    manifest_path = _manifest(tmp_path, payload)
    atlas = load_research_atlas(manifest_path)
    projection_path = _projection(
        tmp_path,
        base_manifest_hash=atlas.manifest_hash,
        fragment_hash=fragment_hash,
    )
    artifact = tmp_path / "artifacts" / "tesla.md"
    artifact.parent.mkdir()
    artifact.write_text(f"# Exact source\n{exact_line}", encoding="utf-8")
    app = create_research_atlas_app(
        manifest_path=manifest_path,
        projection_path=projection_path,
        artifact_root=tmp_path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )

    with TestClient(app) as client:
        overview = client.get("/")
        materials = client.get("/?view=materials&theme=theme_method")
        selected = client.get("/?view=materials&theme=theme_method&material=material_candidate")
        exact = client.get("/artifacts/source_letter?material=material_candidate")
        projection_json = client.get("/projection.json")

    assert "Панорама розуміння" in overview.text
    assert "Поточний статус:" in overview.text
    assert "Кандидатні матеріали залишаються видимими без promotion." in overview.text
    assert "Ширше поле пам’яті" in materials.text
    assert "Ранній кандидатний запис про метод" in materials.text
    assert "не використовується як опора" in selected.text
    assert "Заморожена нотатка на момент збирання evidence core:" in selected.text
    assert "Матеріал-кандидат" in exact.text
    assert 'class="is-evidence"' in exact.text
    assert "<script>alert(1)</script>" not in exact.text
    assert projection_json.status_code == 200
    assert projection_json.json()["schema_id"] == "dithyramba.research_projection/1.0"


def test_projection_rejects_wrong_base_manifest_hash(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    projection_path = _projection(
        tmp_path,
        base_manifest_hash="0" * 64,
        fragment_hash="1" * 64,
    )

    with pytest.raises(ValueError, match="different Atlas hash"):
        create_research_atlas_app(
            manifest_path=manifest_path,
            projection_path=projection_path,
            allowed_origin="http://testserver",
            test_only_allow_testserver=True,
        )


def test_projection_rejects_invalid_internal_references(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    atlas = load_research_atlas(manifest_path)

    def rejected(
        label: str,
        payload: dict[str, object],
        match: str,
    ) -> None:
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with pytest.raises((ValidationError, ValueError), match=match):
            load_research_projection(path, atlas=atlas)

    payload = _projection_payload(atlas.manifest_hash, "0" * 64)
    payload["schema_id"] = "dithyramba.research_projection/0.9"
    rejected("schema", payload, "schema_id must be")

    payload = _projection_payload(atlas.manifest_hash, "0" * 64)
    payload["periods"][0]["date_start"] = "1900"  # type: ignore[index]
    rejected("reversed_period", payload, "date range is reversed")

    payload = _projection_payload(atlas.manifest_hash, "0" * 64)
    collections = payload["collections"]
    assert isinstance(collections, list)
    collections.append(deepcopy(collections[0]))
    rejected("duplicate_collection", payload, "identifiers must be unique")

    payload = _projection_payload(atlas.manifest_hash, "0" * 64)
    payload["materials"][0]["period_id"] = "period_missing"  # type: ignore[index]
    rejected("missing_period", payload, "references missing period")

    payload = _projection_payload(atlas.manifest_hash, "0" * 64)
    payload["materials"][0]["theme_ids"] = ["theme_missing"]  # type: ignore[index]
    rejected("missing_theme", payload, "references missing themes")


def test_projection_rejects_wrong_case_and_atlas_references(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    atlas = load_research_atlas(manifest_path)

    def rejected(
        label: str,
        payload: dict[str, object],
        match: str,
    ) -> None:
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_research_projection(path, atlas=atlas)

    payload = _projection_payload(atlas.manifest_hash, "0" * 64)
    payload["case_id"] = "other_case"
    rejected("wrong_case", payload, "case_id does not match")

    payload = _projection_payload(atlas.manifest_hash, "0" * 64)
    payload["materials"][0]["source_id"] = "source_missing"  # type: ignore[index]
    rejected("missing_source", payload, "references missing Atlas source")

    for field, missing, match in (
        ("question_ids", "question_missing", "projection theme question"),
        ("hypothesis_ids", "hypothesis_missing", "projection theme hypothesis"),
        ("gap_ids", "gap_missing", "projection theme gap"),
    ):
        payload = _projection_payload(atlas.manifest_hash, "0" * 64)
        payload["themes"][0][field] = [missing]  # type: ignore[index]
        rejected(f"missing_{field}", payload, match)

    with pytest.raises(ValueError, match="must be a regular file"):
        load_research_projection(tmp_path, atlas=atlas)


def test_atlas_selection_falls_back_without_disclosing_files(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    app = create_research_atlas_app(
        manifest_path=path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        page = client.get("/?view=sources&selected=source_missing&evidence=evidence_missing")
        invalid = client.get("/?view=technical")
        invalid_span = client.get("/?view=questions&selected=question_origin&span=trace_missing")
        wrong_host = client.get("/", headers={"host": "example.org"})

    assert page.status_code == 200
    assert "My Inventions" in page.text
    assert "artifacts/tesla.md" in page.text
    assert "file://" not in page.text
    assert invalid.status_code == 422
    assert invalid_span.status_code == 404
    assert wrong_host.status_code == 400


def test_atlas_opens_only_manifest_authorized_case_artifacts(tmp_path: Path) -> None:
    payload = _payload()
    exact_line = "<script>alert(1)</script>"
    payload["evidence"][0]["fragment_text_sha256"] = sha256(  # type: ignore[index]
        exact_line.encode("utf-8")
    ).hexdigest()
    path = _manifest(tmp_path, payload)
    artifact = tmp_path / "artifacts" / "tesla.md"
    artifact.parent.mkdir()
    artifact.write_text(f"# Exact source\n{exact_line}", encoding="utf-8")
    app = create_research_atlas_app(
        manifest_path=path,
        artifact_root=tmp_path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    app_without_root = create_research_atlas_app(
        manifest_path=path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        atlas_page = client.get("/?view=questions&evidence=evidence_letter")
        source = client.get("/artifacts/source_letter?evidence=evidence_letter")
        whole_source = client.get("/artifacts/source_letter?evidence=evidence_letter&full=true")
        wrong_evidence = client.get("/artifacts/source_letter?evidence=evidence_study")
        unknown = client.get("/artifacts/source_missing")
        no_artifact = client.get("/artifacts/source_study")
    with TestClient(app_without_root) as client:
        unauthorized = client.get("/artifacts/source_letter")

    assert "Відкрити точний фрагмент" in atlas_page.text
    assert source.status_code == 200
    assert "# Exact source" in source.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in source.text
    assert "<script>alert(1)</script>" not in source.text
    assert 'id="L2"' in source.text
    assert 'class="is-evidence"' in source.text
    assert "fragment_letter" in source.text
    assert "Показати весь локальний документ" in source.text
    assert whole_source.status_code == 200
    assert wrong_evidence.status_code == 404
    assert source.headers["content-security-policy"] == _CSP
    assert unknown.status_code == 404
    assert no_artifact.status_code == 404
    assert unauthorized.status_code == 404

    artifact.write_text("# Exact source\nChanged", encoding="utf-8")
    with TestClient(app) as client:
        stale = client.get("/artifacts/source_letter?evidence=evidence_letter")
    assert stale.status_code == 409
    assert stale.json() == {"detail": "bound source fragment no longer matches the manifest"}


def test_atlas_artifact_viewer_handles_binary_type_size_and_encoding(
    tmp_path: Path,
) -> None:
    payload = _payload()
    source = payload["sources"][0]  # type: ignore[index]
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    source["artifact_path"] = "artifacts/test.pdf"
    (artifact_dir / "test.pdf").write_bytes(b"%PDF-1.4 test")
    app = create_research_atlas_app(
        manifest_path=_manifest(tmp_path, payload),
        artifact_root=tmp_path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        pdf = client.get("/artifacts/source_letter")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"

    payload = _payload()
    payload["sources"][0]["artifact_path"] = "artifacts/test.bin"  # type: ignore[index]
    (artifact_dir / "test.bin").write_bytes(b"\xff")
    app = create_research_atlas_app(
        manifest_path=_manifest(tmp_path, payload),
        artifact_root=tmp_path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        unsupported = client.get("/artifacts/source_letter")
    assert unsupported.status_code == 404

    payload = _payload()
    payload["sources"][0]["artifact_path"] = "artifacts/bad.md"  # type: ignore[index]
    (artifact_dir / "bad.md").write_bytes(b"\xff")
    app = create_research_atlas_app(
        manifest_path=_manifest(tmp_path, payload),
        artifact_root=tmp_path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        invalid_encoding = client.get("/artifacts/source_letter")
    assert invalid_encoding.status_code == 404

    payload = _payload()
    payload["sources"][0]["artifact_path"] = "artifacts/large.md"  # type: ignore[index]
    with (artifact_dir / "large.md").open("wb") as stream:
        stream.truncate(25 * 1024 * 1024 + 1)
    app = create_research_atlas_app(
        manifest_path=_manifest(tmp_path, payload),
        artifact_root=tmp_path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        oversized = client.get("/artifacts/source_letter")
    assert oversized.status_code == 413

    payload = _payload()
    payload["evidence"][0]["source_address"]["line_start"] = 20  # type: ignore[index]
    payload["evidence"][0]["source_address"]["line_end"] = 20  # type: ignore[index]
    payload["sources"][0]["artifact_path"] = "artifacts/short.md"  # type: ignore[index]
    (artifact_dir / "short.md").write_text("one line", encoding="utf-8")
    app = create_research_atlas_app(
        manifest_path=_manifest(tmp_path, payload),
        artifact_root=tmp_path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        out_of_range = client.get("/artifacts/source_letter?evidence=evidence_letter")
    assert out_of_range.status_code == 409


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update(schema_id="wrong"),
            "schema_id must be",
        ),
        (
            lambda payload: payload.update(read_only=False),
            "read-only",
        ),
        (
            lambda payload: payload["corpus"].update(source_count=99),
            "source_count",
        ),
        (
            lambda payload: payload["evidence"][0].update(source_id="source_missing"),
            "evidence source",
        ),
        (
            lambda payload: payload["questions"][0].update(evidence_ids=["evidence_missing"]),
            "question.*evidence",
        ),
        (
            lambda payload: payload["relations"][0].update(
                target_hypothesis_id="hypothesis_missing"
            ),
            "relation hypothesis",
        ),
    ],
)
def test_manifest_rejects_broken_contracts(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    payload = deepcopy(_payload())
    mutate(payload)  # type: ignore[operator]
    path = _manifest(tmp_path, payload)
    with pytest.raises(ValidationError, match=message):
        load_research_atlas(path)


def test_manifest_rejects_invalid_trace_spans(tmp_path: Path) -> None:
    payload = deepcopy(_payload())
    payload["questions"][0]["trace_spans"][0]["text"] = "wrong displayed text"  # type: ignore[index]
    payload["questions"][0]["trace_spans"][0]["end"] = 24  # type: ignore[index]
    with pytest.raises(ValidationError, match="exact displayed text"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    first_span = payload["hypotheses"][0]["trace_spans"][0]  # type: ignore[index]
    second_span = payload["hypotheses"][0]["trace_spans"][1]  # type: ignore[index]
    second_span["start"] = first_span["start"]
    second_span["end"] = first_span["end"]
    second_span["text"] = first_span["text"]
    with pytest.raises(ValidationError, match="ordered and non-overlapping"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    payload["questions"][0]["trace_spans"][0]["evidence_ids"] = [  # type: ignore[index]
        "evidence_study"
    ]
    payload["questions"][0]["evidence_ids"] = ["evidence_letter"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="references missing IDs"):
        load_research_atlas(_manifest(tmp_path, payload))


def test_manifest_rejects_unsafe_source_addresses_and_self_relations(tmp_path: Path) -> None:
    payload = deepcopy(_payload())
    payload["sources"][0]["original_url"] = "javascript:alert(1)"  # type: ignore[index]
    with pytest.raises(ValidationError, match="HTTP"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    payload["sources"][0]["artifact_path"] = "../outside.txt"  # type: ignore[index]
    with pytest.raises(ValidationError, match="case-relative"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    payload["relations"][0]["target_hypothesis_id"] = "hypothesis_context"  # type: ignore[index]
    with pytest.raises(ValidationError, match="cannot target itself"):
        load_research_atlas(_manifest(tmp_path, payload))


def test_manifest_requires_evidence_for_promoted_findings(tmp_path: Path) -> None:
    payload = deepcopy(_payload())
    payload["questions"][0]["evidence_ids"] = []  # type: ignore[index]
    with pytest.raises(ValidationError, match="non-open question"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    payload["hypotheses"][0]["state"] = "established"  # type: ignore[index]
    payload["hypotheses"][0]["evidence_ids"] = []  # type: ignore[index]
    payload["hypotheses"][0]["counterevidence_ids"] = []  # type: ignore[index]
    with pytest.raises(ValidationError, match="established hypothesis"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    payload["timeline"][0]["evidence_ids"] = []  # type: ignore[index]
    with pytest.raises(ValidationError, match="timeline event"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    payload["evidence"][0]["source_address"] = None  # type: ignore[index]
    with pytest.raises(ValidationError, match="binding must be complete"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    payload["evidence"][0]["source_address"]["line_end"] = 1  # type: ignore[index]
    with pytest.raises(ValidationError, match="line range is reversed"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    payload["evidence"][0]["source_address"]["char_end"] = 1  # type: ignore[index]
    with pytest.raises(ValidationError, match="character range is reversed"):
        load_research_atlas(_manifest(tmp_path, payload))

    payload = deepcopy(_payload())
    payload["evidence"][0]["excerpt"] = "x" * 1_001  # type: ignore[index]
    with pytest.raises(ValidationError, match="at most 1000"):
        load_research_atlas(_manifest(tmp_path, payload))


def test_atlas_optional_routes_fail_closed_and_paginate(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    app_without_projection = create_research_atlas_app(
        manifest_path=manifest_path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app_without_projection) as client:
        assert client.get("/projection.json").status_code == 404
        assert client.get("/?view=materials").status_code == 404
        assert client.get("/?view=sources&page=2").status_code == 404
        assert client.get("/artifacts/source_missing").status_code == 404

    artifact_root = tmp_path / "data"
    artifact = artifact_root / "artifacts" / "tesla.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("Exact source\nBound passage\nTail\n", encoding="utf-8")
    payload = _payload()
    exact_hash = sha256(b"Bound passage").hexdigest()
    payload["evidence"][0]["fragment_text_sha256"] = exact_hash  # type: ignore[index]
    payload["evidence"][0]["source_address"]["line_start"] = 2  # type: ignore[index]
    payload["evidence"][0]["source_address"]["line_end"] = 2  # type: ignore[index]
    manifest_path = _manifest(tmp_path, payload)
    projection_path = _projection(
        tmp_path,
        base_manifest_hash=load_research_atlas(manifest_path).manifest_hash,
        fragment_hash=exact_hash,
    )
    app_with_projection = create_research_atlas_app(
        manifest_path=manifest_path,
        projection_path=projection_path,
        artifact_root=artifact_root,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app_with_projection) as client:
        projection = client.get("/projection.json")
        assert projection.status_code == 200
        assert projection.headers["etag"].strip('"') == projection.json()["projection_hash"]
        assert client.get("/?view=materials&theme=missing").status_code == 404
        assert client.get("/?view=materials&period=missing").status_code == 404
        assert client.get("/?view=materials&page=2").status_code == 404
        assert (
            client.get(
                "/artifacts/source_letter?evidence=evidence_letter&material=material_candidate"
            ).status_code
            == 422
        )
        assert client.get("/artifacts/source_letter?evidence=evidence_missing").status_code == 404
        assert client.get("/artifacts/source_letter?material=material_missing").status_code == 404
        material = client.get("/artifacts/source_letter?material=material_candidate&full=true")
        assert material.status_code == 200
        assert "Bound passage" in material.text


def test_atlas_exact_artifact_binding_detects_drift_and_range_errors(tmp_path: Path) -> None:
    artifact_root = tmp_path / "data"
    artifact = artifact_root / "artifacts" / "tesla.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("Exact source\nChanged passage\n", encoding="utf-8")
    payload = _payload()
    payload["evidence"][0]["source_address"]["line_start"] = 2  # type: ignore[index]
    payload["evidence"][0]["source_address"]["line_end"] = 2  # type: ignore[index]
    payload["evidence"][0]["fragment_text_sha256"] = sha256(b"Expected passage").hexdigest()  # type: ignore[index]
    manifest_path = _manifest(tmp_path, payload)
    app = create_research_atlas_app(
        manifest_path=manifest_path,
        artifact_root=artifact_root,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        drift = client.get("/artifacts/source_letter?evidence=evidence_letter")
        assert drift.status_code == 409

    payload["evidence"][0]["source_address"]["line_start"] = 9  # type: ignore[index]
    payload["evidence"][0]["source_address"]["line_end"] = 9  # type: ignore[index]
    manifest_path = _manifest(tmp_path, payload)
    app = create_research_atlas_app(
        manifest_path=manifest_path,
        artifact_root=artifact_root,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        out_of_range = client.get("/artifacts/source_letter?evidence=evidence_letter")
        assert out_of_range.status_code == 409


def test_atlas_view_helpers_keep_selection_and_trace_failures_explicit(
    tmp_path: Path,
) -> None:
    atlas = load_research_atlas(_manifest(tmp_path))
    manifest = atlas.manifest
    evidence_views = _evidence_views(manifest)

    fallback = _selected_object(manifest, view="sources", selected="source_missing")
    assert fallback == manifest.sources[0]
    assert _object_id(object()) is None
    assert _first_evidence(None, evidence_views=evidence_views) is None
    open_question = manifest.questions[1]
    assert _first_evidence(open_question, evidence_views=evidence_views) is None
    span = _selected_trace_span(
        manifest.questions[0],
        requested_span_id="trace_question_method",
    )
    assert span is not None
    assert _first_span_evidence(span, evidence_views=evidence_views) is not None
    parts = _trace_text_parts(
        manifest.questions[0].short_answer,
        manifest.questions[0].trace_spans,
        selected_span_id=None,
    )
    assert parts[-1].text.endswith(".")
    assert _source_binding_view(manifest.evidence[1], None, source_id="source_study") is None
    assert _source_binding_view(None, None, source_id="source_study") is None
    dated_title = manifest.sources[0].model_copy(update={"title": "My Inventions 1919"})
    assert _chip_attribution(dated_title) == "Nikola Tesla"


def test_atlas_rejects_invalid_root_symlink_and_material_without_projection(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path)
    with pytest.raises(ValueError, match="artifact_root"):
        ResearchAtlasWebConfig(
            loopback=LoopbackApiConfig(
                library_id="library_00000000000040008000000000000000",
                data_home=tmp_path,
                allowed_origin="http://testserver",
                test_only_allow_testserver=True,
            ),
            atlas=load_research_atlas(manifest_path),
            artifact_root=Path("."),
        )

    artifact_root = tmp_path / "data"
    artifacts = artifact_root / "artifacts"
    artifacts.mkdir(parents=True)
    (tmp_path / "outside.md").write_text("outside", encoding="utf-8")
    (artifacts / "link.md").symlink_to(tmp_path / "outside.md")
    payload = _payload()
    payload["sources"][0]["artifact_path"] = "artifacts/link.md"  # type: ignore[index]
    app = create_research_atlas_app(
        manifest_path=_manifest(tmp_path, payload),
        artifact_root=artifact_root,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        assert client.get("/artifacts/source_letter").status_code == 404

    (artifacts / "tesla.md").write_text("Exact source\nBound passage\n", encoding="utf-8")
    payload = _payload()
    payload["evidence"][0]["fragment_text_sha256"] = sha256(b"Bound passage").hexdigest()  # type: ignore[index]
    payload["evidence"][0]["source_address"]["line_start"] = 2  # type: ignore[index]
    payload["evidence"][0]["source_address"]["line_end"] = 2  # type: ignore[index]
    app = create_research_atlas_app(
        manifest_path=_manifest(tmp_path, payload),
        artifact_root=artifact_root,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        assert client.get("/artifacts/source_letter?material=material_candidate").status_code == 404


def test_atlas_valid_period_filter_opens_materials(tmp_path: Path) -> None:
    payload = _payload()
    artifact_root = tmp_path / "data"
    artifact = artifact_root / "artifacts" / "tesla.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("Exact source\nBound passage\n", encoding="utf-8")
    exact_hash = sha256(b"Bound passage").hexdigest()
    payload["evidence"][0]["fragment_text_sha256"] = exact_hash  # type: ignore[index]
    manifest_path = _manifest(tmp_path, payload)
    projection_path = _projection(
        tmp_path,
        base_manifest_hash=load_research_atlas(manifest_path).manifest_hash,
        fragment_hash=exact_hash,
    )
    app = create_research_atlas_app(
        manifest_path=manifest_path,
        projection_path=projection_path,
        artifact_root=artifact_root,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    with TestClient(app) as client:
        response = client.get("/?view=materials&period=period_early")
    assert response.status_code == 200
    assert "Ранній кандидатний запис про метод" in response.text


def test_atlas_cli_validates_absolute_manifest_and_starts_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _manifest(tmp_path)
    atlas = load_research_atlas(path)
    projection = _projection(
        tmp_path,
        base_manifest_hash=atlas.manifest_hash,
        fragment_hash="0" * 64,
    )
    called: dict[str, object] = {}

    def fake_run(application: object, **kwargs: object) -> None:
        called["application"] = application
        called.update(kwargs)

    monkeypatch.setattr("dithyramba.cli.uvicorn.run", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "atlas",
            "--manifest",
            str(path),
            "--projection",
            str(projection),
            "--artifact-root",
            str(tmp_path),
            "--port",
            "8361",
        ],
    )
    relative = CliRunner().invoke(
        app,
        ["atlas", "--manifest", "atlas.json"],
    )

    assert result.exit_code == 0
    assert "research_atlas: http://127.0.0.1:8361" in result.stdout
    assert "manifest_hash:" in result.stdout
    assert "projection_hash:" in result.stdout
    assert "mode: read-only" in result.stdout
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8361
    assert relative.exit_code == 2
    assert "manifest must be an absolute path" in relative.output

    relative_root = CliRunner().invoke(
        app,
        [
            "atlas",
            "--manifest",
            str(path),
            "--artifact-root",
            "case",
        ],
    )
    assert relative_root.exit_code == 2
    assert "artifact-root must be an absolute path" in relative_root.output

    relative_projection = CliRunner().invoke(
        app,
        [
            "atlas",
            "--manifest",
            str(path),
            "--projection",
            "projection.json",
        ],
    )
    assert relative_projection.exit_code == 2
    assert "projection must be an absolute path" in relative_projection.output
