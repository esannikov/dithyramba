"""Deterministic migration of historical Research Atlas manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dithyramba.contracts import canonical_json_bytes, sha256_hex

from .models import ResearchAtlasManifest

_VERIFICATION_STATES = {
    "body_proof_ready": "verified",
    "full_text_ready": "verified",
    "abstract": "partial",
    "metadata_only": "unverified",
    "verified": "verified",
    "partial": "partial",
    "unverified": "unverified",
}
_FINDING_STATES = {
    "confirmed": "confirmed",
    "qualified": "qualified",
    "contested": "contested",
    "refuted": "refuted",
    "unknown": "open",
    "open": "open",
    "model": "qualified",
}
_HYPOTHESIS_STATES = {
    "established": "established",
    "working": "working",
    "contested": "contested",
    "refuted": "refuted",
    "open": "open",
    "model": "working",
    "inference": "working",
    "hypothesis": "working",
    "fiction": "working",
}


class AtlasMigrationReceipt(BaseModel):
    """Auditable summary of one deterministic Atlas migration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str = "dithyramba.atlas_migration_receipt/1.0"
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    trace_span_count: int = Field(ge=0)
    exact_binding_count: int = Field(ge=0)
    omitted_binding_count: int = Field(ge=0)
    normalized_identifier_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class MigratedResearchAtlas:
    manifest: ResearchAtlasManifest
    receipt: AtlasMigrationReceipt


class _IdentifierMap:
    def __init__(self) -> None:
        self._values: dict[tuple[str, str], str] = {}
        self._owners: dict[tuple[str, str], str] = {}
        self.normalized_count = 0

    def get(self, raw_value: str, *, domain: str, prefix: str) -> str:
        key = (domain, raw_value)
        if key in self._values:
            return self._values[key]
        value = _safe_identifier(raw_value, prefix=prefix)
        owner_key = (domain, value)
        existing_owner = self._owners.get(owner_key)
        if existing_owner is not None and existing_owner != raw_value:
            digest = sha256_hex(raw_value.encode("utf-8"))[:12]
            value = f"{value[:82]}_{digest}"
        self._values[key] = value
        self._owners[(domain, value)] = raw_value
        if value != raw_value:
            self.normalized_count += 1
        return value


def migrate_legacy_research_atlas(
    payload: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
    trace_overrides: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> MigratedResearchAtlas:
    """Return a strict current manifest without mutating the historical input."""

    raw = deepcopy(dict(payload))
    identifiers = _IdentifierMap()
    overrides = trace_overrides or {}

    sources = raw.get("sources", [])
    evidence = raw.get("evidence", [])
    questions = raw.get("questions", [])
    hypotheses = raw.get("hypotheses", [])
    relations = raw.get("relations", [])
    timeline = raw.get("timeline", [])
    gaps = raw.get("gaps", [])
    if not all(
        isinstance(value, list)
        for value in (sources, evidence, questions, hypotheses, relations, timeline, gaps)
    ):
        raise ValueError("legacy Atlas collections must be JSON arrays")

    source_ids = {
        str(item["source_id"]): identifiers.get(
            str(item["source_id"]),
            domain="source",
            prefix="source",
        )
        for item in sources
    }
    evidence_ids = {
        str(item["evidence_id"]): identifiers.get(
            str(item["evidence_id"]),
            domain="evidence",
            prefix="evidence",
        )
        for item in evidence
    }
    question_ids = {
        str(item["question_id"]): identifiers.get(
            str(item["question_id"]),
            domain="question",
            prefix="question",
        )
        for item in questions
    }
    hypothesis_ids = {
        str(item["hypothesis_id"]): identifiers.get(
            str(item["hypothesis_id"]),
            domain="hypothesis",
            prefix="hypothesis",
        )
        for item in hypotheses
    }

    migrated_sources: list[dict[str, Any]] = []
    source_artifacts: dict[str, str | None] = {}
    for source in sources:
        legacy_id = str(source["source_id"])
        source_artifacts[legacy_id] = source.get("artifact_path")
        migrated = dict(source)
        migrated["source_id"] = source_ids[legacy_id]
        migrated["independence_group"] = identifiers.get(
            str(source["independence_group"]),
            domain="independence_group",
            prefix="group",
        )
        migrated["verification_state"] = _mapped_state(
            str(source["verification_state"]),
            _VERIFICATION_STATES,
            label="verification",
        )
        date_label = migrated.get("date_label")
        if isinstance(date_label, str) and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T.*",
            date_label,
        ):
            migrated["date_label"] = date_label[:10]
        migrated_sources.append(migrated)

    exact_binding_count = 0
    omitted_binding_count = 0
    migrated_evidence: list[dict[str, Any]] = []
    for item in evidence:
        legacy_id = str(item["evidence_id"])
        legacy_source_id = str(item["source_id"])
        migrated = dict(item)
        migrated["evidence_id"] = evidence_ids[legacy_id]
        migrated["source_id"] = source_ids[legacy_source_id]
        fragment_id = item.get("source_fragment_id")
        if fragment_id is not None:
            migrated["source_fragment_id"] = identifiers.get(
                str(fragment_id),
                domain="fragment",
                prefix="fragment",
            )
        binding = _migrate_source_binding(
            item,
            artifact_root=artifact_root,
            artifact_path=source_artifacts[legacy_source_id],
        )
        if binding is None:
            migrated.pop("source_fragment_id", None)
            migrated.pop("fragment_text_sha256", None)
            migrated.pop("source_address", None)
            omitted_binding_count += 1
        else:
            migrated.update(binding)
            exact_binding_count += 1
        migrated_evidence.append(migrated)

    migrated_questions: list[dict[str, Any]] = []
    for item in questions:
        legacy_id = str(item["question_id"])
        migrated = dict(item)
        migrated["question_id"] = question_ids[legacy_id]
        migrated["state"] = _mapped_state(
            str(item["state"]),
            _FINDING_STATES,
            label="question",
        )
        migrated["evidence_ids"] = _map_references(item.get("evidence_ids", []), evidence_ids)
        migrated["trace_spans"] = _build_trace_spans(
            text=str(item["short_answer"]),
            object_id=legacy_id,
            normalized_object_id=question_ids[legacy_id],
            overrides=overrides,
            evidence_ids=evidence_ids,
        )
        migrated_questions.append(migrated)

    migrated_hypotheses: list[dict[str, Any]] = []
    for item in hypotheses:
        legacy_id = str(item["hypothesis_id"])
        migrated = dict(item)
        migrated["hypothesis_id"] = hypothesis_ids[legacy_id]
        migrated["state"] = _mapped_state(
            str(item["state"]),
            _HYPOTHESIS_STATES,
            label="hypothesis",
        )
        migrated["evidence_ids"] = _map_references(item.get("evidence_ids", []), evidence_ids)
        migrated["counterevidence_ids"] = _map_references(
            item.get("counterevidence_ids", []),
            evidence_ids,
        )
        migrated["question_ids"] = _map_references(item.get("question_ids", []), question_ids)
        migrated["trace_spans"] = _build_trace_spans(
            text=str(item["synthesis"]),
            object_id=legacy_id,
            normalized_object_id=hypothesis_ids[legacy_id],
            overrides=overrides,
            evidence_ids=evidence_ids,
        )
        migrated_hypotheses.append(migrated)

    migrated_relations: list[dict[str, Any]] = []
    for item in relations:
        migrated = dict(item)
        migrated["relation_id"] = identifiers.get(
            str(item["relation_id"]),
            domain="relation",
            prefix="relation",
        )
        migrated["source_hypothesis_id"] = hypothesis_ids[str(item["source_hypothesis_id"])]
        migrated["target_hypothesis_id"] = hypothesis_ids[str(item["target_hypothesis_id"])]
        migrated["evidence_ids"] = _map_references(item.get("evidence_ids", []), evidence_ids)
        migrated_relations.append(migrated)

    migrated_timeline: list[dict[str, Any]] = []
    for item in timeline:
        migrated = dict(item)
        migrated["event_id"] = identifiers.get(
            str(item["event_id"]),
            domain="event",
            prefix="event",
        )
        migrated["state"] = _mapped_state(
            str(item["state"]),
            _FINDING_STATES,
            label="timeline",
        )
        migrated["evidence_ids"] = _map_references(item.get("evidence_ids", []), evidence_ids)
        migrated["question_ids"] = _map_references(item.get("question_ids", []), question_ids)
        migrated["hypothesis_ids"] = _map_references(
            item.get("hypothesis_ids", []),
            hypothesis_ids,
        )
        migrated_timeline.append(migrated)

    migrated_gaps: list[dict[str, Any]] = []
    for item in gaps:
        migrated = dict(item)
        migrated["gap_id"] = identifiers.get(
            str(item["gap_id"]),
            domain="gap",
            prefix="gap",
        )
        migrated["related_question_ids"] = _map_references(
            item.get("related_question_ids", []),
            question_ids,
        )
        migrated["related_hypothesis_ids"] = _map_references(
            item.get("related_hypothesis_ids", []),
            hypothesis_ids,
        )
        migrated_gaps.append(migrated)

    corpus = dict(raw["corpus"])
    corpus["source_count"] = len(migrated_sources)
    corpus["source_family_count"] = len({item["independence_group"] for item in migrated_sources})
    migrated_payload = {
        **raw,
        "atlas_id": identifiers.get(
            str(raw["atlas_id"]),
            domain="atlas",
            prefix="atlas",
        ),
        "case_id": identifiers.get(
            str(raw["case_id"]),
            domain="case",
            prefix="case",
        ),
        "corpus": corpus,
        "sources": migrated_sources,
        "evidence": migrated_evidence,
        "questions": migrated_questions,
        "hypotheses": migrated_hypotheses,
        "relations": migrated_relations,
        "timeline": migrated_timeline,
        "gaps": migrated_gaps,
    }
    manifest = ResearchAtlasManifest.model_validate(migrated_payload)
    input_sha256 = sha256_hex(canonical_json_bytes(raw))
    output_sha256 = sha256_hex(
        canonical_json_bytes(manifest.model_dump(mode="json", exclude_none=True))
    )
    trace_span_count = sum(len(item.trace_spans) for item in manifest.questions) + sum(
        len(item.trace_spans) for item in manifest.hypotheses
    )
    return MigratedResearchAtlas(
        manifest=manifest,
        receipt=AtlasMigrationReceipt(
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            source_count=len(manifest.sources),
            evidence_count=len(manifest.evidence),
            trace_span_count=trace_span_count,
            exact_binding_count=exact_binding_count,
            omitted_binding_count=omitted_binding_count,
            normalized_identifier_count=identifiers.normalized_count,
        ),
    )


def _safe_identifier(raw_value: str, *, prefix: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "_", raw_value.casefold()).strip("_-")
    if len(value) < 2 or not value[:1].isalpha():
        value = f"{prefix}_{value or 'item'}"
    if len(value) > 96:
        digest = sha256_hex(raw_value.encode("utf-8"))[:12]
        value = f"{value[:83].rstrip('_-')}_{digest}"
    return value


def _mapped_state(value: str, mapping: Mapping[str, str], *, label: str) -> str:
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unsupported legacy {label} state: {value}") from exc


def _map_references(values: Sequence[Any], mapping: Mapping[str, str]) -> list[str]:
    return [mapping[str(value)] for value in values]


def _migrate_source_binding(
    evidence: Mapping[str, Any],
    *,
    artifact_root: Path | None,
    artifact_path: str | None,
) -> dict[str, Any] | None:
    address = evidence.get("source_address")
    if (
        artifact_root is None
        or artifact_path is None
        or not isinstance(address, Mapping)
        or "char_start" not in address
        or "char_end" not in address
    ):
        return None
    artifact = (artifact_root / artifact_path).resolve(strict=False)
    root = artifact_root.resolve(strict=False)
    if not artifact.is_relative_to(root) or not artifact.is_file():
        return None
    text = artifact.read_text(encoding="utf-8")
    char_start = int(address["char_start"])
    char_end = int(address["char_end"])
    if char_start < 0 or char_end <= char_start or char_end > len(text):
        return None
    line_start = text.count("\n", 0, char_start) + 1
    line_end = text.count("\n", 0, char_end - 1) + 1
    lines = text.splitlines()
    selected_text = "\n".join(lines[line_start - 1 : line_end])
    return {
        "fragment_text_sha256": sha256_hex(selected_text.encode("utf-8")),
        "source_address": {
            "kind": str(address.get("kind", "migrated_exact_fragment")),
            "heading_path": tuple(address.get("heading_path", ())),
            "line_start": line_start,
            "line_end": line_end,
            "char_start": char_start,
            "char_end": char_end,
        },
    }


def _build_trace_spans(
    *,
    text: str,
    object_id: str,
    normalized_object_id: str,
    overrides: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence_ids: Mapping[str, str],
) -> list[dict[str, Any]]:
    specifications = overrides.get(object_id, overrides.get(normalized_object_id, ()))
    trace_object_id = normalized_object_id.replace("-", "_")
    spans: list[dict[str, Any]] = []
    for index, specification in enumerate(specifications, start=1):
        exact_text = str(specification["text"])
        start = text.find(exact_text)
        if start < 0 or text.find(exact_text, start + 1) >= 0:
            raise ValueError(
                f"trace override text must occur exactly once in {object_id}: {exact_text!r}"
            )
        spans.append(
            {
                "span_id": f"trace_{trace_object_id}_{index}",
                "start": start,
                "end": start + len(exact_text),
                "text": exact_text,
                "kind": str(specification["kind"]),
                "evidence_ids": _map_references(
                    specification["evidence_ids"],
                    evidence_ids,
                ),
            }
        )
    return sorted(spans, key=lambda item: (int(item["start"]), int(item["end"])))
