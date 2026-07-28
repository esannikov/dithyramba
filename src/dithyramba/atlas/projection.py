"""Strict companion projection for discoverable, non-promoted research material."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dithyramba.contracts import canonical_json_bytes, sha256_hex

from .loader import LoadedResearchAtlas
from .models import AtlasSourceAddress

_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_DATE_PATTERN = r"^[0-9?]{4}(?:-[0-9?]{2})?(?:-[0-9?]{2})?$"


class _ProjectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ProjectionMaterialState(StrEnum):
    CANDIDATE = "candidate"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    ROUTE_ONLY = "route_only"
    METADATA_ONLY = "metadata_only"


class ProjectionMaterialKind(StrEnum):
    EVENT = "event"
    STATEMENT = "statement"
    SOURCE = "source"
    ARTWORK = "artwork"
    EXHIBITION = "exhibition"
    RECORD = "record"


class ProjectionCollection(_ProjectionModel):
    collection_id: str = Field(pattern=_ID_PATTERN)
    label: str = Field(min_length=1, max_length=300)
    kind: ProjectionMaterialKind
    record_count: int = Field(ge=0)
    description: str = Field(min_length=1, max_length=2_000)
    status_note: str = Field(min_length=1, max_length=1_000)


class ProjectionPeriod(_ProjectionModel):
    period_id: str = Field(pattern=_ID_PATTERN)
    label: str = Field(min_length=1, max_length=300)
    date_start: str = Field(pattern=_DATE_PATTERN)
    date_end: str = Field(pattern=_DATE_PATTERN)
    summary: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def ordered_dates(self) -> Self:
        if self.date_end < self.date_start:
            raise ValueError("projection period date range is reversed")
        return self


class ProjectionTheme(_ProjectionModel):
    theme_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=4_000)
    what_it_changes: str = Field(min_length=1, max_length=2_000)
    hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=64)
    question_ids: tuple[str, ...] = Field(default=(), max_length=128)
    gap_ids: tuple[str, ...] = Field(default=(), max_length=64)


class ProjectionMaterial(_ProjectionModel):
    material_id: str = Field(pattern=_ID_PATTERN)
    kind: ProjectionMaterialKind
    state: ProjectionMaterialState
    title: str = Field(min_length=1, max_length=4_000)
    date_start: str | None = Field(default=None, pattern=_DATE_PATTERN)
    date_label: str | None = Field(default=None, max_length=160)
    period_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    theme_ids: tuple[str, ...] = Field(default=(), max_length=32)
    source_id: str = Field(pattern=_ID_PATTERN)
    fragment_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_address: AtlasSourceAddress
    locator: str = Field(min_length=1, max_length=500)
    asserting_voice: str = Field(min_length=1, max_length=300)
    limitation: str = Field(min_length=1, max_length=2_000)
    review_priority: bool = False


class ResearchProjectionManifest(_ProjectionModel):
    SCHEMA: ClassVar[str] = "dithyramba.research_projection/1.0"

    schema_id: str
    projection_id: str = Field(pattern=_ID_PATTERN)
    case_id: str = Field(pattern=_ID_PATTERN)
    base_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=4_000)
    generated_at: datetime
    collections: tuple[ProjectionCollection, ...] = Field(default=(), max_length=128)
    periods: tuple[ProjectionPeriod, ...] = Field(default=(), max_length=128)
    themes: tuple[ProjectionTheme, ...] = Field(default=(), max_length=128)
    materials: tuple[ProjectionMaterial, ...] = Field(default=(), max_length=50_000)

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")

        _unique_ids(item.collection_id for item in self.collections)
        period_ids = _unique_ids(item.period_id for item in self.periods)
        theme_ids = _unique_ids(item.theme_id for item in self.themes)
        _unique_ids(item.material_id for item in self.materials)
        for material in self.materials:
            if material.period_id is not None and material.period_id not in period_ids:
                raise ValueError(
                    f"projection material references missing period: {material.period_id}"
                )
            missing_themes = sorted(set(material.theme_ids).difference(theme_ids))
            if missing_themes:
                raise ValueError(
                    "projection material references missing themes: " + ", ".join(missing_themes)
                )
        return self


@dataclass(frozen=True, slots=True)
class LoadedResearchProjection:
    manifest: ResearchProjectionManifest
    manifest_path: Path
    manifest_hash: str


def load_research_projection(
    path: str | Path,
    *,
    atlas: LoadedResearchAtlas,
) -> LoadedResearchProjection:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError("Research projection manifest must be a regular file")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ResearchProjectionManifest.model_validate(payload)
    if manifest.case_id != atlas.manifest.case_id:
        raise ValueError("research projection case_id does not match the Atlas")
    if manifest.base_manifest_hash != atlas.manifest_hash:
        raise ValueError("research projection is bound to a different Atlas hash")

    source_ids = {item.source_id for item in atlas.manifest.sources}
    question_ids = {item.question_id for item in atlas.manifest.questions}
    hypothesis_ids = {item.hypothesis_id for item in atlas.manifest.hypotheses}
    gap_ids = {item.gap_id for item in atlas.manifest.gaps}
    for material in manifest.materials:
        if material.source_id not in source_ids:
            raise ValueError(
                f"projection material references missing Atlas source: {material.source_id}"
            )
    for theme in manifest.themes:
        _require_subset(theme.question_ids, question_ids, "projection theme question")
        _require_subset(
            theme.hypothesis_ids,
            hypothesis_ids,
            "projection theme hypothesis",
        )
        _require_subset(theme.gap_ids, gap_ids, "projection theme gap")

    canonical = canonical_json_bytes(manifest.model_dump(mode="json"))
    return LoadedResearchProjection(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_hash=sha256_hex(canonical),
    )


def _unique_ids(values: Iterable[str]) -> frozenset[str]:
    identifiers = tuple(values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("projection identifiers must be unique within their object type")
    return frozenset(identifiers)


def _require_subset(
    values: tuple[str, ...],
    valid: set[str],
    label: str,
) -> None:
    missing = sorted(set(values).difference(valid))
    if missing:
        raise ValueError(f"{label} references missing IDs: {', '.join(missing)}")
