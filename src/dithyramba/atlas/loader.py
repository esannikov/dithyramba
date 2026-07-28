"""Load one immutable Research Atlas manifest with a reproducible hash."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dithyramba.contracts import canonical_json_bytes, sha256_hex

from .models import ResearchAtlasManifest


@dataclass(frozen=True, slots=True)
class LoadedResearchAtlas:
    manifest: ResearchAtlasManifest
    manifest_path: Path
    manifest_hash: str


def load_research_atlas(path: str | Path) -> LoadedResearchAtlas:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError("Research Atlas manifest must be a regular file")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ResearchAtlasManifest.model_validate(payload)
    canonical = canonical_json_bytes(manifest.model_dump(mode="json"))
    return LoadedResearchAtlas(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_hash=sha256_hex(canonical),
    )
