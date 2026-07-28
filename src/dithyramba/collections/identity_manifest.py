"""Strict, hash-pinned administrative source-identity manifests."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from dithyramba.contracts import canonical_sha256_hex, sha256_hex

from .errors import SourceIdentityManifestError
from .models import CollectionRoot

SOURCE_IDENTITY_MANIFEST_SCHEMA = "dithyramba.source_identity_manifest/1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LABEL = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class SourceIdentityDeclaration:
    """One administrator-declared logical source representation part."""

    logical_source_uri: str
    connector: str
    connector_revision: str
    representation: str
    part_number: int
    part_metadata_hash: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class SourceIdentityManifest:
    """Validated, complete root-local map from paths to declarations."""

    path: Path
    sha256: str
    declarations_by_path: dict[str, SourceIdentityDeclaration]

    def declaration_for(self, relative_path: str) -> SourceIdentityDeclaration:
        try:
            return self.declarations_by_path[relative_path]
        except KeyError as exc:
            raise SourceIdentityManifestError(
                f"identity manifest has no declaration for source path: {relative_path}"
            ) from exc


def load_source_identity_manifest(root: CollectionRoot) -> SourceIdentityManifest | None:
    """Load a pinned manifest and prove it is a complete safe root map.

    A missing pin means legacy ingest.  A pin is deliberately administrative:
    the path must be an absolute regular non-symlink file and its raw SHA-256
    must equal the value persisted with the Collection root.
    """

    if root.identity_manifest_path is None:
        if root.identity_manifest_sha256 is not None:
            raise SourceIdentityManifestError("identity manifest hash has no manifest path")
        return None
    if root.identity_manifest_sha256 is None:
        raise SourceIdentityManifestError("identity manifest path has no pinned SHA-256")
    path = root.identity_manifest_path
    try:
        status = os.lstat(path)
        if os.path.islink(path) or not os.path.isfile(path):
            raise SourceIdentityManifestError(
                "identity manifest must be a regular non-symlink file"
            )
        data = path.read_bytes()
    except OSError as exc:
        raise SourceIdentityManifestError(f"identity manifest cannot be read: {path}") from exc
    del status
    actual = sha256_hex(data)
    if actual != root.identity_manifest_sha256:
        raise SourceIdentityManifestError("identity manifest SHA-256 does not match its pin")
    try:
        decoded = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIdentityManifestError("identity manifest is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict) or set(decoded) != {"schema", "identities"}:
        raise SourceIdentityManifestError(
            "identity manifest must contain exactly schema and identities"
        )
    if decoded["schema"] != SOURCE_IDENTITY_MANIFEST_SCHEMA:
        raise SourceIdentityManifestError("identity manifest schema is unsupported")
    identities = decoded["identities"]
    if not isinstance(identities, list) or not identities:
        raise SourceIdentityManifestError("identity manifest identities must be a non-empty list")

    declarations: dict[str, SourceIdentityDeclaration] = {}
    for raw_identity in identities:
        _load_identity(raw_identity, actual, declarations)
    _validate_complete_root_mapping(root, declarations)
    return SourceIdentityManifest(path=path, sha256=actual, declarations_by_path=declarations)


def _load_identity(
    raw: object,
    manifest_sha256: str,
    declarations: dict[str, SourceIdentityDeclaration],
) -> None:
    if not isinstance(raw, dict) or set(raw) != {
        "logical_source_uri",
        "connector",
        "connector_revision",
        "representation",
        "parts",
    }:
        raise SourceIdentityManifestError("each identity must contain exactly its declared fields")
    logical_source_uri = _logical_uri(raw["logical_source_uri"])
    connector = _label(raw["connector"], "connector")
    connector_revision = _text(raw["connector_revision"], "connector_revision")
    representation = _text(raw["representation"], "representation")
    parts = raw["parts"]
    if not isinstance(parts, list) or not parts:
        raise SourceIdentityManifestError("identity parts must be a non-empty list")
    part_numbers: list[int] = []
    for raw_part in parts:
        if not isinstance(raw_part, dict) or set(raw_part) != {"path", "part"}:
            raise SourceIdentityManifestError("identity parts must contain exactly path and part")
        path = _relative_path(raw_part["path"])
        part = raw_part["part"]
        if type(part) is not int or part < 0:
            raise SourceIdentityManifestError("identity part must be a non-negative integer")
        part_numbers.append(part)
        metadata = {
            "logical_source_uri": logical_source_uri,
            "connector": connector,
            "connector_revision": connector_revision,
            "representation": representation,
            "path": path,
            "part": part,
        }
        if path in declarations:
            raise SourceIdentityManifestError(
                f"identity manifest maps a path more than once: {path}"
            )
        declarations[path] = SourceIdentityDeclaration(
            logical_source_uri=logical_source_uri,
            connector=connector,
            connector_revision=connector_revision,
            representation=representation,
            part_number=part,
            part_metadata_hash=canonical_sha256_hex(metadata),
            manifest_sha256=manifest_sha256,
        )
    if sorted(part_numbers) != list(range(len(part_numbers))):
        raise SourceIdentityManifestError("identity parts must be contiguous from zero")


def _validate_complete_root_mapping(
    root: CollectionRoot,
    declarations: dict[str, SourceIdentityDeclaration],
) -> None:
    from dithyramba.ingest.reader import discover_source_paths

    discovered = set(discover_source_paths(root))
    if set(declarations) != discovered:
        missing = sorted(discovered - set(declarations))
        extra = sorted(set(declarations) - discovered)
        detail = "missing=" + ",".join(missing) + " extra=" + ",".join(extra)
        raise SourceIdentityManifestError(
            "identity manifest must map every eligible root path: " + detail
        )
    for relative_path in declarations:
        candidate = root.path / Path(*PurePosixPath(relative_path).parts)
        try:
            if os.path.islink(candidate):
                raise SourceIdentityManifestError(
                    f"identity manifest path must not be a symlink: {relative_path}"
                )
        except OSError as exc:
            raise SourceIdentityManifestError(
                f"identity manifest path cannot be checked: {relative_path}"
            ) from exc


def _relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise SourceIdentityManifestError("identity path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise SourceIdentityManifestError("identity path must stay below its Collection root")
    return path.as_posix()


def _logical_uri(value: object) -> str:
    if type(value) is not str or value != value.strip() or not value or "\x00" in value:
        raise SourceIdentityManifestError("logical_source_uri must be non-empty unpadded text")
    parsed = urlsplit(value)
    if not parsed.scheme or parsed.scheme == "file" or parsed.fragment:
        raise SourceIdentityManifestError(
            "logical_source_uri must be a non-file absolute URI without fragment"
        )
    return value


def _label(value: object, label: str) -> str:
    if type(value) is not str or _LABEL.fullmatch(value) is None:
        raise SourceIdentityManifestError(f"{label} must use stable lowercase label grammar")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise SourceIdentityManifestError(f"{label} must be non-empty unpadded text")
    return value
