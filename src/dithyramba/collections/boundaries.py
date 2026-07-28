"""Canonical and symlink-aware Collection root validation."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TypeAlias

from dithyramba.library import application_data_root, is_path_within

from .errors import (
    CollectionRootNotDirectoryError,
    CollectionRootOverlapError,
    CollectionRootPathError,
    SourcePathEscapeError,
)
from .models import CollectionRoot

PathInput: TypeAlias = str | PathLike[str]


def build_collection_root(
    path: PathInput,
    *,
    data_root: PathInput | None = None,
    include_globs: tuple[str, ...] = (
        "**/*.md",
        "**/*.markdown",
        "**/*.txt",
        "**/*.pdf",
    ),
    exclude_globs: tuple[str, ...] = (),
    identity_manifest_path: PathInput | None = None,
    identity_manifest_sha256: str | None = None,
) -> CollectionRoot:
    """Validate an existing source directory and return its frozen config."""

    candidate = _unambiguous_absolute(path, label="Collection root")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise CollectionRootNotDirectoryError(
            f"Collection root is not an existing directory: {candidate}"
        ) from error
    if not resolved.is_dir():
        raise CollectionRootNotDirectoryError(f"Collection root is not a directory: {resolved}")

    app_root = application_data_root(data_root)
    if is_path_within(resolved, app_root) or is_path_within(app_root, resolved):
        raise CollectionRootOverlapError(
            f"Collection root ({resolved}) overlaps application data ({app_root})"
        )
    return CollectionRoot(
        path=resolved,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        identity_manifest_path=(
            _unambiguous_absolute(identity_manifest_path, label="Identity manifest")
            if identity_manifest_path is not None
            else None
        ),
        identity_manifest_sha256=identity_manifest_sha256,
    )


def resolve_source_path(
    root: CollectionRoot,
    relative_path: PathInput,
    *,
    require_file: bool = True,
) -> Path:
    """Resolve one source path and reject traversal or symlink escape."""

    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SourcePathEscapeError("source path must be relative and contain no '..'")
    try:
        resolved = (root.path / candidate).resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise SourcePathEscapeError(f"source path does not exist: {candidate}") from error
    if not is_path_within(resolved, root.path):
        raise SourcePathEscapeError(f"source path escaped Collection root: {candidate}")
    if require_file and not resolved.is_file():
        raise SourcePathEscapeError(f"source path is not a file: {candidate}")
    return resolved


def _unambiguous_absolute(value: PathInput, *, label: str) -> Path:
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise CollectionRootPathError(f"{label} must not contain '..': {path}")
    if not path.is_absolute():
        raise CollectionRootPathError(f"{label} must be absolute: {path}")
    return path
