"""Safe application-data placement and Library layout creation."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TypeAlias

from platformdirs import user_data_path

from .errors import (
    ApplicationDataPathError,
    LibraryAlreadyExistsError,
    LibraryLayoutError,
    PathOverlapError,
)
from .models import LibraryConfig, validate_library_id

PathInput: TypeAlias = str | PathLike[str]

_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600


@dataclass(frozen=True, slots=True)
class LibraryPaths:
    """Canonical locations belonging to one Library installation."""

    application_data_root: Path
    libraries_root: Path
    root: Path
    database: Path
    events: Path
    blobs: Path
    indexes: Path
    cache: Path
    backups: Path

    @property
    def directories(self) -> tuple[Path, ...]:
        """Return the directories created inside the Library root."""

        return (self.blobs, self.indexes, self.cache, self.backups)

    @property
    def files(self) -> tuple[Path, ...]:
        """Return the required file surfaces inside the Library root."""

        return (self.database, self.events)


def application_data_root(override: PathInput | None = None) -> Path:
    """Return the canonical app-data root without creating it.

    ``override`` is the test/embedding seam. Relative paths and paths containing
    ``..`` are rejected before canonicalization so a caller cannot accidentally
    direct live state somewhere dependent on the process working directory.
    """

    raw = (
        Path(override).expanduser()
        if override is not None
        else user_data_path("Dithyramba", appauthor=False, ensure_exists=False)
    )
    _reject_ambiguous_path(raw, label="application-data root")
    return raw.resolve(strict=False)


def library_paths(library_id: str, *, data_root: PathInput | None = None) -> LibraryPaths:
    """Build, but do not create, the canonical layout for ``library_id``."""

    valid_id = validate_library_id(library_id)
    app_root = application_data_root(data_root)
    libraries_root = app_root / "libraries"
    root = libraries_root / valid_id
    return LibraryPaths(
        application_data_root=app_root,
        libraries_root=libraries_root,
        root=root,
        database=root / "memory.sqlite3",
        events=root / "events.jsonl",
        blobs=root / "blobs",
        indexes=root / "indexes",
        cache=root / "cache",
        backups=root / "backups",
    )


def create_library_layout(
    config: LibraryConfig,
    *,
    data_root: PathInput | None = None,
    declared_source_roots: tuple[PathInput, ...] = (),
    declared_sync_roots: tuple[PathInput, ...] = (),
) -> LibraryPaths:
    """Create one private Library layout after validating physical boundaries.

    Source and sync roots are resolved before any write. Any containment in
    either direction is rejected: a live DB may not sit below such a root, and
    a source root may not sit below application data.
    """

    paths = library_paths(config.library_id, data_root=data_root)
    protected_roots = tuple(declared_source_roots) + tuple(declared_sync_roots)
    for root in protected_roots:
        resolved = _canonical_declared_root(root)
        ensure_paths_disjoint(
            paths.application_data_root,
            resolved,
            left_label="application-data root",
            right_label="declared source/sync root",
        )

    if paths.root.exists() or paths.root.is_symlink():
        raise LibraryAlreadyExistsError(f"Library already exists: {paths.root}")

    created_root = False
    try:
        _create_private_directory(paths.application_data_root)
        _create_private_directory(paths.libraries_root)
        paths.root.mkdir(mode=_DIRECTORY_MODE, exist_ok=False)
        created_root = True
        _set_private_permissions(paths.root, _DIRECTORY_MODE)
        for directory in paths.directories:
            directory.mkdir(mode=_DIRECTORY_MODE)
            _set_private_permissions(directory, _DIRECTORY_MODE)
        for file_path in paths.files:
            _create_private_file(file_path)
        _verify_layout_containment(paths)
    except LibraryAlreadyExistsError:
        raise
    except OSError as error:
        if created_root:
            shutil.rmtree(paths.root, ignore_errors=True)
        raise LibraryLayoutError(f"could not create Library layout: {paths.root}") from error
    except Exception:
        if created_root:
            shutil.rmtree(paths.root, ignore_errors=True)
        raise
    return paths


def validate_library_layout(paths: LibraryPaths) -> None:
    """Fail closed unless every required layout surface exists in-bounds."""

    if not paths.root.is_dir():
        raise LibraryLayoutError(f"Library root is not a directory: {paths.root}")
    for directory in paths.directories:
        if not directory.is_dir():
            raise LibraryLayoutError(f"required Library directory is missing: {directory}")
    for file_path in paths.files:
        if not file_path.is_file():
            raise LibraryLayoutError(f"required Library file is missing: {file_path}")
    _verify_layout_containment(paths)


def ensure_paths_disjoint(
    left: PathInput,
    right: PathInput,
    *,
    left_label: str = "first path",
    right_label: str = "second path",
) -> None:
    """Reject equality or canonical containment in either direction."""

    left_path = Path(left).resolve(strict=False)
    right_path = Path(right).resolve(strict=False)
    if _contains(left_path, right_path) or _contains(right_path, left_path):
        raise PathOverlapError(f"{left_label} ({left_path}) overlaps {right_label} ({right_path})")


def is_path_within(path: PathInput, boundary: PathInput) -> bool:
    """Return whether canonical ``path`` is equal to or below ``boundary``."""

    return _contains(Path(boundary).resolve(strict=False), Path(path).resolve(strict=False))


def _canonical_declared_root(value: PathInput) -> Path:
    path = Path(value).expanduser()
    _reject_ambiguous_path(path, label="declared source/sync root")
    return path.resolve(strict=False)


def _reject_ambiguous_path(path: Path, *, label: str) -> None:
    if ".." in path.parts:
        raise ApplicationDataPathError(f"{label} must not contain '..': {path}")
    if not path.is_absolute():
        raise ApplicationDataPathError(f"{label} must be absolute: {path}")


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _create_private_directory(path: Path) -> None:
    path.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise LibraryLayoutError(f"application-data component is not a real directory: {path}")
    _set_private_permissions(path, _DIRECTORY_MODE)


def _create_private_file(path: Path) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _FILE_MODE)
    os.close(descriptor)
    _set_private_permissions(path, _FILE_MODE)


def _set_private_permissions(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except NotImplementedError:
        if os.name == "posix":
            raise
    except OSError:
        if os.name == "posix":
            raise


def _verify_layout_containment(paths: LibraryPaths) -> None:
    _require_real_directory(paths.application_data_root, label="application-data root")
    _require_real_directory(paths.libraries_root, label="libraries root")
    _require_real_directory(paths.root, label="Library root")
    canonical_root = paths.root.resolve(strict=False)
    if canonical_root.parent != paths.libraries_root.resolve(strict=False):
        raise LibraryLayoutError("Library root escaped its canonical libraries directory")
    for member in paths.directories:
        if not is_path_within(member, canonical_root):
            raise LibraryLayoutError(f"Library layout member escaped its root: {member}")
        _require_real_directory(member, label="required Library directory")
    for member in paths.files:
        if not is_path_within(member, canonical_root):
            raise LibraryLayoutError(f"Library layout member escaped its root: {member}")
        _require_real_file(member, label="required Library file")
    for directory in (
        paths.application_data_root,
        paths.libraries_root,
        paths.root,
        *paths.directories,
    ):
        _require_private_posix_permissions(directory, expected_mode=_DIRECTORY_MODE)
    for file_path in paths.files:
        _require_private_posix_permissions(file_path, expected_mode=_FILE_MODE)


def _require_real_directory(path: Path, *, label: str) -> None:
    """Reject missing, non-directory, and symlink-substituted layout components."""

    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise LibraryLayoutError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise LibraryLayoutError(f"{label} is not a real directory: {path}")


def _require_real_file(path: Path, *, label: str) -> None:
    """Reject symlinks, non-regular files, and hardlinked fixed files."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise LibraryLayoutError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise LibraryLayoutError(f"{label} is not a real file: {path}")
    if metadata.st_nlink != 1:
        raise LibraryLayoutError(f"{label} must not be hardlinked: {path}")


def _require_private_posix_permissions(path: Path, *, expected_mode: int) -> None:
    """Require user ownership and no group/other access on POSIX runtimes."""

    if os.name != "posix":
        return
    metadata = path.lstat()
    if metadata.st_uid != os.geteuid():
        raise LibraryLayoutError(f"Library layout component has a different owner: {path}")
    actual_mode = stat.S_IMODE(metadata.st_mode)
    if actual_mode != expected_mode:
        raise LibraryLayoutError(
            f"Library layout component has unsafe mode {actual_mode:#o}; "
            f"expected {expected_mode:#o}: {path}"
        )
