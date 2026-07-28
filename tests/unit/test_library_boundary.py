from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from dithyramba.collections import (
    CollectionConfig,
    CollectionKind,
    CollectionRootNotDirectoryError,
    CollectionRootOverlapError,
    CollectionRootPathError,
    InvalidCollectionGlobError,
    InvalidCollectionIdError,
    InvalidCollectionNameError,
    SourcePathEscapeError,
    build_collection_root,
    resolve_source_path,
    validate_collection_id,
    validate_collection_name,
    validate_globs,
)
from dithyramba.library import (
    ApplicationDataPathError,
    InvalidLibraryIdError,
    InvalidLibraryNameError,
    LibraryAlreadyExistsError,
    LibraryConfig,
    LibraryLayoutError,
    PathOverlapError,
    application_data_root,
    create_library_layout,
    ensure_paths_disjoint,
    is_path_within,
    library_paths,
    validate_library_id,
    validate_library_layout,
    validate_library_name,
)


def test_application_data_override_is_canonical_and_does_not_create(tmp_path: Path) -> None:
    target = tmp_path / "not-created"

    resolved = application_data_root(target)

    assert resolved == target
    assert not target.exists()


def test_application_data_default_uses_platformdirs_without_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_default = tmp_path / "platform-default"
    monkeypatch.setattr(
        "dithyramba.library.paths.user_data_path",
        lambda *_args, **_kwargs: fake_default,
    )

    assert application_data_root() == fake_default
    assert not fake_default.exists()


@pytest.mark.parametrize("path", ["relative/data", "/tmp/base/../data"])
def test_application_data_root_rejects_ambiguous_paths(path: str) -> None:
    with pytest.raises(ApplicationDataPathError):
        application_data_root(path)


def test_library_and_collection_models_are_frozen_and_use_uuid4_ids() -> None:
    library = LibraryConfig(name="Research")
    collection = CollectionConfig(
        library_id=library.library_id,
        name="PhD",
        kind=CollectionKind.CORPUS,
    )

    assert re.fullmatch(r"library_[0-9a-f]{32}", library.id)
    assert re.fullmatch(r"collection_[0-9a-f]{32}", collection.id)
    assert validate_library_id(library.id) == library.id
    assert validate_collection_id(collection.id) == collection.id
    with pytest.raises(ValidationError, match="frozen"):
        library.name = "Changed"
    with pytest.raises(ValidationError, match="frozen"):
        collection.name = "Changed"


@pytest.mark.parametrize("value", ["library_bad", "collection_" + "0" * 32, 7])
def test_typed_identifier_validators_reject_invalid_values(value: object) -> None:
    with pytest.raises(InvalidLibraryIdError):
        validate_library_id(value)  # type: ignore[arg-type]
    with pytest.raises(InvalidCollectionIdError):
        validate_collection_id(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", " padded", "x" * 201, 7])
def test_typed_name_validators_reject_invalid_values(value: object) -> None:
    with pytest.raises(InvalidLibraryNameError):
        validate_library_name(value)  # type: ignore[arg-type]
    with pytest.raises(InvalidCollectionNameError):
        validate_collection_name(value)  # type: ignore[arg-type]


def test_library_layout_is_complete_private_and_inside_override(tmp_path: Path) -> None:
    injected_root = tmp_path / "private-app-data"
    paths = create_library_layout(LibraryConfig(name="Research"), data_root=injected_root)

    assert paths.application_data_root == injected_root
    assert paths.database.read_bytes() == b""
    assert paths.events.read_bytes() == b""
    assert all(path.is_dir() for path in paths.directories)
    assert all(is_path_within(path, injected_root) for path in paths.directories + paths.files)
    validate_library_layout(paths)
    if os.name == "posix":
        assert paths.root.stat().st_mode & 0o777 == 0o700
        assert paths.events.stat().st_mode & 0o777 == 0o600


def test_library_paths_is_pure_and_duplicate_creation_fails(tmp_path: Path) -> None:
    config = LibraryConfig(name="Research")
    planned = library_paths(config.id, data_root=tmp_path / "data")
    assert not planned.root.exists()

    create_library_layout(config, data_root=tmp_path / "data")
    with pytest.raises(LibraryAlreadyExistsError):
        create_library_layout(config, data_root=tmp_path / "data")


def test_creation_rejects_data_below_declared_source_before_any_write(tmp_path: Path) -> None:
    source_root = tmp_path / "synced-source"
    source_root.mkdir()
    data_root = source_root / "runtime"

    with pytest.raises(PathOverlapError):
        create_library_layout(
            LibraryConfig(name="Unsafe"),
            data_root=data_root,
            declared_source_roots=(source_root,),
        )

    assert not data_root.exists()


def test_creation_rejects_declared_root_below_live_data_before_writing(tmp_path: Path) -> None:
    data_root = tmp_path / "runtime"
    source_root = data_root / "sources"

    with pytest.raises(PathOverlapError):
        create_library_layout(
            LibraryConfig(name="Unsafe"),
            data_root=data_root,
            declared_source_roots=(source_root,),
        )

    assert not data_root.exists()


def test_creation_resolves_sync_root_symlinks_before_writing(tmp_path: Path) -> None:
    real_sync_root = tmp_path / "real-sync"
    real_sync_root.mkdir()
    sync_link = tmp_path / "sync-link"
    sync_link.symlink_to(real_sync_root, target_is_directory=True)

    with pytest.raises(PathOverlapError):
        create_library_layout(
            LibraryConfig(name="Unsafe"),
            data_root=sync_link / "runtime",
            declared_sync_roots=(real_sync_root,),
        )

    assert not (real_sync_root / "runtime").exists()


def test_declared_root_must_be_absolute_and_unambiguous(tmp_path: Path) -> None:
    with pytest.raises(ApplicationDataPathError):
        create_library_layout(
            LibraryConfig(name="Unsafe"),
            data_root=tmp_path / "data",
            declared_source_roots=("relative",),
        )
    with pytest.raises(ApplicationDataPathError):
        create_library_layout(
            LibraryConfig(name="Unsafe"),
            data_root=tmp_path / "data",
            declared_source_roots=(tmp_path / "base" / ".." / "source",),
        )


def test_disjoint_and_containment_helpers_resolve_symlinks(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)

    assert is_path_within(alias / "child", parent)
    with pytest.raises(PathOverlapError):
        ensure_paths_disjoint(alias, child)
    ensure_paths_disjoint(parent, tmp_path / "sibling")


def test_validate_layout_detects_missing_required_surface(tmp_path: Path) -> None:
    paths = create_library_layout(LibraryConfig(name="Research"), data_root=tmp_path / "data")
    paths.events.unlink()

    with pytest.raises(LibraryLayoutError, match="missing"):
        validate_library_layout(paths)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission invariant")
def test_permission_hardening_fails_closed_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "chmod", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))

    with pytest.raises(LibraryLayoutError, match="could not create"):
        create_library_layout(LibraryConfig(name="Research"), data_root=tmp_path / "data")


def test_collection_root_is_canonical_and_outside_application_data(tmp_path: Path) -> None:
    real_root = tmp_path / "corpus"
    real_root.mkdir()
    alias = tmp_path / "corpus-alias"
    alias.symlink_to(real_root, target_is_directory=True)

    root = build_collection_root(alias, data_root=tmp_path / "app-data")

    assert root.path == real_root.resolve()
    assert root.include_globs == (
        "**/*.md",
        "**/*.markdown",
        "**/*.txt",
        "**/*.pdf",
    )


@pytest.mark.parametrize("relative", ["relative", "base/../source"])
def test_collection_root_rejects_ambiguous_paths(relative: str, tmp_path: Path) -> None:
    with pytest.raises(CollectionRootPathError):
        build_collection_root(relative, data_root=tmp_path / "app-data")


def test_collection_root_must_be_an_existing_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "source.md"
    file_path.write_text("source")
    with pytest.raises(CollectionRootNotDirectoryError):
        build_collection_root(file_path, data_root=tmp_path / "app-data")
    with pytest.raises(CollectionRootNotDirectoryError):
        build_collection_root(tmp_path / "missing", data_root=tmp_path / "app-data")


def test_collection_root_rejects_application_data_overlap_in_both_directions(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "app-data"
    nested_source = data_root / "source"
    nested_source.mkdir(parents=True)
    with pytest.raises(CollectionRootOverlapError):
        build_collection_root(nested_source, data_root=data_root)

    broad_source = tmp_path / "broad-source"
    nested_data = broad_source / "app-data"
    broad_source.mkdir()
    with pytest.raises(CollectionRootOverlapError):
        build_collection_root(broad_source, data_root=nested_data)


def test_source_resolution_rejects_parent_absolute_missing_directory_and_symlink_escape(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "safe.md"
    source_file.write_text("safe")
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    escape = source_root / "escape.md"
    escape.symlink_to(outside)
    root = build_collection_root(source_root, data_root=tmp_path / "app-data")

    assert resolve_source_path(root, "safe.md") == source_file
    with pytest.raises(SourcePathEscapeError):
        resolve_source_path(root, "../outside.md")
    with pytest.raises(SourcePathEscapeError):
        resolve_source_path(root, outside)
    with pytest.raises(SourcePathEscapeError):
        resolve_source_path(root, "missing.md")
    with pytest.raises(SourcePathEscapeError):
        resolve_source_path(root, "escape.md")
    with pytest.raises(SourcePathEscapeError):
        resolve_source_path(root, ".")
    assert resolve_source_path(root, ".", require_file=False) == source_root


def test_collection_globs_are_explicit_relative_and_unique(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    root = build_collection_root(
        source_root,
        data_root=tmp_path / "app-data",
        include_globs=("papers/**/*.md",),
        exclude_globs=("private/**",),
    )
    assert root.include_globs == ("papers/**/*.md",)
    assert validate_globs(("papers/**/*.md",)) == ("papers/**/*.md",)

    for globs in (("a", "a"), ("",), ("../private",), (str(tmp_path / "absolute"),)):
        with pytest.raises(InvalidCollectionGlobError):
            validate_globs(globs)
