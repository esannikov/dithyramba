from __future__ import annotations

import errno
import os
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from dithyramba.collections import CollectionRoot
from dithyramba.contracts import sha256_hex
from dithyramba.ingest import reader
from dithyramba.ingest.errors import (
    FileSizeLimitExceededError,
    InvalidSourcePathError,
    SourceChangedDuringIngestError,
    SourceNotRegularError,
    SourceSymlinkError,
    SourceUnavailableError,
    UnsupportedFormatError,
)
from dithyramba.ingest.models import MediaType, ParserLimits


def _root(path: Path, **values: Any) -> CollectionRoot:
    return CollectionRoot(path=path.resolve(), **values)


def test_discovery_is_sorted_globbed_and_supports_root_level_markdown(tmp_path: Path) -> None:
    (tmp_path / "z.md").write_text("z")
    (tmp_path / "a.markdown").write_text("a")
    (tmp_path / "notes.txt").write_text("n")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-invalid")
    (tmp_path / "ignored.csv").write_text("x")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.md").write_text("b")

    assert reader.discover_source_paths(_root(tmp_path)) == (
        "a.markdown",
        "nested/b.md",
        "notes.txt",
        "paper.pdf",
        "z.md",
    )


def test_discovery_applies_segment_globs_and_exclusions(tmp_path: Path) -> None:
    (tmp_path / "root.md").write_text("root")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "keep.md").write_text("keep")
    (docs / "private.md").write_text("private")
    deep = docs / "deep"
    deep.mkdir()
    (deep / "keep.md").write_text("deep")

    root = _root(
        tmp_path,
        include_globs=("docs/**/keep.md",),
        exclude_globs=("docs/private.*",),
    )
    assert reader.discover_source_paths(root) == ("docs/deep/keep.md", "docs/keep.md")


def test_discovery_matches_canonically_equivalent_unicode_globs(tmp_path: Path) -> None:
    decomposed_name = unicodedata.normalize("NFD", "Нейросети.md")
    composed_name = unicodedata.normalize("NFC", decomposed_name)
    assert decomposed_name != composed_name
    (tmp_path / decomposed_name).write_text("included", encoding="utf-8")

    root = _root(tmp_path, include_globs=(composed_name,))

    assert reader.discover_source_paths(root) == (decomposed_name,)


def test_discovery_applies_canonically_equivalent_unicode_exclusions(tmp_path: Path) -> None:
    decomposed_name = unicodedata.normalize("NFD", "приватний Нейрофайл.md")
    composed_name = unicodedata.normalize("NFC", decomposed_name)
    (tmp_path / decomposed_name).write_text("excluded", encoding="utf-8")

    root = _root(
        tmp_path,
        include_globs=("*.md",),
        exclude_globs=(composed_name,),
    )

    assert reader.discover_source_paths(root) == ()


def test_discovery_never_follows_symlinks_or_returns_nonregular_files(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hidden.md").write_text("hidden")
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "visible.md").write_text("visible")
    (sources / "file-link.md").symlink_to(sources / "visible.md")
    (sources / "dir-link").symlink_to(outside, target_is_directory=True)
    fifo = sources / "pipe.md"
    os.mkfifo(fifo)

    assert reader.discover_source_paths(_root(sources)) == ("visible.md",)


def test_discovery_rejects_symlink_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(SourceSymlinkError):
        reader.discover_source_paths(CollectionRoot(path=linked.absolute()))


def test_discovery_wraps_scan_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_scan(_descriptor: int) -> None:
        raise OSError(errno.EIO, "scan failed")

    monkeypatch.setattr(os, "scandir", fail_scan)
    with pytest.raises(SourceUnavailableError, match="scan"):
        reader.discover_source_paths(_root(tmp_path))


@pytest.mark.parametrize(
    ("name", "media_type"),
    [
        ("source.md", MediaType.MARKDOWN),
        ("source.markdown", MediaType.MARKDOWN),
        ("source.txt", MediaType.PLAIN_TEXT),
        ("source.pdf", MediaType.PDF),
    ],
)
def test_reader_captures_exact_bytes_identity_uri_and_media_type(
    tmp_path: Path,
    name: str,
    media_type: MediaType,
) -> None:
    nested = tmp_path / "space dir"
    nested.mkdir()
    data = b"exact\x00bytes"
    path = nested / name
    path.write_bytes(data)

    captured = reader.read_source_bytes(_root(tmp_path), f"space dir/{name}", ParserLimits())

    assert captured.data == data
    assert captured.media_type is media_type
    assert captured.content_sha256 == sha256_hex(data)
    assert captured.canonical_uri == path.as_uri()
    assert captured.source_modified_at.endswith("Z")
    assert len(captured.source_modified_at) == 27
    assert captured.identity.inode == path.stat().st_ino
    assert captured.identity.device == path.stat().st_dev


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "/absolute.md",
        "../escape.md",
        "a/../b.md",
        "./a.md",
        "a//b.md",
        "a\\b.md",
        "a.md/",
        "bad\x00.md",
    ],
)
def test_reader_rejects_ambiguous_paths(tmp_path: Path, relative_path: str) -> None:
    with pytest.raises(InvalidSourcePathError):
        reader.read_source_bytes(_root(tmp_path), relative_path, ParserLimits())


def test_reader_rejects_unsupported_extension_before_open(tmp_path: Path) -> None:
    (tmp_path / "source.csv").write_text("data")
    with pytest.raises(UnsupportedFormatError):
        reader.read_source_bytes(_root(tmp_path), "source.csv", ParserLimits())


def test_reader_rejects_final_and_intermediate_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real.md"
    real.write_text("real")
    (tmp_path / "linked.md").symlink_to(real)
    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "nested.md").write_text("nested")
    (tmp_path / "linked-directory").symlink_to(directory, target_is_directory=True)

    with pytest.raises(SourceSymlinkError):
        reader.read_source_bytes(_root(tmp_path), "linked.md", ParserLimits())
    with pytest.raises(SourceSymlinkError):
        reader.read_source_bytes(_root(tmp_path), "linked-directory/nested.md", ParserLimits())


def test_reader_rejects_missing_directory_and_nonregular_targets(tmp_path: Path) -> None:
    directory = tmp_path / "folder.md"
    directory.mkdir()
    (tmp_path / "component").write_text("not a directory")

    with pytest.raises(SourceUnavailableError):
        reader.read_source_bytes(_root(tmp_path), "missing.md", ParserLimits())
    with pytest.raises(SourceNotRegularError):
        reader.read_source_bytes(_root(tmp_path), "folder.md", ParserLimits())
    with pytest.raises(SourceNotRegularError):
        reader.read_source_bytes(_root(tmp_path), "component/child.md", ParserLimits())


def test_reader_enforces_size_before_allocation(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_bytes(b"1234")
    with pytest.raises(FileSizeLimitExceededError):
        reader.read_source_bytes(
            _root(tmp_path),
            "large.txt",
            ParserLimits(max_file_bytes=3),
        )


def test_reader_enforces_size_if_file_grows_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "growing.txt"
    path.write_bytes(b"123")
    original = reader._bounded_read

    def grow_then_read(descriptor: int, max_bytes: int) -> bytes:
        with path.open("ab") as stream:
            stream.write(b"4")
        return original(descriptor, max_bytes)

    monkeypatch.setattr(reader, "_bounded_read", grow_then_read)
    with pytest.raises(FileSizeLimitExceededError, match="during read"):
        reader.read_source_bytes(
            _root(tmp_path),
            "growing.txt",
            ParserLimits(max_file_bytes=3),
        )


def test_reader_detects_content_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "changing.md"
    path.write_text("before")
    original = reader._bounded_read

    def mutate_after_read(descriptor: int, max_bytes: int) -> bytes:
        data = original(descriptor, max_bytes)
        path.write_text("different content")
        return data

    monkeypatch.setattr(reader, "_bounded_read", mutate_after_read)
    with pytest.raises(SourceChangedDuringIngestError):
        reader.read_source_bytes(_root(tmp_path), "changing.md", ParserLimits())


def test_reader_detects_directory_entry_replaced_by_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "swapped.md"
    path.write_text("original")
    displaced = tmp_path / "displaced.md"
    original = reader._bounded_read

    def swap_after_read(descriptor: int, max_bytes: int) -> bytes:
        data = original(descriptor, max_bytes)
        path.rename(displaced)
        path.symlink_to(displaced)
        return data

    monkeypatch.setattr(reader, "_bounded_read", swap_after_read)
    with pytest.raises(SourceChangedDuringIngestError):
        reader.read_source_bytes(_root(tmp_path), "swapped.md", ParserLimits())


def test_reader_detects_removed_directory_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "removed.md"
    path.write_text("original")
    displaced = tmp_path / "removed-old.md"
    original = reader._bounded_read

    def remove_after_read(descriptor: int, max_bytes: int) -> bytes:
        data = original(descriptor, max_bytes)
        path.rename(displaced)
        return data

    monkeypatch.setattr(reader, "_bounded_read", remove_after_read)
    with pytest.raises(SourceChangedDuringIngestError):
        reader.read_source_bytes(_root(tmp_path), "removed.md", ParserLimits())


def test_reader_detects_configured_root_ancestor_replacement_and_never_returns_alias_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = tmp_path / "ancestor"
    collection = ancestor / "collection"
    collection.mkdir(parents=True)
    source = collection / "source.md"
    source.write_bytes(b"original ancestor bytes")
    displaced = tmp_path / "ancestor-displaced"
    observed_reads: list[bytes] = []
    original_read = reader._bounded_read

    def replace_ancestor_after_read(descriptor: int, max_bytes: int) -> bytes:
        data = original_read(descriptor, max_bytes)
        observed_reads.append(data)
        ancestor.rename(displaced)
        replacement = ancestor / "collection"
        replacement.mkdir(parents=True)
        (replacement / "source.md").write_bytes(b"alternate ancestor bytes")
        return data

    monkeypatch.setattr(reader, "_bounded_read", replace_ancestor_after_read)
    with pytest.raises(SourceChangedDuringIngestError):
        reader.read_source_bytes(_root(collection), "source.md", ParserLimits())
    assert observed_reads == [b"original ancestor bytes"]
    assert source.as_uri() == (ancestor / "collection" / "source.md").as_uri()
    assert (ancestor / "collection" / "source.md").read_bytes() == b"alternate ancestor bytes"


def test_reader_detects_intermediate_directory_replacement_and_never_returns_alias_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intermediate = tmp_path / "section"
    intermediate.mkdir()
    source = intermediate / "source.md"
    source.write_bytes(b"original intermediate bytes")
    displaced = tmp_path / "section-displaced"
    observed_reads: list[bytes] = []
    original_read = reader._bounded_read

    def replace_intermediate_after_read(descriptor: int, max_bytes: int) -> bytes:
        data = original_read(descriptor, max_bytes)
        observed_reads.append(data)
        intermediate.rename(displaced)
        intermediate.mkdir()
        (intermediate / "source.md").write_bytes(b"alternate intermediate bytes")
        return data

    monkeypatch.setattr(reader, "_bounded_read", replace_intermediate_after_read)
    with pytest.raises(SourceChangedDuringIngestError):
        reader.read_source_bytes(_root(tmp_path), "section/source.md", ParserLimits())
    assert observed_reads == [b"original intermediate bytes"]
    assert source.as_uri() == (intermediate / "source.md").as_uri()
    assert (intermediate / "source.md").read_bytes() == b"alternate intermediate bytes"


def test_reader_detects_configured_root_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "source.md").write_bytes(b"original root bytes")
    displaced = tmp_path / "collection-displaced"
    root = _root(collection)
    original_read = reader._bounded_read

    def replace_root_after_read(descriptor: int, max_bytes: int) -> bytes:
        data = original_read(descriptor, max_bytes)
        collection.rename(displaced)
        collection.mkdir()
        (collection / "source.md").write_bytes(b"alternate root bytes")
        return data

    monkeypatch.setattr(reader, "_bounded_read", replace_root_after_read)
    with pytest.raises(SourceChangedDuringIngestError):
        reader.read_source_bytes(root, "source.md", ParserLimits())


def test_reader_detects_configured_root_rename_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "source.md").write_bytes(b"source")
    displaced = tmp_path / "collection-displaced"
    root = _root(collection)
    original_read = reader._bounded_read

    def rename_root_after_read(descriptor: int, max_bytes: int) -> bytes:
        data = original_read(descriptor, max_bytes)
        collection.rename(displaced)
        return data

    monkeypatch.setattr(reader, "_bounded_read", rename_root_after_read)
    with pytest.raises(SourceChangedDuringIngestError):
        reader.read_source_bytes(root, "source.md", ParserLimits())


def test_reader_maps_intermediate_directory_symlink_introduced_during_read_to_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intermediate = tmp_path / "section"
    intermediate.mkdir()
    (intermediate / "source.md").write_bytes(b"source")
    displaced = tmp_path / "section-displaced"
    original_read = reader._bounded_read

    def replace_directory_with_symlink(descriptor: int, max_bytes: int) -> bytes:
        data = original_read(descriptor, max_bytes)
        intermediate.rename(displaced)
        intermediate.symlink_to(displaced, target_is_directory=True)
        return data

    monkeypatch.setattr(reader, "_bounded_read", replace_directory_with_symlink)
    with pytest.raises(SourceChangedDuringIngestError):
        reader.read_source_bytes(_root(tmp_path), "section/source.md", ParserLimits())


def test_post_parse_validation_uses_exact_reread_and_maps_path_replacement_to_changed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.md"
    path.write_bytes(b"captured")
    root = _root(tmp_path)
    captured = reader.read_source_bytes(root, "source.md", ParserLimits())
    reader.validate_source_unchanged(root, captured, ParserLimits())

    path.unlink()
    path.symlink_to(tmp_path / "missing.md")
    with pytest.raises(SourceChangedDuringIngestError, match="after capture"):
        reader.validate_source_unchanged(root, captured, ParserLimits())


def test_post_parse_validation_rejects_changed_bytes_and_propagates_reader_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.md"
    path.write_bytes(b"captured")
    root = _root(tmp_path)
    captured = reader.read_source_bytes(root, "source.md", ParserLimits())
    path.write_bytes(b"changed!")
    with pytest.raises(SourceChangedDuringIngestError, match="after capture"):
        reader.validate_source_unchanged(root, captured, ParserLimits())

    def changed_during_reread(
        _root: CollectionRoot,
        _relative_path: str,
        _limits: ParserLimits,
    ) -> None:
        raise SourceChangedDuringIngestError("raced")

    monkeypatch.setattr(reader, "read_source_bytes", changed_during_reread)
    with pytest.raises(SourceChangedDuringIngestError, match="raced"):
        reader.validate_source_unchanged(root, captured, ParserLimits())


def test_internal_root_chain_requires_an_absolute_normalized_path() -> None:
    with pytest.raises(SourceUnavailableError, match="absolute"):
        reader._open_configured_root_chain(Path("relative"))


def test_reader_wraps_descriptor_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "source.md").write_text("source")

    def fail_read(_descriptor: int, _size: int) -> bytes:
        raise OSError(errno.EIO, "read failed")

    monkeypatch.setattr(os, "read", fail_read)
    with pytest.raises(SourceUnavailableError, match="read failed"):
        reader.read_source_bytes(_root(tmp_path), "source.md", ParserLimits())


@pytest.mark.parametrize(
    ("error_number", "expected"),
    [
        (errno.ELOOP, SourceSymlinkError),
        (errno.ENOTDIR, SourceNotRegularError),
        (errno.ENOENT, SourceUnavailableError),
    ],
)
def test_open_errors_have_stable_typed_mapping(
    error_number: int,
    expected: type[Exception],
) -> None:
    with pytest.raises(expected):
        reader._raise_open_error(OSError(error_number, "failure"), "source", directory=False)


def test_no_follow_unavailable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(SourceUnavailableError, match="lacks"):
        reader._no_follow_flag()
