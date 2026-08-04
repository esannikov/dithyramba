from __future__ import annotations

import io
import json
import stat
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest

import dithyramba.connectors.books as books
from dithyramba.connectors import (
    BOOK_PROJECTION_SCHEMA,
    DEFAULT_BOOK_PROFILE,
    LARGE_BOOK_PROFILE,
    BookMediaType,
    BookProjection,
    BookProjectionStatus,
    BookSource,
    BookSourceMember,
    project_book,
    project_book_catalog,
    project_book_path,
)
from dithyramba.contracts import sha256_hex
from dithyramba.ingest import (
    FragmentKind,
    ParsedFragment,
    ParseResult,
    ParserLimits,
    PdfSourceAddress,
    SourceBytes,
)
from dithyramba.ingest.errors import ParserTimeoutError


def _epub(
    documents: list[tuple[str, bytes]],
    *,
    rootfile: str = "OEBPS/content.opf",
    hrefs: list[str] | None = None,
    extras: list[tuple[zipfile.ZipInfo | str, bytes]] | None = None,
) -> bytes:
    hrefs = hrefs or [name.removeprefix("OEBPS/") for name, _content in documents]
    items = "".join(
        f'<item id="item{index}" href="{href}" media-type="application/xhtml+xml"/>'
        for index, href in enumerate(hrefs)
    )
    spine = "".join(f'<itemref idref="item{index}"/>' for index in range(len(hrefs)))
    opf = (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf">'
        f"<manifest>{items}</manifest><spine>{spine}</spine></package>"
    ).encode()
    container = (
        '<?xml version="1.0"?>'
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        f'<rootfiles><rootfile full-path="{rootfile}"/></rootfiles></container>'
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr(rootfile, opf)
        for path, content in documents:
            archive.writestr(path, content)
        for extra_path, content in extras or []:
            archive.writestr(extra_path, content)
    return output.getvalue()


def _xhtml(body: str) -> bytes:
    return (
        '<?xml version="1.0"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        f"{body}</body></html>"
    ).encode()


def _payload(projection: books.BookProjection) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(projection.sidecar_json))


def _failure_code(source: BookSource) -> str:
    projection = project_book(source)
    assert projection.status is BookProjectionStatus.FAILURE
    assert projection.markdown == ""
    return str(_payload(projection)["failure"]["code"])


def test_epub_projection_is_deterministic_spine_ordered_and_exactly_located() -> None:
    data = _epub(
        [
            ("OEBPS/second.xhtml", _xhtml('<h1 id="second">Second</h1><p>Beta</p>')),
            ("OEBPS/first.xhtml", _xhtml("<h1>First</h1><p>Alpha</p>")),
        ]
    )
    source = BookSource(name="Novel.epub", media_type=BookMediaType.EPUB, data=data)

    first = project_book(source)
    second = project_book(source)

    assert first == second
    assert first.status is BookProjectionStatus.COMPLETE
    assert first.markdown.index("Second") < first.markdown.index("First")
    payload = _payload(first)
    assert payload["schema"] == BOOK_PROJECTION_SCHEMA
    assert payload["source"]["sha256"] == sha256_hex(data)
    assert payload["metadata"]["roles"] == ["private"]
    assert payload["optional_sidecars"]["docling"] == {
        "contract": "dithyramba.book_docling_sidecar/1.0",
        "status": "not_run",
    }
    assert payload["markdown"]["sha256"] == first.markdown_sha256
    assert [unit["source_locator"]["spine_index"] for unit in payload["units"]] == [0, 0, 1, 1]
    first_locator = payload["units"][0]["source_locator"]
    assert first_locator == {
        "archive_path": "OEBPS/second.xhtml",
        "element_id": "second",
        "kind": "epub",
        "spine_index": 0,
        "xml_path": (
            "/{http://www.w3.org/1999/xhtml}html[1]"
            "/{http://www.w3.org/1999/xhtml}body[1]"
            "/{http://www.w3.org/1999/xhtml}h1[1]"
        ),
    }
    for unit in payload["units"]:
        projected_text = first.markdown[unit["markdown_char_start"] : unit["markdown_char_end"]]
        assert sha256_hex(projected_text.encode()) == unit["source_text_sha256"]


def test_epub_reports_partial_when_a_spine_item_has_no_text() -> None:
    data = _epub(
        [
            ("OEBPS/blank.xhtml", _xhtml("<div/>")),
            ("OEBPS/chapter.xhtml", _xhtml("<p>Readable</p>")),
        ]
    )
    projection = project_book(BookSource("partial.epub", BookMediaType.EPUB, data))

    assert projection.status is BookProjectionStatus.PARTIAL
    assert _payload(projection)["warnings"] == ["empty_spine_item:0"]
    assert "Readable" in projection.markdown


def test_epub_preserves_structural_text_without_polluting_markdown() -> None:
    data = _epub(
        [
            (
                "OEBPS/chapter.xhtml",
                _xhtml(
                    "<table><tr><td>Measured value</td></tr></table>"
                    "<dl><dt>Term</dt><dd>Definition</dd></dl>"
                    "<aside>Scholarly note</aside>"
                    "<figure><figcaption>Archive caption</figcaption></figure>"
                ),
            )
        ]
    )

    projection = project_book(BookSource("apparatus.epub", BookMediaType.EPUB, data))
    payload = _payload(projection)

    assert projection.status is BookProjectionStatus.COMPLETE
    assert payload["warnings"] == []
    assert [item["kind"] for item in payload["units"]] == [
        "table",
        "definition_list",
        "note",
        "caption",
    ]
    assert all(
        text in projection.markdown
        for text in ("Measured value", "Term Definition", "Scholarly note", "Archive caption")
    )
    assert "<!-- unit:" not in projection.markdown


def test_epub_reports_unknown_dropped_text_as_partial() -> None:
    data = _epub(
        [
            (
                "OEBPS/chapter.xhtml",
                _xhtml("<custom>Unclassified text</custom><p>Retained text</p>"),
            )
        ]
    )

    projection = project_book(BookSource("unknown.epub", BookMediaType.EPUB, data))

    assert projection.status is BookProjectionStatus.PARTIAL
    assert _payload(projection)["warnings"] == ["dropped_block:custom:spine:0"]
    assert "Unclassified text" not in projection.markdown
    assert "Retained text" in projection.markdown


@pytest.mark.parametrize(
    ("extra", "expected_code"),
    [
        (("../escape", b"x"), "archive_path_traversal"),
        (("/absolute", b"x"), "archive_path_traversal"),
    ],
)
def test_epub_rejects_archive_traversal(extra: tuple[str, bytes], expected_code: str) -> None:
    data = _epub([("OEBPS/ch.xhtml", _xhtml("<p>Text</p>"))], extras=[extra])
    assert _failure_code(BookSource("bad.epub", BookMediaType.EPUB, data)) == expected_code


def test_epub_rejects_duplicate_members_and_symlinks() -> None:
    document = ("OEBPS/ch.xhtml", _xhtml("<p>Text</p>"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name")
        duplicate = _epub([document], extras=[document])
    assert _failure_code(BookSource("duplicate.epub", BookMediaType.EPUB, duplicate)) == (
        "duplicate_archive_member"
    )

    link = zipfile.ZipInfo("OEBPS/link.xhtml")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    symlink = _epub([document], extras=[(link, b"target.xhtml")])
    assert _failure_code(BookSource("link.epub", BookMediaType.EPUB, symlink)) == (
        "archive_symlink_rejected"
    )


def test_epub_rejects_external_fetch_reference_and_xml_entities() -> None:
    external = _epub(
        [("OEBPS/ch.xhtml", _xhtml("<p>Text</p>"))],
        hrefs=["https://example.invalid/ch.xhtml"],
    )
    assert _failure_code(BookSource("external.epub", BookMediaType.EPUB, external)) == (
        "external_reference_rejected"
    )

    unsafe = _epub(
        [
            (
                "OEBPS/ch.xhtml",
                b'<!DOCTYPE html [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
                b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>&xxe;</p></body></html>',
            )
        ]
    )
    assert _failure_code(BookSource("entity.epub", BookMediaType.EPUB, unsafe)) == (
        "unsafe_or_invalid_xml"
    )


def test_epub_strips_only_known_external_xhtml_doctype_without_resolution() -> None:
    doctype = (
        b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" '
        b'"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">'
    )
    xhtml = b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Compatible</p></body></html>'
    benign = _epub(
        [
            (
                "OEBPS/ch.xhtml",
                b'<?xml version="1.0" encoding="utf-8"?>\n' + doctype + xhtml,
            )
        ]
    )
    projection = project_book(BookSource("doctype.epub", BookMediaType.EPUB, benign))
    assert projection.status is BookProjectionStatus.COMPLETE
    assert "Compatible" in projection.markdown

    unknown_external = _epub(
        [
            (
                "OEBPS/ch.xhtml",
                b'<!DOCTYPE html SYSTEM "file:///etc/passwd">'
                b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Unsafe</p></body></html>',
            )
        ]
    )
    multiple = _epub(
        [
            (
                "OEBPS/ch.xhtml",
                doctype
                + doctype
                + b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Ambiguous</p>'
                b"</body></html>",
            )
        ]
    )
    for name, data in (("unknown.epub", unknown_external), ("multiple.epub", multiple)):
        assert _failure_code(BookSource(name, BookMediaType.EPUB, data)) == (
            "unsafe_or_invalid_xml"
        )


def test_fb2_projection_follows_document_order_and_rejects_dtd() -> None:
    fb2 = (
        b'<?xml version="1.0"?>'
        b'<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        b"<body><section><title><p>Chapter</p></title><p>One</p>"
        b"<section><subtitle>Inside</subtitle><p>Two</p></section></section></body>"
        b"</FictionBook>"
    )
    projection = project_book(BookSource("story.fb2", BookMediaType.FB2, fb2))

    assert projection.status is BookProjectionStatus.COMPLETE
    assert projection.markdown.index("Chapter") < projection.markdown.index("One")
    assert projection.markdown.index("One") < projection.markdown.index("Inside")
    payload = _payload(projection)
    assert payload["units"][0]["source_locator"]["kind"] == "fb2"
    assert payload["units"][0]["source_locator"]["body_index"] == 0
    assert payload["units"][0]["source_locator"]["xml_path"].endswith("}p[1]")

    unsafe = b'<!DOCTYPE x [<!ENTITY x "boom">]><FictionBook><body><p>&x;</p></body></FictionBook>'
    assert _failure_code(BookSource("unsafe.fb2", BookMediaType.FB2, unsafe)) == (
        "unsafe_or_invalid_xml"
    )


def test_fb2_table_text_is_preserved_as_a_typed_unit() -> None:
    fb2 = (
        b'<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        b"<body><table><tr><td>Cell value</td></tr></table></body></FictionBook>"
    )

    projection = project_book(BookSource("table.fb2", BookMediaType.FB2, fb2))

    assert projection.status is BookProjectionStatus.COMPLETE
    assert "Cell value" in projection.markdown
    assert _payload(projection)["units"][0]["kind"] == "table"


def test_pdf_delegates_to_isolated_parser_and_keeps_page_bbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_source: list[SourceBytes] = []
    observed_limits: list[ParserLimits] = []

    def fake_parser(source: SourceBytes, limits: ParserLimits) -> ParseResult:
        observed_source.append(source)
        observed_limits.append(limits)
        return ParseResult.processed(
            (
                ParsedFragment(
                    ordinal=0,
                    kind=FragmentKind.PAGE_TEXT,
                    text="PDF text",
                    address=PdfSourceAddress(
                        page=7,
                        bbox=("1.000", "2.000", "3.000", "4.000"),
                        char_start=0,
                        char_end=8,
                    ),
                ),
            ),
            parser_revision="test-pdf",
            normalized_codepoints=8,
        )

    monkeypatch.setattr(books, "parse_pdf_source", fake_parser)
    projection = project_book(
        BookSource("book.pdf", BookMediaType.PDF, b"%PDF-frozen"),
        profile=LARGE_BOOK_PROFILE,
    )

    assert projection.status is BookProjectionStatus.COMPLETE
    assert observed_source[0].data == b"%PDF-frozen"
    assert observed_limits[0].max_pdf_pages == LARGE_BOOK_PROFILE.max_pdf_pages
    assert _payload(projection)["units"][0]["source_locator"] == {
        "bbox": ["1.000", "2.000", "3.000", "4.000"],
        "kind": "pdf",
        "page": 7,
    }

    def timeout(_source: SourceBytes, _limits: ParserLimits) -> ParseResult:
        raise ParserTimeoutError("timeout")

    monkeypatch.setattr(books, "parse_pdf_source", timeout)
    failed = project_book(BookSource("book.pdf", BookMediaType.PDF, b"%PDF-frozen"))
    assert failed.status is BookProjectionStatus.FAILURE
    assert _payload(failed)["failure"] == {"code": "parser_timeout"}


def test_profiles_are_distinct_bounded_and_fail_closed() -> None:
    landscapes_of_mars_bytes = 182_205_617
    assert LARGE_BOOK_PROFILE.profile_id != DEFAULT_BOOK_PROFILE.profile_id
    assert LARGE_BOOK_PROFILE.max_file_bytes > DEFAULT_BOOK_PROFILE.max_file_bytes
    assert LARGE_BOOK_PROFILE.max_file_bytes == 512 * 1024 * 1024
    assert landscapes_of_mars_bytes < LARGE_BOOK_PROFILE.max_file_bytes
    assert landscapes_of_mars_bytes > DEFAULT_BOOK_PROFILE.max_file_bytes
    assert LARGE_BOOK_PROFILE.max_pdf_pages > DEFAULT_BOOK_PROFILE.max_pdf_pages
    assert LARGE_BOOK_PROFILE.max_extracted_codepoints == 20_000_000
    assert LARGE_BOOK_PROFILE.timeout_seconds == 180
    tiny = replace(DEFAULT_BOOK_PROFILE, profile_id="book/tiny", max_file_bytes=3)
    source = BookSource("story.fb2", BookMediaType.FB2, b"four")

    projection = project_book(source, profile=tiny)

    payload = _payload(projection)
    assert projection.status is BookProjectionStatus.FAILURE
    assert payload["failure"] == {"code": "file_size_limit_exceeded"}
    assert payload["source"]["sha256"] == sha256_hex(b"four")
    assert payload["profile"] == {"id": "book/tiny"}


def test_source_requires_immutable_bytes_and_unique_roles() -> None:
    assert BookSource("generic.fb2", BookMediaType.FB2, b"x").roles == ("private",)
    with pytest.raises(TypeError, match="immutable bytes"):
        BookSource("bad.fb2", BookMediaType.FB2, bytearray(b"x"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        BookSource("bad.fb2", BookMediaType.FB2, b"x", roles=("private", "private"))


def test_expanded_epub_directory_is_captured_read_only_and_catalogued(tmp_path: Path) -> None:
    input_root = tmp_path / "raw" / "private" / "books"
    expanded = input_root / "Destination Mars.epub"
    (expanded / "META-INF").mkdir(parents=True)
    (expanded / "OEBPS").mkdir()
    (expanded / "mimetype").write_bytes(b"application/epub+zip")
    (expanded / "META-INF" / "container.xml").write_bytes(
        b'<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        b'<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>'
    )
    (expanded / "OEBPS" / "content.opf").write_bytes(
        b'<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
        b'<item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>'
        b'</manifest><spine><itemref idref="chapter"/></spine></package>'
    )
    (expanded / "OEBPS" / "chapter.xhtml").write_bytes(_xhtml("<p>Mars text</p>"))

    first = project_book_path(expanded, profile=DEFAULT_BOOK_PROFILE)
    second = project_book_path(expanded, profile=DEFAULT_BOOK_PROFILE)

    assert first == second
    assert first.status is BookProjectionStatus.COMPLETE
    source_payload = _payload(first)["source"]
    assert source_payload["form"] == "expanded_epub_repack"
    assert [member["path"] for member in source_payload["members"]] == [
        "META-INF/container.xml",
        "OEBPS/chapter.xhtml",
        "OEBPS/content.opf",
        "mimetype",
    ]
    output_root = tmp_path / "derived" / "private" / "books"
    summary = project_book_catalog(
        input_root,
        output_root,
        profile=DEFAULT_BOOK_PROFILE,
    )
    assert (summary.complete, summary.partial, summary.failure, summary.total) == (1, 0, 0, 1)
    assert (output_root / "Destination Mars.epub.md").read_text() == first.markdown
    assert (
        output_root / "Destination Mars.epub.book-projection.json"
    ).read_bytes() == first.sidecar_json
    assert stat.S_IMODE((output_root / "Destination Mars.epub.md").stat().st_mode) == 0o600
    assert stat.S_IMODE(output_root.stat().st_mode) == 0o700


def test_expanded_epub_symlink_is_explicitly_rejected(tmp_path: Path) -> None:
    expanded = tmp_path / "Mapping Mars.epub"
    expanded.mkdir()
    (expanded / "mimetype").write_bytes(b"application/epub+zip")
    (expanded / "linked").symlink_to(expanded / "mimetype")

    with pytest.raises(ValueError, match="symlink rejected"):
        project_book_path(expanded)


def test_archive_and_text_bounds_return_stable_failures() -> None:
    data = _epub([("OEBPS/ch.xhtml", _xhtml("<p>Long enough text</p>"))])
    source = BookSource("bounded.epub", BookMediaType.EPUB, data)
    member_limited = replace(
        DEFAULT_BOOK_PROFILE,
        profile_id="book/member-limited",
        max_archive_members=3,
    )
    text_limited = replace(
        DEFAULT_BOOK_PROFILE,
        profile_id="book/text-limited",
        max_extracted_codepoints=4,
    )
    xml_limited = replace(
        DEFAULT_BOOK_PROFILE,
        profile_id="book/xml-limited",
        max_xml_elements=2,
    )

    assert _payload(project_book(source, profile=member_limited))["failure"] == {
        "code": "archive_member_limit_exceeded"
    }
    assert _payload(project_book(source, profile=text_limited))["failure"] == {
        "code": "extracted_text_limit_exceeded"
    }
    assert _payload(project_book(source, profile=xml_limited))["failure"] == {
        "code": "xml_element_limit_exceeded"
    }


def test_module_help_documents_private_only_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        books.main(["--help"])

    assert exit_info.value.code == 0
    assert "manifest role; repeatable (default: private)" in capsys.readouterr().out


def test_connector_contract_objects_fail_closed_on_invalid_values() -> None:
    with pytest.raises(ValueError, match="profile_id"):
        replace(DEFAULT_BOOK_PROFILE, profile_id="")
    with pytest.raises(ValueError, match="max_units"):
        replace(DEFAULT_BOOK_PROFILE, profile_id="book/test", max_units=0)

    digest = "0" * 64
    with pytest.raises(ValueError, match="byte_size"):
        BookSourceMember("chapter.xhtml", -1, digest)
    with pytest.raises(ValueError, match="sha256"):
        BookSourceMember("chapter.xhtml", 1, "NOT-A-HASH")
    with pytest.raises(ValueError, match="name"):
        BookSource("", BookMediaType.FB2, b"x")
    with pytest.raises(TypeError, match="media_type"):
        BookSource("x", "fb2", b"x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="roles"):
        BookSource("x", BookMediaType.FB2, b"x", roles=("",))
    with pytest.raises(ValueError, match="source_form"):
        BookSource("x", BookMediaType.FB2, b"x", source_form="unknown")
    with pytest.raises(TypeError, match="members"):
        BookSource("x", BookMediaType.FB2, b"x", members=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="file sources"):
        BookSource(
            "x",
            BookMediaType.EPUB,
            b"x",
            members=(BookSourceMember("chapter.xhtml", 1, digest),),
        )
    with pytest.raises(ValueError, match="require EPUB"):
        BookSource("x", BookMediaType.FB2, b"x", source_form="expanded_epub_repack")
    with pytest.raises(ValueError, match="path-sorted"):
        BookSource(
            "x.epub",
            BookMediaType.EPUB,
            b"x",
            source_form="expanded_epub_repack",
            members=(
                BookSourceMember("z.xhtml", 1, digest),
                BookSourceMember("a.xhtml", 1, digest),
            ),
        )

    with pytest.raises(TypeError, match="status"):
        BookProjection("complete", "", b"{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="markdown"):
        BookProjection(BookProjectionStatus.COMPLETE, b"", b"{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sidecar_json"):
        BookProjection(BookProjectionStatus.COMPLETE, "", "{}")  # type: ignore[arg-type]
    projection = BookProjection(BookProjectionStatus.COMPLETE, "x", b"{}")
    assert projection.sidecar_sha256 == sha256_hex(b"{}")


def test_connector_public_entrypoints_reject_wrong_types_and_paths(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="source must"):
        project_book(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="profile must"):
        project_book(BookSource("x.fb2", BookMediaType.FB2, b"x"), profile=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absolute"):
        project_book_path(Path("relative.epub"))
    with pytest.raises(ValueError, match="unavailable"):
        project_book_path((tmp_path / "absent.epub").resolve())

    ordinary_directory = tmp_path / "ordinary"
    ordinary_directory.mkdir()
    with pytest.raises(ValueError, match=r"only \.epub"):
        project_book_path(ordinary_directory.resolve())

    unsupported = tmp_path / "book.txt"
    unsupported.write_text("text")
    with pytest.raises(ValueError, match="unsupported book source suffix"):
        project_book_path(unsupported.resolve())

    with pytest.raises(ValueError, match="input directory must be absolute"):
        project_book_catalog(Path("relative"), tmp_path / "output")
    with pytest.raises(ValueError, match="input directory is unavailable"):
        project_book_catalog((tmp_path / "missing").resolve(), tmp_path / "output")
    file_root = tmp_path / "not-a-directory"
    file_root.write_text("x")
    with pytest.raises(ValueError, match="real directory"):
        project_book_catalog(file_root.resolve(), tmp_path / "output")

    input_root = tmp_path / "catalog"
    input_root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        project_book_catalog(input_root.resolve(), (input_root / "derived").resolve())


def test_invalid_epub_and_fb2_structures_have_stable_failure_codes() -> None:
    assert _failure_code(BookSource("bad.epub", BookMediaType.EPUB, b"not-a-zip")) == (
        "invalid_epub"
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name")
        bad_mimetype = _epub(
            [("OEBPS/ch.xhtml", _xhtml("<p>Text</p>"))],
            extras=[("mimetype", b"text/plain")],
        )
        assert (
            _failure_code(BookSource("mimetype.epub", BookMediaType.EPUB, bad_mimetype))
            == "duplicate_archive_member"
        )

    no_body = b'<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"/>'
    wrong_root = b"<Book><body><p>Text</p></body></Book>"
    assert _failure_code(BookSource("nobody.fb2", BookMediaType.FB2, no_body)) == (
        "invalid_fb2_body"
    )
    assert _failure_code(BookSource("root.fb2", BookMediaType.FB2, wrong_root)) == (
        "invalid_fb2_root"
    )

    empty = b'<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"><body/></FictionBook>'
    projection = project_book(BookSource("empty.fb2", BookMediaType.FB2, empty))
    assert projection.status is BookProjectionStatus.FAILURE
    assert _payload(projection)["failure"] == {"code": "no_extractable_text"}


def test_reference_xml_and_unit_helpers_reject_ambiguous_inputs() -> None:
    with pytest.raises(books._ProjectionFailure, match="not local"):
        books._resolve_archive_reference("OEBPS/content.opf", "https://example.test/ch")
    with pytest.raises(books._ProjectionFailure, match="escapes root"):
        books._resolve_archive_reference("", "../outside.xhtml")
    with pytest.raises(books._ProjectionFailure, match="invalid"):
        books._resolve_archive_reference("", "")
    with pytest.raises(TypeError, match="immutable bytes"):
        books._parse_xml("<x/>", 2)  # type: ignore[arg-type]
    with pytest.raises(books._ProjectionFailure, match="non-text tag"):
        books._split_tag(object())
    with pytest.raises(TypeError, match="spine_index"):
        books._locator_label({"kind": "epub", "spine_index": "0"})
    with pytest.raises(TypeError, match="body_index"):
        books._locator_label({"kind": "fb2", "body_index": "0"})

    unit = books._Unit(
        kind="paragraph",
        label="Body 1",
        text="text",
        locator=MappingProxyType({"kind": "fb2", "body_index": 0}),
    )
    with pytest.raises(books._ProjectionFailure, match="too many text units"):
        books._validate_unit_bounds(
            [unit, unit], replace(DEFAULT_BOOK_PROFILE, profile_id="units", max_units=1)
        )


def test_module_main_reports_success_failure_and_usage_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    assert (
        books.main(
            [
                "--input-dir",
                str(source),
                "--output-dir",
                str(output),
                "--profile",
                "default",
                "--role",
                "scholarly",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "complete": 0,
        "failure": 0,
        "partial": 0,
        "total": 0,
    }

    assert (
        books.main(
            [
                "--input-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "other"),
            ]
        )
        == 2
    )
    assert "book projection failed" in capsys.readouterr().err
