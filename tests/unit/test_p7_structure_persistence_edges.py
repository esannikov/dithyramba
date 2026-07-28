"""Integrity edges for metadata-only StructureUnit generation persistence."""

from __future__ import annotations

import pytest

from dithyramba.contracts import canonical_json_bytes
from dithyramba.ingest import FragmentKind, MarkdownSourceAddress, PdfSourceAddress
from dithyramba.persistence.structure import (
    SQLiteStructureRepository,
    _parse_structure_source_address,
    _PermittedStructureMetadata,
)
from dithyramba.structure import StructureIntegrityError


def _markdown_payload() -> dict[str, object]:
    return MarkdownSourceAddress(("Evidence",), 1, 2, 0, 10).payload()


def test_source_address_parser_reconstructs_only_markdown() -> None:
    markdown = canonical_json_bytes(_markdown_payload()).decode("utf-8")
    pdf = canonical_json_bytes(
        PdfSourceAddress(1, ("0.000", "0.000", "1.000", "1.000"), 0, 1).payload()
    ).decode("utf-8")

    assert _parse_structure_source_address(markdown) == ("markdown", ("Evidence",))
    assert _parse_structure_source_address(pdf) == ("pdf", None)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{", "valid JSON"),
        ("[]", "JSON object"),
        ('{"kind":"pdf","value":1.5}', "canonicalized"),
        ('{ "kind": "pdf" }', "canonical JSON"),
        (canonical_json_bytes({"kind": None}).decode(), "kind"),
        (canonical_json_bytes({"kind": ""}).decode(), "kind"),
        (canonical_json_bytes({"kind": " pdf "}).decode(), "kind"),
        (canonical_json_bytes({"kind": "x" * 65}).decode(), "kind"),
        (canonical_json_bytes({"kind": "x\x00"}).decode(), "kind"),
        (
            canonical_json_bytes(_markdown_payload() | {"extra": True}).decode(),
            "fields",
        ),
        (
            canonical_json_bytes(_markdown_payload() | {"heading_path": "Evidence"}).decode(),
            "heading_path",
        ),
        (
            canonical_json_bytes(_markdown_payload() | {"heading_path": [7]}).decode(),
            "heading_path",
        ),
        (
            canonical_json_bytes(_markdown_payload() | {"line_start": True}).decode(),
            "integer field",
        ),
        (
            canonical_json_bytes(_markdown_payload() | {"line_start": 0}).decode(),
            "integer field",
        ),
        (
            canonical_json_bytes(_markdown_payload() | {"line_end": 0}).decode(),
            "integer field",
        ),
        (
            canonical_json_bytes(_markdown_payload() | {"char_start": -1}).decode(),
            "integer field",
        ),
        (
            canonical_json_bytes(_markdown_payload() | {"char_end": -1}).decode(),
            "integer field",
        ),
        (
            canonical_json_bytes(_markdown_payload() | {"line_end": 1, "line_start": 2}).decode(),
            "invalid",
        ),
        (
            canonical_json_bytes(_markdown_payload() | {"schema": "other"}).decode(),
            "typed reconstruction",
        ),
    ],
)
def test_source_address_parser_fails_closed_on_malformed_metadata(
    raw: str,
    message: str,
) -> None:
    with pytest.raises(StructureIntegrityError, match=message):
        _parse_structure_source_address(raw)


def _metadata(
    ordinal: int,
    *,
    address_kind: str = "markdown",
    heading_path: tuple[str, ...] | None = ("Evidence",),
    fragment_kind: FragmentKind = FragmentKind.PARAGRAPH,
) -> _PermittedStructureMetadata:
    return _PermittedStructureMetadata(
        source_fragment_id=f"fragment_a_{ordinal}",
        source_version_id="source_version_a",
        source_id="source_a",
        ordinal=ordinal,
        fragment_kind=fragment_kind,
        address_kind=address_kind,
        heading_path=heading_path,
    )


def _structure_repository_without_store() -> SQLiteStructureRepository:
    return object.__new__(SQLiteStructureRepository)


def test_markdown_input_preparation_rejects_mixed_address_kinds() -> None:
    structures = _structure_repository_without_store()

    with pytest.raises(StructureIntegrityError, match="mixed"):
        structures._prepare_markdown_inputs(
            (_metadata(0), _metadata(1, address_kind="pdf", heading_path=None))
        )


def test_markdown_input_preparation_maps_descriptor_contract_failures() -> None:
    structures = _structure_repository_without_store()

    with pytest.raises(StructureIntegrityError, match="descriptor"):
        structures._prepare_markdown_inputs((_metadata(0, fragment_kind=FragmentKind.PAGE_TEXT),))
    assert structures._prepare_markdown_inputs(()) == ((), ())
