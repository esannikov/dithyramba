"""Typed P7 contracts for explicit StructureUnit retrieval context."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from dithyramba.access import RequestScope
from dithyramba.contracts import canonical_sha256_hex, sha256_hex

from .errors import StructureContractError

_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ID = re.compile(r"^[a-z][a-z0-9_]+$")


class StructureUnitKind(StrEnum):
    """Domain-neutral ordered StructureUnit kinds."""

    DOCUMENT = "document"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SCENE = "scene"
    BLOCK = "block"
    PARAGRAPH = "paragraph"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class StructureGenerationRequest:
    """Policy-, snapshot- and profile-bound StructureUnit generation request."""

    corpus_snapshot_id: str
    access_policy_id: str
    scope: RequestScope
    profile_id: str
    profile_version: str

    def __post_init__(self) -> None:
        _prefixed(self.corpus_snapshot_id, "snapshot_", "corpus_snapshot_id")
        _prefixed(self.access_policy_id, "policy_", "access_policy_id")
        if not isinstance(self.scope, RequestScope):
            raise StructureContractError("scope must be a RequestScope")
        if not _KEY.fullmatch(self.profile_id):
            raise StructureContractError("profile_id must be a lowercase canonical key")
        _text(self.profile_version, "profile_version", maximum=128)

    @property
    def scope_hash(self) -> str:
        return self.scope.exclusion_hash

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.structure_generation_request/1.0",
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "scope": {
                "library_id": self.scope.library_id,
                "snapshot_hash": self.scope.snapshot_hash,
                "purpose": self.scope.purpose,
                "collection_ids": list(self.scope.collection_ids),
                "exclusions": self.scope.exclusions.payload(),
            },
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
        }


@dataclass(frozen=True, slots=True)
class StructureUnitProposal:
    """One ordered, contiguous set of exact SourceFragments."""

    kind: StructureUnitKind
    source_fragment_ids: tuple[str, ...]
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, StructureUnitKind):
            raise StructureContractError("kind must be a StructureUnitKind")
        ids = _ids(self.source_fragment_ids, "fragment_", "SourceFragment IDs", 128)
        if not ids:
            raise StructureContractError("a StructureUnit requires at least one fragment")
        object.__setattr__(self, "source_fragment_ids", ids)
        if self.label is not None:
            _text(self.label, "label", maximum=500)

    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "label": self.label,
            "source_fragment_ids": list(self.source_fragment_ids),
        }


@dataclass(frozen=True, slots=True)
class StructureUnitMember:
    source_fragment_id: str
    member_order: int
    ordinal: int
    text_sha256: str

    def __post_init__(self) -> None:
        _prefixed(self.source_fragment_id, "fragment_", "source_fragment_id")
        _integer(self.member_order, 0, 127, "member_order")
        _integer(self.ordinal, 0, 2_147_483_647, "ordinal")
        _hash(self.text_sha256, "text_sha256")

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "member_order": self.member_order,
            "ordinal": self.ordinal,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True, slots=True)
class StructureUnit:
    structure_unit_id: str
    generation_id: str
    source_id: str
    source_version_id: str
    kind: StructureUnitKind
    label: str | None
    members: tuple[StructureUnitMember, ...]
    boundary_hash: str
    content_hash: str

    def __post_init__(self) -> None:
        _prefixed(self.structure_unit_id, "structure_unit_", "structure_unit_id")
        _prefixed(self.generation_id, "structure_generation_", "generation_id")
        _prefixed(self.source_id, "source_", "source_id")
        _prefixed(self.source_version_id, "source_version_", "source_version_id")
        if not isinstance(self.kind, StructureUnitKind):
            raise StructureContractError("kind must be a StructureUnitKind")
        if self.label is not None:
            _text(self.label, "label", maximum=500)
        if type(self.members) is not tuple or not self.members:
            raise StructureContractError("members must be a non-empty tuple")
        if any(not isinstance(item, StructureUnitMember) for item in self.members):
            raise StructureContractError("members must contain StructureUnitMember values")
        if tuple(item.member_order for item in self.members) != tuple(range(len(self.members))):
            raise StructureContractError("StructureUnit member_order must be contiguous")
        ordinals = tuple(item.ordinal for item in self.members)
        if ordinals != tuple(range(ordinals[0], ordinals[0] + len(ordinals))):
            raise StructureContractError("StructureUnit fragment ordinals must be contiguous")
        _hash(self.boundary_hash, "boundary_hash")
        _hash(self.content_hash, "content_hash")

    @property
    def first_ordinal(self) -> int:
        return self.members[0].ordinal

    @property
    def last_ordinal(self) -> int:
        return self.members[-1].ordinal

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.structure_unit/1.0",
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "kind": self.kind.value,
            "label": self.label,
            "members": [item.payload() for item in self.members],
            "boundary_hash": self.boundary_hash,
        }


@dataclass(frozen=True, slots=True)
class StructureGeneration:
    generation_id: str
    library_id: str
    corpus_snapshot_id: str
    access_policy_id: str
    policy_hash: str
    scope_hash: str
    permitted_set_hash: str
    profile_id: str
    profile_version: str
    units: tuple[StructureUnit, ...]
    generation_hash: str

    def __post_init__(self) -> None:
        _prefixed(self.generation_id, "structure_generation_", "generation_id")
        _prefixed(self.library_id, "library_", "library_id")
        _prefixed(self.corpus_snapshot_id, "snapshot_", "corpus_snapshot_id")
        _prefixed(self.access_policy_id, "policy_", "access_policy_id")
        for value, label in (
            (self.policy_hash, "policy_hash"),
            (self.scope_hash, "scope_hash"),
            (self.permitted_set_hash, "permitted_set_hash"),
            (self.generation_hash, "generation_hash"),
        ):
            _hash(value, label)
        if not _KEY.fullmatch(self.profile_id):
            raise StructureContractError("profile_id must be a lowercase canonical key")
        _text(self.profile_version, "profile_version", maximum=128)
        if type(self.units) is not tuple or not 1 <= len(self.units) <= 5_000:
            raise StructureContractError("units must contain between 1 and 5,000 values")
        if any(not isinstance(item, StructureUnit) for item in self.units):
            raise StructureContractError("units must contain StructureUnit values")
        if any(item.generation_id != self.generation_id for item in self.units):
            raise StructureContractError("StructureUnit generation IDs do not match")
        if tuple(item.structure_unit_id for item in self.units) != tuple(
            sorted(item.structure_unit_id for item in self.units)
        ):
            raise StructureContractError("StructureUnits must use canonical ID order")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.structure_generation/1.0",
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "policy_hash": self.policy_hash,
            "scope_hash": self.scope_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "units": [
                item.semantic_payload()
                | {"structure_unit_id": item.structure_unit_id, "content_hash": item.content_hash}
                for item in self.units
            ],
        }


@dataclass(frozen=True, slots=True)
class StructureReadReceiptItem:
    source_fragment_id: str
    member_order: int
    text_sha256: str
    rendered_start: int
    rendered_end: int

    def __post_init__(self) -> None:
        _prefixed(self.source_fragment_id, "fragment_", "source_fragment_id")
        _integer(self.member_order, 0, 127, "member_order")
        _hash(self.text_sha256, "text_sha256")
        _integer(self.rendered_start, 0, 9_999_999, "rendered_start")
        _integer(self.rendered_end, 1, 10_000_000, "rendered_end")
        if self.rendered_end <= self.rendered_start:
            raise StructureContractError("rendered_end must be greater than rendered_start")

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "member_order": self.member_order,
            "text_sha256": self.text_sha256,
            "rendered_start": self.rendered_start,
            "rendered_end": self.rendered_end,
        }


@dataclass(frozen=True, slots=True)
class StructureReadReceipt:
    receipt_id: str
    structure_unit_id: str
    library_id: str
    corpus_snapshot_id: str
    access_policy_id: str
    policy_hash: str
    scope_hash: str
    permitted_set_hash: str
    delimiter: str
    rendered_text_sha256: str
    items: tuple[StructureReadReceiptItem, ...]
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _prefixed(self.receipt_id, "structure_read_", "receipt_id")
        _prefixed(self.structure_unit_id, "structure_unit_", "structure_unit_id")
        _prefixed(self.library_id, "library_", "library_id")
        _prefixed(self.corpus_snapshot_id, "snapshot_", "corpus_snapshot_id")
        _prefixed(self.access_policy_id, "policy_", "access_policy_id")
        for value, label in (
            (self.policy_hash, "policy_hash"),
            (self.scope_hash, "scope_hash"),
            (self.permitted_set_hash, "permitted_set_hash"),
            (self.rendered_text_sha256, "rendered_text_sha256"),
        ):
            _hash(value, label)
        if self.delimiter != "\n\n":
            raise StructureContractError("StructureUnit delimiter is frozen to two newlines")
        if type(self.items) is not tuple or not self.items:
            raise StructureContractError("StructureReadReceipt requires member items")
        if any(not isinstance(item, StructureReadReceiptItem) for item in self.items):
            raise StructureContractError("receipt items must be StructureReadReceiptItem values")
        if tuple(item.member_order for item in self.items) != tuple(range(len(self.items))):
            raise StructureContractError("receipt member order must be contiguous")
        object.__setattr__(self, "receipt_hash", canonical_sha256_hex(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.structure_read_receipt/1.0",
            "structure_unit_id": self.structure_unit_id,
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "policy_hash": self.policy_hash,
            "scope_hash": self.scope_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "delimiter": self.delimiter,
            "rendered_text_sha256": self.rendered_text_sha256,
            "items": [item.payload() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class StructureUnitText:
    """Rendered context plus the exact member spans that make it honest."""

    structure_unit: StructureUnit
    text: str
    receipt: StructureReadReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.structure_unit, StructureUnit):
            raise StructureContractError("structure_unit must be a StructureUnit")
        if type(self.text) is not str or not self.text:
            raise StructureContractError("rendered StructureUnit text must be non-empty")
        if not isinstance(self.receipt, StructureReadReceipt):
            raise StructureContractError("receipt must be a StructureReadReceipt")
        if self.receipt.structure_unit_id != self.structure_unit.structure_unit_id:
            raise StructureContractError("StructureUnit and receipt IDs differ")
        if sha256_hex(self.text.encode("utf-8")) != self.receipt.rendered_text_sha256:
            raise StructureContractError("rendered StructureUnit text hash differs from receipt")


def _prefixed(value: object, prefix: str, label: str) -> str:
    if type(value) is not str or not value.startswith(prefix) or _ID.fullmatch(value) is None:
        raise StructureContractError(f"{label} must use the {prefix} prefix")
    return value


def _hash(value: object, label: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise StructureContractError(f"{label} must be lowercase SHA-256")
    return value


def _text(value: object, label: str, *, maximum: int) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or not value
        or len(value) > maximum
        or "\x00" in value
    ):
        raise StructureContractError(f"{label} must be non-empty trimmed text")
    return value


def _integer(value: object, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise StructureContractError(f"{label} must be between {minimum} and {maximum}")
    return value


def _ids(values: object, prefix: str, label: str, maximum: int) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) > maximum:
        raise StructureContractError(f"{label} must be a tuple of at most {maximum} IDs")
    result = tuple(_prefixed(item, prefix, label) for item in values)
    if len(set(result)) != len(result):
        raise StructureContractError(f"{label} must be unique")
    return result
