"""Content-addressed compatibility contracts for external recall inputs.

The contracts in this module freeze three boundaries that used to be caller
convention only:

* how raw fragment text became the canonical retrieval projection;
* how external identifiers resolve into one explicit Dithyramba Library; and
* which exact permitted fragments have explicit proof metadata.

They contain no persistence or provider logic.  A later adapter may store
their canonical payloads, but it must not infer omitted proof eligibility or
silently repair a scope, coverage, collision, or hash mismatch.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*_[a-z0-9]+(?:_[a-z0-9]+)*$")
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_MAX_FRAGMENTS = 5_000


class CompatibilityContractError(ValueError):
    """A compatibility artifact violates its frozen fail-closed contract."""


class _FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class FragmentTextProjectionReceipt(_FrozenContract):
    """Text-free receipt for one raw-to-canonical fragment projection.

    ``raw_trim_start`` is inclusive and ``raw_trim_end`` is exclusive.  The
    frozen ``unicode_nfc_trim_v1`` profile permits only removal of surrounding
    whitespace followed by NFC normalization.  :meth:`from_texts` and
    :meth:`verify_texts` perform the content-level replay when both texts are
    available; the persisted receipt itself retains only hashes, counts and
    offsets.
    """

    SCHEMA: ClassVar[str] = "dithyramba.fragment_text_projection_receipt/1.0"

    library_id: str
    source_version_id: str
    source_fragment_id: str
    projection_profile: Literal["unicode_nfc_trim_v1"] = "unicode_nfc_trim_v1"
    raw_text_sha256: str
    projected_text_sha256: str
    raw_character_count: int = Field(ge=1, le=2_000_000)
    projected_character_count: int = Field(ge=1, le=2_000_000)
    raw_trim_start: int = Field(ge=0, le=2_000_000)
    raw_trim_end: int = Field(ge=1, le=2_000_000)

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("source_version_id")
    @classmethod
    def _source_version_id(cls, value: str) -> str:
        return _require_id(value, "source_version")

    @field_validator("source_fragment_id")
    @classmethod
    def _source_fragment_id(cls, value: str) -> str:
        return _require_id(value, "fragment")

    @field_validator("raw_text_sha256", "projected_text_sha256")
    @classmethod
    def _hashes(cls, value: str, info: Any) -> str:
        return _require_hash(value, info.field_name)

    @model_validator(mode="after")
    def _closed_offsets(self) -> Self:
        if self.raw_trim_end > self.raw_character_count:
            raise CompatibilityContractError("raw_trim_end exceeds the raw character count")
        if self.raw_trim_start >= self.raw_trim_end:
            raise CompatibilityContractError("raw trim offsets must select a non-empty span")
        if self.projected_character_count > self.raw_trim_end - self.raw_trim_start:
            raise CompatibilityContractError(
                "NFC projection cannot exceed the selected raw character span"
            )
        if self.raw_text_sha256 == self.projected_text_sha256 and (
            self.raw_trim_start != 0
            or self.raw_trim_end != self.raw_character_count
            or self.projected_character_count != self.raw_character_count
        ):
            raise CompatibilityContractError(
                "identical raw/projected hashes require an identity projection"
            )
        return self

    @classmethod
    def from_texts(
        cls,
        *,
        library_id: str,
        source_version_id: str,
        source_fragment_id: str,
        raw_text: str,
        raw_trim_start: int,
        raw_trim_end: int,
    ) -> FragmentTextProjectionReceipt:
        """Create and immediately verify the frozen projection from exact text."""

        projected_text = _project_raw_text(
            raw_text,
            raw_trim_start=raw_trim_start,
            raw_trim_end=raw_trim_end,
        )
        return cls(
            library_id=library_id,
            source_version_id=source_version_id,
            source_fragment_id=source_fragment_id,
            raw_text_sha256=sha256_hex(raw_text.encode("utf-8")),
            projected_text_sha256=sha256_hex(projected_text.encode("utf-8")),
            raw_character_count=len(raw_text),
            projected_character_count=len(projected_text),
            raw_trim_start=raw_trim_start,
            raw_trim_end=raw_trim_end,
        )

    def verify_texts(self, *, raw_text: str, projected_text: str) -> None:
        """Fail closed unless exact texts replay every recorded field."""

        expected = _project_raw_text(
            raw_text,
            raw_trim_start=self.raw_trim_start,
            raw_trim_end=self.raw_trim_end,
        )
        if projected_text != expected:
            raise CompatibilityContractError(
                "projected text does not replay the frozen trim/NFC profile"
            )
        if len(raw_text) != self.raw_character_count:
            raise CompatibilityContractError("raw character count does not match the receipt")
        if len(projected_text) != self.projected_character_count:
            raise CompatibilityContractError("projected character count does not match the receipt")
        if sha256_hex(raw_text.encode("utf-8")) != self.raw_text_sha256:
            raise CompatibilityContractError("raw text hash does not match the receipt")
        if sha256_hex(projected_text.encode("utf-8")) != self.projected_text_sha256:
            raise CompatibilityContractError("projected text hash does not match the receipt")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            **self.model_dump(mode="json"),
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


class ExternalReferenceMapEntry(_FrozenContract):
    """One external identifier and original alias bound to a canonical object."""

    library_id: str
    external_id: str
    original_alias: str
    canonical_kind: Literal["collection", "source", "source_version", "fragment"]
    canonical_id: str

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("external_id")
    @classmethod
    def _external_id(cls, value: str) -> str:
        return _require_text(value, "external_id", maximum=512)

    @field_validator("original_alias")
    @classmethod
    def _original_alias(cls, value: str) -> str:
        return _require_text(value, "original_alias", maximum=512)

    @model_validator(mode="after")
    def _canonical_kind_matches_id(self) -> Self:
        _require_id(self.canonical_id, self.canonical_kind)
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class ExternalReferenceMap(_FrozenContract):
    """Exact, single-Library external-reference mapping.

    The expected external identifier set is part of the hash domain.  Missing
    or additional entries therefore fail construction, while lookup always
    requires an explicit Library and rejects unknown references.
    """

    SCHEMA: ClassVar[str] = "dithyramba.external_reference_map/1.0"

    library_id: str
    namespace: str
    expected_external_ids: tuple[str, ...]
    entries: tuple[ExternalReferenceMapEntry, ...]

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("namespace")
    @classmethod
    def _namespace(cls, value: str) -> str:
        if type(value) is not str or _KEY_PATTERN.fullmatch(value) is None:
            raise CompatibilityContractError("namespace must be stable lowercase snake_case")
        return value

    @field_validator("expected_external_ids")
    @classmethod
    def _expected_external_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > _MAX_FRAGMENTS:
            raise CompatibilityContractError("external reference coverage exceeds the bound")
        normalized = tuple(
            _require_text(item, "expected_external_ids", maximum=512) for item in value
        )
        return _unique_sorted_texts(normalized, "expected_external_ids")

    @field_validator("entries")
    @classmethod
    def _entries(
        cls, value: tuple[ExternalReferenceMapEntry, ...]
    ) -> tuple[ExternalReferenceMapEntry, ...]:
        if len(value) > _MAX_FRAGMENTS:
            raise CompatibilityContractError("external reference map exceeds the bound")
        if any(type(item) is not ExternalReferenceMapEntry for item in value):
            raise CompatibilityContractError(
                "external reference map requires exact ExternalReferenceMapEntry values"
            )
        external_ids = tuple(item.external_id for item in value)
        if len(set(external_ids)) != len(external_ids):
            raise CompatibilityContractError("external identifiers must be unique")
        canonical_targets = tuple((item.canonical_kind, item.canonical_id) for item in value)
        if len(set(canonical_targets)) != len(canonical_targets):
            raise CompatibilityContractError(
                "canonical targets must be unique; undeclared many-to-one mappings are forbidden"
            )
        alias_keys = tuple(_alias_key(item.original_alias) for item in value)
        if len(set(alias_keys)) != len(alias_keys):
            raise CompatibilityContractError(
                "normalized original aliases collide inside the reference map"
            )
        return tuple(sorted(value, key=_external_entry_sort_key))

    @model_validator(mode="after")
    def _closed_scope_and_coverage(self) -> Self:
        if any(item.library_id != self.library_id for item in self.entries):
            raise CompatibilityContractError(
                "external reference entries cannot cross the map Library"
            )
        actual_ids = tuple(item.external_id for item in self.entries)
        if set(actual_ids) != set(self.expected_external_ids):
            missing = sorted(set(self.expected_external_ids).difference(actual_ids))
            extra = sorted(set(actual_ids).difference(self.expected_external_ids))
            raise CompatibilityContractError(
                f"external reference coverage mismatch; missing={missing!r}, extra={extra!r}"
            )
        return self

    def resolve(self, external_id: str, *, library_id: str) -> ExternalReferenceMapEntry:
        """Resolve one exact external ID without implicit cross-Library access."""

        if library_id != self.library_id:
            raise CompatibilityContractError("cross-Library external reference lookup is forbidden")
        canonical_external_id = _require_text(external_id, "external_id", maximum=512)
        for item in self.entries:
            if item.external_id == canonical_external_id:
                return item
        raise CompatibilityContractError("unknown external reference")

    def resolve_alias(self, alias: str, *, library_id: str) -> ExternalReferenceMapEntry:
        """Resolve one original alias under the same explicit Library boundary."""

        if library_id != self.library_id:
            raise CompatibilityContractError("cross-Library external alias lookup is forbidden")
        key = _alias_key(_require_text(alias, "original_alias", maximum=512))
        for item in self.entries:
            if _alias_key(item.original_alias) == key:
                return item
        raise CompatibilityContractError("unknown external alias")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "namespace": self.namespace,
            "expected_external_ids": list(self.expected_external_ids),
            "entries": [item.semantic_payload() for item in self.entries],
        }

    @property
    def map_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


class ProofMetadataEntry(_FrozenContract):
    """Explicit proof role for exactly one permitted SourceFragment."""

    library_id: str
    source_id: str
    source_version_id: str
    source_fragment_id: str
    projection_receipt_hash: str
    external_reference_id: str | None = None
    fragment_kind: str
    source_kind: str
    source_family: str
    authority: str
    independence_group: str
    evidence_tags: tuple[str, ...] = ()
    anchor_keys: tuple[str, ...] = ()
    direction_keys: tuple[str, ...] = ()
    body_proof_eligible: bool

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("source_id")
    @classmethod
    def _source_id(cls, value: str) -> str:
        return _require_id(value, "source")

    @field_validator("source_version_id")
    @classmethod
    def _source_version_id(cls, value: str) -> str:
        return _require_id(value, "source_version")

    @field_validator("source_fragment_id")
    @classmethod
    def _source_fragment_id(cls, value: str) -> str:
        return _require_id(value, "fragment")

    @field_validator("projection_receipt_hash")
    @classmethod
    def _projection_hash(cls, value: str) -> str:
        return _require_hash(value, "projection_receipt_hash")

    @field_validator("external_reference_id")
    @classmethod
    def _external_reference_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_text(value, "external_reference_id", maximum=512)

    @field_validator(
        "fragment_kind",
        "source_kind",
        "source_family",
        "authority",
        "independence_group",
    )
    @classmethod
    def _role_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name, maximum=256)

    @field_validator("evidence_tags", "anchor_keys", "direction_keys")
    @classmethod
    def _canonical_metadata_tuple(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        if len(value) > 128:
            raise CompatibilityContractError(f"{info.field_name} exceeds the bound")
        normalized = tuple(_require_text(item, info.field_name, maximum=128) for item in value)
        return _unique_sorted_texts(normalized, info.field_name)

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class ProofMetadataManifest(_FrozenContract):
    """Exact content-addressed proof sidecar for one permitted corpus.

    ``expected_fragment_ids`` is the frozen coverage oracle.  Every expected
    fragment must have exactly one text projection and one proof entry; extra,
    missing, duplicate, cross-Library, unknown-reference and hash-mismatched
    objects all fail construction.
    """

    SCHEMA: ClassVar[str] = "dithyramba.proof_metadata_manifest/1.0"

    library_id: str
    retrieval_corpus_hash: str
    expected_fragment_ids: tuple[str, ...]
    text_projections: tuple[FragmentTextProjectionReceipt, ...]
    external_references: ExternalReferenceMap
    entries: tuple[ProofMetadataEntry, ...]

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("retrieval_corpus_hash")
    @classmethod
    def _corpus_hash(cls, value: str) -> str:
        return _require_hash(value, "retrieval_corpus_hash")

    @field_validator("expected_fragment_ids")
    @classmethod
    def _expected_fragment_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > _MAX_FRAGMENTS:
            raise CompatibilityContractError("proof manifest fragment coverage exceeds the bound")
        canonical = tuple(_require_id(item, "fragment") for item in value)
        return _unique_sorted_texts(canonical, "expected_fragment_ids")

    @field_validator("text_projections")
    @classmethod
    def _text_projections(
        cls, value: tuple[FragmentTextProjectionReceipt, ...]
    ) -> tuple[FragmentTextProjectionReceipt, ...]:
        if len(value) > _MAX_FRAGMENTS:
            raise CompatibilityContractError("text projection coverage exceeds the bound")
        if any(type(item) is not FragmentTextProjectionReceipt for item in value):
            raise CompatibilityContractError(
                "text_projections require exact FragmentTextProjectionReceipt values"
            )
        fragment_ids = tuple(item.source_fragment_id for item in value)
        if len(set(fragment_ids)) != len(fragment_ids):
            raise CompatibilityContractError("text projections contain duplicate fragments")
        hashes = tuple(item.receipt_hash for item in value)
        if len(set(hashes)) != len(hashes):
            raise CompatibilityContractError("text projection receipt hashes must be unique")
        return tuple(sorted(value, key=lambda item: item.source_fragment_id.encode("ascii")))

    @field_validator("external_references")
    @classmethod
    def _external_references(cls, value: ExternalReferenceMap) -> ExternalReferenceMap:
        if type(value) is not ExternalReferenceMap:
            raise CompatibilityContractError(
                "external_references must be an exact ExternalReferenceMap"
            )
        return value

    @field_validator("entries")
    @classmethod
    def _entries(cls, value: tuple[ProofMetadataEntry, ...]) -> tuple[ProofMetadataEntry, ...]:
        if len(value) > _MAX_FRAGMENTS:
            raise CompatibilityContractError("proof metadata coverage exceeds the bound")
        if any(type(item) is not ProofMetadataEntry for item in value):
            raise CompatibilityContractError("entries require exact ProofMetadataEntry values")
        fragment_ids = tuple(item.source_fragment_id for item in value)
        if len(set(fragment_ids)) != len(fragment_ids):
            raise CompatibilityContractError("proof metadata contains duplicate fragments")
        return tuple(sorted(value, key=lambda item: item.source_fragment_id.encode("ascii")))

    @model_validator(mode="after")
    def _closed_manifest(self) -> Self:
        expected = set(self.expected_fragment_ids)
        projection_ids = {item.source_fragment_id for item in self.text_projections}
        entry_ids = {item.source_fragment_id for item in self.entries}
        _require_exact_coverage(expected, projection_ids, "text projection")
        _require_exact_coverage(expected, entry_ids, "proof metadata")

        if self.external_references.library_id != self.library_id:
            raise CompatibilityContractError(
                "external reference map cannot cross the proof manifest Library"
            )
        if any(item.library_id != self.library_id for item in self.text_projections):
            raise CompatibilityContractError(
                "text projections cannot cross the proof manifest Library"
            )
        if any(item.library_id != self.library_id for item in self.entries):
            raise CompatibilityContractError(
                "proof metadata cannot cross the proof manifest Library"
            )

        projections = {item.source_fragment_id: item for item in self.text_projections}
        for item in self.entries:
            projection = projections[item.source_fragment_id]
            if projection.source_version_id != item.source_version_id:
                raise CompatibilityContractError(
                    "proof metadata source version differs from its text projection"
                )
            if projection.receipt_hash != item.projection_receipt_hash:
                raise CompatibilityContractError(
                    "proof metadata projection receipt hash does not match"
                )
            if item.external_reference_id is None:
                continue
            reference = self.external_references.resolve(
                item.external_reference_id,
                library_id=self.library_id,
            )
            if reference.canonical_kind != "fragment":
                raise CompatibilityContractError(
                    "proof metadata external reference must resolve to a fragment"
                )
            if reference.canonical_id != item.source_fragment_id:
                raise CompatibilityContractError(
                    "proof metadata external reference resolves to another fragment"
                )
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "retrieval_corpus_hash": self.retrieval_corpus_hash,
            "expected_fragment_ids": list(self.expected_fragment_ids),
            "text_projections": [item.semantic_payload() for item in self.text_projections],
            "external_references": self.external_references.semantic_payload(),
            "entries": [item.semantic_payload() for item in self.entries],
        }

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


def _project_raw_text(raw_text: str, *, raw_trim_start: int, raw_trim_end: int) -> str:
    if type(raw_text) is not str or not raw_text or "\x00" in raw_text:
        raise CompatibilityContractError("raw text must be non-empty text without NUL")
    if type(raw_trim_start) is not int or type(raw_trim_end) is not int:
        raise CompatibilityContractError("raw trim offsets must be integers")
    if not 0 <= raw_trim_start < raw_trim_end <= len(raw_text):
        raise CompatibilityContractError("raw trim offsets are outside the raw text")
    if raw_text[:raw_trim_start].strip() or raw_text[raw_trim_end:].strip():
        raise CompatibilityContractError("the frozen projection may trim only surrounding space")
    projected = unicodedata.normalize("NFC", raw_text[raw_trim_start:raw_trim_end])
    if not projected.strip() or projected != projected.strip() or "\x00" in projected:
        raise CompatibilityContractError(
            "projected text must be nonblank, unpadded NFC text without NUL"
        )
    return projected


def _require_hash(value: str, field: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise CompatibilityContractError(f"{field} must be lowercase SHA-256 hex")
    return value


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if (
        type(value) is not str
        or not value.startswith(expected)
        or _ID_PATTERN.fullmatch(value) is None
    ):
        raise CompatibilityContractError(f"identifier must use the {expected} prefix")
    return value


def _require_text(value: str, field: str, *, maximum: int) -> str:
    if type(value) is not str:
        raise CompatibilityContractError(f"{field} must be text")
    if (
        not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or "\x00" in value
        or len(value) > maximum
    ):
        raise CompatibilityContractError(f"{field} must be bounded, unpadded NFC text without NUL")
    return value


def _alias_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _unique_sorted_texts(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise CompatibilityContractError(f"{field} must contain unique values")
    return tuple(sorted(values, key=lambda item: item.encode("utf-8")))


def _external_entry_sort_key(
    item: ExternalReferenceMapEntry,
) -> tuple[bytes, bytes, bytes]:
    return (
        item.external_id.encode("utf-8"),
        item.original_alias.encode("utf-8"),
        item.canonical_id.encode("ascii"),
    )


def _require_exact_coverage(expected: set[str], actual: set[str], label: str) -> None:
    if actual == expected:
        return
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    raise CompatibilityContractError(
        f"{label} coverage mismatch; missing={missing!r}, extra={extra!r}"
    )
