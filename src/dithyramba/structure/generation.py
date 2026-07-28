"""Pure, deterministic Markdown StructureUnit proposal generation.

The builder consumes metadata only.  It never receives source text, and a
human-readable section label remains presentation metadata rather than
evidence.  Exact SourceFragment IDs are the only proposed members.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import groupby
from typing import ClassVar, Literal

from dithyramba.contracts import canonical_sha256_hex
from dithyramba.ingest import FragmentKind

from .errors import StructureContractError
from .models import (
    StructureGeneration,
    StructureGenerationRequest,
    StructureUnitKind,
    StructureUnitProposal,
)

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]+$")
_PROFILE_ID = "markdown_exact_heading_path"
_PROFILE_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class MarkdownStructureProfile:
    """Frozen v1 policy for natural Markdown structural boundaries."""

    SCHEMA: ClassVar[str] = "dithyramba.markdown_structure_profile/1.0"

    profile_id: Literal["markdown_exact_heading_path"] = "markdown_exact_heading_path"
    profile_version: Literal["1.0"] = "1.0"
    minimum_members: Literal[2] = 2
    maximum_members: int = 128

    def __post_init__(self) -> None:
        if self.profile_id != _PROFILE_ID:
            raise StructureContractError("unsupported Markdown structure profile_id")
        if self.profile_version != _PROFILE_VERSION:
            raise StructureContractError("unsupported Markdown structure profile_version")
        if type(self.minimum_members) is not int or self.minimum_members != 2:
            raise StructureContractError(
                "Markdown StructureUnits require exactly 2 minimum members"
            )
        if type(self.maximum_members) is not int or not 2 <= self.maximum_members <= 128:
            raise StructureContractError("maximum_members must be between 2 and 128")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "minimum_members": self.minimum_members,
            "maximum_members": self.maximum_members,
        }

    @property
    def profile_hash(self) -> str:
        return canonical_sha256_hex(self.payload())


@dataclass(frozen=True, slots=True)
class MarkdownFragmentDescriptor:
    """Text-free structural metadata for one exact Markdown SourceFragment."""

    source_id: str
    source_version_id: str
    source_fragment_id: str
    ordinal: int
    fragment_kind: FragmentKind
    heading_path: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.source_id, "source_", "source_id")
        _require_id(self.source_version_id, "source_version_", "source_version_id")
        _require_id(self.source_fragment_id, "fragment_", "source_fragment_id")
        if type(self.ordinal) is not int or not 0 <= self.ordinal <= 2_147_483_647:
            raise StructureContractError("ordinal must be a non-negative integer")
        if not isinstance(self.fragment_kind, FragmentKind):
            raise StructureContractError("fragment_kind must be a FragmentKind")
        if self.fragment_kind is FragmentKind.PAGE_TEXT:
            raise StructureContractError("Markdown descriptors cannot use page_text fragments")
        if type(self.heading_path) is not tuple or len(self.heading_path) > 64:
            raise StructureContractError("heading_path must be a tuple with at most 64 headings")
        for heading in self.heading_path:
            if (
                type(heading) is not str
                or heading != heading.strip()
                or not heading
                or len(heading) > 500
                or "\x00" in heading
                or unicodedata.normalize("NFC", heading) != heading
            ):
                raise StructureContractError(
                    "heading_path entries must be non-empty trimmed NFC text"
                )
        if self.fragment_kind is FragmentKind.HEADING and not self.heading_path:
            raise StructureContractError("Markdown heading fragments require a heading_path")

    def payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "source_fragment_id": self.source_fragment_id,
            "ordinal": self.ordinal,
            "fragment_kind": self.fragment_kind.value,
            "heading_path": list(self.heading_path),
        }


class StructureProposalOmissionReason(StrEnum):
    """Reasons why one complete natural Markdown unit was not proposed."""

    BELOW_MINIMUM_MEMBERS = "below_minimum_members"
    OVERSIZED_NATURAL_SECTION = "oversized_natural_section"


class StructureInputExclusionReason(StrEnum):
    """Why permitted metadata was deliberately not sent to the Markdown builder."""

    UNSUPPORTED_SOURCE_ADDRESS = "unsupported_source_address"
    FRAGMENT_LEVEL_EXCLUSION_PRESENT = "fragment_level_exclusion_present"
    NONCONTIGUOUS_PERMITTED_VERSION = "noncontiguous_permitted_version"


@dataclass(frozen=True, slots=True, order=True)
class StructureInputExclusion:
    """One permitted, text-free fragment excluded before proposal generation."""

    source_id: str
    source_version_id: str
    source_fragment_id: str
    ordinal: int
    fragment_kind: FragmentKind
    address_kind: str
    reason: StructureInputExclusionReason

    def __post_init__(self) -> None:
        _require_id(self.source_id, "source_", "source_id")
        _require_id(self.source_version_id, "source_version_", "source_version_id")
        _require_id(self.source_fragment_id, "fragment_", "source_fragment_id")
        if type(self.ordinal) is not int or not 0 <= self.ordinal <= 2_147_483_647:
            raise StructureContractError("ordinal must be a non-negative integer")
        if not isinstance(self.fragment_kind, FragmentKind):
            raise StructureContractError("fragment_kind must be a FragmentKind")
        if (
            type(self.address_kind) is not str
            or self.address_kind != self.address_kind.strip()
            or not self.address_kind
            or len(self.address_kind) > 64
            or "\x00" in self.address_kind
        ):
            raise StructureContractError("address_kind must be non-empty trimmed text")
        if not isinstance(self.reason, StructureInputExclusionReason):
            raise StructureContractError("reason must be a StructureInputExclusionReason")

    def payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "source_fragment_id": self.source_fragment_id,
            "ordinal": self.ordinal,
            "fragment_kind": self.fragment_kind.value,
            "address_kind": self.address_kind,
            "reason": self.reason.value,
        }


@dataclass(frozen=True, slots=True)
class StructureProposalOmission:
    """Explicit omission of one unsplit, naturally bounded Markdown unit."""

    reason: StructureProposalOmissionReason
    source_id: str
    source_version_id: str
    kind: StructureUnitKind
    heading_path: tuple[str, ...]
    source_fragment_ids: tuple[str, ...]
    first_ordinal: int
    last_ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.reason, StructureProposalOmissionReason):
            raise StructureContractError("reason must be a StructureProposalOmissionReason")
        _require_id(self.source_id, "source_", "source_id")
        _require_id(self.source_version_id, "source_version_", "source_version_id")
        if not isinstance(self.kind, StructureUnitKind):
            raise StructureContractError("kind must be a StructureUnitKind")
        if type(self.heading_path) is not tuple:
            raise StructureContractError("heading_path must be a tuple")
        if self.kind is StructureUnitKind.DOCUMENT and self.heading_path:
            raise StructureContractError("DOCUMENT omissions require an empty heading_path")
        if self.kind is StructureUnitKind.SECTION and not self.heading_path:
            raise StructureContractError("SECTION omissions require a heading_path")
        fragment_ids = _require_fragment_ids(self.source_fragment_ids)
        object.__setattr__(self, "source_fragment_ids", fragment_ids)
        if type(self.first_ordinal) is not int or self.first_ordinal < 0:
            raise StructureContractError("first_ordinal must be a non-negative integer")
        if type(self.last_ordinal) is not int or self.last_ordinal < self.first_ordinal:
            raise StructureContractError("last_ordinal must be at least first_ordinal")
        if self.last_ordinal - self.first_ordinal + 1 != len(fragment_ids):
            raise StructureContractError("omission ordinals must match its contiguous members")

    @property
    def member_count(self) -> int:
        return len(self.source_fragment_ids)

    @property
    def label(self) -> str | None:
        return self.heading_path[-1] if self.heading_path else None

    def payload(self) -> dict[str, object]:
        return {
            "reason": self.reason.value,
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "kind": self.kind.value,
            "label": self.label,
            "heading_path": list(self.heading_path),
            "source_fragment_ids": list(self.source_fragment_ids),
            "first_ordinal": self.first_ordinal,
            "last_ordinal": self.last_ordinal,
            "member_count": self.member_count,
        }


@dataclass(frozen=True, slots=True)
class MarkdownStructureProposalBuild:
    """Canonical proposals and explicit natural-boundary omissions."""

    SCHEMA: ClassVar[str] = "dithyramba.markdown_structure_proposal_build/1.0"

    profile: MarkdownStructureProfile
    proposals: tuple[StructureUnitProposal, ...]
    omissions: tuple[StructureProposalOmission, ...]
    build_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile, MarkdownStructureProfile):
            raise StructureContractError("profile must be a MarkdownStructureProfile")
        if type(self.proposals) is not tuple or any(
            not isinstance(item, StructureUnitProposal) for item in self.proposals
        ):
            raise StructureContractError("proposals must be StructureUnitProposal values")
        if type(self.omissions) is not tuple or any(
            not isinstance(item, StructureProposalOmission) for item in self.omissions
        ):
            raise StructureContractError("omissions must be StructureProposalOmission values")
        member_ids = tuple(
            fragment_id
            for proposal in self.proposals
            for fragment_id in proposal.source_fragment_ids
        ) + tuple(
            fragment_id
            for omission in self.omissions
            for fragment_id in omission.source_fragment_ids
        )
        if len(member_ids) != len(set(member_ids)):
            raise StructureContractError("proposal build cannot reuse a SourceFragment")
        object.__setattr__(self, "build_hash", canonical_sha256_hex(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "profile": self.profile.payload(),
            "proposals": [item.payload() for item in self.proposals],
            "omissions": [item.payload() for item in self.omissions],
        }


class StructureGenerationBuildStatus(StrEnum):
    """Terminal status of one metadata-only StructureUnit generation attempt."""

    PERSISTED = "persisted"
    NO_PROPOSALS = "no_proposals"


@dataclass(frozen=True, slots=True)
class MarkdownStructureGenerationResult:
    """Versioned binding of a proposal build to its optional persisted generation."""

    SCHEMA: ClassVar[str] = "dithyramba.markdown_structure_generation_result/1.0"

    request: StructureGenerationRequest
    build: MarkdownStructureProposalBuild
    generation: StructureGeneration | None
    exclusions: tuple[StructureInputExclusion, ...] = ()
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, StructureGenerationRequest):
            raise StructureContractError("request must be a StructureGenerationRequest")
        if not isinstance(self.build, MarkdownStructureProposalBuild):
            raise StructureContractError("build must be a MarkdownStructureProposalBuild")
        if (
            self.request.profile_id != self.build.profile.profile_id
            or self.request.profile_version != self.build.profile.profile_version
        ):
            raise StructureContractError("request and Markdown profile identities differ")
        if type(self.exclusions) is not tuple or any(
            not isinstance(item, StructureInputExclusion) for item in self.exclusions
        ):
            raise StructureContractError("exclusions must be StructureInputExclusion values")
        canonical_exclusions = tuple(sorted(self.exclusions))
        if canonical_exclusions != self.exclusions:
            raise StructureContractError("StructureInput exclusions must use canonical order")
        excluded_ids = tuple(item.source_fragment_id for item in self.exclusions)
        if len(excluded_ids) != len(set(excluded_ids)):
            raise StructureContractError("StructureInput exclusions must be unique")
        build_ids = tuple(
            fragment_id
            for proposal in self.build.proposals
            for fragment_id in proposal.source_fragment_ids
        ) + tuple(
            fragment_id
            for omission in self.build.omissions
            for fragment_id in omission.source_fragment_ids
        )
        if set(excluded_ids).intersection(build_ids):
            raise StructureContractError("excluded fragments cannot appear in the Markdown build")
        if self.build.proposals:
            if not isinstance(self.generation, StructureGeneration):
                raise StructureContractError("proposals require a persisted StructureGeneration")
            if (
                self.generation.library_id != self.request.scope.library_id
                or self.generation.corpus_snapshot_id != self.request.corpus_snapshot_id
                or self.generation.access_policy_id != self.request.access_policy_id
                or self.generation.scope_hash != self.request.scope_hash
                or self.generation.profile_id != self.request.profile_id
                or self.generation.profile_version != self.request.profile_version
                or len(self.generation.units) != len(self.build.proposals)
            ):
                raise StructureContractError(
                    "persisted StructureGeneration differs from its request or build"
                )
        elif self.generation is not None:
            raise StructureContractError("an empty proposal build cannot persist a generation")
        object.__setattr__(self, "result_hash", canonical_sha256_hex(self.payload()))

    @property
    def status(self) -> StructureGenerationBuildStatus:
        if self.generation is None:
            return StructureGenerationBuildStatus.NO_PROPOSALS
        return StructureGenerationBuildStatus.PERSISTED

    def payload(self) -> dict[str, object]:
        generation_identity = (
            None
            if self.generation is None
            else {
                "generation_id": self.generation.generation_id,
                "generation_hash": self.generation.generation_hash,
            }
        )
        return {
            "schema": self.SCHEMA,
            "status": self.status.value,
            "request": self.request.payload(),
            "build_hash": self.build.build_hash,
            "build": self.build.payload(),
            "exclusions": [item.payload() for item in self.exclusions],
            "generation": generation_identity,
        }


DEFAULT_MARKDOWN_STRUCTURE_PROFILE = MarkdownStructureProfile()


def build_markdown_structure_proposals(
    fragments: tuple[MarkdownFragmentDescriptor, ...],
    *,
    profile: MarkdownStructureProfile | None = None,
) -> MarkdownStructureProposalBuild:
    """Build natural Markdown units in canonical SourceVersion/ordinal order.

    Each contiguous run with one exact ``heading_path`` remains whole.  A run
    above the profile limit is reported and never split into arbitrary windows.
    """

    resolved_profile = DEFAULT_MARKDOWN_STRUCTURE_PROFILE if profile is None else profile
    if not isinstance(resolved_profile, MarkdownStructureProfile):
        raise StructureContractError("profile must be a MarkdownStructureProfile")
    if type(fragments) is not tuple or not fragments:
        raise StructureContractError("fragments must be a non-empty tuple")
    if any(not isinstance(item, MarkdownFragmentDescriptor) for item in fragments):
        raise StructureContractError("fragments must contain MarkdownFragmentDescriptor values")

    _validate_input_identity(fragments)
    canonical = tuple(
        sorted(
            fragments,
            key=lambda item: (item.source_id, item.source_version_id, item.ordinal),
        )
    )
    proposals: list[StructureUnitProposal] = []
    omissions: list[StructureProposalOmission] = []

    def version_key(item: MarkdownFragmentDescriptor) -> tuple[str, str]:
        return item.source_id, item.source_version_id

    for _, version_items_iterator in groupby(canonical, key=version_key):
        version_items = tuple(version_items_iterator)
        _require_contiguous_version(version_items)
        for heading_path, natural_iterator in groupby(
            version_items, key=lambda item: item.heading_path
        ):
            natural = tuple(natural_iterator)
            kind = StructureUnitKind.DOCUMENT if not heading_path else StructureUnitKind.SECTION
            label = heading_path[-1] if heading_path else None
            fragment_ids = tuple(item.source_fragment_id for item in natural)
            if resolved_profile.minimum_members <= len(natural) <= resolved_profile.maximum_members:
                proposals.append(StructureUnitProposal(kind, fragment_ids, label))
                continue
            reason = (
                StructureProposalOmissionReason.BELOW_MINIMUM_MEMBERS
                if len(natural) < resolved_profile.minimum_members
                else StructureProposalOmissionReason.OVERSIZED_NATURAL_SECTION
            )
            omissions.append(
                StructureProposalOmission(
                    reason=reason,
                    source_id=natural[0].source_id,
                    source_version_id=natural[0].source_version_id,
                    kind=kind,
                    heading_path=heading_path,
                    source_fragment_ids=fragment_ids,
                    first_ordinal=natural[0].ordinal,
                    last_ordinal=natural[-1].ordinal,
                )
            )

    return MarkdownStructureProposalBuild(resolved_profile, tuple(proposals), tuple(omissions))


def _validate_input_identity(fragments: tuple[MarkdownFragmentDescriptor, ...]) -> None:
    fragment_ids: set[str] = set()
    ordinal_keys: set[tuple[str, int]] = set()
    version_sources: dict[str, str] = {}
    for fragment in fragments:
        if fragment.source_fragment_id in fragment_ids:
            raise StructureContractError("duplicate source_fragment_id in Markdown input")
        fragment_ids.add(fragment.source_fragment_id)
        ordinal_key = (fragment.source_version_id, fragment.ordinal)
        if ordinal_key in ordinal_keys:
            raise StructureContractError("duplicate ordinal within one SourceVersion")
        ordinal_keys.add(ordinal_key)
        prior_source = version_sources.setdefault(fragment.source_version_id, fragment.source_id)
        if prior_source != fragment.source_id:
            raise StructureContractError("one SourceVersion cannot belong to multiple Sources")


def _require_contiguous_version(fragments: tuple[MarkdownFragmentDescriptor, ...]) -> None:
    ordinals = tuple(item.ordinal for item in fragments)
    expected = tuple(range(ordinals[0], ordinals[0] + len(ordinals)))
    if ordinals != expected:
        raise StructureContractError(
            "Markdown fragments must be ordinal-contiguous per SourceVersion"
        )


def _require_id(value: object, prefix: str, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or _ID_PATTERN.fullmatch(value) is None
    ):
        raise StructureContractError(f"{label} must use the {prefix} prefix")
    return value


def _require_fragment_ids(values: object) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise StructureContractError("source_fragment_ids must be a non-empty tuple")
    result = tuple(_require_id(item, "fragment_", "source_fragment_id") for item in values)
    if len(result) != len(set(result)):
        raise StructureContractError("source_fragment_ids must be unique")
    return result
