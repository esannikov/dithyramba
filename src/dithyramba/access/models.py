"""Immutable, text-free contracts for Dithyramba's access boundary.

The compiler deliberately operates on identifiers and membership metadata.  It
never receives fragment text, paths, titles, retrieval scores, or other corpus
content.  A repository may fetch text only after verifying a ``PermittedToken``
against the exact five-component tuple frozen below.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from dithyramba.contracts import canonical_sha256_hex

from .errors import InvalidAccessPolicyError, InvalidAccessScopeError, PermittedTokenError

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PURPOSE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class PolicyEffect(StrEnum):
    """An explicit policy rule effect."""

    ALLOW = "allow"
    DENY = "deny"


class MembershipState(StrEnum):
    """The state captured for a source in an immutable corpus snapshot."""

    ACTIVE = "active"
    EXCLUDED = "excluded"
    HOLDOUT = "holdout"


@dataclass(frozen=True, slots=True, order=True)
class CollectionRule:
    """Policy effect for one Collection."""

    collection_id: str
    effect: PolicyEffect

    def __post_init__(self) -> None:
        _require_id(self.collection_id, "collection")
        _require_enum(self.effect, PolicyEffect, "collection rule effect")


@dataclass(frozen=True, slots=True, order=True)
class SourceRule:
    """Policy effect for one Source; deny always overrides Collection allow."""

    source_id: str
    effect: PolicyEffect

    def __post_init__(self) -> None:
        _require_id(self.source_id, "source")
        _require_enum(self.effect, PolicyEffect, "source rule effect")


@dataclass(frozen=True, slots=True)
class AccessPolicySnapshot:
    """A canonical immutable snapshot of the policy inputs used for one query."""

    SCHEMA: ClassVar[str] = "dithyramba.access_policy_snapshot/1.0"

    access_policy_id: str
    library_id: str
    allowed_purposes: tuple[str, ...]
    collection_rules: tuple[CollectionRule, ...]
    source_rules: tuple[SourceRule, ...] = ()
    allow_export: bool = False
    allow_external_provider: bool = False
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_id(self.access_policy_id, "policy")
        _require_id(self.library_id, "library")
        purposes = _sorted_unique_strings(
            self.allowed_purposes,
            label="allowed purposes",
            pattern=_PURPOSE_PATTERN,
        )
        if not purposes:
            raise InvalidAccessPolicyError("an access policy must allow at least one purpose")

        collection_rules = _canonical_rules(
            self.collection_rules,
            CollectionRule,
            key_name="collection_id",
            label="collection rules",
        )
        source_rules = _canonical_rules(
            self.source_rules,
            SourceRule,
            key_name="source_id",
            label="source rules",
        )
        if type(self.allow_export) is not bool or type(self.allow_external_provider) is not bool:
            raise InvalidAccessPolicyError("policy capability flags must be booleans")

        object.__setattr__(self, "allowed_purposes", purposes)
        object.__setattr__(self, "collection_rules", collection_rules)
        object.__setattr__(self, "source_rules", source_rules)
        object.__setattr__(self, "policy_hash", canonical_sha256_hex(self.semantic_payload()))

    def semantic_payload(self) -> dict[str, Any]:
        """Return the stable semantic payload; random policy identity is excluded."""

        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "allowed_purposes": list(self.allowed_purposes),
            "collection_rules": [
                {"collection_id": rule.collection_id, "effect": rule.effect.value}
                for rule in self.collection_rules
            ],
            "source_rules": [
                {"source_id": rule.source_id, "effect": rule.effect.value}
                for rule in self.source_rules
            ],
            "allow_export": self.allow_export,
            "allow_external_provider": self.allow_external_provider,
        }


@dataclass(frozen=True, slots=True)
class QueryExclusions:
    """Explicit caller exclusions applied before any fragment content read."""

    source_ids: tuple[str, ...] = ()
    source_family_ids: tuple[str, ...] = ()
    source_fragment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_ids",
            _sorted_unique_ids(self.source_ids, "source", "excluded Source IDs"),
        )
        object.__setattr__(
            self,
            "source_family_ids",
            _sorted_unique_ids(
                self.source_family_ids,
                "family",
                "excluded SourceFamily IDs",
            ),
        )
        object.__setattr__(
            self,
            "source_fragment_ids",
            _sorted_unique_ids(
                self.source_fragment_ids,
                "fragment",
                "excluded SourceFragment IDs",
            ),
        )

    def payload(self) -> dict[str, list[str]]:
        """Return a canonical explicit-exclusion payload."""

        return {
            "source_ids": list(self.source_ids),
            "source_family_ids": list(self.source_family_ids),
            "source_fragment_ids": list(self.source_fragment_ids),
        }


@dataclass(frozen=True, slots=True)
class RequestScope:
    """The request fields that can change the permitted set."""

    library_id: str
    snapshot_hash: str
    purpose: str
    collection_ids: tuple[str, ...]
    exclusions: QueryExclusions = field(default_factory=QueryExclusions)

    def __post_init__(self) -> None:
        _require_id(self.library_id, "library")
        _require_hash(self.snapshot_hash, "snapshot_hash", InvalidAccessScopeError)
        if type(self.purpose) is not str or _PURPOSE_PATTERN.fullmatch(self.purpose) is None:
            raise InvalidAccessScopeError("purpose must match [a-z][a-z0-9_-]{0,63}")
        collection_ids = _sorted_unique_ids(
            self.collection_ids,
            "collection",
            "requested Collection IDs",
        )
        if not collection_ids:
            raise InvalidAccessScopeError("a request must select at least one Collection")
        if len(collection_ids) > 16:
            raise InvalidAccessScopeError("a request may select at most 16 Collections")
        if not isinstance(self.exclusions, QueryExclusions):
            raise InvalidAccessScopeError("exclusions must be a QueryExclusions value")
        object.__setattr__(self, "collection_ids", collection_ids)

    @property
    def exclusion_hash(self) -> str:
        """Hash every scope filter that affects permission compilation.

        The frozen schema calls this ``exclusion_hash``.  Binding purpose and
        selected Collections as well as explicit exclusions prevents a token
        compiled for one legal request scope from being replayed in another.
        """

        return canonical_sha256_hex(
            {
                "schema": "dithyramba.query_access_scope/1.0",
                "purpose": self.purpose,
                "collection_ids": list(self.collection_ids),
                "exclusions": self.exclusions.payload(),
            }
        )


@dataclass(frozen=True, slots=True, order=True)
class SnapshotMembership:
    """Content-free Collection membership metadata captured by a snapshot."""

    collection_id: str
    state: MembershipState

    def __post_init__(self) -> None:
        _require_id(self.collection_id, "collection")
        _require_enum(self.state, MembershipState, "membership state")


@dataclass(frozen=True, slots=True)
class SnapshotCandidate:
    """Content-free candidate metadata supplied to the policy compiler.

    Fragment granularity is intentional: fragment exclusions can therefore be
    applied before the repository is authorized to fetch ``text``.
    """

    source_fragment_id: str
    source_version_id: str
    source_id: str
    source_family_id: str | None
    memberships: tuple[SnapshotMembership, ...]

    def __post_init__(self) -> None:
        _require_id(self.source_fragment_id, "fragment")
        _require_id(self.source_version_id, "source_version")
        _require_id(self.source_id, "source")
        if self.source_family_id is not None:
            _require_id(self.source_family_id, "family")
        memberships = _canonical_memberships(self.memberships)
        if not memberships:
            raise InvalidAccessScopeError("a snapshot candidate must have a membership")
        object.__setattr__(self, "memberships", memberships)


@dataclass(frozen=True, slots=True, order=True)
class PermittedManifestItem:
    """One fragment that a token-gated repository may read."""

    source_fragment_id: str
    source_version_id: str
    source_id: str
    source_family_id: str | None
    collection_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.source_fragment_id, "fragment")
        _require_id(self.source_version_id, "source_version")
        _require_id(self.source_id, "source")
        if self.source_family_id is not None:
            _require_id(self.source_family_id, "family")
        collection_ids = _sorted_unique_ids(
            self.collection_ids,
            "collection",
            "permitted Collection IDs",
        )
        if not collection_ids:
            raise InvalidAccessScopeError("a permitted item must retain an allowed Collection")
        object.__setattr__(self, "collection_ids", collection_ids)

    def payload(self) -> dict[str, Any]:
        """Return the stable identifier-only manifest entry."""

        return {
            "source_fragment_id": self.source_fragment_id,
            "source_version_id": self.source_version_id,
            "source_id": self.source_id,
            "source_family_id": self.source_family_id,
            "collection_ids": list(self.collection_ids),
        }


@dataclass(frozen=True, slots=True)
class PermittedManifest:
    """The complete identifier-only set that may cross the content boundary."""

    SCHEMA: ClassVar[str] = "dithyramba.permitted_manifest/1.0"

    library_id: str
    snapshot_hash: str
    items: tuple[PermittedManifestItem, ...]
    permitted_set_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_id(self.library_id, "library")
        _require_hash(self.snapshot_hash, "snapshot_hash", InvalidAccessScopeError)
        items = _canonical_manifest_items(self.items)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "permitted_set_hash", canonical_sha256_hex(self.payload()))

    def payload(self) -> dict[str, Any]:
        """Return the stable manifest used to derive ``permitted_set_hash``."""

        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "snapshot_hash": self.snapshot_hash,
            "items": [item.payload() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class PermittedToken:
    """Capability bound to the exact policy-safe retrieval tuple."""

    SCHEMA: ClassVar[str] = "dithyramba.permitted_token/1.0"

    library_id: str
    snapshot_hash: str
    policy_hash: str
    exclusion_hash: str
    permitted_set_hash: str
    token_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_id(self.library_id, "library")
        for name in (
            "snapshot_hash",
            "policy_hash",
            "exclusion_hash",
            "permitted_set_hash",
        ):
            _require_hash(getattr(self, name), name, PermittedTokenError)
        object.__setattr__(self, "token_hash", canonical_sha256_hex(self.binding_payload()))

    def binding_payload(self) -> dict[str, str]:
        """Return the complete five-component binding plus schema marker."""

        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "snapshot_hash": self.snapshot_hash,
            "policy_hash": self.policy_hash,
            "exclusion_hash": self.exclusion_hash,
            "permitted_set_hash": self.permitted_set_hash,
        }

    def public_payload(self) -> dict[str, str]:
        """Return a safe token representation containing no corpus identifiers."""

        return {**self.binding_payload(), "token_hash": self.token_hash}


@dataclass(frozen=True, slots=True)
class PublicAccessResult:
    """The complete public policy-omission disclosure for VS0."""

    policy_omission_present: bool

    def __post_init__(self) -> None:
        if type(self.policy_omission_present) is not bool:
            raise InvalidAccessScopeError("policy_omission_present must be a boolean")

    def payload(self) -> dict[str, bool]:
        """Expose presence only: policy-denied counts and identities never exist here."""

        return {"policy_omission_present": self.policy_omission_present}


@dataclass(frozen=True, slots=True)
class CompiledAccess:
    """Internal compiler output and its deliberately minimal public disclosure."""

    manifest: PermittedManifest
    token: PermittedToken
    public_result: PublicAccessResult

    def __post_init__(self) -> None:
        if self.manifest.library_id != self.token.library_id:
            raise PermittedTokenError("manifest and token Library do not match")
        if self.manifest.snapshot_hash != self.token.snapshot_hash:
            raise PermittedTokenError("manifest and token snapshot do not match")
        if self.manifest.permitted_set_hash != self.token.permitted_set_hash:
            raise PermittedTokenError("manifest and token permitted set do not match")


def _require_id(value: object, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected) or len(value) <= len(expected):
        raise InvalidAccessScopeError(f"identifier must have prefix {expected!r}")
    if _ID_PATTERN.fullmatch(value) is None:
        raise InvalidAccessScopeError("identifiers must contain lowercase ASCII letters/digits/_")
    return value


def _require_hash(
    value: object,
    name: str,
    error_type: type[InvalidAccessScopeError] | type[PermittedTokenError],
) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise error_type(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return value


def _require_enum(value: object, expected_type: type[StrEnum], label: str) -> None:
    if not isinstance(value, expected_type):
        raise InvalidAccessScopeError(f"{label} must be a {expected_type.__name__}")


def _as_tuple(values: Iterable[Any], label: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAccessScopeError(f"{label} must be an iterable of values")
    return tuple(values)


def _sorted_unique_strings(
    values: Iterable[Any],
    *,
    label: str,
    pattern: re.Pattern[str],
) -> tuple[str, ...]:
    result = _as_tuple(values, label)
    if any(type(value) is not str or pattern.fullmatch(value) is None for value in result):
        raise InvalidAccessPolicyError(f"{label} contain an invalid value")
    if len(set(result)) != len(result):
        raise InvalidAccessPolicyError(f"{label} must be unique")
    return tuple(sorted(result))


def _sorted_unique_ids(values: Iterable[Any], prefix: str, label: str) -> tuple[str, ...]:
    result = _as_tuple(values, label)
    for value in result:
        _require_id(value, prefix)
    if len(set(result)) != len(result):
        raise InvalidAccessScopeError(f"{label} must be unique")
    return tuple(sorted(result))


def _canonical_rules(
    values: Iterable[Any],
    expected_type: type[CollectionRule] | type[SourceRule],
    *,
    key_name: str,
    label: str,
) -> tuple[Any, ...]:
    rules = _as_tuple(values, label)
    if any(not isinstance(rule, expected_type) for rule in rules):
        raise InvalidAccessPolicyError(f"{label} contain an invalid rule type")
    identifiers = [getattr(rule, key_name) for rule in rules]
    if len(set(identifiers)) != len(identifiers):
        raise InvalidAccessPolicyError(f"{label} may contain only one rule per identifier")
    return tuple(sorted(rules, key=lambda rule: getattr(rule, key_name)))


def _canonical_memberships(
    values: Iterable[SnapshotMembership],
) -> tuple[SnapshotMembership, ...]:
    memberships = _as_tuple(values, "snapshot memberships")
    if any(not isinstance(item, SnapshotMembership) for item in memberships):
        raise InvalidAccessScopeError("snapshot memberships contain an invalid value")
    collection_ids = [item.collection_id for item in memberships]
    if len(set(collection_ids)) != len(collection_ids):
        raise InvalidAccessScopeError("a candidate may have one state per Collection")
    return tuple(sorted(memberships, key=lambda item: item.collection_id))


def _canonical_manifest_items(
    values: Iterable[PermittedManifestItem],
) -> tuple[PermittedManifestItem, ...]:
    items = _as_tuple(values, "permitted manifest items")
    if any(not isinstance(item, PermittedManifestItem) for item in items):
        raise InvalidAccessScopeError("permitted manifest contains an invalid item")
    fragment_ids = [item.source_fragment_id for item in items]
    if len(set(fragment_ids)) != len(fragment_ids):
        raise InvalidAccessScopeError("permitted manifest fragment IDs must be unique")
    return tuple(sorted(items, key=lambda item: item.source_fragment_id))
