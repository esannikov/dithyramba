"""Pure policy compilation and exact permitted-token verification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from dithyramba.contracts import canonical_sha256_hex

from .errors import (
    InvalidAccessScopeError,
    PermittedTokenError,
    PermittedTokenMismatchError,
    PurposeNotAllowedError,
)
from .models import (
    AccessPolicySnapshot,
    CompiledAccess,
    MembershipState,
    PermittedManifest,
    PermittedManifestItem,
    PermittedToken,
    PolicyEffect,
    PublicAccessResult,
    RequestScope,
    SnapshotCandidate,
)

_TOKEN_COMPONENTS: Final[tuple[str, ...]] = (
    "library_id",
    "snapshot_hash",
    "policy_hash",
    "exclusion_hash",
    "permitted_set_hash",
)


def compile_access(
    *,
    policy: AccessPolicySnapshot,
    scope: RequestScope,
    candidates: Iterable[SnapshotCandidate],
) -> CompiledAccess:
    """Compile content-free snapshot metadata into a permitted capability.

    Unlisted Collections are denied.  Any relevant Collection deny, non-active
    membership, explicit query exclusion, or Source deny removes the fragment
    before a repository can fetch content.  Only policy denials affect the
    public presence bit; denied identities and counts are never accumulated.
    """

    if policy.library_id != scope.library_id:
        raise InvalidAccessScopeError("policy and request belong to different Libraries")
    if scope.purpose not in policy.allowed_purposes:
        raise PurposeNotAllowedError("request purpose is not authorized by the policy")

    collection_effects = {rule.collection_id: rule.effect for rule in policy.collection_rules}
    source_effects = {rule.source_id: rule.effect for rule in policy.source_rules}
    requested_collections = frozenset(scope.collection_ids)
    excluded_sources = frozenset(scope.exclusions.source_ids)
    excluded_families = frozenset(scope.exclusions.source_family_ids)
    excluded_fragments = frozenset(scope.exclusions.source_fragment_ids)

    permitted_by_fragment: dict[str, PermittedManifestItem] = {}
    policy_omission_present = False

    for candidate in candidates:
        if not isinstance(candidate, SnapshotCandidate):
            raise InvalidAccessScopeError("candidates must contain SnapshotCandidate values")

        relevant_memberships = tuple(
            membership
            for membership in candidate.memberships
            if membership.collection_id in requested_collections
        )
        if not relevant_memberships:
            continue

        # Holdout and explicit exclusions are evaluated using metadata only.
        if any(
            membership.state is not MembershipState.ACTIVE for membership in relevant_memberships
        ):
            continue
        if candidate.source_id in excluded_sources:
            continue
        if (
            candidate.source_family_id is not None
            and candidate.source_family_id in excluded_families
        ):
            continue
        if candidate.source_fragment_id in excluded_fragments:
            continue

        effects = tuple(
            collection_effects.get(membership.collection_id) for membership in relevant_memberships
        )
        # None means unlisted and therefore denied.  A single denied/unlisted
        # relevant membership wins over any other allowed membership.
        if any(effect is not PolicyEffect.ALLOW for effect in effects):
            policy_omission_present = True
            continue
        if source_effects.get(candidate.source_id) is PolicyEffect.DENY:
            policy_omission_present = True
            continue

        item = PermittedManifestItem(
            source_fragment_id=candidate.source_fragment_id,
            source_version_id=candidate.source_version_id,
            source_id=candidate.source_id,
            source_family_id=candidate.source_family_id,
            collection_ids=tuple(membership.collection_id for membership in relevant_memberships),
        )
        previous = permitted_by_fragment.setdefault(item.source_fragment_id, item)
        if previous != item:
            raise InvalidAccessScopeError(
                "one fragment ID has conflicting snapshot candidate metadata"
            )

    manifest = PermittedManifest(
        library_id=scope.library_id,
        snapshot_hash=scope.snapshot_hash,
        items=tuple(permitted_by_fragment.values()),
    )
    token = PermittedToken(
        library_id=scope.library_id,
        snapshot_hash=scope.snapshot_hash,
        policy_hash=policy.policy_hash,
        exclusion_hash=scope.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
    )
    return CompiledAccess(
        manifest=manifest,
        token=token,
        public_result=PublicAccessResult(
            policy_omission_present=policy_omission_present,
        ),
    )


def verify_permitted_token(
    token: PermittedToken,
    *,
    library_id: str,
    snapshot_hash: str,
    policy_hash: str,
    exclusion_hash: str,
    permitted_set_hash: str,
) -> None:
    """Fail closed unless ``token`` matches every expected tuple component."""

    if not isinstance(token, PermittedToken):
        raise PermittedTokenError("repository access requires a PermittedToken")

    expected = {
        "library_id": library_id,
        "snapshot_hash": snapshot_hash,
        "policy_hash": policy_hash,
        "exclusion_hash": exclusion_hash,
        "permitted_set_hash": permitted_set_hash,
    }
    mismatches = tuple(name for name in _TOKEN_COMPONENTS if getattr(token, name) != expected[name])
    if mismatches:
        # Component names aid diagnosis while deliberately avoiding values that
        # could disclose identifiers from either security domain.
        raise PermittedTokenMismatchError(
            "permitted token tuple mismatch: " + ", ".join(mismatches)
        )

    expected_token_hash = canonical_sha256_hex(token.binding_payload())
    if token.token_hash != expected_token_hash:
        raise PermittedTokenError("permitted token integrity check failed")
