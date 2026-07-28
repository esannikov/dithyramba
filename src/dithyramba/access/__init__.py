"""Policy-safe, content-free access compilation for Dithyramba VS0."""

from .compiler import compile_access, verify_permitted_token
from .errors import (
    AccessContractError,
    InvalidAccessPolicyError,
    InvalidAccessScopeError,
    PermittedTokenError,
    PermittedTokenMismatchError,
    PurposeNotAllowedError,
)
from .models import (
    AccessPolicySnapshot,
    CollectionRule,
    CompiledAccess,
    MembershipState,
    PermittedManifest,
    PermittedManifestItem,
    PermittedToken,
    PolicyEffect,
    PublicAccessResult,
    QueryExclusions,
    RequestScope,
    SnapshotCandidate,
    SnapshotMembership,
    SourceRule,
)

__all__ = [
    "AccessContractError",
    "AccessPolicySnapshot",
    "CollectionRule",
    "CompiledAccess",
    "InvalidAccessPolicyError",
    "InvalidAccessScopeError",
    "MembershipState",
    "PermittedManifest",
    "PermittedManifestItem",
    "PermittedToken",
    "PermittedTokenError",
    "PermittedTokenMismatchError",
    "PolicyEffect",
    "PublicAccessResult",
    "PurposeNotAllowedError",
    "QueryExclusions",
    "RequestScope",
    "SnapshotCandidate",
    "SnapshotMembership",
    "SourceRule",
    "compile_access",
    "verify_permitted_token",
]
