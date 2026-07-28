from __future__ import annotations

import dataclasses
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    InvalidAccessScopeError,
    MembershipState,
    PermittedTokenMismatchError,
    PolicyEffect,
    PurposeNotAllowedError,
    QueryExclusions,
    RequestScope,
    SnapshotCandidate,
    SnapshotMembership,
    SourceRule,
    compile_access,
    verify_permitted_token,
)

LIBRARY = "library_alpha"
SNAPSHOT_HASH = "a" * 64
COLLECTION_A = "collection_alpha"
COLLECTION_B = "collection_beta"
COLLECTION_PRIVATE = "collection_private"


def _policy(
    *,
    collection_rules: tuple[CollectionRule, ...] | None = None,
    source_rules: tuple[SourceRule, ...] = (),
    allowed_purposes: tuple[str, ...] = ("research",),
) -> AccessPolicySnapshot:
    return AccessPolicySnapshot(
        access_policy_id="policy_research",
        library_id=LIBRARY,
        allowed_purposes=allowed_purposes,
        collection_rules=collection_rules
        or (
            CollectionRule(COLLECTION_A, PolicyEffect.ALLOW),
            CollectionRule(COLLECTION_B, PolicyEffect.ALLOW),
        ),
        source_rules=source_rules,
    )


def _scope(
    *,
    collections: tuple[str, ...] = (COLLECTION_A,),
    purpose: str = "research",
    exclusions: QueryExclusions | None = None,
) -> RequestScope:
    return RequestScope(
        library_id=LIBRARY,
        snapshot_hash=SNAPSHOT_HASH,
        purpose=purpose,
        collection_ids=collections,
        exclusions=exclusions or QueryExclusions(),
    )


def _candidate(
    suffix: str,
    *,
    collections: tuple[tuple[str, MembershipState], ...] = (
        (COLLECTION_A, MembershipState.ACTIVE),
    ),
    source_family_id: str | None = None,
) -> SnapshotCandidate:
    return SnapshotCandidate(
        source_fragment_id=f"fragment_{suffix}",
        source_version_id=f"source_version_{suffix}",
        source_id=f"source_{suffix}",
        source_family_id=source_family_id,
        memberships=tuple(
            SnapshotMembership(collection_id, state) for collection_id, state in collections
        ),
    )


def test_compiler_allows_active_listed_collections_and_multiple_collections() -> None:
    first = _candidate("first")
    second = _candidate(
        "second",
        collections=(
            (COLLECTION_A, MembershipState.ACTIVE),
            (COLLECTION_B, MembershipState.ACTIVE),
        ),
    )

    compiled = compile_access(
        policy=_policy(),
        scope=_scope(collections=(COLLECTION_B, COLLECTION_A)),
        candidates=(first, second),
    )

    assert tuple(item.source_fragment_id for item in compiled.manifest.items) == (
        "fragment_first",
        "fragment_second",
    )
    assert compiled.manifest.items[1].collection_ids == (COLLECTION_A, COLLECTION_B)
    assert compiled.public_result.payload() == {"policy_omission_present": False}


def test_unlisted_collection_is_denied_and_disclosed_only_as_presence() -> None:
    denied_id = "fragment_classified"
    denied = _candidate(
        "classified",
        collections=((COLLECTION_PRIVATE, MembershipState.ACTIVE),),
    )
    compiled = compile_access(
        policy=_policy(),
        scope=_scope(collections=(COLLECTION_A, COLLECTION_PRIVATE)),
        candidates=(_candidate("public"), denied),
    )

    assert tuple(item.source_fragment_id for item in compiled.manifest.items) == (
        "fragment_public",
    )
    public_json = json.dumps(compiled.public_result.payload(), sort_keys=True)
    token_json = json.dumps(compiled.token.public_payload(), sort_keys=True)
    assert compiled.public_result.payload() == {"policy_omission_present": True}
    assert set(compiled.public_result.payload()) == {"policy_omission_present"}
    assert denied_id not in public_json
    assert denied_id not in token_json
    assert "count" not in public_json
    assert "denied" not in token_json


def test_collection_deny_wins_over_allow_for_shared_fragment() -> None:
    shared = _candidate(
        "shared",
        collections=(
            (COLLECTION_A, MembershipState.ACTIVE),
            (COLLECTION_B, MembershipState.ACTIVE),
        ),
    )
    policy = _policy(
        collection_rules=(
            CollectionRule(COLLECTION_A, PolicyEffect.ALLOW),
            CollectionRule(COLLECTION_B, PolicyEffect.DENY),
        )
    )

    compiled = compile_access(
        policy=policy,
        scope=_scope(collections=(COLLECTION_A, COLLECTION_B)),
        candidates=(shared,),
    )

    assert compiled.manifest.items == ()
    assert compiled.public_result.policy_omission_present is True


def test_source_deny_wins_over_collection_allow() -> None:
    candidate = _candidate("source_denied")
    policy = _policy(
        source_rules=(SourceRule(candidate.source_id, PolicyEffect.DENY),),
    )

    compiled = compile_access(policy=policy, scope=_scope(), candidates=(candidate,))

    assert compiled.manifest.items == ()
    assert compiled.public_result.policy_omission_present is True


def test_holdout_excluded_and_query_exclusions_are_removed_without_content_read() -> None:
    holdout = _candidate(
        "holdout",
        collections=((COLLECTION_A, MembershipState.HOLDOUT),),
    )
    excluded_state = _candidate(
        "excluded_state",
        collections=((COLLECTION_A, MembershipState.EXCLUDED),),
    )
    excluded_source = _candidate("excluded_source")
    excluded_family = _candidate("excluded_family", source_family_id="family_secret")
    excluded_fragment = _candidate("excluded_fragment")
    permitted = _candidate("permitted")
    exclusions = QueryExclusions(
        source_ids=(excluded_source.source_id,),
        source_family_ids=("family_secret",),
        source_fragment_ids=(excluded_fragment.source_fragment_id,),
    )

    compiled = compile_access(
        policy=_policy(),
        scope=_scope(exclusions=exclusions),
        candidates=(
            holdout,
            excluded_state,
            excluded_source,
            excluded_family,
            excluded_fragment,
            permitted,
        ),
    )

    assert tuple(item.source_fragment_id for item in compiled.manifest.items) == (
        permitted.source_fragment_id,
    )
    assert compiled.public_result.policy_omission_present is False


def test_unauthorized_purpose_fails_closed_instead_of_returning_empty_set() -> None:
    with pytest.raises(PurposeNotAllowedError, match="not authorized"):
        compile_access(
            policy=_policy(),
            scope=_scope(purpose="commercial"),
            candidates=(_candidate("one"),),
        )


def test_policy_hash_is_semantic_and_stable_across_rule_order() -> None:
    first = _policy(
        collection_rules=(
            CollectionRule(COLLECTION_A, PolicyEffect.ALLOW),
            CollectionRule(COLLECTION_B, PolicyEffect.DENY),
        ),
        source_rules=(
            SourceRule("source_zeta", PolicyEffect.DENY),
            SourceRule("source_alpha", PolicyEffect.ALLOW),
        ),
        allowed_purposes=("teaching", "research"),
    )
    second = AccessPolicySnapshot(
        access_policy_id="policy_different_opaque_id",
        library_id=LIBRARY,
        collection_rules=tuple(reversed(first.collection_rules)),
        source_rules=tuple(reversed(first.source_rules)),
        allowed_purposes=tuple(reversed(first.allowed_purposes)),
    )

    assert first.policy_hash == second.policy_hash
    assert first.semantic_payload() == second.semantic_payload()


@given(order=st.permutations(("alpha", "beta", "gamma")))
def test_manifest_and_token_hashes_are_stable_for_every_candidate_order(
    order: list[str],
) -> None:
    candidates = {suffix: _candidate(suffix) for suffix in ("alpha", "beta", "gamma")}

    compiled = compile_access(
        policy=_policy(),
        scope=_scope(),
        candidates=tuple(candidates[suffix] for suffix in order),
    )
    canonical = compile_access(
        policy=_policy(),
        scope=_scope(),
        candidates=tuple(candidates.values()),
    )

    assert compiled.manifest.permitted_set_hash == canonical.manifest.permitted_set_hash
    assert compiled.token.token_hash == canonical.token.token_hash


def test_scope_hash_changes_with_purpose_collection_or_explicit_exclusion() -> None:
    base = _scope()
    teaching = _scope(purpose="teaching")
    collections = _scope(collections=(COLLECTION_A, COLLECTION_B))
    excluded = _scope(
        exclusions=QueryExclusions(source_fragment_ids=("fragment_one",)),
    )

    assert (
        len(
            {
                base.exclusion_hash,
                teaching.exclusion_hash,
                collections.exclusion_hash,
                excluded.exclusion_hash,
            }
        )
        == 4
    )


def test_verifier_accepts_exact_tuple() -> None:
    compiled = compile_access(
        policy=_policy(),
        scope=_scope(),
        candidates=(_candidate("one"),),
    )
    token = compiled.token

    verify_permitted_token(
        token,
        library_id=token.library_id,
        snapshot_hash=token.snapshot_hash,
        policy_hash=token.policy_hash,
        exclusion_hash=token.exclusion_hash,
        permitted_set_hash=token.permitted_set_hash,
    )


@pytest.mark.parametrize(
    ("component", "replacement"),
    [
        ("library_id", "library_other"),
        ("snapshot_hash", "b" * 64),
        ("policy_hash", "c" * 64),
        ("exclusion_hash", "d" * 64),
        ("permitted_set_hash", "e" * 64),
    ],
)
def test_verifier_rejects_reuse_across_each_tuple_component(
    component: str,
    replacement: str,
) -> None:
    compiled = compile_access(
        policy=_policy(),
        scope=_scope(),
        candidates=(_candidate("one"),),
    )
    token = compiled.token
    expected: dict[str, str] = {
        "library_id": token.library_id,
        "snapshot_hash": token.snapshot_hash,
        "policy_hash": token.policy_hash,
        "exclusion_hash": token.exclusion_hash,
        "permitted_set_hash": token.permitted_set_hash,
    }
    expected[component] = replacement

    with pytest.raises(PermittedTokenMismatchError, match=component):
        verify_permitted_token(token, **expected)


def test_candidate_contract_cannot_carry_fragment_text() -> None:
    field_names = {item.name for item in dataclasses.fields(SnapshotCandidate)}

    assert "text" not in field_names
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SnapshotCandidate(  # type: ignore[call-arg]
            source_fragment_id="fragment_one",
            source_version_id="source_version_one",
            source_id="source_one",
            source_family_id=None,
            memberships=(SnapshotMembership(COLLECTION_A, MembershipState.ACTIVE),),
            text="this must never reach the access compiler",
        )


def test_access_models_are_deeply_immutable_at_the_collection_boundary() -> None:
    scope = _scope()
    candidate = _candidate("one")

    with pytest.raises(FrozenInstanceError):
        scope.purpose = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        candidate.memberships = ()  # type: ignore[misc]
    assert isinstance(scope.collection_ids, tuple)
    assert isinstance(candidate.memberships, tuple)


def test_conflicting_metadata_for_one_fragment_fails_closed() -> None:
    first = _candidate("same")
    conflicting = SnapshotCandidate(
        source_fragment_id=first.source_fragment_id,
        source_version_id="source_version_other",
        source_id=first.source_id,
        source_family_id=None,
        memberships=first.memberships,
    )

    with pytest.raises(InvalidAccessScopeError, match="conflicting"):
        compile_access(
            policy=_policy(),
            scope=_scope(),
            candidates=(first, conflicting),
        )


def test_public_and_token_payloads_do_not_gain_denial_fields() -> None:
    compiled = compile_access(
        policy=_policy(collection_rules=(CollectionRule(COLLECTION_A, PolicyEffect.DENY),)),
        scope=_scope(),
        candidates=(_candidate("secret"),),
    )
    public_payload: dict[str, Any] = compiled.public_result.payload()
    token_payload = compiled.token.public_payload()

    assert public_payload == {"policy_omission_present": True}
    assert set(token_payload) == {
        "schema",
        "library_id",
        "snapshot_hash",
        "policy_hash",
        "exclusion_hash",
        "permitted_set_hash",
        "token_hash",
    }
    combined = repr(public_payload) + repr(token_payload)
    assert "secret" not in combined
    assert "denied_ids" not in combined
    assert "denied_count" not in combined
