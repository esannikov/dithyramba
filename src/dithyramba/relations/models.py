"""Typed P7 contracts for Relations and complete 1-3-hop path receipts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from dithyramba.access import RequestScope
from dithyramba.contracts import canonical_sha256_hex

from .errors import RelationContractError

_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ID = re.compile(r"^[a-z][a-z0-9_]+$")


class RelationType(StrEnum):
    """Frozen, non-generic Relation vocabulary for the P7 core."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"
    LIMITS = "limits"
    DEFINES = "defines"
    NARROWS = "narrows"
    BROADENS = "broadens"
    ANALOGOUS_TO = "analogous_to"
    DISTINGUISHES_FROM = "distinguishes_from"
    PRECEDES = "precedes"
    OVERLAPS = "overlaps"
    SUPERSEDES = "supersedes"
    ENABLES = "enables"
    INHIBITS = "inhibits"
    POSSIBLY_CAUSES = "possibly_causes"
    DERIVED_FROM = "derived_from"
    QUOTES = "quotes"
    DEPENDS_ON = "depends_on"
    TRANSFERS_METHOD_TO = "transfers_method_to"
    OPENS_GAP = "opens_gap"
    SUGGESTS_SCENE = "suggests_scene"


class RelationNodeType(StrEnum):
    """Core object types that a typed Relation may connect."""

    STATEMENT = "statement"
    CONCEPT = "concept"
    CONCEPT_MEANING = "concept_meaning"
    ENTITY = "entity"
    VOICE = "voice"
    TIME_CONTEXT = "time_context"
    STRUCTURE_UNIT = "structure_unit"

    @property
    def id_prefix(self) -> str:
        return {
            self.STATEMENT: "statement_",
            self.CONCEPT: "concept_",
            self.CONCEPT_MEANING: "concept_meaning_",
            self.ENTITY: "entity_",
            self.VOICE: "voice_",
            self.TIME_CONTEXT: "time_context_",
            self.STRUCTURE_UNIT: "structure_unit_",
        }[self]


class RelationDirection(StrEnum):
    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


@dataclass(frozen=True, slots=True, order=True)
class RelationNodeRef:
    node_type: RelationNodeType
    node_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_type, RelationNodeType):
            raise RelationContractError("node_type must be a RelationNodeType")
        _prefixed(self.node_id, self.node_type.id_prefix, "node_id")

    def payload(self) -> dict[str, str]:
        return {"node_type": self.node_type.value, "node_id": self.node_id}


@dataclass(frozen=True, slots=True)
class RelationNodeGrounding:
    """Exact SourceFragments that make one Relation endpoint admissible."""

    node: RelationNodeRef
    source_fragment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node, RelationNodeRef):
            raise RelationContractError("node must be a RelationNodeRef")
        object.__setattr__(
            self,
            "source_fragment_ids",
            _sorted_ids(
                self.source_fragment_ids,
                "fragment_",
                "source_fragment_ids",
                2_000,
            ),
        )
        if not self.source_fragment_ids:
            raise RelationContractError("Relation node grounding cannot be empty")

    def payload(self) -> dict[str, object]:
        return {
            "node": self.node.payload(),
            "source_fragment_ids": list(self.source_fragment_ids),
        }


@dataclass(frozen=True, slots=True)
class RelationImportRequest:
    corpus_snapshot_id: str
    access_policy_id: str
    scope: RequestScope
    code_version: str
    profile_version: str
    max_relations: int = 400

    def __post_init__(self) -> None:
        _prefixed(self.corpus_snapshot_id, "snapshot_", "corpus_snapshot_id")
        _prefixed(self.access_policy_id, "policy_", "access_policy_id")
        if not isinstance(self.scope, RequestScope):
            raise RelationContractError("scope must be a RequestScope")
        _text(self.code_version, "code_version", maximum=128)
        _text(self.profile_version, "profile_version", maximum=128)
        _integer(self.max_relations, 1, 400, "max_relations")

    @property
    def scope_hash(self) -> str:
        return self.scope.exclusion_hash

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.relation_import_request/1.0",
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "scope": _scope_payload(self.scope),
            "code_version": self.code_version,
            "profile_version": self.profile_version,
            "max_relations": self.max_relations,
        }


@dataclass(frozen=True, slots=True)
class RelationProposal:
    """Candidate edge plus an untrusted internal authoring annotation.

    ``reason_text`` is intentionally *not* policy-certified provenance.  It may
    help a reviewer understand why an extractor proposed the edge, but only the
    text-free :class:`RelationGroundingProjection` is safe to carry into future
    policy-facing traversal and admission receipts.
    """

    relation_type: RelationType
    subject: RelationNodeRef
    object_node: RelationNodeRef
    evidence_link_ids: tuple[str, ...]
    reason_text: str
    extraction_method: str
    valid_start: str | None = None
    valid_end: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation_type, RelationType):
            raise RelationContractError("relation_type must be a RelationType")
        if not isinstance(self.subject, RelationNodeRef) or not isinstance(
            self.object_node, RelationNodeRef
        ):
            raise RelationContractError("Relation endpoints must be RelationNodeRef values")
        if self.subject == self.object_node:
            raise RelationContractError("a Relation cannot connect a node to itself")
        object.__setattr__(
            self,
            "evidence_link_ids",
            _sorted_ids(self.evidence_link_ids, "evidence_", "evidence_link_ids", 16),
        )
        if not self.evidence_link_ids:
            raise RelationContractError("a Relation requires at least one EvidenceLink")
        _text(self.reason_text, "reason_text", maximum=4_000)
        _text(self.extraction_method, "extraction_method", maximum=128)
        for value, label in ((self.valid_start, "valid_start"), (self.valid_end, "valid_end")):
            if value is not None:
                _text(value, label, maximum=128)

    @property
    def content_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    def grounding_projection(self) -> RelationGroundingProjection:
        """Return the text-free edge/evidence claim certified by policy reads."""

        return RelationGroundingProjection(
            relation_type=self.relation_type,
            subject=self.subject,
            object_node=self.object_node,
            evidence_link_ids=self.evidence_link_ids,
            valid_start=self.valid_start,
            valid_end=self.valid_end,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.relation/1.0",
            "relation_type": self.relation_type.value,
            "subject": self.subject.payload(),
            "object": self.object_node.payload(),
            "evidence_link_ids": list(self.evidence_link_ids),
            "reason_text": self.reason_text,
            "extraction_method": self.extraction_method,
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
            "status": "candidate",
        }


@dataclass(frozen=True, slots=True)
class Relation:
    relation_id: str
    import_run_id: str
    library_id: str
    relation_type: RelationType
    subject: RelationNodeRef
    object_node: RelationNodeRef
    evidence_link_ids: tuple[str, ...]
    reason_text: str
    extraction_method: str
    valid_start: str | None
    valid_end: str | None
    content_hash: str

    def __post_init__(self) -> None:
        _prefixed(self.relation_id, "relation_", "relation_id")
        _prefixed(self.import_run_id, "relation_run_", "import_run_id")
        _prefixed(self.library_id, "library_", "library_id")
        proposal = self.as_proposal()
        _hash(self.content_hash, "content_hash")
        if proposal.content_hash != self.content_hash:
            raise RelationContractError("Relation content_hash does not match its payload")

    def as_proposal(self) -> RelationProposal:
        return RelationProposal(
            relation_type=self.relation_type,
            subject=self.subject,
            object_node=self.object_node,
            evidence_link_ids=self.evidence_link_ids,
            reason_text=self.reason_text,
            extraction_method=self.extraction_method,
            valid_start=self.valid_start,
            valid_end=self.valid_end,
        )

    def grounding_projection(self) -> RelationGroundingProjection:
        """Return the policy-facing claim without the untrusted annotation."""

        return self.as_proposal().grounding_projection()

    @property
    def creation_import_run_id(self) -> str:
        """Return the immutable run that first created this canonical Relation."""

        return self.import_run_id

    def payload(self) -> dict[str, object]:
        return self.as_proposal().payload() | {
            "relation_id": self.relation_id,
            "import_run_id": self.import_run_id,
            "library_id": self.library_id,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class RelationGroundingProjection:
    """Text-free Relation claim bound only to typed endpoints and exact evidence.

    This projection is the compatibility boundary for future accepted-only path
    receipts.  It deliberately excludes ``reason_text`` and extraction metadata:
    neither proves that free-form text was produced exclusively from the declared
    ``AuthorizedRead``.  A displayable generated rationale will require its own
    repository-issued generation receipt.
    """

    relation_type: RelationType
    subject: RelationNodeRef
    object_node: RelationNodeRef
    evidence_link_ids: tuple[str, ...]
    valid_start: str | None = None
    valid_end: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation_type, RelationType):
            raise RelationContractError("relation_type must be a RelationType")
        if not isinstance(self.subject, RelationNodeRef) or not isinstance(
            self.object_node, RelationNodeRef
        ):
            raise RelationContractError("Relation endpoints must be RelationNodeRef values")
        if self.subject == self.object_node:
            raise RelationContractError("a Relation cannot connect a node to itself")
        object.__setattr__(
            self,
            "evidence_link_ids",
            _sorted_ids(self.evidence_link_ids, "evidence_", "evidence_link_ids", 16),
        )
        if not self.evidence_link_ids:
            raise RelationContractError("a Relation requires at least one EvidenceLink")
        for value, label in ((self.valid_start, "valid_start"), (self.valid_end, "valid_end")):
            if value is not None:
                _text(value, label, maximum=128)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.relation_grounding_projection/1.0",
            "relation_type": self.relation_type.value,
            "subject": self.subject.payload(),
            "object": self.object_node.payload(),
            "evidence_link_ids": list(self.evidence_link_ids),
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
        }

    @property
    def projection_hash(self) -> str:
        return canonical_sha256_hex(self.payload())


@dataclass(frozen=True, slots=True)
class RelationImportReceipt:
    import_run_id: str
    library_id: str
    corpus_snapshot_id: str
    access_policy_id: str
    policy_hash: str
    scope_hash: str
    permitted_set_hash: str
    code_version: str
    profile_version: str
    input_hash: str
    output_hash: str
    relation_ids: tuple[str, ...]
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _prefixed(self.import_run_id, "relation_run_", "import_run_id")
        _prefixed(self.library_id, "library_", "library_id")
        _prefixed(self.corpus_snapshot_id, "snapshot_", "corpus_snapshot_id")
        _prefixed(self.access_policy_id, "policy_", "access_policy_id")
        for value, label in (
            (self.policy_hash, "policy_hash"),
            (self.scope_hash, "scope_hash"),
            (self.permitted_set_hash, "permitted_set_hash"),
            (self.input_hash, "input_hash"),
            (self.output_hash, "output_hash"),
        ):
            _hash(value, label)
        _text(self.code_version, "code_version", maximum=128)
        _text(self.profile_version, "profile_version", maximum=128)
        object.__setattr__(
            self,
            "relation_ids",
            _sorted_ids(self.relation_ids, "relation_", "relation_ids", 400),
        )
        if not self.relation_ids:
            raise RelationContractError("Relation import receipt cannot be empty")
        object.__setattr__(self, "receipt_hash", canonical_sha256_hex(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.relation_import_receipt/1.0",
            "import_run_id": self.import_run_id,
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "policy_hash": self.policy_hash,
            "scope_hash": self.scope_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "code_version": self.code_version,
            "profile_version": self.profile_version,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "relation_ids": list(self.relation_ids),
        }


@dataclass(frozen=True, slots=True)
class RelationImportResult:
    relations: tuple[Relation, ...]
    receipt: RelationImportReceipt

    def __post_init__(self) -> None:
        if type(self.relations) is not tuple or not self.relations:
            raise RelationContractError("Relation import result cannot be empty")
        if any(not isinstance(item, Relation) for item in self.relations):
            raise RelationContractError("relations must contain Relation values")
        if tuple(item.relation_id for item in self.relations) != self.receipt.relation_ids:
            raise RelationContractError("Relation import output differs from its receipt")


@dataclass(frozen=True, slots=True)
class RelationPathRequest:
    corpus_snapshot_id: str
    access_policy_id: str
    scope: RequestScope
    seeds: tuple[RelationNodeRef, ...]
    relation_types: tuple[RelationType, ...]
    direction: RelationDirection = RelationDirection.BOTH
    max_hops: int = 2
    max_paths: int = 50

    def __post_init__(self) -> None:
        _prefixed(self.corpus_snapshot_id, "snapshot_", "corpus_snapshot_id")
        _prefixed(self.access_policy_id, "policy_", "access_policy_id")
        if not isinstance(self.scope, RequestScope):
            raise RelationContractError("scope must be a RequestScope")
        if type(self.seeds) is not tuple or not 1 <= len(self.seeds) <= 32:
            raise RelationContractError("seeds must contain between 1 and 32 nodes")
        if any(not isinstance(item, RelationNodeRef) for item in self.seeds):
            raise RelationContractError("seeds must contain RelationNodeRef values")
        canonical_seeds = tuple(sorted(set(self.seeds)))
        object.__setattr__(self, "seeds", canonical_seeds)
        if type(self.relation_types) is not tuple or not self.relation_types:
            raise RelationContractError("relation_types must be a non-empty tuple")
        if any(not isinstance(item, RelationType) for item in self.relation_types):
            raise RelationContractError("relation_types must contain RelationType values")
        object.__setattr__(
            self,
            "relation_types",
            tuple(sorted(set(self.relation_types), key=lambda item: item.value)),
        )
        if not isinstance(self.direction, RelationDirection):
            raise RelationContractError("direction must be a RelationDirection")
        _integer(self.max_hops, 1, 3, "max_hops")
        _integer(self.max_paths, 1, 100, "max_paths")

    @property
    def request_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.relation_path_request/1.0",
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "scope": _scope_payload(self.scope),
            "seeds": [item.payload() for item in self.seeds],
            "relation_types": [item.value for item in self.relation_types],
            "direction": self.direction.value,
            "max_hops": self.max_hops,
            "max_paths": self.max_paths,
        }


@dataclass(frozen=True, slots=True)
class RelationPathStep:
    relation_id: str
    relation_type: RelationType
    from_node: RelationNodeRef
    to_node: RelationNodeRef
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _prefixed(self.relation_id, "relation_", "relation_id")
        if not isinstance(self.relation_type, RelationType):
            raise RelationContractError("relation_type must be a RelationType")
        if not isinstance(self.from_node, RelationNodeRef) or not isinstance(
            self.to_node, RelationNodeRef
        ):
            raise RelationContractError("path endpoints must be RelationNodeRef values")
        if self.from_node == self.to_node:
            raise RelationContractError("path step cannot be a self-loop")
        object.__setattr__(
            self,
            "evidence_link_ids",
            _sorted_ids(self.evidence_link_ids, "evidence_", "evidence_link_ids", 16),
        )
        if not self.evidence_link_ids:
            raise RelationContractError("path step requires exact EvidenceLinks")

    def payload(self) -> dict[str, object]:
        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type.value,
            "from": self.from_node.payload(),
            "to": self.to_node.payload(),
            "evidence_link_ids": list(self.evidence_link_ids),
        }


@dataclass(frozen=True, slots=True)
class RelationPath:
    seed_order: int
    seed: RelationNodeRef
    steps: tuple[RelationPathStep, ...]
    path_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _integer(self.seed_order, 0, 31, "seed_order")
        if not isinstance(self.seed, RelationNodeRef):
            raise RelationContractError("seed must be a RelationNodeRef")
        if type(self.steps) is not tuple or not 1 <= len(self.steps) <= 3:
            raise RelationContractError("RelationPath must contain 1 to 3 steps")
        if any(not isinstance(item, RelationPathStep) for item in self.steps):
            raise RelationContractError("steps must contain RelationPathStep values")
        current = self.seed
        seen = {current}
        relation_ids: set[str] = set()
        for step in self.steps:
            if step.from_node != current:
                raise RelationContractError("RelationPath steps are not contiguous")
            if step.to_node in seen or step.relation_id in relation_ids:
                raise RelationContractError("RelationPath contains a cycle")
            seen.add(step.to_node)
            relation_ids.add(step.relation_id)
            current = step.to_node
        object.__setattr__(self, "path_hash", canonical_sha256_hex(self.payload()))

    @property
    def terminal(self) -> RelationNodeRef:
        return self.steps[-1].to_node

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.relation_path/1.0",
            "seed_order": self.seed_order,
            "seed": self.seed.payload(),
            "steps": [item.payload() for item in self.steps],
        }


@dataclass(frozen=True, slots=True)
class RelationPathReceipt:
    receipt_id: str
    library_id: str
    corpus_snapshot_id: str
    access_policy_id: str
    policy_hash: str
    scope_hash: str
    permitted_set_hash: str
    request: RelationPathRequest
    paths: tuple[RelationPath, ...]
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _prefixed(self.receipt_id, "relation_path_", "receipt_id")
        _prefixed(self.library_id, "library_", "library_id")
        _prefixed(self.corpus_snapshot_id, "snapshot_", "corpus_snapshot_id")
        _prefixed(self.access_policy_id, "policy_", "access_policy_id")
        for value, label in (
            (self.policy_hash, "policy_hash"),
            (self.scope_hash, "scope_hash"),
            (self.permitted_set_hash, "permitted_set_hash"),
        ):
            _hash(value, label)
        if not isinstance(self.request, RelationPathRequest):
            raise RelationContractError("request must be a RelationPathRequest")
        if type(self.paths) is not tuple or len(self.paths) > self.request.max_paths:
            raise RelationContractError("paths exceed the request max_paths")
        if any(not isinstance(item, RelationPath) for item in self.paths):
            raise RelationContractError("paths must contain RelationPath values")
        if tuple(item.path_hash for item in self.paths) != tuple(
            sorted(item.path_hash for item in self.paths)
        ):
            raise RelationContractError("paths must use canonical hash order")
        object.__setattr__(self, "receipt_hash", canonical_sha256_hex(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.relation_path_receipt/1.0",
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "policy_hash": self.policy_hash,
            "scope_hash": self.scope_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "request": self.request.payload(),
            "paths": [item.payload() | {"path_hash": item.path_hash} for item in self.paths],
        }


def _scope_payload(scope: RequestScope) -> dict[str, object]:
    return {
        "library_id": scope.library_id,
        "snapshot_hash": scope.snapshot_hash,
        "purpose": scope.purpose,
        "collection_ids": list(scope.collection_ids),
        "exclusions": scope.exclusions.payload(),
    }


def _prefixed(value: object, prefix: str, label: str) -> str:
    if type(value) is not str or not value.startswith(prefix) or _ID.fullmatch(value) is None:
        raise RelationContractError(f"{label} must use the {prefix} prefix")
    return value


def _hash(value: object, label: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise RelationContractError(f"{label} must be lowercase SHA-256")
    return value


def _text(value: object, label: str, *, maximum: int) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or not value
        or len(value) > maximum
        or "\x00" in value
    ):
        raise RelationContractError(f"{label} must be non-empty trimmed text")
    return value


def _integer(value: object, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise RelationContractError(f"{label} must be between {minimum} and {maximum}")
    return value


def _sorted_ids(
    values: object,
    prefix: str,
    label: str,
    maximum: int,
) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) > maximum:
        raise RelationContractError(f"{label} must be a tuple of at most {maximum} IDs")
    result = tuple(sorted(_prefixed(item, prefix, label) for item in values))
    if len(set(result)) != len(result):
        raise RelationContractError(f"{label} must be unique")
    return result
