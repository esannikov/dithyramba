"""Narrow SQLite repository for scoped, append-only ReviewDecisions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from dithyramba.access import QueryExclusions
from dithyramba.contracts import (
    canonical_content_id,
    canonical_sha256_hex,
    new_id,
)
from dithyramba.ingest import (
    SOURCE_ADDRESS_SCHEMA,
    MarkdownSourceAddress,
    PdfSourceAddress,
)
from dithyramba.recall import (
    EvidencePacketResultContract,
    QueryRequest,
    RetrievalBudget,
)
from dithyramba.review.errors import (
    ReviewConflictError,
    ReviewDecisionNotFoundError,
    ReviewIntegrityError,
    ReviewTargetNotFoundError,
)
from dithyramba.review.models import (
    ReviewAction,
    ReviewDecision,
    ReviewDecisionRequest,
    ReviewScope,
    ReviewTargetType,
)

from .repository import (
    LibraryRepository,
    _insert_outbox_event,
    _load_canonical_object,
    _timestamp,
    _validated_timestamp,
)


@dataclass(frozen=True, slots=True)
class ReviewTarget:
    """A verified target and the Collections in which it can be reviewed."""

    target_type: ReviewTargetType
    target_id: str
    target_hash: str
    collection_ids: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "target_type": self.target_type.value,
            "target_id": self.target_id,
            "target_hash": self.target_hash,
            "collection_ids": list(self.collection_ids),
        }


class ReviewDecisionRepository:
    """Persist and strictly load ReviewDecisions inside one Library."""

    def __init__(self, repository: LibraryRepository) -> None:
        self._repository = repository

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def get_target(self, target_type: ReviewTargetType, target_id: str) -> ReviewTarget:
        if target_type is ReviewTargetType.SOURCE_FRAGMENT:
            return self._source_fragment_target(target_id)
        if target_type is ReviewTargetType.EVIDENCE_PACKET:
            return self._evidence_packet_target(target_id)
        if target_type is ReviewTargetType.PACKET_ITEM:
            return self._packet_item_target(target_id)
        raise ReviewIntegrityError("unsupported review target type")

    def list_packet_item_targets(self, evidence_packet_id: str) -> tuple[ReviewTarget, ...]:
        """Return stable review identities for every evidence item in one packet."""

        packet = self._evidence_packet_target(evidence_packet_id)
        rows = self._repository._store.connection.execute(
            """
            SELECT source_fragment_id, role, rank, score_text
            FROM packet_items
            WHERE evidence_packet_id = ?
            ORDER BY rank
            """,
            (evidence_packet_id,),
        ).fetchall()
        return tuple(self._packet_item_from_row(packet, row) for row in rows)

    def create(self, request: ReviewDecisionRequest) -> ReviewDecision:
        target = self.get_target(request.target_type, request.target_id)
        if request.target_hash != target.target_hash:
            raise ReviewConflictError("review target hash is stale or belongs to another object")
        if not set(request.scope.collection_ids).issubset(target.collection_ids):
            raise ReviewIntegrityError("ReviewScope is outside the target Collection scope")

        created_at = _timestamp(self._repository._clock)
        review_decision_id = new_id("review")
        scope_json = request.scope.canonical_bytes.decode("utf-8")
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                if request.action is ReviewAction.SUPERSEDE:
                    prior = self._load_row(
                        request.supersedes_review_decision_id or "",
                        connection=connection,
                    )
                    self._validate_supersede(prior, request, connection)
                connection.execute(
                    """
                    INSERT INTO review_decisions(
                        review_decision_id, library_id, target_type, target_id,
                        target_hash, action, reason, authority, scope_json,
                        supersedes_review_decision_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_decision_id,
                        self.library_id,
                        request.target_type.value,
                        request.target_id,
                        request.target_hash,
                        request.action.value,
                        request.reason,
                        request.authority,
                        scope_json,
                        request.supersedes_review_decision_id,
                        created_at,
                    ),
                )
                event_payload = {
                    "schema": "dithyramba.review_decision_created/1.0",
                    "review_decision_id": review_decision_id,
                    "library_id": self.library_id,
                    **request.semantic_payload(),
                }
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="review_decision.created",
                    aggregate_type="review_decision",
                    aggregate_id=review_decision_id,
                    payload=event_payload,
                    occurred_at=created_at,
                )
        except ReviewDecisionNotFoundError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ReviewConflictError(
                "ReviewDecision transaction conflicted with append-only state"
            ) from exc
        return self.get(review_decision_id)

    def get(self, review_decision_id: str) -> ReviewDecision:
        return self._load_row(review_decision_id, connection=self._repository._store.connection)

    def list(
        self,
        *,
        target_type: ReviewTargetType | None = None,
        target_id: str | None = None,
        limit: int = 100,
    ) -> tuple[ReviewDecision, ...]:
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ReviewIntegrityError("ReviewDecision list limit must be between 1 and 500")
        if (target_type is None) != (target_id is None):
            raise ReviewIntegrityError(
                "target_type and target_id filters must be supplied together"
            )
        parameters: list[object] = [self.library_id]
        where = "library_id = ?"
        if target_type is not None and target_id is not None:
            self.get_target(target_type, target_id)
            where += " AND target_type = ? AND target_id = ?"
            parameters.extend((target_type.value, target_id))
        parameters.append(limit)
        rows = self._repository._store.connection.execute(
            f"""
            SELECT review_decision_id
            FROM review_decisions
            WHERE {where}
            ORDER BY created_at ASC, review_decision_id ASC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return tuple(self.get(str(row[0])) for row in rows)

    def _load_row(
        self,
        review_decision_id: str,
        *,
        connection: sqlite3.Connection,
    ) -> ReviewDecision:
        row = connection.execute(
            """
            SELECT review_decision_id, library_id, target_type, target_id,
                   target_hash, action, reason, authority, scope_json,
                   supersedes_review_decision_id, created_at
            FROM review_decisions
            WHERE review_decision_id = ? AND library_id = ?
            """,
            (review_decision_id, self.library_id),
        ).fetchone()
        if row is None:
            raise ReviewDecisionNotFoundError("ReviewDecision does not exist in this Library")
        scope_payload = _load_canonical_object(str(row[8]), "ReviewDecision scope")
        try:
            if set(scope_payload) != {"schema", "collection_ids", "use"}:
                raise ValueError("unexpected ReviewScope fields")
            raw_collections = scope_payload["collection_ids"]
            if type(raw_collections) is not list or any(
                type(value) is not str for value in raw_collections
            ):
                raise ValueError("invalid ReviewScope Collection IDs")
            scope = ReviewScope(
                collection_ids=tuple(raw_collections),
                use=_strict_str(scope_payload["use"], "ReviewScope use"),
            )
            if scope.semantic_payload() != scope_payload:
                raise ValueError("noncanonical ReviewScope")
            decision = ReviewDecision(
                review_decision_id=str(row[0]),
                library_id=str(row[1]),
                target_type=ReviewTargetType(str(row[2])),
                target_id=str(row[3]),
                target_hash=str(row[4]),
                action=ReviewAction(str(row[5])),
                reason=str(row[6]),
                authority=str(row[7]),
                scope=scope,
                supersedes_review_decision_id=(None if row[9] is None else str(row[9])),
                created_at=_validated_timestamp(str(row[10])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewIntegrityError("persisted ReviewDecision is not canonical") from exc

        target = self.get_target(decision.target_type, decision.target_id)
        if target.target_hash != decision.target_hash:
            raise ReviewIntegrityError("persisted ReviewDecision target hash is inconsistent")
        if not set(decision.scope.collection_ids).issubset(target.collection_ids):
            raise ReviewIntegrityError("persisted ReviewDecision scope is inconsistent")
        return decision

    def _validate_supersede(
        self,
        prior: ReviewDecision,
        request: ReviewDecisionRequest,
        connection: sqlite3.Connection,
    ) -> None:
        if (
            prior.target_type is not request.target_type
            or prior.target_id != request.target_id
            or prior.target_hash != request.target_hash
            or prior.scope != request.scope
        ):
            raise ReviewIntegrityError(
                "supersede must retain the prior target, hash, and ReviewScope"
            )
        already_superseded = connection.execute(
            """
            SELECT 1 FROM review_decisions
            WHERE library_id = ? AND supersedes_review_decision_id = ?
            LIMIT 1
            """,
            (self.library_id, prior.review_decision_id),
        ).fetchone()
        if already_superseded is not None:
            raise ReviewConflictError("ReviewDecision has already been superseded")

    def _source_fragment_target(self, source_fragment_id: str) -> ReviewTarget:
        row = self._repository._store.connection.execute(
            """
            SELECT sf.source_fragment_id, sf.source_version_id, sf.ordinal,
                   sf.fragment_kind, sf.text_sha256, sf.source_address_json,
                   sf.address_hash, sv.source_id, s.library_id
            FROM source_fragments AS sf
            JOIN source_versions AS sv ON sv.source_version_id = sf.source_version_id
            JOIN sources AS s ON s.source_id = sv.source_id
            WHERE sf.source_fragment_id = ? AND s.library_id = ?
            """,
            (source_fragment_id, self.library_id),
        ).fetchone()
        if row is None:
            raise ReviewTargetNotFoundError("review target does not exist in this Library")
        address = _validated_source_address_payload(
            _load_canonical_object(str(row[5]), "SourceFragment address")
        )
        address_hash = str(row[6])
        if canonical_sha256_hex(address) != address_hash:
            raise ReviewIntegrityError("SourceFragment address hash is inconsistent")
        identity_payload: dict[str, object] = {
            "schema": "dithyramba.source_fragment_identity/1.0",
            "source_version_id": str(row[1]),
            "ordinal": _strict_int(row[2], "SourceFragment ordinal", minimum=0),
            "fragment_kind": str(row[3]),
            "text_sha256": str(row[4]),
            "address_hash": address_hash,
        }
        if canonical_content_id("fragment", identity_payload) != source_fragment_id:
            raise ReviewIntegrityError("SourceFragment content identity is inconsistent")
        payload: dict[str, object] = {
            "schema": "dithyramba.source_fragment_target/1.0",
            "source_fragment_id": str(row[0]),
            "source_version_id": str(row[1]),
            "ordinal": identity_payload["ordinal"],
            "fragment_kind": str(row[3]),
            "text_sha256": str(row[4]),
            "source_address": address,
            "address_hash": address_hash,
        }
        collection_ids = self._source_collection_ids(str(row[7]))
        return ReviewTarget(
            target_type=ReviewTargetType.SOURCE_FRAGMENT,
            target_id=source_fragment_id,
            target_hash=canonical_sha256_hex(payload),
            collection_ids=collection_ids,
        )

    def _evidence_packet_target(self, evidence_packet_id: str) -> ReviewTarget:
        row = self._repository._store.connection.execute(
            """
            SELECT ep.evidence_packet_id, ep.query_request_id, ep.corpus_snapshot_id,
                   ep.packet_json, ep.packet_hash, qr.library_id, qr.request_json,
                   qr.request_hash
            FROM evidence_packets AS ep
            JOIN query_requests AS qr ON qr.query_request_id = ep.query_request_id
            WHERE ep.evidence_packet_id = ? AND qr.library_id = ?
            """,
            (evidence_packet_id, self.library_id),
        ).fetchone()
        if row is None:
            raise ReviewTargetNotFoundError("review target does not exist in this Library")
        packet_payload = _load_canonical_object(str(row[3]), "EvidencePacket")
        packet_hash = str(row[4])
        if packet_payload.get("evidence_packet_id") != evidence_packet_id:
            raise ReviewIntegrityError("EvidencePacket JSON identity is inconsistent")
        if packet_payload.get("packet_hash") != packet_hash:
            raise ReviewIntegrityError("EvidencePacket JSON hash field is inconsistent")
        semantic = dict(packet_payload)
        semantic.pop("evidence_packet_id", None)
        semantic.pop("packet_hash", None)
        if canonical_sha256_hex(semantic) != packet_hash:
            raise ReviewIntegrityError("EvidencePacket hash is inconsistent")
        if canonical_content_id("packet", semantic) != evidence_packet_id:
            raise ReviewIntegrityError("EvidencePacket content identity is inconsistent")
        if packet_payload.get("query_request_id") != str(row[1]):
            raise ReviewIntegrityError("EvidencePacket QueryRequest relation is inconsistent")
        if packet_payload.get("corpus_snapshot_id") != str(row[2]):
            raise ReviewIntegrityError("EvidencePacket CorpusSnapshot relation is inconsistent")
        query = _query_from_storage(
            query_request_id=str(row[1]),
            request_json=str(row[6]),
            request_hash=str(row[7]),
        )
        if query.library_id != self.library_id:
            raise ReviewIntegrityError("EvidencePacket QueryRequest belongs to another Library")
        return ReviewTarget(
            target_type=ReviewTargetType.EVIDENCE_PACKET,
            target_id=evidence_packet_id,
            target_hash=packet_hash,
            collection_ids=query.collection_ids,
        )

    def _packet_item_target(self, packet_item_id: str) -> ReviewTarget:
        packet_rows = self._repository._store.connection.execute(
            """
            SELECT DISTINCT pi.evidence_packet_id
            FROM packet_items AS pi
            JOIN evidence_packets AS ep ON ep.evidence_packet_id = pi.evidence_packet_id
            JOIN query_requests AS qr ON qr.query_request_id = ep.query_request_id
            WHERE qr.library_id = ?
            ORDER BY pi.evidence_packet_id
            """,
            (self.library_id,),
        ).fetchall()
        for packet_row in packet_rows:
            for candidate in self.list_packet_item_targets(str(packet_row[0])):
                if candidate.target_id == packet_item_id:
                    return candidate
        raise ReviewTargetNotFoundError("review target does not exist in this Library")

    def _packet_item_from_row(
        self,
        packet: ReviewTarget,
        row: sqlite3.Row | tuple[object, ...],
    ) -> ReviewTarget:
        fragment = self._source_fragment_target(str(row[0]))
        intersection = tuple(
            sorted(set(packet.collection_ids).intersection(fragment.collection_ids))
        )
        if not intersection:
            raise ReviewIntegrityError("PacketItem fragment is outside packet Collections")
        payload: dict[str, object] = {
            "schema": "dithyramba.packet_item_target/1.0",
            "evidence_packet_id": packet.target_id,
            "evidence_packet_hash": packet.target_hash,
            "source_fragment_id": fragment.target_id,
            "source_fragment_target_hash": fragment.target_hash,
            "role": str(row[1]),
            "rank": _strict_int(row[2], "PacketItem rank", minimum=1),
            "score": str(row[3]),
        }
        return ReviewTarget(
            target_type=ReviewTargetType.PACKET_ITEM,
            target_id=canonical_content_id("packet_item", payload),
            target_hash=canonical_sha256_hex(payload),
            collection_ids=intersection,
        )

    def _source_collection_ids(self, source_id: str) -> tuple[str, ...]:
        rows = self._repository._store.connection.execute(
            """
            SELECT cm.collection_id
            FROM collection_memberships AS cm
            JOIN collections AS c ON c.collection_id = cm.collection_id
            WHERE cm.source_id = ? AND c.library_id = ?
            ORDER BY cm.collection_id
            """,
            (source_id, self.library_id),
        ).fetchall()
        collection_ids = tuple(str(row[0]) for row in rows)
        if not collection_ids:
            raise ReviewIntegrityError("SourceFragment has no Library-scoped Collection")
        return collection_ids


def _query_from_storage(
    *,
    query_request_id: str,
    request_json: str,
    request_hash: str,
) -> QueryRequest:
    payload = _load_canonical_object(request_json, "QueryRequest")
    expected_keys = {
        "schema",
        "question",
        "library_id",
        "collection_ids",
        "corpus_snapshot_id",
        "access_policy_id",
        "purpose",
        "exclusions",
        "retrieval",
        "result_contract",
    }
    try:
        if set(payload) != expected_keys or payload["schema"] != QueryRequest.SCHEMA:
            raise ValueError("unexpected QueryRequest shape")
        collections = payload["collection_ids"]
        exclusions = payload["exclusions"]
        retrieval = payload["retrieval"]
        result_contract = payload["result_contract"]
        if type(collections) is not list or any(type(item) is not str for item in collections):
            raise ValueError("invalid Collection IDs")
        if type(exclusions) is not dict or set(exclusions) != {
            "source_ids",
            "source_family_ids",
            "source_fragment_ids",
        }:
            raise ValueError("invalid exclusions")
        exclusion_lists = tuple(exclusions[key] for key in sorted(exclusions))
        if any(
            type(values) is not list or any(type(item) is not str for item in values)
            for values in exclusion_lists
        ):
            raise ValueError("invalid exclusion identifiers")
        if type(retrieval) is not dict or type(result_contract) is not dict:
            raise ValueError("invalid nested QueryRequest contract")
        query = QueryRequest(
            question=_strict_str(payload["question"], "QueryRequest question"),
            library_id=_strict_str(payload["library_id"], "QueryRequest Library ID"),
            collection_ids=tuple(collections),
            corpus_snapshot_id=_strict_str(
                payload["corpus_snapshot_id"], "QueryRequest CorpusSnapshot ID"
            ),
            access_policy_id=_strict_str(
                payload["access_policy_id"], "QueryRequest AccessPolicy ID"
            ),
            purpose=_strict_str(payload["purpose"], "QueryRequest purpose"),
            exclusions=QueryExclusions(
                source_ids=tuple(exclusions["source_ids"]),
                source_family_ids=tuple(exclusions["source_family_ids"]),
                source_fragment_ids=tuple(exclusions["source_fragment_ids"]),
            ),
            retrieval=RetrievalBudget(**retrieval),
            result_contract=EvidencePacketResultContract(**result_contract),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewIntegrityError("persisted QueryRequest is not canonical") from exc
    if query.canonical_bytes.decode("utf-8") != request_json:
        raise ReviewIntegrityError("persisted QueryRequest differs from typed reconstruction")
    if query.request_hash != request_hash or query.query_request_id != query_request_id:
        raise ReviewIntegrityError("persisted QueryRequest identity/hash is inconsistent")
    return query


def _strict_int(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ReviewIntegrityError(f"{label} is invalid")
    return value


def _strict_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise ReviewIntegrityError(f"{label} is invalid")
    return value


def _validated_source_address_payload(payload: dict[str, object]) -> dict[str, object]:
    address: MarkdownSourceAddress | PdfSourceAddress
    try:
        kind = payload.get("kind")
        if kind == "markdown":
            if set(payload) != {
                "schema",
                "kind",
                "heading_path",
                "line_start",
                "line_end",
                "char_start",
                "char_end",
            }:
                raise ValueError("unexpected Markdown address fields")
            heading_path = payload["heading_path"]
            if type(heading_path) is not list or any(
                type(item) is not str for item in heading_path
            ):
                raise ValueError("invalid Markdown heading path")
            address = MarkdownSourceAddress(
                heading_path=tuple(heading_path),
                line_start=_strict_int(
                    payload["line_start"], "Markdown address line_start", minimum=1
                ),
                line_end=_strict_int(payload["line_end"], "Markdown address line_end", minimum=1),
                char_start=_strict_int(
                    payload["char_start"], "Markdown address char_start", minimum=0
                ),
                char_end=_strict_int(payload["char_end"], "Markdown address char_end", minimum=0),
            )
        elif kind == "pdf":
            if set(payload) != {
                "schema",
                "kind",
                "page",
                "bbox",
                "char_start",
                "char_end",
            }:
                raise ValueError("unexpected PDF address fields")
            bbox = payload["bbox"]
            if (
                type(bbox) is not list
                or len(bbox) != 4
                or any(type(item) is not str for item in bbox)
            ):
                raise ValueError("invalid PDF address box")
            address = PdfSourceAddress(
                page=_strict_int(payload["page"], "PDF address page", minimum=1),
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                char_start=_strict_int(payload["char_start"], "PDF address char_start", minimum=0),
                char_end=_strict_int(payload["char_end"], "PDF address char_end", minimum=0),
            )
        else:
            raise ValueError("unsupported SourceAddress kind")
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewIntegrityError("SourceFragment address is invalid") from exc
    rebuilt = address.payload()
    if payload.get("schema") != SOURCE_ADDRESS_SCHEMA or rebuilt != payload:
        raise ReviewIntegrityError("SourceFragment address differs from typed reconstruction")
    return rebuilt
