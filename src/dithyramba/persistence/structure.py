"""Append-only persistence and exact rendering for P7 StructureUnits."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import cast

from dithyramba.access import AccessContractError
from dithyramba.contracts import (
    CanonicalizationError,
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)
from dithyramba.ingest import FragmentKind, MarkdownSourceAddress
from dithyramba.structure import (
    MarkdownFragmentDescriptor,
    MarkdownStructureGenerationResult,
    MarkdownStructureProfile,
    MarkdownStructureProposalBuild,
    StructureAuthorizationError,
    StructureContractError,
    StructureGeneration,
    StructureGenerationRequest,
    StructureInputExclusion,
    StructureInputExclusionReason,
    StructureIntegrityError,
    StructureNotFoundError,
    StructureReadReceipt,
    StructureReadReceiptItem,
    StructureUnit,
    StructureUnitKind,
    StructureUnitMember,
    StructureUnitProposal,
    StructureUnitText,
    build_markdown_structure_proposals,
)

from .errors import (
    AccessPolicyNotFoundError,
    AuthorizationError,
    CollectionNotFoundError,
    CorpusSnapshotNotFoundError,
)
from .models import AuthorizedRead, SourceFragmentText
from .repository import LibraryRepository, _timestamp

_DELIMITER = "\n\n"
_PERSIST_BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class _FragmentMetadata:
    source_fragment_id: str
    source_version_id: str
    source_id: str
    ordinal: int
    text_sha256: str


@dataclass(frozen=True, slots=True)
class _PermittedStructureMetadata:
    """Text-free metadata selected for one permitted SourceFragment."""

    source_fragment_id: str
    source_version_id: str
    source_id: str
    ordinal: int
    fragment_kind: FragmentKind
    address_kind: str
    heading_path: tuple[str, ...] | None


class SQLiteStructureRepository:
    """Persist and read explicit broader context without hiding its fragments."""

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteStructureRepository requires a LibraryRepository")
        self._repository = repository
        # Structure generations are append-only. Keeping fully verified pages
        # avoids rebuilding many-thousand-unit objects for every query or
        # final StructureUnit read in one benchmark run.
        self._generation_cache: dict[str, StructureGeneration] = {}

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def generate_markdown_structure(
        self,
        request: StructureGenerationRequest,
        profile: MarkdownStructureProfile,
    ) -> MarkdownStructureGenerationResult:
        """Generate and persist policy-safe Markdown StructureUnits from metadata only.

        Valid canonical non-Markdown addresses are explicitly excluded. A
        fragment-level request exclusion prevents every Markdown proposal for
        that request: without reading forbidden metadata, no complete natural
        boundary can be proven. If a permitted SourceVersion otherwise has an
        ordinal gap, all of its permitted fragments are excluded rather than
        converted into misleading partial sections. No ``source_fragments.text``
        read is authorized by this path.
        """

        build, exclusions = self.prepare_markdown_structure(request, profile)
        generation = self.persist_generation(request, build.proposals) if build.proposals else None
        return MarkdownStructureGenerationResult(
            request=request,
            build=build,
            generation=generation,
            exclusions=exclusions,
        )

    def prepare_markdown_structure(
        self,
        request: StructureGenerationRequest,
        profile: MarkdownStructureProfile,
    ) -> tuple[MarkdownStructureProposalBuild, tuple[StructureInputExclusion, ...]]:
        """Build the complete authorized Markdown proposal set without persisting it.

        This is the scalable seam for callers that need to persist a complete
        proposal set in deterministic pages. It has the same policy-first,
        metadata-only behavior as :meth:`generate_markdown_structure` and
        never reads ``source_fragments.text``.
        """

        if not isinstance(request, StructureGenerationRequest):
            raise TypeError("request must be a StructureGenerationRequest")
        if not isinstance(profile, MarkdownStructureProfile):
            raise TypeError("profile must be a MarkdownStructureProfile")
        if request.scope.library_id != self.library_id:
            raise StructureAuthorizationError("StructureUnit scope targets another Library")
        if (
            request.profile_id != profile.profile_id
            or request.profile_version != profile.profile_version
        ):
            raise StructureAuthorizationError(
                "StructureGenerationRequest differs from the Markdown structure profile"
            )
        self._validate_snapshot_scope(
            request.corpus_snapshot_id,
            request.scope.snapshot_hash,
        )
        try:
            authorization = self._repository.authorize_read(
                access_policy_id=request.access_policy_id,
                scope=request.scope,
            )
        except (
            AccessContractError,
            AccessPolicyNotFoundError,
            AuthorizationError,
            CollectionNotFoundError,
        ) as exc:
            raise StructureAuthorizationError(str(exc)) from exc
        token = authorization.compiled.token
        if (
            token.snapshot_hash != request.scope.snapshot_hash
            or token.exclusion_hash != request.scope_hash
        ):
            raise StructureAuthorizationError("StructureUnit authorization binding changed")

        metadata = self._permitted_structure_metadata(request, authorization)
        descriptors, exclusions = self._prepare_markdown_inputs(metadata)
        if request.scope.exclusions.source_fragment_ids:
            descriptors = ()
            exclusions = tuple(
                item
                for item in exclusions
                if item.reason is StructureInputExclusionReason.UNSUPPORTED_SOURCE_ADDRESS
            ) + tuple(
                _structure_input_exclusion(
                    item,
                    StructureInputExclusionReason.FRAGMENT_LEVEL_EXCLUSION_PRESENT,
                )
                for item in metadata
                if item.address_kind == "markdown"
            )
        try:
            build = (
                build_markdown_structure_proposals(descriptors, profile=profile)
                if descriptors
                else MarkdownStructureProposalBuild(profile, (), ())
            )
        except StructureContractError as exc:
            raise StructureIntegrityError(
                "permitted Markdown metadata cannot form canonical StructureUnits"
            ) from exc
        return build, tuple(sorted(exclusions))

    def persist_generation(
        self,
        request: StructureGenerationRequest,
        proposals: tuple[StructureUnitProposal, ...],
    ) -> StructureGeneration:
        """Validate and atomically persist one complete StructureUnit generation."""

        if not isinstance(request, StructureGenerationRequest):
            raise TypeError("request must be a StructureGenerationRequest")
        if (
            type(proposals) is not tuple
            or not 1 <= len(proposals) <= 5_000
            or any(not isinstance(item, StructureUnitProposal) for item in proposals)
        ):
            raise TypeError("proposals must contain 1 to 5,000 StructureUnitProposal values")
        if request.scope.library_id != self.library_id:
            raise StructureAuthorizationError("StructureUnit scope targets another Library")
        self._validate_snapshot_scope(
            request.corpus_snapshot_id,
            request.scope.snapshot_hash,
        )

        try:
            authorization = self._repository.authorize_read(
                access_policy_id=request.access_policy_id,
                scope=request.scope,
            )
        except (
            AccessContractError,
            AccessPolicyNotFoundError,
            AuthorizationError,
            CollectionNotFoundError,
        ) as exc:
            raise StructureAuthorizationError(str(exc)) from exc
        token = authorization.compiled.token
        if token.snapshot_hash != request.scope.snapshot_hash:
            raise StructureAuthorizationError("StructureUnit snapshot binding changed")

        requested_ids = tuple(
            fragment_id for proposal in proposals for fragment_id in proposal.source_fragment_ids
        )
        permitted_ids = {item.source_fragment_id for item in authorization.compiled.manifest.items}
        if not set(requested_ids).issubset(permitted_ids):
            raise StructureAuthorizationError(
                "StructureUnit proposal contains a fragment outside the permitted manifest"
            )
        metadata = self._fragment_metadata(tuple(sorted(set(requested_ids))))
        prepared = [self._prepare_unit(proposal, metadata) for proposal in proposals]
        unit_ids = [item[0] for item in prepared]
        if len(set(unit_ids)) != len(unit_ids):
            raise StructureIntegrityError("StructureUnit proposals contain duplicate content")
        prepared.sort(key=lambda item: item[0])

        unit_payloads = [
            semantic | {"structure_unit_id": unit_id, "content_hash": content_hash}
            for unit_id, semantic, content_hash, _members in prepared
        ]
        generation_payload: dict[str, object] = {
            "schema": "dithyramba.structure_generation/1.0",
            "library_id": self.library_id,
            "corpus_snapshot_id": request.corpus_snapshot_id,
            "access_policy_id": request.access_policy_id,
            "policy_hash": token.policy_hash,
            "scope_hash": token.exclusion_hash,
            "permitted_set_hash": token.permitted_set_hash,
            "profile_id": request.profile_id,
            "profile_version": request.profile_version,
            "units": unit_payloads,
        }
        generation_hash = canonical_sha256_hex(generation_payload)
        generation_id = canonical_content_id(
            "structure_generation",
            {
                "schema": "dithyramba.structure_generation_identity/1.0",
                "generation_hash": generation_hash,
            },
        )
        units = tuple(
            StructureUnit(
                structure_unit_id=unit_id,
                generation_id=generation_id,
                source_id=str(semantic["source_id"]),
                source_version_id=str(semantic["source_version_id"]),
                kind=StructureUnitKind(str(semantic["kind"])),
                label=cast(str | None, proposal_label),
                members=members,
                boundary_hash=str(semantic["boundary_hash"]),
                content_hash=content_hash,
            )
            for unit_id, semantic, content_hash, members in prepared
            for proposal_label in (semantic["label"],)
        )
        generation = StructureGeneration(
            generation_id=generation_id,
            library_id=self.library_id,
            corpus_snapshot_id=request.corpus_snapshot_id,
            access_policy_id=request.access_policy_id,
            policy_hash=token.policy_hash,
            scope_hash=token.exclusion_hash,
            permitted_set_hash=token.permitted_set_hash,
            profile_id=request.profile_id,
            profile_version=request.profile_version,
            units=units,
            generation_hash=generation_hash,
        )
        if canonical_sha256_hex(generation.payload()) != generation_hash:
            raise StructureIntegrityError("StructureUnit generation hash is inconsistent")

        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO structure_unit_generations(
                        structure_unit_generation_id, library_id, corpus_snapshot_id,
                        access_policy_id, policy_hash, scope_hash, permitted_set_hash,
                        profile_id, profile_version, unit_count, generation_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation.generation_id,
                        generation.library_id,
                        generation.corpus_snapshot_id,
                        generation.access_policy_id,
                        generation.policy_hash,
                        generation.scope_hash,
                        generation.permitted_set_hash,
                        generation.profile_id,
                        generation.profile_version,
                        len(generation.units),
                        generation.generation_hash,
                        created_at,
                    ),
                )
                if cursor.rowcount == 1:
                    self._insert_units(connection, generation.units, created_at)
                persisted = self._load_generation(connection, generation.generation_id)
                if persisted != generation:
                    raise StructureIntegrityError(
                        "persisted StructureUnit generation conflicts with its content address"
                    )
        except sqlite3.IntegrityError as exc:
            raise StructureIntegrityError("StructureUnit generation insert conflicted") from exc
        self._generation_cache[generation.generation_id] = generation
        return generation

    def load_generation(self, generation_id: str) -> StructureGeneration:
        cached = self._generation_cache.get(generation_id)
        if cached is not None:
            return cached
        with self._repository._store.transaction() as connection:
            generation = self._load_generation(connection, generation_id)
        self._generation_cache[generation_id] = generation
        return generation

    def load_unit(self, structure_unit_id: str) -> StructureUnit:
        with self._repository._store.transaction() as connection:
            return self._load_unit(connection, structure_unit_id)

    def resolve_permitted_units(
        self,
        *,
        generation_id: str,
        access_policy_id: str,
        scope: object,
        anchor_fragment_ids: tuple[str, ...],
    ) -> tuple[StructureUnit, ...]:
        """Resolve metadata-only units around exact authorized atomic anchors.

        The method deliberately requires the generation's exact policy/scope
        closure. It does not read source text or create a read receipt; callers
        must use ``read_unit`` only for their bounded final shortlist.
        """

        from dithyramba.access import RequestScope

        if not isinstance(scope, RequestScope):
            raise TypeError("scope must be a RequestScope")
        if (
            type(anchor_fragment_ids) is not tuple
            or not 1 <= len(anchor_fragment_ids) <= 500
            or any(
                type(item) is not str or not item.startswith("fragment_")
                for item in anchor_fragment_ids
            )
            or len(set(anchor_fragment_ids)) != len(anchor_fragment_ids)
        ):
            raise TypeError(
                "anchor_fragment_ids must be a unique tuple of 1 to 500 SourceFragment IDs"
            )
        if scope.library_id != self.library_id:
            raise StructureAuthorizationError("StructureUnit resolution targets another Library")
        generation = self.load_generation(generation_id)
        self._validate_snapshot_scope(generation.corpus_snapshot_id, scope.snapshot_hash)
        try:
            authorization = self._repository.authorize_read(
                access_policy_id=access_policy_id,
                scope=scope,
            )
        except (
            AccessContractError,
            AccessPolicyNotFoundError,
            AuthorizationError,
            CollectionNotFoundError,
        ) as exc:
            raise StructureAuthorizationError(str(exc)) from exc
        token = authorization.compiled.token
        if (
            generation.access_policy_id != access_policy_id
            or generation.policy_hash != token.policy_hash
            or generation.scope_hash != token.exclusion_hash
            or generation.permitted_set_hash != token.permitted_set_hash
        ):
            raise StructureAuthorizationError(
                "StructureUnit generation differs from the current policy scope"
            )
        permitted_ids = {item.source_fragment_id for item in authorization.compiled.manifest.items}
        if not set(anchor_fragment_ids).issubset(permitted_ids):
            raise StructureAuthorizationError(
                "StructureUnit anchor is outside the permitted manifest"
            )
        unit_ids: set[str] = set()
        connection = self._repository._store.connection
        for offset in range(0, len(anchor_fragment_ids), 500):
            chunk = anchor_fragment_ids[offset : offset + 500]
            placeholders = ", ".join("?" for _ in chunk)
            unit_ids.update(
                str(row[0])
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT unit.structure_unit_id
                    FROM structure_units AS unit
                    JOIN structure_unit_members AS member
                      ON member.structure_unit_id = unit.structure_unit_id
                    WHERE unit.structure_unit_generation_id = ?
                      AND member.source_fragment_id IN ({placeholders})
                    """,
                    (generation.generation_id, *chunk),
                ).fetchall()
            )
        units = tuple(
            self.load_unit(unit_id)
            for unit_id in sorted(unit_ids, key=lambda item: item.encode("ascii"))
        )
        if any(
            not {member.source_fragment_id for member in unit.members}.issubset(permitted_ids)
            for unit in units
        ):
            raise StructureIntegrityError(
                "resolved StructureUnit crosses its generation permitted set"
            )
        return units

    def read_unit(
        self,
        *,
        structure_unit_id: str,
        corpus_snapshot_id: str,
        access_policy_id: str,
        scope: object,
    ) -> StructureUnitText:
        """Render one unit and persist exact member spans under current policy."""

        return self.read_units(
            structure_unit_ids=(structure_unit_id,),
            corpus_snapshot_id=corpus_snapshot_id,
            access_policy_id=access_policy_id,
            scope=scope,
        )[0]

    def read_units(
        self,
        *,
        structure_unit_ids: tuple[str, ...],
        corpus_snapshot_id: str,
        access_policy_id: str,
        scope: object,
    ) -> tuple[StructureUnitText, ...]:
        """Render an ordered unit batch under one freshly revalidated authorization.

        Every returned unit keeps its own exact, content-addressed read receipt.
        The batch boundary only prevents repeated whole-manifest authorization
        compilation and reads the union of requested member fragments once.
        """

        from dithyramba.access import RequestScope

        if not isinstance(scope, RequestScope):
            raise TypeError("scope must be a RequestScope")
        if (
            type(structure_unit_ids) is not tuple
            or not 1 <= len(structure_unit_ids) <= 500
            or any(
                type(item) is not str or not item.startswith("structure_unit_")
                for item in structure_unit_ids
            )
            or len(set(structure_unit_ids)) != len(structure_unit_ids)
        ):
            raise TypeError(
                "structure_unit_ids must be a unique tuple of 1 to 500 StructureUnit IDs"
            )
        if scope.library_id != self.library_id:
            raise StructureAuthorizationError("StructureUnit read targets another Library")
        units = tuple(self.load_unit(unit_id) for unit_id in structure_unit_ids)
        generations = {
            unit.generation_id: self.load_generation(unit.generation_id) for unit in units
        }
        if any(
            generation.corpus_snapshot_id != corpus_snapshot_id
            for generation in generations.values()
        ):
            raise StructureAuthorizationError(
                "StructureUnit read must use the generation CorpusSnapshot"
            )
        self._validate_snapshot_scope(corpus_snapshot_id, scope.snapshot_hash)
        requested_fragment_ids = tuple(
            dict.fromkeys(member.source_fragment_id for unit in units for member in unit.members)
        )
        try:
            authorization = self._repository.authorize_read(
                access_policy_id=access_policy_id,
                scope=scope,
            )
            fragments = self._repository.read_permitted_fragment_subset(
                authorization,
                requested_fragment_ids,
            )
        except (
            AccessContractError,
            AccessPolicyNotFoundError,
            AuthorizationError,
            CollectionNotFoundError,
        ) as exc:
            raise StructureAuthorizationError(str(exc)) from exc
        token = authorization.compiled.token
        fragment_by_id = {item.source_fragment_id: item for item in fragments}
        rendered_units = tuple(
            self._render_unit(
                unit=unit,
                corpus_snapshot_id=corpus_snapshot_id,
                access_policy_id=access_policy_id,
                policy_hash=token.policy_hash,
                scope_hash=token.exclusion_hash,
                permitted_set_hash=token.permitted_set_hash,
                fragment_by_id=fragment_by_id,
            )
            for unit in units
        )
        for rendered in rendered_units:
            self._persist_read_receipt(rendered.receipt)
        return rendered_units

    def _render_unit(
        self,
        *,
        unit: StructureUnit,
        corpus_snapshot_id: str,
        access_policy_id: str,
        policy_hash: str,
        scope_hash: str,
        permitted_set_hash: str,
        fragment_by_id: dict[str, SourceFragmentText],
    ) -> StructureUnitText:
        """Render one already-authorized unit without opening another read boundary."""

        fragments: list[SourceFragmentText] = []
        for member in unit.members:
            fragment = fragment_by_id.get(member.source_fragment_id)
            if not isinstance(fragment, SourceFragmentText):
                raise StructureIntegrityError(
                    "StructureUnit member is missing from its authorized batch read"
                )
            fragments.append(fragment)
        rendered_parts: list[str] = []
        receipt_items: list[StructureReadReceiptItem] = []
        cursor = 0
        for order, (member, fragment) in enumerate(zip(unit.members, fragments, strict=True)):
            if (
                member.source_fragment_id != fragment.source_fragment_id
                or member.text_sha256 != fragment.text_sha256
                or member.ordinal != fragment.ordinal
                or unit.source_id != fragment.source_id
                or unit.source_version_id != fragment.source_version_id
            ):
                raise StructureIntegrityError("StructureUnit member lineage changed before read")
            if not fragment.text:
                raise StructureIntegrityError("StructureUnit cannot render an empty fragment")
            if order:
                cursor += len(_DELIMITER)
            start = cursor
            cursor += len(fragment.text)
            receipt_items.append(
                StructureReadReceiptItem(
                    source_fragment_id=fragment.source_fragment_id,
                    member_order=order,
                    text_sha256=fragment.text_sha256,
                    rendered_start=start,
                    rendered_end=cursor,
                )
            )
            rendered_parts.append(fragment.text)
        rendered = _DELIMITER.join(rendered_parts)
        rendered_hash = sha256_hex(rendered.encode("utf-8"))
        receipt_payload: dict[str, object] = {
            "schema": "dithyramba.structure_read_receipt/1.0",
            "structure_unit_id": unit.structure_unit_id,
            "library_id": self.library_id,
            "corpus_snapshot_id": corpus_snapshot_id,
            "access_policy_id": access_policy_id,
            "policy_hash": policy_hash,
            "scope_hash": scope_hash,
            "permitted_set_hash": permitted_set_hash,
            "delimiter": _DELIMITER,
            "rendered_text_sha256": rendered_hash,
            "items": [item.payload() for item in receipt_items],
        }
        receipt_hash = canonical_sha256_hex(receipt_payload)
        receipt = StructureReadReceipt(
            receipt_id=canonical_content_id("structure_read", receipt_payload),
            structure_unit_id=unit.structure_unit_id,
            library_id=self.library_id,
            corpus_snapshot_id=corpus_snapshot_id,
            access_policy_id=access_policy_id,
            policy_hash=policy_hash,
            scope_hash=scope_hash,
            permitted_set_hash=permitted_set_hash,
            delimiter=_DELIMITER,
            rendered_text_sha256=rendered_hash,
            items=tuple(receipt_items),
        )
        if receipt.receipt_hash != receipt_hash:
            raise StructureIntegrityError("StructureUnit receipt hash is inconsistent")
        return StructureUnitText(unit, rendered, receipt)

    def _validate_snapshot_scope(
        self,
        corpus_snapshot_id: str,
        snapshot_hash: str,
    ) -> None:
        try:
            snapshot = self._repository.get_corpus_snapshot(corpus_snapshot_id)
        except CorpusSnapshotNotFoundError as exc:
            raise StructureAuthorizationError(
                "StructureUnit CorpusSnapshot does not exist in this Library"
            ) from exc
        if snapshot.manifest_hash != snapshot_hash:
            raise StructureAuthorizationError(
                "StructureUnit CorpusSnapshot ID/hash are inconsistent"
            )

    def load_read_receipt(self, receipt_id: str) -> StructureReadReceipt:
        with self._repository._store.transaction() as connection:
            row = connection.execute(
                """
                SELECT structure_unit_id, library_id, corpus_snapshot_id,
                       access_policy_id, policy_hash, scope_hash, permitted_set_hash,
                       delimiter, rendered_text_sha256, receipt_hash
                FROM structure_unit_read_receipts
                WHERE structure_read_receipt_id = ? AND library_id = ?
                """,
                (receipt_id, self.library_id),
            ).fetchone()
            if row is None:
                raise StructureNotFoundError("StructureUnit read receipt does not exist")
            item_rows = connection.execute(
                """
                SELECT source_fragment_id, member_order, text_sha256,
                       rendered_start, rendered_end
                FROM structure_unit_read_receipt_items
                WHERE structure_read_receipt_id = ?
                ORDER BY member_order
                """,
                (receipt_id,),
            ).fetchall()
        receipt = StructureReadReceipt(
            receipt_id=receipt_id,
            structure_unit_id=str(row[0]),
            library_id=str(row[1]),
            corpus_snapshot_id=str(row[2]),
            access_policy_id=str(row[3]),
            policy_hash=str(row[4]),
            scope_hash=str(row[5]),
            permitted_set_hash=str(row[6]),
            delimiter=str(row[7]),
            rendered_text_sha256=str(row[8]),
            items=tuple(
                StructureReadReceiptItem(
                    source_fragment_id=str(item[0]),
                    member_order=int(item[1]),
                    text_sha256=str(item[2]),
                    rendered_start=int(item[3]),
                    rendered_end=int(item[4]),
                )
                for item in item_rows
            ),
        )
        if receipt.receipt_hash != str(row[9]):
            raise StructureIntegrityError("persisted StructureUnit receipt hash mismatch")
        return receipt

    def _fragment_metadata(
        self, source_fragment_ids: tuple[str, ...]
    ) -> dict[str, _FragmentMetadata]:
        rows: list[sqlite3.Row] = []
        connection = self._repository._store.connection
        for offset in range(0, len(source_fragment_ids), 500):
            chunk = source_fragment_ids[offset : offset + 500]
            placeholders = ", ".join("?" for _ in chunk)
            rows.extend(
                connection.execute(
                    f"""
                    SELECT fragment.source_fragment_id, fragment.source_version_id,
                           version.source_id, fragment.ordinal, fragment.text_sha256
                    FROM source_fragments AS fragment
                    JOIN source_versions AS version
                      ON version.source_version_id = fragment.source_version_id
                    JOIN sources AS source ON source.source_id = version.source_id
                    WHERE source.library_id = ?
                      AND fragment.source_fragment_id IN ({placeholders})
                    """,
                    (self.library_id, *chunk),
                ).fetchall()
            )
        if {str(row[0]) for row in rows} != set(source_fragment_ids):
            raise StructureIntegrityError("StructureUnit fragment metadata is incomplete")
        return {
            str(row[0]): _FragmentMetadata(
                source_fragment_id=str(row[0]),
                source_version_id=str(row[1]),
                source_id=str(row[2]),
                ordinal=int(row[3]),
                text_sha256=str(row[4]),
            )
            for row in rows
        }

    def _permitted_structure_metadata(
        self,
        request: StructureGenerationRequest,
        authorization: AuthorizedRead,
    ) -> tuple[_PermittedStructureMetadata, ...]:
        """Read the exact permitted manifest's structural columns, never source text."""

        manifest = authorization.compiled.manifest
        expected = {item.source_fragment_id: item for item in manifest.items}
        if not expected:
            return ()
        try:
            with self._repository._store.transaction() as connection:
                current = self._repository._compile_db_access(
                    access_policy_id=request.access_policy_id,
                    scope=request.scope,
                )
                if current != authorization.compiled:
                    raise StructureAuthorizationError(
                        "authorized policy or snapshot metadata changed"
                    )
                rows: list[sqlite3.Row] = []
                identifiers = tuple(expected)
                for offset in range(0, len(identifiers), 500):
                    chunk = identifiers[offset : offset + 500]
                    placeholders = ", ".join("?" for _ in chunk)
                    rows.extend(
                        connection.execute(
                            f"""
                            SELECT fragment.source_fragment_id,
                                   fragment.source_version_id,
                                   version.source_id,
                                   fragment.ordinal,
                                   fragment.fragment_kind,
                                   fragment.source_address_json
                            FROM source_fragments AS fragment
                            JOIN source_versions AS version
                              ON version.source_version_id = fragment.source_version_id
                            JOIN sources AS source ON source.source_id = version.source_id
                            WHERE source.library_id = ?
                              AND fragment.source_fragment_id IN ({placeholders})
                            ORDER BY fragment.source_fragment_id
                            """,
                            (self.library_id, *chunk),
                        ).fetchall()
                    )
        except (
            AccessContractError,
            AccessPolicyNotFoundError,
            AuthorizationError,
            CollectionNotFoundError,
        ) as exc:
            raise StructureAuthorizationError(str(exc)) from exc
        if {str(row[0]) for row in rows} != set(expected):
            raise StructureIntegrityError(
                "permitted structural metadata does not resolve to the exact manifest"
            )

        result: list[_PermittedStructureMetadata] = []
        for row in rows:
            fragment_id = str(row[0])
            manifest_item = expected[fragment_id]
            if (
                str(row[1]) != manifest_item.source_version_id
                or str(row[2]) != manifest_item.source_id
            ):
                raise StructureIntegrityError(
                    "permitted manifest lineage differs from structural metadata"
                )
            try:
                fragment_kind = FragmentKind(str(row[4]))
            except ValueError as exc:
                raise StructureIntegrityError(
                    "permitted structural metadata has an invalid fragment kind"
                ) from exc
            address_kind, heading_path = _parse_structure_source_address(str(row[5]))
            result.append(
                _PermittedStructureMetadata(
                    source_fragment_id=fragment_id,
                    source_version_id=str(row[1]),
                    source_id=str(row[2]),
                    ordinal=int(row[3]),
                    fragment_kind=fragment_kind,
                    address_kind=address_kind,
                    heading_path=heading_path,
                )
            )
        return tuple(
            sorted(
                result,
                key=lambda item: (
                    item.source_id,
                    item.source_version_id,
                    item.ordinal,
                    item.source_fragment_id,
                ),
            )
        )

    def _prepare_markdown_inputs(
        self,
        metadata: tuple[_PermittedStructureMetadata, ...],
    ) -> tuple[tuple[MarkdownFragmentDescriptor, ...], tuple[StructureInputExclusion, ...]]:
        descriptors: list[MarkdownFragmentDescriptor] = []
        exclusions: list[StructureInputExclusion] = []
        by_version: dict[str, list[_PermittedStructureMetadata]] = {}
        for item in metadata:
            by_version.setdefault(item.source_version_id, []).append(item)

        for source_version_id in sorted(by_version, key=lambda value: value.encode("ascii")):
            values = tuple(
                sorted(
                    by_version[source_version_id],
                    key=lambda item: (item.ordinal, item.source_fragment_id),
                )
            )
            address_kinds = {item.address_kind for item in values}
            if len(address_kinds) != 1:
                raise StructureIntegrityError(
                    "one SourceVersion contains mixed SourceAddress kinds"
                )
            if address_kinds != {"markdown"}:
                exclusions.extend(
                    _structure_input_exclusion(
                        item,
                        StructureInputExclusionReason.UNSUPPORTED_SOURCE_ADDRESS,
                    )
                    for item in values
                )
                continue
            ordinals = tuple(item.ordinal for item in values)
            if ordinals != tuple(range(ordinals[0], ordinals[0] + len(ordinals))):
                exclusions.extend(
                    _structure_input_exclusion(
                        item,
                        StructureInputExclusionReason.NONCONTIGUOUS_PERMITTED_VERSION,
                    )
                    for item in values
                )
                continue
            try:
                descriptors.extend(
                    MarkdownFragmentDescriptor(
                        source_id=item.source_id,
                        source_version_id=item.source_version_id,
                        source_fragment_id=item.source_fragment_id,
                        ordinal=item.ordinal,
                        fragment_kind=item.fragment_kind,
                        heading_path=cast(tuple[str, ...], item.heading_path),
                    )
                    for item in values
                )
            except StructureContractError as exc:
                raise StructureIntegrityError(
                    "permitted Markdown descriptor metadata is invalid"
                ) from exc
        return tuple(descriptors), tuple(exclusions)

    def _prepare_unit(
        self,
        proposal: StructureUnitProposal,
        metadata: dict[str, _FragmentMetadata],
    ) -> tuple[str, dict[str, object], str, tuple[StructureUnitMember, ...]]:
        values = tuple(metadata[item] for item in proposal.source_fragment_ids)
        if (
            len({item.source_id for item in values}) != 1
            or len({item.source_version_id for item in values}) != 1
        ):
            raise StructureIntegrityError(
                "StructureUnit members must belong to one exact SourceVersion"
            )
        ordinals = tuple(item.ordinal for item in values)
        if ordinals != tuple(range(ordinals[0], ordinals[0] + len(ordinals))):
            raise StructureIntegrityError(
                "StructureUnit members must follow contiguous SourceFragment ordinals"
            )
        members = tuple(
            StructureUnitMember(
                source_fragment_id=item.source_fragment_id,
                member_order=order,
                ordinal=item.ordinal,
                text_sha256=item.text_sha256,
            )
            for order, item in enumerate(values)
        )
        boundary_payload = {
            "schema": "dithyramba.structure_boundary/1.0",
            "source_version_id": values[0].source_version_id,
            "first_ordinal": ordinals[0],
            "last_ordinal": ordinals[-1],
            "source_fragment_ids": list(proposal.source_fragment_ids),
        }
        boundary_hash = canonical_sha256_hex(boundary_payload)
        semantic: dict[str, object] = {
            "schema": "dithyramba.structure_unit/1.0",
            "source_id": values[0].source_id,
            "source_version_id": values[0].source_version_id,
            "kind": proposal.kind.value,
            "label": proposal.label,
            "members": [item.payload() for item in members],
            "boundary_hash": boundary_hash,
        }
        content_hash = canonical_sha256_hex(semantic)
        return (
            canonical_content_id("structure_unit", semantic),
            semantic,
            content_hash,
            members,
        )

    def _insert_units(
        self,
        connection: sqlite3.Connection,
        units: tuple[StructureUnit, ...],
        created_at: str,
    ) -> None:
        for offset in range(0, len(units), _PERSIST_BATCH_SIZE):
            batch = units[offset : offset + _PERSIST_BATCH_SIZE]
            connection.executemany(
                """
                INSERT INTO structure_units(
                    structure_unit_id, structure_unit_generation_id, source_id,
                    source_version_id, kind, label, member_count, first_ordinal,
                    last_ordinal, boundary_hash, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        unit.structure_unit_id,
                        unit.generation_id,
                        unit.source_id,
                        unit.source_version_id,
                        unit.kind.value,
                        unit.label,
                        len(unit.members),
                        unit.first_ordinal,
                        unit.last_ordinal,
                        unit.boundary_hash,
                        unit.content_hash,
                        created_at,
                    )
                    for unit in batch
                ),
            )
            connection.executemany(
                """
                INSERT INTO structure_unit_members(
                    structure_unit_id, source_fragment_id, member_order,
                    ordinal, text_sha256
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        unit.structure_unit_id,
                        item.source_fragment_id,
                        item.member_order,
                        item.ordinal,
                        item.text_sha256,
                    )
                    for unit in batch
                    for item in unit.members
                ),
            )

    def _load_generation(
        self, connection: sqlite3.Connection, generation_id: str
    ) -> StructureGeneration:
        row = connection.execute(
            """
            SELECT library_id, corpus_snapshot_id, access_policy_id, policy_hash,
                   scope_hash, permitted_set_hash, profile_id, profile_version,
                   unit_count, generation_hash
            FROM structure_unit_generations
            WHERE structure_unit_generation_id = ? AND library_id = ?
            """,
            (generation_id, self.library_id),
        ).fetchone()
        if row is None:
            raise StructureNotFoundError("StructureUnit generation does not exist")
        unit_rows = connection.execute(
            """
            SELECT structure_unit_id, source_id, source_version_id, kind, label,
                   member_count, first_ordinal, last_ordinal, boundary_hash,
                   content_hash
            FROM structure_units
            WHERE structure_unit_generation_id = ? ORDER BY structure_unit_id
            """,
            (generation_id,),
        ).fetchall()
        if len(unit_rows) != int(row[8]):
            raise StructureIntegrityError("StructureUnit generation count mismatch")
        member_rows = connection.execute(
            """
            SELECT member.structure_unit_id, member.source_fragment_id,
                   member.member_order, member.ordinal, member.text_sha256
            FROM structure_unit_members AS member
            JOIN structure_units AS unit
              ON unit.structure_unit_id = member.structure_unit_id
            WHERE unit.structure_unit_generation_id = ?
            ORDER BY member.structure_unit_id, member.member_order
            """,
            (generation_id,),
        ).fetchall()
        members_by_unit: dict[str, list[StructureUnitMember]] = {}
        for member_row in member_rows:
            members_by_unit.setdefault(str(member_row[0]), []).append(
                StructureUnitMember(
                    source_fragment_id=str(member_row[1]),
                    member_order=int(member_row[2]),
                    ordinal=int(member_row[3]),
                    text_sha256=str(member_row[4]),
                )
            )
        units: list[StructureUnit] = []
        for unit_row in unit_rows:
            unit_id = str(unit_row[0])
            members = tuple(members_by_unit.pop(unit_id, ()))
            if len(members) != int(unit_row[5]):
                raise StructureIntegrityError("StructureUnit member count mismatch")
            unit = StructureUnit(
                structure_unit_id=unit_id,
                generation_id=generation_id,
                source_id=str(unit_row[1]),
                source_version_id=str(unit_row[2]),
                kind=StructureUnitKind(str(unit_row[3])),
                label=None if unit_row[4] is None else str(unit_row[4]),
                members=members,
                boundary_hash=str(unit_row[8]),
                content_hash=str(unit_row[9]),
            )
            if unit.first_ordinal != int(unit_row[6]) or unit.last_ordinal != int(unit_row[7]):
                raise StructureIntegrityError("StructureUnit stored boundaries mismatch")
            if canonical_sha256_hex(unit.semantic_payload()) != unit.content_hash:
                raise StructureIntegrityError("StructureUnit content hash mismatch")
            expected_unit_id = canonical_content_id("structure_unit", unit.semantic_payload())
            if unit.structure_unit_id != expected_unit_id:
                raise StructureIntegrityError("StructureUnit content ID mismatch")
            units.append(unit)
        if members_by_unit:
            raise StructureIntegrityError("StructureUnit member references are inconsistent")
        generation = StructureGeneration(
            generation_id=generation_id,
            library_id=str(row[0]),
            corpus_snapshot_id=str(row[1]),
            access_policy_id=str(row[2]),
            policy_hash=str(row[3]),
            scope_hash=str(row[4]),
            permitted_set_hash=str(row[5]),
            profile_id=str(row[6]),
            profile_version=str(row[7]),
            units=tuple(units),
            generation_hash=str(row[9]),
        )
        if canonical_sha256_hex(generation.payload()) != generation.generation_hash:
            raise StructureIntegrityError("persisted StructureUnit generation hash mismatch")
        expected_id = canonical_content_id(
            "structure_generation",
            {
                "schema": "dithyramba.structure_generation_identity/1.0",
                "generation_hash": generation.generation_hash,
            },
        )
        if generation.generation_id != expected_id:
            raise StructureIntegrityError("persisted StructureUnit generation ID mismatch")
        return generation

    def _load_unit(self, connection: sqlite3.Connection, structure_unit_id: str) -> StructureUnit:
        row = connection.execute(
            """
            SELECT unit.structure_unit_generation_id, unit.source_id,
                   unit.source_version_id, unit.kind, unit.label, unit.member_count,
                   unit.first_ordinal, unit.last_ordinal, unit.boundary_hash,
                   unit.content_hash
            FROM structure_units AS unit
            JOIN structure_unit_generations AS generation
              ON generation.structure_unit_generation_id = unit.structure_unit_generation_id
            WHERE unit.structure_unit_id = ? AND generation.library_id = ?
            """,
            (structure_unit_id, self.library_id),
        ).fetchone()
        if row is None:
            raise StructureNotFoundError("StructureUnit does not exist")
        member_rows = connection.execute(
            """
            SELECT source_fragment_id, member_order, ordinal, text_sha256
            FROM structure_unit_members
            WHERE structure_unit_id = ? ORDER BY member_order
            """,
            (structure_unit_id,),
        ).fetchall()
        if len(member_rows) != int(row[5]):
            raise StructureIntegrityError("StructureUnit member count mismatch")
        unit = StructureUnit(
            structure_unit_id=structure_unit_id,
            generation_id=str(row[0]),
            source_id=str(row[1]),
            source_version_id=str(row[2]),
            kind=StructureUnitKind(str(row[3])),
            label=None if row[4] is None else str(row[4]),
            members=tuple(
                StructureUnitMember(
                    source_fragment_id=str(item[0]),
                    member_order=int(item[1]),
                    ordinal=int(item[2]),
                    text_sha256=str(item[3]),
                )
                for item in member_rows
            ),
            boundary_hash=str(row[8]),
            content_hash=str(row[9]),
        )
        if unit.first_ordinal != int(row[6]) or unit.last_ordinal != int(row[7]):
            raise StructureIntegrityError("StructureUnit stored boundaries mismatch")
        if canonical_sha256_hex(unit.semantic_payload()) != unit.content_hash:
            raise StructureIntegrityError("StructureUnit content hash mismatch")
        if (
            canonical_content_id("structure_unit", unit.semantic_payload())
            != unit.structure_unit_id
        ):
            raise StructureIntegrityError("StructureUnit content ID mismatch")
        return unit

    def _persist_read_receipt(self, receipt: StructureReadReceipt) -> None:
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO structure_unit_read_receipts(
                        structure_read_receipt_id, structure_unit_id, library_id,
                        corpus_snapshot_id, access_policy_id, policy_hash, scope_hash,
                        permitted_set_hash, delimiter, member_count,
                        rendered_text_sha256, receipt_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.receipt_id,
                        receipt.structure_unit_id,
                        receipt.library_id,
                        receipt.corpus_snapshot_id,
                        receipt.access_policy_id,
                        receipt.policy_hash,
                        receipt.scope_hash,
                        receipt.permitted_set_hash,
                        receipt.delimiter,
                        len(receipt.items),
                        receipt.rendered_text_sha256,
                        receipt.receipt_hash,
                        created_at,
                    ),
                )
                if cursor.rowcount == 1:
                    connection.executemany(
                        """
                        INSERT INTO structure_unit_read_receipt_items(
                            structure_read_receipt_id, source_fragment_id,
                            member_order, text_sha256, rendered_start, rendered_end
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            (
                                receipt.receipt_id,
                                item.source_fragment_id,
                                item.member_order,
                                item.text_sha256,
                                item.rendered_start,
                                item.rendered_end,
                            )
                            for item in receipt.items
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            raise StructureIntegrityError("StructureUnit receipt insert conflicted") from exc
        if self.load_read_receipt(receipt.receipt_id) != receipt:
            raise StructureIntegrityError(
                "persisted StructureUnit receipt conflicts with its content address"
            )


def _parse_structure_source_address(
    source_address_json: str,
) -> tuple[str, tuple[str, ...] | None]:
    """Validate canonical JSON and reconstruct only the supported Markdown address."""

    try:
        decoded: object = json.loads(source_address_json)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise StructureIntegrityError("SourceAddress is not valid JSON") from exc
    if type(decoded) is not dict:
        raise StructureIntegrityError("SourceAddress must be a JSON object")
    payload = cast(dict[str, object], decoded)
    try:
        canonical = canonical_json_bytes(payload).decode("utf-8")
    except (CanonicalizationError, UnicodeError) as exc:
        raise StructureIntegrityError("SourceAddress cannot be canonicalized") from exc
    if canonical != source_address_json:
        raise StructureIntegrityError("SourceAddress is not canonical JSON")
    kind = payload.get("kind")
    if (
        type(kind) is not str
        or kind != kind.strip()
        or not kind
        or len(kind) > 64
        or "\x00" in kind
    ):
        raise StructureIntegrityError("SourceAddress kind is invalid")
    if kind != "markdown":
        return kind, None

    expected_keys = {
        "schema",
        "kind",
        "heading_path",
        "line_start",
        "line_end",
        "char_start",
        "char_end",
    }
    if set(payload) != expected_keys:
        raise StructureIntegrityError("Markdown SourceAddress fields are invalid")
    heading_path = payload["heading_path"]
    if type(heading_path) is not list or any(type(item) is not str for item in heading_path):
        raise StructureIntegrityError("Markdown SourceAddress heading_path is invalid")
    try:
        address = MarkdownSourceAddress(
            heading_path=tuple(cast(list[str], heading_path)),
            line_start=_strict_structure_address_int(payload["line_start"], minimum=1),
            line_end=_strict_structure_address_int(payload["line_end"], minimum=1),
            char_start=_strict_structure_address_int(payload["char_start"], minimum=0),
            char_end=_strict_structure_address_int(payload["char_end"], minimum=0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StructureIntegrityError("Markdown SourceAddress is invalid") from exc
    if address.payload() != payload:
        raise StructureIntegrityError("Markdown SourceAddress differs from typed reconstruction")
    return "markdown", address.heading_path


def _strict_structure_address_int(value: object, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise StructureIntegrityError("Markdown SourceAddress integer field is invalid")
    return value


def _structure_input_exclusion(
    item: _PermittedStructureMetadata,
    reason: StructureInputExclusionReason,
) -> StructureInputExclusion:
    return StructureInputExclusion(
        source_id=item.source_id,
        source_version_id=item.source_version_id,
        source_fragment_id=item.source_fragment_id,
        ordinal=item.ordinal,
        fragment_kind=item.fragment_kind,
        address_kind=item.address_kind,
        reason=reason,
    )
