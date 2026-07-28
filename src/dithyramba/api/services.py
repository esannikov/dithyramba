"""Library-scoped packet projection and read-only SourceChip resolution."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, urlencode

from dithyramba.contracts import canonical_json_bytes
from dithyramba.persistence.errors import SourceNotFoundError, SourceVersionNotFoundError
from dithyramba.persistence.recall import SQLiteRecallBackend
from dithyramba.persistence.repository import LibraryRepository
from dithyramba.recall import EvidenceFragment, EvidencePacket

from .models import SourceChipResponse, address_response


class ViewerNotFoundError(RuntimeError):
    """The requested packet-backed SourceChip is absent from this Library."""


@dataclass(frozen=True, slots=True)
class PacketProjection:
    packet: EvidencePacket
    source_chips: tuple[SourceChipResponse, ...]


@dataclass(frozen=True, slots=True)
class ViewerProjection:
    packet: EvidencePacket
    fragment: EvidenceFragment
    source_chip: SourceChipResponse
    source_address_json: str


@dataclass(frozen=True, slots=True)
class FragmentProjection:
    """One SourceFragment proven to occur in a Library-scoped EvidencePacket."""

    packet: EvidencePacket
    fragment: EvidenceFragment
    source_chip: SourceChipResponse


class PacketViewService:
    """Resolve text only by loading a strictly closed persisted EvidencePacket."""

    def __init__(
        self,
        repository: LibraryRepository,
        recall_backend: SQLiteRecallBackend,
    ) -> None:
        self._repository = repository
        self._recall_backend = recall_backend

    def load_packet(self, evidence_packet_id: str) -> PacketProjection:
        packet = self._recall_backend.load_evidence_packet(evidence_packet_id)
        chips = tuple(
            self._chip_for_fragment(packet, fragment) for fragment in packet.source_fragments
        )
        return PacketProjection(packet=packet, source_chips=chips)

    def load_viewer(
        self,
        *,
        source_id: str,
        source_version_id: str,
        source_fragment_id: str,
        evidence_packet_id: str,
    ) -> ViewerProjection:
        projection = self.load_fragment(
            source_fragment_id,
            evidence_packet_id=evidence_packet_id,
        )
        chip = projection.source_chip
        if chip.source_id != source_id or chip.source_version_id != source_version_id:
            raise ViewerNotFoundError("packet-backed SourceChip does not exist")
        return ViewerProjection(
            packet=projection.packet,
            fragment=projection.fragment,
            source_chip=chip,
            source_address_json=canonical_json_bytes(
                projection.fragment.source_address.payload()
            ).decode("utf-8"),
        )

    def load_fragment(
        self,
        source_fragment_id: str,
        *,
        evidence_packet_id: str,
    ) -> FragmentProjection:
        """Load text only through a persisted packet, never by raw fragment SELECT."""

        packet_projection = self.load_packet(evidence_packet_id)
        matching_fragments = tuple(
            fragment
            for fragment in packet_projection.packet.source_fragments
            if fragment.source_fragment_id == source_fragment_id
        )
        matching_chips = tuple(
            chip
            for chip in packet_projection.source_chips
            if chip.source_fragment_id == source_fragment_id
        )
        if len(matching_fragments) != 1 or len(matching_chips) != 1:
            raise ViewerNotFoundError("packet-backed SourceChip does not exist")
        fragment = matching_fragments[0]
        chip = matching_chips[0]
        return FragmentProjection(
            packet=packet_projection.packet,
            fragment=fragment,
            source_chip=chip,
        )

    def _chip_for_fragment(
        self,
        packet: EvidencePacket,
        fragment: EvidenceFragment,
    ) -> SourceChipResponse:
        packet_matches = tuple(
            item
            for item in packet.source_fragments
            if item.source_fragment_id == fragment.source_fragment_id
        )
        if not packet_matches:
            raise ViewerNotFoundError("packet fragment does not resolve in this Library")
        if len(packet_matches) != 1 or packet_matches[0] != fragment:
            raise ViewerNotFoundError("packet fragment provenance is inconsistent")
        try:
            source_version = self._repository.get_source_version(fragment.source_version_id)
            matching_metadata = tuple(
                item
                for item in self._repository.list_source_fragments(fragment.source_version_id)
                if item.source_fragment_id == fragment.source_fragment_id
            )
        except (SourceNotFoundError, SourceVersionNotFoundError):
            raise ViewerNotFoundError("packet fragment does not resolve in this Library") from None
        if len(matching_metadata) != 1:
            raise ViewerNotFoundError("packet fragment does not resolve in this Library")
        metadata = matching_metadata[0]
        source_id = source_version.source_id
        stored_address = metadata.source_address_json
        expected_address = canonical_json_bytes(fragment.source_address.payload()).decode("utf-8")
        if (
            metadata.source_version_id != fragment.source_version_id
            or metadata.text_sha256 != fragment.text_sha256
            or stored_address != expected_address
        ):
            raise ViewerNotFoundError("packet fragment provenance is inconsistent")
        source = self._repository.get_source(source_id)
        if (
            fragment.source_family_id is not None
            and source.source_family_id != fragment.source_family_id
        ):
            raise ViewerNotFoundError("packet fragment lineage is inconsistent")
        viewer_path = (
            f"/sources/{quote(source_id, safe='')}/versions/"
            f"{quote(fragment.source_version_id, safe='')}?"
            + urlencode(
                {
                    "packet": packet.evidence_packet_id,
                    "fragment": fragment.source_fragment_id,
                }
            )
        )
        return SourceChipResponse(
            evidence_packet_id=packet.evidence_packet_id,
            packet_hash=packet.packet_hash,
            source_id=source_id,
            source_family_id=source.source_family_id,
            root_source_id=source.root_source_id,
            family_role=source.family_role.value,
            source_version_id=fragment.source_version_id,
            source_fragment_id=fragment.source_fragment_id,
            text_sha256=fragment.text_sha256,
            source_address=address_response(fragment.source_address),
            viewer_url=viewer_path,
        )
