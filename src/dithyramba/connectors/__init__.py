"""Bounded projections for external sources and downstream consumers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .books import (
        BOOK_PROJECTION_SCHEMA,
        DEFAULT_BOOK_PROFILE,
        LARGE_BOOK_PROFILE,
        BookMediaType,
        BookProjection,
        BookProjectionProfile,
        BookProjectionStatus,
        BookSource,
        BookSourceMember,
        CatalogProjectionSummary,
        project_book,
        project_book_catalog,
        project_book_path,
    )
    from .compact_packets import (
        CONNECTOR_PACKET_PROFILE,
        CONNECTOR_PACKET_SCHEMA,
        CompactConnectorPacket,
        ConnectorEvidence,
        ConnectorGap,
        ConnectorPacketBudget,
        ConnectorPacketError,
        ConnectorPacketRequest,
        ConnectorPacketStatus,
        ConnectorRole,
        ConnectorRoleAssignment,
        ConnectorSource,
        ConnectorSourceAddress,
        project_compact_memory_packet,
    )

__all__ = [
    "BOOK_PROJECTION_SCHEMA",
    "CONNECTOR_PACKET_PROFILE",
    "CONNECTOR_PACKET_SCHEMA",
    "DEFAULT_BOOK_PROFILE",
    "LARGE_BOOK_PROFILE",
    "BookMediaType",
    "BookProjection",
    "BookProjectionProfile",
    "BookProjectionStatus",
    "BookSource",
    "BookSourceMember",
    "CatalogProjectionSummary",
    "CompactConnectorPacket",
    "ConnectorEvidence",
    "ConnectorGap",
    "ConnectorPacketBudget",
    "ConnectorPacketError",
    "ConnectorPacketRequest",
    "ConnectorPacketStatus",
    "ConnectorRole",
    "ConnectorRoleAssignment",
    "ConnectorSource",
    "ConnectorSourceAddress",
    "project_book",
    "project_book_catalog",
    "project_book_path",
    "project_compact_memory_packet",
]

_COMPACT_PACKET_EXPORTS = frozenset(
    {
        "CONNECTOR_PACKET_PROFILE",
        "CONNECTOR_PACKET_SCHEMA",
        "CompactConnectorPacket",
        "ConnectorEvidence",
        "ConnectorGap",
        "ConnectorPacketBudget",
        "ConnectorPacketError",
        "ConnectorPacketRequest",
        "ConnectorPacketStatus",
        "ConnectorRole",
        "ConnectorRoleAssignment",
        "ConnectorSource",
        "ConnectorSourceAddress",
        "project_compact_memory_packet",
    }
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module_name = ".compact_packets" if name in _COMPACT_PACKET_EXPORTS else ".books"
    return getattr(import_module(module_name, __name__), name)
