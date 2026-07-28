"""Content-addressed provenance and canonical coverage contracts."""

from .blobs import BlobStore
from .coverage import CoverageDraft, OmissionDraft, build_ingest_coverage
from .errors import (
    BlobIntegrityError,
    BlobStoreError,
    ProvenanceError,
    SourceHistoryConflictError,
    SourceLineageIntegrityError,
)
from .models import (
    BlobRecord,
    CollectionMembershipRecord,
    CoverageReportRecord,
    IngestBatchResult,
    IngestDisposition,
    IngestInputOutcome,
    OmissionRecord,
    PersistedSourceOutcome,
    ProcessingRunRecord,
    ProcessingRunStatus,
    PromotedBlob,
    SourceFamilyRole,
    SourceFragmentRecord,
    SourceRecord,
    SourceVersionRecord,
    TerminalInputOutcome,
)

__all__ = [
    "BlobIntegrityError",
    "BlobRecord",
    "BlobStore",
    "BlobStoreError",
    "CollectionMembershipRecord",
    "CoverageDraft",
    "CoverageReportRecord",
    "IngestBatchResult",
    "IngestDisposition",
    "IngestInputOutcome",
    "OmissionDraft",
    "OmissionRecord",
    "PersistedSourceOutcome",
    "ProcessingRunRecord",
    "ProcessingRunStatus",
    "PromotedBlob",
    "ProvenanceError",
    "SourceFamilyRole",
    "SourceFragmentRecord",
    "SourceHistoryConflictError",
    "SourceLineageIntegrityError",
    "SourceRecord",
    "SourceVersionRecord",
    "TerminalInputOutcome",
    "build_ingest_coverage",
]
