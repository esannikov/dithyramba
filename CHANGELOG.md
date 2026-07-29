# Changelog

This file records package changes. Detailed experiment metrics and decision
history belong in dated evaluation reports, not in the product changelog.

## Unreleased — 2026-07-28

### Added

- Schema v10 `CorpusReadSet` storage and `RecallService.recall_batch(...)`:
  several questions in one exact scope reuse one protected corpus read and one
  FTS session while retaining separate request, packet, receipt, and review
  identities. Public `ReadReceipt/1.0` and `EvidencePacket/1.0` stay unchanged.
- An explicit bounded `large-document` ingest profile for book-length local
  sources; the conservative default profile remains unchanged.
- Live library-only adaptive recall route with bounded FTS50/100 discovery,
  deterministic lexical repair, q0-anchored Harrier ranking, body-proof
  admission, Wide `EvidenceCoverageGate`, and optional QueryCloud q1/q2.
- `AdaptiveRecallResult/2.2` with stage, selection, Gate-scan, and matched-proof
  receipts.
- Content-addressed `FragmentTextProjectionReceipt/1.0`,
  `ExternalReferenceMap/1.0`, and `ProofMetadataManifest/1.0` compatibility
  contracts with fail-closed scope, coverage, collision, and tamper checks.
- Causal regression coverage for forged receipts, prior proof preservation,
  bounded candidate selection, boundary projection, neighbour budgets, and
  result closure.
- Concise user, reference, explanation, and development documentation.
- Apache License 2.0 with SPDX package and citation metadata.
- A compact, rights-safe public verification replay and a repository guide that
  separates installed runtime code from contributor-only support files.

### Changed

- Harrier input validation now uses an explicit two-million-character safety
  bound and the exact no-truncation token audit as the decisive model limit,
  avoiding false rejection of valid long passages.
- Model provisioning can resolve the pinned Hugging Face CLI from the active
  project environment even when that executable is not on the shell `PATH`.
- Coverage output now uses two decimal places so a value below 95% cannot pass
  by integer display rounding.
- The guarded full update checks the unrounded JSON coverage value.
- Granite and Qwen remain outside the active retrieval target; FTS is the
  default and Harrier is a bounded library-only reranker.
- Case-specific Artists, P7, and model-comparison harnesses are no longer part
  of the public release tree; their dated results remain in the evaluation
  record with explicit limitations.

### Not yet public surface

- Binding compatibility artifacts into adaptive execution, adaptive
  persistence, packet v2, cold replay, and opt-in CLI/HTTP routing remain open.
- The workspace has no public remote, release, or compatibility promise.

## 0.1.0rc0 — local pre-alpha foundation

- Implemented isolated Libraries and Collections, immutable default-deny
  AccessPolicies, CorpusSnapshots, Markdown/TXT/PDF ingest, exact source
  addressing, FTS recall, `EvidencePacket/1.0`, Read/Access/Retrieval receipts,
  append-only review, backup/restore, and a pinned loopback API.
- Added meaning candidates, Voices, Entities, Concepts, ConceptMeanings,
  TimeContexts, typed Relations, StructureUnits, ReadingRoom, Research Atlas,
  and compact Connector packets behind scoped contracts.
- Added optional semantic/vector/reranking infrastructure and development-only
  evaluation harnesses without changing the stable FTS default.
