# Changelog

This file records package changes. Detailed experiment metrics and decision
history belong in dated evaluation reports, not in the product changelog.

## Unreleased

### Added

- Durable 0.2 `ResearchSession` journals with compact least-context agent turns,
  exact command retry receipts, and cold-reopen state reconstruction.
- Process-local `RecallScopeSession` reuse: one authorized fragment read-set and
  in-memory FTS index can serve later questions in the same exact session scope
  while every question retains its own durable packet and audit trail.
- Thin newline-delimited JSON-RPC stdio MCP over the existing Python facade,
  with six bounded research tools and no human acceptance or promotion tool.
- GET-only Session Lens for briefs, chronological questions, packet-backed
  evidence, drafts, gaps, rejected paths, and exact source inspection.
- `AgentEvidencePacket/1.1` human-readable source references beside every exact
  selected fragment, with replay support for 1.0 development receipts.

- `AnswerProjection/1.0` and `PropositionCoverageGate`: an optional
  display-governance layer that partitions the exact final answer into facts,
  bounded syntheses, disclosed hypotheses, research questions, and framing.
- Schema v12 append-only `answer_projections` and
  `answer_projection_receipts`, with canonical reopen, corruption checks,
  Library isolation, and audit events.
- Sparse `AtlasTraceSpan` bindings for Research Atlas answers and hypotheses.
  Lens colours only exact source-traceable phrases; clicking one activates its
  evidence chip and exact-passage inspector.
- Falsifiable hypothesis probes with an explicit test question, falsifier, and
  next evidence, while retaining exact premise claim routes.
- Deterministic fail-closed checks for complete span coverage, stale answer and
  receipt bindings, unsupported overclaim, uncertain roles, and policy drift.

### Changed

- Rebuilt the public architecture figure and product documentation around the
  two-plane 0.2 design: durable evidence core plus bounded interactive session
  loop. Updated the clean-install and wheel-closure checks to require schema
  v13 and the packaged research-session migration.
- A six-session Mars screen measured a 23.85% mean warm-turn reduction while
  preserving 17/17 comparable top-10 result lists, 6/6 exact retries, and 6/6
  cold reopens with zero model tokens. Automatic relation-aware repair remains
  out of production after a one-case challenger repaired only one of two query
  variants.
- Large-scope session completion now validates the shared immutable corpus
  closure once per explicit live scope and revalidates selected stored
  fragments on later questions. Modern Session Lens events reopen compact
  command receipts and direct fragment metadata instead of reconstructing a
  corpus-wide `ReadReceipt` to render a page.
- Session Lens now distinguishes the source title from the exact file/URI
  locator. It no longer labels a book or file title as the person who asserted
  the passage, and it never exposes a full local filesystem path in the human
  view. The same safe locator is now present in the stable JSON projection;
  web locators retain their host and path while credentials, query strings,
  and fragments stay hidden. A source without a stored title falls back to
  this safe locator instead of exposing its raw canonical URI.

- Simplified the research route to local FTS, bounded lexical expansion,
  conditional QueryCloud, exact proof admission, and deterministic evidence
  gates.
- Reworked Lens into a sparse reading workspace: one central column, compact
  route navigation, an inspector opened only by a selected exact trace or
  source, and paginated full-corpus sources.
- Added strict migration of legacy Atlas projections to the current schema
  without rewriting the original artifact.
- Replaced the proposed whole-answer equivalence rule with proposition-level
  coverage. Factual spans remain strict; exploratory spans may be useful and
  expressive without being promoted to fact.
- Documented the default profile as lexical-first, evidence-first, and
  embedding-optional. No claim is made that lexical retrieval wins every
  semantic-recall case.
- Added a public Mars IdeaTrace-24 evaluation card that separates the frozen
  reasoning test from the earlier 53,747-unit retrieval calibration.
- Preserved byte-compatible `CompactMemoryPacket/1.0` and `/1.1` identities by
  omitting empty trace metadata and rejecting non-empty Lens spans when a
  legacy connector cannot preserve them.

### Removed

- Removed Harrier from runtime architecture, provisioning, public interfaces,
  and current documentation. Historical comparison results remain labelled as
  retired evaluation evidence; model weights, caches, and case artifacts are
  not part of the package.

## 0.1.0rc1 — 2026-07-31

### Added

- `IdeaTrace/1.0`, a bounded public reasoning artifact made of short
  statements, named operations, explicit premises, concise warrants,
  qualifiers, exact claim-evidence bindings, and open gaps. It never stores or
  requests private chain-of-thought text.
- Deterministic `ReasoningClosureGate` and
  `ReasoningClosureResult/1.0`. Closure checks trace topology, stale bindings,
  semantic claim verdicts, and operation policy without a model call. A pass is
  `review_eligible`, not an automatic promotion.
- Schema v11 append-only `idea_traces` and `reasoning_closure_results` storage,
  exact reopen/cold replay, corruption checks, and audit outbox events.
- `dithyramba reasoning-check` for provider-free verification of one trace,
  case set, and claim-evidence result.
- Scoped `CandidateOntologyManifest/1.0` and GET-only Concept Lens as the
  smaller replacement for the rejected global-cartography experiment.

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

- Replaced the optional `cartography` dependency group with `ontology`; global
  automatic corpus maps are no longer presented as a product module.
- Candidate ontology and evidence-grounded reasoning remain separate from the
  accepted truth path. Both can orient or synthesize, but neither can create a
  human `ReviewDecision` or grant itself evidential authority.

- Migration verification accepts the single audited private pre-release
  checksum of `0003_recall_run_artifacts.sql` while keeping the published
  source-preview checksum canonical; every other checksum drift still fails
  closed and the schema fingerprint remains mandatory.
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
- The source preview is public; no stable tag or GitHub Release has been
  declared before independent fresh-install acceptance.

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
