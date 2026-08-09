# Dithyramba reference

This reference describes `1.0.0rc3`, the current v1 release candidate. The
command itself is authoritative for exact
options and defaults:

```bash
uv run dithyramba <command> --help
```

## Runtime requirements

| Item | Requirement |
|---|---|
| Python | 3.11 or newer |
| SQLite | FTS5 enabled |
| Release-qualified hosts | Linux and macOS 13+ through CI and clean-install checks |
| Not yet qualified | Windows; current filesystem contracts are POSIX-specific |
| Package environment | `uv` for development; wheel build uses Hatchling |
| Base runtime | no model, GPU, external service, or Docker required |
| Embedding or reranking runtime | not included in v1 |
| Optional ontology extra | `numpy>=2,<3`; `scikit-learn>=1.8,<2` |
| Current database schema | one v1 baseline (`0001_v1.sql`) |
| Package version | `1.0.0rc3` |

## Language scope

Core source preservation and SQLite FTS are Unicode-capable. Release-qualified
multilingual fixtures and candidate-hygiene rules cover Ukrainian and English;
other Latin- and Cyrillic-script languages are best-effort until a dated corpus
evaluation is added. The experimental Candidate Ontology accepts English only.
Corpora written outside Latin and Cyrillic scripts are outside the current
product-evaluation scope.

## Command inventory

### General

```text
dithyramba --version
dithyramba about
dithyramba doctor --library <library-id> --data-home <absolute-path>
```

`doctor` is an alias for `library doctor` and requires an initialized Library.
Use `dithyramba about` or the public demo before any Library exists.

### Library and Collection

```text
dithyramba library init
dithyramba library list
dithyramba library describe
dithyramba library doctor
dithyramba collection add
dithyramba collection list
dithyramba collection freeze
```

`library describe --json` returns one text-free operator overview: verified
Library identity and health, persisted-object counts, Collections,
AccessPolicies, immutable snapshot summaries, and recent ProcessingRuns. It is
the preferred first call for a human or agent resuming work. The command does
not read source text, contact external services, mutate state, or claim that a
successful processing run establishes scholarly correctness. Backup bundles
remain external portable artifacts and are therefore reported as externally
tracked rather than guessed from the live database.

### Policy, ingest, and sources

```text
dithyramba access-policy create
dithyramba access-policy show
dithyramba access-policy check
dithyramba index
dithyramba source add
dithyramba source list
dithyramba source versions
```

`index` and `source add` accept `--parser-profile default` or
`--parser-profile large-document`. The default remains deliberately small and
conservative. `large-document` is an explicit bounded profile for book-length
PDF, Markdown, and text inputs: up to 512 MiB per file, 2,000 PDF pages,
20 million extracted characters, 180 seconds, and 2,560 MiB worker RSS. It does
not remove parser limits or change source identity. A `SourceVersion` is unique
for its source bytes and parser profile together: a changed profile is parsed
once into a new immutable representation, while the next identical run reuses
that exact representation.

### Recall and review

```text
dithyramba recall
dithyramba packet inspect
dithyramba packet replay
dithyramba packet read-receipt
dithyramba review decide
dithyramba review queue
```

### Portability and local views

```text
dithyramba backup
dithyramba restore
dithyramba serve
dithyramba lens library
dithyramba lens session
dithyramba lens atlas
dithyramba lens concepts
dithyramba lens flow
dithyramba mcp
dithyramba reasoning-check
```

`Lens` is the single researcher-facing CLI surface. Its modes retain separate
validated input contracts rather than pretending that a live Library, a durable
session, and an immutable Atlas are the same artifact.

`lens library` requires `--library`, `--snapshot`, `--access-policy`, at least
one `--collection`, `--purpose`, and `--data-home`. It starts a separate GET-only
loopback server. `lens atlas`, `lens flow`, `lens concepts`, and `lens session`
also start read-only loopback views. Session mode requires one existing
`ResearchSession` ID and exposes `/projection.json` beside its human journal.

There is no v1 in-place legacy migration command. A pre-v1 Library fails closed
before connection-profile writes; rebuild from its read-only sources and use the
matching pre-v1 release only if review or session artifacts must be exported.

`mcp` runs a stdio server over one existing Library. It reserves stdout for
newline-delimited JSON-RPC and exposes `open_session`, `recall`,
`session_context`, `prepare_answer`, `record_draft`, `record_gap`, and
`reject_path`. `prepare_answer` applies deterministic candidate hygiene,
evaluates an explicit `EvidenceGateSpec`, and may run a bounded search inside
already found Sources before returning `answer`, `gap`, or `blocked`.
`record_draft` requires an exactly replayable `answer` preparation. The MCP
surface has no human acceptance, decision, promotion, deletion, or
session-closure tool.

The repository also contains a thin Codex skill in `skills/dithyramba/`. It
orchestrates the CLI, MCP adapter, and Lens while preserving the same evidence
and human-review boundaries. It is a source-repository integration resource,
not part of the runtime wheel and not a second implementation of the memory.

`reasoning-check` consumes one absolute-path `IdeaTrace`, exact claim-evidence
case set, and semantic entailment result. It makes no provider call and emits a
canonical closure receipt. A non-passed closure exits non-zero.

## Loopback HTTP surface

```bash
dithyramba serve \
  --library <library-id> \
  --data-home <absolute-path> \
  --port 8347
```

The service binds to `127.0.0.1`, pins one existing Library, disables OpenAPI
and interactive documentation routes, and prints a fresh bearer token to
stderr. Every mutating request requires that token and the exact loopback
`Origin`; read requests remain subject to Library, packet, policy, and snapshot
checks.

| Method | Route group | Purpose |
|---|---|---|
| `GET` | `/health`, `/v1/libraries` | verify and identify the pinned Library |
| `GET`/`POST` | `/v1/libraries/{library_id}/collections` | list or create Collections |
| `POST` | `/v1/libraries/{library_id}/collections/{collection_id}/snapshots` | freeze a Collection snapshot |
| `POST` | `/v1/sources` | ingest one source |
| `GET` | `/v1/sources/{source_id}/versions` | list immutable source versions |
| `POST` | `/v1/recall` | run the persisted FTS route |
| `GET`/`POST` | `/v1/evidence-packets/...` | inspect, replay, or export a packet |
| `GET`/`POST` | `/v1/review-decisions...` | inspect or append scoped review decisions |

Library creation and migration remain CLI-only. The HTTP models are executable
v1 release-candidate contracts. Their declared schema identifiers keep their
documented meaning, while transport details may still change before stable
`1.0.0`.

## Core input formats

| Suffix | Core ingest | Address |
|---|---|---|
| `.md`, `.markdown` | yes | heading and character range |
| `.txt` | yes | character range |
| `.pdf` | yes | page and normalized bounding box |
| `.epub` | book Connector only | spine item and local range |
| `.fb2` | book Connector only | XML structure and local range |

The book Connector returns readable Markdown plus a canonical sidecar with
typed units and exact local ranges. The `books/1.2` profile preserves common
EPUB and FB2 structural blocks; an unknown text-bearing block is disclosed as
partial projection coverage instead of being silently omitted. The Connector
does not promote the projection to source evidence by itself.

This book-projection `partial` means representational loss recorded in the
Connector receipt. It is separate from the evidence gate's `partial`, which
means that only some declared evidence roles are covered.

## Versioned core contracts

These schemas are versioned within the v1 release candidate. A schema identifier
keeps its documented byte meaning; a meaning change requires a new identifier.
The complete public compatibility promise will be frozen at stable `1.0.0`.

| Contract | Schema | Purpose |
|---|---|---|
| `AccessPolicySnapshot` | `dithyramba.access_policy_snapshot/1.0` | immutable default-deny rules |
| `CorpusSnapshot` | `dithyramba.corpus_snapshot_manifest/1.0` | exact frozen source-version scope |
| `QueryRequest` | `dithyramba.query_request/1.0` | question, scope, policy, exclusions, budget |
| `ReadReceipt` | `dithyramba.read_receipt/1.0` | exact protected fragments actually read |
| `AccessReceipt` | `dithyramba.access_receipt/1.0` | policy decision used by a query |
| `RetrievalReceipt` | `dithyramba.retrieval_receipt/1.0` | local retrieval identity and profile |
| `CoverageReport` | `dithyramba.coverage_report/1.0` | processed, skipped, failed, omitted |
| `EvidencePacket` | `dithyramba.evidence_packet/1.0` | persisted bounded FTS result |
| `ReviewDecision` | `dithyramba.review_decision/1.0` | append-only scoped human decision |
| `BackupBundle` | `dithyramba.backup_bundle/1.0` | portable hash-closed Library backup |
| `IdeaTrace` | `dithyramba.idea_trace/1.0` | short public reasoning candidate over exact claim-evidence cases |
| `ReasoningClosureResult` | `dithyramba.reasoning_closure/1.0` | deterministic structural closure; review eligibility only |
| `ResearchSessionBrief` | `dithyramba.research_session_brief/1.0` | bounded purpose, success criteria, and limits |
| `ResearchSession` | `dithyramba.research_session/1.0` | immutable Library/snapshot/policy scope |
| `SessionEvent` | `dithyramba.session_event/1.0` | typed append-only research-journal step |

Interactive transport adds compact, derived schemas rather than changing the
durable `EvidencePacket/1.0`:

| View | Schema | Boundary |
|---|---|---|
| `AgentSessionContext` | `dithyramba.agent_session_context/1.0` | bounded recent journal state plus explicit omission counts |
| `AgentEvidencePacket` | `dithyramba.agent_evidence_packet/1.2` | selected exact fragments, readable source references, candidate-only admission state, source-diversity diagnostics, coverage, and IDs/hashes of the full audit receipts |
| `AgentResearchTurn` | `dithyramba.agent_research_turn/1.2` | one compact evidence packet plus the exact session context that follows it |
| `AgentSourceDrilldown` | `dithyramba.agent_source_drilldown/1.0` | bounded source-local retrieval query, source IDs, result hash, and selected fragments |
| `AgentAnswerPreparation` | `dithyramba.agent_answer_preparation/1.1` | exact candidates, hygiene decisions, gate result, optional drilldown, response mode, and preparation profile required before a draft |

`AgentEvidencePacket/1.0` and `/1.1` remain readable for development-session
replay. Version 1.0 lacks readable source references; version 1.1 has source
references but predates the explicit `retrieved_candidates` admission state and
source-diversity diagnostics. The full materialized `ReadReceipt` stays local
in every version and is reopened only through the strict audit route.

`AgentResearchTurn/1.1` and `AgentAnswerPreparation/1.0` also remain readable.
Preparation `/1.0` retains the original meaningful-token drilldown behavior;
`/1.1` declares `candidate_hygiene_v1_literal_drilldown_v2` and therefore keeps
short explicit anchors such as `AI` and `ШІ`. MCP `tools/list` exposes an
executable `outputSchema` for every tool. `scripts/contract_receipt.py` records
those schemas, the CLI command tree, SQLite fingerprint, and checked local
documentation links against one exact commit.

The v1 baseline includes an internal append-only `CorpusReadSet`: one exact protected
fragment manifest can be shared by several recall requests over the same
Library, snapshot, policy, Collections, and purpose. Public `ReadReceipt/1.0`
and `EvidencePacket/1.0` payloads remain unchanged.

The v1 baseline includes append-only `idea_traces` and
`reasoning_closure_results`. It stores only canonical public trace artifacts,
not private chain-of-thought text. A passed closure does not create a human
`ReviewDecision` and does not promote a claim into accepted memory.

It also includes append-only `answer_projections` and
`answer_projection_receipts`. The first stores exact role-labelled public
prose; the second stores the semantic judgment bound to that projection.
`PropositionCoverageResult` is deterministic and rebuildable, so it is not a
separate durable table.

Changing the meaning or required fields of one of these schemas requires a
new version. Applied migrations remain immutable.

## Implemented library-only contracts

| Contract | Schema | Persistence |
|---|---|---|
| `EvidenceRequirement` | `dithyramba.evidence_requirement/1.0` | caller-owned input |
| `EvidenceGateSpec` | `dithyramba.evidence_gate_spec/1.0` | caller-owned input |
| `EvidenceCoverageResult` | `dithyramba.evidence_coverage_result/1.0` | in-memory |
| `AnswerProjection` | `dithyramba.answer_projection/1.0` | append-only canonical JSON in the v1 baseline |
| `AnswerProjectionJudgmentReceipt` | `dithyramba.answer_projection_receipt/1.0` | append-only canonical JSON in the v1 baseline |
| `PropositionCoverageResult` | typed result | in-memory; display eligibility only |
| `RouteCandidateReceipt` | `dithyramba.route_candidate_receipt/1.0` | library artifact |
| `CompactMemoryPacket` | `dithyramba.compact_memory_packet/1.1` | library artifact |
| `CompactConnectorPacket` | `dithyramba.compact_connector_packet/1.0` | library artifact |

These contracts are executable and tested. Library-only call shapes may still
change before stable `1.0.0`; persisted schema meaning may not.

`AnswerProjection` never changes `ResearchAnswer/1.0`. It binds exact character
spans of the displayed answer to accepted claims and labels them as fact,
synthesis, hypothesis, question, or framing. Hypotheses require a falsifiable
probe. `PropositionCoverageGate` is deterministic and provider-free; a separate
semantic receipt is required to verify that the prose actually matches its
declared roles.

Research Atlas may additionally carry sparse `AtlasTraceSpan` view records for
question answers and hypothesis syntheses. A span stores exact character
offsets, an epistemic display kind, and one or more Atlas evidence IDs. Lens
uses those bindings to highlight only traceable phrases and to select the
corresponding source evidence. Trace spans are part of the Atlas projection,
not a new evidence or acceptance contract.

## Experimental scoped-ontology types

| Type | Schema or identity | Current boundary |
|---|---|---|
| `OntologyConfig` | `dithyramba.ontology_config/1.0` | frozen local extraction profile |
| `OntologyEvidence` | content-addressed exact fragment support | source/version/address/text closure |
| `OntologyConcept` | member of `CandidateOntologyManifest/1.0` | `scope_anchor` or `emergent`; always candidate |
| `OntologyRelation` | `co_occurs_with` only | exact shared fragments; never causation |
| `CandidateOntologyManifest` | `dithyramba.candidate_ontology/1.0` | rebuildable scoped navigation projection |

`build_candidate_ontology(...)` consumes one bounded, already-authorized
neighbourhood. The optional `ontology` extra supplies deterministic TF-IDF/NMF
extraction. `dithyramba lens concepts --projection /absolute/ontology.json`
opens the validated result on loopback. `--presentation
/absolute/presentation.json` optionally adds an ontology-bound human view with
one title, question, summary, and entry concept per cluster. The command is
GET-only and does not promote candidates or mutate a Library.

## Fragment compatibility contracts

| Contract | Schema | Current boundary |
|---|---|---|
| `FragmentTextProjectionReceipt` | `dithyramba.fragment_text_projection_receipt/1.0` | tested library artifact; not run-bound |
| `ExternalReferenceMap` | `dithyramba.external_reference_map/1.0` | tested library artifact; not run-bound |
| `ProofMetadataManifest` | `dithyramba.proof_metadata_manifest/1.0` | tested library artifact; not persisted |

These contracts are present and tested in the v1 release candidate. They fail closed
when an external fragment projection no longer matches Dithyramba's exact
source identity or proof metadata.

## Default budgets

### Stable FTS request

| Field | Default | Bounds |
|---|---:|---:|
| `max_candidates` | 100 | 1–500 |
| `max_source_fragments` | 30 | 1–100 and not above candidates |

The Python `RecallService.recall_batch(...)` route accepts several requests for
one exact scope, performs one protected corpus read and one reusable FTS
session, then persists a separate ordinary request, run, packet, and receipt
identity for each question. Mixed scopes fail before execution. A batch does
not merge questions, evidence, or review history.

`AgentResearchFacade` opens a process-local `RecallScopeSession` on the first
recall for one session. The cache is bound to exact Library, snapshot, policy,
purpose, Collections, and exclusions; retrieval budget remains per question.
The default LRU capacity is four live scope sessions. `close_scope_sessions()`
destroys every in-memory SQLite index. `cache_stats(session_id)` reports only
non-canonical runtime counts: permitted fragments, builds, and searches.

### Expanded discovery route

Development evaluations use bounded FTS50, conditional FTS100, a bounded total
candidate cap, and at most two QueryCloud queries. These are evaluation
parameters, not a public persisted request contract.

## Evidence gate results

| Result | Meaning |
|---|---|
| `ready` | every frozen requirement has accepted exact support |
| `partial` | some requirements are supported and some remain open |
| `insufficient` | candidates exist but do not satisfy the requirements |
| `gap_preserved` | a declared absence remains unchallenged in the searched scope |
| `gap_challenged` | new exact evidence conflicts with a declared gap |

The gate is deterministic and provider-free. It does not estimate truth
probability.

The default `exact_fragment_unicode_v3` profile uses word-bounded matching,
preserves Latin and Cyrillic letter/diacritic distinctions, treats combining
marks as word characters, requires a positive condition, and validates
caller-supplied lineage consistency. Explicit v1 and v2 replays preserve their
prior behavior; v2 remains Latin-diacritic-insensitive. The supported product
evaluation scope is Latin and Cyrillic corpora. `EvidenceCandidate.source_family`
retains its v1 public name but carries the canonical `source_family_id`.

## Retrieval profile

| Profile | Role | Current status |
|---|---|---|
| none / FTS5 | exact lexical recall | default |

The v1 package contains no embedding model, reranker, vector store, model
provisioner, or model cache. Historical Harrier and dense-retrieval experiments
remain evaluation records only. Candidate rank never sets
`body_proof_eligible`, source authority, independence, or a human
`ReviewDecision`.

## Data and isolation rules

- One SQLite database per Library.
- Collections are logical scopes inside a Library.
- Heavy writes to one Library are sequential.
- Live data is outside repositories and sync roots.
- API binds to `127.0.0.1`, pins one Library, and creates a fresh mutation
  bearer token.
- Library bootstrap is CLI-only.
- Cross-Library reads require an explicit contract; no global corpus index is
  created.

## Failure semantics

Infrastructure or integrity failures return a non-zero command status and a
terminal failed `ProcessingRun`. A legitimate lack of evidence is a normal
gap/insufficient result, not an exception.

The system fails closed on policy/snapshot mismatch, changed source during
read, malformed identifiers, hash mismatch, migration drift, unsafe paths,
ambiguous external references, incomplete proof manifests, or replay identity
mismatch.

## Quality commands

```bash
uv sync --frozen --dev
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src tests
uv run pytest
uv run pip-audit
uv run check
./scripts/acceptance.sh --quick
./scripts/acceptance.sh
```

`uv run check` is the development aggregate. Read
`artifacts/coverage.json` for unrounded combined, statement, and branch values.
The full acceptance script is the clean-install packaging gate; `--quick` is only a
rehearsal. Both use the development aggregate's combined-coverage threshold.
Statement and branch coverage are reported separately. Never use the combined
value to imply that every decision path has been exercised.
