# Dithyramba reference

This reference describes the `0.1.0rc1` package base plus the current pre-alpha
`0.2` development candidate. The command itself is authoritative for exact
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
| Optional semantic extra | `sentence-transformers==5.6.0` |
| Optional ontology extra | `numpy>=2,<3`; `scikit-learn>=1.8,<2` |
| Current database schema | v13 |
| Package version | `0.1.0rc1` |

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
dithyramba library doctor
dithyramba library migrate
dithyramba collection add
dithyramba collection list
dithyramba collection freeze
```

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
PDF, Markdown, and text inputs: up to 512 MiB per file, 1,500 PDF pages,
20 million extracted characters, 180 seconds, and 1,024 MiB worker RSS. It does
not remove parser limits or change source identity.

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
dithyramba reading-room
dithyramba atlas
dithyramba flow-view
dithyramba concept-lens
dithyramba session-lens
dithyramba mcp
dithyramba reasoning-check
```

`Lens` is the umbrella name for researcher-facing views, not a separate CLI
command. Today it is assembled from ReadingRoom, Research Atlas, Concept Lens,
Flow View, and Session Lens.

`reading-room` requires `--library`, `--snapshot`, `--access-policy`, at least
one `--collection`, `--purpose`, and `--data-home`. It starts a separate GET-only
loopback server. `atlas`, `flow-view`, `concept-lens`, and `session-lens` also
start read-only loopback views. Session Lens requires one existing
`ResearchSession` ID and exposes `/projection.json` beside its human journal.

`mcp` runs a stdio server over one existing Library. It reserves stdout for
newline-delimited JSON-RPC and exposes `open_session`, `recall`,
`session_context`, `record_draft`, `record_gap`, and `reject_path`. It has no
human acceptance, decision, promotion, deletion, or session-closure tool.

`reasoning-check` consumes one absolute-path `IdeaTrace`, exact claim-evidence
case set, and semantic entailment result. It makes no provider call and emits a
canonical closure receipt. A non-passed closure exits non-zero.

### Models

```text
dithyramba model provision
```

Provisioning requires an explicit network permission for first download. An
offline call verifies a cached pinned revision. A provisioned model is not
automatically enabled in the default recall route.

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
pre-alpha contracts, not a public long-term compatibility promise.

## Core input formats

| Suffix | Core ingest | Address |
|---|---|---|
| `.md`, `.markdown` | yes | heading and character range |
| `.txt` | yes | character range |
| `.pdf` | yes | page and normalized bounding box |
| `.epub` | book Connector only | spine item and local range |
| `.fb2` | book Connector only | XML structure and local range |

The book Connector returns Markdown plus a canonical sidecar. It does not
promote the projection to source evidence by itself.

## Versioned core contracts

These schemas are versioned within the source preview. Their version labels do
not create a compatibility promise beyond the declared `0.1.x` preview.

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

Interactive transport adds two compact, derived schemas rather than changing
the durable `EvidencePacket/1.0`:

| View | Schema | Boundary |
|---|---|---|
| `AgentSessionContext` | `dithyramba.agent_session_context/1.0` | bounded recent journal state plus explicit omission counts |
| `AgentEvidencePacket` | `dithyramba.agent_evidence_packet/1.1` | selected exact fragments, readable source references, coverage, and IDs/hashes of the full audit receipts |

`AgentEvidencePacket/1.0` remains readable for development-session replay. It
lacks the 1.1 source-reference list but preserves the selected exact fragments
and provenance hashes. The full materialized `ReadReceipt` stays local in both
versions and is reopened only through the strict audit route.

Schema v10 added an internal append-only `CorpusReadSet`: one exact protected
fragment manifest can be shared by several recall requests over the same
Library, snapshot, policy, Collections, and purpose. Public `ReadReceipt/1.0`
and `EvidencePacket/1.0` payloads remain unchanged, and v9 Libraries remain
readable after migration.

Schema v11 adds append-only `idea_traces` and
`reasoning_closure_results`. It stores only canonical public trace artifacts,
not private chain-of-thought text. A passed closure does not create a human
`ReviewDecision` and does not promote a claim into accepted memory.

Schema v12 adds append-only `answer_projections` and
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
| `AnswerProjection` | `dithyramba.answer_projection/1.0` | append-only canonical JSON in schema v12 |
| `AnswerProjectionJudgmentReceipt` | `dithyramba.answer_projection_receipt/1.0` | append-only canonical JSON in schema v12 |
| `PropositionCoverageResult` | typed result | in-memory; display eligibility only |
| `RouteCandidateReceipt` | `dithyramba.route_candidate_receipt/1.0` | library artifact |
| `CompactMemoryPacket` | `dithyramba.compact_memory_packet/1.1` | library artifact |
| `CompactConnectorPacket` | `dithyramba.compact_connector_packet/1.0` | library artifact |

These contracts are executable and tested, but they do not imply a stable
CLI/HTTP compatibility promise.

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
extraction. `dithyramba concept-lens --projection /absolute/ontology.json`
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

These contracts are present and tested in the source preview. They fail closed
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

## Model profiles

| Profile | Role | Current status |
|---|---|---|
| none / FTS5 | exact lexical recall | default |
| multilingual E5-small | compact dense control; rejected as the tested global cartography geometry | optional |

Model scores never set `body_proof_eligible`, source authority, independence,
or a human ReviewDecision.

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
