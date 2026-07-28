# Dithyramba reference

This reference describes the `0.1.0rc0` pre-alpha source preview. The command
itself is authoritative for exact options and defaults:

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
| Current database schema | v9 |
| Package version | `0.1.0rc0` |

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
```

`Lens` is the umbrella name for researcher-facing views, not a separate CLI
command. Today it is assembled from ReadingRoom, Research Atlas, and Flow View.

`reading-room` requires `--library`, `--snapshot`, `--access-policy`, at least
one `--collection`, `--purpose`, and `--data-home`. It starts a separate GET-only
loopback server. `atlas` and `flow-view` also start read-only loopback views.

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

Changing the meaning or required fields of one of these schemas requires a
new version. Applied migrations remain immutable.

## Implemented library-only contracts

| Contract | Schema | Persistence |
|---|---|---|
| `EvidenceRequirement` | `dithyramba.evidence_requirement/1.0` | caller-owned input |
| `EvidenceGateSpec` | `dithyramba.evidence_gate_spec/1.0` | caller-owned input |
| `EvidenceCoverageResult` | `dithyramba.evidence_coverage_result/1.0` | in-memory |
| `QueryRepairPlan` | `dithyramba.query_repair_plan/1.0` | in-memory |
| `HarrierScoreBatch` | `dithyramba.harrier_score_batch/1.0` | in-memory |
| `AdaptiveStageReceipt` | `dithyramba.adaptive_stage_receipt/2.0` | in-memory |
| `AdaptiveRecallResult` | `dithyramba.adaptive_recall_result/2.2` | in-memory |
| `RouteCandidateReceipt` | `dithyramba.route_candidate_receipt/1.0` | library artifact |
| `CompactMemoryPacket` | `dithyramba.compact_memory_packet/1.1` | library artifact |
| `CompactConnectorPacket` | `dithyramba.compact_connector_packet/1.0` | library artifact |

These contracts are executable and tested, but they do not imply a stable
CLI/HTTP compatibility promise.

## Adaptive compatibility contracts

| Contract | Schema | Current boundary |
|---|---|---|
| `FragmentTextProjectionReceipt` | `dithyramba.fragment_text_projection_receipt/1.0` | tested library artifact; not run-bound |
| `ExternalReferenceMap` | `dithyramba.external_reference_map/1.0` | tested library artifact; not run-bound |
| `ProofMetadataManifest` | `dithyramba.proof_metadata_manifest/1.0` | tested library artifact; not persisted |

These contracts are present and tested in the source preview. They fail closed,
but the adaptive route does not yet bind their instances into one persisted run.

## Planned durable adaptive contracts

The following names are reserved for the durable adaptive slice:

| Contract | Responsibility |
|---|---|
| `AdaptiveQueryPlan/1.0` | freeze scope, budgets, Gate, providers, and compatibility hashes |
| `AdaptiveEvidencePacket/2.0` | persist matched proof, gaps, and text-free discovery trace |

`AdaptiveEvidencePacket/2.0` is distinct from the existing
`ExpandedEvidencePacket/2.0`; neither extends `EvidencePacket/1.0` in place.
The three compatibility artifacts listed in the previous section are already
executable and fail closed, but are not yet bound to a persisted adaptive run.

## Default budgets

### Stable FTS request

| Field | Default | Bounds |
|---|---:|---:|
| `max_candidates` | 100 | 1–500 |
| `max_source_fragments` | 30 | 1–100 and not above candidates |

### Adaptive route

The route uses bounded FTS50, conditional FTS100, a Harrier scoring cap, a
bounded total discovered-candidate cap, and at most two QueryCloud queries.
Exact constants are versioned in `AdaptiveRetrievalConfig`; callers should not
reimplement them from prose.

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
| multilingual E5-small | compact dense control | optional |
| Harrier 270M | bounded q0 candidate reranking | adaptive library route |

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
