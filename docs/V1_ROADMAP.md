# Dithyramba v1 cleanup roadmap

## Goal

Dithyramba v1 is a small local research-memory engine with one default route:

```text
read-only sources
  → immutable SourceVersion and exact SourceFragment
  → frozen snapshot and compiled AccessPolicy
  → local SQLite FTS with bounded lexical repair
  → candidate hygiene and EvidenceCoverageGate
  → compact AgentEvidencePacket
  → MCP or Session Lens
  → append-only human review
```

The default installation must not require a model, embeddings, a vector store,
a graph database, or a network service. Optional research projections may exist
only when they have a named user job, an isolated dependency boundary, and an
evaluation that proves they improve that job.

## Baseline

Work begins from Git commit `7ce7f7e`, the green head of PR #7. Its ancestry
contains the proposition-level answer work, Harrier removal, interactive session
spine, MCP, Session Lens, PhD-scale repairs, exact-passage drilldown, and compact
answer-ready packets. The baseline passed Linux, macOS/PDF, and Chromium CI.

The cleanup does not weaken these invariants:

- exact source identity and immutable versions;
- default-deny access before fragment materialization;
- content-addressed receipts and byte-exact replay;
- visible candidate, evidence, synthesis, hypothesis, and human-decision states;
- fail-closed gaps and parser failures;
- local, zero-model FTS recall;
- exact source coordinates in every proof-eligible packet.

## Implementation status

### 2026-08-03 — V1.1 to V1.3 complete locally

The incremental-ingest fast path is implemented on branch
`codex/v1-clean-core`. Repeated input now reuses an existing SourceVersion
before parser invocation only after exact byte, parser-profile, Collection,
membership, lineage, fragment, and physical Blob checks. Changed bytes still
invoke the parser. A profile mismatch also invokes the parser. Corrupt Blob or
logical state fails closed.

V1.2 required no speculative rewrite. On the 615-source, 259,165-fragment PhD
scope, the existing process-local scope session built one authorized FTS index
and reused it for all later queries. The measured core median was `7.109 s`;
warm public-facade recalls were `13.908 s` and `13.666 s`, or `1.93–1.96×`
core. The separate `56.581 s` scope build and `72.776 s` first public recall are
the intentional one-time full receipt audit, not warm-query latency.

V1.3 removes the inactive semantic, hybrid, vector, reranker, model-provisioning,
and expanded-reranking runtime together with its CLI and optional dependency.
The default dependency lock no longer contains `sentence-transformers`,
Transformers, Torch, Hugging Face model tooling, or their CUDA dependency tree.
Runtime Python fell from 153 files / 65,226 physical lines to 136 files / 52,145
lines. The paired obsolete test surface was removed with the runtime rather than
left as misleading product evidence.

Verification after V1.3: `1946 passed`, `2 skipped`, branch-aware coverage
`95.26%`; Ruff format/lint and strict mypy passed. Model calls/tokens remained
`0/0`.

Earlier V1.1 verification: `2691 passed`, `2 skipped`, coverage `95.01%`;
Ruff format/lint, strict mypy, terminology, and dependency audit passed. This
closed V1.1 before runtime removal.

## Current excess

The verified baseline is conceptually smaller than its implementation. Before
cleanup it contains 153 runtime Python files, 65,226 physical Python lines,
13 historical schema migrations, eight default runtime dependencies, and
several old or experimental routes.

| Area | v1 decision | Reason |
|---|---|---|
| `contracts`, `library`, `collections`, `ingest`, `provenance` | core | source and identity boundary |
| `snapshots`, `access`, `recall/fts`, bounded expansion | core | governed candidate discovery |
| `evidence`, compact packets, source-local drilldown | core | proof preparation |
| `sessions`, `interactive`, stdio MCP | core | context-window and agent boundary |
| `review` | core | human promotion boundary |
| Session Lens and exact source inspector | core surface | human audit path |
| `answers` and bounded public `reasoning` artifacts | retain, narrow | source-grounded synthesis without hidden truth promotion |
| backup and restore | retain | local durable memory requires recovery |
| candidate ontology | optional until human coherence gate | useful orientation is not yet independently validated |
| Research Atlas, Reading Room, Concept Lens, Flow View | consolidate | one Lens product surface is enough for v1 |
| semantic vectors, hybrid retrieval, rerankers, model provisioning | remove from default runtime | no active default route and no measured current win |
| `sentence-transformers` extra | remove or move to a separate experimental package | contradicts the model-free v1 promise |
| old semantic/hybrid tables | preserve only through an explicit legacy importer | historical compatibility must not define the clean v1 schema |
| case-specific fixtures, evaluations, model caches, runtime DBs | exclude from package | development evidence or rebuildable local state |

The table is a deletion plan, not deletion authority. A module moves to
`remove` only after import, persistence, CLI, public API, test, documentation,
and existing-Library compatibility audits agree.

## Ordered delivery

### V1.1 — Incremental ingest

Add a read-only pre-parse reuse check for an unchanged Source. It must verify
the exact Collection/root/URI, media type, current byte hash, parser profile,
membership state, lineage, identity declaration, fragments, and physical Blob.
Only then may it return the existing SourceVersion without invoking a parser.

Acceptance:

1. [x] A repeated unchanged ingest invokes the parser zero times.
2. [x] Changed bytes still run the parser and create or restore the correct version.
3. [x] A different parser profile never uses the fast path.
4. [x] Missing/corrupt Blob, lineage conflict, membership conflict, and source
   mutation fail closed.
5. [x] Coverage and ProcessingRun receipts remain complete and deterministic.

### V1.2 — Public route latency

Make the public session facade reuse the same exact process-local scope used by
the core route. Remove duplicate packet reconstruction and projection work.

Acceptance:

1. [x] The same 259,165-fragment scope is built once per live session.
2. [x] Warm public recall is no more than twice the measured core-route wall time.
3. [x] Full audit receipts remain available by immutable ID and hash.
4. [x] Compact events never exceed their reference bound or lose packet closure.

### V1.3 — Runtime diet

Build an import and dependency inventory, then remove inactive semantic/hybrid
execution from the default package. Consolidate human views around Session Lens.

Acceptance:

1. [x] Default install contains no model runtime or model provisioning code path.
2. [x] Default `pyproject.toml` has no embedding/reranking dependency extra.
3. [x] No CLI, MCP, HTTP, or public import advertises a removed route.
4. [x] Historical evaluation reports remain in Git history or a clearly labelled
   archive, never in the user workflow.
5. [x] The stable route remains green; clean-install acceptance is repeated in V1.5.

### V1.4 — Clean schema boundary

Generate one reviewed v1 schema snapshot from the retained tables. Existing
release-candidate Libraries are never modified in place by cleanup code. They
must either reopen through a separately tested legacy importer/exporter or be
rebuilt from their read-only sources with explicit loss reporting for reviews
and session artifacts.

Acceptance:

1. A new v1 Library starts from one documented schema baseline.
2. No retired semantic/vector table is created in a new v1 Library.
3. Legacy conversion is all-or-nothing, backup-first, and hash-verified.
4. Failure leaves the original Library byte-for-byte untouched.

### V1.5 — Release closure

Rebuild the public explanation, diagrams, repository guide, examples, and
clean-install protocol around the one stable route.

Acceptance:

1. Linux and macOS clean installs pass from sdist and wheel.
2. Full Ruff, strict mypy, pytest, dependency audit, PDF, MCP, Session Lens,
   and Chromium gates pass.
3. Test coverage remains at least 95% without excluding retained runtime code.
4. A fresh 300–800-source corpus completes ingest, incremental update, 30
   frozen questions, exact replay, and source inspection.
5. Engineering checks and human relevance judgments are reported separately.

## Release measurements

Every phase records four different costs instead of one vague runtime number:

1. cold ingest: new or changed bytes parsed;
2. incremental ingest: unchanged, changed, failed, and parser invocation counts;
3. cold and warm recall: core and public facade measured separately;
4. agent context: exact evidence words/tokens delivered after local search.

Model calls and model tokens remain `0/0` for the default ingest and recall
route. Any synthesis or semantic reviewer reports its own separate budget.

## Out of scope for v1

- autonomous promotion of generated claims;
- global automatic ontology of an entire mixed Library;
- a second graph or vector database;
- training or fine-tuning an embedding model;
- cloud hosting, multi-user authorization, or remote corpus sync;
- a claim that code coverage proves scholarly relevance or truth.

## Rollback

Each phase is a separate commit. Deletion begins only after a passing inventory
and compatibility test. Existing Libraries are opened read-only during schema
experiments, backed up before any approved conversion, and never used as the
first test fixture. Reverting a phase must restore the previous wheel without
requiring a data rollback.
