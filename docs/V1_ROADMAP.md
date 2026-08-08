# Dithyramba v1 release roadmap

## Goal

Dithyramba v1 is a small local research-memory engine with one default route:

```text
read-only sources
  → immutable versions and exact fragments
  → frozen scope and default-deny access
  → local SQLite FTS
  → evidence coverage check
  → compact source-addressed packet
  → MCP or Lens
  → append-only human review
```

The default installation requires no model, embeddings, vector store, graph
database, API key, or network service.

## Completed

- Immutable source identity, exact addresses, snapshots, access policies,
  evidence packets, receipts, review, backup, and restore.
- Incremental ingest that reuses unchanged inputs and preserves alternative
  parser representations without rewriting source files.
- One clean v1 SQLite schema; old experimental vector and reranking paths are
  absent from new Libraries and the default dependency tree.
- Process-local session reuse, compact agent context, seven bounded stdio MCP
  tools, and a GET-only Lens.
- Source and wheel distribution checks from isolated environments.
- A thin Codex skill that orchestrates the public CLI and MCP surfaces without
  duplicating research logic.

## Current evidence

| Measure | Current result |
|---|---:|
| Persisted Library scale | 3–2,047 sources; 7–276,165 fragments |
| Unchanged repeat ingest | 100% reused in 265-, 400-, and 816-source runs |
| Deterministic replay | 16/16 sampled and 30/30 fresh-corpus cases |
| Default model budget | 0 calls; 0 tokens |
| Engineering gate | 1,927 collected; 1,925 passed; 2 host-dependent skips; 95.18% combined coverage |

These are engineering and retrieval measurements. They do not establish source
truth, scholarly validity, or the quality of a generated conclusion. Scale
maxima may come from different runs. Detailed
protocols and limitations are kept in [EVALUATION.md](EVALUATION.md).

## Remaining before stable 1.0.0

1. Keep Linux, macOS, browser, dependency, and distribution CI green on the
   release commit.
2. Repeat the published install and demo on a machine that has no project-local
   state.
3. Freeze the public CLI, MCP tool list, packet contracts, schema fingerprint,
   and documentation links.
4. Record commit-bound acceptance results, tag the release, and publish release
   notes.

## Post-v1 candidates

These stay outside the stable path until a focused evaluation shows a clear
user benefit without weakening provenance:

- replayable QueryCloud plans for difficult lexical queries;
- a faster optional PDF preprocessing profile with automatic fallback and
  completeness checks;
- scoped concept views with a human coherence test;
- optional synthesis adapters that report their own model and token budget.

## Measures to keep separate

1. **Cold ingest:** new or changed bytes parsed.
2. **Incremental ingest:** unchanged, changed, failed, and parser invocations.
3. **Cold and warm recall:** first scope build versus later questions.
4. **Retrieval quality:** correct source and exact fragment measured separately.
5. **Evidence closure:** declared evidence roles found, blocked, or still open.
6. **Agent cost:** context delivered and any optional model calls or tokens.

## Out of scope for v1

- autonomous promotion of generated claims;
- a global ontology of an entire mixed Library;
- a second graph or vector database;
- training or fine-tuning an embedding model;
- cloud hosting, multi-user authorization, or remote corpus sync;
- claims that code coverage or retrieval rank prove research truth.

## Rollback rule

Existing Libraries are never modified in place by cleanup code. Any future
schema or parser experiment must start from a backup or a rebuild from read-only
sources, fail closed, and preserve the previous accepted release path.
