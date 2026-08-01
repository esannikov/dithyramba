# Roadmap 0.2

The roadmap is ordered by dependency. A later stage must not be started merely
because it is visually attractive; its preceding contract must first pass focused
tests and cold replay.

## 0.2a — Session Spine

Status: complete

- freeze the technical specification and terminology;
- implement immutable session, brief, artifact-reference, event, and state models;
- implement deterministic append and replay;
- prove role boundaries and fail-closed hash-chain behavior;
- document the plain-language research loop.

Exit condition: focused tests, Ruff, and strict MyPy pass without a new dependency.

## 0.2b — Durable session store

Status: complete

- add migration 0013 to both SQL mirrors;
- persist sessions and append-only events in SQLite;
- validate referenced artifacts inside the same Library;
- add insert-or-verify, outbox events, cold reopen, tamper, backup, and migration tests;
- derive state from persisted events rather than storing an editable summary.

Exit condition: fresh install, 0012 -> 0013 migration, backup, reopen, and corruption
tests pass.

## 0.2c — Research tools for agents

Status: next

- expose a small Python facade: `open`, `recall`, `attach`, `draft`, `propose`,
  `record_gap`, `state`;
- assemble least-context packets from the session state and current question;
- keep semantic candidate acceptance human-only;
- add a thin local stdio MCP adapter over the Python facade.

Exit condition: one agent can continue a session after restart without receiving
the entire transcript or corpus.

## 0.2d — Human review loop

- project pending candidates, gaps, and decisions into Lens;
- reuse the existing meaning review sessions and append-only decisions;
- show exact source text for every promotable claim;
- let a human accept, reject, revise, defer, or supersede without granting the
  agent a commit tool.

Exit condition: candidate -> exact evidence -> human decision -> replay receipt is
auditable end to end.

## 0.2e — Controlled research expansion

- add an explicit web-research connector with source capture and rights metadata;
- quarantine snippets and summaries until a source artifact is ingested;
- deduplicate against existing source identity;
- record negative search paths to avoid repeated token spend.

Exit condition: external research can enrich a session without provenance
laundering or silent corpus mutation.

## 0.2f — Evaluation and release candidate

- run 5–10 real investigations across mixed and focused corpora;
- measure supported-answer rate, exact-source recovery, review burden, repeated
  search reduction, context size, and continuation after restart;
- compare session memory against raw-file agent search under the same task set;
- fix only recurring causal failures;
- run the full clean-install acceptance route on a clean second machine.

Exit condition: the documentation reports measured benefits and negative findings,
not only passing implementation tests.

## Deferred until evidence requires them

- embeddings or a vector store;
- a graph database;
- autonomous ontology maintenance;
- background self-learning;
- model-specific fine-tuning;
- multi-user synchronization.
