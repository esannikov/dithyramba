# Roadmap 0.2

> Completed implementation history. `0.2` was an internal development
> milestone, not the current package version. For the current release path, use
> [`V1_ROADMAP.md`](../V1_ROADMAP.md) and [`REFERENCE.md`](../REFERENCE.md).

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

Status: complete

- [x] expose a small Python facade: `open`, `recall`, `prepare_answer`,
  `attach_evidence`, `record_draft`, `link_candidates`, `record_gap`,
  `reject_path`, `context`;
- [x] assemble a bounded context projection from the session state and return
  exact evidence text separately in a compact `AgentEvidencePacket`;
- [x] keep the full `EvidencePacket` and `ReadReceipt` local while exposing their
  immutable IDs and hashes to the agent;
- [x] keep semantic candidate acceptance, operator decisions, and session closure
  outside the agent facade;
- [x] add command idempotency and partial-turn reconciliation through atomic
  question/completion receipts in the existing append-only outbox;
- [x] pass the full repository gate after the answer-route repair: 2,678
  passed, 2 host/browser skips, branch-aware coverage 95.02%, Ruff and strict
  MyPy green;
- [x] reuse one process-local authorized read-set and FTS index for questions
  inside the same exact session scope;
- [x] add a thin local stdio MCP adapter over the Python facade.
- [x] expose human-readable source title/URI references beside every selected
  fragment in `AgentEvidencePacket/1.2`, mark raw recall as
  `retrieved_candidates`, report source dominance, and retain 1.0/1.1 replay;
- [x] validate the full immutable scope closure once per explicit live session
  and revalidate selected stored fragments on each later completion;

Exit condition: one agent can continue a session after restart without receiving
the entire transcript or corpus.

## 0.2d — Human review loop

Status: projection slice complete; mutation and review remain on existing human surfaces

- [x] project the brief, questions, exact evidence, drafts, gaps, and rejected
  paths into GET-only Session Lens;
- [x] keep source names and exact text primary while hiding internal IDs under
  progressive technical disclosure;
- [x] reopen modern Lens events from compact command receipts and direct
  fragment metadata instead of reconstructing a corpus-wide `ReadReceipt`;
- project pending candidate and human decision status into Lens after the
  existing review route is bound to a session;
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

- [x] run a 20-turn Mars session over 2,047 sources / 53,747 retrieval units
  with zero model calls, cold reopen after every turn, and repeated command IDs;
- [x] measure and repair the accidental full-`ReadReceipt` agent transport;
- [x] run six realistic three-turn Mars investigations plus one isolated
  relation-repair challenger over 2,047 sources / 53,747 units;
- [x] verify one FTS build per three-turn session, 6/6 exact retries, 6/6 cold
  reopens, and 17/17 unchanged comparable top-10 result lists;
- [x] measure warm-turn latency: 38.76 s versus 50.90 s first-turn mean, a
  23.85% reduction with zero model tokens;
- [x] run the M1 PhD scale diagnostic: 622 sources / 263,363 fragments, second
  same-session completion 13.557 s, and 24-fragment Lens projection
  0.062–0.064 s with zero model tokens;
- [x] freeze and run the first 24-question / 72-query PhD scholarly holdout on
  a corrected 615-source / 259,165-fragment scope: 20/72 direct top-three
  supports, 61/120 useful top-five fragments, zero model tokens;
- [x] keep raw recall explicitly candidate-only in `AgentEvidencePacket/1.2`
  and expose exact source/family dominance without breaking 1.0/1.1 replay;
- [x] bind `EvidenceCoverageGate` to the interactive answer route: deterministic
  noise guards, conditional drilldown inside up to three already found Sources,
  exact replay before `record_draft`, and fail-closed blocked/gap states;
- add reviewed work/source identity for cross-format copies;
- replace development agent labels with independent human adjudication and
  measure correction load and answer usefulness;
- measure exact-source recovery, repeated search reduction, context size, and
  continuation after restart on the same frozen questions;
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
- relation-aware automatic query repair; one known Mars miss improved for only
  one of two variants, which is insufficient independent evidence for a new
  production layer.
