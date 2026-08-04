---
name: dithyramba
description: Operate Dithyramba research memory end to end. Use when the user asks to create, ingest, update, inspect, query, freeze, back up, continue, or diagnose a Dithyramba Library; retrieve source-grounded evidence; open Lens; or configure the local MCP adapter. Coordinate the CLI for lifecycle work and stdio MCP for bounded interactive research. Never treat retrieved candidates as accepted truth, invoke OCR, network access, or models without approval, mutate source files, or delete data without confirmation.
---

# Dithyramba

Use Dithyramba as a local research memory that maps questions to exact source
passages and preserves the route back to them. Keep the skill thin: orchestrate
the installed CLI, MCP adapter, and Lens; do not reproduce their logic in prose
or write directly to the Library database.

## Start by orienting

1. Locate the installed `dithyramba` executable and the explicit Library ID and
   data home. Never guess a data home from a source folder.
2. Run `dithyramba library describe --library <id> --data-home <path> --json`.
3. Run `dithyramba library doctor --library <id> --data-home <path> --json`
   before ingest, migration, restore, or debugging.
4. Report the Library name, source and fragment counts, latest snapshot, recent
   processing run, and any missing prerequisite in ordinary language.

If no Library exists, ask for or infer only the Library name, source roots, and
runtime data home. Keep source files and runtime data physically separate.

## Choose one operating mode

- **Orient** — describe and diagnose an existing Library without reading source
  text.
- **Build** — initialize a Library, add Collections, create a default-deny
  AccessPolicy, ingest sources, and freeze a snapshot.
- **Research** — open a bounded session, recall candidates, prepare an answer,
  then record a draft or an explicit gap.
- **Refresh** — re-index changed sources incrementally, freeze a new snapshot,
  and preserve the previous snapshot for replay.
- **Audit** — inspect packets, sessions, reviews, backups, and Lens without
  silently changing acceptance state.

Read [references/lifecycle.md](references/lifecycle.md) for lifecycle commands.
Read [references/mcp-tools.md](references/mcp-tools.md) for interactive MCP use.
Read [references/evidence-contract.md](references/evidence-contract.md) whenever
the task produces or evaluates a factual answer, hypothesis, or synthesis.

## Follow the research loop

Use MCP for an agent conversation when it is configured; otherwise use the
equivalent Python facade or CLI for bounded operations.

1. `open_session` with one explicit question, intended use, success criteria,
   snapshot, policy, and Collections.
2. `recall` to retrieve candidates. Treat rank as discovery priority, not truth.
3. Inspect exact passages and source coordinates. Broaden the query only when
   the first route leaves a named evidence role uncovered.
4. `prepare_answer` with the exact candidate IDs and declared evidence
   requirements. This applies candidate hygiene and `EvidenceCoverageGate`.
5. Generate only from the returned eligible evidence. Keep interpretation and
   speculation visibly separate from directly supported claims.
6. `record_draft` with the complete, unchanged preparation. If the gate blocks,
   use `record_gap`; if a route is misleading, use `reject_path`.
7. Use `session_context` to continue later without rereading the corpus.

Never bypass a blocked gate by rewriting the answer as certainty. A useful gap
is a valid research result.

## Keep expensive actions conditional

Default to deterministic local parsing, SQLite FTS, and exact passage checks.
Do not enable OCR, network research, embedding models, rerankers, or generative
providers merely because they are available. State the expected benefit and
obtain approval when an action adds cost, external transmission, or a new data
dependency.

## Preserve operator control

- Never modify original source files.
- Never delete a Library, source, snapshot, packet, or backup without explicit
  confirmation and a named target.
- Never expand a session beyond its AccessPolicy or snapshot.
- Never call a candidate, hypothesis, or draft an accepted fact.
- Human review remains the only route to acceptance or rejection.
- Back up before migration, restore rehearsal, or broad structural change.
- Keep credentials and bearer tokens out of notes, logs, and chat output.

## Report progress and results

For long corpus work, report at meaningful boundaries rather than every file:

- discovered / admitted / excluded inputs;
- indexed sources and fragments;
- elapsed parsing and ingest time;
- model calls and estimated agent tokens, explicitly `0` when none were used;
- snapshot ID, doctor result, and unresolved failures.

Finish with four short sections:

1. **What changed** — durable state created or updated.
2. **How to use it** — the next concrete command or research question.
3. **What is verified** — deterministic checks and exact evidence boundary.
4. **What remains open** — gaps, human review, or optional expensive work.
