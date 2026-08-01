# Interactive research memory 0.2

Status: Session Spine, durable store, and least-context agent transport implemented locally

Base: `0.1.0rc1` plus the proposition-level answer work in PR #5

Implemented slices: session contracts, deterministic replay, SQLite persistence,
idempotent recall commands, and compact evidence projection

Verification: 2,614 tests passed, 2 host/browser tests skipped, branch-aware
coverage 95.03%, Ruff and strict MyPy passed, with no new runtime dependency

## Product objective

Dithyramba 0.2 adds a research loop around the existing source-grounded memory.
An operator and an agent may investigate a problem over many conversations
without putting the whole corpus or the whole conversation into the model's
context window each time.

The system keeps four things separate:

1. **Conversation** — questions, drafts, rejected paths, and operator decisions.
2. **Recall** — compact packets selected from an explicit corpus snapshot.
3. **Evidence** — exact source fragments and their immutable addresses.
4. **Memory candidates** — statements, hypotheses, concepts, and public reasoning
   traces that require the existing evidence and human-review route.

Conversation may cause retrieval or produce a candidate. It is never evidence by
itself.

## Why this is an evolution rather than a second system

Version 0.1 already contains the expensive trust machinery:

- immutable source fragments and snapshots;
- access-policy-scoped FTS recall;
- compact evidence packets;
- exact citation validation;
- answer-facet coverage;
- receipt-bound claim/evidence review;
- candidate statements, evidence links, IdeaTrace, and human review decisions.

Version 0.2 must coordinate these objects across a continuing investigation. It
must not introduce another statement table, another evidence type, or another
notion of acceptance.

## First slice: Session Spine

### Public objects

| Object | Purpose | Trust status |
|---|---|---|
| `ResearchSessionBrief` | States the problem, intended use, success conditions, and boundaries. | Operator input, not evidence. |
| `ResearchSession` | Binds the brief to one Library, corpus snapshot, policy, purpose, and collection scope. | Immutable scope receipt. |
| `SessionEvent` | Records one typed step in an append-only hash chain. | Journal fact only; payload is not automatically a corpus fact. |
| `SessionArtifactReference` | Pins an existing Dithyramba artifact by kind, ID, and hash. | A pointer; inherits no stronger status than its target. |
| `ResearchSessionState` | Deterministic projection rebuilt from the event stream. | Derived and disposable. |

### Event types

- `question_asked`
- `evidence_attached`
- `answer_drafted`
- `candidate_linked`
- `gap_recorded`
- `path_rejected`
- `decision_recorded`
- `session_closed`

Free-form events are deliberately excluded. Every event has a bounded meaning.
Agent output may be recorded as a draft, gap, or candidate reference. Only a
human actor may record an operator decision or close a session.

### State machine

```text
open session
    -> question
    -> recall and attach evidence packet
    -> draft answer / record gap / reject path
    -> link validated candidate
    -> human decision
    -> repeat or close
```

Every event names the preceding event hash. Replay must fail closed for a gap,
reordering, foreign session binding, changed payload, or an event after closure.
The session state is rebuilt from events; it is never an independently editable
summary.

## Agent boundary

The first agent-facing Python surface may:

- open a bound session;
- record a question or draft;
- attach existing evidence artifacts;
- link existing candidate artifacts;
- record gaps and rejected paths;
- replay the event stream and request its compact state.

It may not:

- turn chat text or a search snippet into evidence;
- bypass access policy or snapshot binding;
- accept a semantic candidate;
- create a human review decision;
- rewrite or delete a session event.

Human promotion continues through the existing review contracts. The future MCP
adapter will be a thin transport over the same Python services and will not own
business logic.

The implemented `AgentResearchFacade` exposes only bounded research actions. A
turn returns two different objects: a compact `AgentSessionContext` containing
recent journal state and explicit omission counts, and an
`AgentEvidencePacket` containing only the selected exact source fragments,
their addresses, compact coverage counts, and the IDs and hashes of the full
audit receipts. The complete `EvidencePacket`, including its materialized
`ReadReceipt`, remains local in the Library and can be inspected explicitly; it
is not copied into every agent prompt. This avoids replaying the whole transcript
or corpus-read manifest while preserving a cryptographic route back to the full
audit artifact.

Every recall command now requires a stable `command_id`. Its question event and
start receipt commit atomically; its evidence event and exact response receipt
also commit atomically. A retry with identical input returns the persisted turn
without a second FTS run or journal event. A retry after interruption resumes
from the one durable question, while reuse of the command ID for different input
fails closed. These receipts reuse the existing append-only outbox, so no third
editable session store or schema migration is introduced.

## Failure semantics

- Invalid or stale artifact hashes are rejected by the persistence adapter.
- A broken event chain cannot be partially replayed.
- An agent-authored `decision_recorded` or `session_closed` event is rejected.
- Session state can be regenerated after loss because events are authoritative.
- A completed command can be replayed byte-for-byte after restart; a partial
  command cannot duplicate its question when the transport retries.
- Ordinary command replay reconstructs the compact projection from its persisted
  response receipt and does not rematerialize the full `EvidencePacket`. Legacy
  0.2 development receipts are upgraded in memory by loading the full packet once.
- A missing semantic judgment remains `review_required`; no model verdict becomes
  historical truth without its bound receipt.

## Durable boundary

Migration `0013_research_sessions.sql` adds only two append-only tables:
`research_sessions` and `research_session_events`. The store validates the exact
Library, CorpusSnapshot hash, AccessPolicy scope, exclusions, and every attached
artifact ID/hash before accepting an event. Reopening a Library reconstructs the
same event stream and state; the state itself is not stored as editable truth.

Both tables emit ordinary outbox events. SQL triggers independently reject event
gaps, broken previous hashes, events after closure, model-authored decisions, and
any update or deletion. The Python layer repeats the semantic checks so a failure
is caught before or during the transaction rather than trusted to one boundary.

## Acceptance criteria for the Session Spine

1. Contracts are strict, immutable, content-addressed, and reject unknown fields.
2. A session is bound to one Library, snapshot, policy, purpose, and collection set.
3. Event sequence and previous-hash closure are deterministic and fail closed.
4. Replaying the same stream produces the same compact state and state hash.
5. No event after closure is accepted.
6. Only a human actor may record decisions or close a session.
7. Evidence and candidate events require typed, hash-pinned artifact references.
8. Unit tests cover happy paths, tampering, ordering, scope, role, and closure errors.
9. The package remains offline-first and adds no runtime dependency.
10. The agent transport omits the materialized `ReadReceipt` while retaining its
    immutable ID/hash and every selected exact fragment.

## Explicitly outside the implemented contour

- MCP and HTTP routes;
- automatic capture of arbitrary chat transcripts;
- automatic candidate promotion;
- global ontology generation;
- background web research;
- embeddings, a vector database, or a graph database;
- a redesigned Lens interface.

These are staged only after the Session Spine is stable. This prevents transport,
UI, and orchestration concerns from becoming part of the memory core.
