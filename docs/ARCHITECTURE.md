# Architecture

Dithyramba separates four jobs that are often collapsed into one opaque
“knowledge” step: preserving sources, finding candidates, checking evidence,
and recording human decisions.

![Dithyramba memory architecture](assets/dithyramba-memory-map.svg)

The figure separates two cooperating planes. The durable evidence core turns
read-only source bytes into exact packets and scoped human decisions. The
interactive session loop lets an agent continue an investigation from compact
state and packet references without creating a second truth store. Rebuildable
caches, transports, and visual views may disappear; the source, packet, event,
and review records remain.

## The smallest durable unit

The authoritative atom is a `SourceFragment`: an exact passage that belongs to
an immutable `SourceVersion` and carries a stable `SourceAddress`. An embedding
is not the atom. A generated summary is not the atom. Both may be rebuilt or
revised while the passage identity remains stable.

A useful memory object is therefore a small provenance-bearing bundle:

```text
SourceFragment
  source_version_id
  exact text hash
  source address
  parser and read receipts
  ↓
Statement / EvidenceLink / ReviewDecision
  who makes the assertion
  what role the passage plays
  which scope and snapshot were used
  whether a human accepted, revised, rejected, or deferred it
```

Typed `Relations`, `Concepts`, `Voices`, and timeline objects can connect these
bundles. The connection never replaces the path back to exact evidence.

## Four layers

### 1. Bounded sources

A `Library` is the physical privacy, database, and backup boundary. A
`Collection` is a logical scope inside it. Source files remain read-only inputs.
Ingest stores immutable versions, exact fragments, content hashes, parser
receipts, and non-leaking logical addresses.

### 2. Governed recall

A `CorpusSnapshot` freezes which source versions participate. An `AccessPolicy`
is compiled before fragment text is selected. The stable recall route uses local
SQLite FTS5 to produce a bounded candidate list.

Aliases, neighbouring fragments, and bounded derived queries may improve
discovery. They are deterministic, replaceable aids and cannot grant access or
certify evidence. V1 contains no embedding model, reranker, or vector store.

### 3. Evidence

The stable route persists `EvidencePacket/1.0`: the query identity, exact
snapshot and policy scope, selected passages, source addresses, profile, read
receipts, and packet hash required for exact replay.

When several questions share one exact scope, schema v10 stores their permitted
fragment manifest once as a content-addressed `CorpusReadSet`. Each question
still has an independent `QueryRequest`, processing run, `ReadReceipt`, packet,
and review history. Sharing storage never merges questions or permits a scope
to cross a Library, snapshot, policy, Collection set, or purpose boundary.

The optional evidence-sufficiency route adds `EvidenceCoverageGate`. Its caller
freezes explicit requirements such as actor, date, source role, mechanism,
direction, or independent provenance group. The gate scans proof-eligible
passage bodies and reports which requirements are covered. Missing requirements
remain an `EvidenceGap`; a high search score cannot fill them.

### 4. Memory in use

Review decisions are scoped and append-only. Reading Room and Lens project the
stored state for a person. Connectors produce compact source-closed packets for
another agent. These surfaces do not become new truth stores and do not mutate
memory during read-only inspection.

## Evidence-grounded reasoning

An accepted answer may be followed by one bounded `IdeaTrace/1.0`. This is a
public research artifact, not a hidden chain of thought:

```text
accepted ResearchAnswer claim
  → exact ClaimEvidenceCase
  → bound semantic judgment receipt
  → extract / compare / connect / infer step
  → ReasoningClosureGate
  → passed, failed, or review_required
  → separate human review
```

Every step repeats the exact claim text, names its operation and earlier
premises, and binds to one claim-evidence case. Derived steps require a concise
public warrant. The trace is a small ordered DAG with at most 32 steps; all
steps must contribute to the final step, so cycles and decorative orphan nodes
fail closed.

`ReasoningClosureGate` is deterministic and provider-free. It verifies trace,
answer, packet, case-set, and judgment-receipt identities; then applies a small
operation/qualifier policy to the already judged claim states. It does not
rejudge historical truth. A passed closure is `review_eligible`, never accepted
memory by itself.

Schema v11 stores the canonical trace and closure receipt in two append-only
SQLite tables. The graph is rebuilt from their JSON; no second graph database
or vector index is required.

### A factual spine with room for exploration

`ResearchAnswer/1.0` remains the strict factual spine. An optional
`AnswerProjection/1.0` divides its exact displayed `short_answer` into
contiguous propositions with one declared role:

```text
fact | synthesis | hypothesis | question | framing
```

`PropositionCoverageGate` checks each span against the already accepted
claim-evidence receipt. A fact must close over a directly supported factual
claim. A synthesis may combine supported premises without claiming more than
they establish. A hypothesis must name its premises and carry a test question,
falsifier, and next evidence. A research question must point to the premises
that made it relevant. Framing may carry no claim binding and must be judged
non-propositional.

This avoids forcing a useful research answer to be a literal restatement of one
verified claim. It also prevents fluent speculation from entering memory as
fact. The projection pass is display governance only: even a passed synthesis
or hypothesis remains exploratory and needs a separate human decision before
promotion.

The deterministic gate makes no provider call. A semantic author or reviewer
may be used on the final short answer only; it does not reread or embed the
corpus.

Schema v12 stores the canonical `AnswerProjection` and its semantic judgment
receipt in two append-only tables. Persistence preserves what was judged; it
does not generate the projection, accept its content, or replace the separate
human review path. `PropositionCoverageResult` remains a deterministic,
rebuildable result.

Research Atlas may expose a sparse exact subset of the displayed answer or
hypothesis as `AtlasTraceSpan` records. Each record binds one half-open
character range to one or more evidence IDs and labels it as fact, synthesis,
hypothesis, or question. Lens colours only these bound characters. Selecting a
span activates its source chip and the exact-passage inspector; ordinary
framing remains uncoloured so the interface does not imply that every sentence
is verified.

## Experimental scoped candidate ontology

Question-led recall works when the operator already knows what to ask. A scoped
ontology helps before that moment without pretending to understand the entire
Library at once:

```text
one named scope + compact query expansion
  → one permitted local FTS result, balanced across sources
  → bounded exact-fragment neighbourhood
  → local TF-IDF/NMF candidate areas
  → scope anchors plus emergent phrases
  → exact multi-source co-occurrence links
  → GET-only Concept Lens
  → ordinary question-led recall and EvidenceCoverageGate
  → separate query branches only when a named gap remains
```

`CandidateOntologyManifest` stores clusters, candidate concepts, explicit
concept origin, exact evidence addresses, and `co_occurs_with` relations. It is
content-addressed and replayable. A scope anchor comes from the expanded scope
vocabulary only when the literal term recurs in enough independent sources. An
emergent concept comes from the bounded corpus neighbourhood. Neither is
automatically accepted.

The ontology is deliberately outside the truth path. It cannot add support to a
claim, infer causation, or promote a relation. Selecting a concept merely opens
its exact excerpts or starts the normal evidence route. Lens is a projection,
not a second store.

The optional Concept Lens presentation is a separate, ontology-bound view
artifact. It can replace raw automatic labels with reviewed questions and
plain-language area names, but it cannot add concepts, relations, sources, or
support. The interface begins with a decision map and opens only the selected
area, so internal cluster indices never become the researcher's navigation
model.

The removed global cartography experiment remains a negative evaluation record.
Its flat lexical, retired neural-reranking, and document-first E5 maps were
deterministic but not navigationally coherent. The replacement is smaller:
local scope, FTS-first neighbourhood, no required embedding model, and no
global ontology claim.

## Interactive research sessions (0.2 development candidate)

The 0.2 development candidate adds a continuing investigation around the stable
evidence route. It does not store a chat transcript as source truth and does not
create a second semantic layer.

```text
ResearchSessionBrief
  → ResearchSession bound to Library + snapshot + policy + Collections
  → typed append-only SessionEvents
  → exact references to existing packets, fragments, candidates, and answers
  → deterministic compact ResearchSessionState
  → existing evidence gates and human ReviewDecision
```

The event stream is authoritative; its state is a disposable projection. Every
event is content-addressed and names the previous event hash. Reordering,
omission, foreign-session binding, mutation, and appending after closure fail
closed. Conversation events can explain why a path was tried, rejected, or left
open, but they cannot support a claim unless they link to an existing exact
evidence artifact.

The Session Spine, durable SQLite store, least-context Python facade, thin stdio
MCP transport, and GET-only Session Lens are implemented. Migration
0013 persists the immutable session receipt and typed hash-chained events,
validates exact artifact closure, and rebuilds compact state after a cold reopen.
The facade and MCP return compact journal state separately from the current
exact evidence projection. `AgentEvidencePacket/1.2` pairs every selected exact
fragment with a human-readable source title/URI and immutable source identity,
marks the set as `retrieved_candidates`, and reports compact source-dominance
diagnostics; it still omits the materialized corpus-wide read manifest. Neither
surface exposes acceptance, operator decisions, or closure. Session Lens renders the
brief, chronological journal, gaps, drafts, rejected paths, and exact
packet-backed passages without becoming another truth store. See
[the 0.2 specification](INTERACTIVE_RESEARCH_MEMORY_SPEC.md) and
[roadmap](ROADMAP_0.2.md).

### Gate-bound answer route

Raw recall remains an immutable `fts_v1` trace. Answer preparation is a
derived, provider-free step over that trace:

```text
AgentEvidencePacket(retrieved_candidates)
  → deterministic candidate-quality guards
  → EvidenceCoverageGate
  → if incomplete: source-local FTS over at most three already found Sources
  → EvidenceCoverageGate replay
  → answer | gap | blocked
```

The quality guards remove only explicit bibliography/reference headings,
index-like layouts, table fragments, and candidates with no meaningful query
overlap. They do not rewrite the persisted ranking receipt. Source-local repair
reuses the exact session scope and searches only already authorized fragments;
its result is bound into `AgentAnswerPreparation/1.0` and cannot replace the
original EvidencePacket.

`record_draft` accepts an answer only with an exact preparation. The facade
repeats filtering, local repair, and the Gate immediately before appending the
draft, then attaches the original packet and matched exact fragments to the
session event. A blocked preparation cannot be recorded as an answer. This
controls the Dithyramba journal boundary; it cannot prevent an external model
from emitting ungrounded text outside the system.

### Ephemeral session recall cache

The first recall in one process builds a `RecallScopeSession` from the exact
Library, snapshot, policy, purpose, Collections, and exclusions. It owns the
already-authorized read-set and one in-memory SQLite FTS5 index. Later questions
in that same scope query the existing index instead of reading and indexing the
corpus again.

The cache is an expendable capability, not durable memory:

- it is process-local, bounded by an LRU capacity, and explicitly closable;
- scope drift, snapshot drift, service substitution, or reuse after close fails
  before a new run is written;
- it stores no new source authority or ranking truth;
- every question still receives an independent `QueryRequest`, ProcessingRun,
  `ReadReceipt`, `EvidencePacket`, and session event;
- the first completion fully validates the immutable shared read set; later
  questions may reuse only the explicit process-local scope capability and
  revalidate every selected fragment before commit;
- a restart simply rebuilds it from the immutable source and policy record.

The M1 PhD stress corpus measured 56.887 s to prepare 263,363 permitted
fragments, 72.647 s for the first strict completion, and 13.557 s for a second
question in the same session. This does not make cold startup instant; it makes
repeated work proportional to the bounded query and selected proof instead of
revalidating the full corpus closure on every turn.

### Agent and human adapters

```text
Python AgentResearchFacade
  ├─ stdio MCP: agent tools; JSON-RPC protocol on stdout, logs on stderr
  └─ Session Lens: GET-only human projection; no mutation or acceptance route
```

The MCP server implements protocol lifecycle, `tools/list`, and `tools/call`
without duplicating research logic. It exposes `open_session`, `recall`,
`session_context`, `prepare_answer`, `record_draft`, `record_gap`, and
`reject_path`. It does not expose candidate acceptance, human decisions,
promotion, or closure.

Modern Session Lens events reuse the compact command receipt and direct
fragment metadata lookup. Full packet reconstruction remains available for
audit and as a legacy fallback, but it is no longer part of ordinary page
rendering. On the same M1 corpus, a 24-fragment session projection measured
0.062–0.064 s with a stable hash and zero model calls.

## Storage model

| Material | Storage | Authority |
|---|---|---|
| source versions, fragments, identities, policies, snapshots, shared read sets, packets, reviews | SQLite plus content-addressed blobs | durable record |
| IdeaTrace candidates, answer projections, and their closure/judgment receipts | append-only canonical JSON in SQLite | durable candidate/audit record; not accepted truth |
| research sessions and hash-chained events | append-only canonical JSON plus relational bindings in SQLite | durable research journal; chat and drafts remain non-evidence |
| session authorized read-set and FTS index | process-local memory plus SQLite FTS5 | rebuildable scoped discovery capability; destroyed on close or eviction |
| one-shot FTS index | SQLite FTS5 | rebuildable discovery index |
| semantic vectors and reranker caches | optional local model data | rebuildable discovery aid |
| graph and Atlas projections | versioned derived artifacts | navigational view |
| original corpus | operator-owned files | read-only input |

Runtime databases, WAL files, indexes, model weights, and backups belong in an
explicit application-data root outside the repository and outside synchronized
corpus folders.

## Stable route

```text
source files
  → SourceVersion + SourceFragment + SourceAddress
  → CorpusSnapshot + compiled AccessPolicy
  → one protected read + local FTS candidate discovery
  → one persisted EvidencePacket/1.0 per question
  → inspect / read receipt / exact replay
  → scoped human review
```

This route needs no external model.

## Default computational profile

The core is **lexical-first, evidence-first, and embedding-optional**. Source
identity, FTS5 recall, exact addresses, packets, gates, receipts, and review
work on CPU without model weights, a vector database, a graph database, or a
corpus-wide semantic pass.

This is an operational advantage, not a claim that lexical retrieval is always
more accurate. FTS is strong for names, quotations, dates, terminology, and
deterministic replay. It can miss distant paraphrases or implicit concepts.
Optional rerankers or embeddings may repair those measured cases, but they stay
rebuildable adapters: they cannot grant access, certify a claim, or replace the
route to exact source text.

## Expanded discovery and evidence route

```text
FTS50
  → deterministic lexical repair
  → EvidenceCoverageGate
  → FTS100 only if a requirement remains uncovered
  → optional QueryCloud q1/q2 for the named gap
  → final proof scan
```

The default remains the persisted FTS route. Query expansion is a bounded
discovery aid and cannot replace the original question or become evidence. Any
future default expansion must atomically persist its plan and survive cold
byte-exact replay.

## Security and failure semantics

- Access defaults to deny.
- Scope identity is checked before text materialization.
- Corpus text is untrusted data, never instructions.
- The HTTP surface binds to `127.0.0.1` and pins one existing Library.
- Parser work is bounded by size, time, and subprocess limits.
- Empty, partial, skipped, refused, failed, stale, and truncated states remain
  explicit; they are not rewritten as success.
- Backup and restore verify paths, modes, hashes, schema, migrations, rows,
  event stream, and blob closure.

## What the architecture does not claim

Dithyramba can preserve what a source says and how that passage was used. It
cannot guarantee that a source is authentic, unbiased, independent, or complete.
Those are research-method decisions that must remain visible to the operator.
