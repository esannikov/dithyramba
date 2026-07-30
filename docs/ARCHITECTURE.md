# Architecture

Dithyramba separates four jobs that are often collapsed into one opaque
“knowledge” step: preserving sources, finding candidates, checking evidence,
and recording human decisions.

![Dithyramba memory architecture](assets/dithyramba-memory-map.svg)

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

Optional embeddings, rerankers, aliases, neighbouring fragments, or derived
queries may improve discovery. They are replaceable ranking aids and cannot
grant access or certify evidence.

### 3. Evidence

The stable route persists `EvidencePacket/1.0`: the query identity, exact
snapshot and policy scope, selected passages, source addresses, profile, read
receipts, and packet hash required for exact replay.

When several questions share one exact scope, schema v10 stores their permitted
fragment manifest once as a content-addressed `CorpusReadSet`. Each question
still has an independent `QueryRequest`, processing run, `ReadReceipt`, packet,
and review history. Sharing storage never merges questions or permits a scope
to cross a Library, snapshot, policy, Collection set, or purpose boundary.

The experimental adaptive route adds `EvidenceCoverageGate`. Its caller freezes
explicit requirements such as actor, date, source role, mechanism, direction,
or independent provenance group. The gate scans proof-eligible passage bodies
and reports which requirements are covered. Missing requirements remain an
`EvidenceGap`; a high search score cannot fill them.

### 4. Memory in use

Review decisions are scoped and append-only. Reading Room and Lens project the
stored state for a person. Connectors produce compact source-closed packets for
another agent. These surfaces do not become new truth stores and do not mutate
memory during read-only inspection.

## Experimental universal cartography

Question-led recall works when the operator already knows what to ask. A mixed
notebook or research library may first need a map. The experimental
`dithyramba.cartography` package adds a rebuildable navigation layer without
promoting clusters into source truth:

```text
authorized SourceFragments
  → conservative projection admission
  → one exact local vector-generation receipt
  → HDBSCAN hard areas plus explicit unassigned mass
  → content-addressed, text-free AreaMap
  → deterministic InquirySignals
  → human or on-demand model formulates an Inquiry candidate
  → selective recall and evidence closure
  → reviewable BoundedTrace
```

`AreaProjection` stores canonical fragment/source/version references,
text/address/vector hashes, memberships, representatives, and a boundary hash.
It stores no source text or lexical labels. A readable area name or excerpt is
an independent local view that can be rebuilt or translated without changing
map identity.

Cartography admission never deletes source memory. It can keep obvious layout
markup or unreadable OCR out of dense geometry while the exact original remains
available to FTS and provenance inspection. The transform profile belongs to
the vector-generation receipt. `InquirySignal` records an observable bridge,
source concentration, or unmapped mass; it is not itself a research question.

This path remains experimental until independent representative review, a
cold-process replay, and selective inquiry/evidence tests are complete. It is
not invoked by the stable CLI route.

The first two-corpus screen rejected every tested global partition profile.
TF-IDF/SVD, raw Harrier, Harrier with batch PCA, and a document-first
multilingual-E5-small hierarchy each collapsed, rejected too much mass, or
changed topology after a small input refinement. This is a representation
failure, not evidence that cluster labels should be promoted into knowledge.
The next design step separates a deterministic structural-facet map from local
semantic neighbourhoods created around a selected source, concept, or inquiry.
No further model sweep is part of the current release path.

## Storage model

| Material | Storage | Authority |
|---|---|---|
| source versions, fragments, identities, policies, snapshots, shared read sets, packets, reviews | SQLite plus content-addressed blobs | durable record |
| FTS index | SQLite FTS5 | rebuildable discovery index |
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

## Adaptive route

```text
FTS50
  → deterministic lexical repair
  → optional q0-anchored reranking
  → EvidenceCoverageGate
  → FTS100 only if a requirement remains uncovered
  → optional QueryCloud q1/q2 for the named gap
  → final proof scan
```

The adaptive route is currently an in-memory Python API. It will not replace the
stable route until every intermediate artifact is atomically persisted and a
cold process can reconstruct the exact result bytes and hashes.

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
