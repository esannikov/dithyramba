# Dithyramba 0.1.0rc1

> Historical release note. It describes the `0.1.0rc1` package boundary and
> is not the current package version. Current release notes are in
> [`../RELEASE_NOTES_1.0.0.md`](../RELEASE_NOTES_1.0.0.md). It retains
> its original migration-era verification receipt. The current stable v1 package
> keeps the retained tables in one clean schema baseline and adds scoped
> cache reuse, stdio MCP, and Session Lens; see the
> [changelog](../../CHANGELOG.md), [architecture](../ARCHITECTURE.md), and
> [v1 cleanup roadmap](../V1_ROADMAP.md).

This release candidate makes two experimental paths smaller and more explicit:
orientation before a good question, and inspectable synthesis after evidence
has been checked.

## What changed

### Candidate Ontology replaces global Cartography

The former attempt to partition an entire mixed corpus into one automatic map
was deterministic but not consistently useful to navigate. It has been removed
from the package surface.

The replacement starts from one named scope, retrieves a bounded and
source-balanced neighbourhood with local FTS, and extracts candidate concepts
and exact `co_occurs_with` links. Concept Lens shows this projection without
promoting a concept or relation into accepted memory.

### IdeaTrace makes synthesis inspectable

`IdeaTrace/1.0` records a small public chain of statements. Each step names
whether it extracts, compares, connects, or infers; derived steps name their
premises and carry a concise warrant. Every step is bound to the exact
claim-evidence case and semantic judgment receipt that supports it.

`ReasoningClosureGate` then verifies those bindings and the step topology. It
does not generate prose, call a model, or decide historical truth. Its strongest
result is `review_eligible`, leaving acceptance to a separate human decision.

### Proposition-level prose remains inspectable

`AnswerProjection/1.0` labels exact final-answer spans as fact, synthesis,
hypothesis, question, or framing. The deterministic
`PropositionCoverageGate` applies a role-specific closure rule instead of
forcing the whole paragraph to equal one verified claim.

The migration-era schema v12 stored the projection and its semantic judgment
receipt as append-only canonical JSON; the clean v1 baseline retains those
tables without replaying historical migrations. Research Atlas may expose a sparse subset of those
source-traceable phrases: clicking one in Lens activates its named evidence and
opens the exact bound passage. Unbound framing stays visually plain.

### Storage stays small

The migration-era v11/v12 sequence added four append-only tables containing
canonical JSON; the clean v1 package includes them in its single baseline:

- `idea_traces`;
- `reasoning_closure_results`.

- `answer_projections`;
- `answer_projection_receipts`.

No graph server or vector database is required. Graph and Lens views can be
rebuilt from these bounded public artifacts.

## New command

```bash
dithyramba reasoning-check \
  --trace /absolute/trace.json \
  --case-set /absolute/case-set.json \
  --entailment /absolute/entailment.json \
  --json
```

The command exits with status 0 only for a closed trace. Missing, stale,
uncertain, or rejected evidence exits non-zero with an explicit decision.

## Compatibility notes

- At the time of this historical release, existing Libraries required the
  checksummed migrations through v12. The clean v1 candidate does not convert
  pre-v1 Libraries in place; it fails closed and requires an explicit rebuild
  or separately reviewed importer.
- The optional dependency name changes from `cartography` to `ontology`.
- Stable FTS recall, `EvidencePacket/1.0`, and existing review decisions remain
  unchanged.
- IdeaTrace is an experimental `0.1.x` contract. Trace generation and human
  acceptance UI are intentionally not part of this release candidate.
- Answer projections are durable but remain opt-in candidate records. Their
  persistence does not grant acceptance or generate a projection automatically.

## Validation boundary

The canonical local gate passed all 2,591 collected tests, including the two
real-Chromium browser cases, with zero skips. Strict typing across 260 files,
95.0053% exact combined line/branch coverage, and the dependency audit passed.

The committed tests cover canonical identities, topology, stale and tampered
artifacts, failed semantic claims, append-only persistence, corruption,
cold-process-equivalent replay, migration, and CLI invocation. These checks
establish engineering closure. They do not yet establish that generated traces
are insightful, complete, or superior to expert human synthesis.

Empty Lens trace metadata is omitted from canonical serialization, so existing
`CompactMemoryPacket/1.0` and `/1.1` identities remain byte-compatible.
Non-empty view-only trace spans fail closed when a legacy compact connector
cannot preserve their meaning.
