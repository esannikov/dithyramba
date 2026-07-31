# Dithyramba 0.1.0rc1

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

Schema v12 stores the projection and its semantic judgment receipt as
append-only canonical JSON. Research Atlas may expose a sparse subset of those
source-traceable phrases: clicking one in Lens activates its named evidence and
opens the exact bound passage. Unbound framing stays visually plain.

### Storage stays small

Schema v11 adds two append-only tables containing canonical JSON:

- `idea_traces`;
- `reasoning_closure_results`.

Schema v12 adds:

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

- Existing Libraries require the checksummed migrations through v12 before
  normal open.
- The optional dependency name changes from `cartography` to `ontology`.
- Stable FTS recall, `EvidencePacket/1.0`, and existing review decisions remain
  unchanged.
- IdeaTrace is an experimental `0.1.x` contract. Trace generation and human
  acceptance UI are intentionally not part of this release candidate.
- Answer projections are durable but remain opt-in candidate records. Their
  persistence does not grant acceptance or generate a projection automatically.

## Validation boundary

The canonical local gate collected 2,701 tests: 2,699 passed and two declared
browser cases were skipped until Chromium was provisioned. A separate
real-Chromium run passed both browser cases with zero skips. Strict typing
across 260 files, 95.0172% exact combined line/branch coverage, and the
dependency audit passed.

The committed tests cover canonical identities, topology, stale and tampered
artifacts, failed semantic claims, append-only persistence, corruption,
cold-process-equivalent replay, migration, and CLI invocation. These checks
establish engineering closure. They do not yet establish that generated traces
are insightful, complete, or superior to expert human synthesis.

Empty Lens trace metadata is omitted from canonical serialization, so existing
`CompactMemoryPacket/1.0` and `/1.1` identities remain byte-compatible.
Non-empty view-only trace spans fail closed when a legacy compact connector
cannot preserve their meaning.
