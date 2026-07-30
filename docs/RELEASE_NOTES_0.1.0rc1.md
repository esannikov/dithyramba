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

### Storage stays small

Schema v11 adds two append-only tables containing canonical JSON:

- `idea_traces`;
- `reasoning_closure_results`.

No graph server or vector database is required. A graph view can be rebuilt
from the bounded step list.

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

- Existing Libraries require the checksummed v11 migration before normal open.
- The optional dependency name changes from `cartography` to `ontology`.
- Stable FTS recall, `EvidencePacket/1.0`, and existing review decisions remain
  unchanged.
- IdeaTrace is an experimental `0.1.x` contract. Trace generation and human
  acceptance UI are intentionally not part of this release candidate.

## Validation boundary

The canonical local gate completed with 2,666 passing tests, two declared
browser skips, strict typing across 256 files, 95.0573% exact combined
line/branch coverage, and no known vulnerable third-party dependency.

The committed tests cover canonical identities, topology, stale and tampered
artifacts, failed semantic claims, append-only persistence, corruption,
cold-process-equivalent replay, migration, and CLI invocation. These checks
establish engineering closure. They do not yet establish that generated traces
are insightful, complete, or superior to expert human synthesis.
