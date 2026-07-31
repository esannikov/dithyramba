# Evidence-grounded reasoning

Dithyramba can now preserve a short, inspectable path from exact evidence to a
working conclusion. The artifact is called `IdeaTrace/1.0`.

It is deliberately **not** a model's private chain of thought. An IdeaTrace
contains only material a researcher can inspect:

- a short statement for each step;
- the operation used: `extract`, `compare`, `connect`, or `infer`;
- explicit earlier steps used as premises;
- a concise public warrant for a derived step;
- a qualifier: `direct`, `bounded`, or `contested`;
- exact bindings to an existing claim-evidence case and semantic judgment
  receipt;
- explicit unresolved questions and the next evidence needed.

## Where it sits

```text
source fragment + exact address
  → compact evidence packet
  → research answer claims
  → exact citation gate
  → required-facet coverage gate
  → claim/evidence semantic receipt
  → IdeaTrace candidate
  → ReasoningClosureGate
  → review-eligible trace or an explicit failure/review requirement
  → separate human decision
```

The closure gate makes no model call. It checks content identities, stale
bindings, step topology, claim text, claim state, semantic verdicts, and the
operation/qualifier policy. A failed or uncertain source claim cannot silently
become a verified reasoning step.

## Letting the answer breathe

An `IdeaTrace` explains how accepted claims were connected. It does not by
itself decide which sentences in the final prose are facts, syntheses, or
research hypotheses. The optional `AnswerProjection/1.0` supplies that missing
display layer.

It partitions the exact `short_answer` into contiguous spans:

| Role | What must close |
|---|---|
| `fact` | at least one directly supported factual claim |
| `synthesis` | named supported premises plus a `bounded_synthesis` judgment |
| `hypothesis` | named supported premises, explicit hypothesis label, falsifier, test question, and next evidence |
| `question` | named premises that explain why the question is worth asking |
| `framing` | no factual binding and a `non_propositional` judgment |

The `PropositionCoverageGate` returns `passed`, `failed`, or
`review_required`. Only a passed projection may be displayed as a checked rich
answer. Passing does not turn a synthesis or hypothesis into accepted memory.

This is stricter than checking citations at the end of a paragraph, but less
restrictive than requiring the whole paragraph to be logically equivalent to
one verified claim. The strictness follows the epistemic role of each span.

## What the decisions mean

| Closure result | Meaning |
|---|---|
| `passed` | Every public step closes over the exact accepted claim-evidence receipt. The trace may be shown to a reviewer. |
| `failed` | At least one step relies on a claim that the semantic evidence gate rejected, or violates the frozen operation policy. |
| `review_required` | The artifacts are stale, missing, uncertain, or not exactly bound. No automatic conclusion is allowed. |

`passed` means structurally closed, not historically or scientifically true.
The result is called `review_eligible`; it is never an automatic promotion to
accepted memory.

## Minimal storage

Schema v11 adds two append-only SQLite tables:

- `idea_traces` stores one canonical, content-addressed candidate JSON;
- `reasoning_closure_results` stores its canonical deterministic closure
  receipt.

No graph database, vector database, or hidden transcript is added. A graph view
can be rebuilt from the at-most-32 explicit steps. The original source chain
remains in the existing packet, case-set, and judgment artifacts.

## Verify an artifact

```bash
uv run dithyramba reasoning-check \
  --trace /absolute/path/idea-trace.json \
  --case-set /absolute/path/claim-evidence-case-set.json \
  --entailment /absolute/path/claim-evidence-result.json \
  --json
```

The command is provider-free and returns a canonical
`dithyramba.reasoning_closure/1.0` receipt. It exits non-zero when closure fails
or needs review.

## Current limit

The release implements contracts, validation, replay, persistence, corruption
checks, and a CLI verifier for IdeaTrace. `AnswerProjection` and
`PropositionCoverageGate` are currently typed, tested in-memory contracts. They
are not yet persisted, exposed through CLI/HTTP, automatically generated, or
connected to a human acceptance workflow. Those surfaces remain separate so a
convenient synthesis cannot grant itself authority.
