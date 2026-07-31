---
type: architecture_note
id: dithyramba.compact_research_memory.v1_1
status: implemented_library_contracts
updated: 2026-07-27
---

# Compact research memory and answer validation

## Problem

An accepted Research Atlas can improve research quality, but sending the whole
Atlas to every agent repeats irrelevant context.  A raw model can also quote a
real fragment while writing a broader claim than that fragment supports.

The compact path therefore separates three questions:

1. Which accepted Atlas records are relevant to this task?
2. Did the answer use authentic evidence and cover the required aspects?
3. Does each exact quotation support the whole wording of its claim?

## Implemented contract v1.1

```text
human task
  → upstream semantic and/or strong lexical scorer
  → RouteCandidateReceipt/1.0
  → TaskRouter.route_v2 + RouteBudget
  → CompactMemoryPacket/1.1
  → research model
  → ResearchAnswer
  → ResearchAnswerValidator
      1. exact AnswerContractGate
      2. required-facet AnswerCoverageGate
      3. claim/evidence semantic gate
  → accepted | rejected | review_required
```

The core is provider-agnostic.  It records and verifies externally computed
candidate scores and judgments; it does not call an embedding or generative
model itself.

### RouteCandidateReceipt/1.0

The receipt binds one normalized query to the exact Atlas and records:

- Atlas, case and manifest hashes;
- model, runtime and index profile hashes;
- the exact semantic threshold;
- ordered semantic candidates with canonical `float.hex` scores;
- caller-filtered strong lexical-only candidates with integer scores;
- an explicit `routed` or `abstained` decision.

Semantic and lexical score scales are never added or compared.  The receipt is
content-addressed, rejects duplicate/unknown question IDs and fails closed when
the query or manifest has changed.

### Direct seeds and graph expansion

`route_trace` keeps retrieval and context expansion separate:

- `seed` means the upstream search selected that question; it carries channel,
  rank and exact score;
- `graph_expansion` means an accepted Atlas reference linked another question,
  hypothesis, relation, event or gap; it carries its admitted parent and no
  retrieval score.

Consequently `direct_seed_hit@k` and `packet_contains_target` are different
metrics.  A target added only by graph closure is useful context, but it is not
a successful direct retrieval.

### CompactMemoryPacket/1.1 and RouteBudget

The packet contains accepted objects only.  `RouteBudget` has independent hard
caps for questions, linked hypotheses, relations, timeline events, gaps,
evidence and sources.  Admission is atomic: a question is not admitted unless
its required evidence and source closure also fit.  Every refused object gets a
deterministic omission reason.

Three route states are explicit:

- `routed` — at least one complete seed bundle was admitted;
- `no_match` — the scorer abstained or returned no candidates;
- `budget_exceeded` — candidates existed, but no complete seed bundle fit.

Packet ID/hash cover the query, manifest, accepted objects, receipt binding,
budget, trace and omissions.  A consumer must retain and replay the referenced
candidate receipt; the ID/hash pair alone is not a model signature.

Candidate Research View materials remain outside this accepted-memory
packet.

## ResearchAnswerValidator

The façade exposes one final decision while preserving three independent
checks.  It stops at the first failed prerequisite and makes no provider call.

### Stage 1 — exact answer contract

`AnswerContractGate` verifies strict schema, task and packet identity, accepted
IDs, local source boundary, SourceAddress hash and verbatim quotation.  It asks
whether the evidence is authentic and correctly addressed, not whether it
proves the prose.

### Stage 2 — required-facet coverage

`AnswerCoverageGate` counts only evidence IDs actually assigned to claims.  A
frozen `AnswerCoverageSpec` defines required aspects, allowed epistemic states,
minimum source diversity and any required open gap.

### Stage 3 — claim/evidence semantic judgment

`ClaimEvidenceCaseSet/1.0` exposes the least possible context: one exact claim,
its claim state and only its already verified quotations with source/voice/role
metadata.  Wider Atlas synthesis and source passages are excluded.

An external human or model returns a content-addressed
`ClaimEvidenceJudgmentReceipt/1.0`.  The deterministic gate checks the case-set,
judge profile, prompt and raw-response bindings, exact judgment closure and any
unsupported spans.  Supported or qualified claims require
`directly_supported`; an explicit inference may also accept
`supported_inference`.

Only `exact accepted + coverage complete + semantic passed` makes the unified
answer `accepted` and promotion-eligible.  A missing, stale, wrong-profile or
uncertain receipt produces `review_required`.  A model judgment is one review
artifact, not historical truth.

## Compatibility with packet v1.0

The original lexical `TaskRouter.route()` and `CompactMemoryPacket/1.0` remain
replay-compatible.  They use weighted term overlap, `RouteLimits` and legacy
`route_matches`.  They are retained as an auditable baseline, not the current
semantic delivery contract.

This Atlas packet closure is distinct from general Relation traversal.
Accepted-only relation ranking remains disabled until it has its own versioned
receipt and graph-off/on evaluation.  Packet v1.1 follows only explicit
accepted Atlas references for one bounded context bundle.

## Current boundary

The compact packet, exact answer contract, required-facet gate, and
claim/evidence judgment receipts are implemented library contracts. They remain
outside the default persisted CLI recall route.

The active discovery target starts with bounded FTS, may add compact lexical
variants or a conditional QueryCloud, and invokes `EvidenceCoverageGate` only
against explicit requirements. Historic model comparisons and private-corpus
measurements are development records, not public performance claims.

These contracts verify identity, declared coverage, and recorded judgments.
They do not establish universal retrieval accuracy, source truth, or human
usefulness. Promotion still requires a task-specific specification and human
review.
