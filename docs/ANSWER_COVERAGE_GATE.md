# AnswerCoverageGate

`AnswerContractGate` proves that an answer used real, packet-authorized source
fragments. `AnswerCoverageGate` checks whether that exact answer used every
facet that the frozen task declared mandatory.

```text
CompactMemoryPacket
  + ResearchAnswer
  + AnswerCoverageSpec
  → AnswerCoverageGate
  → complete | incomplete
```

The specification is operator-authored and content-addressed. Each
`RequiredFacet` names one human-readable facet and the accepted evidence IDs
that may satisfy it. `min_evidence` allows a facet to require one of several
acceptable anchors or multiple complementary anchors.

The task-level contract can also require:

- one of a frozen set of epistemic answer statuses;
- an explicit actionable `open_gap`;
- a minimum number of distinct cited sources.

The result records every covered and missing facet, integer coverage basis
points, status calibration, source diversity, violations, and the exact
content IDs/hashes of both the answer and coverage specification.

## Boundary

The gate performs no semantic inference. It proves actual use of declared
accepted evidence IDs, not historical truth or claim entailment. It must run
after `AnswerContractGate`, which separately replays source paths, hashes and
verbatim quotations.

Facet specifications are part of a Research Brief or frozen evaluation task,
not universal properties of a source corpus. A new research question may need
a new specification without changing the underlying memory. Corpus-specific
evaluation results remain development evidence and are not public performance
claims.
