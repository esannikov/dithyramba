# EvidenceCoverageGate

`EvidenceCoverageGate` is the deterministic verification step between retrieved
candidates and an evidence-backed answer.

Retrieval answers: “Which fragments look relevant?” The gate answers the
narrower question: “Do exact fragments cover every evidence role that this
question requires?”

It does not call a model, infer historical truth, or promote a claim into human
canon.

## Why it exists

A high search or reranker score proves similarity, not sufficiency. Typical
false positives include:

- the correct document title without the passage that states the mechanism;
- a court case without the exact holding;
- one side of a two-voice comparison;
- a patent presented as proof of experimental performance;
- a scholarly discussion presented as an official award record;
- two copies or reprints counted as two independent confirmations.

`EvidenceRequirement` makes each missing role explicit. A requirement can bind:

- allowed source identities;
- source kind, family, and authority;
- one or more literal anchor groups;
- precomputed evidence tags;
- forbidden anchors for the wrong claim direction;
- a minimum number of independent provenance groups.

Every anchor group is an OR-list, while all groups in one requirement are
required. All populated metadata selectors are also required. Text matching is
Unicode-normalized, case-insensitive, whitespace-collapsed exact substring
matching. It deliberately does not perform semantic inference.

## Decisions

For questions expected to be answerable:

- `ready` — every requirement and the total independence threshold are covered;
- `partial` — some roles are covered or all roles lack the required total
  independence;
- `insufficient` — no required role is covered.

For questions declared `not_in_corpus`:

- `gap_preserved` — the complete hypothetical evidence needed to overturn the
  gap was not found;
- `gap_challenged` — all declared overturning requirements were found. This
  opens human review; it does not automatically produce an answer.

The result records covered/missing requirement IDs, matched fragment/source/
independence identities, stable rejection codes for the nearest candidates, a
candidate-set hash, spec hash, result hash, and canonical bytes.

## Example

```python
from dithyramba.evidence import (
    EvidenceAnswerability,
    EvidenceCandidate,
    EvidenceCoverageGate,
    EvidenceGateSpec,
    EvidenceRequirement,
)

requirement = EvidenceRequirement(
    key="court_holding",
    label="The exact court holding",
    source_kinds_any=("court decision",),
    anchor_groups=(("marconi showed no invention over stone",),),
)
spec = EvidenceGateSpec(
    query_key="radio_priority",
    question="What did the court actually decide?",
    expected_answerability=EvidenceAnswerability.ANSWERABLE,
    requirements=(requirement,),
)

result = EvidenceCoverageGate(spec).evaluate(candidates)
```

`EvidenceCandidate` requires exact fragment text, a canonical `SourceAddress`,
source-role metadata, and an independence group. Candidate order does not
change the canonical result; rank remains part of the input.

## Current boundary

The module is a pure public Python contract under `dithyramba.evidence`. It is
not yet a default `recall` CLI step and creates no persistent database object.
Consumers may apply it to an authorized `EvidencePacket` or another exact
fragment projection. Persistence and UI promotion require a later versioned
integration contract.

Corpus-specific replays and measurements remain development evidence. They do
not establish general retrieval quality or evidence sufficiency.
