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

## Interactive answer boundary

The pure evaluator remains public under `dithyramba.evidence`; ordinary
`recall` still returns candidates and never invents a pass. The interactive
facade and stdio MCP now bind it to the answer route:

1. `recall` returns `AgentEvidencePacket/1.2` with
   `admission_state: retrieved_candidates`;
2. `prepare_answer` removes deterministic bibliography, index, table, and
   no-overlap noise;
3. the Gate evaluates the exact remaining fragments;
4. if an answerable question is incomplete and the missing requirement has
   literal anchors, a bounded FTS repair searches inside at most three Sources
   already found by the broad recall;
5. the Gate evaluates the combined exact candidates again;
6. when the result is `ready`, the answer packet is reduced to an
   inclusion-minimal set of Gate-matched fragments: removing any remaining
   fragment would make the declared evidence roles incomplete;
7. `record_draft` replays the same preparation and refuses any state other than
   `ready`.

The original FTS receipt is never rewritten. The local drilldown is recorded in
`AgentAnswerPreparation/1.0` with its query, source IDs, result hash, selected
fragment IDs, and filtered count. The draft event links the original packet and
the exact matched source fragments.

The compact answer packet is a projection, not a destructive rewrite. The
original retrieval packet, local-search receipt, rejected-fragment assessments,
and provenance addresses remain durable and inspectable. Evidence-gap diagnostics retain
their broader candidate set because the operator may need to understand what is
still missing.

## Precision discipline

Literal matching is only as precise as the Gate specification. Put the domain
anchor and the evidence role in the same `EvidenceRequirement` when they must
co-occur in one passage. For example, an art-research explainability claim
should require an art anchor, an explainability anchor, and a limitation anchor
inside one fragment. Separate broad requirements could otherwise combine an
unrelated art passage with a generic machine-learning passage.

Avoid weak alternatives such as `model`, `human`, `project`, or `expert` unless
another anchor group narrows their meaning in the same requirement. Named-example
questions whose entities are not known in advance should remain `partial` or
`insufficient` until a stronger specification, human review, or semantic
claim-evidence check is available.

This proves declared evidence-role coverage immediately before the journal
accepts a source-backed draft. It does not prove entailment of every sentence,
historical truth, or human acceptance. An external model can still emit text
outside Dithyramba; the system controls its own answer/journal boundary.

For generated prose, the optional `ClaimEvidenceEntailmentGate` and
`PropositionCoverageGate` provide a separate post-generation boundary. They do
not turn retrieval scores or literal role coverage into truth; their result and
any model or human reviewer used by the application must remain disclosed.

Corpus-specific replays and measurements remain development evidence. They do
not establish general retrieval quality or evidence sufficiency.
