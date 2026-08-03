# Mars IdeaTrace-24 evaluation card

This internal development evaluation asks a narrow question: once the relevant
passages have already been selected, does an inspectable reasoning artifact
help Dithyramba detect and repair answers that say more than their evidence?

It does **not** remeasure retrieval. Every mode received the same frozen
evidence packets. The source corpus itself is not redistributed with the public
repository.

## Frozen design

The 24 questions were divided evenly:

| Group | Count | Required behaviour |
|---|---:|---|
| Direct facts | 6 | Return one precise value or mechanism from one passage. |
| Multi-source synthesis | 6 | Combine two evidence roles without inventing a stronger conclusion. |
| Contested claims | 6 | Preserve the limitation or counterevidence and qualify the answer. |
| Unsupported traps | 6 | Refuse the attractive but unproved premise. |

Three result stages were compared:

1. a normal answer with citations;
2. the same task expressed as public, inspectable `IdeaTrace/1.0` steps;
3. one repair pass in which the agent received the exact failed gate reasons
   for the 12 complex answers.

`IdeaTrace` contains public statements, named operations, premises, short
warrants, and exact claim-evidence bindings. It is not a private
chain-of-thought transcript.

## Results

| Mode | Complete answer | Correct answer status | Overclaim | Complete evidence closure | Safe abstention |
|---|---:|---:|---:|---:|---:|
| Normal answer, first pass | 21/24 | 22/24 | 2/24 | 6/18 | 6/6 |
| IdeaTrace, first pass | 21/24 | 22/24 | 3/24 | 6/18 | 6/6 |
| IdeaTrace after one gate-directed repair | **23/24** | **24/24** | **1/24** | **16/18** | **6/6** |

The first IdeaTrace generation did not make the model intrinsically better.
Its value was diagnostic: the system could identify which public proposition
had no adequate support, which qualified answer omitted its open gap, and
which citation no longer matched the exact allowed span. Ten of the 12 complex
answers closed after one targeted repair.

The 18-case closure denominator excludes the six unsupported traps. Those
traps are measured separately as safe abstentions.

## What the 24 questions asked

### Direct facts

| ID | Question | Final answer | Result |
|---|---|---|---|
| D01 | What communication delay does an Earth-to-Mars spacecraft rapidly approach? | Approximately 20 minutes. | passed |
| D02 | Why does one traditional parachute not scale to human Mars landing? | Human-scale mass needs continuous retropropulsive braking. | passed |
| D03 | How long is the fast abort-return window for the specified high-thrust trajectory? | The first 40 days. | passed |
| D04 | How deep should Icebreaker sample icy regolith? | Up to 2 metres. | passed |
| D05 | What wind-sampling resolution does the SuperCam microphone provide? | One sample every 0.01 seconds. | passed |
| D06 | What global equivalent water layer does the cited ancient-ocean model require? | Approximately 700 metres. | passed |

### Multi-source synthesis

| ID | Question | Final answer after repair | Result |
|---|---|---|---|
| M01 | What operational need follows from the near-20-minute delay and the research gap around anomaly response? | Onboard tactical decision support is needed; the stronger claim about sufficient anomaly response was not established. | blocked for review |
| M02 | Why should landing-site elevation be considered in both EDL and ascent architecture? | Elevation is an ascent trade-space parameter, while the packet leaves separate human-EDL requirements open; it does not prove their full causal link. | passed after narrowing |
| M03 | What do Gale data and future accessible-ice mapping contribute together? | They provide two comparison axes for exploration zones, not an automatic winner. | passed |
| M04 | Does a plan to produce 300 tonnes of LO2/LCH4 make an early base independent of Earth? | No. It closes a local fuel loop while food and other systems still depend on supplies. | passed |
| M05 | Why do high water recovery and fresh produce not yet form a closed life-support system? | Water recovery does not close the oxygen loop, and fresh food initially supplements stored food. | passed |
| M06 | What conservative planning principle follows from long abort trajectories and unresolved human-system risks? | Do not assume rapid return after the early window; the packet supplies no numeric reserve for the remaining risks. | content acceptable; exact quote blocked |

### Contested claims

| ID | Question | Final answer | Result |
|---|---|---|---|
| C01 | Is Gale proved to be the unconditionally best first human landing site? | No. It has strong resource arguments, but the packet does not compare all candidates. | passed |
| C02 | Can the first base be fully operationally independent of Earth? | Not on this evidence: local fuel is planned, while oxygen and stored-food gaps remain. | passed |
| C03 | Is full Mars terraforming feasible on human timescales? | Not established. Possible warming does not close atmospheric, resource, and biological prerequisites. | passed |
| C04 | Is multigenerational human reproduction at 0.38 g known to be safe? | No. The required long-duration human evidence does not exist in the packet. | passed |
| C05 | Are human EDL and Mars ascent risks practically solved? | No. A future sample-return MAV test and unvalidated human powered-descent models do not close those risks. | passed |
| C06 | Do the sources show that solar power alone can supply the first base? | No. SEP cargo transport and a surface-load range do not prove a solar-only surface architecture. | passed |

### Unsupported traps

| ID | Unsupported premise | Behaviour | Result |
|---|---|---|---|
| U01 | A one-bar atmosphere was created and retained in Jezero. | Evidence packet does not contain such an experiment. | correct abstention |
| U02 | Sub-millisecond Earth-Mars communication was demonstrated. | Packet instead describes delay approaching 20 minutes. | correct abstention |
| U03 | A crewed MAV already carried people from Mars to orbit. | Packet describes a future sample-return MAV. | correct abstention |
| U04 | A long-term human study proved pregnancy safe at 0.38 g. | No such human study appears in the packet. | correct abstention |
| U05 | The first human birth on Mars was documented. | No such event appears in the packet. | correct abstention |
| U06 | An experiment increased global Mars pressure by 10%. | No such completed experiment appears in the packet. | correct abstention |

## The two unresolved cases

**M01 — semantic scope.** The response reasonably inferred a need for onboard
decision support. The judge treated its wording about anomaly-response
sufficiency as broader than a passage that only established a research gap.
This may be a real overclaim or an over-strict judgment; it requires human
adjudication or an independently configured second judge.

**M06 — exact quotation.** The final meaning was acceptable, but the quote
changed a source comma to a period. The exact-quote gate correctly refused to
call it verbatim. The appropriate repair is to show the allowed span for
recopying, not to silently rewrite and accept it.

## Architectural consequences

The run exposed a gap between accepted atomic claims and fluent final prose.
All 29 atomic claims in the normal-answer mode could pass while two short
answers still overclaimed. Dithyramba therefore added:

- `AnswerProjection/1.0`, which divides the exact displayed answer into
  proposition spans labelled `fact`, `synthesis`, `hypothesis`, `question`, or
  `framing`;
- `AnswerProjectionJudgmentReceipt/1.0`, which records the semantic review of
  those roles;
- `PropositionCoverageGate`, which applies a different closure rule to each
  role without turning a synthesis or hypothesis into fact;
- schema v12 append-only persistence for the projection and its judgment
  receipt;
- sparse Atlas trace spans, so a reader can select a coloured phrase in Lens
  and open the exact source evidence attached to that phrase.

The deterministic gates remain local and provider-free. The semantic judgment
that authored the receipt may use a model or a human reviewer.

## Cost and limitations

The development experiment used five model passes: two first generations, one
joint semantic review, one targeted repair, and one review of the repair. The
recorded total was approximately **188,106 input tokens** and **37,992 output
tokens**.

That is the cost of designing and diagnosing the new contract, not the normal
cost of one query. A normal runtime path needs one answer generation; exact
span validation, proposition coverage, persistence, and reasoning closure make
no model call. Semantic judgment can be reserved for derived or high-stakes
propositions.

This evaluation is not an independent benchmark:

- frozen packets excluded retrieval quality from the comparison;
- one model judge is not ground truth;
- the source corpus and heavy model outputs are not redistributed;
- 24 cases reveal failure modes but do not establish cross-domain superiority.

## Next validation

Before making a broader quality claim, the recommended follow-up is:

1. freeze 12 held-out Mars questions unseen by the generator;
2. include temporal, causal, quantitative, and conflict-resolution cases;
3. require at least six cases with independent source families;
4. human-adjudicate complex propositions before examining model verdicts;
5. replay the same protocol on a non-Mars corpus.
