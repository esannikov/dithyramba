# Design system

## Register and direction

Register: warm archival research workbench.

The Reading Room should feel like a precise research atlas open on a quiet desk in
warm daylight: calm enough for long inspection, dense enough to compare evidence,
and explicit about uncertainty. The core pattern combines question-led navigation,
a three-part reading flow, progressive disclosure, a focused hypothesis graph, and
an evidence inspector that opens exact source passages.

## Palette

Use explicit solid colors and semantic tokens. Do not use gradients,
glassmorphism, neon, or alpha-heavy overlays. Warm off-white surfaces are
intentional; copper, teal, olive, violet, archive blue, and coral distinguish
source roles without turning the interface into a status rainbow. A restrained
violet is permitted only for qualified synthesis/hypothesis states and scholarly
method sources; it never means truth.

| Token | Value | Role |
|---|---:|---|
| `--canvas` | `#f5efe5` | warm page background |
| `--paper` | `#fffdf8` | primary reading surface |
| `--paper-muted` | `#f8f1e7` | supporting surface |
| `--ink` | `#17130f` | headings and primary text |
| `--ink-soft` | `#3f372f` | body and secondary text |
| `--ink-muted` | `#655b51` | metadata with readable contrast |
| `--line` | `#dfd1c0` | hairline boundaries |
| `--line-strong` | `#c8b49d` | focused structural boundaries |
| `--teal` | `#176f64` | active scope and verified evidence path |
| `--blue` | `#456d83` | archive and informational navigation |
| `--violet` | `#725582` | qualified synthesis and method |
| `--amber` | `#9a691d` | first-person or historical voice |
| `--copper` | `#a65e36` | exact-source identity and archival warmth |
| `--coral` | `#ad4939` | unresolved tension, refusal, warning |
| `--green` | `#2c704e` | supported evidence closure |

Color is never the only state carrier. Every status has text and, where useful, a
shape or glyph. Coral is scarce. Teal means a verified path or an explicit human
acceptance, never model confidence.

## Typography

This is a research product surface, so use locally available,
zero-layout-shift stacks:

- UI, navigation, and factual labels: `-apple-system`, `BlinkMacSystemFont`,
  `"SF Pro Text"`, `"Helvetica Neue"`, Arial, sans-serif.
- Answers, quotations, and source excerpts: `"New York"`,
  `"Iowan Old Style"`, Charter, Georgia, serif.
- IDs, hashes, counts, and timestamps: `"SF Mono", "JetBrains Mono",
  ui-monospace, Menlo, monospace`.

Serif is semantic, not ornamental: it marks a human-readable answer or exact
source voice, never navigation or metadata. Technical IDs never dominate a
research view.

## Layout and spacing

Use a 4-point spacing scale: `4, 8, 12, 16, 24, 32, 48, 64` pixels exposed as
semantic custom properties.

Desktop topology:

1. Context bar: Library, snapshot freshness, policy/scope, read-only state.
2. Question-led navigation on the left.
3. Central workspace: topic orientation, thematic panorama, question/answer,
   hypothesis graph, timeline, or wider memory materials.
4. Evidence inspector on the right, always labelled as the proof for the selected
   accepted answer or finding, or as the non-promoted context for a selected
   candidate.

The central view may use columns or swimlanes for typed objects. It must not use a
force-directed hairball. The preferred graph pattern places source-bound findings
on one side, hypotheses on the other, and uses explicit
`supports`/`qualifies`/`refutes` edges, focus-on-click, and named SourceChips in
the selected hypothesis sidecar. Cards are reserved for distinct selectable
objects; use spacing, rules, lists, and tables for surrounding structure. Never
nest cards.

At narrower widths the inspector follows the selected object in document order.
On phone, question navigation becomes a horizontally scrollable set of 44-pixel
targets and typed connections become a structured list. The evidence inspector
remains present below the main view.

## Interaction

- Reading Room is GET-only and performs no domain mutation.
- Session Lens is GET-only and separates chronological journal material from
  exact packet-backed evidence. A draft, gap, or rejected path never receives
  evidence styling merely because it appears in the session.
- Agent session context, MCP tool names, cache counters, and command receipts
  belong under progressive technical disclosure; the primary human view begins
  with the Brief, the research path, named sources, and exact passages.
- Navigation, filters, pagination, and selection have shareable query parameters.
- Keyboard focus uses a visible 2-pixel blue ring with 3-pixel offset.
- Links and controls have at least 44 by 44 pixel touch targets on coarse pointers.
- Hover may reinforce but never reveal otherwise unavailable information.
- Provide a skip link to the central reading area.
- Avoid modal dialogs. Use in-flow disclosure and native `details` where needed.
- Respect `prefers-reduced-motion`; no information depends on animation.
- Clicking a research question must reveal a plain-language answer in the central
  workspace, not only ranked fragment IDs.
- Every visible SourceChip names the specific source and encodes its role
  (primary voice, patent, archive, scholarship, critical analysis, or method).
- Clicking a SourceChip opens the exact excerpt; raw rank/score lists live only
  under method disclosure.
- Candidate materials use a different inspector contract from accepted
  evidence. The interface states that the material is visible and replayable
  but does not support an accepted conclusion until review.
- Concept Lens begins with a question-led decision map rather than cluster
  indices or a force-directed graph. Its center names the scope; seven bounded
  paths use reviewed plain-language titles and orientation questions. Only the
  selected area reveals automatic corpus terms, relations, and exact excerpts.
- Automatic ontology and human presentation remain separate artifacts. The
  presentation can rename an area or choose a useful entry concept, but it
  cannot add evidence or alter ontology identity.
- A relation labelled `зустрічається поруч із` means literal co-occurrence
  only. It must never be styled or phrased as influence, causation, agreement,
  or proof.
- Overview simplicity comes from progressive disclosure: theme, period and
  status filters fold the wider memory into stable routes instead of deleting
  context.
- The right inspector also names who makes the assertion, the source kind,
  evidence role, independence group, and explicit limitation.
- Main question rows use human sequence numbers. Internal query IDs and model
  labels never occupy a primary column.
- The smallest normal metadata is at least 11px; research-facing explanatory
  text is at least 13px and uses dark ink rather than pale grey.

## Copy

Research Atlas navigation uses these stable human routes:

- What is in this corpus?
- Which questions and hypotheses are source-grounded?
- What supports, qualifies, or refutes each hypothesis?
- How does the subject unfold over time?
- What else does the memory contain, and what deserves review next?
- What is the evidence passport and how should this source be read?

Within an answer or hypothesis, use role colour only for exact stored trace
spans:

- fact — teal underline;
- synthesis — violet underline;
- hypothesis — amber underline;
- research question — blue underline;
- currently selected span — warm gold field plus the active source chip.

The trace decoration must remain sparse. Unbound connective prose keeps normal
ink, and colour never substitutes for the visible source title, asserting
voice, evidence role, limitation, and exact passage.

Ukrainian UI translations should remain plain and consistent. Schema identifiers
appear as supporting metadata, not as the primary label. Never show a generic
`Error 403`; explain that the selected scope does not permit the requested view.

## Mock fidelity inventory

Carry into code:

- explicit context bar and read-only status;
- question-led navigation;
- trustworthy-state summary before the graph;
- typed columns and edge verbs;
- evidence inspector with source/version/fragment lineage;
- mobile inspector preserved below content;
- restrained paper/ink/teal/blue/coral system.

Do not literalize:

- mock-only counts, confidence scores, source names, or timestamps;
- a `confirmed` candidate state not present in the domain model;
- automatic refresh, export, or filters until backed by real contracts;
- untyped visual proximity or co-occurrence edges;
- a source quote outside the authorized packet-backed evidence path.

## Concept Lens finish review — 2026-07-30

- Desktop pass: the first viewport shows one central research question and
  seven readable perspectives; no internal IDs or numeric cluster indices are
  visible.
- Mobile pass at 390 × 844: cards become one column, touch targets remain at
  least 44 pixels, and exact evidence follows the selected concept in document
  order.
- Epistemic pass: `co_occurs_with` is rendered only as “зустрічається поруч
  із”; candidate state and the route to an exact book passage remain visible.
- Automated Impeccable detector: no findings on the final template or CSS.

**Verdict:** accept as an experimental research Lens. The layout is complete;
independent human coherence review of the seven automatic areas remains a
product-quality gate, not a visual defect.
