# Product context

## Product

Dithyramba is a local, source-grounded interpretive memory engine for research and
creative corpora. It turns explicitly scoped source fragments into typed,
reviewable memory objects without collapsing source, interpretation, hypothesis,
and human decision into one truth layer.

The current release line is pre-alpha. Product surfaces must describe what the
running system proves; they must not imply public validation, autonomous learning,
or semantic confidence that has not been measured.

The `0.2` development line adds bounded research sessions so an agent can
continue an investigation from compact state and exact artifact references.
Session notes, drafts, and rejected paths remain journal material. They do not
become evidence or accepted memory merely because an agent recorded them.

## Interactive research memory

The product has one evidence core and two adapters around it:

- an agent uses the local stdio MCP adapter to open a bounded session, recall
  exact evidence, and record drafts, gaps, or rejected paths;
- a human uses Session Lens to inspect the brief, chronological journal, named
  sources, and exact passages without mutating the Library.

The first recall in a live process may build a session-scoped authorized FTS
cache. Later questions in the same exact scope may reuse it. The cache is not
memory authority: it is destroyed on process exit, close, or eviction and can be
rebuilt from the immutable source and policy record. Every question still gets
its own durable request, receipt, packet, and journal event.

This is the product's context-window strategy. The agent receives a small
session state and selected evidence packet rather than the complete corpus or
chat transcript. Full audit artifacts stay local and addressable by immutable
ID and hash.

## Primary operator

The primary operator is a researcher in the broad sense: scientist, historian,
director, screenwriter, producer, curator, writer, or analyst working with a
local Library. They need to answer seven questions quickly:

1. What is this corpus about, and where should I begin?
2. Which questions already have a source-grounded answer?
3. Which hypotheses are supported, qualified, contradicted, or still open?
4. Which exact sources and excerpts support each statement?
5. Who is making each assertion, what kind of source is it, and how independent
   are its corroborating lines?
6. How do events, concepts, findings, and hypotheses connect?
7. What remains unknown, contested, stale, or in need of human review?

Internal schema names may appear in details, hashes, and receipts. Navigation and
headings should use the operator's questions and the accepted Dithyramba
terminology.

## Product principles

- Evidence precedes interpretation.
- Candidate recall, relevance ranking, and evidence sufficiency are three
  separate decisions. Candidate ordering may improve the second; only exact
  proof and the coverage gate decide the third.
- A generated or rewritten query may discover a source fragment but is never
  itself evidence and never replaces the operator's original question.
- AccessPolicy is compiled before any read or projection.
- A visual connection is never stronger than its stored typed edge and evidence
  closure.
- Candidate, accepted, rejected, revised, and superseded are visibly distinct.
- Read-only surfaces perform no hidden mutation, extraction, indexing, or review.
- Empty, partial, refused, failed, stale, and truncated states are first-class.
- Exact source text is opened only through an authorized evidence path.
- The local Library remains usable without a network connection.

## Reading Room

Reading Room is the visual control surface for one explicit Library, snapshot,
policy, and scope. It is not an administration console and not a free-form graph
browser. Its first screen communicates the trustworthy state of memory, then lets
the operator follow typed connections into packet-backed evidence.

The surface is useful when it shortens an evidence audit. It must not turn
co-occurrence, visual proximity, or shared styling into an asserted relation.

## Research Atlas

Research Atlas is the corpus-facing workbench built on the same evidence
contracts. Its first screen orients a human in the subject; it is not a benchmark
viewer. Questions open plain-language answers in the central workspace. Named
SourceChips open exact evidence in a stable right inspector. The hypothesis graph
connects source-bound findings to hypotheses only through explicit relation
verbs such as `supports`, `qualifies`, and `refutes`. A timeline is a first-class
research route, not decoration: every event must resolve to an answer or
hypothesis and then to exact sources.

The evidence passport keeps two axes distinct:

- source voice and kind: memoir, first-person statement, patent, archive,
  museum object, court decision, or scholarship;
- coverage: one, two, or three-plus independent provenance groups, conflict, or
  an explicit gap.

The inspector always names the exact source and asserting voice, states the
evidence role, shows the independence group and limitation, and then opens the
exact local excerpt and original source.

Answers and hypothesis syntheses may contain sparse source-traceable phrases.
Only exact phrases with stored evidence bindings receive an epistemic
underline. Selecting one phrase highlights the corresponding SourceChip and
opens its exact passage in the inspector. Prose without that binding remains
plain, so visual emphasis never implies support that the manifest does not
contain.

Search scores, query IDs, model names, and evaluation labels belong in
progressive method disclosure. They may explain how a candidate was found, but
must not substitute for the answer, the hypothesis state, or its evidence
closure.

## Candidate Concept Lens

Concept Lens is the orientation surface for one deliberately narrow research
scope. It shows automatic candidate areas, concepts, and exact source excerpts
before the operator knows which final question to ask. It does not display a
global ontology of the Library and it does not promote concepts into accepted
memory.

The view distinguishes QueryCloud anchors from phrases that emerged inside the
bounded corpus neighbourhood. Its only automatic relation is
`co_occurs_with`, backed by exact fragments from the named sources. The human
may use a concept to start ordinary recall; only that later evidence route and
review can support a conclusion.

Primary navigation uses a small ontology-bound presentation: readable
perspectives and questions around one central scope. Raw cluster labels and
concept terms appear only after a perspective is chosen. The presentation may
improve explanation, but it cannot change the automatic projection or its
evidence.

### Full-memory projection

The accepted Atlas manifest remains the compact evidence release. An optional,
hash-bound Research View makes the wider memory visible as periods,
themes, collection inventories, candidate materials and review priorities.
The projection is not a second truth store and cannot silently add support to
an accepted answer or hypothesis.

The interface keeps two independent axes:

- **epistemic status:** accepted evidence, working synthesis, candidate,
  deferred, rejected, route-only or metadata-only;
- **reading scale:** panorama, theme, period, material and exact source.

This permits progressive disclosure without destructive filtering. A calm
overview is a reversible compression of the whole memory, not a claim that the
hidden material does not exist.

## Technical boundaries

- Python 3.11+ and the existing FastAPI/Jinja2/SQLite stack.
- Loopback-only server with the existing host/origin security boundary.
- No second frontend framework and no network-loaded runtime assets.
- Accessible semantic HTML, keyboard operation, 200% zoom, and reduced-motion
  support.
- Desktop research use is primary; phone use is a real inspection path, not a
  shrunken desktop.
- Original corpora are read-only inputs. Derived application data lives in the
  explicit Dithyramba data root.
