# How Dithyramba turns sources into research memory

## The simple idea

A normal file search answers: “Where do these words occur?”

Dithyramba tries to preserve a longer chain:

```text
question
  → candidate passage
  → exact source and address
  → evidence requirement it satisfies
  → supported conclusion or explicit gap
  → human decision and later reuse
```

That chain is why the system is a memory rather than a search preset. It keeps
the source, the relation between source and conclusion, the context in which
the relation was accepted, and the history needed to replay or challenge it.

## How it reduces context-window pressure

Dithyramba does not enlarge a model's context window. It changes what must enter
that window. A first local pass reads the permitted corpus and stores exact,
addressable evidence. A continuing session then sends the agent only:

- the bounded research brief;
- a compact projection of recent questions, drafts, gaps, and rejected paths;
- the exact passages selected for the current turn;
- immutable IDs and hashes for the larger local audit record.

The full corpus, full FTS candidate union, materialized read receipt, and entire
conversation remain outside the prompt. A session-scoped FTS cache can reuse the
same authorized read-set for later questions, but it is disposable and grants no
new authority.

```text
agent question
  → stdio MCP
  → AgentResearchFacade
  → exact session scope + local recall
  → EvidencePacket
  → compact evidence for the turn
  → packet reference appended to the session journal
  → human inspection in Session Lens
```

Drafts and chat events explain the path of inquiry. They are not evidence unless
the ordinary source, packet, gate, and human-review path supports them.

## The smallest useful unit

The authoritative atom is not an embedding and not a generated summary. It is
an exact `SourceFragment` belonging to an immutable `SourceVersion`, with a
`SourceAddress` that lets a person reopen the passage.

A fragment can participate in larger objects:

- a `Statement` says what a source, researcher, or system asserts;
- an `EvidenceLink` says how a fragment supports, limits, or contradicts it;
- a `Voice` records who makes the Statement;
- a `Relation` connects accepted objects without replacing their evidence;
- a `ReviewDecision` records what a human accepted, rejected, or deferred.

This resembles a graph because objects have typed links. It is not a neural
network: the links are explicit records with provenance and validation rules,
not learned weights distributed across a model.

## Why text is not converted directly into “memory weights”

An embedding is a vector that places similar texts near one another. Creating
it runs a model; it does not fine-tune that model and does not accumulate new
weights. The vector is useful for finding paraphrases, but it cannot by itself
show who said something or where the supporting passage is.

Dithyramba therefore stores:

- source bytes and exact hashes;
- structured records and typed links in SQLite;
- optional vectors as rebuildable indexes;
- human decisions as separate append-only records.

If a better embedding model appears, vectors can be rebuilt while source IDs,
citations, and decisions stay intact.

## Three questions that must remain separate

### 1. Was it discovered?

FTS, aliases, phrase repair, neighbouring fragments, a vector model, or a
derived query can place a fragment in a bounded candidate list.

### 2. Is it relevant?

FTS rank, phrase overlap, aliases, and bounded query variants may order the
candidate list. A high rank means the passage may be useful, not that it proves
the answer.

### 3. Is the evidence sufficient?

`EvidenceCoverageGate` checks exact passages against explicit
`EvidenceRequirements`: required source role, actor, mechanism, date,
direction, status, independent provenance group, or literal anchor.

Only the third step can mark a requirement covered. If one part is missing,
the result stays partial or becomes an `EvidenceGap`.

## Why the research route starts with FTS

FTS is cheap, deterministic, local, and excellent for proper names, numbers,
patent IDs, dates, quotations, and rare terms. It is also easy to audit.

The route adds complexity only when needed:

```text
FTS50
  → controlled lexical repair
  → Wide Gate
  → FTS100 only if coverage is incomplete
  → QueryCloud q1/q2 only if a named gap remains
  → Wide Gate again
```

This avoids embedding or reranking the whole corpus merely to answer a few
questions. Exact FTS remains available without any model.

## What “Wide Gate” means

Earlier search routes inspected only a short top-ranked window. A correct
passage at rank 60 could be present in the candidate union but invisible to
the proof check.

The Wide Gate scans every body-proof-eligible passage in the bounded union.
It may accept a lower-ranked exact proof, but the result exposes only the
passages that satisfied requirements, not the text of the entire union.

This separates two budgets:

- a larger private budget for finding proof;
- a small outward packet for a person or downstream agent.

## What QueryCloud does

QueryCloud is not a second answer generator. When the Gate names an uncovered
requirement, one optional provider may propose at most two retrieval queries.
They can introduce a synonym, split evidence roles, or guard relation state.

The original question remains unchanged. New candidates are reranked against
that original question. Generated query text is never evidence.

The agent-facing packet also says `admission_state: retrieved_candidates`.
This is intentional: FTS can prove that a passage was retrieved from the
permitted snapshot, but retrieval alone cannot prove that the passage supports
the answer. The packet reports distinct sources, distinct SourceFamilies, and
the largest number of selected fragments from one source or family so an agent
can see source dominance before invoking `EvidenceCoverageGate`.
These counts do not infer that differently encoded PDF and EPUB files are the
same scholarly work. Cross-format editions must be declared in a pinned source
identity manifest; otherwise a SourceFamily count can overstate independence.

## Why provenance saves downstream context

Without Dithyramba, a consumer agent may receive a large Markdown manual or a
pile of books and repeatedly search, read, and judge them inside its expensive
context window.

With Dithyramba, a Connector can request a small source-closed packet:

```text
recipe
precondition
validation
contraindication
gap
```

Each item points to exact fragments. The consumer spends context on the
decision it must make instead of rediscovering where the evidence came from.
Tracking provenance also allows later correction: if a source version changes
or a passage is rejected, dependent packets can be found and refreshed.

Dropping provenance can make a tiny semantic index, but the result cannot
support source chips, challenge a conclusion, distinguish independent sources,
or safely update dependent work.

## Why a graph is useful but not sufficient

A graph is excellent for navigation:

- which people, places, works, and events connect;
- which Statements support or contradict one another;
- which concepts change meaning across Voices and time;
- which research gaps block several downstream ideas.

But a graph edge is an assertion. Dithyramba requires the edge to preserve its
path to exact evidence and review state. Graph traversal remains out of the
default recall route until accepted-only traversal and replay receipts exist.

## What the human interface is for

Lens and ReadingRoom are an API from memory to a person. They should make the
chain readable:

- the question or hypothesis;
- the concise current answer;
- who makes each supporting statement;
- source type and number of independent supports;
- exact highlighted passage;
- counterevidence and open gaps;
- timeline and connections;
- technical details only on demand.

When a question answer or hypothesis contains exact trace bindings, Lens
underlines only those source-traceable phrases. Clicking one selects the named
evidence chip and opens the bound passage. The rest of the prose is deliberately
uncoloured: it may be useful framing, but the view does not pretend that it has
the same proof route.

The interface does not make evidence stronger. It makes the stored relation
between conclusion and proof inspectable.

## Why the stable CLI is FTS-first

The default persisted query must be replayable after the process exits. FTS
provides a local, deterministic candidate route whose exact scope and result
can be stored with `EvidencePacket/1.0`. Bounded lexical variants and
QueryCloud may widen discovery, but they do not replace the original question
or become evidence.

This keeps the auditable route small. A future discovery aid must demonstrate
better evidence recall and preserve cold replay before it enters the default
path.

## What Dithyramba cannot guarantee

- An exact quotation can still come from a weak, biased, or forged source.
- Several citations can belong to one dependent `SourceFamily`.
- A corpus can omit the decisive document.
- Human review can be wrong or apply only to one scope.
- A development fixture can overestimate general performance.

Dithyramba makes these limits representable and visible. It does not remove
the need for source criticism or expert judgment.
