<p align="center">
  <a href="docs/assets/dithyramba-memory-map.svg">
    <img src="docs/assets/dithyramba-memory-map.svg" alt="Dithyramba architecture: a durable source-to-evidence core, a bounded interactive session loop for agents and researchers, and separate rebuildable and experimental paths" width="100%">
  </a>
</p>

<p align="center"><sub>Open the figure for the full two-plane architecture. [S] source of record · [D] durable · [R] rebuildable · [E] experimental.</sub></p>

# Dithyramba

**Local research memory that keeps every retrieved passage attached to its
source, scope, and review history.**

Dithyramba is a Python and SQLite system for researchers—and for agents working
on their behalf—who return to the same body of sources over time. It turns a
local corpus into versioned, addressable evidence packets that can be inspected,
replayed, challenged, and reused without asking a model to reread everything.

> **Current status:** `1.0.0`, the first stable v1 release. The supported path is
> a local, replayable source-to-evidence memory with CLI, stdio MCP, and a read-only
> Lens. Optional synthesis and navigation views remain review-only: they cannot
> turn generated text into accepted evidence.

## Why it exists

A file search can tell you where words occur. A chat over files can write a
plausible answer. Research usually needs a longer-lived record:

```text
question → exact passage → source and address → evidence role
         → explicit gap or packet → human decision → later reuse
```

Dithyramba preserves that chain. “Memory” here means durable source identity,
exact fragments, scope, receipts, and review decisions. It does **not** mean
fine-tuning a model or silently turning generated prose into fact.

| Approach | What it returns | What persists |
|---|---|---|
| File search | matching files or lines | usually the query and matches |
| One-off LLM analysis | generated prose from the supplied context | whatever the operator saves |
| Dithyramba | bounded evidence packets or explicit gaps | source version, exact address, scope, retrieval receipts, and review history |

File search remains the simplest tool for a one-off name or date. Direct LLM
analysis can be reasonable for a small disposable corpus. Dithyramba becomes
useful when sources are revisited, provenance matters, or downstream agents
need compact context that a human can audit.

## Who it is for

Dithyramba is designed for source-heavy work by historians, scientists,
journalists, curators, analysts, directors, screenwriters, producers, and
research agents. Typical uses include:

- building an evidence-backed biography, documentary world, or scientific case;
- separating first-person testimony, archival records, and later scholarship;
- checking which parts of a question are supported and which remain open;
- following a claim back to the exact local passage that supports or limits it;
- giving another tool a compact evidence packet instead of an entire library.

## How the memory is built

1. **Bound the corpus.** A `Library` is the physical privacy and database
   boundary. `Collections` keep source layers distinct.
2. **Freeze identity.** Each input becomes an immutable `SourceVersion` made of
   exact `SourceFragments` with stable `SourceAddresses`.
3. **Compile scope first.** A `CorpusSnapshot` and default-deny `AccessPolicy`
   decide what a query may read before retrieval begins.
4. **Discover locally.** SQLite FTS5 finds a bounded candidate set. Deterministic
   lexical repair may add query variants, but candidate rank never certifies truth.
5. **Return inspectable evidence.** The stable route persists an
   `EvidencePacket/1.0` with exact passages and receipts, or records that no
   evidence was found.
6. **Keep human judgment separate.** Review decisions are scoped, append-only,
   and never overwritten by a later model run.

An optional `IdeaTrace/1.0` can then expose a short public path from accepted
answer claims to a working conclusion. `ReasoningClosureGate` verifies that
every step closes over the exact claim-evidence receipt. Passing makes the
trace eligible for review; it never promotes the conclusion automatically.

An optional `AnswerProjection/1.0` lets the final prose be richer than a literal
claim restatement. It marks each exact span as fact, bounded synthesis,
disclosed hypothesis, research question, or framing. Facts stay strict;
hypotheses must expose their premises, falsifier, test question, and next
evidence. `PropositionCoverageGate` checks the roles without promoting
exploratory text into accepted memory. The v1 baseline stores the projection and its
judgment receipt as append-only canonical JSON. Lens can render a sparse subset
of these source-traceable spans: clicking a coloured phrase selects the exact
evidence chip and opens the bound passage, while unbound framing remains plain.

Several questions over one exact scope may use `RecallService.recall_batch`.
The v1 baseline stores the shared protected fragment manifest once as a
content-addressed `CorpusReadSet`, while every question still receives its own
request, packet, receipt, and review identity. This changes storage and repeated
read work, not the meaning of the public packet contracts.

### Continue across many agent turns

For continuing agent work, `AgentResearchFacade` binds a `ResearchSessionBrief`
to one frozen snapshot and records questions, exact packet references, drafts,
gaps, and rejected paths as a typed append-only journal. Its authorized read-set
and in-memory FTS index are reused inside the process for later questions in the
same exact scope; closing or eviction destroys that cache. Every question still
persists an ordinary request, run, receipt, and packet. The same narrow facade is
available through local stdio MCP, while Lens session mode exposes a read-only human
journal with exact source inspection. Agent packets include a readable source
title/URI beside every selected fragment, while their full corpus-read audit
manifest stays local and reopenable by ID/hash.

### Durable record and rebuildable aids

| Durable, authoritative record | Rebuildable aid or view |
|---|---|
| source bytes and hashes | FTS index |
| source versions, fragments, and addresses | process-local FTS index |
| Library, Collection, Snapshot, and policy identity | candidate rankings |
| evidence packets and read receipts | graph projections |
| shared, content-addressed corpus read sets | candidate caches |
| append-only review decisions | Lens pages |
| append-only IdeaTrace candidates and closure receipts | rebuilt trace graph views |
| append-only answer projections and judgment receipts | sparse clickable Lens spans |

This boundary lets Dithyramba replace a search model or rebuild a visual view
without losing citations or human decisions.

## Five-minute local demo

### Requirements

- Linux or macOS 13 or newer;
- Python 3.11; the release gate pins Python `3.11.12` exactly;
- [`uv`](https://docs.astral.sh/uv/) for the reproducible environment.

Package metadata currently permits newer Python versions, but they are not
release-qualified until they enter the continuous-integration matrix.

The stable route uses SQLite FTS5 and does not require an API key, GPU, model
download, or network connection after installation.

```bash
git clone https://github.com/esannikov/dithyramba.git
cd dithyramba
uv sync --frozen
uv run dithyramba --version
uv run python scripts/demo.py
```

The demo creates a temporary Library from a small CC0 synthetic corpus, indexes
it, freezes a snapshot, runs one recall, inspects the resulting packet, and
replays it byte-for-byte. It prints the temporary data path and the exact IDs so
you can inspect the result yourself.

For a real corpus, follow [the complete local workflow](docs/HOW_TO_USE.md).
Keep live SQLite files, rebuildable caches, and backups outside the corpus and outside
Obsidian, Syncthing, Dropbox, or another file-sync root.

Book-length sources can be ingested with the explicit bounded
`--parser-profile large-document`; the conservative default remains unchanged.
The larger profile raises limits without disabling time, size, page, character,
or worker-memory guards.

The active parser profile is persisted as part of the immutable source
representation. Repeating the same bytes with the same profile reuses the
existing fragments without parser work; changing the profile creates a new
representation and keeps the previous one available for exact replay.

## Stable interfaces

| Surface | Role | Maturity |
|---|---|---|
| CLI | Library setup, ingest, snapshot, FTS recall, packet inspection and replay, review, backup and restore | default public path |
| `EvidencePacket/1.0` | persisted, source-closed retrieval result | versioned v1 contract |
| Loopback HTTP service | local programmatic access to one pinned Library | implemented; not remotely exposed |
| stdio MCP | bounded session, recall, answer preparation, gated draft, gap, and rejected-path tools over the Python facade | implemented; no acceptance or promotion tools |
| Lens | one CLI/API surface with library, session, atlas, concepts, and flow modes | implemented GET-only projections |
| Compact connectors | small source-closed packets for downstream agents | implemented library contracts |
| Candidate Ontology / Lens | scoped concepts and exact co-occurrence links over one bounded question neighbourhood | experimental, GET-only projection |
| Evidence-grounded reasoning | short public IdeaTrace steps closed over exact semantic receipts | experimental contracts, persistence, and CLI verifier |
| Rich answer projection | persisted proposition-level fact, synthesis, hypothesis, question, and framing governance; exact Lens span-to-source routes | experimental durable contract and deterministic gate |

## Optional research aids

The default route is deliberately small: local FTS finds candidates and
`EvidenceCoverageGate` checks whether the requested evidence roles are present.
If a literal query is insufficient, bounded query expansion or QueryCloud may
try a few explicit variants.

```text
raw FTS candidates
  → strict bibliography / index / table / topic-noise guards
  → EvidenceCoverageGate
  → conditional search inside up to three already found Sources
  → EvidenceCoverageGate again
  → inclusion-minimal exact fragments for a ready answer
  → answer-ready preparation or explicit blocked/gap state
```

Query variants, rankings, concept maps, IdeaTrace, and answer projections are
navigation or drafting aids. Every factual path must still close over exact
source fragments, and every promotion remains a human decision. The default
package contains no embedding model, reranker, vector store, or model download.
The detailed contracts are documented separately for integrators.

## Evidence at a glance

The public repository ships only rights-safe fixtures. Private research texts
are not distributed; their runs are summarized as ranges so the README describes
the system rather than individual projects.

| Universal measure | Observed range or result | What it establishes |
|---|---:|---|
| Persisted Library scale | 3–2,047 sources; 7–276,165 exact fragments | The same schema and workflow operate from a demo to a large mixed library. |
| Incremental ingest | 100% of unchanged inputs reused in 265-, 400-, and 816-source runs | Repeating an unchanged corpus created zero new source versions or fragments. |
| Two labeled retrieval suites | exact fragment @10: 58–82%; correct source @10: 87–96% | The right document is often found before the exact passage; passage drill-down remains necessary. |
| Deterministic replay | 16/16 sampled cold replays and 30/30 fresh-corpus replays | Stored packets and source addresses can be reproduced exactly. |
| Default model cost | 0 model calls; 0 model tokens | Ingest, FTS recall, gates, packets, and replay are local and deterministic. |
| Engineering gate | Full suite plus a strict `>=95.00%` combined-coverage threshold | The current implementation is broadly exercised; exact commit-bound values belong in the release acceptance record and do not validate research conclusions. |

The retrieval ranges combine different frozen internal test sets and are not a
cross-domain leaderboard; the scale maxima also come from different runs. An
exact passage proves what a source says, not that the source or a proposed
conclusion is true. Detailed protocols, negative
results, and limitations remain in the [evaluation record](docs/EVALUATION.md).

## Trust boundaries

- An exact quotation proves what a source says, not that the source is true.
- Several citations may still belong to one dependent source family.
- An incomplete corpus may omit the decisive document.
- Search scores, embeddings, graph edges, and UI proximity are not proof.
- Corpus text is treated as untrusted data, not as instructions.
- The server binds to loopback; read/write scope is pinned before startup.
- Input files are not rewritten or deleted during ingest.
- Dithyramba can represent uncertainty and provenance; it cannot replace source
  criticism or expert judgment.

## Project map

```text
src/dithyramba/        package and CLI
tests/                 unit, integration, smoke, API, and browser checks
migrations/            migration policy and pointer to the packaged SQL baseline
fixtures/              rights-safe synthetic regression corpora
verification/          deterministic public evidence-path replay
examples/              small CC0 corpus used by the local demo
schemas/               contract inventory policy, not duplicate schemas
scripts/               demo and clean-machine acceptance tools
skills/dithyramba/     thin Codex orchestration skill and evidence guidance
docs/                  concepts, workflow, reference, and development guide
.github/                continuous-integration and issue templates
```

Only `src/dithyramba/` is installed into the runtime wheel. The Codex skill is
an optional source-repository integration resource; the other
directories support learning, development, reproducibility, and release
verification. See the [repository guide](docs/REPOSITORY_GUIDE.md) for the
responsibility of each directory and package group.

## Documentation

- [Start with your own corpus](docs/HOW_TO_USE.md)
- [Codex skill orchestration](skills/dithyramba/SKILL.md)
- [Understand the architecture](docs/ARCHITECTURE.md)
- [How source-grounded memory works](docs/EXPLANATION.md)
- [CLI and contract reference](docs/REFERENCE.md)
- [Evidence coverage explained](docs/EVIDENCE_COVERAGE_GATE.md)
- [Evidence-grounded reasoning and IdeaTrace](docs/REASONING.md)
- [1.0.0 release notes](docs/RELEASE_NOTES_1.0.0.md)
- [Canonical terminology](docs/TERMINOLOGY.md)
- [Historical design and release records](docs/history/README.md)
- [v1 release boundary and post-v1 roadmap](docs/V1_ROADMAP.md)
- [Development and verification](docs/DEVELOPMENT.md)
- [Repository and module guide](docs/REPOSITORY_GUIDE.md)
- [Evaluation evidence and limitations](docs/EVALUATION.md)
- [Clean-install acceptance](docs/ACCEPTANCE.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Verification

The repository treats engineering checks and research validity as different
claims. `uv run check` runs format, lint, strict typing, the full test suite with
branch measurement, and dependency audit. A clean release build is then tested
from both wheel and source distribution.

```bash
uv sync --frozen --dev
uv run check
uv run python -m verification.generate_synthetic_1000 --check
uv run python -m verification.run_public_replay --warmups 1 --measured 2 \
  --output /tmp/dithyramba-public-replay.json
```

Every release commit must pass strict typing, terminology checks, the full
test suite, dependency audit, and the `>=95.00%` combined-coverage gate.
Exact counts and percentages are commit- and environment-bound release
evidence rather than evergreen documentation.

Run clean-install acceptance from a fresh checkout before building or syncing
inside that checkout:

```bash
./scripts/acceptance.sh --quick
```

The acceptance route generates its distributions and state under an isolated
temporary root. It verifies the schema, executable CLI/MCP contract receipt,
documentation links, Lens assets, deterministic replay, and both source and
wheel builds without making the release tree dirty.

Passing these checks shows that the implementation behaves as specified by its
tests. It does not establish historical or scientific truth, nor superiority
over other research systems. Commit-bound verification results will be recorded
with each published release after the clean-install run.

## License

Copyright 2026 Eugene Sannikov.

Dithyramba is licensed under the
[Apache License, Version 2.0](LICENSE). It permits commercial and private use,
modification, and redistribution under the license terms, and includes an
explicit patent grant from contributors. The license does not grant rights to
project names or trademarks.
