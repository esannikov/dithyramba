# Repository guide

This guide explains the stable responsibilities in the Dithyramba repository.
It deliberately does not catalog every file. On GitHub, the useful convention
is to document public entry points, directory boundaries, and module families;
individual implementation files are documented by their module docstrings,
typed contracts, and tests.

## What a user installs

The runtime wheel contains `src/dithyramba/` plus package metadata and the
Apache-2.0 license. Tests, fixtures, documentation, verification scripts,
examples, and GitHub automation are repository resources. They help people
understand and verify the project, but they are not imported into a normal
Dithyramba runtime.

The source distribution is broader because it must let a maintainer inspect,
rebuild, and verify the same wheel from source.

## Top-level map

| Path | Responsibility | Installed in wheel? |
|---|---|---:|
| `src/dithyramba/` | Runtime library, CLI, local HTTP service, and packaged SQL/templates | yes |
| `tests/` | Causal unit, integration, API, browser, smoke, and acceptance tests | no |
| `fixtures/` | Small repository-authored CC0 inputs with frozen invariants | no |
| `verification/` | Deterministic 1,000-fragment public replay of the evidence path | no |
| `examples/` | Three CC0 notes used by the five-minute demo | no |
| `migrations/` | Human-inspectable mirror of the packaged immutable SQL migrations | no |
| `schemas/` | Policy for versioned schemas and a pointer to executable contracts | no |
| `scripts/` | Demo, release audit, distribution inspection, and clean-install gate | no |
| `docs/` | User workflow, product/design rationale, contracts, architecture, evaluation, and development | no |
| `.github/` | Continuous integration plus issue and pull-request templates | no |
| `pyproject.toml` | Package metadata, dependencies, build, lint, typing, and test configuration | no |
| `uv.lock` | Frozen transitive dependency graph used by development and release gates | no |
| `LICENSE`, `CITATION.cff` | Legal and citation metadata | license is copied into the wheel |

## Runtime package map

The Python package is grouped by responsibility rather than by a single
all-knowing service.

| Package family | Responsibility |
|---|---|
| `library`, `collections`, `access`, `snapshots` | Physical boundary, logical scope, default-deny authorization, and frozen corpus identity |
| `ingest`, `provenance`, `contracts` | Read-only source intake, exact addresses, hashes, receipts, and canonical serialization |
| `store`, `persistence`, `backup` | SQLite schema, repositories, immutable records, backup, restore, and migration |
| `recall`, `evidence`, `answers` | Candidate discovery, replayable packets, explicit evidence coverage, and answer validation |
| `reasoning`, `answers` | Public IdeaTrace and proposition-level answer contracts, deterministic closure/coverage, and no automatic promotion |
| `sessions` | 0.2 contracts for scoped research sessions, typed hash-chained events, and deterministic compact state; durable writes live in `persistence.sessions` |
| `ontology` | Experimental scoped candidate concepts, exact evidence closure, and co-occurrence links |
| `meaning`, `relations`, `structure`, `research` | Typed interpretive objects and source-bound research structures |
| `review` | Append-only human acceptance, revision, rejection, and deferral |
| `connectors` | Compact source-closed packets for downstream tools and agents |
| `reading_room`, `atlas`, `api` | Human and local programmatic views over one pinned Library and scope |
| `tooling` | Contributor checks and terminology validation; not a research runtime service |

Several internal names retain early contract labels such as `VS0` or phase
prefixes such as `P2` and `P7`. They are compatibility and history labels, not
separate products, hardware requirements, or recommended user workflows.

## Tests and verification

The test suite mirrors observable boundaries:

- `tests/unit/` checks one contract or failure rule in isolation;
- `tests/ontology/` checks candidate-ontology construction, tamper resistance, and deterministic replay;
- `tests/integration/` checks persistence and multi-module closure;
- `tests/api/` and `tests/browser/` check loopback and human-facing surfaces;
- `tests/smoke/` checks small complete routes;
- `tests/acceptance/` checks public fixtures and release-facing invariants.

`verification/` serves a different purpose. It is a small runnable example of
the complete public evidence path over 1,000 generated fragments. It records
real ingest, scope, recall, persistence, and replay behavior without claiming
natural-language quality or research truth.

## Where to begin

- **First-time user:** [How to use Dithyramba](HOW_TO_USE.md)
- **Agent or systems specialist:** [Architecture](ARCHITECTURE.md) and
  [reference](REFERENCE.md)
- **Reasoning integrator:** [Evidence-grounded reasoning](REASONING.md)
- **Interactive-memory integrator:** [0.2 specification](INTERACTIVE_RESEARCH_MEMORY_SPEC.md)
  and [roadmap](ROADMAP_0.2.md)
- **Research-method reader:** [Explanation](EXPLANATION.md) and
  [evaluation limits](EVALUATION.md)
- **Contributor:** [Development](DEVELOPMENT.md),
  [contributing rules](../CONTRIBUTING.md), and
  [clean-install acceptance](ACCEPTANCE.md)

## Specialized documents

These pages are narrower than the main user path, but they make advanced
contracts and design decisions inspectable:

| Document | Question it answers |
|---|---|
| [Product context](PRODUCT.md) | Who is the system for, and what must its surfaces never imply? |
| [Design system](DESIGN.md) | How should Reading Room and Lens communicate evidence and uncertainty? |
| [EvidenceCoverageGate](EVIDENCE_COVERAGE_GATE.md) | When do retrieved passages cover the declared evidence roles? |
| [AnswerCoverageGate](ANSWER_COVERAGE_GATE.md) | Did a proposed answer use every required accepted facet? |
| [Compact research memory](COMPACT_RESEARCH_MEMORY.md) | How are small source-closed packets routed and validated? |
| [Research View](RESEARCH_PROJECTION.md) | How can wider candidate material remain visible without becoming accepted evidence? |
| [Mars IdeaTrace-24](MARS_IDEATRACE_24.md) | What did the first frozen multi-source reasoning diagnostic test, improve, and leave unresolved? |
| [Interactive research memory 0.2](INTERACTIVE_RESEARCH_MEMORY_SPEC.md) | How can an agent continue research without turning chat into evidence or duplicating the truth layer? |
| [Roadmap 0.2](ROADMAP_0.2.md) | In what dependency order will sessions, persistence, agent tools, review, and evaluation be built? |

## Maintenance rule

Update this guide when a top-level directory is added, removed, or changes its
stable responsibility, or when a new public package family appears. Do not add
every leaf file. A leaf file should have one clear responsibility, a module
docstring when useful, typed interfaces, and tests that reveal why it exists.
