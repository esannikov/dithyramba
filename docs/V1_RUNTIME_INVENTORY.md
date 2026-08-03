# Dithyramba v1 runtime inventory

## Purpose

This is the deletion boundary for v1. It records what the verified package
contains before removal, what the default route actually needs, and which
historical subsystems must disappear without weakening provenance.

Baseline: Git `7ce7f7e`; working branch `codex/v1-clean-core`.

## Verified size before cleanup

- 153 runtime Python files;
- 65,226 physical Python lines including package initializers and generated
  public surfaces;
- 123 Python test files;
- 13 ordered SQL migrations;
- eight default runtime dependencies;
- one optional semantic dependency group and one optional ontology group.

Ignored local state (`.venv`, test/type/lint caches, coverage artifacts, and
`__pycache__`) is not part of Git, sdist, or wheel. It may be deleted at release
closure, but it is not evidence of package bloat.

## Stable v1 route

| Boundary | Keep | Why |
|---|---:|---|
| Library, Collection, access policy | yes | exact scope and default deny |
| Source, SourceVersion, SourceFragment, Blob | yes | immutable evidence identity |
| Parser and incremental ingest | yes | build and update local memory |
| Snapshot and permitted FTS scope | yes | reproducible candidate discovery |
| lexical expansion and conditional QueryCloud | yes | measured deterministic repair |
| candidate hygiene and EvidenceCoverageGate | yes | keep noise and unsupported answers out |
| compact AgentEvidencePacket | yes | solve context-window pressure |
| research session, stdio MCP, Session Lens | yes | agent and human access |
| append-only review | yes | human promotion boundary |
| backup and restore | yes | local durability |

## Retired runtime candidate

The following semantic/hybrid/model family is not used by the best tested
default route:

- `recall/hybrid_*`;
- `recall/semantic_*`;
- `recall/vector.py`;
- `recall/rerank.py` and `recall/reranker_provider.py`;
- `recall/provisioning.py` and the `model provision` CLI;
- `persistence/hybrid.py`, `persistence/hybrid_recall.py`, and
  `persistence/semantic.py`;
- the `sentence-transformers` optional dependency group;
- SQL tables created only for retired semantic/hybrid execution.

This family currently accounts for 9,847 runtime lines and 10,079 directly
paired test lines. Seventeen source/test modules import its public symbols.
Those imports, package exports, CLI commands, migration compatibility, and
documentation must be removed as one audited slice. Deleting files before that
closure would create a package that looks smaller but is not safely installable.

## Optional or consolidating surfaces

- Candidate Ontology remains optional until independent human coherence review.
- Research Atlas, Reading Room, Concept Lens, and Flow View overlap. V1 keeps
  one Session Lens product surface plus exact source inspection; useful
  projections should become Lens modes rather than separate products.
- Answer projection and bounded reasoning stay only where they expose factual,
  synthesis, hypothesis, question, and human-decision states without hidden
  promotion.

## Removal protocol

For every retired family:

1. prove no default CLI, MCP, HTTP, or Python route imports it;
2. remove public exports and optional dependencies;
3. remove paired runtime and tests in the same commit;
4. keep historical evaluation findings as documentation, not executable user
   workflow;
5. run clean sdist/wheel installation and the full release gate;
6. test legacy Library behavior before touching old schema tables;
7. never mutate a user's existing Library in place for cleanup.

## Next measured slice

Profile the public interactive route against the core route on one frozen
scope. The target is process-local scope construction once per session and
warm public recall no more than twice core recall. Only after that result is
green does runtime deletion begin.
