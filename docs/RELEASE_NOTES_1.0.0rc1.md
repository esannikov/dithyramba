# Dithyramba 1.0.0rc1

`1.0.0rc1` is the first package candidate for the v1 architecture. It is ready
for clean-machine installation and real-corpus testing, but it is not the final
stable `1.0.0` release.

## What this candidate establishes

- One local Python/SQLite research-memory core with a single clean v1 schema.
- Immutable source versions, exact fragments and addresses, frozen snapshots,
  default-deny access policies, replayable evidence packets, and append-only
  review history.
- A deterministic FTS5-first retrieval route with bounded query repair,
  candidate hygiene, optional source-local drill-down, and
  `EvidenceCoverageGate` before answer preparation.
- Continuing research sessions through the Python facade and local stdio MCP,
  without giving agents a human-acceptance or promotion tool.
- Read-only Lens views for Libraries, sessions, atlases, concepts, and process
  flow, with exact span-to-source inspection where a projection provides trace
  bindings.
- A thin Codex skill that orchestrates the public CLI, MCP, and Lens surfaces
  without duplicating storage, retrieval, or review logic.

## What was removed from the current architecture

Harrier, embedding runtimes, vector stores, model provisioning, model caches,
and case-specific evaluation harnesses are not part of the v1 package or
default route. Historical comparisons remain in the evaluation record only.

## Current engineering evidence

The canonical local gate collects 1,914 tests: 1,912 pass and two
host-dependent checks skip in the recorded environment. Combined branch-aware
coverage is 95.15% against a strict 95.00% threshold. Format, lint, terminology,
strict typing, dependency audit, deterministic replay, distribution inspection,
and isolated wheel/source installation are separate release gates.

These checks establish implementation consistency against the repository's
contracts. They do not establish that a source is true, a corpus is complete,
or a generated research conclusion is valid.

## Before stable 1.0.0

The remaining gate is to repeat the documented install and demo from the
published repository on a machine with no Dithyramba project state, record the
commit-bound result, freeze the public surfaces, and then tag stable `1.0.0`.
See the [v1 roadmap](V1_ROADMAP.md) and
[acceptance protocol](ACCEPTANCE.md).

For upgrade and compatibility details, read the
[reference](REFERENCE.md), [changelog](../CHANGELOG.md), and
[historical records](history/README.md).
