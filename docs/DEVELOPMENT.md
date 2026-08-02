# Development

Dithyramba package metadata is `0.1.0rc1`; the current branch is the pre-alpha
`0.2` development candidate with schema v13 interactive sessions. This guide
covers work from a repository checkout; it does not define a public
compatibility promise.

## Supported development target

- Linux or macOS 13 or newer;
- Python 3.11 or newer;
- [`uv`](https://docs.astral.sh/uv/);
- SQLite with FTS5.

The POSIX acceptance script records the operating system and architecture as
receipt metadata. Windows is not yet release-qualified because current locking
and descriptor-safety contracts depend on POSIX behavior.

## Bootstrap

```bash
uv sync --frozen --dev
uv run dithyramba --version
uv run dithyramba about
uv run python -m dithyramba.store.migrations verify
uv run python scripts/demo.py
```

The development group intentionally includes the optional Candidate Ontology
runtime because the canonical test aggregate exercises that surface. The base
package remains lightweight; ordinary users install the ontology dependencies
only with `dithyramba[ontology]`.

The demo creates a temporary Library, verifies its local runtime, ingests the
public example corpus, freezes a snapshot, recalls one packet, and replays it.
Use `dithyramba library doctor --library <id> --data-home <path>` for an existing
Library; `doctor` is not a pre-initialization environment probe.

## The development loop

1. Freeze one defect, expected behavior, and failure semantics.
2. Add the smallest contract and causal tests that expose it.
3. Run focused tests while editing.
4. Run format, lint, and strict typing on changed files.
5. Run the full aggregate and inspect raw artifacts.
6. Update package documentation and changelog only after the surface is real.

Do not add a module without its first executable contract and test. Do not add
a model or service without a measured defect that requires it.

## Focused retrieval and evidence checks

```bash
uv run pytest --no-cov \
  tests/unit/test_fts_pool.py \
  tests/unit/test_recall_compatibility.py \
  tests/unit/test_evidence_coverage_gate.py -q
uv run ruff format --check \
  src/dithyramba/recall/fts.py \
  src/dithyramba/recall/compatibility.py \
  src/dithyramba/recall/__init__.py \
  tests/unit/test_recall_compatibility.py \
  tests/unit/test_evidence_coverage_gate.py
uv run ruff check \
  src/dithyramba/recall/fts.py \
  src/dithyramba/recall/compatibility.py \
  src/dithyramba/recall/__init__.py \
  tests/unit/test_recall_compatibility.py \
  tests/unit/test_evidence_coverage_gate.py
uv run mypy \
  src/dithyramba/recall/fts.py \
  src/dithyramba/recall/compatibility.py \
  src/dithyramba/recall/__init__.py \
  tests/unit/test_recall_compatibility.py \
  tests/unit/test_evidence_coverage_gate.py
```

Evidence-gate input must already be authorized and contain explicit proof
metadata. Do not infer proof eligibility from a search score or source
extension.

## Development aggregate

```bash
uv run check
```

The aggregate runs formatting, lint, strict typing, full tests with branch
measurement, and dependency audit. Pytest enforces exact combined coverage of
at least `95.00%`. Coverage output uses two decimal places, but the raw JSON is
authoritative:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

totals = json.loads(Path("artifacts/coverage.json").read_text())["totals"]
print(f"combined:   {float(totals['percent_covered']):.4f}%")
print(f"statements: {float(totals['percent_statements_covered']):.4f}%")
print(f"branches:   {float(totals['percent_branches_covered']):.4f}%")
PY
```

Passing this aggregate verifies the implementation against its test suite. It
does not close the release or research-validity gates below.

## Clean-install acceptance

For host preparation, expected output, retained logs, and failure diagnosis,
follow the [clean-install acceptance protocol](ACCEPTANCE.md).

Use quick mode while preparing a candidate:

```bash
./scripts/acceptance.sh --quick
```

Quick mode runs the release audit, frozen dependency sync, public demo, smoke
and migration tests, format/lint checks, distribution build, package-content
inspection, and isolated wheel/sdist imports. It is a rehearsal, not release
evidence.

The full gate must run without `--quick` in a clean Linux or macOS checkout:

```bash
./scripts/acceptance.sh
```

Full mode replaces the focused tests with `uv run check` and then verifies both
built distributions. Record the commit and complete output with any release
candidate; a successful run on a dirty or different checkout is not equivalent.

### Coverage reporting

The development and clean-install routes enforce exact combined coverage
`>=95.00%`. Statement and branch coverage remain visible as separate diagnostic
values. Do not use the combined number to hide an untested decision path.

## Persistence and migration rules

1. Keep SQL in repository/persistence modules and parameterize values.
2. Never modify an applied migration; add the next contiguous checksummed SQL
   migration to both packaged mirrors.
3. Preserve byte compatibility of existing v1 artifacts.
4. Use insert-or-verify for content-addressed immutable records.
5. Reconstruct and rehash persisted objects before returning them.
6. Test fresh schema, previous-schema migration, fault rollback, row tampering,
   read-only replay, and cold reopen.

`LibraryRepository` is the production persistence boundary. `Store.open` is a
low-level migration/test/backup seam and must not become a corpus-facing API.

## Runtime and security rules

- Treat corpus text and metadata as untrusted data, never instructions.
- Runtime databases, WAL, blobs, indexes, and model caches remain outside the
  repository and all synchronization roots.
- Parser subprocesses use bounded input/output, timeout, and resource limits.
- Do not expose absolute corpus paths or full source text in ordinary metadata.
- Policy and snapshot scope are compiled before text selection.
- Library bootstrap stays CLI-only; loopback API pins one existing Library.
- Backup/restore must verify all paths, modes, hashes, migrations, rows, event
  stream, and blob closure.

## Contract evolution

- Stable v1 schema meaning changes require a new version.
- `EvidencePacket/1.0` remains the persisted FTS packet.
- An expanded discovery route must not change the meaning of
  `EvidencePacket/1.0` or `ExpandedEvidencePacket/2.0`.
- Default CLI/HTTP behavior cannot change until a new route has durable
  manifests, atomic persistence, and cold exact replay.

## Documentation ownership

- `README.md`: product entry and honest surface table;
- `docs/PRODUCT.md`: product audience, surface, and trust boundaries;
- `docs/DESIGN.md`: human-interface language and interaction constraints;
- `docs/ARCHITECTURE.md`: durable layers and invariants;
- `docs/HOW_TO_USE.md`: user workflow;
- `docs/REFERENCE.md`: commands and contracts;
- `docs/EXPLANATION.md`: architecture in plain language;
- `docs/EVIDENCE_COVERAGE_GATE.md`: deterministic evidence sufficiency;
- `docs/ANSWER_COVERAGE_GATE.md`: answer-facet validation;
- `docs/COMPACT_RESEARCH_MEMORY.md`: compact connector and answer-validation contracts;
- `docs/RESEARCH_PROJECTION.md`: wider-memory projection outside accepted evidence;
- `docs/INTERACTIVE_RESEARCH_MEMORY_SPEC.md`: implemented 0.2 session and
  persistence contracts, role boundaries, and acceptance criteria;
- `docs/ROADMAP_0.2.md`: dependency-ordered implementation and release gates for
  interactive research memory;
- `docs/EVALUATION.md`: corpus scales, results, negative findings, and limits;
- `docs/ACCEPTANCE.md`: clean-install release verification;
- `docs/DEVELOPMENT.md`: contribution and verification rules;
- `docs/REPOSITORY_GUIDE.md`: stable directory and module responsibilities;
- `CHANGELOG.md`: package changes, not experiment notebooks.

## Release boundary

This repository is a pre-alpha source preview. Clean-install acceptance
establishes that the checkout, demo, tests, build, wheel, and sdist agree in one
recorded environment. It does not establish semantic accuracy, source truth,
human usefulness, production security, or compatibility beyond the declared
preview contracts.
