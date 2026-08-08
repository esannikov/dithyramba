# Contributing

Dithyramba `1.0.0rc2` is a release candidate. Small, evidence-backed changes are
easier to review than broad framework additions.

## Before changing code

1. State one observed defect or missing contract.
2. Define the expected result and fail-closed behavior.
3. Keep source identity, retrieval relevance, evidence sufficiency, and human
   acceptance as separate decisions.
4. Avoid adding a model or service until a measured defect requires it.

## Local setup

```bash
uv sync --frozen --dev
uv run dithyramba --version
uv run python scripts/demo.py
```

## Required checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src tests verification
uv run pytest
python scripts/release_audit.py
```

The canonical aggregate is:

```bash
uv run check
```

Tests must use synthetic or clearly licensed fixtures. Do not commit personal
corpora, source excerpts without redistribution rights, absolute user paths,
runtime databases, model weights, credentials, generated benchmark artifacts,
or local agent state.

For the role of each top-level directory and package family, read the
[repository guide](docs/REPOSITORY_GUIDE.md). Update that guide when a stable
responsibility changes; ordinary leaf files should explain themselves through
their module docstring, type contract, and tests.

## Contract rules

- Never modify an applied SQL migration; add the next contiguous migration to
  both migration mirrors.
- Existing versioned artifacts keep their byte meaning. A meaning change needs a
  new schema version.
- Search scores and generated queries are discovery metadata, not evidence.
- Read-only views perform no hidden indexing, review, or mutation.
- New public behavior needs a causal test and plain-language documentation.

## Pull requests

Describe the defect, the changed contract, new tests, compatibility impact, and
any open research-validity limitation. Passing engineering checks is not a claim
that a corpus or conclusion is factually correct.

## Contribution license

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in Dithyramba is provided under the Apache License,
Version 2.0, without additional terms or conditions, as described in section 5
of the [project license](LICENSE).
