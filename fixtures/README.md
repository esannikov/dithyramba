# Rights-safe fixtures

Every fixture in this directory is repository-authored synthetic data released
under `CC0-1.0`. The files contain no copied corpus passages, private material,
personal data, or third-party prose. Names such as `caillebotte_regression` are
regression labels only; that fixture makes no historical or semantic claim.

JSON artifacts use the repository canonical profile: UTF-8, NFC, sorted object
keys, compact separators, no floats, and exactly one terminal LF. Artifact
hashes are SHA-256 over canonical JSON bytes without the terminal LF. Stable
fixture IDs are deliberately non-random and must not be treated as production
entity IDs.

Fixture sets:

- `public_multilingual/`: twelve original fragments and twelve preregistered
  lexical queries (six Ukrainian, six English) with expected source references.
- `caillebotte_regression/`: opaque assertion-accounting data demonstrating
  that an exact duplicate and an AI ledger entry add no independent source
  family.
- `synthetic_isolation/`: public, private, holdout, and excluded canaries across
  two logical Libraries.

Validate canonical bytes, frozen hashes, counts, isolation expectations, source
family accounting, and live FTS recall gates with:

```bash
uv run python fixtures/validate.py
```

The validator reads fixture text only after it has loaded the explicit fixture
scope. It performs no network access and writes no files.
