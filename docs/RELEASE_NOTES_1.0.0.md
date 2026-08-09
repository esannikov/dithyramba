# Dithyramba 1.0.0

`1.0.0` is the first stable release of Dithyramba's local source-to-evidence
memory. It promotes the audited `1.0.0rc3` runtime without adding a retrieval
subsystem, database, model dependency, migration, or architectural layer.

## Stable v1 surface

- One local route from immutable source versions and exact fragments through a
  frozen `CorpusSnapshot`, default-deny `AccessPolicy`, SQLite FTS5 recall,
  exact `EvidencePacket`, and append-only human review.
- A versioned CLI, seven bounded stdio MCP tools with executable input/output
  schemas, and GET-only Lens views.
- Deterministic packet replay, portable backup/restore, strict Library
  isolation, exact source-family accounting, and commit-bound contract
  receipts.
- Optional ontology, reasoning, and answer-projection paths remain explicitly
  candidate or review-only. They cannot promote generated material into
  accepted evidence.

## Stable-release gate

The stable commit was published only after:

- ten independent Latin/Cyrillic simulation scenarios covering EN/UK recall,
  strict letter and diacritic handling, short `AI`/`ШІ` anchors, duplicate and
  SourceFamily accounting, default-deny isolation, explicit gaps, safe FTS
  input, MCP lifecycle, and exact HTTP replay;
- the full clean-install acceptance route from source, sdist, and wheel on the
  pinned Python and `uv` versions;
- strict formatting, linting, typing, terminology, dependency, migration,
  browser, PDF, schema, documentation-link, and `>=95.00%` combined-coverage
  checks;
- green Linux and macOS GitHub CI on the exact release change.

Exact counts, commit identity, receipt hash, host metadata, and CI links belong
to the GitHub Release so they remain bound to the published tag.

## Language scope

The release-qualified corpus scope is Latin and Cyrillic. Ukrainian and English
have public end-to-end fixtures; other Latin- and Cyrillic-script languages are
best-effort until a dated corpus evaluation exists. Other scripts are outside
the current product-evaluation claim.

## Upgrade from 1.0.0rc3

No database migration or corpus rebuild is required. Existing Libraries,
snapshots, packets, and explicit `exact_fragment_unicode_v1`, `/v2`, and `/v3`
gate profiles remain readable and replayable. Reinstall the package and confirm
the target Library with `library describe` and `library doctor`.

## Trust boundary

Stable means that the documented software contracts passed the release gate.
It does not establish source truth, scholarly validity, universal retrieval
quality, human usefulness, or acceptance of any generated conclusion.
