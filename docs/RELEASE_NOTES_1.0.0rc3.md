# Dithyramba 1.0.0rc3

`1.0.0rc3` closes the final contract-audit findings without adding a retrieval
subsystem, database, model dependency, or architectural layer.

## What changed

- Missing requirements can drill down on explicit two-letter anchors such as
  `AI` and `ШІ`. Those literal terms no longer inherit the topic-drift filter.
- New `EvidenceGateSpec` values default to `exact_fragment_unicode_v3`, which
  preserves Latin and Cyrillic diacritics and treats combining marks as part of
  a word. Explicit `exact_fragment_unicode_v1` and `/v2` values retain their
  historical meaning and hashes.
- `AgentAnswerPreparation/1.1` binds the current deterministic preparation
  profile. `/1.0` remains readable and replays with its original drilldown
  behavior.
- MCP `tools/list` declares executable `outputSchema` values. The release gate
  generates one commit-bound receipt from the CLI tree, MCP input/output
  contracts, SQLite schema fingerprint, and checked local documentation links.
- Contributor and acceptance instructions now separate ordinary development
  state from the clean release checkout.
- The development test stack follows Starlette's supported `httpx2` client and
  no longer emits the legacy-client warning.

## Trust boundary

Strict literal matching reduces false evidence admission; it does not establish
source truth, entailment, independence, or human acceptance. Candidate rank
continues to govern discovery order only. The supported corpus scope is Latin-
and Cyrillic-script research; other scripts are outside this release's product
evaluation scope.

## Upgrade

No database migration or corpus rebuild is required. Existing saved gate specs
continue to carry their explicit matching profile. Re-run `prepare_answer` to
receive the new `/1.1` preparation and use the full returned object unchanged
when recording a draft.
