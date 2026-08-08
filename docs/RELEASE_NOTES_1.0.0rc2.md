# Dithyramba 1.0.0rc2

`1.0.0rc2` is a focused correctness update to the v1 release candidate. It
does not add another retrieval subsystem or database. It closes two audit
findings at existing trust boundaries: book projection completeness and exact
evidence admission.

## What changed

- EPUB and FB2 projection preserves common structural blocks, including
  tables, definition lists, notes, captions, and verse. If an unknown
  text-bearing block cannot be projected, coverage becomes `partial` and the
  omission is named in the processing receipt.
- Internal Markdown unit comments no longer appear in evidence text. Unit type,
  source coordinate, and character span remain available in the sidecar.
- One shared `nfkd_latin_marks_casefold_v2` lexical fold now serves candidate
  hygiene and evidence anchors. It is deterministic, case-insensitive, strips
  Latin diacritics, preserves meaning-bearing marks in Ukrainian and other
  non-Latin scripts, and remains word-bounded.
- `exact_fragment_unicode_v2` is the default gate profile. A negative-only
  requirement is invalid under v2, and candidate sets with contradictory
  SourceFamily or independence lineage fail closed before coverage is counted.
  Explicit `exact_fragment_unicode_v1` replays keep their previous substring,
  negative-only, and caller-supplied lineage semantics.
- The interactive source-local drilldown runs only when an answerable requirement is
  genuinely incomplete. Corpus-gap review states are preserved as such.

## Trust boundary

The lineage check proves internal consistency, not the external truth of labels
provided to the pure evaluator. The standard interactive route obtains lineage
from the verified Library repository. Direct evaluator integrations must
provide an equivalent trusted mapping.

The strict replay immediately before `record_draft`, content-addressed source
checks, access policies, and immutable historical packets are unchanged.

## Upgrade

No schema migration is required. Install the updated package and re-index EPUB
or FB2 sources when the richer `books/1.2` projection is desired. Existing
Libraries and prior packet identities remain readable.

For v1 compatibility, `EvidenceCandidate.source_family` keeps its public field
name while containing the canonical `source_family_id` used elsewhere. Likewise,
`CandidateQualityAssessment.admitted` means that a fragment is quality-eligible
to be offered to the gate; it is not evidence acceptance or human review.
