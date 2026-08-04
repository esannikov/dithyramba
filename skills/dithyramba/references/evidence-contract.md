# Evidence and truth contract

Keep these states separate in every answer, report, and interface.

| State | Meaning | Allowed wording |
|---|---|---|
| Retrieved candidate | Search found a potentially relevant passage | “The search found…” |
| Eligible evidence | A passage passed deterministic hygiene and declared coverage checks | “This passage supports…” |
| Draft claim | An agent formulated a claim from eligible evidence | “Working conclusion…” |
| Hypothesis | A testable interpretation or connection | “Hypothesis…” |
| EvidenceGap | A required role or exact passage is missing | “The corpus does not yet establish…” |
| Human-reviewed decision | A person accepted, rejected, or revised a scoped claim | “Accepted in review…” |

Ranking measures discovery priority. It does not establish truth, source
quality, independence, or sufficient evidence.

## Before a source-backed answer

1. State the question and intended use.
2. Declare required evidence roles: for example primary account, independent
   corroboration, counterevidence, date, mechanism, or methodological context.
3. Inspect exact passages, official titles, authors, and source coordinates.
4. Remove bibliography, index, table, navigation, and topic-drift fragments.
5. Run `prepare_answer`; accept its gate decision as a boundary on certainty.
6. Bind each factual proposition to one or more exact source fragments.
7. Mark synthesis that combines sources and speculation that goes beyond them.

## Source presentation

Show human-readable metadata first:

- author or asserting institution;
- official title;
- year when available;
- page, section, line, timestamp, or other stable coordinate;
- exact supporting excerpt;
- role in the answer: supports, qualifies, contradicts, or provides context.

Keep internal fragment IDs and hashes available for replay, but do not make
them the primary label shown to a researcher.

## What the gate proves

`EvidenceCoverageGate` proves only that the selected clean passages satisfy the
explicit structural requirements supplied to it. It does not prove that a
source is true, independent, complete, unbiased, or accepted by a human.

Each requirement needs at least one positive selector. Forbidden text alone is
not evidence. Literal anchors are case- and diacritic-insensitive but remain
word-bounded: the gate does not silently add synonyms or semantic equivalence.

The gate also rejects contradictory Source/SourceFamily/independence mappings
inside one candidate set. This prevents accidental double-counting, but it does
not authenticate labels invented by an external caller. The normal
`prepare_answer` route derives lineage from the verified Library repository;
direct users of the pure evaluator must establish equivalent trust themselves.

## Synthesis discipline

Let the answer breathe by separating three layers:

1. **Supported spine** — propositions directly grounded in exact passages.
2. **Interpretive synthesis** — a transparent connection among supported
   propositions, with all contributing sources visible.
3. **Open hypothesis** — a useful idea that requires further evidence and must
   not be phrased as established fact.

This separation preserves creativity without weakening provenance.
