# Evaluation evidence and limits

This page records what Dithyramba has actually been exercised on as of the
`0.1.0rc0` source preview. It separates three different claims:

1. **Public reproducibility:** checks anyone can run from this repository.
2. **Internal development evidence:** corpus-specific tests whose sources or
   gold labels cannot be redistributed here.
3. **Independent validity:** external, blinded, human evaluation. This has not
   yet been established.

Counts use the unit native to each corpus. A source, file, structured record,
fragment, retrieval unit, memory question, and evidence item are different
objects and must not be added together as one global corpus size.

## How to read the metrics

- **Exact fragment @k** asks whether the needed passage is present in the first
  `k` candidates. It measures recovery, not truth.
- **Correct source @k** is weaker: the right document is present, but the exact
  supporting passage may still be missing.
- **Full evidence coverage** requires every preregistered evidence role or
  facet, not merely one relevant quotation.
- **Safe abstention** checks whether deliberate gaps, excluded sources, and
  unrelated questions remain unsupported.
- **Statement support** compares the wording of an answer with the displayed
  passages. A genuine quotation does not justify a broader sentence by itself.
- **Context reduction** is reported only beside retained evidence coverage. A
  smaller prompt is not an improvement when it drops required anchors.

## Publicly reproducible checks

All committed fixture text is repository-authored and released as `CC0-1.0`.

| Fixture | Scale | Current staging result | What it proves |
|---|---:|---:|---|
| `public_multilingual` | 12 fragments; 12 preregistered queries: 6 Ukrainian and 6 English | recall@10 `12/12`; macro, UA, and EN recall `1.000` | Exact multilingual lexical retrieval over a tiny frozen fixture. |
| `synthetic_1000` | 1,000 generated fragments; 20 deterministic lexical probes | expected fragment selected; packet hashes `1/1`; provider calls `0` | Ingest, scope compilation, FTS recall, receipts, persistence, and deterministic packet replay. |
| Isolation fixtures | 2 Libraries; 4 public/private/holdout/excluded canaries | all fixture invariants passed | Scope is compiled before reading and excluded data does not silently enter recall. |

The 1,000-fragment run is an engineering workload, not a semantic research
benchmark. Its text and queries are deliberately easy and synthetic.

## Internal corpus screens

### Mars working corpus

| Field | Value |
|---|---:|
| Retrieval units | 53,747 |
| Normalized text | approximately 103.8 million characters |
| Evaluation cases | 72 bilingual cases: 51 positive, 21 controls |
| Candidate union in the Harrier ablation | 7,273 units |
| Exact fragment @1 / @5 / @10 | 18/51 · 38/51 · 43/51 |
| Correct source @10 | 48/51 |
| Exact MRR | 0.510 |
| Non-body items after structural admission | 0 |
| Network, generative calls, generative tokens | 0 · 0 · 0 |

This was the largest retrieval-scale calibration. It showed that exact FTS
candidates followed by bounded Harrier reranking were stronger and simpler
than placing another semantic candidate generator before Harrier. It did not
test complete generated answers. Candidate availability was known during
development, so these numbers are calibration evidence rather than a held-out
benchmark.

The reports record units and characters, but not a trustworthy aggregate page
or original-document count. No page estimate is inferred here.

### Parisian Ten structured records

| Field | Value |
|---|---:|
| Raw inputs | 40 files; 7 source families; 10 artists |
| Structured records | 28,006 |
| Prepared memory layer | 28 reviewed question units; 36 exact evidence fragments |
| Evaluation | 28 cases; 56 Ukrainian/English formulations; 10 unrelated controls |
| Full evidence in memory packet | 56/56 |
| Full raw evidence at matched packet budget | FTS 43/56 · hybrid 42/56 · E5 37/56 |
| Unrelated-query abstention | 10/10 |
| Cross-source cases | memory 10/10; raw FTS 1/10 at top 4; E5 and hybrid 0/10 |

The useful effect here was not generic similarity. The memory retained a
reviewed join between several records, entity-disambiguation decisions, and
explicit gaps. Raw search ranked 28,006 records; the memory route ranked 28
prepared question units and then expanded their accepted evidence. These are
different search spaces. The result measures reuse of prior research work,
not superiority of one embedding model.

### Van Gogh source-grounded research

Two related tests exposed both the benefit and the remaining risk.

#### Equal-source compact-packet A/B

- both conditions had the same 18 Markdown source files and four tasks;
- direct search returned 3/7 verbatim quotations and 2/13 strictly covered
  facets;
- compact memory returned 16/16 verbatim quotations and 13/13 facets;
- input tokens fell from 109,733 to 91,631 (`−16.5%`);
- total runtime tokens fell from 117,204 to 101,745 (`−13.2%`);
- the compact sandbox was physically larger, so the token reduction came from
  routing and structure rather than withholding source files.

#### Statement-level audit

The four compact answers initially passed quotation and facet checks. A later
least-context claim audit found:

- 14 claims and 16 exact citations;
- 5/14 claims directly supported;
- 9/14 claims broader than the displayed evidence;
- 1/4 answers eligible for promotion without revision;
- 3/4 correctly returned as `revision_required`.

This negative result changed the architecture: a real quotation is no longer
enough to promote a broader claim. Exact citation, required-facet coverage,
and statement support remain separate gates. Statement support was judged by one model
without human adjudication, so the result is diagnostic rather than final.

### Tesla biographical and technical corpus

| Field | Value |
|---|---:|
| Sources | 50 |
| Canonical fragments | 6,183 |
| Frozen questions | 40: 35 answerable and 5 deliberate gaps |
| Historical best lane: exact required evidence role @10 | 58.1% |
| Historical best lane: correct source @10 | 86.7% |
| Hit @1 | 31.4% |
| EvidenceGap recognition | 40.0% |

These figures belong to an earlier Qwen/graph/BGE development route, not the
current default FTS path. They are retained as evaluation history and should
not be read as the performance of `0.1.0rc0`. A later coverage-gate replay on
four known failures across three lanes produced six preserved gaps, two
partials, four insufficients, and no unsupported `ready` promotion.

### Artists corpus

| Field | Value |
|---|---:|
| Markdown files | 404 |
| Text size | 31.2 MB; approximately 2.40 million words |
| Active evaluation scope | 401 cards plus 2 holdouts |
| Indexed fragments | 48,072 |
| Natural Ukrainian questions | 18 |
| Exact address groups @10 | FTS 17/31 · E5 18/31 · BGE 22/31 · R2 25/31 |

This was primarily a parser, late-fragment access, holdout-isolation, and
retrieval-mechanics stress test. Gold labels and review were produced inside
the project, with an agent proxy rather than an independent human reviewer.
No semantic-winner, research-usefulness, or historical-truth claim is made.

### Maulstick knowledge connector

| Field | Current router | Compact hybrid |
|---|---:|---:|
| Source units | 383 | 383 |
| Answerable craft cases | 31 | 31 |
| At least one useful anchor | 26/31 | 27/31 |
| All required groups | 13/31 | 10/31 |
| Required anchor groups | 49/77 | 43/77 |
| Median context | 21,174 tokens | 1,147 tokens |
| Forbidden-source intrusions | 4/4 cases | 0/4 cases |
| Correct no-fit abstention | 0/5 | 5/5 |

The compact connector reduced median context by 94.6%, but lost required
anchors. The verdict was `ITERATE`. This is a useful example of why context
reduction and evidence coverage must be reported together.

### Directing and screenwriting craft corpora

| Field | Value |
|---|---:|
| Admitted runtime sources | directing 116 · screenwriting/dramaturgy 152 |
| Frozen fragments | 40,886 · 66,145 |
| Stable FTS rows | 120/120 completed |
| Sampled cold replays | 16/16 byte-exact |
| Model/provider calls in stable baseline | 0 |
| Blinded Ukrainian directing slice | 5 questions · 191 de-duplicated candidate spans |

Two answerable Ukrainian questions returned no lexical candidate and remained
explicit gaps. Of the three non-empty pools, Harrier produced three different
effects: it moved one axis-of-action passage from FTS rank 31 to rank 7; it was
effectively neutral on room tone, whose direct and qualified passages already
appeared at FTS ranks 2 and 4; and it substantially repaired a blocking query
whose FTS top 10 consisted entirely of drawing-related homonyms.

The blocking pool received a second independent blind agent review. Reviewers
agreed on 28/33 exact labels (84.8%), confirmed nine strict-support passages and
17 unsafe homonyms in common, and produced identical label totals. Harrier
placed five consensus-supported passages in its first ten unique results, but
two unsafe homonyms remained there. This supports a bounded reranking role plus
an exact evidence gate; it does not support using a model score as proof.

The corpus is not distributed, only three questions had non-empty pools, and
the reviewers were agents rather than independent human researchers. The result
is therefore a failure diagnosis, not an accuracy or state-of-the-art claim.

## Mixed adaptive-route replay

One ten-task development replay combined four Tesla tasks, four Mars tasks,
one Van Gogh task, and one Parisian Ten task:

- previous proof-window route: 7/10 correct final states;
- experimental Wide Gate: 10/10 final states;
- positive tasks ready: 4/7 → 7/7;
- declared gaps preserved: 3/3;
- declared hard negative admitted: 0/1;
- locally inspected text: 1,042,323 characters;
- final matched proof: 13,313 characters.

This does **not** mean 100% Dithyramba accuracy. The tasks were known, source
identities and literal anchors were specified in advance, and no claim-level
human evaluation was performed. The result isolated a mechanical defect: the
coverage gate had inspected only the first 30 reranked candidates even when
the exact proof was already present deeper in a bounded union.

## Engineering verification

The current public update collects 2,595 engineering cases. In its canonical
local gate:

- 2,593 passed and two declared browser cases were skipped;
- exact combined line/branch coverage was `95.0138%` at a strict `95.00%` gate;
- terminology, format, lint, and strict typing across 238 files passed;
- the dependency audit found no known vulnerabilities;
- a separate sdist-to-wheel closure installed non-editably in isolated Python
  3.11.12, imported from temporary `site-packages`, and included the schema-v10
  migration;
- the deterministic 1,000-fragment replay remains the rights-safe public
  research-path fixture.

Format, lint, terminology, strict typing, fixture validation, and the
release-surface audit are separate gates. These are engineering signals, not
research-quality metrics.

## What remains unproven

Dithyramba has not yet established:

- an independent cross-domain held-out benchmark;
- blinded evaluation by external researchers;
- human-adjudicated claim-to-citation entailment;
- calibrated near-domain abstention;
- automatic construction of equally strong memory units from arbitrary raw
  corpora;
- durable cold replay of the opt-in adaptive route;
- superiority over conversational-memory or research-agent systems measured
  under the same corpus, task, model, and review protocol.

Future public quality claims should ship a rights-cleared corpus or an
independently auditable evaluation card with frozen questions, denominators,
gold-construction policy, code revision, run hashes, and human-review method.
