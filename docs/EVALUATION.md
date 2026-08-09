# Evaluation evidence and limits

This page records what Dithyramba has actually been exercised on for the
historical package candidates and the current stable v1 release. It separates
three different claims:

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
| `v1_fresh_corpus` | 400 newly generated sources; 2,800 fragments; 30 questions | top-one document, source address, and exact replay `30/30`; unchanged reuse `400/400`; model calls/tokens `0/0` | Clean ingest, incremental reuse, lexical retrieval, addressability, and replay on previously unseen synthetic inputs. |
| Isolation fixtures | 2 Libraries; 4 public/private/holdout/excluded canaries | all fixture invariants passed | Scope is compiled before reading and excluded data does not silently enter recall. |
| IdeaTrace closure fixtures | 2-step public chain plus stale, tampered, orphaned, failed-claim, persistence, corruption, and CLI cases | deterministic replay matched; unsafe paths failed or required review; provider calls `0` | Contract closure and fail-closed persistence, not the usefulness or truth of a generated idea. |
| Answer projection and Lens trace fixtures | exact proposition spans, semantic receipts, clean v1 reopen, corruption, Library isolation, sparse source-linked Atlas spans | exact replay matched; invalid, overlapping, out-of-scope, or stale spans failed closed; provider calls `0` | Durable display governance and exact phrase-to-evidence routing, not semantic truth or automatic acceptance. |

The 1,000-fragment run is an engineering workload, not a semantic research
benchmark. Its text and queries are deliberately easy and synthetic.

## Internal corpus screens

### Mars working corpus

| Field | Value |
|---|---:|
| Retrieval units | 53,747 |
| Normalized text | approximately 103.8 million characters |
| Evaluation cases | 72 bilingual cases: 51 positive, 21 controls |
| Current route | bounded flat FTS expansion + exact proof admission |
| Exact fragment @1 / @5 / @10 | 22/51 · 39/51 · 42/51 |
| Correct source @10 | 49/51 |
| Exact MRR | 0.572 |
| Non-body items after structural admission | 0 |
| Network, generative calls, generative tokens | 0 · 0 · 0 |

This was the largest retrieval-scale calibration. The current model-free route
recovered the exact fragment at top 10 for 42/51 positive cases (82.4%) and the
correct source for 49/51 (96.1%). It was byte-equivalent on replay and required
no model. It did not test complete generated answers. Candidate availability
was known during development, so these numbers are calibration evidence rather
than a held-out benchmark.

The reports record units and characters, but not a trustworthy aggregate page
or original-document count. No page estimate is inferred here.

### Mars multi-session screen (0.2 development)

The internal protocol revision 0.3 reused the same frozen normalized Library:
2,047
admitted Library sources and 53,747 retrieval units. It ran six realistic
three-turn investigations—environment, transport, habitat, human factors,
society, and unsupported controls—plus one isolated three-query relation-repair
session. It used no model, embedding, network inference, or generative token.

| Measure | Result |
|---|---:|
| Sessions with exactly one authorized read / FTS build | 6/6 |
| Sessions with three ordinary searches | 6/6 |
| Mean first turn | 50.90 s |
| Mean second/third turn | 38.76 s |
| Mean warm-turn reduction | 23.85% |
| Comparable v0.2 top-10 lists unchanged | 17/17 |
| Positive correct-source @1 / @5 / @10 | 12/15 · 13/15 · 14/15 |
| Exact idempotent retry | 6/6 |
| Exact context after cold Library reopen | 6/6 |

This result supports one narrow claim: reusing a process-local authorized
read-set and in-memory FTS index avoids rebuilding it for later questions
without changing ranking. It does not make large-corpus turns instantaneous.
Most remaining time is spent assembling, hashing, and transactionally storing
the full local audit packet and its corpus-wide receipt.

The one known EDL relation-paraphrase miss was preregistered as an isolated
challenger. The original query missed at top 10; a neutral relation formulation
recovered the expected source at rank 2; an explicit keyword formulation missed
again. Because only one of two repairs worked on one known failure, automatic
relation-aware repair was not added to production. The gate requires at least
two independent recurring failures and a frozen holdout gain without control
degradation.

Full internal result hash:
`a3dba797739f04a5d34c3e63fb9d10ef369455a01073e329d32ddd52368149e3`.
The redistributable evaluation runner and report live outside the package tree;
the private heavy result and corpus remain on the operator's STORAGE volume.

### Mars IdeaTrace-24

This test held retrieval constant and compared ordinary cited answers with
inspectable reasoning over 24 frozen evidence packets:

- 6 direct facts;
- 6 multi-source syntheses;
- 6 contested claims;
- 6 unsupported traps.

The first ordinary and IdeaTrace passes both produced 21/24 complete answers.
The IdeaTrace structure became useful when the gates returned exact failure
reasons. After one repair of the 12 complex answers, complete answers improved
to 23/24, correct answer status to 24/24, and full evidence closure to 16/18.
All six unsupported traps remained safe abstentions. One semantic-scope
overclaim and one punctuation-level exact-quote mismatch remained blocked.

The run used approximately 188,106 input and 37,992 output model tokens across
five development passes. Deterministic span, coverage, persistence, and closure
checks themselves used no model tokens. This is model-judged internal
diagnostic evidence, not an independent benchmark. See the
[full historical protocol, all questions, answers, and limits](history/evaluations/MARS_IDEATRACE_24.md).

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
not be read as the performance of the current v1 package. A later coverage-gate replay on
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

### Retired universal-cartography profile screen

The former global-cartography contracts were exercised on source-stratified
samples from the same two craft corpora without generative calls. Every sample
retained all source documents and bound exact fragment, address, text, vector,
runtime, policy, and input-manifest hashes.

| Profile | Corpus | Areas | Assigned | Largest assigned area |
|---|---|---:|---:|---:|
| TF-IDF/SVD | directing | 17 | 604/1,342 | 163/604 |
| TF-IDF/SVD | screenwriting/dramaturgy | 2 | 1,776/1,787 | 1,756/1,776 |
| raw Harrier | directing | 2 | 980/1,353 | 956/980 |
| Harrier with batch PCA, before refinement | directing | 10 | 841/1,353 | 268/841 |
| Harrier with batch PCA, before refinement | screenwriting/dramaturgy | 16 | 842/1,789 | 244/842 |
| document-first E5-small, no PCA | directing | 2 document / 5 fragment | 605/1,342 | 542/605 |
| document-first E5-small, no PCA | screenwriting/dramaturgy | 6 document / 2 fragment | 57/1,787 | 39/57 |

All recorded final maps were equivalent under reordered input and canonical
reopen. None passed the preregistered automatic navigation gate across both
corpora: 8–35 fragment areas, 50–85% assigned mass, no area above 25% of
assigned mass, and a median of at least three sources per area. Independent
human coherence scoring was therefore not opened.

This is a useful negative result. It rejected the claim that one flat global
embedding geometry is already a useful universal map. The public cartography
module was removed rather than retained as architectural ballast.

### Scoped dialogue-directing ontology

The replacement was tested on seven locally held foundational craft books:
5,432 extracted fragments and a bounded 500-result FTS neighbourhood.
Construction used no generative calls or tokens. The first run balanced twelve
QueryCloud branches before local TF-IDF/NMF extraction so a popular camera
vocabulary could not consume the entire neighbourhood.

The result is a candidate navigation projection, not a knowledge-quality score.
The preregistered diagnostic checks are:

- at least 8/10 named craft facets visibly represented;
- no HTML/markup labels in the visible concept set;
- every shown `co_occurs_with` relation supported by at least two sources;
- byte-exact replay under reversed input order.

The final automatic gate passed: 7 clusters, 41 concepts, 36 multi-source
co-occurrences, 9/10 diagnostic facets, no visible markup concepts, and
byte-exact reordered-input replay. `subtext` was the missing visible facet.
These counts describe projection closure and breadth, not truth or human
coherence.

A local neural reorder was tested only as an A/B comparator. It reduced facet
breadth in the final run (`9/10 → 8/10`) and has since been removed from the
runtime. The minimal ontology route does not require an embedding model. Human
cluster-coherence review is still needed; lexical facet coverage cannot
establish that an area name is useful to a director.

A follow-up ablation held the corpus and final fragment budget fixed and
compared five retrieval routes at 180 and 360 fragments: one question, one flat
expanded question, multi-query fusion, balanced branches, and a historical
model-assisted branch. The flat expanded query reached 10/10 visible diagnostic
facets and all seven sources at both budgets. Multi-query routes reached 9/10
and the retired model-assisted branch 8/10. Balanced branches did retain all
top-20 branch results at the
360-fragment budget, but that mechanical breadth did not improve the final
candidate map. A complete rerun produced an identical metrics hash.

The resulting simplification is evidence-specific: a clear scope defaults to
one compact expansion and one FTS search. Separate branches are reserved for an
ambiguous scope or a coverage gap. The 180-fragment view is a useful preview,
but only 67.3% of flat-route concepts overlapped with the 360-fragment map, so
the larger budget remains appropriate for a final orientation view. These are
lexical and structural diagnostics, not a human relevance or truth score.

## Historical mixed-route replay

One ten-task development replay combined four Tesla tasks, four Mars tasks,
one Van Gogh task, and one Parisian Ten task:

- previous proof-window route: 7/10 correct final states;
- Wide Gate proof scan: 10/10 final states;
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

## M1 PhD scholarly holdout

A private research corpus was used to test source-grounded navigation without
shipping its copyrighted contents. The questions and three query variants per
question were frozen before retrieval. The corrected run excluded seven
author-side or duplicated SourceFamilies and searched **615 sources / 259,165
fragments**.

| Measure | Result |
|---|---:|
| Frozen questions / local searches | 24 / 72 |
| Total wall time / cold scope | 1,219.752 s / 56.818 s |
| Median per question, three searches | 39.959 s |
| Exact replay | 0.006 s, unchanged hash |
| Direct support in top 3 | 20/72 (27.8%) |
| Useful evidence in top 5 | 61/120 (50.8%) |
| Questions with any direct top-3 support | 12/24 |
| Questions with at least four useful top-5 fragments | 9/24 |
| Dithyramba model calls / tokens | 0 / 0 |

The test deliberately keeps retrieval separate from evidence admission. All
three unsupported trap questions still returned lexical candidates. A broad
query also found sources about neural art while leaving exact named-work
examples below the top five. Therefore candidate abundance and source-family
diversity are diagnostics, not answer confidence. The labels are a conservative
agent audit and require independent human adjudication before any external
validity claim.

The run caused one product contract correction: `AgentEvidencePacket/1.2` labels
raw recall as `retrieved_candidates` and carries exact source/family dominance
diagnostics. It does not claim that ranking precision improved. Planned repairs
remain bounded. The answer-route drilldown now exists: deterministic bibliography/
index/table/topic-noise guards, conditional source-local drilldown, and
interactive `EvidenceCoverageGate` binding. Reviewed cross-format source
identity and human calibration remain open; the engineering checks below do
not substitute for that research-quality review.

## M1 PhD clean v1 rebuild

The old M1 Libraries and rebuildable caches were removed under explicit owner
authorization. The source corpus remained read-only. A new v1 Library was then
built from 829 supported physical PDF, Markdown, and text inputs distributed
across five Collections.

| Measure | Result |
|---|---:|
| Collection entries discovered / processed | 849 / 836 |
| Unique sources / versions / families | 816 / 816 / 816 |
| Exact source fragments | 276,165 |
| Cold ingest | 2,882.42 s (48:02) |
| Unchanged five-Collection repeat | 39.62 s |
| New versions / fragments on repeat | 0 / 0 |
| Corpus omissions | 8 no-text, 4 encrypted, 1 invalid PDF signature |
| Doctor | schema 1, FTS5 available, no external service, `ok` |
| Model calls / tokens | 0 / 0 |

The rebuild exposed five general ingest defects rather than one corpus-specific
special case. Recoverable NUL glyphs and invalid zero-area boxes no longer fail
an otherwise readable PDF. The active parser profile is persisted as part of
source representation identity. The same source bytes parsed under a different
profile now create a separate immutable version instead of silently resolving
to an older representation. The bounded large-document worker admitted a
568-page book and a valid 1,698-page encyclopedia while retaining the 180-second,
512 MiB file, 20-million-character, and 2,560 MiB worker limits.

The same Library then served 30 philosophy-of-art and art-history questions.
Each question used its original lexical query plus one deterministic branch per
evidence requirement: 90 local searches and 2,700 candidate assessments in
total.

| Measure | Result |
|---|---:|
| Total / cold scope | 1,986.217 s / 62.128 s |
| Median question | 58.574 s |
| Source cards / unique works | 240 / 139 |
| Literal Gate-support / related-reading cards | 60 / 180 |
| Human-report file, locator, and bibliography QA | 240/240 |
| Gate decisions | 30/30 `ready` |
| Model calls / tokens | 0 / 0 |

`Ready` here proves only that the selected passages contain the declared
literal topic and evidence-role anchors. It does not prove source truth,
semantic entailment of an answer, or equal usefulness of all 240 cards.

One control question also exposed a scope-design problem: in a mixed all-scope,
the first six results were the researcher's own drafts. They were relevant as
author context but methodologically unsuitable as independent confirmation.
The existing snapshot and policy boundary was sufficient to separate 662
independent evidence sources from 174 author-context sources. On the evidence
scope the drafts disappeared from the leading results, and packet
`packet_ee753ce17d21783b346e41b40c2c31b5` replayed with the same packet SHA-256
`ee753ce17d21783b346e41b40c2c31b5c66bc54d7f0c4cc4ef79ac0d597dff0b`.

An isolated parser comparison did not change the v1 dependency set.
Firecrawl's Rust `pdf-inspector` was roughly 15–18 times faster on three
text-native PDFs (34, 176, and 1,698 pages) and correctly identified one true
scan as OCR-required. It nevertheless returned zero text for a 568-page source
from which the current `pdfplumber` route extracted 566 pages. The candidate is
therefore recorded only as a possible future fast-first subprocess with an
automatic fallback and paired completeness gate.

## M1 Cinema strict text-native screen

An additional real-corpus screen tested an isolated optional preprocessing
route for film and screenwriting literature. It is reported separately from the
default parser because the candidate route deliberately excluded every file
that requested OCR, looked mixed or scanned, or returned a structural error.

| Measurement | Observed result |
|---|---:|
| Candidate files inspected | 365 |
| Admitted text-native/prepared sources | 265 |
| Explicit exclusions | 100 |
| Candidate bytes | 1,765,950,412 |
| Prepared characters | 33,710,000 |
| Parser preparation | 37.58 s |
| Cold Dithyramba ingest | 15.88 s |
| Unchanged repeat | 265/265 in 9.77 s |
| Stored fragments | 134,771 |
| Collections / snapshot members | 5 / 265 |
| Complete memory cycle | 44.61 s |
| Model calls / model tokens | 0 / 0 |

The complete cycle included Library initialization, ingest, unchanged repeat,
two doctor checks, snapshot freeze, and portable backup. The final SQLite quick
check and Dithyramba doctor passed; all 265 derived hashes matched and no NUL
metadata remained. The test found two general engineering issues: backup output
must be owner-only, and imported PDF titles need control-character
normalization. Both were corrected before the clean rebuild.

This screen demonstrates fast bounded intake when a researcher accepts strict
exclusion. It does not prove that the 100 excluded books were unimportant, that
the parser is complete, or that the resulting corpus supports a particular
research conclusion. The optional preprocessor remains isolated and does not
replace Dithyramba's default PDF route.

## Engineering verification

The current stable v1 release runs the complete engineering suite. In its canonical
gate:

- all required tests must pass, with any host-dependent skips reported rather
  than hidden;
- branch-aware combined coverage must meet the strict `95.00%` threshold;
- terminology, format, lint, and strict typing passed;
- the dependency audit found no known vulnerabilities;
- a separate sdist-to-wheel closure installed non-editably in isolated Python
  3.11.12, imported from temporary `site-packages`, and included the single
  clean v1 schema baseline;
- the deterministic 1,000-fragment replay remains the rights-safe public
  research-path fixture;
- the fresh-corpus acceptance ingested 400 sources / 2,800 fragments in
  `1.955 s`, reused all unchanged sources in `0.782 s`, and answered 30 exact
  lexical probes in `20.323 s` total (`0.675 s` median) with 30/30 top-one
  document hits, source coordinates, and exact replays.

Format, lint, terminology, strict typing, fixture validation, and the
release-surface audit are separate gates. Exact test counts and coverage values
belong to the commit-bound release acceptance record because randomized test
paths and environment details can change the measured total. These are
engineering signals, not research-quality metrics.

## What remains unproven

Dithyramba has not yet established:

- an independent cross-domain held-out benchmark;
- blinded evaluation by external researchers;
- human-adjudicated claim-to-citation entailment;
- calibrated near-domain abstention;
- automatic construction of equally strong memory units from arbitrary raw
  corpora;
- durable cold replay of optional query-expansion plans;
- superiority over conversational-memory or research-agent systems measured
  under the same corpus, task, model, and review protocol.

Future public quality claims should ship a rights-cleared corpus or an
independently auditable evaluation card with frozen questions, denominators,
gold-construction policy, code revision, run hashes, and human-review method.
