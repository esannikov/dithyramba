# Research View

`dithyramba.research_projection/1.0` is an optional, case-owned companion to an
accepted Research Atlas manifest. It makes the wider research memory
discoverable without promoting that material into accepted evidence.

## Why it exists

An evidence release is intentionally selective. That selectivity is useful for
audit, but it can make a rich memory look like a thin illustrated biography.
The projection restores the surrounding chronology, themes, collections and
review priorities while preserving a hard distinction:

- **accepted Atlas objects** may support a displayed answer or hypothesis;
- **projection materials** may orient, suggest, complicate or become a review
  task, but cannot support an accepted conclusion until promoted through the
  ordinary evidence gates.

Visibility and evidential admissibility are therefore separate properties.

## Binding and invariants

Each projection declares:

- its own schema and projection identity;
- the exact `case_id`;
- the SHA-256 of the accepted Atlas manifest it accompanies;
- collection inventories, periods, themes and individual materials;
- exact source, locator, fragment hash and asserting voice for each material.

The loader fails closed when:

- the case or accepted manifest hash differs;
- an ID is duplicated or a reference dangles;
- a material points outside the accepted source boundary;
- a period or theme reference does not exist;
- a theme points to a missing accepted question, hypothesis or gap.

The renderer never merges projection materials into `questions[].evidence_ids`,
`hypotheses[].evidence_ids` or timeline evidence. Promotion requires a new,
reviewed Atlas release.

## Multiscale human flow

The same memory can now be read at five levels:

1. **Panorama** — themes, collection sizes and evidence passports.
2. **Research line** — one theme with its accepted questions, hypotheses,
   gaps and wider candidate field.
3. **Period** — a bounded life or project phase.
4. **Material** — one dated candidate, its voice, limitation and review state.
5. **Exact source** — a hash-verified local fragment.

This is reversible semantic compression: the first screen is calm because
details are folded into meaningful routes, not deleted.

## Run

```bash
uv run dithyramba atlas \
  --manifest /absolute/case/path/research/atlas/manifest.json \
  --projection /absolute/case/path/research/projection/manifest.json \
  --artifact-root /absolute/case/path
```

The projection is optional. Existing accepted manifests render unchanged when
`--projection` is omitted.
