# Public verification

This directory contains the smallest rights-safe replay that checks the public
Dithyramba evidence path without a private corpus, model, API key, or network
call. It is release verification, not a general claim about research quality.

## What it contains

| File | Purpose |
|---|---|
| `generate_synthetic_1000.py` | Deterministically creates 1,000 synthetic fragments and 20 exact lexical probes. |
| `synthetic_1000_manifest.json` | Freezes the expected fragment IDs, counts, profile, and canonical payload hash. |
| `run_public_replay.py` | Exercises ingest, default-deny scope, FTS recall, receipts, persistence, and exact packet replay. |

The generated text is repository-authored synthetic data released under
`CC0-1.0`. Generated payloads and run outputs stay outside Git.

## Verify the input

```bash
uv run python -m verification.generate_synthetic_1000 --check
```

To inspect the complete canonical payload:

```bash
uv run python -m verification.generate_synthetic_1000 \
  --output /tmp/dithyramba-synthetic-1000.json
```

## Replay the public evidence path

```bash
uv run python -m verification.run_public_replay \
  --warmups 1 --measured 2 \
  --output /tmp/dithyramba-public-replay.json
```

A passing replay proves that one frozen synthetic input follows the expected
local engineering path and produces one deterministic evidence packet. It does
not prove source truth, semantic recall on natural language, claim entailment,
or usefulness to an independent researcher. Those require separate evaluation
with disclosed corpora and human review.
