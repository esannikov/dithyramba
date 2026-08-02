# How to use Dithyramba

This guide builds one local, replayable evidence memory from source files with
the persisted FTS-only CLI route. Package metadata remains `0.1.0rc1`; the
current `0.2` development candidate also adds durable research sessions, local
stdio MCP, and Session Lens. Replace the example identifiers with the values
printed by each command.

To verify a fresh checkout, follow the separate
[clean-install acceptance protocol](ACCEPTANCE.md).

## 1. Choose separate source and runtime locations

```text
/absolute/path/to/corpus/       source files owned by you
/absolute/private/dithyramba/   live application data
/absolute/private/backups/      verified backup bundles
```

The runtime path must not be inside the corpus, an Obsidian vault, or a synced
folder. Dithyramba refuses unsafe overlaps when they are declared.

## 2. Create a Library

```bash
uv run dithyramba library init "Example research" \
  --data-home /absolute/private/dithyramba \
  --source-root /absolute/path/to/corpus \
  --json
```

Save the returned `library_id`. A Library is the physical isolation, database,
and backup boundary.

Verify it before adding data:

```bash
uv run dithyramba library doctor \
  --library <library-id> \
  --data-home /absolute/private/dithyramba
```

## 3. Add one or more Collections

A Collection is a logical scope inside the Library. Keep primary material and
later interpretation separate when their rank order could be misleading.

```bash
uv run dithyramba collection add "Primary sources" \
  --library <library-id> \
  --data-home /absolute/private/dithyramba \
  --root /absolute/path/to/corpus/primary \
  --kind corpus \
  --json

uv run dithyramba collection add "Research literature" \
  --library <library-id> \
  --data-home /absolute/private/dithyramba \
  --root /absolute/path/to/corpus/research \
  --kind corpus \
  --json
```

By default the collection discovers Markdown, plain text, and PDF files. Use
repeatable `--include` and `--exclude` globs to narrow the scope.

## 4. Create a default-deny AccessPolicy

```bash
uv run dithyramba access-policy create "Local research" \
  --library <library-id> \
  --data-home /absolute/private/dithyramba \
  --purpose research \
  --allow-collection <primary-collection-id> \
  --allow-collection <research-collection-id> \
  --json
```

External-provider disclosure and export are false unless explicitly enabled.
Check policy without reading corpus text:

```bash
uv run dithyramba access-policy check <policy-id> \
  --library <library-id> \
  --data-home /absolute/private/dithyramba \
  --purpose research \
  --collection <primary-collection-id>
```

## 5. Ingest sources

Ingest the whole Collection:

```bash
uv run dithyramba index \
  --library <library-id> \
  --collection <primary-collection-id> \
  --data-home /absolute/private/dithyramba \
  --json
```

Or add one exact relative path:

```bash
uv run dithyramba source add notes/example.md \
  --library <library-id> \
  --collection <primary-collection-id> \
  --data-home /absolute/private/dithyramba \
  --json
```

Read the `CoverageReport`. A skipped encrypted PDF, parser failure, or resource
limit is part of the result and must not be treated as successful coverage.

## 6. Freeze an exact snapshot

```bash
uv run dithyramba collection freeze \
  --library <library-id> \
  --collection <primary-collection-id> \
  --data-home /absolute/private/dithyramba \
  --json
```

Save the `corpus_snapshot_id`. A later source update does not change this
snapshot, which makes query replay meaningful.

Freeze the research Collection separately if you want a clean two-scope
comparison.

## 7. Recall evidence

```bash
uv run dithyramba recall "Which source states the exact date?" \
  --library <library-id> \
  --collection <primary-collection-id> \
  --snapshot <primary-snapshot-id> \
  --access-policy <policy-id> \
  --purpose research \
  --max-candidates 100 \
  --max-source-fragments 30 \
  --data-home /absolute/private/dithyramba \
  --json
```

The stable route creates a persisted `EvidencePacket/1.0`. Record its
`evidence_packet_id` and `packet_hash`.

For a two-scope workflow, ask the same question again with the research
Collection and its own snapshot. You now have two independently replayable
packets: what the root sources show and what later researchers infer.

### Continue the investigation through an agent

Start the local stdio MCP server from an MCP-capable client:

```bash
uv run dithyramba mcp \
  --library <library-id> \
  --data-home /absolute/private/dithyramba \
  --agent-id agent:research-assistant
```

The client launches that command and speaks newline-delimited JSON-RPC over
standard input/output. Dithyramba exposes six bounded tools:

```text
open_session → recall → session_context
             → record_draft / record_gap / reject_path
```

`open_session` requires the research question, intended use, success criteria,
snapshot, policy, purpose, and Collection IDs. Save its `session_id`. Every
`recall` also requires a stable `command_id`; retrying the same command with the
same input returns the stored turn, while changing the input fails closed.

The first recall builds one process-local authorized read-set and FTS index for
that exact session scope. Later questions reuse it. The cache is destroyed when
the process exits or closes its scope sessions, and it is rebuilt after restart.
This optimization does not merge packets or review history.

Open the human journal for that session in a separate terminal:

```bash
uv run dithyramba session-lens \
  --library <library-id> \
  --session <research-session-id> \
  --data-home /absolute/private/dithyramba \
  --port 8353
```

Visit `http://127.0.0.1:8353`. Session Lens is GET-only. Select a source slip to
read the exact packet-backed passage. Drafts, gaps, and rejected routes remain
journal entries; visible evidence is not automatically accepted.

## 8. Inspect and replay

```bash
uv run dithyramba packet inspect <packet-id> \
  --library <library-id> \
  --data-home /absolute/private/dithyramba

uv run dithyramba packet read-receipt <packet-id> \
  --library <library-id> \
  --data-home /absolute/private/dithyramba

uv run dithyramba packet replay <packet-id> \
  --library <library-id> \
  --data-home /absolute/private/dithyramba
```

`inspect` reconstructs and revalidates the stored packet. `read-receipt` shows
the exact protected fragments materialized for it. `replay` reruns the frozen
local profile and requires the same packet hash.

## 9. Review and human reading

```bash
uv run dithyramba review queue --help
uv run dithyramba review decide --help

uv run dithyramba reading-room \
  --library <library-id> \
  --snapshot <primary-snapshot-id> \
  --access-policy <policy-id> \
  --collection <primary-collection-id> \
  --purpose research \
  --data-home /absolute/private/dithyramba
```

Machine candidates remain separate from accepted human decisions. Review
scope and authority are explicit and append-only.

## 10. Back up before migration or major work

```bash
uv run dithyramba backup \
  --library <library-id> \
  --data-home /absolute/private/dithyramba \
  --output-directory /absolute/private/backups
```

Restore only into a new isolated root:

```bash
uv run dithyramba restore /absolute/private/backups/<bundle> \
  --data-home /absolute/private/restored-dithyramba
```

Restore verifies paths, file modes, hashes, schema, migrations, rows, event
stream, and blob closure before opening the Library.

## Optional query expansion

Use the original question first. A compact deterministic expansion may add
aliases, translated names, and close domain phrases. QueryCloud is reserved for
a named coverage gap and is bounded to two derived queries.

Derived queries are discovery instructions, not evidence. Persist the original
question, exact source fragments, and evidence decision; do not cite a generated
query or a search score.

## Common mistakes

- Putting runtime SQLite or model caches in a synced folder.
- Using a combined primary/research scope when provenance layer matters.
- Treating a search score as proof.
- Ignoring skips in the ingest CoverageReport.
- Reusing a snapshot after adding sources and assuming it changed.
- Treating a derived query as evidence or as a replacement for the operator's
  original question.
