# Storage schema

Dithyramba v1 installs one reviewed SQLite baseline from:

```text
src/dithyramba/store/sql/0001_v1.sql
```

That packaged file is the runtime source of truth. It contains the retained
source, scope, FTS evidence, review, reasoning, answer-projection, and research
session tables. It deliberately creates no embedding, vector, reranker,
model-runtime, or hybrid-retrieval tables.

The pre-v1 numbered migration chain remains available in Git history only. A
v1 process detects that history before applying connection-profile writes and
refuses to migrate the Library in place. Rebuild a new v1 Library from the
original read-only sources; keep the old Library unchanged until any review or
session artifacts that matter have been exported with the matching pre-v1
release.

Future v1 migrations must be numbered, immutable, checksummed, contiguous, and
backup-first. They may extend the v1 baseline, but must not silently reinterpret
or reintroduce retired storage.
