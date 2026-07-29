# Migrations

Numbered, immutable, checksummed SQLite migrations evolve the public storage
contract. The initial migration preserves the legacy `VS0` contract name used
inside schema identifiers; no generated migration framework is used.

Schema head v10 normalizes repeated protected-read manifests into append-only
`CorpusReadSet` records. Several receipts may reference one exact set only when
Library, snapshot, policy, Collections, and purpose all match; public
`ReadReceipt/1.0` remains unchanged and legacy v9 receipts remain readable.
Schema v9 adds hash-pinned connector-declared logical Source identity tables.
Schema v7 adds append-only, text-free SemanticSpan plans and vector-v2
generations. SQLite enforces source lineage, profile/model/runtime bindings,
token and character bounds, exact plan counts before generation, and float32
blob length. The persistence layer must still reconstruct every canonical hash
and verify each vector blob SHA-256 inside its write transaction; stock SQLite
cannot perform canonical JSON hashing or defer an application-defined hash
check until commit. SemanticSpan validation resolves source lineage through
the stored fragment hash and never selects the protected
`source_fragments.text` column. Migration v7 backfills the append-only
`source_fragment_text_metrics` projection once; an `AFTER INSERT` trigger uses
only the just-inserted `NEW.text` value to capture hash and character count for
future fragments. SemanticSpan validation reads that metadata-only projection.
