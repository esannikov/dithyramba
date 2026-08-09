# Schemas

Dithyramba's versioned contracts are defined by the canonical typed models under
`src/dithyramba/`. Their validated payloads, canonical bytes, schema identifiers,
and content hashes are authoritative.

The `1.0.0rc3` release candidate does not ship a second set of hand-maintained JSON
Schema files. That would let documentation drift away from executable validation.
The current contract inventory and schema identifiers are listed in
[`docs/REFERENCE.md`](../docs/REFERENCE.md).

Clean-install acceptance runs `scripts/contract_receipt.py`. It derives the CLI
tree and MCP input/output schemas from executable code, records the SQLite
fingerprint, checks local documentation links, and binds the result to one
commit hash. The generated receipt is release evidence, not a second source of
schema truth.

If generated JSON Schema artifacts are added later, they must be produced from the
canonical models, identify the generator and contract version, and pass parity
tests before release.
