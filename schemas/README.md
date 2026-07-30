# Schemas

Dithyramba's versioned contracts are defined by the canonical typed models under
`src/dithyramba/`. Their validated payloads, canonical bytes, schema identifiers,
and content hashes are authoritative.

The `0.1.0rc1` source preview does not ship a second set of hand-maintained JSON
Schema files. That would let documentation drift away from executable validation.
The current contract inventory and schema identifiers are listed in
[`docs/REFERENCE.md`](../docs/REFERENCE.md).

If generated JSON Schema artifacts are added later, they must be produced from the
canonical models, identify the generator and contract version, and pass parity
tests before release.
