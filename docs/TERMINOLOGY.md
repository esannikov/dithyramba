# Dithyramba terminology

These terms keep discovery, evidence admission, drafting, and human judgment
separate across the CLI, MCP, Lens, documentation, and tests.

## Research path

| Term | Meaning |
|---|---|
| Retrieved candidate | A fragment returned by bounded search. Rank sets inspection order only. |
| Quality-eligible candidate | A retrieved fragment that passed deterministic bibliography, index, table, and topic-drift guards. It has not yet satisfied an evidence requirement. |
| Gate-matched evidence | A quality-eligible exact fragment that satisfies one declared evidence requirement under the selected matching profile. This remains evidence of what the source says. |
| Draft | Agent-authored prose recorded from one exactly replayed ready preparation. It is not a human decision. |
| Human-reviewed result | A separately scoped append-only review decision. This is the only route to acceptance or rejection. |

The compact path is:

```text
retrieved candidate
  → quality-eligible candidate
  → gate-matched evidence
  → draft
  → human review
```

## Source identity and independence

| Term | Meaning |
|---|---|
| `Source` | One logical work with one or more immutable byte versions. |
| `SourceFamily` | Canonical lineage identity for related Sources. The legacy public field `EvidenceCandidate.source_family` carries this identifier. |
| Independence group | Caller-supplied corroboration unit checked for consistency. Several Sources may belong to one group and therefore do not count as independent support. |
| Authority | Declared source-role metadata used by an evidence requirement. It is not inferred from file type or search rank. |

## Two meanings of partial

| Display phrase | Meaning |
|---|---|
| Partial projection coverage | A Connector preserved usable text but disclosed representational loss while projecting a source. |
| Partial evidence coverage | The evidence gate covered some declared requirements while others remain open. |

These states belong to different layers and must not be merged into one generic
confidence score.

## Contract evolution

Schema identifiers freeze byte meaning. Compatibility aliases remain documented
until their versioned artifacts no longer need replay. New public prose should
use the canonical terms in this file and the contract inventory in
[REFERENCE.md](REFERENCE.md).
