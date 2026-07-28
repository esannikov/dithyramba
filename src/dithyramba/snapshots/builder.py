"""Pure CorpusSnapshot construction."""

from __future__ import annotations

from collections.abc import Iterable

from .models import CorpusSnapshot, SnapshotMember


def build_corpus_snapshot(
    *,
    library_id: str,
    collection_ids: Iterable[str],
    members: Iterable[SnapshotMember] = (),
) -> CorpusSnapshot:
    """Build a deterministic snapshot without reading or persisting state."""

    return CorpusSnapshot(
        library_id=library_id,
        collection_ids=tuple(collection_ids),
        members=tuple(members),
    )
