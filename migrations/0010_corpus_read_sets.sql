-- Normalize exact protected-read manifests without changing ReadReceipt/1.0.
CREATE TABLE corpus_read_sets (
    corpus_read_set_id TEXT PRIMARY KEY CHECK (substr(corpus_read_set_id, 1, 16) = 'corpus_read_set_' AND length(corpus_read_set_id) > 16),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    retrieval_corpus_hash TEXT NOT NULL CHECK (length(retrieval_corpus_hash) = 64 AND retrieval_corpus_hash NOT GLOB '*[^0-9a-f]*'),
    item_count INTEGER NOT NULL CHECK (item_count >= 0),
    set_hash TEXT NOT NULL UNIQUE CHECK (length(set_hash) = 64 AND set_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE corpus_read_set_items (
    corpus_read_set_id TEXT NOT NULL REFERENCES corpus_read_sets(corpus_read_set_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    read_order INTEGER NOT NULL CHECK (read_order >= 0),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (corpus_read_set_id, source_fragment_id),
    UNIQUE (corpus_read_set_id, read_order)
);

CREATE TABLE read_receipt_corpus_sets (
    read_receipt_id TEXT PRIMARY KEY REFERENCES read_receipts(read_receipt_id) ON DELETE RESTRICT,
    corpus_read_set_id TEXT NOT NULL REFERENCES corpus_read_sets(corpus_read_set_id) ON DELETE RESTRICT
);

CREATE INDEX corpus_read_sets_library_corpus_idx
ON corpus_read_sets(library_id, retrieval_corpus_hash, corpus_read_set_id);

CREATE INDEX corpus_read_set_items_fragment_idx
ON corpus_read_set_items(source_fragment_id, corpus_read_set_id);

CREATE INDEX read_receipt_corpus_sets_set_idx
ON read_receipt_corpus_sets(corpus_read_set_id, read_receipt_id);

CREATE TRIGGER corpus_read_set_items_validate_insert
BEFORE INSERT ON corpus_read_set_items
BEGIN
    SELECT RAISE(ABORT, 'CorpusReadSet item crosses its Library or SourceVersion boundary')
    WHERE NOT EXISTS (
        SELECT 1
        FROM corpus_read_sets AS read_set
        JOIN source_fragments AS fragment
          ON fragment.source_fragment_id = NEW.source_fragment_id
        JOIN source_versions AS version
          ON version.source_version_id = NEW.source_version_id
         AND version.source_version_id = fragment.source_version_id
        JOIN sources AS source
          ON source.source_id = version.source_id
        WHERE read_set.corpus_read_set_id = NEW.corpus_read_set_id
          AND source.library_id = read_set.library_id
          AND fragment.text_sha256 = NEW.text_sha256
    );
END;

CREATE TRIGGER read_receipt_corpus_sets_validate_insert
BEFORE INSERT ON read_receipt_corpus_sets
BEGIN
    SELECT RAISE(ABORT, 'ReadReceipt cannot have legacy and shared item representations')
    WHERE EXISTS (
        SELECT 1 FROM read_receipt_items AS legacy
        WHERE legacy.read_receipt_id = NEW.read_receipt_id
    );

    SELECT RAISE(ABORT, 'ReadReceipt CorpusReadSet is incomplete or non-contiguous')
    WHERE NOT EXISTS (
        SELECT 1
        FROM corpus_read_sets AS read_set
        WHERE read_set.corpus_read_set_id = NEW.corpus_read_set_id
          AND read_set.item_count = (
              SELECT COUNT(*) FROM corpus_read_set_items AS item
              WHERE item.corpus_read_set_id = read_set.corpus_read_set_id
          )
          AND (
              (read_set.item_count = 0 AND NOT EXISTS (
                  SELECT 1 FROM corpus_read_set_items AS item
                  WHERE item.corpus_read_set_id = read_set.corpus_read_set_id
              ))
              OR
              (read_set.item_count > 0
               AND 0 = (
                   SELECT MIN(item.read_order) FROM corpus_read_set_items AS item
                   WHERE item.corpus_read_set_id = read_set.corpus_read_set_id
               )
               AND read_set.item_count - 1 = (
                   SELECT MAX(item.read_order) FROM corpus_read_set_items AS item
                   WHERE item.corpus_read_set_id = read_set.corpus_read_set_id
               ))
          )
    );

    SELECT RAISE(ABORT, 'ReadReceipt CorpusReadSet crosses its Library boundary')
    WHERE NOT EXISTS (
        SELECT 1
        FROM read_receipts AS receipt
        JOIN processing_runs AS run
          ON run.processing_run_id = receipt.processing_run_id
        JOIN query_requests AS request
          ON request.query_request_id = run.query_request_id
        JOIN corpus_read_sets AS read_set
          ON read_set.corpus_read_set_id = NEW.corpus_read_set_id
        WHERE receipt.read_receipt_id = NEW.read_receipt_id
          AND request.library_id = read_set.library_id
    );
END;

CREATE TRIGGER read_receipt_items_reject_shared_insert
BEFORE INSERT ON read_receipt_items
WHEN EXISTS (
    SELECT 1 FROM read_receipt_corpus_sets AS shared
    WHERE shared.read_receipt_id = NEW.read_receipt_id
)
BEGIN
    SELECT RAISE(ABORT, 'ReadReceipt cannot have shared and legacy item representations');
END;

CREATE TRIGGER corpus_read_sets_no_update BEFORE UPDATE ON corpus_read_sets BEGIN SELECT RAISE(ABORT, 'corpus_read_sets is append-only'); END;
CREATE TRIGGER corpus_read_sets_no_delete BEFORE DELETE ON corpus_read_sets BEGIN SELECT RAISE(ABORT, 'corpus_read_sets is append-only'); END;
CREATE TRIGGER corpus_read_set_items_no_update BEFORE UPDATE ON corpus_read_set_items BEGIN SELECT RAISE(ABORT, 'corpus_read_set_items is append-only'); END;
CREATE TRIGGER corpus_read_set_items_no_delete BEFORE DELETE ON corpus_read_set_items BEGIN SELECT RAISE(ABORT, 'corpus_read_set_items is append-only'); END;
CREATE TRIGGER read_receipt_corpus_sets_no_update BEFORE UPDATE ON read_receipt_corpus_sets BEGIN SELECT RAISE(ABORT, 'read_receipt_corpus_sets is append-only'); END;
CREATE TRIGGER read_receipt_corpus_sets_no_delete BEFORE DELETE ON read_receipt_corpus_sets BEGIN SELECT RAISE(ABORT, 'read_receipt_corpus_sets is append-only'); END;
