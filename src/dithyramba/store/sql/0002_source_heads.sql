-- Track the currently observed immutable content version without rewriting history.
CREATE TABLE source_heads (
    source_id TEXT PRIMARY KEY REFERENCES sources(source_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL UNIQUE REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    observed_at TEXT NOT NULL,
    source_modified_at TEXT
);

CREATE TRIGGER source_heads_validate_insert BEFORE INSERT ON source_heads
WHEN NOT EXISTS (
    SELECT 1
    FROM source_versions
    WHERE source_version_id = NEW.source_version_id
      AND source_id = NEW.source_id
)
BEGIN
    SELECT RAISE(ABORT, 'source head version belongs to another Source');
END;

CREATE TRIGGER source_heads_validate_update BEFORE UPDATE ON source_heads
WHEN OLD.source_id IS NOT NEW.source_id
    OR NOT EXISTS (
        SELECT 1
        FROM source_versions
        WHERE source_version_id = NEW.source_version_id
          AND source_id = NEW.source_id
    )
BEGIN
    SELECT RAISE(ABORT, 'source head update is invalid');
END;

CREATE TRIGGER source_heads_no_delete BEFORE DELETE ON source_heads
BEGIN
    SELECT RAISE(ABORT, 'source_heads cannot be deleted');
END;

INSERT INTO source_heads(source_id, source_version_id, observed_at, source_modified_at)
SELECT sv.source_id, sv.source_version_id, sv.observed_at, sv.source_modified_at
FROM source_versions AS sv
WHERE sv.version_number = (
    SELECT MAX(candidate.version_number)
    FROM source_versions AS candidate
    WHERE candidate.source_id = sv.source_id
);
