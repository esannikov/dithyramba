-- Dithyramba schema-v8 Relation import membership and review-admission foundation.

CREATE TABLE relation_import_run_items (
    relation_import_run_id TEXT NOT NULL REFERENCES relation_import_runs(relation_import_run_id) ON DELETE RESTRICT,
    relation_id TEXT NOT NULL REFERENCES relations(relation_id) ON DELETE RESTRICT,
    output_order INTEGER NOT NULL CHECK (output_order >= 0 AND output_order < 400),
    PRIMARY KEY (relation_import_run_id, relation_id),
    UNIQUE (relation_import_run_id, output_order)
);

CREATE TRIGGER relation_import_run_items_validate_insert
BEFORE INSERT ON relation_import_run_items
BEGIN
    SELECT RAISE(ABORT, 'Relation import membership must retain same-Library lineage and output bounds')
    WHERE NOT EXISTS (
        SELECT 1
        FROM relation_import_runs AS run
        JOIN relations AS relation ON relation.relation_id = NEW.relation_id
        WHERE run.relation_import_run_id = NEW.relation_import_run_id
          AND run.library_id = relation.library_id
          AND NEW.output_order < run.relation_count
    );
END;

INSERT INTO relation_import_run_items(
    relation_import_run_id, relation_id, output_order
)
SELECT
    relation_import_run_id,
    relation_id,
    ROW_NUMBER() OVER (
        PARTITION BY relation_import_run_id
        ORDER BY relation_id
    ) - 1
FROM relations;

CREATE TABLE relation_import_backfill_guard (
    valid INTEGER NOT NULL CHECK (valid = 1)
);

INSERT INTO relation_import_backfill_guard(valid)
SELECT CASE WHEN EXISTS (
    SELECT 1
    FROM relation_import_runs AS run
    LEFT JOIN relation_import_run_items AS item
      ON item.relation_import_run_id = run.relation_import_run_id
    GROUP BY run.relation_import_run_id, run.relation_count
    HAVING COUNT(item.relation_id) <> run.relation_count
) THEN 0 ELSE 1 END;

DROP TABLE relation_import_backfill_guard;

CREATE INDEX relation_import_run_items_relation_idx
ON relation_import_run_items(relation_id, relation_import_run_id);

CREATE TABLE relation_review_decisions (
    review_decision_id TEXT PRIMARY KEY CHECK (substr(review_decision_id, 1, 16) = 'review_relation_' AND length(review_decision_id) > 16),
    review_session_id TEXT NOT NULL REFERENCES meaning_review_sessions(review_session_id) ON DELETE RESTRICT,
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    target_type TEXT NOT NULL CHECK (target_type = 'relation'),
    target_id TEXT NOT NULL REFERENCES relations(relation_id) ON DELETE RESTRICT,
    target_hash TEXT NOT NULL CHECK (length(target_hash) = 64 AND target_hash NOT GLOB '*[^0-9a-f]*'),
    action TEXT NOT NULL CHECK (action IN ('accept', 'reject', 'revise', 'defer', 'supersede')),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    authority TEXT NOT NULL CHECK (length(trim(authority)) > 0),
    scope_json TEXT NOT NULL,
    replacement_target_id TEXT REFERENCES relations(relation_id) ON DELETE RESTRICT,
    supersedes_review_decision_id TEXT REFERENCES relation_review_decisions(review_decision_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    CHECK ((action = 'revise' AND replacement_target_id IS NOT NULL) OR (action <> 'revise' AND replacement_target_id IS NULL)),
    CHECK ((action = 'supersede' AND supersedes_review_decision_id IS NOT NULL) OR (action <> 'supersede' AND supersedes_review_decision_id IS NULL)),
    CHECK (replacement_target_id IS NULL OR replacement_target_id <> target_id)
);

CREATE INDEX relation_review_decisions_target_idx
ON relation_review_decisions(
    library_id, target_id, target_hash, scope_json, created_at, review_decision_id
);

CREATE INDEX relation_review_decisions_session_idx
ON relation_review_decisions(review_session_id, created_at, review_decision_id);

CREATE TABLE meaning_review_budget_backfill_guard (
    valid INTEGER NOT NULL CHECK (valid = 1)
);

INSERT INTO meaning_review_budget_backfill_guard(valid)
SELECT CASE WHEN EXISTS (
    SELECT 1
    FROM meaning_review_sessions AS session
    LEFT JOIN meaning_review_decisions AS decision
      ON decision.review_session_id = session.review_session_id
    GROUP BY session.review_session_id, session.decision_limit
    HAVING COUNT(decision.review_decision_id) > session.decision_limit
) THEN 0 ELSE 1 END;

DROP TABLE meaning_review_budget_backfill_guard;

CREATE TRIGGER meaning_review_decisions_validate_shared_budget_insert
BEFORE INSERT ON meaning_review_decisions
BEGIN
    SELECT RAISE(ABORT, 'Semantic ReviewDecision session must belong to its Library')
    WHERE NOT EXISTS (
        SELECT 1
        FROM meaning_review_sessions AS session
        WHERE session.review_session_id = NEW.review_session_id
          AND session.library_id = NEW.library_id
    );

    SELECT RAISE(ABORT, 'Semantic review session decision budget is exhausted')
    WHERE (
        SELECT COUNT(*)
        FROM meaning_review_decisions AS decision
        WHERE decision.review_session_id = NEW.review_session_id
    ) + (
        SELECT COUNT(*)
        FROM relation_review_decisions AS decision
        WHERE decision.review_session_id = NEW.review_session_id
    ) >= (
        SELECT session.decision_limit
        FROM meaning_review_sessions AS session
        WHERE session.review_session_id = NEW.review_session_id
          AND session.library_id = NEW.library_id
    );
END;

CREATE TRIGGER relation_review_decisions_validate_insert
BEFORE INSERT ON relation_review_decisions
BEGIN
    SELECT RAISE(ABORT, 'Relation ReviewDecision session must belong to its Library')
    WHERE NOT EXISTS (
        SELECT 1
        FROM meaning_review_sessions AS session
        WHERE session.review_session_id = NEW.review_session_id
          AND session.library_id = NEW.library_id
    );

    SELECT RAISE(ABORT, 'Relation ReviewDecision target hash must match a Relation in its Library')
    WHERE NOT EXISTS (
        SELECT 1
        FROM relations AS relation
        WHERE relation.relation_id = NEW.target_id
          AND relation.library_id = NEW.library_id
          AND relation.content_hash = NEW.target_hash
    );

    SELECT RAISE(ABORT, 'Relation ReviewDecision replacement must be a distinct Relation in its Library')
    WHERE NEW.action = 'revise'
      AND NOT EXISTS (
          SELECT 1
          FROM relations AS replacement
          WHERE replacement.relation_id = NEW.replacement_target_id
            AND replacement.library_id = NEW.library_id
            AND replacement.relation_id <> NEW.target_id
      );

    SELECT RAISE(ABORT, 'Relation review session decision budget is exhausted')
    WHERE (
        SELECT COUNT(*)
        FROM meaning_review_decisions AS decision
        WHERE decision.review_session_id = NEW.review_session_id
    ) + (
        SELECT COUNT(*)
        FROM relation_review_decisions AS decision
        WHERE decision.review_session_id = NEW.review_session_id
    ) >= (
        SELECT session.decision_limit
        FROM meaning_review_sessions AS session
        WHERE session.review_session_id = NEW.review_session_id
          AND session.library_id = NEW.library_id
    );
END;

CREATE TRIGGER relation_review_decisions_validate_active_insert
BEFORE INSERT ON relation_review_decisions
WHEN NEW.action <> 'supersede'
BEGIN
    SELECT RAISE(ABORT, 'Relation ReviewScope already has an active decision')
    WHERE EXISTS (
        SELECT 1
        FROM relation_review_decisions AS active
        WHERE active.library_id = NEW.library_id
          AND active.target_type = NEW.target_type
          AND active.target_id = NEW.target_id
          AND active.target_hash = NEW.target_hash
          AND active.scope_json = NEW.scope_json
          AND active.action <> 'supersede'
          AND NOT EXISTS (
              SELECT 1
              FROM relation_review_decisions AS later
              WHERE later.supersedes_review_decision_id = active.review_decision_id
          )
    );
END;

CREATE TRIGGER relation_review_decisions_validate_supersede_insert
BEFORE INSERT ON relation_review_decisions
WHEN NEW.action = 'supersede'
BEGIN
    SELECT RAISE(ABORT, 'Relation ReviewDecision supersede must retain Library, target, hash, and ReviewScope')
    WHERE NOT EXISTS (
        SELECT 1
        FROM relation_review_decisions AS prior
        WHERE prior.review_decision_id = NEW.supersedes_review_decision_id
          AND prior.library_id = NEW.library_id
          AND prior.action <> 'supersede'
          AND prior.target_type = NEW.target_type
          AND prior.target_id = NEW.target_id
          AND prior.target_hash = NEW.target_hash
          AND prior.scope_json = NEW.scope_json
    );

    SELECT RAISE(ABORT, 'Relation ReviewDecision is already superseded')
    WHERE EXISTS (
        SELECT 1
        FROM relation_review_decisions AS later
        WHERE later.supersedes_review_decision_id = NEW.supersedes_review_decision_id
    );
END;

CREATE TRIGGER relation_import_run_items_no_update BEFORE UPDATE ON relation_import_run_items BEGIN SELECT RAISE(ABORT, 'relation_import_run_items is append-only'); END;
CREATE TRIGGER relation_import_run_items_no_delete BEFORE DELETE ON relation_import_run_items BEGIN SELECT RAISE(ABORT, 'relation_import_run_items is append-only'); END;
CREATE TRIGGER relation_review_decisions_no_update BEFORE UPDATE ON relation_review_decisions BEGIN SELECT RAISE(ABORT, 'relation_review_decisions is append-only'); END;
CREATE TRIGGER relation_review_decisions_no_delete BEFORE DELETE ON relation_review_decisions BEGIN SELECT RAISE(ABORT, 'relation_review_decisions is append-only'); END;
