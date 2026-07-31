-- Append-only proposition projections and their independent semantic receipts.
CREATE TABLE answer_projections (
    projection_id TEXT PRIMARY KEY CHECK (substr(projection_id, 1, 18) = 'answer_projection_' AND length(projection_id) = 50 AND substr(projection_id, 19) NOT GLOB '*[^0-9a-f]*'),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    schema_id TEXT NOT NULL CHECK (schema_id = 'dithyramba.answer_projection/1.0'),
    projection_hash TEXT NOT NULL UNIQUE CHECK (length(projection_hash) = 64 AND projection_hash NOT GLOB '*[^0-9a-f]*'),
    answer_id TEXT NOT NULL CHECK (substr(answer_id, 1, 16) = 'research_answer_' AND length(answer_id) = 48 AND substr(answer_id, 17) NOT GLOB '*[^0-9a-f]*'),
    answer_hash TEXT NOT NULL CHECK (length(answer_hash) = 64 AND answer_hash NOT GLOB '*[^0-9a-f]*'),
    case_set_id TEXT NOT NULL CHECK (substr(case_set_id, 1, 24) = 'claim_evidence_case_set_' AND length(case_set_id) = 56 AND substr(case_set_id, 25) NOT GLOB '*[^0-9a-f]*'),
    case_set_hash TEXT NOT NULL CHECK (length(case_set_hash) = 64 AND case_set_hash NOT GLOB '*[^0-9a-f]*'),
    entailment_receipt_id TEXT NOT NULL CHECK (substr(entailment_receipt_id, 1, 32) = 'claim_evidence_judgment_receipt_' AND length(entailment_receipt_id) = 64 AND substr(entailment_receipt_id, 33) NOT GLOB '*[^0-9a-f]*'),
    entailment_receipt_hash TEXT NOT NULL CHECK (length(entailment_receipt_hash) = 64 AND entailment_receipt_hash NOT GLOB '*[^0-9a-f]*'),
    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'model', 'deterministic')),
    projection_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (projection_id, library_id)
);

CREATE TABLE answer_projection_receipts (
    receipt_id TEXT PRIMARY KEY CHECK (substr(receipt_id, 1, 26) = 'answer_projection_receipt_' AND length(receipt_id) = 58 AND substr(receipt_id, 27) NOT GLOB '*[^0-9a-f]*'),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    projection_id TEXT NOT NULL,
    projection_hash TEXT NOT NULL CHECK (length(projection_hash) = 64 AND projection_hash NOT GLOB '*[^0-9a-f]*'),
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*'),
    judge_kind TEXT NOT NULL CHECK (judge_kind IN ('human', 'model', 'deterministic')),
    receipt_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (projection_id),
    FOREIGN KEY (projection_id, library_id) REFERENCES answer_projections(projection_id, library_id) ON DELETE RESTRICT
);

CREATE INDEX answer_projections_library_answer_idx
ON answer_projections(library_id, answer_id, created_at, projection_id);

CREATE INDEX answer_projection_receipts_library_judge_idx
ON answer_projection_receipts(library_id, judge_kind, created_at, receipt_id);

CREATE TRIGGER answer_projections_no_update
BEFORE UPDATE ON answer_projections BEGIN SELECT RAISE(ABORT, 'answer_projections is append-only'); END;

CREATE TRIGGER answer_projections_no_delete
BEFORE DELETE ON answer_projections BEGIN SELECT RAISE(ABORT, 'answer_projections is append-only'); END;

CREATE TRIGGER answer_projection_receipts_no_update
BEFORE UPDATE ON answer_projection_receipts BEGIN SELECT RAISE(ABORT, 'answer_projection_receipts is append-only'); END;

CREATE TRIGGER answer_projection_receipts_no_delete
BEFORE DELETE ON answer_projection_receipts BEGIN SELECT RAISE(ABORT, 'answer_projection_receipts is append-only'); END;
