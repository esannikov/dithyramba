-- Append-only public IdeaTrace candidates and deterministic closure receipts.
CREATE TABLE idea_traces (
    trace_id TEXT PRIMARY KEY CHECK (substr(trace_id, 1, 11) = 'idea_trace_' AND length(trace_id) = 43 AND substr(trace_id, 12) NOT GLOB '*[^0-9a-f]*'),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    schema_id TEXT NOT NULL CHECK (schema_id = 'dithyramba.idea_trace/1.0'),
    trace_hash TEXT NOT NULL UNIQUE CHECK (length(trace_hash) = 64 AND trace_hash NOT GLOB '*[^0-9a-f]*'),
    task_id TEXT NOT NULL,
    answer_id TEXT NOT NULL CHECK (substr(answer_id, 1, 16) = 'research_answer_' AND length(answer_id) = 48 AND substr(answer_id, 17) NOT GLOB '*[^0-9a-f]*'),
    answer_hash TEXT NOT NULL CHECK (length(answer_hash) = 64 AND answer_hash NOT GLOB '*[^0-9a-f]*'),
    packet_id TEXT NOT NULL CHECK (substr(packet_id, 1, 14) = 'memory_packet_' AND length(packet_id) = 46 AND substr(packet_id, 15) NOT GLOB '*[^0-9a-f]*'),
    packet_hash TEXT NOT NULL CHECK (length(packet_hash) = 64 AND packet_hash NOT GLOB '*[^0-9a-f]*'),
    case_set_id TEXT NOT NULL CHECK (substr(case_set_id, 1, 24) = 'claim_evidence_case_set_' AND length(case_set_id) = 56 AND substr(case_set_id, 25) NOT GLOB '*[^0-9a-f]*'),
    case_set_hash TEXT NOT NULL CHECK (length(case_set_hash) = 64 AND case_set_hash NOT GLOB '*[^0-9a-f]*'),
    receipt_id TEXT NOT NULL CHECK (substr(receipt_id, 1, 32) = 'claim_evidence_judgment_receipt_' AND length(receipt_id) = 64 AND substr(receipt_id, 33) NOT GLOB '*[^0-9a-f]*'),
    receipt_hash TEXT NOT NULL CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*'),
    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'model', 'deterministic')),
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (trace_id, library_id)
);

CREATE TABLE reasoning_closure_results (
    closure_id TEXT PRIMARY KEY CHECK (substr(closure_id, 1, 18) = 'reasoning_closure_' AND length(closure_id) = 50 AND substr(closure_id, 19) NOT GLOB '*[^0-9a-f]*'),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    trace_id TEXT NOT NULL,
    trace_hash TEXT NOT NULL CHECK (length(trace_hash) = 64 AND trace_hash NOT GLOB '*[^0-9a-f]*'),
    closure_hash TEXT NOT NULL UNIQUE CHECK (length(closure_hash) = 64 AND closure_hash NOT GLOB '*[^0-9a-f]*'),
    decision TEXT NOT NULL CHECK (decision IN ('passed', 'failed', 'review_required')),
    closure_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (trace_id),
    FOREIGN KEY (trace_id, library_id) REFERENCES idea_traces(trace_id, library_id) ON DELETE RESTRICT
);

CREATE INDEX idea_traces_library_task_idx
ON idea_traces(library_id, task_id, created_at, trace_id);

CREATE INDEX reasoning_closure_library_decision_idx
ON reasoning_closure_results(library_id, decision, created_at, closure_id);

CREATE TRIGGER idea_traces_no_update
BEFORE UPDATE ON idea_traces BEGIN SELECT RAISE(ABORT, 'idea_traces is append-only'); END;

CREATE TRIGGER idea_traces_no_delete
BEFORE DELETE ON idea_traces BEGIN SELECT RAISE(ABORT, 'idea_traces is append-only'); END;

CREATE TRIGGER reasoning_closure_results_no_update
BEFORE UPDATE ON reasoning_closure_results BEGIN SELECT RAISE(ABORT, 'reasoning_closure_results is append-only'); END;

CREATE TRIGGER reasoning_closure_results_no_delete
BEFORE DELETE ON reasoning_closure_results BEGIN SELECT RAISE(ABORT, 'reasoning_closure_results is append-only'); END;
