-- Dithyramba P6 typed, append-only meaning core.
CREATE TABLE meaning_processing_runs (
    processing_run_id TEXT PRIMARY KEY CHECK (substr(processing_run_id, 1, 4) = 'run_' AND length(processing_run_id) > 4),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (kind IN ('meaning_extract', 'meaning_import')),
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    scope_hash TEXT NOT NULL CHECK (length(scope_hash) = 64 AND scope_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    extraction_profile TEXT NOT NULL CHECK (extraction_profile IN ('index', 'light', 'deep')),
    code_version TEXT NOT NULL CHECK (length(trim(code_version)) > 0),
    generator_json TEXT NOT NULL,
    generator_hash TEXT NOT NULL CHECK (length(generator_hash) = 64 AND generator_hash NOT GLOB '*[^0-9a-f]*'),
    review_budget_json TEXT NOT NULL,
    review_budget_hash TEXT NOT NULL CHECK (length(review_budget_hash) = 64 AND review_budget_hash NOT GLOB '*[^0-9a-f]*'),
    input_hash TEXT NOT NULL CHECK (length(input_hash) = 64 AND input_hash NOT GLOB '*[^0-9a-f]*'),
    proposal_hash TEXT CHECK (proposal_hash IS NULL OR (length(proposal_hash) = 64 AND proposal_hash NOT GLOB '*[^0-9a-f]*')),
    import_key TEXT NOT NULL UNIQUE CHECK (length(import_key) = 64 AND import_key NOT GLOB '*[^0-9a-f]*'),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'partial', 'failed', 'refused')),
    candidate_count INTEGER NOT NULL CHECK (candidate_count >= 0 AND candidate_count <= 400),
    unresolved_before INTEGER NOT NULL CHECK (unresolved_before >= 0),
    backlog_warning INTEGER NOT NULL CHECK (backlog_warning IN (0, 1)),
    error_code TEXT,
    output_manifest_json TEXT NOT NULL,
    output_hash TEXT NOT NULL CHECK (length(output_hash) = 64 AND output_hash NOT GLOB '*[^0-9a-f]*'),
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*'),
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    CHECK ((status = 'succeeded' AND error_code IS NULL) OR (status <> 'succeeded' AND error_code IS NOT NULL)),
    CHECK ((status = 'succeeded' AND candidate_count >= 0) OR (status <> 'succeeded' AND candidate_count = 0))
);

CREATE TABLE meaning_coverage_reports (
    coverage_report_id TEXT PRIMARY KEY CHECK (substr(coverage_report_id, 1, 9) = 'coverage_' AND length(coverage_report_id) > 9),
    processing_run_id TEXT NOT NULL UNIQUE REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    extraction_profile TEXT NOT NULL CHECK (extraction_profile IN ('index', 'light', 'deep')),
    state TEXT NOT NULL CHECK (state IN ('complete', 'partial', 'failed', 'refused')),
    read_fragment_count INTEGER NOT NULL CHECK (read_fragment_count >= 0 AND read_fragment_count <= 2000),
    candidate_count INTEGER NOT NULL CHECK (candidate_count >= 0 AND candidate_count <= 400),
    omissions_json TEXT NOT NULL,
    report_hash TEXT NOT NULL UNIQUE CHECK (length(report_hash) = 64 AND report_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE meaning_read_receipts (
    read_receipt_id TEXT PRIMARY KEY CHECK (substr(read_receipt_id, 1, 5) = 'read_' AND length(read_receipt_id) > 5),
    processing_run_id TEXT NOT NULL UNIQUE REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    input_hash TEXT NOT NULL CHECK (length(input_hash) = 64 AND input_hash NOT GLOB '*[^0-9a-f]*'),
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE meaning_read_receipt_items (
    read_receipt_id TEXT NOT NULL REFERENCES meaning_read_receipts(read_receipt_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    collection_ids_json TEXT NOT NULL,
    read_order INTEGER NOT NULL CHECK (read_order >= 0),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (read_receipt_id, source_fragment_id),
    UNIQUE (read_receipt_id, read_order)
);

CREATE TABLE voices (
    voice_id TEXT PRIMARY KEY CHECK (substr(voice_id, 1, 6) = 'voice_' AND length(voice_id) > 6),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (kind IN ('author', 'narrator', 'character', 'institution', 'school', 'collective')),
    label TEXT NOT NULL CHECK (length(trim(label)) > 0),
    status TEXT NOT NULL CHECK (status = 'candidate'),
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    first_processing_run_id TEXT NOT NULL REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE (library_id, content_hash)
);

CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY CHECK (substr(entity_id, 1, 7) = 'entity_' AND length(entity_id) > 7),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (kind IN ('person', 'organization', 'work', 'place', 'event', 'object')),
    label TEXT NOT NULL CHECK (length(trim(label)) > 0),
    status TEXT NOT NULL CHECK (status = 'candidate'),
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    first_processing_run_id TEXT NOT NULL REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE (library_id, content_hash)
);

CREATE TABLE entity_aliases (
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE RESTRICT,
    alias TEXT NOT NULL CHECK (length(trim(alias)) > 0),
    PRIMARY KEY (entity_id, alias)
);

CREATE TABLE entity_mentions (
    entity_mention_id TEXT PRIMARY KEY CHECK (substr(entity_mention_id, 1, 15) = 'entity_mention_' AND length(entity_mention_id) > 15),
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE RESTRICT,
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    quote_start INTEGER NOT NULL CHECK (quote_start >= 0),
    quote_end INTEGER NOT NULL CHECK (quote_end > quote_start),
    quote_text TEXT NOT NULL CHECK (length(quote_text) > 0),
    quote_sha256 TEXT NOT NULL CHECK (length(quote_sha256) = 64 AND quote_sha256 NOT GLOB '*[^0-9a-f]*'),
    content_hash TEXT NOT NULL UNIQUE CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE concepts (
    concept_id TEXT PRIMARY KEY CHECK (substr(concept_id, 1, 8) = 'concept_' AND length(concept_id) > 8),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    label TEXT NOT NULL CHECK (length(trim(label)) > 0),
    status TEXT NOT NULL CHECK (status = 'candidate'),
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    first_processing_run_id TEXT NOT NULL REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE (library_id, content_hash)
);

CREATE TABLE concept_aliases (
    concept_id TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE RESTRICT,
    alias TEXT NOT NULL CHECK (length(trim(alias)) > 0),
    PRIMARY KEY (concept_id, alias)
);

CREATE TABLE concept_mentions (
    concept_mention_id TEXT PRIMARY KEY CHECK (substr(concept_mention_id, 1, 16) = 'concept_mention_' AND length(concept_mention_id) > 16),
    concept_id TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE RESTRICT,
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    quote_start INTEGER NOT NULL CHECK (quote_start >= 0),
    quote_end INTEGER NOT NULL CHECK (quote_end > quote_start),
    quote_text TEXT NOT NULL CHECK (length(quote_text) > 0),
    quote_sha256 TEXT NOT NULL CHECK (length(quote_sha256) = 64 AND quote_sha256 NOT GLOB '*[^0-9a-f]*'),
    content_hash TEXT NOT NULL UNIQUE CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE time_contexts (
    time_context_id TEXT PRIMARY KEY CHECK (substr(time_context_id, 1, 13) = 'time_context_' AND length(time_context_id) > 13),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    event_valid_start TEXT,
    event_valid_end TEXT,
    recorded_at TEXT,
    available_at TEXT,
    revealed_at TEXT,
    reviewed_at TEXT,
    status TEXT NOT NULL CHECK (status = 'candidate'),
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    first_processing_run_id TEXT NOT NULL REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    CHECK (event_valid_start IS NOT NULL OR event_valid_end IS NOT NULL OR recorded_at IS NOT NULL OR available_at IS NOT NULL OR revealed_at IS NOT NULL OR reviewed_at IS NOT NULL),
    UNIQUE (library_id, content_hash)
);

CREATE TABLE statements (
    statement_id TEXT PRIMARY KEY CHECK (substr(statement_id, 1, 10) = 'statement_' AND length(statement_id) > 10),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (kind IN ('source_claim', 'observation', 'interpretation', 'hypothesis', 'proposal', 'rule')),
    statement_text TEXT NOT NULL CHECK (length(trim(statement_text)) > 0),
    voice_id TEXT NOT NULL REFERENCES voices(voice_id) ON DELETE RESTRICT,
    time_context_id TEXT REFERENCES time_contexts(time_context_id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (status = 'candidate'),
    lifecycle TEXT NOT NULL CHECK (lifecycle = 'active'),
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    first_processing_run_id TEXT NOT NULL REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE (library_id, content_hash)
);

CREATE TABLE evidence_links (
    evidence_link_id TEXT PRIMARY KEY CHECK (substr(evidence_link_id, 1, 9) = 'evidence_' AND length(evidence_link_id) > 9),
    statement_id TEXT NOT NULL REFERENCES statements(statement_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    attributed_voice_id TEXT NOT NULL REFERENCES voices(voice_id) ON DELETE RESTRICT,
    polarity TEXT NOT NULL CHECK (polarity IN ('supports', 'contradicts', 'contextualizes')),
    alignment TEXT NOT NULL CHECK (alignment IN ('exact', 'partial', 'contextual')),
    limits_text TEXT,
    quote_start INTEGER NOT NULL CHECK (quote_start >= 0),
    quote_end INTEGER NOT NULL CHECK (quote_end > quote_start),
    quote_text TEXT NOT NULL CHECK (length(quote_text) > 0),
    quote_sha256 TEXT NOT NULL CHECK (length(quote_sha256) = 64 AND quote_sha256 NOT GLOB '*[^0-9a-f]*'),
    extraction_method TEXT NOT NULL CHECK (length(trim(extraction_method)) > 0),
    status TEXT NOT NULL CHECK (status = 'candidate'),
    content_hash TEXT NOT NULL UNIQUE CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    first_processing_run_id TEXT NOT NULL REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL
);

CREATE TABLE concept_meanings (
    concept_meaning_id TEXT PRIMARY KEY CHECK (substr(concept_meaning_id, 1, 16) = 'concept_meaning_' AND length(concept_meaning_id) > 16),
    concept_id TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE RESTRICT,
    voice_id TEXT NOT NULL REFERENCES voices(voice_id) ON DELETE RESTRICT,
    time_context_id TEXT REFERENCES time_contexts(time_context_id) ON DELETE RESTRICT,
    meaning_text TEXT NOT NULL CHECK (length(trim(meaning_text)) > 0),
    status TEXT NOT NULL CHECK (status = 'candidate'),
    content_hash TEXT NOT NULL UNIQUE CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    first_processing_run_id TEXT NOT NULL REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL
);

CREATE TABLE concept_meaning_statements (
    concept_meaning_id TEXT NOT NULL REFERENCES concept_meanings(concept_meaning_id) ON DELETE RESTRICT,
    statement_id TEXT NOT NULL REFERENCES statements(statement_id) ON DELETE RESTRICT,
    PRIMARY KEY (concept_meaning_id, statement_id)
);

CREATE TABLE meaning_review_sessions (
    review_session_id TEXT PRIMARY KEY CHECK (substr(review_session_id, 1, 15) = 'review_session_' AND length(review_session_id) > 15),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    review_budget_json TEXT NOT NULL,
    review_budget_hash TEXT NOT NULL CHECK (length(review_budget_hash) = 64 AND review_budget_hash NOT GLOB '*[^0-9a-f]*'),
    decision_limit INTEGER NOT NULL CHECK (decision_limit > 0),
    created_at TEXT NOT NULL
);

CREATE TABLE meaning_review_decisions (
    review_decision_id TEXT PRIMARY KEY CHECK (substr(review_decision_id, 1, 7) = 'review_' AND length(review_decision_id) > 7),
    review_session_id TEXT NOT NULL REFERENCES meaning_review_sessions(review_session_id) ON DELETE RESTRICT,
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    target_type TEXT NOT NULL CHECK (target_type IN ('statement', 'evidence_link', 'voice', 'entity', 'concept', 'concept_meaning', 'time_context')),
    target_id TEXT NOT NULL,
    target_hash TEXT NOT NULL CHECK (length(target_hash) = 64 AND target_hash NOT GLOB '*[^0-9a-f]*'),
    action TEXT NOT NULL CHECK (action IN ('accept', 'reject', 'revise', 'defer', 'supersede')),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    authority TEXT NOT NULL CHECK (length(trim(authority)) > 0),
    scope_json TEXT NOT NULL,
    replacement_target_id TEXT,
    supersedes_review_decision_id TEXT REFERENCES meaning_review_decisions(review_decision_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    CHECK ((action = 'revise' AND replacement_target_id IS NOT NULL) OR (action <> 'revise' AND replacement_target_id IS NULL)),
    CHECK ((action = 'supersede' AND supersedes_review_decision_id IS NOT NULL) OR (action <> 'supersede' AND supersedes_review_decision_id IS NULL))
);

CREATE TRIGGER entity_mentions_validate_insert
BEFORE INSERT ON entity_mentions
BEGIN
    SELECT RAISE(ABORT, 'Entity mention quote does not match exact SourceFragment')
    WHERE NOT EXISTS (
        SELECT 1 FROM source_fragments AS sf
        WHERE sf.source_fragment_id = NEW.source_fragment_id
          AND NEW.quote_text = substr(sf.text, NEW.quote_start + 1, NEW.quote_end - NEW.quote_start)
    );
END;

CREATE TRIGGER concept_mentions_validate_insert
BEFORE INSERT ON concept_mentions
BEGIN
    SELECT RAISE(ABORT, 'Concept mention quote does not match exact SourceFragment')
    WHERE NOT EXISTS (
        SELECT 1 FROM source_fragments AS sf
        WHERE sf.source_fragment_id = NEW.source_fragment_id
          AND NEW.quote_text = substr(sf.text, NEW.quote_start + 1, NEW.quote_end - NEW.quote_start)
    );
END;

CREATE TRIGGER evidence_links_validate_insert
BEFORE INSERT ON evidence_links
BEGIN
    SELECT RAISE(ABORT, 'EvidenceLink quote or Voice does not match its exact source')
    WHERE NOT EXISTS (
        SELECT 1
        FROM statements AS s
        JOIN source_fragments AS sf ON sf.source_fragment_id = NEW.source_fragment_id
        WHERE s.statement_id = NEW.statement_id
          AND s.voice_id = NEW.attributed_voice_id
          AND NEW.quote_text = substr(sf.text, NEW.quote_start + 1, NEW.quote_end - NEW.quote_start)
    );
END;

CREATE TRIGGER concept_meaning_statements_validate_insert
BEFORE INSERT ON concept_meaning_statements
BEGIN
    SELECT RAISE(ABORT, 'ConceptMeaning support must retain the same Voice')
    WHERE NOT EXISTS (
        SELECT 1
        FROM concept_meanings AS cm
        JOIN statements AS s ON s.statement_id = NEW.statement_id
        WHERE cm.concept_meaning_id = NEW.concept_meaning_id
          AND cm.voice_id = s.voice_id
    );
END;

CREATE INDEX statements_collection_idx ON statements(collection_id, created_at, statement_id);
CREATE INDEX evidence_links_fragment_idx ON evidence_links(source_fragment_id, statement_id);
CREATE INDEX entity_mentions_fragment_idx ON entity_mentions(source_fragment_id, entity_id);
CREATE INDEX concept_mentions_fragment_idx ON concept_mentions(source_fragment_id, concept_id);
CREATE INDEX concept_meanings_voice_idx ON concept_meanings(voice_id, concept_id);
CREATE INDEX meaning_review_decisions_target_idx ON meaning_review_decisions(target_type, target_id, created_at, review_decision_id);
CREATE INDEX meaning_read_receipt_items_fragment_idx ON meaning_read_receipt_items(source_fragment_id, read_receipt_id);

CREATE TRIGGER meaning_processing_runs_no_update BEFORE UPDATE ON meaning_processing_runs BEGIN SELECT RAISE(ABORT, 'meaning_processing_runs is append-only'); END;
CREATE TRIGGER meaning_processing_runs_no_delete BEFORE DELETE ON meaning_processing_runs BEGIN SELECT RAISE(ABORT, 'meaning_processing_runs is append-only'); END;
CREATE TRIGGER meaning_coverage_reports_no_update BEFORE UPDATE ON meaning_coverage_reports BEGIN SELECT RAISE(ABORT, 'meaning_coverage_reports is append-only'); END;
CREATE TRIGGER meaning_coverage_reports_no_delete BEFORE DELETE ON meaning_coverage_reports BEGIN SELECT RAISE(ABORT, 'meaning_coverage_reports is append-only'); END;
CREATE TRIGGER meaning_read_receipts_no_update BEFORE UPDATE ON meaning_read_receipts BEGIN SELECT RAISE(ABORT, 'meaning_read_receipts is append-only'); END;
CREATE TRIGGER meaning_read_receipts_no_delete BEFORE DELETE ON meaning_read_receipts BEGIN SELECT RAISE(ABORT, 'meaning_read_receipts is append-only'); END;
CREATE TRIGGER meaning_read_receipt_items_no_update BEFORE UPDATE ON meaning_read_receipt_items BEGIN SELECT RAISE(ABORT, 'meaning_read_receipt_items is append-only'); END;
CREATE TRIGGER meaning_read_receipt_items_no_delete BEFORE DELETE ON meaning_read_receipt_items BEGIN SELECT RAISE(ABORT, 'meaning_read_receipt_items is append-only'); END;
CREATE TRIGGER voices_no_update BEFORE UPDATE ON voices BEGIN SELECT RAISE(ABORT, 'voices is append-only'); END;
CREATE TRIGGER voices_no_delete BEFORE DELETE ON voices BEGIN SELECT RAISE(ABORT, 'voices is append-only'); END;
CREATE TRIGGER entities_no_update BEFORE UPDATE ON entities BEGIN SELECT RAISE(ABORT, 'entities is append-only'); END;
CREATE TRIGGER entities_no_delete BEFORE DELETE ON entities BEGIN SELECT RAISE(ABORT, 'entities is append-only'); END;
CREATE TRIGGER entity_aliases_no_update BEFORE UPDATE ON entity_aliases BEGIN SELECT RAISE(ABORT, 'entity_aliases is append-only'); END;
CREATE TRIGGER entity_aliases_no_delete BEFORE DELETE ON entity_aliases BEGIN SELECT RAISE(ABORT, 'entity_aliases is append-only'); END;
CREATE TRIGGER entity_mentions_no_update BEFORE UPDATE ON entity_mentions BEGIN SELECT RAISE(ABORT, 'entity_mentions is append-only'); END;
CREATE TRIGGER entity_mentions_no_delete BEFORE DELETE ON entity_mentions BEGIN SELECT RAISE(ABORT, 'entity_mentions is append-only'); END;
CREATE TRIGGER concepts_no_update BEFORE UPDATE ON concepts BEGIN SELECT RAISE(ABORT, 'concepts is append-only'); END;
CREATE TRIGGER concepts_no_delete BEFORE DELETE ON concepts BEGIN SELECT RAISE(ABORT, 'concepts is append-only'); END;
CREATE TRIGGER concept_aliases_no_update BEFORE UPDATE ON concept_aliases BEGIN SELECT RAISE(ABORT, 'concept_aliases is append-only'); END;
CREATE TRIGGER concept_aliases_no_delete BEFORE DELETE ON concept_aliases BEGIN SELECT RAISE(ABORT, 'concept_aliases is append-only'); END;
CREATE TRIGGER concept_mentions_no_update BEFORE UPDATE ON concept_mentions BEGIN SELECT RAISE(ABORT, 'concept_mentions is append-only'); END;
CREATE TRIGGER concept_mentions_no_delete BEFORE DELETE ON concept_mentions BEGIN SELECT RAISE(ABORT, 'concept_mentions is append-only'); END;
CREATE TRIGGER time_contexts_no_update BEFORE UPDATE ON time_contexts BEGIN SELECT RAISE(ABORT, 'time_contexts is append-only'); END;
CREATE TRIGGER time_contexts_no_delete BEFORE DELETE ON time_contexts BEGIN SELECT RAISE(ABORT, 'time_contexts is append-only'); END;
CREATE TRIGGER statements_no_update BEFORE UPDATE ON statements BEGIN SELECT RAISE(ABORT, 'statements is append-only'); END;
CREATE TRIGGER statements_no_delete BEFORE DELETE ON statements BEGIN SELECT RAISE(ABORT, 'statements is append-only'); END;
CREATE TRIGGER evidence_links_no_update BEFORE UPDATE ON evidence_links BEGIN SELECT RAISE(ABORT, 'evidence_links is append-only'); END;
CREATE TRIGGER evidence_links_no_delete BEFORE DELETE ON evidence_links BEGIN SELECT RAISE(ABORT, 'evidence_links is append-only'); END;
CREATE TRIGGER concept_meanings_no_update BEFORE UPDATE ON concept_meanings BEGIN SELECT RAISE(ABORT, 'concept_meanings is append-only'); END;
CREATE TRIGGER concept_meanings_no_delete BEFORE DELETE ON concept_meanings BEGIN SELECT RAISE(ABORT, 'concept_meanings is append-only'); END;
CREATE TRIGGER concept_meaning_statements_no_update BEFORE UPDATE ON concept_meaning_statements BEGIN SELECT RAISE(ABORT, 'concept_meaning_statements is append-only'); END;
CREATE TRIGGER concept_meaning_statements_no_delete BEFORE DELETE ON concept_meaning_statements BEGIN SELECT RAISE(ABORT, 'concept_meaning_statements is append-only'); END;
CREATE TRIGGER meaning_review_sessions_no_update BEFORE UPDATE ON meaning_review_sessions BEGIN SELECT RAISE(ABORT, 'meaning_review_sessions is append-only'); END;
CREATE TRIGGER meaning_review_sessions_no_delete BEFORE DELETE ON meaning_review_sessions BEGIN SELECT RAISE(ABORT, 'meaning_review_sessions is append-only'); END;
CREATE TRIGGER meaning_review_decisions_no_update BEFORE UPDATE ON meaning_review_decisions BEGIN SELECT RAISE(ABORT, 'meaning_review_decisions is append-only'); END;
CREATE TRIGGER meaning_review_decisions_no_delete BEFORE DELETE ON meaning_review_decisions BEGIN SELECT RAISE(ABORT, 'meaning_review_decisions is append-only'); END;
