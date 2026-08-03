-- Dithyramba v1 clean schema baseline.
-- New Libraries create only the model-free source-to-evidence runtime.

CREATE TABLE access_policies (
    access_policy_id TEXT PRIMARY KEY CHECK (substr(access_policy_id, 1, 7) = 'policy_' AND length(access_policy_id) > 7),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    allowed_purposes_json TEXT NOT NULL,
    allow_export INTEGER NOT NULL CHECK (allow_export IN (0, 1)),
    allow_external_provider INTEGER NOT NULL DEFAULT 0 CHECK (allow_external_provider IN (0, 1)),
    policy_hash TEXT NOT NULL CHECK (length(policy_hash) = 64 AND policy_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (library_id, name),
    UNIQUE (library_id, policy_hash)
);

CREATE TABLE access_policy_collection_rules (
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    effect TEXT NOT NULL CHECK (effect IN ('allow', 'deny')),
    PRIMARY KEY (access_policy_id, collection_id)
);

CREATE TABLE access_policy_source_rules (
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    effect TEXT NOT NULL CHECK (effect IN ('allow', 'deny')),
    PRIMARY KEY (access_policy_id, source_id)
);

CREATE TABLE access_receipts (
    access_receipt_id TEXT PRIMARY KEY CHECK (substr(access_receipt_id, 1, 7) = 'access_' AND length(access_receipt_id) > 7),
    processing_run_id TEXT NOT NULL REFERENCES processing_runs(processing_run_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    policy_hash TEXT NOT NULL CHECK (length(policy_hash) = 64 AND policy_hash NOT GLOB '*[^0-9a-f]*'),
    snapshot_hash TEXT NOT NULL CHECK (length(snapshot_hash) = 64 AND snapshot_hash NOT GLOB '*[^0-9a-f]*'),
    exclusion_hash TEXT NOT NULL CHECK (length(exclusion_hash) = 64 AND exclusion_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    retrieval_corpus_hash TEXT NOT NULL CHECK (length(retrieval_corpus_hash) = 64 AND retrieval_corpus_hash NOT GLOB '*[^0-9a-f]*'),
    policy_omission_present INTEGER NOT NULL CHECK (policy_omission_present IN (0, 1)),
    receipt_json TEXT NOT NULL,
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*')
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

CREATE TABLE blobs (
    content_sha256 TEXT PRIMARY KEY CHECK (length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*'),
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    relative_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE collection_memberships (
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    state TEXT NOT NULL CHECK (state IN ('active', 'excluded', 'holdout')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (collection_id, source_id)
);

CREATE TABLE collection_root_identity_manifests (
    collection_root_id TEXT PRIMARY KEY REFERENCES collection_roots(collection_root_id) ON DELETE RESTRICT,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL CHECK (length(manifest_sha256) = 64 AND manifest_sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);

CREATE TABLE collection_roots (
    collection_root_id TEXT PRIMARY KEY CHECK (substr(collection_root_id, 1, 5) = 'root_' AND length(collection_root_id) > 5),
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    resolved_path TEXT NOT NULL,
    include_globs_json TEXT NOT NULL,
    exclude_globs_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (collection_id, resolved_path)
);

CREATE TABLE collections (
    collection_id TEXT PRIMARY KEY CHECK (substr(collection_id, 1, 11) = 'collection_' AND length(collection_id) > 11),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('corpus', 'case', 'branch', 'holdout')),
    created_at TEXT NOT NULL,
    UNIQUE (library_id, name)
);

CREATE TABLE concept_aliases (
    concept_id TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE RESTRICT,
    alias TEXT NOT NULL CHECK (length(trim(alias)) > 0),
    PRIMARY KEY (concept_id, alias)
);

CREATE TABLE concept_meaning_statements (
    concept_meaning_id TEXT NOT NULL REFERENCES concept_meanings(concept_meaning_id) ON DELETE RESTRICT,
    statement_id TEXT NOT NULL REFERENCES statements(statement_id) ON DELETE RESTRICT,
    PRIMARY KEY (concept_meaning_id, statement_id)
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

CREATE TABLE corpus_read_set_items (
    corpus_read_set_id TEXT NOT NULL REFERENCES corpus_read_sets(corpus_read_set_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    read_order INTEGER NOT NULL CHECK (read_order >= 0),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (corpus_read_set_id, source_fragment_id),
    UNIQUE (corpus_read_set_id, read_order)
);

CREATE TABLE corpus_read_sets (
    corpus_read_set_id TEXT PRIMARY KEY CHECK (substr(corpus_read_set_id, 1, 16) = 'corpus_read_set_' AND length(corpus_read_set_id) > 16),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    retrieval_corpus_hash TEXT NOT NULL CHECK (length(retrieval_corpus_hash) = 64 AND retrieval_corpus_hash NOT GLOB '*[^0-9a-f]*'),
    item_count INTEGER NOT NULL CHECK (item_count >= 0),
    set_hash TEXT NOT NULL UNIQUE CHECK (length(set_hash) = 64 AND set_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE corpus_snapshots (
    corpus_snapshot_id TEXT PRIMARY KEY CHECK (substr(corpus_snapshot_id, 1, 9) = 'snapshot_' AND length(corpus_snapshot_id) > 9),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    scope_json TEXT NOT NULL,
    manifest_hash TEXT NOT NULL UNIQUE CHECK (length(manifest_hash) = 64 AND manifest_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);

CREATE TABLE coverage_reports (
    coverage_report_id TEXT PRIMARY KEY CHECK (substr(coverage_report_id, 1, 9) = 'coverage_' AND length(coverage_report_id) > 9),
    processing_run_id TEXT NOT NULL REFERENCES processing_runs(processing_run_id) ON DELETE RESTRICT,
    stage TEXT NOT NULL CHECK (stage IN ('ingest', 'recall')),
    processed_count INTEGER NOT NULL CHECK (processed_count >= 0),
    skipped_count INTEGER NOT NULL CHECK (skipped_count >= 0),
    failed_count INTEGER NOT NULL CHECK (failed_count >= 0),
    policy_omission_present INTEGER NOT NULL CHECK (policy_omission_present IN (0, 1)),
    report_json TEXT NOT NULL,
    report_hash TEXT NOT NULL UNIQUE CHECK (length(report_hash) = 64 AND report_hash NOT GLOB '*[^0-9a-f]*')
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

CREATE TABLE event_outbox (
    event_id TEXT PRIMARY KEY CHECK (substr(event_id, 1, 6) = 'event_' AND length(event_id) > 6),
    event_type TEXT NOT NULL,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL CHECK (length(payload_hash) = 64 AND payload_hash NOT GLOB '*[^0-9a-f]*'),
    occurred_at TEXT NOT NULL,
    delivered_at TEXT
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

CREATE TABLE evidence_packets (
    evidence_packet_id TEXT PRIMARY KEY CHECK (substr(evidence_packet_id, 1, 7) = 'packet_' AND length(evidence_packet_id) > 7),
    query_request_id TEXT NOT NULL REFERENCES query_requests(query_request_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    result_status TEXT NOT NULL CHECK (result_status IN ('evidence_found', 'no_evidence')),
    packet_json TEXT NOT NULL,
    packet_hash TEXT NOT NULL UNIQUE CHECK (length(packet_hash) = 64 AND packet_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);

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

CREATE TABLE libraries (
    library_id TEXT PRIMARY KEY CHECK (substr(library_id, 1, 8) = 'library_' AND length(library_id) > 8),
    name TEXT NOT NULL,
    logical_identity_hash TEXT NOT NULL UNIQUE CHECK (length(logical_identity_hash) = 64 AND logical_identity_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
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

CREATE TABLE meaning_read_receipts (
    read_receipt_id TEXT PRIMARY KEY CHECK (substr(read_receipt_id, 1, 5) = 'read_' AND length(read_receipt_id) > 5),
    processing_run_id TEXT NOT NULL UNIQUE REFERENCES meaning_processing_runs(processing_run_id) ON DELETE RESTRICT,
    input_hash TEXT NOT NULL CHECK (length(input_hash) = 64 AND input_hash NOT GLOB '*[^0-9a-f]*'),
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*')
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

CREATE TABLE meaning_review_sessions (
    review_session_id TEXT PRIMARY KEY CHECK (substr(review_session_id, 1, 15) = 'review_session_' AND length(review_session_id) > 15),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    review_budget_json TEXT NOT NULL,
    review_budget_hash TEXT NOT NULL CHECK (length(review_budget_hash) = 64 AND review_budget_hash NOT GLOB '*[^0-9a-f]*'),
    decision_limit INTEGER NOT NULL CHECK (decision_limit > 0),
    created_at TEXT NOT NULL
);

CREATE TABLE omissions (
    omission_id TEXT PRIMARY KEY CHECK (substr(omission_id, 1, 9) = 'omission_' AND length(omission_id) > 9),
    coverage_report_id TEXT NOT NULL REFERENCES coverage_reports(coverage_report_id) ON DELETE RESTRICT,
    category TEXT NOT NULL CHECK (category IN ('explicit_exclusion', 'budget', 'unsupported', 'parser_failure', 'policy')),
    disclosure TEXT NOT NULL CHECK (disclosure IN ('counted', 'redacted')),
    count INTEGER CHECK (count IS NULL OR count >= 0),
    reason_code TEXT NOT NULL,
    CHECK ((category = 'policy' AND disclosure = 'redacted' AND count IS NULL) OR category <> 'policy')
);

CREATE TABLE packet_items (
    evidence_packet_id TEXT NOT NULL REFERENCES evidence_packets(evidence_packet_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    role TEXT NOT NULL CHECK (role = 'evidence'),
    rank INTEGER NOT NULL CHECK (rank > 0),
    score_text TEXT NOT NULL,
    PRIMARY KEY (evidence_packet_id, source_fragment_id),
    UNIQUE (evidence_packet_id, rank)
);

CREATE TABLE processing_runs (
    processing_run_id TEXT PRIMARY KEY CHECK (substr(processing_run_id, 1, 4) = 'run_' AND length(processing_run_id) > 4),
    kind TEXT NOT NULL CHECK (kind IN ('ingest', 'recall', 'replay', 'backup', 'restore')),
    query_request_id TEXT REFERENCES query_requests(query_request_id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    code_version TEXT NOT NULL,
    profile_version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error_code TEXT,
    output_hash TEXT CHECK (output_hash IS NULL OR (length(output_hash) = 64 AND output_hash NOT GLOB '*[^0-9a-f]*'))
);

CREATE TABLE query_requests (
    query_request_id TEXT PRIMARY KEY CHECK (substr(query_request_id, 1, 6) = 'query_' AND length(query_request_id) > 6),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    request_json TEXT NOT NULL,
    request_hash TEXT NOT NULL UNIQUE CHECK (length(request_hash) = 64 AND request_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);

CREATE TABLE read_receipt_corpus_sets (
    read_receipt_id TEXT PRIMARY KEY REFERENCES read_receipts(read_receipt_id) ON DELETE RESTRICT,
    corpus_read_set_id TEXT NOT NULL REFERENCES corpus_read_sets(corpus_read_set_id) ON DELETE RESTRICT
);

CREATE TABLE read_receipt_items (
    read_receipt_id TEXT NOT NULL REFERENCES read_receipts(read_receipt_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    read_order INTEGER NOT NULL CHECK (read_order >= 0),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (read_receipt_id, source_fragment_id),
    UNIQUE (read_receipt_id, read_order)
);

CREATE TABLE read_receipts (
    read_receipt_id TEXT PRIMARY KEY CHECK (substr(read_receipt_id, 1, 5) = 'read_' AND length(read_receipt_id) > 5),
    processing_run_id TEXT NOT NULL REFERENCES processing_runs(processing_run_id) ON DELETE RESTRICT,
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*')
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

CREATE TABLE recall_run_artifacts (
    processing_run_id TEXT PRIMARY KEY REFERENCES processing_runs(processing_run_id) ON DELETE RESTRICT,
    coverage_report_id TEXT NOT NULL REFERENCES coverage_reports(coverage_report_id) ON DELETE RESTRICT,
    read_receipt_id TEXT NOT NULL REFERENCES read_receipts(read_receipt_id) ON DELETE RESTRICT,
    access_receipt_id TEXT NOT NULL REFERENCES access_receipts(access_receipt_id) ON DELETE RESTRICT,
    retrieval_receipt_id TEXT NOT NULL REFERENCES retrieval_receipts(retrieval_receipt_id) ON DELETE RESTRICT,
    evidence_packet_id TEXT NOT NULL REFERENCES evidence_packets(evidence_packet_id) ON DELETE RESTRICT
);

CREATE TABLE relation_evidence_links (
    relation_id TEXT NOT NULL REFERENCES relations(relation_id) ON DELETE RESTRICT,
    evidence_link_id TEXT NOT NULL REFERENCES evidence_links(evidence_link_id) ON DELETE RESTRICT,
    evidence_order INTEGER NOT NULL CHECK (evidence_order >= 0 AND evidence_order < 16),
    PRIMARY KEY (relation_id, evidence_link_id),
    UNIQUE (relation_id, evidence_order)
);

CREATE TABLE relation_import_run_items (
    relation_import_run_id TEXT NOT NULL REFERENCES relation_import_runs(relation_import_run_id) ON DELETE RESTRICT,
    relation_id TEXT NOT NULL REFERENCES relations(relation_id) ON DELETE RESTRICT,
    output_order INTEGER NOT NULL CHECK (output_order >= 0 AND output_order < 400),
    PRIMARY KEY (relation_import_run_id, relation_id),
    UNIQUE (relation_import_run_id, output_order)
);

CREATE TABLE relation_import_runs (
    relation_import_run_id TEXT PRIMARY KEY CHECK (substr(relation_import_run_id, 1, 13) = 'relation_run_' AND length(relation_import_run_id) > 13),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    policy_hash TEXT NOT NULL CHECK (length(policy_hash) = 64 AND policy_hash NOT GLOB '*[^0-9a-f]*'),
    scope_hash TEXT NOT NULL CHECK (length(scope_hash) = 64 AND scope_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    code_version TEXT NOT NULL CHECK (length(trim(code_version)) BETWEEN 1 AND 128),
    profile_version TEXT NOT NULL CHECK (length(trim(profile_version)) BETWEEN 1 AND 128),
    input_hash TEXT NOT NULL CHECK (length(input_hash) = 64 AND input_hash NOT GLOB '*[^0-9a-f]*'),
    relation_count INTEGER NOT NULL CHECK (relation_count BETWEEN 1 AND 400),
    output_hash TEXT NOT NULL CHECK (length(output_hash) = 64 AND output_hash NOT GLOB '*[^0-9a-f]*'),
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (library_id, input_hash, policy_hash, scope_hash, permitted_set_hash, profile_version)
);

CREATE TABLE relation_path_collections (
    relation_path_receipt_id TEXT NOT NULL REFERENCES relation_path_receipts(relation_path_receipt_id) ON DELETE RESTRICT,
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    PRIMARY KEY (relation_path_receipt_id, collection_id)
);

CREATE TABLE relation_path_exclusions (
    relation_path_receipt_id TEXT NOT NULL REFERENCES relation_path_receipts(relation_path_receipt_id) ON DELETE RESTRICT,
    target_type TEXT NOT NULL CHECK (target_type IN ('source', 'source_family', 'source_fragment')),
    target_id TEXT NOT NULL,
    PRIMARY KEY (relation_path_receipt_id, target_type, target_id)
);

CREATE TABLE relation_path_receipts (
    relation_path_receipt_id TEXT PRIMARY KEY CHECK (substr(relation_path_receipt_id, 1, 14) = 'relation_path_' AND length(relation_path_receipt_id) > 14),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    snapshot_hash TEXT NOT NULL CHECK (length(snapshot_hash) = 64 AND snapshot_hash NOT GLOB '*[^0-9a-f]*'),
    policy_hash TEXT NOT NULL CHECK (length(policy_hash) = 64 AND policy_hash NOT GLOB '*[^0-9a-f]*'),
    scope_hash TEXT NOT NULL CHECK (length(scope_hash) = 64 AND scope_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    request_hash TEXT NOT NULL CHECK (length(request_hash) = 64 AND request_hash NOT GLOB '*[^0-9a-f]*'),
    purpose TEXT NOT NULL CHECK (length(purpose) BETWEEN 1 AND 64 AND purpose = lower(purpose) AND purpose NOT GLOB '*[^a-z0-9_-]*'),
    direction TEXT NOT NULL CHECK (direction IN ('outgoing', 'incoming', 'both')),
    max_hops INTEGER NOT NULL CHECK (max_hops BETWEEN 1 AND 3),
    max_paths INTEGER NOT NULL CHECK (max_paths BETWEEN 1 AND 100),
    seed_count INTEGER NOT NULL CHECK (seed_count BETWEEN 1 AND 32),
    path_count INTEGER NOT NULL CHECK (path_count BETWEEN 0 AND 100),
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);

CREATE TABLE relation_path_relation_types (
    relation_path_receipt_id TEXT NOT NULL REFERENCES relation_path_receipts(relation_path_receipt_id) ON DELETE RESTRICT,
    relation_type TEXT NOT NULL CHECK (relation_type IN (
        'supports', 'contradicts', 'contextualizes', 'limits',
        'defines', 'narrows', 'broadens', 'analogous_to', 'distinguishes_from',
        'precedes', 'overlaps', 'supersedes',
        'enables', 'inhibits', 'possibly_causes',
        'derived_from', 'quotes', 'depends_on',
        'transfers_method_to', 'opens_gap', 'suggests_scene'
    )),
    PRIMARY KEY (relation_path_receipt_id, relation_type)
);

CREATE TABLE relation_path_seeds (
    relation_path_receipt_id TEXT NOT NULL REFERENCES relation_path_receipts(relation_path_receipt_id) ON DELETE RESTRICT,
    seed_order INTEGER NOT NULL CHECK (seed_order >= 0 AND seed_order < 32),
    node_type TEXT NOT NULL CHECK (node_type IN ('statement', 'concept', 'concept_meaning', 'entity', 'voice', 'time_context', 'structure_unit')),
    node_id TEXT NOT NULL,
    PRIMARY KEY (relation_path_receipt_id, seed_order),
    UNIQUE (relation_path_receipt_id, node_type, node_id)
);

CREATE TABLE relation_path_steps (
    relation_path_receipt_id TEXT NOT NULL,
    path_order INTEGER NOT NULL,
    hop_order INTEGER NOT NULL CHECK (hop_order BETWEEN 0 AND 2),
    relation_id TEXT NOT NULL REFERENCES relations(relation_id) ON DELETE RESTRICT,
    from_type TEXT NOT NULL CHECK (from_type IN ('statement', 'concept', 'concept_meaning', 'entity', 'voice', 'time_context', 'structure_unit')),
    from_id TEXT NOT NULL,
    to_type TEXT NOT NULL CHECK (to_type IN ('statement', 'concept', 'concept_meaning', 'entity', 'voice', 'time_context', 'structure_unit')),
    to_id TEXT NOT NULL,
    PRIMARY KEY (relation_path_receipt_id, path_order, hop_order),
    UNIQUE (relation_path_receipt_id, path_order, relation_id),
    FOREIGN KEY (relation_path_receipt_id, path_order) REFERENCES relation_paths(relation_path_receipt_id, path_order) ON DELETE RESTRICT
);

CREATE TABLE relation_paths (
    relation_path_receipt_id TEXT NOT NULL REFERENCES relation_path_receipts(relation_path_receipt_id) ON DELETE RESTRICT,
    path_order INTEGER NOT NULL CHECK (path_order >= 0 AND path_order < 100),
    seed_order INTEGER NOT NULL CHECK (seed_order >= 0 AND seed_order < 32),
    terminal_type TEXT NOT NULL CHECK (terminal_type IN ('statement', 'concept', 'concept_meaning', 'entity', 'voice', 'time_context', 'structure_unit')),
    terminal_id TEXT NOT NULL,
    hop_count INTEGER NOT NULL CHECK (hop_count BETWEEN 1 AND 3),
    path_hash TEXT NOT NULL CHECK (length(path_hash) = 64 AND path_hash NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (relation_path_receipt_id, path_order),
    UNIQUE (relation_path_receipt_id, path_hash),
    FOREIGN KEY (relation_path_receipt_id, seed_order) REFERENCES relation_path_seeds(relation_path_receipt_id, seed_order) ON DELETE RESTRICT
);

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

CREATE TABLE relations (
    relation_id TEXT PRIMARY KEY CHECK (substr(relation_id, 1, 9) = 'relation_' AND length(relation_id) > 9),
    relation_import_run_id TEXT NOT NULL REFERENCES relation_import_runs(relation_import_run_id) ON DELETE RESTRICT,
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    relation_type TEXT NOT NULL CHECK (relation_type IN (
        'supports', 'contradicts', 'contextualizes', 'limits',
        'defines', 'narrows', 'broadens', 'analogous_to', 'distinguishes_from',
        'precedes', 'overlaps', 'supersedes',
        'enables', 'inhibits', 'possibly_causes',
        'derived_from', 'quotes', 'depends_on',
        'transfers_method_to', 'opens_gap', 'suggests_scene'
    )),
    subject_type TEXT NOT NULL CHECK (subject_type IN ('statement', 'concept', 'concept_meaning', 'entity', 'voice', 'time_context', 'structure_unit')),
    subject_id TEXT NOT NULL,
    object_type TEXT NOT NULL CHECK (object_type IN ('statement', 'concept', 'concept_meaning', 'entity', 'voice', 'time_context', 'structure_unit')),
    object_id TEXT NOT NULL,
    reason_text TEXT NOT NULL CHECK (length(trim(reason_text)) BETWEEN 1 AND 4000 AND instr(reason_text, char(0)) = 0),
    extraction_method TEXT NOT NULL CHECK (length(trim(extraction_method)) BETWEEN 1 AND 128),
    valid_start TEXT,
    valid_end TEXT,
    evidence_count INTEGER NOT NULL CHECK (evidence_count BETWEEN 1 AND 16),
    status TEXT NOT NULL CHECK (status = 'candidate'),
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (library_id, content_hash),
    CHECK (subject_type <> object_type OR subject_id <> object_id),
    CHECK (valid_start IS NULL OR length(trim(valid_start)) > 0),
    CHECK (valid_end IS NULL OR length(trim(valid_end)) > 0)
);

CREATE TABLE research_session_events (
    event_id TEXT PRIMARY KEY CHECK (
        substr(event_id, 1, 14) = 'session_event_'
        AND length(event_id) = 46
        AND substr(event_id, 15) NOT GLOB '*[^0-9a-f]*'
    ),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    session_id TEXT NOT NULL,
    schema_id TEXT NOT NULL CHECK (schema_id = 'dithyramba.session_event/1.0'),
    event_hash TEXT NOT NULL UNIQUE CHECK (
        length(event_hash) = 64 AND event_hash NOT GLOB '*[^0-9a-f]*'
    ),
    sequence INTEGER NOT NULL CHECK (sequence BETWEEN 1 AND 1000000),
    previous_event_hash TEXT NOT NULL CHECK (
        length(previous_event_hash) = 64
        AND previous_event_hash NOT GLOB '*[^0-9a-f]*'
    ),
    kind TEXT NOT NULL CHECK (
        kind IN (
            'question_asked',
            'evidence_attached',
            'answer_drafted',
            'candidate_linked',
            'gap_recorded',
            'path_rejected',
            'decision_recorded',
            'session_closed'
        )
    ),
    actor_kind TEXT NOT NULL CHECK (actor_kind IN ('human', 'model', 'deterministic')),
    event_json TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    UNIQUE (session_id, sequence),
    FOREIGN KEY (session_id, library_id)
        REFERENCES research_sessions(session_id, library_id) ON DELETE RESTRICT
);

CREATE TABLE research_sessions (
    session_id TEXT PRIMARY KEY CHECK (
        substr(session_id, 1, 17) = 'research_session_'
        AND length(session_id) = 49
        AND substr(session_id, 18) NOT GLOB '*[^0-9a-f]*'
    ),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    schema_id TEXT NOT NULL CHECK (schema_id = 'dithyramba.research_session/1.0'),
    session_hash TEXT NOT NULL UNIQUE CHECK (
        length(session_hash) = 64 AND session_hash NOT GLOB '*[^0-9a-f]*'
    ),
    brief_id TEXT NOT NULL CHECK (
        substr(brief_id, 1, 14) = 'session_brief_'
        AND length(brief_id) = 46
        AND substr(brief_id, 15) NOT GLOB '*[^0-9a-f]*'
    ),
    brief_hash TEXT NOT NULL CHECK (
        length(brief_hash) = 64 AND brief_hash NOT GLOB '*[^0-9a-f]*'
    ),
    corpus_snapshot_id TEXT NOT NULL
        REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    snapshot_hash TEXT NOT NULL CHECK (
        length(snapshot_hash) = 64 AND snapshot_hash NOT GLOB '*[^0-9a-f]*'
    ),
    access_policy_id TEXT NOT NULL
        REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    scope_hash TEXT NOT NULL CHECK (
        length(scope_hash) = 64 AND scope_hash NOT GLOB '*[^0-9a-f]*'
    ),
    created_by_kind TEXT NOT NULL CHECK (created_by_kind IN ('human', 'model')),
    session_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, library_id)
);

CREATE TABLE retrieval_receipts (
    retrieval_receipt_id TEXT PRIMARY KEY CHECK (substr(retrieval_receipt_id, 1, 10) = 'retrieval_' AND length(retrieval_receipt_id) > 10),
    processing_run_id TEXT NOT NULL REFERENCES processing_runs(processing_run_id) ON DELETE RESTRICT,
    profile TEXT NOT NULL CHECK (profile = 'fts_v1'),
    profile_version TEXT NOT NULL,
    candidate_count INTEGER NOT NULL CHECK (candidate_count >= 0),
    selected_count INTEGER NOT NULL CHECK (selected_count >= 0),
    trace_json TEXT NOT NULL,
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE review_decisions (
    review_decision_id TEXT PRIMARY KEY CHECK (substr(review_decision_id, 1, 7) = 'review_' AND length(review_decision_id) > 7),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    target_type TEXT NOT NULL CHECK (target_type IN ('evidence_packet', 'packet_item', 'source_fragment')),
    target_id TEXT NOT NULL,
    target_hash TEXT NOT NULL CHECK (length(target_hash) = 64 AND target_hash NOT GLOB '*[^0-9a-f]*'),
    action TEXT NOT NULL CHECK (action IN ('accept', 'reject', 'revise', 'defer', 'supersede')),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    authority TEXT NOT NULL CHECK (length(trim(authority)) > 0),
    scope_json TEXT NOT NULL,
    supersedes_review_decision_id TEXT REFERENCES review_decisions(review_decision_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    CHECK ((action = 'supersede' AND supersedes_review_decision_id IS NOT NULL) OR (action <> 'supersede' AND supersedes_review_decision_id IS NULL))
);

CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY CHECK (version > 0),
    name TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    applied_at TEXT NOT NULL
);

CREATE TABLE snapshot_members (
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    membership_state TEXT NOT NULL CHECK (membership_state IN ('active', 'excluded', 'holdout')),
    PRIMARY KEY (corpus_snapshot_id, collection_id, source_version_id)
);

CREATE TABLE source_families (
    source_family_id TEXT PRIMARY KEY CHECK (substr(source_family_id, 1, 7) = 'family_' AND length(source_family_id) > 7),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE source_family_identities (
    source_family_id TEXT PRIMARY KEY REFERENCES source_families(source_family_id) ON DELETE RESTRICT,
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    logical_source_uri TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (library_id, logical_source_uri)
);

CREATE TABLE source_family_members (
    source_family_id TEXT NOT NULL REFERENCES source_families(source_family_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    role TEXT NOT NULL CHECK (role IN ('root', 'derivative', 'duplicate')),
    root_source_id TEXT REFERENCES sources(source_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_family_id, source_id),
    UNIQUE (source_id),
    CHECK ((role = 'root' AND root_source_id IS NULL) OR (role IN ('derivative', 'duplicate') AND root_source_id IS NOT NULL))
);

CREATE TABLE source_fragments (
    source_fragment_id TEXT PRIMARY KEY CHECK (substr(source_fragment_id, 1, 9) = 'fragment_' AND length(source_fragment_id) > 9),
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    fragment_kind TEXT NOT NULL CHECK (fragment_kind IN ('heading', 'paragraph', 'page_text')),
    text TEXT NOT NULL CHECK (length(text) > 0),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    source_address_json TEXT NOT NULL,
    address_hash TEXT NOT NULL CHECK (length(address_hash) = 64 AND address_hash NOT GLOB '*[^0-9a-f]*'),
    UNIQUE (source_version_id, ordinal),
    UNIQUE (source_version_id, address_hash)
);

CREATE TABLE source_heads (
    source_id TEXT PRIMARY KEY REFERENCES sources(source_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL UNIQUE REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    observed_at TEXT NOT NULL,
    source_modified_at TEXT
);

CREATE TABLE source_identity_bindings (
    source_identity_binding_id TEXT PRIMARY KEY CHECK (substr(source_identity_binding_id, 1, 16) = 'source_identity_' AND length(source_identity_binding_id) > 16),
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    source_family_id TEXT NOT NULL REFERENCES source_families(source_family_id) ON DELETE RESTRICT,
    connector TEXT NOT NULL,
    connector_revision TEXT NOT NULL,
    representation TEXT NOT NULL,
    part_number INTEGER NOT NULL CHECK (part_number >= 0),
    part_metadata_hash TEXT NOT NULL CHECK (length(part_metadata_hash) = 64 AND part_metadata_hash NOT GLOB '*[^0-9a-f]*'),
    content_sha256 TEXT NOT NULL REFERENCES blobs(content_sha256) ON DELETE RESTRICT CHECK (length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*'),
    manifest_sha256 TEXT NOT NULL CHECK (length(manifest_sha256) = 64 AND manifest_sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (source_version_id)
);

CREATE TABLE source_versions (
    source_version_id TEXT PRIMARY KEY CHECK (substr(source_version_id, 1, 15) = 'source_version_' AND length(source_version_id) > 15),
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    content_sha256 TEXT NOT NULL REFERENCES blobs(content_sha256) ON DELETE RESTRICT,
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    observed_at TEXT NOT NULL,
    source_modified_at TEXT,
    parser_profile TEXT NOT NULL,
    parse_status TEXT NOT NULL CHECK (parse_status IN ('processed', 'skipped', 'failed')),
    failure_code TEXT,
    UNIQUE (source_id, version_number),
    UNIQUE (source_id, content_sha256, parser_profile)
);

CREATE TABLE sources (
    source_id TEXT PRIMARY KEY CHECK (substr(source_id, 1, 7) = 'source_' AND length(source_id) > 7),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    canonical_uri TEXT NOT NULL,
    media_type TEXT NOT NULL CHECK (media_type IN ('text/markdown', 'text/plain', 'application/pdf')),
    title TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (library_id, canonical_uri)
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

CREATE TABLE structure_unit_generations (
    structure_unit_generation_id TEXT PRIMARY KEY CHECK (substr(structure_unit_generation_id, 1, 21) = 'structure_generation_' AND length(structure_unit_generation_id) > 21),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    policy_hash TEXT NOT NULL CHECK (length(policy_hash) = 64 AND policy_hash NOT GLOB '*[^0-9a-f]*'),
    scope_hash TEXT NOT NULL CHECK (length(scope_hash) = 64 AND scope_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    profile_id TEXT NOT NULL CHECK (length(profile_id) BETWEEN 1 AND 64 AND profile_id = lower(profile_id) AND profile_id NOT GLOB '*[^a-z0-9_-]*'),
    profile_version TEXT NOT NULL CHECK (length(trim(profile_version)) BETWEEN 1 AND 128),
    unit_count INTEGER NOT NULL CHECK (unit_count BETWEEN 1 AND 5000),
    generation_hash TEXT NOT NULL UNIQUE CHECK (length(generation_hash) = 64 AND generation_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (library_id, corpus_snapshot_id, policy_hash, scope_hash, permitted_set_hash, profile_id, profile_version, generation_hash)
);

CREATE TABLE structure_unit_members (
    structure_unit_id TEXT NOT NULL REFERENCES structure_units(structure_unit_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    member_order INTEGER NOT NULL CHECK (member_order >= 0 AND member_order < 128),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (structure_unit_id, source_fragment_id),
    UNIQUE (structure_unit_id, member_order),
    UNIQUE (structure_unit_id, ordinal)
);

CREATE TABLE structure_unit_read_receipt_items (
    structure_read_receipt_id TEXT NOT NULL REFERENCES structure_unit_read_receipts(structure_read_receipt_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    member_order INTEGER NOT NULL CHECK (member_order >= 0 AND member_order < 128),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    rendered_start INTEGER NOT NULL CHECK (rendered_start >= 0),
    rendered_end INTEGER NOT NULL CHECK (rendered_end > rendered_start),
    PRIMARY KEY (structure_read_receipt_id, source_fragment_id),
    UNIQUE (structure_read_receipt_id, member_order),
    CHECK (rendered_end <= 10000000)
);

CREATE TABLE structure_unit_read_receipts (
    structure_read_receipt_id TEXT PRIMARY KEY CHECK (substr(structure_read_receipt_id, 1, 15) = 'structure_read_' AND length(structure_read_receipt_id) > 15),
    structure_unit_id TEXT NOT NULL REFERENCES structure_units(structure_unit_id) ON DELETE RESTRICT,
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    policy_hash TEXT NOT NULL CHECK (length(policy_hash) = 64 AND policy_hash NOT GLOB '*[^0-9a-f]*'),
    scope_hash TEXT NOT NULL CHECK (length(scope_hash) = 64 AND scope_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    delimiter TEXT NOT NULL CHECK (delimiter = char(10) || char(10)),
    member_count INTEGER NOT NULL CHECK (member_count BETWEEN 1 AND 128),
    rendered_text_sha256 TEXT NOT NULL CHECK (length(rendered_text_sha256) = 64 AND rendered_text_sha256 NOT GLOB '*[^0-9a-f]*'),
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);

CREATE TABLE structure_units (
    structure_unit_id TEXT PRIMARY KEY CHECK (substr(structure_unit_id, 1, 15) = 'structure_unit_' AND length(structure_unit_id) > 15),
    structure_unit_generation_id TEXT NOT NULL REFERENCES structure_unit_generations(structure_unit_generation_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (kind IN ('document', 'part', 'chapter', 'section', 'scene', 'block', 'paragraph', 'other')),
    label TEXT CHECK (label IS NULL OR (length(trim(label)) BETWEEN 1 AND 500 AND instr(label, char(0)) = 0)),
    member_count INTEGER NOT NULL CHECK (member_count BETWEEN 1 AND 128),
    first_ordinal INTEGER NOT NULL CHECK (first_ordinal >= 0),
    last_ordinal INTEGER NOT NULL CHECK (last_ordinal >= first_ordinal),
    boundary_hash TEXT NOT NULL CHECK (length(boundary_hash) = 64 AND boundary_hash NOT GLOB '*[^0-9a-f]*'),
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (structure_unit_generation_id, content_hash),
    CHECK (last_ordinal - first_ordinal + 1 = member_count)
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

CREATE INDEX answer_projection_receipts_library_judge_idx
ON answer_projection_receipts(library_id, judge_kind, created_at, receipt_id);

CREATE INDEX answer_projections_library_answer_idx
ON answer_projections(library_id, answer_id, created_at, projection_id);

CREATE INDEX collection_memberships_source_idx ON collection_memberships(source_id, collection_id);

CREATE INDEX concept_meanings_voice_idx ON concept_meanings(voice_id, concept_id);

CREATE INDEX concept_mentions_fragment_idx ON concept_mentions(source_fragment_id, concept_id);

CREATE INDEX corpus_read_set_items_fragment_idx
ON corpus_read_set_items(source_fragment_id, corpus_read_set_id);

CREATE INDEX corpus_read_sets_library_corpus_idx
ON corpus_read_sets(library_id, retrieval_corpus_hash, corpus_read_set_id);

CREATE INDEX entity_mentions_fragment_idx ON entity_mentions(source_fragment_id, entity_id);

CREATE INDEX event_outbox_pending_idx ON event_outbox(delivered_at, occurred_at, event_id);

CREATE INDEX evidence_links_fragment_idx ON evidence_links(source_fragment_id, statement_id);

CREATE INDEX idea_traces_library_task_idx
ON idea_traces(library_id, task_id, created_at, trace_id);

CREATE INDEX meaning_read_receipt_items_fragment_idx ON meaning_read_receipt_items(source_fragment_id, read_receipt_id);

CREATE INDEX meaning_review_decisions_target_idx ON meaning_review_decisions(target_type, target_id, created_at, review_decision_id);

CREATE INDEX read_receipt_corpus_sets_set_idx
ON read_receipt_corpus_sets(corpus_read_set_id, read_receipt_id);

CREATE INDEX reasoning_closure_library_decision_idx
ON reasoning_closure_results(library_id, decision, created_at, closure_id);

CREATE INDEX recall_run_artifacts_access_idx
ON recall_run_artifacts(access_receipt_id);

CREATE INDEX recall_run_artifacts_coverage_idx
ON recall_run_artifacts(coverage_report_id);

CREATE INDEX recall_run_artifacts_packet_idx
ON recall_run_artifacts(evidence_packet_id);

CREATE INDEX recall_run_artifacts_read_idx
ON recall_run_artifacts(read_receipt_id);

CREATE INDEX recall_run_artifacts_retrieval_idx
ON recall_run_artifacts(retrieval_receipt_id);

CREATE INDEX relation_import_run_items_relation_idx
ON relation_import_run_items(relation_id, relation_import_run_id);

CREATE INDEX relation_review_decisions_session_idx
ON relation_review_decisions(review_session_id, created_at, review_decision_id);

CREATE INDEX relation_review_decisions_target_idx
ON relation_review_decisions(
    library_id, target_id, target_hash, scope_json, created_at, review_decision_id
);

CREATE INDEX research_session_events_session_sequence_idx
ON research_session_events(session_id, sequence);

CREATE INDEX research_sessions_library_created_idx
ON research_sessions(library_id, created_at, session_id);

CREATE INDEX source_identity_bindings_family_content_idx
ON source_identity_bindings(source_family_id, content_sha256, source_id);

CREATE INDEX statements_collection_idx ON statements(collection_id, created_at, statement_id);

CREATE TRIGGER access_policies_no_delete BEFORE DELETE ON access_policies BEGIN SELECT RAISE(ABORT, 'access_policies is append-only'); END;

CREATE TRIGGER access_policies_no_update BEFORE UPDATE ON access_policies BEGIN SELECT RAISE(ABORT, 'access_policies is append-only'); END;

CREATE TRIGGER access_policy_collection_rules_no_delete BEFORE DELETE ON access_policy_collection_rules BEGIN SELECT RAISE(ABORT, 'access_policy_collection_rules is append-only'); END;

CREATE TRIGGER access_policy_collection_rules_no_update BEFORE UPDATE ON access_policy_collection_rules BEGIN SELECT RAISE(ABORT, 'access_policy_collection_rules is append-only'); END;

CREATE TRIGGER access_policy_source_rules_no_delete BEFORE DELETE ON access_policy_source_rules BEGIN SELECT RAISE(ABORT, 'access_policy_source_rules is append-only'); END;

CREATE TRIGGER access_policy_source_rules_no_update BEFORE UPDATE ON access_policy_source_rules BEGIN SELECT RAISE(ABORT, 'access_policy_source_rules is append-only'); END;

CREATE TRIGGER access_receipts_no_delete BEFORE DELETE ON access_receipts BEGIN SELECT RAISE(ABORT, 'access_receipts is append-only'); END;

CREATE TRIGGER access_receipts_no_update BEFORE UPDATE ON access_receipts BEGIN SELECT RAISE(ABORT, 'access_receipts is append-only'); END;

CREATE TRIGGER answer_projection_receipts_no_delete
BEFORE DELETE ON answer_projection_receipts BEGIN SELECT RAISE(ABORT, 'answer_projection_receipts is append-only'); END;

CREATE TRIGGER answer_projection_receipts_no_update
BEFORE UPDATE ON answer_projection_receipts BEGIN SELECT RAISE(ABORT, 'answer_projection_receipts is append-only'); END;

CREATE TRIGGER answer_projections_no_delete
BEFORE DELETE ON answer_projections BEGIN SELECT RAISE(ABORT, 'answer_projections is append-only'); END;

CREATE TRIGGER answer_projections_no_update
BEFORE UPDATE ON answer_projections BEGIN SELECT RAISE(ABORT, 'answer_projections is append-only'); END;

CREATE TRIGGER blobs_no_delete BEFORE DELETE ON blobs BEGIN SELECT RAISE(ABORT, 'blobs is append-only'); END;

CREATE TRIGGER blobs_no_update BEFORE UPDATE ON blobs BEGIN SELECT RAISE(ABORT, 'blobs is append-only'); END;

CREATE TRIGGER collection_root_identity_manifests_no_delete BEFORE DELETE ON collection_root_identity_manifests BEGIN SELECT RAISE(ABORT, 'collection_root_identity_manifests is append-only'); END;

CREATE TRIGGER collection_root_identity_manifests_no_update BEFORE UPDATE ON collection_root_identity_manifests BEGIN SELECT RAISE(ABORT, 'collection_root_identity_manifests is append-only'); END;

CREATE TRIGGER collection_root_identity_manifests_validate_insert BEFORE INSERT ON collection_root_identity_manifests
BEGIN
    SELECT RAISE(ABORT, 'identity manifest path must be absolute')
    WHERE substr(NEW.manifest_path, 1, 1) <> '/';
END;

CREATE TRIGGER concept_aliases_no_delete BEFORE DELETE ON concept_aliases BEGIN SELECT RAISE(ABORT, 'concept_aliases is append-only'); END;

CREATE TRIGGER concept_aliases_no_update BEFORE UPDATE ON concept_aliases BEGIN SELECT RAISE(ABORT, 'concept_aliases is append-only'); END;

CREATE TRIGGER concept_meaning_statements_no_delete BEFORE DELETE ON concept_meaning_statements BEGIN SELECT RAISE(ABORT, 'concept_meaning_statements is append-only'); END;

CREATE TRIGGER concept_meaning_statements_no_update BEFORE UPDATE ON concept_meaning_statements BEGIN SELECT RAISE(ABORT, 'concept_meaning_statements is append-only'); END;

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

CREATE TRIGGER concept_meanings_no_delete BEFORE DELETE ON concept_meanings BEGIN SELECT RAISE(ABORT, 'concept_meanings is append-only'); END;

CREATE TRIGGER concept_meanings_no_update BEFORE UPDATE ON concept_meanings BEGIN SELECT RAISE(ABORT, 'concept_meanings is append-only'); END;

CREATE TRIGGER concept_mentions_no_delete BEFORE DELETE ON concept_mentions BEGIN SELECT RAISE(ABORT, 'concept_mentions is append-only'); END;

CREATE TRIGGER concept_mentions_no_update BEFORE UPDATE ON concept_mentions BEGIN SELECT RAISE(ABORT, 'concept_mentions is append-only'); END;

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

CREATE TRIGGER concepts_no_delete BEFORE DELETE ON concepts BEGIN SELECT RAISE(ABORT, 'concepts is append-only'); END;

CREATE TRIGGER concepts_no_update BEFORE UPDATE ON concepts BEGIN SELECT RAISE(ABORT, 'concepts is append-only'); END;

CREATE TRIGGER corpus_read_set_items_no_delete BEFORE DELETE ON corpus_read_set_items BEGIN SELECT RAISE(ABORT, 'corpus_read_set_items is append-only'); END;

CREATE TRIGGER corpus_read_set_items_no_update BEFORE UPDATE ON corpus_read_set_items BEGIN SELECT RAISE(ABORT, 'corpus_read_set_items is append-only'); END;

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

CREATE TRIGGER corpus_read_sets_no_delete BEFORE DELETE ON corpus_read_sets BEGIN SELECT RAISE(ABORT, 'corpus_read_sets is append-only'); END;

CREATE TRIGGER corpus_read_sets_no_update BEFORE UPDATE ON corpus_read_sets BEGIN SELECT RAISE(ABORT, 'corpus_read_sets is append-only'); END;

CREATE TRIGGER corpus_snapshots_no_delete BEFORE DELETE ON corpus_snapshots BEGIN SELECT RAISE(ABORT, 'corpus_snapshots is append-only'); END;

CREATE TRIGGER corpus_snapshots_no_update BEFORE UPDATE ON corpus_snapshots BEGIN SELECT RAISE(ABORT, 'corpus_snapshots is append-only'); END;

CREATE TRIGGER coverage_reports_no_delete BEFORE DELETE ON coverage_reports BEGIN SELECT RAISE(ABORT, 'coverage_reports is append-only'); END;

CREATE TRIGGER coverage_reports_no_update BEFORE UPDATE ON coverage_reports BEGIN SELECT RAISE(ABORT, 'coverage_reports is append-only'); END;

CREATE TRIGGER entities_no_delete BEFORE DELETE ON entities BEGIN SELECT RAISE(ABORT, 'entities is append-only'); END;

CREATE TRIGGER entities_no_update BEFORE UPDATE ON entities BEGIN SELECT RAISE(ABORT, 'entities is append-only'); END;

CREATE TRIGGER entity_aliases_no_delete BEFORE DELETE ON entity_aliases BEGIN SELECT RAISE(ABORT, 'entity_aliases is append-only'); END;

CREATE TRIGGER entity_aliases_no_update BEFORE UPDATE ON entity_aliases BEGIN SELECT RAISE(ABORT, 'entity_aliases is append-only'); END;

CREATE TRIGGER entity_mentions_no_delete BEFORE DELETE ON entity_mentions BEGIN SELECT RAISE(ABORT, 'entity_mentions is append-only'); END;

CREATE TRIGGER entity_mentions_no_update BEFORE UPDATE ON entity_mentions BEGIN SELECT RAISE(ABORT, 'entity_mentions is append-only'); END;

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

CREATE TRIGGER event_outbox_delivery_only BEFORE UPDATE ON event_outbox
WHEN OLD.delivered_at IS NOT NULL
    OR NEW.delivered_at IS NULL
    OR length(trim(NEW.delivered_at)) = 0
    OR NEW.event_id IS NOT OLD.event_id
    OR NEW.event_type IS NOT OLD.event_type
    OR NEW.aggregate_type IS NOT OLD.aggregate_type
    OR NEW.aggregate_id IS NOT OLD.aggregate_id
    OR NEW.payload_json IS NOT OLD.payload_json
    OR NEW.payload_hash IS NOT OLD.payload_hash
    OR NEW.occurred_at IS NOT OLD.occurred_at
BEGIN
    SELECT RAISE(ABORT, 'event_outbox permits only first delivery');
END;

CREATE TRIGGER event_outbox_no_delete BEFORE DELETE ON event_outbox BEGIN SELECT RAISE(ABORT, 'event_outbox is append-only'); END;

CREATE TRIGGER evidence_links_no_delete BEFORE DELETE ON evidence_links BEGIN SELECT RAISE(ABORT, 'evidence_links is append-only'); END;

CREATE TRIGGER evidence_links_no_update BEFORE UPDATE ON evidence_links BEGIN SELECT RAISE(ABORT, 'evidence_links is append-only'); END;

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

CREATE TRIGGER evidence_packets_no_delete BEFORE DELETE ON evidence_packets BEGIN SELECT RAISE(ABORT, 'evidence_packets is append-only'); END;

CREATE TRIGGER evidence_packets_no_update BEFORE UPDATE ON evidence_packets BEGIN SELECT RAISE(ABORT, 'evidence_packets is append-only'); END;

CREATE TRIGGER idea_traces_no_delete
BEFORE DELETE ON idea_traces BEGIN SELECT RAISE(ABORT, 'idea_traces is append-only'); END;

CREATE TRIGGER idea_traces_no_update
BEFORE UPDATE ON idea_traces BEGIN SELECT RAISE(ABORT, 'idea_traces is append-only'); END;

CREATE TRIGGER libraries_no_delete BEFORE DELETE ON libraries BEGIN SELECT RAISE(ABORT, 'libraries is append-only'); END;

CREATE TRIGGER libraries_no_update BEFORE UPDATE ON libraries BEGIN SELECT RAISE(ABORT, 'libraries is append-only'); END;

CREATE TRIGGER libraries_one_row_insert BEFORE INSERT ON libraries
WHEN EXISTS (SELECT 1 FROM libraries)
BEGIN
    SELECT RAISE(ABORT, 'one Library per database');
END;

CREATE TRIGGER meaning_coverage_reports_no_delete BEFORE DELETE ON meaning_coverage_reports BEGIN SELECT RAISE(ABORT, 'meaning_coverage_reports is append-only'); END;

CREATE TRIGGER meaning_coverage_reports_no_update BEFORE UPDATE ON meaning_coverage_reports BEGIN SELECT RAISE(ABORT, 'meaning_coverage_reports is append-only'); END;

CREATE TRIGGER meaning_processing_runs_no_delete BEFORE DELETE ON meaning_processing_runs BEGIN SELECT RAISE(ABORT, 'meaning_processing_runs is append-only'); END;

CREATE TRIGGER meaning_processing_runs_no_update BEFORE UPDATE ON meaning_processing_runs BEGIN SELECT RAISE(ABORT, 'meaning_processing_runs is append-only'); END;

CREATE TRIGGER meaning_read_receipt_items_no_delete BEFORE DELETE ON meaning_read_receipt_items BEGIN SELECT RAISE(ABORT, 'meaning_read_receipt_items is append-only'); END;

CREATE TRIGGER meaning_read_receipt_items_no_update BEFORE UPDATE ON meaning_read_receipt_items BEGIN SELECT RAISE(ABORT, 'meaning_read_receipt_items is append-only'); END;

CREATE TRIGGER meaning_read_receipts_no_delete BEFORE DELETE ON meaning_read_receipts BEGIN SELECT RAISE(ABORT, 'meaning_read_receipts is append-only'); END;

CREATE TRIGGER meaning_read_receipts_no_update BEFORE UPDATE ON meaning_read_receipts BEGIN SELECT RAISE(ABORT, 'meaning_read_receipts is append-only'); END;

CREATE TRIGGER meaning_review_decisions_no_delete BEFORE DELETE ON meaning_review_decisions BEGIN SELECT RAISE(ABORT, 'meaning_review_decisions is append-only'); END;

CREATE TRIGGER meaning_review_decisions_no_update BEFORE UPDATE ON meaning_review_decisions BEGIN SELECT RAISE(ABORT, 'meaning_review_decisions is append-only'); END;

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

CREATE TRIGGER meaning_review_sessions_no_delete BEFORE DELETE ON meaning_review_sessions BEGIN SELECT RAISE(ABORT, 'meaning_review_sessions is append-only'); END;

CREATE TRIGGER meaning_review_sessions_no_update BEFORE UPDATE ON meaning_review_sessions BEGIN SELECT RAISE(ABORT, 'meaning_review_sessions is append-only'); END;

CREATE TRIGGER omissions_no_delete BEFORE DELETE ON omissions BEGIN SELECT RAISE(ABORT, 'omissions is append-only'); END;

CREATE TRIGGER omissions_no_update BEFORE UPDATE ON omissions BEGIN SELECT RAISE(ABORT, 'omissions is append-only'); END;

CREATE TRIGGER packet_items_no_delete BEFORE DELETE ON packet_items BEGIN SELECT RAISE(ABORT, 'packet_items is append-only'); END;

CREATE TRIGGER packet_items_no_update BEFORE UPDATE ON packet_items BEGIN SELECT RAISE(ABORT, 'packet_items is append-only'); END;

CREATE TRIGGER query_requests_no_delete BEFORE DELETE ON query_requests BEGIN SELECT RAISE(ABORT, 'query_requests is append-only'); END;

CREATE TRIGGER query_requests_no_update BEFORE UPDATE ON query_requests BEGIN SELECT RAISE(ABORT, 'query_requests is append-only'); END;

CREATE TRIGGER read_receipt_corpus_sets_no_delete BEFORE DELETE ON read_receipt_corpus_sets BEGIN SELECT RAISE(ABORT, 'read_receipt_corpus_sets is append-only'); END;

CREATE TRIGGER read_receipt_corpus_sets_no_update BEFORE UPDATE ON read_receipt_corpus_sets BEGIN SELECT RAISE(ABORT, 'read_receipt_corpus_sets is append-only'); END;

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

CREATE TRIGGER read_receipt_items_no_delete BEFORE DELETE ON read_receipt_items BEGIN SELECT RAISE(ABORT, 'read_receipt_items is append-only'); END;

CREATE TRIGGER read_receipt_items_no_update BEFORE UPDATE ON read_receipt_items BEGIN SELECT RAISE(ABORT, 'read_receipt_items is append-only'); END;

CREATE TRIGGER read_receipt_items_reject_shared_insert
BEFORE INSERT ON read_receipt_items
WHEN EXISTS (
    SELECT 1 FROM read_receipt_corpus_sets AS shared
    WHERE shared.read_receipt_id = NEW.read_receipt_id
)
BEGIN
    SELECT RAISE(ABORT, 'ReadReceipt cannot have shared and legacy item representations');
END;

CREATE TRIGGER read_receipts_no_delete BEFORE DELETE ON read_receipts BEGIN SELECT RAISE(ABORT, 'read_receipts is append-only'); END;

CREATE TRIGGER read_receipts_no_update BEFORE UPDATE ON read_receipts BEGIN SELECT RAISE(ABORT, 'read_receipts is append-only'); END;

CREATE TRIGGER reasoning_closure_results_no_delete
BEFORE DELETE ON reasoning_closure_results BEGIN SELECT RAISE(ABORT, 'reasoning_closure_results is append-only'); END;

CREATE TRIGGER reasoning_closure_results_no_update
BEFORE UPDATE ON reasoning_closure_results BEGIN SELECT RAISE(ABORT, 'reasoning_closure_results is append-only'); END;

CREATE TRIGGER recall_run_artifacts_no_delete
BEFORE DELETE ON recall_run_artifacts
BEGIN
    SELECT RAISE(ABORT, 'recall_run_artifacts is append-only');
END;

CREATE TRIGGER recall_run_artifacts_no_update
BEFORE UPDATE ON recall_run_artifacts
BEGIN
    SELECT RAISE(ABORT, 'recall_run_artifacts is append-only');
END;

CREATE TRIGGER recall_run_artifacts_validate_insert
BEFORE INSERT ON recall_run_artifacts
BEGIN
    SELECT RAISE(ABORT, 'recall run artifacts are inconsistent with the run query or packet')
    WHERE NOT EXISTS (
        SELECT 1
        FROM processing_runs AS target_run
        JOIN query_requests AS target_query
          ON target_query.query_request_id = target_run.query_request_id
        JOIN corpus_snapshots AS target_snapshot
          ON target_snapshot.corpus_snapshot_id = target_query.corpus_snapshot_id
        JOIN coverage_reports AS coverage
          ON coverage.coverage_report_id = NEW.coverage_report_id
         AND coverage.stage = 'recall'
        JOIN processing_runs AS coverage_run
          ON coverage_run.processing_run_id = coverage.processing_run_id
        JOIN read_receipts AS read_receipt
          ON read_receipt.read_receipt_id = NEW.read_receipt_id
        JOIN processing_runs AS read_run
          ON read_run.processing_run_id = read_receipt.processing_run_id
        JOIN access_receipts AS access_receipt
          ON access_receipt.access_receipt_id = NEW.access_receipt_id
        JOIN processing_runs AS access_run
          ON access_run.processing_run_id = access_receipt.processing_run_id
        JOIN retrieval_receipts AS retrieval_receipt
          ON retrieval_receipt.retrieval_receipt_id = NEW.retrieval_receipt_id
        JOIN processing_runs AS retrieval_run
          ON retrieval_run.processing_run_id = retrieval_receipt.processing_run_id
        JOIN evidence_packets AS packet
          ON packet.evidence_packet_id = NEW.evidence_packet_id
        WHERE target_run.processing_run_id = NEW.processing_run_id
          AND target_run.kind IN ('recall', 'replay')
          AND coverage_run.kind IN ('recall', 'replay')
          AND read_run.kind IN ('recall', 'replay')
          AND access_run.kind IN ('recall', 'replay')
          AND retrieval_run.kind IN ('recall', 'replay')
          AND coverage_run.query_request_id = target_query.query_request_id
          AND read_run.query_request_id = target_query.query_request_id
          AND access_run.query_request_id = target_query.query_request_id
          AND retrieval_run.query_request_id = target_query.query_request_id
          AND access_receipt.access_policy_id = target_query.access_policy_id
          AND access_receipt.snapshot_hash = target_snapshot.manifest_hash
          AND packet.query_request_id = target_query.query_request_id
          AND packet.corpus_snapshot_id = target_query.corpus_snapshot_id
    );
END;

CREATE TRIGGER relation_evidence_links_no_delete BEFORE DELETE ON relation_evidence_links BEGIN SELECT RAISE(ABORT, 'relation_evidence_links is append-only'); END;

CREATE TRIGGER relation_evidence_links_no_update BEFORE UPDATE ON relation_evidence_links BEGIN SELECT RAISE(ABORT, 'relation_evidence_links is append-only'); END;

CREATE TRIGGER relation_evidence_links_validate_insert
BEFORE INSERT ON relation_evidence_links
BEGIN
    SELECT RAISE(ABORT, 'Relation evidence must belong to the same Library')
    WHERE NOT EXISTS (
        SELECT 1
        FROM relations AS relation
        JOIN evidence_links AS evidence ON evidence.evidence_link_id = NEW.evidence_link_id
        JOIN statements AS statement ON statement.statement_id = evidence.statement_id
        WHERE relation.relation_id = NEW.relation_id
          AND statement.library_id = relation.library_id
    );
END;

CREATE TRIGGER relation_import_run_items_no_delete BEFORE DELETE ON relation_import_run_items BEGIN SELECT RAISE(ABORT, 'relation_import_run_items is append-only'); END;

CREATE TRIGGER relation_import_run_items_no_update BEFORE UPDATE ON relation_import_run_items BEGIN SELECT RAISE(ABORT, 'relation_import_run_items is append-only'); END;

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

CREATE TRIGGER relation_import_runs_no_delete BEFORE DELETE ON relation_import_runs BEGIN SELECT RAISE(ABORT, 'relation_import_runs is append-only'); END;

CREATE TRIGGER relation_import_runs_no_update BEFORE UPDATE ON relation_import_runs BEGIN SELECT RAISE(ABORT, 'relation_import_runs is append-only'); END;

CREATE TRIGGER relation_path_collections_no_delete BEFORE DELETE ON relation_path_collections BEGIN SELECT RAISE(ABORT, 'relation_path_collections is append-only'); END;

CREATE TRIGGER relation_path_collections_no_update BEFORE UPDATE ON relation_path_collections BEGIN SELECT RAISE(ABORT, 'relation_path_collections is append-only'); END;

CREATE TRIGGER relation_path_exclusions_no_delete BEFORE DELETE ON relation_path_exclusions BEGIN SELECT RAISE(ABORT, 'relation_path_exclusions is append-only'); END;

CREATE TRIGGER relation_path_exclusions_no_update BEFORE UPDATE ON relation_path_exclusions BEGIN SELECT RAISE(ABORT, 'relation_path_exclusions is append-only'); END;

CREATE TRIGGER relation_path_receipts_no_delete BEFORE DELETE ON relation_path_receipts BEGIN SELECT RAISE(ABORT, 'relation_path_receipts is append-only'); END;

CREATE TRIGGER relation_path_receipts_no_update BEFORE UPDATE ON relation_path_receipts BEGIN SELECT RAISE(ABORT, 'relation_path_receipts is append-only'); END;

CREATE TRIGGER relation_path_relation_types_no_delete BEFORE DELETE ON relation_path_relation_types BEGIN SELECT RAISE(ABORT, 'relation_path_relation_types is append-only'); END;

CREATE TRIGGER relation_path_relation_types_no_update BEFORE UPDATE ON relation_path_relation_types BEGIN SELECT RAISE(ABORT, 'relation_path_relation_types is append-only'); END;

CREATE TRIGGER relation_path_seeds_no_delete BEFORE DELETE ON relation_path_seeds BEGIN SELECT RAISE(ABORT, 'relation_path_seeds is append-only'); END;

CREATE TRIGGER relation_path_seeds_no_update BEFORE UPDATE ON relation_path_seeds BEGIN SELECT RAISE(ABORT, 'relation_path_seeds is append-only'); END;

CREATE TRIGGER relation_path_steps_no_delete BEFORE DELETE ON relation_path_steps BEGIN SELECT RAISE(ABORT, 'relation_path_steps is append-only'); END;

CREATE TRIGGER relation_path_steps_no_update BEFORE UPDATE ON relation_path_steps BEGIN SELECT RAISE(ABORT, 'relation_path_steps is append-only'); END;

CREATE TRIGGER relation_path_steps_validate_insert
BEFORE INSERT ON relation_path_steps
BEGIN
    SELECT RAISE(ABORT, 'Relation path step does not match its typed Relation')
    WHERE NOT EXISTS (
        SELECT 1 FROM relations AS relation
        WHERE relation.relation_id = NEW.relation_id
          AND (
            (relation.subject_type = NEW.from_type AND relation.subject_id = NEW.from_id AND relation.object_type = NEW.to_type AND relation.object_id = NEW.to_id)
            OR
            (relation.object_type = NEW.from_type AND relation.object_id = NEW.from_id AND relation.subject_type = NEW.to_type AND relation.subject_id = NEW.to_id)
          )
    );
END;

CREATE TRIGGER relation_paths_no_delete BEFORE DELETE ON relation_paths BEGIN SELECT RAISE(ABORT, 'relation_paths is append-only'); END;

CREATE TRIGGER relation_paths_no_update BEFORE UPDATE ON relation_paths BEGIN SELECT RAISE(ABORT, 'relation_paths is append-only'); END;

CREATE TRIGGER relation_review_decisions_no_delete BEFORE DELETE ON relation_review_decisions BEGIN SELECT RAISE(ABORT, 'relation_review_decisions is append-only'); END;

CREATE TRIGGER relation_review_decisions_no_update BEFORE UPDATE ON relation_review_decisions BEGIN SELECT RAISE(ABORT, 'relation_review_decisions is append-only'); END;

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

CREATE TRIGGER relations_no_delete BEFORE DELETE ON relations BEGIN SELECT RAISE(ABORT, 'relations is append-only'); END;

CREATE TRIGGER relations_no_update BEFORE UPDATE ON relations BEGIN SELECT RAISE(ABORT, 'relations is append-only'); END;

CREATE TRIGGER relations_validate_endpoints_insert
BEFORE INSERT ON relations
BEGIN
    SELECT RAISE(ABORT, 'Relation subject endpoint does not exist in its Library')
    WHERE
        (NEW.subject_type = 'statement' AND NOT EXISTS (SELECT 1 FROM statements WHERE statement_id = NEW.subject_id AND library_id = NEW.library_id)) OR
        (NEW.subject_type = 'concept' AND NOT EXISTS (SELECT 1 FROM concepts WHERE concept_id = NEW.subject_id AND library_id = NEW.library_id)) OR
        (NEW.subject_type = 'concept_meaning' AND NOT EXISTS (SELECT 1 FROM concept_meanings AS meaning JOIN concepts AS concept ON concept.concept_id = meaning.concept_id WHERE meaning.concept_meaning_id = NEW.subject_id AND concept.library_id = NEW.library_id)) OR
        (NEW.subject_type = 'entity' AND NOT EXISTS (SELECT 1 FROM entities WHERE entity_id = NEW.subject_id AND library_id = NEW.library_id)) OR
        (NEW.subject_type = 'voice' AND NOT EXISTS (SELECT 1 FROM voices WHERE voice_id = NEW.subject_id AND library_id = NEW.library_id)) OR
        (NEW.subject_type = 'time_context' AND NOT EXISTS (SELECT 1 FROM time_contexts WHERE time_context_id = NEW.subject_id AND library_id = NEW.library_id)) OR
        (NEW.subject_type = 'structure_unit' AND NOT EXISTS (SELECT 1 FROM structure_units AS unit JOIN structure_unit_generations AS generation ON generation.structure_unit_generation_id = unit.structure_unit_generation_id WHERE unit.structure_unit_id = NEW.subject_id AND generation.library_id = NEW.library_id));

    SELECT RAISE(ABORT, 'Relation object endpoint does not exist in its Library')
    WHERE
        (NEW.object_type = 'statement' AND NOT EXISTS (SELECT 1 FROM statements WHERE statement_id = NEW.object_id AND library_id = NEW.library_id)) OR
        (NEW.object_type = 'concept' AND NOT EXISTS (SELECT 1 FROM concepts WHERE concept_id = NEW.object_id AND library_id = NEW.library_id)) OR
        (NEW.object_type = 'concept_meaning' AND NOT EXISTS (SELECT 1 FROM concept_meanings AS meaning JOIN concepts AS concept ON concept.concept_id = meaning.concept_id WHERE meaning.concept_meaning_id = NEW.object_id AND concept.library_id = NEW.library_id)) OR
        (NEW.object_type = 'entity' AND NOT EXISTS (SELECT 1 FROM entities WHERE entity_id = NEW.object_id AND library_id = NEW.library_id)) OR
        (NEW.object_type = 'voice' AND NOT EXISTS (SELECT 1 FROM voices WHERE voice_id = NEW.object_id AND library_id = NEW.library_id)) OR
        (NEW.object_type = 'time_context' AND NOT EXISTS (SELECT 1 FROM time_contexts WHERE time_context_id = NEW.object_id AND library_id = NEW.library_id)) OR
        (NEW.object_type = 'structure_unit' AND NOT EXISTS (SELECT 1 FROM structure_units AS unit JOIN structure_unit_generations AS generation ON generation.structure_unit_generation_id = unit.structure_unit_generation_id WHERE unit.structure_unit_id = NEW.object_id AND generation.library_id = NEW.library_id));
END;

CREATE TRIGGER research_session_events_no_delete
BEFORE DELETE ON research_session_events
BEGIN
    SELECT RAISE(ABORT, 'research_session_events is append-only');
END;

CREATE TRIGGER research_session_events_no_update
BEFORE UPDATE ON research_session_events
BEGIN
    SELECT RAISE(ABORT, 'research_session_events is append-only');
END;

CREATE TRIGGER research_session_events_validate_insert
BEFORE INSERT ON research_session_events
BEGIN
    SELECT CASE
        WHEN NEW.sequence = 1
         AND NEW.previous_event_hash <> '0000000000000000000000000000000000000000000000000000000000000000'
        THEN RAISE(ABORT, 'first research session event must use the genesis hash')
    END;
    SELECT CASE
        WHEN NEW.sequence > 1
         AND NOT EXISTS (
            SELECT 1
            FROM research_session_events AS prior
            WHERE prior.session_id = NEW.session_id
              AND prior.sequence = NEW.sequence - 1
              AND prior.event_hash = NEW.previous_event_hash
        )
        THEN RAISE(ABORT, 'research session event does not close the previous hash')
    END;
    SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM research_session_events AS prior
            WHERE prior.session_id = NEW.session_id
              AND prior.kind = 'session_closed'
        )
        THEN RAISE(ABORT, 'research session is already closed')
    END;
    SELECT CASE
        WHEN NEW.kind IN ('decision_recorded', 'session_closed')
         AND NEW.actor_kind <> 'human'
        THEN RAISE(ABORT, 'research session decision requires a human actor')
    END;
END;

CREATE TRIGGER research_sessions_no_delete
BEFORE DELETE ON research_sessions
BEGIN
    SELECT RAISE(ABORT, 'research_sessions is append-only');
END;

CREATE TRIGGER research_sessions_no_update
BEFORE UPDATE ON research_sessions
BEGIN
    SELECT RAISE(ABORT, 'research_sessions is append-only');
END;

CREATE TRIGGER retrieval_receipts_no_delete BEFORE DELETE ON retrieval_receipts BEGIN SELECT RAISE(ABORT, 'retrieval_receipts is append-only'); END;

CREATE TRIGGER retrieval_receipts_no_update BEFORE UPDATE ON retrieval_receipts BEGIN SELECT RAISE(ABORT, 'retrieval_receipts is append-only'); END;

CREATE TRIGGER review_decisions_no_delete BEFORE DELETE ON review_decisions BEGIN SELECT RAISE(ABORT, 'review_decisions is append-only'); END;

CREATE TRIGGER review_decisions_no_update BEFORE UPDATE ON review_decisions BEGIN SELECT RAISE(ABORT, 'review_decisions is append-only'); END;

CREATE TRIGGER schema_migrations_no_delete BEFORE DELETE ON schema_migrations BEGIN SELECT RAISE(ABORT, 'schema_migrations is append-only'); END;

CREATE TRIGGER schema_migrations_no_update BEFORE UPDATE ON schema_migrations BEGIN SELECT RAISE(ABORT, 'schema_migrations is append-only'); END;

CREATE TRIGGER snapshot_members_no_delete BEFORE DELETE ON snapshot_members BEGIN SELECT RAISE(ABORT, 'snapshot_members is append-only'); END;

CREATE TRIGGER snapshot_members_no_update BEFORE UPDATE ON snapshot_members BEGIN SELECT RAISE(ABORT, 'snapshot_members is append-only'); END;

CREATE TRIGGER source_families_no_delete BEFORE DELETE ON source_families BEGIN SELECT RAISE(ABORT, 'source_families is append-only'); END;

CREATE TRIGGER source_families_no_update BEFORE UPDATE ON source_families BEGIN SELECT RAISE(ABORT, 'source_families is append-only'); END;

CREATE TRIGGER source_family_identities_no_delete BEFORE DELETE ON source_family_identities BEGIN SELECT RAISE(ABORT, 'source_family_identities is append-only'); END;

CREATE TRIGGER source_family_identities_no_update BEFORE UPDATE ON source_family_identities BEGIN SELECT RAISE(ABORT, 'source_family_identities is append-only'); END;

CREATE TRIGGER source_family_identities_validate_insert BEFORE INSERT ON source_family_identities
BEGIN
    SELECT RAISE(ABORT, 'SourceFamily identity Library must match SourceFamily')
    WHERE NOT EXISTS (
        SELECT 1 FROM source_families AS family
        WHERE family.source_family_id = NEW.source_family_id
          AND family.library_id = NEW.library_id
    );
END;

CREATE TRIGGER source_family_members_no_delete BEFORE DELETE ON source_family_members BEGIN SELECT RAISE(ABORT, 'source_family_members is append-only'); END;

CREATE TRIGGER source_family_members_no_update BEFORE UPDATE ON source_family_members BEGIN SELECT RAISE(ABORT, 'source_family_members is append-only'); END;

CREATE TRIGGER source_fragments_no_delete BEFORE DELETE ON source_fragments BEGIN SELECT RAISE(ABORT, 'source_fragments is append-only'); END;

CREATE TRIGGER source_fragments_no_update BEFORE UPDATE ON source_fragments BEGIN SELECT RAISE(ABORT, 'source_fragments is append-only'); END;

CREATE TRIGGER source_heads_no_delete BEFORE DELETE ON source_heads
BEGIN
    SELECT RAISE(ABORT, 'source_heads cannot be deleted');
END;

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

CREATE TRIGGER source_identity_bindings_no_delete BEFORE DELETE ON source_identity_bindings BEGIN SELECT RAISE(ABORT, 'source_identity_bindings is append-only'); END;

CREATE TRIGGER source_identity_bindings_no_update BEFORE UPDATE ON source_identity_bindings BEGIN SELECT RAISE(ABORT, 'source_identity_bindings is append-only'); END;

CREATE TRIGGER source_identity_bindings_validate_insert BEFORE INSERT ON source_identity_bindings
BEGIN
    SELECT RAISE(ABORT, 'Source identity binding requires a declared SourceFamily identity')
    WHERE NOT EXISTS (
        SELECT 1 FROM source_family_identities AS identity
        WHERE identity.source_family_id = NEW.source_family_id
    );

    SELECT RAISE(ABORT, 'Source identity binding SourceVersion does not belong to Source')
    WHERE NOT EXISTS (
        SELECT 1 FROM source_versions AS version
        WHERE version.source_version_id = NEW.source_version_id
          AND version.source_id = NEW.source_id
          AND version.content_sha256 = NEW.content_sha256
    );

    SELECT RAISE(ABORT, 'Source identity binding Source must belong to SourceFamily')
    WHERE NOT EXISTS (
        SELECT 1 FROM source_family_members AS member
        WHERE member.source_id = NEW.source_id
          AND member.source_family_id = NEW.source_family_id
    );
END;

CREATE TRIGGER source_versions_no_delete BEFORE DELETE ON source_versions BEGIN SELECT RAISE(ABORT, 'source_versions is append-only'); END;

CREATE TRIGGER source_versions_no_update BEFORE UPDATE ON source_versions BEGIN SELECT RAISE(ABORT, 'source_versions is append-only'); END;

CREATE TRIGGER sources_no_delete BEFORE DELETE ON sources BEGIN SELECT RAISE(ABORT, 'sources is append-only'); END;

CREATE TRIGGER sources_no_update BEFORE UPDATE ON sources BEGIN SELECT RAISE(ABORT, 'sources is append-only'); END;

CREATE TRIGGER statements_no_delete BEFORE DELETE ON statements BEGIN SELECT RAISE(ABORT, 'statements is append-only'); END;

CREATE TRIGGER statements_no_update BEFORE UPDATE ON statements BEGIN SELECT RAISE(ABORT, 'statements is append-only'); END;

CREATE TRIGGER structure_unit_generations_no_delete BEFORE DELETE ON structure_unit_generations BEGIN SELECT RAISE(ABORT, 'structure_unit_generations is append-only'); END;

CREATE TRIGGER structure_unit_generations_no_update BEFORE UPDATE ON structure_unit_generations BEGIN SELECT RAISE(ABORT, 'structure_unit_generations is append-only'); END;

CREATE TRIGGER structure_unit_members_no_delete BEFORE DELETE ON structure_unit_members BEGIN SELECT RAISE(ABORT, 'structure_unit_members is append-only'); END;

CREATE TRIGGER structure_unit_members_no_update BEFORE UPDATE ON structure_unit_members BEGIN SELECT RAISE(ABORT, 'structure_unit_members is append-only'); END;

CREATE TRIGGER structure_unit_members_validate_insert
BEFORE INSERT ON structure_unit_members
BEGIN
    SELECT RAISE(ABORT, 'StructureUnit member must match its exact SourceVersion and ordinal')
    WHERE NOT EXISTS (
        SELECT 1
        FROM structure_units AS unit
        JOIN source_fragments AS fragment ON fragment.source_fragment_id = NEW.source_fragment_id
        JOIN source_versions AS version ON version.source_version_id = fragment.source_version_id
        WHERE unit.structure_unit_id = NEW.structure_unit_id
          AND unit.source_version_id = fragment.source_version_id
          AND unit.source_id = version.source_id
          AND fragment.ordinal = NEW.ordinal
          AND fragment.text_sha256 = NEW.text_sha256
          AND NEW.ordinal = unit.first_ordinal + NEW.member_order
    );
END;

CREATE TRIGGER structure_unit_read_receipt_items_no_delete BEFORE DELETE ON structure_unit_read_receipt_items BEGIN SELECT RAISE(ABORT, 'structure_unit_read_receipt_items is append-only'); END;

CREATE TRIGGER structure_unit_read_receipt_items_no_update BEFORE UPDATE ON structure_unit_read_receipt_items BEGIN SELECT RAISE(ABORT, 'structure_unit_read_receipt_items is append-only'); END;

CREATE TRIGGER structure_unit_read_receipts_no_delete BEFORE DELETE ON structure_unit_read_receipts BEGIN SELECT RAISE(ABORT, 'structure_unit_read_receipts is append-only'); END;

CREATE TRIGGER structure_unit_read_receipts_no_update BEFORE UPDATE ON structure_unit_read_receipts BEGIN SELECT RAISE(ABORT, 'structure_unit_read_receipts is append-only'); END;

CREATE TRIGGER structure_units_no_delete BEFORE DELETE ON structure_units BEGIN SELECT RAISE(ABORT, 'structure_units is append-only'); END;

CREATE TRIGGER structure_units_no_update BEFORE UPDATE ON structure_units BEGIN SELECT RAISE(ABORT, 'structure_units is append-only'); END;

CREATE TRIGGER structure_units_validate_insert
BEFORE INSERT ON structure_units
BEGIN
    SELECT RAISE(ABORT, 'StructureUnit lineage must belong to its Library and CorpusSnapshot')
    WHERE NOT EXISTS (
        SELECT 1
        FROM structure_unit_generations AS generation
        JOIN sources AS source ON source.source_id = NEW.source_id
        JOIN source_versions AS version ON version.source_version_id = NEW.source_version_id
        JOIN snapshot_members AS member
          ON member.corpus_snapshot_id = generation.corpus_snapshot_id
         AND member.source_version_id = NEW.source_version_id
         AND member.membership_state = 'active'
        WHERE generation.structure_unit_generation_id = NEW.structure_unit_generation_id
          AND source.library_id = generation.library_id
          AND version.source_id = NEW.source_id
    );
END;

CREATE TRIGGER time_contexts_no_delete BEFORE DELETE ON time_contexts BEGIN SELECT RAISE(ABORT, 'time_contexts is append-only'); END;

CREATE TRIGGER time_contexts_no_update BEFORE UPDATE ON time_contexts BEGIN SELECT RAISE(ABORT, 'time_contexts is append-only'); END;

CREATE TRIGGER voices_no_delete BEFORE DELETE ON voices BEGIN SELECT RAISE(ABORT, 'voices is append-only'); END;

CREATE TRIGGER voices_no_update BEFORE UPDATE ON voices BEGIN SELECT RAISE(ABORT, 'voices is append-only'); END;
