-- Dithyramba VS0 core schema. Applied atomically by MigrationRunner.
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY CHECK (version > 0),
    name TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    applied_at TEXT NOT NULL
);

CREATE TABLE libraries (
    library_id TEXT PRIMARY KEY CHECK (substr(library_id, 1, 8) = 'library_' AND length(library_id) > 8),
    name TEXT NOT NULL,
    logical_identity_hash TEXT NOT NULL UNIQUE CHECK (length(logical_identity_hash) = 64 AND logical_identity_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);
CREATE TRIGGER libraries_one_row_insert BEFORE INSERT ON libraries
WHEN EXISTS (SELECT 1 FROM libraries)
BEGIN
    SELECT RAISE(ABORT, 'one Library per database');
END;

CREATE TABLE collections (
    collection_id TEXT PRIMARY KEY CHECK (substr(collection_id, 1, 11) = 'collection_' AND length(collection_id) > 11),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('corpus', 'case', 'branch', 'holdout')),
    created_at TEXT NOT NULL,
    UNIQUE (library_id, name)
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

CREATE TABLE blobs (
    content_sha256 TEXT PRIMARY KEY CHECK (length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*'),
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    relative_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
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

CREATE TABLE source_families (
    source_family_id TEXT PRIMARY KEY CHECK (substr(source_family_id, 1, 7) = 'family_' AND length(source_family_id) > 7),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL
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

CREATE TABLE access_policy_source_rules (
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    effect TEXT NOT NULL CHECK (effect IN ('allow', 'deny')),
    PRIMARY KEY (access_policy_id, source_id)
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
    UNIQUE (source_id, content_sha256)
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

CREATE TABLE collection_memberships (
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    state TEXT NOT NULL CHECK (state IN ('active', 'excluded', 'holdout')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (collection_id, source_id)
);

CREATE TABLE corpus_snapshots (
    corpus_snapshot_id TEXT PRIMARY KEY CHECK (substr(corpus_snapshot_id, 1, 9) = 'snapshot_' AND length(corpus_snapshot_id) > 9),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    scope_json TEXT NOT NULL,
    manifest_hash TEXT NOT NULL UNIQUE CHECK (length(manifest_hash) = 64 AND manifest_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);

CREATE TABLE snapshot_members (
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    membership_state TEXT NOT NULL CHECK (membership_state IN ('active', 'excluded', 'holdout')),
    PRIMARY KEY (corpus_snapshot_id, collection_id, source_version_id)
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

CREATE TABLE omissions (
    omission_id TEXT PRIMARY KEY CHECK (substr(omission_id, 1, 9) = 'omission_' AND length(omission_id) > 9),
    coverage_report_id TEXT NOT NULL REFERENCES coverage_reports(coverage_report_id) ON DELETE RESTRICT,
    category TEXT NOT NULL CHECK (category IN ('explicit_exclusion', 'budget', 'unsupported', 'parser_failure', 'policy')),
    disclosure TEXT NOT NULL CHECK (disclosure IN ('counted', 'redacted')),
    count INTEGER CHECK (count IS NULL OR count >= 0),
    reason_code TEXT NOT NULL,
    CHECK ((category = 'policy' AND disclosure = 'redacted' AND count IS NULL) OR category <> 'policy')
);

CREATE TABLE read_receipts (
    read_receipt_id TEXT PRIMARY KEY CHECK (substr(read_receipt_id, 1, 5) = 'read_' AND length(read_receipt_id) > 5),
    processing_run_id TEXT NOT NULL REFERENCES processing_runs(processing_run_id) ON DELETE RESTRICT,
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE read_receipt_items (
    read_receipt_id TEXT NOT NULL REFERENCES read_receipts(read_receipt_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    read_order INTEGER NOT NULL CHECK (read_order >= 0),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (read_receipt_id, source_fragment_id),
    UNIQUE (read_receipt_id, read_order)
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

CREATE TABLE evidence_packets (
    evidence_packet_id TEXT PRIMARY KEY CHECK (substr(evidence_packet_id, 1, 7) = 'packet_' AND length(evidence_packet_id) > 7),
    query_request_id TEXT NOT NULL REFERENCES query_requests(query_request_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    result_status TEXT NOT NULL CHECK (result_status IN ('evidence_found', 'no_evidence')),
    packet_json TEXT NOT NULL,
    packet_hash TEXT NOT NULL UNIQUE CHECK (length(packet_hash) = 64 AND packet_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
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

CREATE TRIGGER schema_migrations_no_update BEFORE UPDATE ON schema_migrations BEGIN SELECT RAISE(ABORT, 'schema_migrations is append-only'); END;
CREATE TRIGGER schema_migrations_no_delete BEFORE DELETE ON schema_migrations BEGIN SELECT RAISE(ABORT, 'schema_migrations is append-only'); END;
CREATE TRIGGER libraries_no_update BEFORE UPDATE ON libraries BEGIN SELECT RAISE(ABORT, 'libraries is append-only'); END;
CREATE TRIGGER libraries_no_delete BEFORE DELETE ON libraries BEGIN SELECT RAISE(ABORT, 'libraries is append-only'); END;
CREATE TRIGGER access_policies_no_update BEFORE UPDATE ON access_policies BEGIN SELECT RAISE(ABORT, 'access_policies is append-only'); END;
CREATE TRIGGER access_policies_no_delete BEFORE DELETE ON access_policies BEGIN SELECT RAISE(ABORT, 'access_policies is append-only'); END;
CREATE TRIGGER access_policy_collection_rules_no_update BEFORE UPDATE ON access_policy_collection_rules BEGIN SELECT RAISE(ABORT, 'access_policy_collection_rules is append-only'); END;
CREATE TRIGGER access_policy_collection_rules_no_delete BEFORE DELETE ON access_policy_collection_rules BEGIN SELECT RAISE(ABORT, 'access_policy_collection_rules is append-only'); END;
CREATE TRIGGER access_policy_source_rules_no_update BEFORE UPDATE ON access_policy_source_rules BEGIN SELECT RAISE(ABORT, 'access_policy_source_rules is append-only'); END;
CREATE TRIGGER access_policy_source_rules_no_delete BEFORE DELETE ON access_policy_source_rules BEGIN SELECT RAISE(ABORT, 'access_policy_source_rules is append-only'); END;
CREATE TRIGGER blobs_no_update BEFORE UPDATE ON blobs BEGIN SELECT RAISE(ABORT, 'blobs is append-only'); END;
CREATE TRIGGER blobs_no_delete BEFORE DELETE ON blobs BEGIN SELECT RAISE(ABORT, 'blobs is append-only'); END;
CREATE TRIGGER sources_no_update BEFORE UPDATE ON sources BEGIN SELECT RAISE(ABORT, 'sources is append-only'); END;
CREATE TRIGGER sources_no_delete BEFORE DELETE ON sources BEGIN SELECT RAISE(ABORT, 'sources is append-only'); END;
CREATE TRIGGER source_families_no_update BEFORE UPDATE ON source_families BEGIN SELECT RAISE(ABORT, 'source_families is append-only'); END;
CREATE TRIGGER source_families_no_delete BEFORE DELETE ON source_families BEGIN SELECT RAISE(ABORT, 'source_families is append-only'); END;
CREATE TRIGGER source_family_members_no_update BEFORE UPDATE ON source_family_members BEGIN SELECT RAISE(ABORT, 'source_family_members is append-only'); END;
CREATE TRIGGER source_family_members_no_delete BEFORE DELETE ON source_family_members BEGIN SELECT RAISE(ABORT, 'source_family_members is append-only'); END;
CREATE TRIGGER source_versions_no_update BEFORE UPDATE ON source_versions BEGIN SELECT RAISE(ABORT, 'source_versions is append-only'); END;
CREATE TRIGGER source_versions_no_delete BEFORE DELETE ON source_versions BEGIN SELECT RAISE(ABORT, 'source_versions is append-only'); END;
CREATE TRIGGER source_fragments_no_update BEFORE UPDATE ON source_fragments BEGIN SELECT RAISE(ABORT, 'source_fragments is append-only'); END;
CREATE TRIGGER source_fragments_no_delete BEFORE DELETE ON source_fragments BEGIN SELECT RAISE(ABORT, 'source_fragments is append-only'); END;
CREATE TRIGGER corpus_snapshots_no_update BEFORE UPDATE ON corpus_snapshots BEGIN SELECT RAISE(ABORT, 'corpus_snapshots is append-only'); END;
CREATE TRIGGER corpus_snapshots_no_delete BEFORE DELETE ON corpus_snapshots BEGIN SELECT RAISE(ABORT, 'corpus_snapshots is append-only'); END;
CREATE TRIGGER snapshot_members_no_update BEFORE UPDATE ON snapshot_members BEGIN SELECT RAISE(ABORT, 'snapshot_members is append-only'); END;
CREATE TRIGGER snapshot_members_no_delete BEFORE DELETE ON snapshot_members BEGIN SELECT RAISE(ABORT, 'snapshot_members is append-only'); END;
CREATE TRIGGER query_requests_no_update BEFORE UPDATE ON query_requests BEGIN SELECT RAISE(ABORT, 'query_requests is append-only'); END;
CREATE TRIGGER query_requests_no_delete BEFORE DELETE ON query_requests BEGIN SELECT RAISE(ABORT, 'query_requests is append-only'); END;
CREATE TRIGGER coverage_reports_no_update BEFORE UPDATE ON coverage_reports BEGIN SELECT RAISE(ABORT, 'coverage_reports is append-only'); END;
CREATE TRIGGER coverage_reports_no_delete BEFORE DELETE ON coverage_reports BEGIN SELECT RAISE(ABORT, 'coverage_reports is append-only'); END;
CREATE TRIGGER omissions_no_update BEFORE UPDATE ON omissions BEGIN SELECT RAISE(ABORT, 'omissions is append-only'); END;
CREATE TRIGGER omissions_no_delete BEFORE DELETE ON omissions BEGIN SELECT RAISE(ABORT, 'omissions is append-only'); END;
CREATE TRIGGER read_receipts_no_update BEFORE UPDATE ON read_receipts BEGIN SELECT RAISE(ABORT, 'read_receipts is append-only'); END;
CREATE TRIGGER read_receipts_no_delete BEFORE DELETE ON read_receipts BEGIN SELECT RAISE(ABORT, 'read_receipts is append-only'); END;
CREATE TRIGGER read_receipt_items_no_update BEFORE UPDATE ON read_receipt_items BEGIN SELECT RAISE(ABORT, 'read_receipt_items is append-only'); END;
CREATE TRIGGER read_receipt_items_no_delete BEFORE DELETE ON read_receipt_items BEGIN SELECT RAISE(ABORT, 'read_receipt_items is append-only'); END;
CREATE TRIGGER access_receipts_no_update BEFORE UPDATE ON access_receipts BEGIN SELECT RAISE(ABORT, 'access_receipts is append-only'); END;
CREATE TRIGGER access_receipts_no_delete BEFORE DELETE ON access_receipts BEGIN SELECT RAISE(ABORT, 'access_receipts is append-only'); END;
CREATE TRIGGER retrieval_receipts_no_update BEFORE UPDATE ON retrieval_receipts BEGIN SELECT RAISE(ABORT, 'retrieval_receipts is append-only'); END;
CREATE TRIGGER retrieval_receipts_no_delete BEFORE DELETE ON retrieval_receipts BEGIN SELECT RAISE(ABORT, 'retrieval_receipts is append-only'); END;
CREATE TRIGGER evidence_packets_no_update BEFORE UPDATE ON evidence_packets BEGIN SELECT RAISE(ABORT, 'evidence_packets is append-only'); END;
CREATE TRIGGER evidence_packets_no_delete BEFORE DELETE ON evidence_packets BEGIN SELECT RAISE(ABORT, 'evidence_packets is append-only'); END;
CREATE TRIGGER packet_items_no_update BEFORE UPDATE ON packet_items BEGIN SELECT RAISE(ABORT, 'packet_items is append-only'); END;
CREATE TRIGGER packet_items_no_delete BEFORE DELETE ON packet_items BEGIN SELECT RAISE(ABORT, 'packet_items is append-only'); END;
CREATE TRIGGER review_decisions_no_update BEFORE UPDATE ON review_decisions BEGIN SELECT RAISE(ABORT, 'review_decisions is append-only'); END;
CREATE TRIGGER review_decisions_no_delete BEFORE DELETE ON review_decisions BEGIN SELECT RAISE(ABORT, 'review_decisions is append-only'); END;
CREATE TRIGGER event_outbox_no_delete BEFORE DELETE ON event_outbox BEGIN SELECT RAISE(ABORT, 'event_outbox is append-only'); END;
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

CREATE INDEX event_outbox_pending_idx ON event_outbox(delivered_at, occurred_at, event_id);
CREATE INDEX collection_memberships_source_idx ON collection_memberships(source_id, collection_id);
