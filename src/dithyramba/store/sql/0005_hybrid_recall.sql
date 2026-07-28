-- Dithyramba P7 hybrid recall. All P7 records are immutable and coexist with VS0.
CREATE TABLE embedding_model_profiles (
    embedding_model_profile_id TEXT PRIMARY KEY CHECK (substr(embedding_model_profile_id, 1, 18) = 'embedding_profile_' AND length(embedding_model_profile_id) > 18),
    provider TEXT NOT NULL CHECK (provider = 'sentence_transformers'),
    model_id TEXT NOT NULL CHECK (length(model_id) BETWEEN 3 AND 255 AND instr(model_id, '/') > 1),
    revision TEXT NOT NULL CHECK (length(revision) = 40 AND revision NOT GLOB '*[^0-9a-f]*'),
    model_license TEXT NOT NULL CHECK (length(model_license) BETWEEN 1 AND 128),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 8192),
    max_tokens INTEGER NOT NULL CHECK (max_tokens BETWEEN 1 AND 32768),
    query_prefix TEXT NOT NULL CHECK (length(query_prefix) <= 128 AND instr(query_prefix, char(0)) = 0),
    passage_prefix TEXT NOT NULL CHECK (length(passage_prefix) <= 128 AND instr(passage_prefix, char(0)) = 0),
    normalize_embeddings INTEGER NOT NULL CHECK (normalize_embeddings = 1),
    trust_remote_code INTEGER NOT NULL CHECK (trust_remote_code = 0),
    profile_hash TEXT NOT NULL UNIQUE CHECK (length(profile_hash) = 64 AND profile_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (provider, model_id, revision, dimensions, max_tokens, query_prefix, passage_prefix)
);

CREATE TABLE model_runtime_profiles (
    model_runtime_profile_id TEXT PRIMARY KEY CHECK (substr(model_runtime_profile_id, 1, 16) = 'runtime_profile_' AND length(model_runtime_profile_id) > 16),
    sentence_transformers_version TEXT NOT NULL CHECK (length(sentence_transformers_version) BETWEEN 1 AND 128),
    transformers_version TEXT NOT NULL CHECK (length(transformers_version) BETWEEN 1 AND 128),
    torch_version TEXT NOT NULL CHECK (length(torch_version) BETWEEN 1 AND 128),
    device TEXT NOT NULL CHECK (device IN ('cpu', 'mps', 'cuda')),
    dtype TEXT NOT NULL CHECK (dtype = 'float32'),
    batch_size INTEGER NOT NULL CHECK (batch_size BETWEEN 1 AND 512),
    local_files_only INTEGER NOT NULL CHECK (local_files_only = 1),
    trust_remote_code INTEGER NOT NULL CHECK (trust_remote_code = 0),
    profile_hash TEXT NOT NULL UNIQUE CHECK (length(profile_hash) = 64 AND profile_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (sentence_transformers_version, transformers_version, torch_version, device, dtype, batch_size)
);

CREATE TABLE fusion_profiles (
    profile_hash TEXT PRIMARY KEY CHECK (length(profile_hash) = 64 AND profile_hash NOT GLOB '*[^0-9a-f]*'),
    algorithm TEXT NOT NULL CHECK (algorithm = 'rrf_v1'),
    rrf_k INTEGER NOT NULL CHECK (rrf_k = 60),
    fts_weight INTEGER NOT NULL CHECK (fts_weight = 1),
    dense_weight INTEGER NOT NULL CHECK (dense_weight = 1),
    max_per_derived_source_top_five INTEGER NOT NULL CHECK (max_per_derived_source_top_five = 2),
    created_at TEXT NOT NULL,
    UNIQUE (algorithm, rrf_k, fts_weight, dense_weight, max_per_derived_source_top_five)
);

CREATE TABLE reranker_profiles (
    reranker_profile_id TEXT PRIMARY KEY CHECK (substr(reranker_profile_id, 1, 17) = 'reranker_profile_' AND length(reranker_profile_id) > 17),
    provider TEXT NOT NULL CHECK (provider = 'sentence_transformers_cross_encoder'),
    model_id TEXT NOT NULL CHECK (length(model_id) BETWEEN 3 AND 255 AND instr(model_id, '/') > 1),
    revision TEXT NOT NULL CHECK (length(revision) = 40 AND revision NOT GLOB '*[^0-9a-f]*'),
    model_license TEXT NOT NULL CHECK (length(model_license) BETWEEN 1 AND 128),
    max_tokens INTEGER NOT NULL CHECK (max_tokens BETWEEN 1 AND 32768),
    local_files_only INTEGER NOT NULL CHECK (local_files_only = 1),
    trust_remote_code INTEGER NOT NULL CHECK (trust_remote_code = 0),
    profile_hash TEXT NOT NULL UNIQUE CHECK (length(profile_hash) = 64 AND profile_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (provider, model_id, revision, max_tokens)
);

CREATE TABLE corpus_layer_profiles (
    corpus_layer_profile_id TEXT PRIMARY KEY CHECK (substr(corpus_layer_profile_id, 1, 14) = 'layer_profile_' AND length(corpus_layer_profile_id) > 14),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    exclusion_hash TEXT NOT NULL CHECK (length(exclusion_hash) = 64 AND exclusion_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    item_count INTEGER NOT NULL CHECK (item_count >= 0),
    profile_hash TEXT NOT NULL UNIQUE CHECK (length(profile_hash) = 64 AND profile_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (library_id, corpus_snapshot_id, exclusion_hash, permitted_set_hash)
);

CREATE TABLE corpus_layer_items (
    corpus_layer_profile_id TEXT NOT NULL REFERENCES corpus_layer_profiles(corpus_layer_profile_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    layer TEXT NOT NULL CHECK (layer IN ('primary', 'derived', 'synthesis', 'unclassified')),
    PRIMARY KEY (corpus_layer_profile_id, source_id)
);

CREATE TABLE corpus_vector_generations (
    corpus_vector_generation_id TEXT PRIMARY KEY CHECK (substr(corpus_vector_generation_id, 1, 18) = 'vector_generation_' AND length(corpus_vector_generation_id) > 18),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    snapshot_hash TEXT NOT NULL CHECK (length(snapshot_hash) = 64 AND snapshot_hash NOT GLOB '*[^0-9a-f]*'),
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    policy_hash TEXT NOT NULL CHECK (length(policy_hash) = 64 AND policy_hash NOT GLOB '*[^0-9a-f]*'),
    exclusion_hash TEXT NOT NULL CHECK (length(exclusion_hash) = 64 AND exclusion_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    model_profile_hash TEXT NOT NULL REFERENCES embedding_model_profiles(profile_hash) ON DELETE RESTRICT,
    runtime_profile_hash TEXT NOT NULL REFERENCES model_runtime_profiles(profile_hash) ON DELETE RESTRICT,
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 8192),
    item_count INTEGER NOT NULL CHECK (item_count >= 0),
    generation_hash TEXT NOT NULL UNIQUE CHECK (length(generation_hash) = 64 AND generation_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    UNIQUE (library_id, corpus_snapshot_id, access_policy_id, policy_hash, exclusion_hash, permitted_set_hash, model_profile_hash, runtime_profile_hash, dimensions)
);

CREATE TABLE corpus_vector_items (
    corpus_vector_generation_id TEXT NOT NULL REFERENCES corpus_vector_generations(corpus_vector_generation_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    vector_sha256 TEXT NOT NULL CHECK (length(vector_sha256) = 64 AND vector_sha256 NOT GLOB '*[^0-9a-f]*'),
    vector_blob BLOB NOT NULL CHECK (typeof(vector_blob) = 'blob' AND length(vector_blob) > 0),
    PRIMARY KEY (corpus_vector_generation_id, source_fragment_id)
);

CREATE TABLE hybrid_query_requests (
    hybrid_query_request_id TEXT PRIMARY KEY CHECK (substr(hybrid_query_request_id, 1, 13) = 'hybrid_query_' AND length(hybrid_query_request_id) > 13),
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    question TEXT NOT NULL CHECK (length(question) BETWEEN 1 AND 2000 AND question = trim(question) AND instr(question, char(0)) = 0),
    purpose TEXT NOT NULL CHECK (length(purpose) BETWEEN 1 AND 64 AND substr(purpose, 1, 1) BETWEEN 'a' AND 'z' AND purpose = lower(purpose) AND purpose NOT GLOB '*[^a-z0-9_-]*'),
    model_profile_hash TEXT NOT NULL REFERENCES embedding_model_profiles(profile_hash) ON DELETE RESTRICT,
    runtime_profile_hash TEXT NOT NULL REFERENCES model_runtime_profiles(profile_hash) ON DELETE RESTRICT,
    layer_profile_hash TEXT NOT NULL REFERENCES corpus_layer_profiles(profile_hash) ON DELETE RESTRICT,
    fusion_profile_hash TEXT NOT NULL REFERENCES fusion_profiles(profile_hash) ON DELETE RESTRICT,
    reranker_profile_hash TEXT REFERENCES reranker_profiles(profile_hash) ON DELETE RESTRICT,
    exclusion_hash TEXT NOT NULL CHECK (length(exclusion_hash) = 64 AND exclusion_hash NOT GLOB '*[^0-9a-f]*'),
    collection_count INTEGER NOT NULL CHECK (collection_count BETWEEN 1 AND 16),
    exclusion_count INTEGER NOT NULL CHECK (exclusion_count >= 0),
    max_fts_candidates INTEGER NOT NULL CHECK (max_fts_candidates BETWEEN 1 AND 500),
    max_dense_candidates INTEGER NOT NULL CHECK (max_dense_candidates BETWEEN 1 AND 500),
    max_fused_candidates INTEGER NOT NULL CHECK (max_fused_candidates BETWEEN 1 AND 500),
    max_source_fragments INTEGER NOT NULL CHECK (max_source_fragments BETWEEN 1 AND 100 AND max_source_fragments <= max_fused_candidates),
    result_format TEXT NOT NULL CHECK (result_format = 'hybrid_evidence_packet'),
    request_hash TEXT NOT NULL UNIQUE CHECK (length(request_hash) = 64 AND request_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);

CREATE TABLE hybrid_query_collections (
    hybrid_query_request_id TEXT NOT NULL REFERENCES hybrid_query_requests(hybrid_query_request_id) ON DELETE RESTRICT,
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE RESTRICT,
    PRIMARY KEY (hybrid_query_request_id, collection_id)
);

CREATE TABLE hybrid_query_exclusions (
    hybrid_query_request_id TEXT NOT NULL REFERENCES hybrid_query_requests(hybrid_query_request_id) ON DELETE RESTRICT,
    target_type TEXT NOT NULL CHECK (target_type IN ('source', 'source_family', 'source_fragment')),
    target_id TEXT NOT NULL,
    PRIMARY KEY (hybrid_query_request_id, target_type, target_id)
);

CREATE TABLE hybrid_processing_runs (
    hybrid_processing_run_id TEXT PRIMARY KEY CHECK (substr(hybrid_processing_run_id, 1, 4) = 'run_' AND length(hybrid_processing_run_id) > 4),
    hybrid_query_request_id TEXT NOT NULL REFERENCES hybrid_query_requests(hybrid_query_request_id) ON DELETE RESTRICT,
    corpus_vector_generation_id TEXT NOT NULL REFERENCES corpus_vector_generations(corpus_vector_generation_id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (kind IN ('recall', 'replay')),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed', 'cancelled')),
    code_version TEXT NOT NULL CHECK (length(trim(code_version)) > 0),
    profile_version TEXT NOT NULL CHECK (length(trim(profile_version)) > 0),
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    error_code TEXT,
    output_hash TEXT CHECK (output_hash IS NULL OR (length(output_hash) = 64 AND output_hash NOT GLOB '*[^0-9a-f]*')),
    CHECK ((status = 'succeeded' AND error_code IS NULL AND output_hash IS NOT NULL) OR (status <> 'succeeded' AND error_code IS NOT NULL AND output_hash IS NULL))
);

CREATE TABLE hybrid_coverage_reports (
    hybrid_coverage_report_id TEXT PRIMARY KEY CHECK (substr(hybrid_coverage_report_id, 1, 16) = 'hybrid_coverage_' AND length(hybrid_coverage_report_id) > 16),
    hybrid_processing_run_id TEXT NOT NULL REFERENCES hybrid_processing_runs(hybrid_processing_run_id) ON DELETE RESTRICT,
    hybrid_query_request_id TEXT NOT NULL REFERENCES hybrid_query_requests(hybrid_query_request_id) ON DELETE RESTRICT,
    processed_count INTEGER NOT NULL CHECK (processed_count >= 0),
    skipped_count INTEGER NOT NULL CHECK (skipped_count >= 0),
    failed_count INTEGER NOT NULL CHECK (failed_count >= 0),
    policy_omission_present INTEGER NOT NULL CHECK (policy_omission_present IN (0, 1)),
    report_hash TEXT NOT NULL UNIQUE CHECK (length(report_hash) = 64 AND report_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE hybrid_read_receipts (
    hybrid_read_receipt_id TEXT PRIMARY KEY CHECK (substr(hybrid_read_receipt_id, 1, 12) = 'hybrid_read_' AND length(hybrid_read_receipt_id) > 12),
    hybrid_processing_run_id TEXT NOT NULL REFERENCES hybrid_processing_runs(hybrid_processing_run_id) ON DELETE RESTRICT,
    hybrid_query_request_id TEXT NOT NULL REFERENCES hybrid_query_requests(hybrid_query_request_id) ON DELETE RESTRICT,
    item_count INTEGER NOT NULL CHECK (item_count >= 0),
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE hybrid_read_receipt_items (
    hybrid_read_receipt_id TEXT NOT NULL REFERENCES hybrid_read_receipts(hybrid_read_receipt_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    read_order INTEGER NOT NULL CHECK (read_order >= 0),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (hybrid_read_receipt_id, source_fragment_id),
    UNIQUE (hybrid_read_receipt_id, read_order)
);

CREATE TABLE hybrid_access_receipts (
    hybrid_access_receipt_id TEXT PRIMARY KEY CHECK (substr(hybrid_access_receipt_id, 1, 14) = 'hybrid_access_' AND length(hybrid_access_receipt_id) > 14),
    hybrid_processing_run_id TEXT NOT NULL REFERENCES hybrid_processing_runs(hybrid_processing_run_id) ON DELETE RESTRICT,
    hybrid_query_request_id TEXT NOT NULL REFERENCES hybrid_query_requests(hybrid_query_request_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    policy_hash TEXT NOT NULL CHECK (length(policy_hash) = 64 AND policy_hash NOT GLOB '*[^0-9a-f]*'),
    snapshot_hash TEXT NOT NULL CHECK (length(snapshot_hash) = 64 AND snapshot_hash NOT GLOB '*[^0-9a-f]*'),
    exclusion_hash TEXT NOT NULL CHECK (length(exclusion_hash) = 64 AND exclusion_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    retrieval_corpus_hash TEXT NOT NULL CHECK (length(retrieval_corpus_hash) = 64 AND retrieval_corpus_hash NOT GLOB '*[^0-9a-f]*'),
    policy_omission_present INTEGER NOT NULL CHECK (policy_omission_present IN (0, 1)),
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE hybrid_retrieval_receipts (
    hybrid_retrieval_receipt_id TEXT PRIMARY KEY CHECK (substr(hybrid_retrieval_receipt_id, 1, 17) = 'hybrid_retrieval_' AND length(hybrid_retrieval_receipt_id) > 17),
    hybrid_processing_run_id TEXT NOT NULL REFERENCES hybrid_processing_runs(hybrid_processing_run_id) ON DELETE RESTRICT,
    hybrid_query_request_id TEXT NOT NULL REFERENCES hybrid_query_requests(hybrid_query_request_id) ON DELETE RESTRICT,
    request_hash TEXT NOT NULL CHECK (length(request_hash) = 64 AND request_hash NOT GLOB '*[^0-9a-f]*'),
    exclusion_hash TEXT NOT NULL CHECK (length(exclusion_hash) = 64 AND exclusion_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    model_profile_hash TEXT NOT NULL REFERENCES embedding_model_profiles(profile_hash) ON DELETE RESTRICT,
    runtime_profile_hash TEXT NOT NULL REFERENCES model_runtime_profiles(profile_hash) ON DELETE RESTRICT,
    layer_profile_hash TEXT NOT NULL REFERENCES corpus_layer_profiles(profile_hash) ON DELETE RESTRICT,
    fusion_profile_hash TEXT NOT NULL REFERENCES fusion_profiles(profile_hash) ON DELETE RESTRICT,
    vector_generation_hash TEXT NOT NULL REFERENCES corpus_vector_generations(generation_hash) ON DELETE RESTRICT,
    fts_profile_version TEXT NOT NULL CHECK (length(fts_profile_version) BETWEEN 1 AND 128 AND instr(fts_profile_version, char(0)) = 0),
    fts_result_hash TEXT NOT NULL CHECK (length(fts_result_hash) = 64 AND fts_result_hash NOT GLOB '*[^0-9a-f]*'),
    query_vector_hash TEXT NOT NULL CHECK (length(query_vector_hash) = 64 AND query_vector_hash NOT GLOB '*[^0-9a-f]*'),
    reranker_profile_hash TEXT REFERENCES reranker_profiles(profile_hash) ON DELETE RESTRICT,
    receipt_hash TEXT NOT NULL UNIQUE CHECK (length(receipt_hash) = 64 AND receipt_hash NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE hybrid_retrieval_trace_items (
    hybrid_retrieval_receipt_id TEXT NOT NULL REFERENCES hybrid_retrieval_receipts(hybrid_retrieval_receipt_id) ON DELETE RESTRICT,
    channel TEXT NOT NULL CHECK (channel IN ('fts', 'dense', 'fused', 'reranked')),
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    rank INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 500),
    score_repr TEXT NOT NULL CHECK (length(score_repr) BETWEEN 1 AND 128 AND instr(score_repr, char(0)) = 0),
    PRIMARY KEY (hybrid_retrieval_receipt_id, channel, source_fragment_id),
    UNIQUE (hybrid_retrieval_receipt_id, channel, rank)
);

CREATE TABLE hybrid_evidence_packets (
    hybrid_evidence_packet_id TEXT PRIMARY KEY CHECK (substr(hybrid_evidence_packet_id, 1, 14) = 'hybrid_packet_' AND length(hybrid_evidence_packet_id) > 14),
    hybrid_query_request_id TEXT NOT NULL REFERENCES hybrid_query_requests(hybrid_query_request_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    read_receipt_hash TEXT NOT NULL REFERENCES hybrid_read_receipts(receipt_hash) ON DELETE RESTRICT,
    access_receipt_hash TEXT NOT NULL REFERENCES hybrid_access_receipts(receipt_hash) ON DELETE RESTRICT,
    coverage_report_hash TEXT NOT NULL REFERENCES hybrid_coverage_reports(report_hash) ON DELETE RESTRICT,
    retrieval_receipt_hash TEXT NOT NULL REFERENCES hybrid_retrieval_receipts(receipt_hash) ON DELETE RESTRICT,
    result_status TEXT NOT NULL CHECK (result_status IN ('evidence_found', 'no_evidence')),
    item_count INTEGER NOT NULL CHECK (item_count BETWEEN 0 AND 100),
    packet_hash TEXT NOT NULL UNIQUE CHECK (length(packet_hash) = 64 AND packet_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    CHECK ((result_status = 'evidence_found' AND item_count > 0) OR (result_status = 'no_evidence' AND item_count = 0))
);

CREATE TABLE hybrid_packet_items (
    hybrid_evidence_packet_id TEXT NOT NULL REFERENCES hybrid_evidence_packets(hybrid_evidence_packet_id) ON DELETE RESTRICT,
    rank INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 100),
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    source_family_id TEXT NOT NULL REFERENCES source_families(source_family_id) ON DELETE RESTRICT,
    layer TEXT NOT NULL CHECK (layer IN ('primary', 'derived', 'synthesis', 'unclassified')),
    PRIMARY KEY (hybrid_evidence_packet_id, source_fragment_id),
    UNIQUE (hybrid_evidence_packet_id, rank)
);

CREATE TABLE hybrid_run_observations (
    hybrid_run_observation_id TEXT PRIMARY KEY CHECK (substr(hybrid_run_observation_id, 1, 19) = 'hybrid_observation_' AND length(hybrid_run_observation_id) > 19),
    hybrid_processing_run_id TEXT NOT NULL UNIQUE REFERENCES hybrid_processing_runs(hybrid_processing_run_id) ON DELETE RESTRICT,
    hybrid_query_request_id TEXT NOT NULL REFERENCES hybrid_query_requests(hybrid_query_request_id) ON DELETE RESTRICT,
    cold_model_load_ns INTEGER NOT NULL CHECK (cold_model_load_ns >= 0),
    passage_embedding_ns INTEGER NOT NULL CHECK (passage_embedding_ns >= 0),
    query_embedding_ns INTEGER NOT NULL CHECK (query_embedding_ns >= 0),
    retrieval_ns INTEGER NOT NULL CHECK (retrieval_ns >= 0),
    reranker_ns INTEGER NOT NULL CHECK (reranker_ns >= 0),
    peak_rss_bytes INTEGER NOT NULL CHECK (peak_rss_bytes >= 0),
    cache_bytes INTEGER NOT NULL CHECK (cache_bytes >= 0),
    provider_calls INTEGER NOT NULL CHECK (provider_calls >= 0),
    observed_at TEXT NOT NULL
);

CREATE TABLE hybrid_run_artifacts (
    hybrid_processing_run_id TEXT PRIMARY KEY REFERENCES hybrid_processing_runs(hybrid_processing_run_id) ON DELETE RESTRICT,
    hybrid_coverage_report_id TEXT NOT NULL REFERENCES hybrid_coverage_reports(hybrid_coverage_report_id) ON DELETE RESTRICT,
    hybrid_read_receipt_id TEXT NOT NULL REFERENCES hybrid_read_receipts(hybrid_read_receipt_id) ON DELETE RESTRICT,
    hybrid_access_receipt_id TEXT NOT NULL REFERENCES hybrid_access_receipts(hybrid_access_receipt_id) ON DELETE RESTRICT,
    hybrid_retrieval_receipt_id TEXT NOT NULL REFERENCES hybrid_retrieval_receipts(hybrid_retrieval_receipt_id) ON DELETE RESTRICT,
    hybrid_evidence_packet_id TEXT NOT NULL REFERENCES hybrid_evidence_packets(hybrid_evidence_packet_id) ON DELETE RESTRICT
);

CREATE INDEX corpus_layer_profiles_scope_idx ON corpus_layer_profiles(library_id, corpus_snapshot_id, exclusion_hash, permitted_set_hash);
CREATE INDEX corpus_layer_items_source_idx ON corpus_layer_items(source_id, corpus_layer_profile_id);
CREATE INDEX corpus_vector_generations_scope_idx ON corpus_vector_generations(library_id, corpus_snapshot_id, access_policy_id, exclusion_hash, permitted_set_hash);
CREATE INDEX corpus_vector_items_fragment_idx ON corpus_vector_items(source_fragment_id, corpus_vector_generation_id);
CREATE INDEX hybrid_query_requests_scope_idx ON hybrid_query_requests(library_id, corpus_snapshot_id, access_policy_id, exclusion_hash);
CREATE INDEX hybrid_query_collections_collection_idx ON hybrid_query_collections(collection_id, hybrid_query_request_id);
CREATE INDEX hybrid_query_exclusions_target_idx ON hybrid_query_exclusions(target_type, target_id, hybrid_query_request_id);
CREATE INDEX hybrid_processing_runs_request_idx ON hybrid_processing_runs(hybrid_query_request_id, finished_at, hybrid_processing_run_id);
CREATE INDEX hybrid_read_receipt_items_fragment_idx ON hybrid_read_receipt_items(source_fragment_id, hybrid_read_receipt_id);
CREATE INDEX hybrid_retrieval_trace_fragment_idx ON hybrid_retrieval_trace_items(source_fragment_id, hybrid_retrieval_receipt_id, channel);
CREATE INDEX hybrid_packet_items_fragment_idx ON hybrid_packet_items(source_fragment_id, hybrid_evidence_packet_id);
CREATE INDEX hybrid_run_artifacts_coverage_idx ON hybrid_run_artifacts(hybrid_coverage_report_id);
CREATE INDEX hybrid_run_artifacts_read_idx ON hybrid_run_artifacts(hybrid_read_receipt_id);
CREATE INDEX hybrid_run_artifacts_access_idx ON hybrid_run_artifacts(hybrid_access_receipt_id);
CREATE INDEX hybrid_run_artifacts_retrieval_idx ON hybrid_run_artifacts(hybrid_retrieval_receipt_id);
CREATE INDEX hybrid_run_artifacts_packet_idx ON hybrid_run_artifacts(hybrid_evidence_packet_id);

CREATE TRIGGER corpus_layer_profiles_validate_insert
BEFORE INSERT ON corpus_layer_profiles
BEGIN
    SELECT RAISE(ABORT, 'CorpusLayerProfile snapshot must belong to its Library')
    WHERE NOT EXISTS (
        SELECT 1 FROM corpus_snapshots AS snapshot
        WHERE snapshot.corpus_snapshot_id = NEW.corpus_snapshot_id
          AND snapshot.library_id = NEW.library_id
    );
END;

CREATE TRIGGER corpus_layer_items_validate_insert
BEFORE INSERT ON corpus_layer_items
BEGIN
    SELECT RAISE(ABORT, 'CorpusLayerItem Source must belong to the profile Library')
    WHERE NOT EXISTS (
        SELECT 1
        FROM corpus_layer_profiles AS profile
        JOIN sources AS source ON source.source_id = NEW.source_id
        WHERE profile.corpus_layer_profile_id = NEW.corpus_layer_profile_id
          AND source.library_id = profile.library_id
    );
    SELECT RAISE(ABORT, 'CorpusLayerProfile item count exceeded')
    WHERE (SELECT COUNT(*) FROM corpus_layer_items WHERE corpus_layer_profile_id = NEW.corpus_layer_profile_id)
          >= (SELECT item_count FROM corpus_layer_profiles WHERE corpus_layer_profile_id = NEW.corpus_layer_profile_id);
END;

CREATE TRIGGER corpus_vector_generations_validate_insert
BEFORE INSERT ON corpus_vector_generations
BEGIN
    SELECT RAISE(ABORT, 'vector generation scope or profile is inconsistent')
    WHERE NOT EXISTS (
        SELECT 1
        FROM corpus_snapshots AS snapshot
        JOIN access_policies AS policy ON policy.access_policy_id = NEW.access_policy_id
        JOIN embedding_model_profiles AS model ON model.profile_hash = NEW.model_profile_hash
        JOIN model_runtime_profiles AS runtime ON runtime.profile_hash = NEW.runtime_profile_hash
        WHERE snapshot.corpus_snapshot_id = NEW.corpus_snapshot_id
          AND snapshot.library_id = NEW.library_id
          AND snapshot.manifest_hash = NEW.snapshot_hash
          AND policy.library_id = NEW.library_id
          AND policy.policy_hash = NEW.policy_hash
          AND model.dimensions = NEW.dimensions
    );
END;

CREATE TRIGGER corpus_vector_items_validate_insert
BEFORE INSERT ON corpus_vector_items
BEGIN
    SELECT RAISE(ABORT, 'vector item must match its exact snapshot Fragment and dimensions')
    WHERE NOT EXISTS (
        SELECT 1
        FROM corpus_vector_generations AS generation
        JOIN source_fragments AS fragment ON fragment.source_fragment_id = NEW.source_fragment_id
        JOIN source_versions AS version ON version.source_version_id = fragment.source_version_id
        JOIN snapshot_members AS member
          ON member.corpus_snapshot_id = generation.corpus_snapshot_id
         AND member.source_version_id = version.source_version_id
         AND member.membership_state = 'active'
        WHERE generation.corpus_vector_generation_id = NEW.corpus_vector_generation_id
          AND fragment.text_sha256 = NEW.text_sha256
          AND length(NEW.vector_blob) = generation.dimensions * 4
    );
    SELECT RAISE(ABORT, 'vector generation item count exceeded')
    WHERE (SELECT COUNT(*) FROM corpus_vector_items WHERE corpus_vector_generation_id = NEW.corpus_vector_generation_id)
          >= (SELECT item_count FROM corpus_vector_generations WHERE corpus_vector_generation_id = NEW.corpus_vector_generation_id);
END;

CREATE TRIGGER hybrid_query_requests_validate_insert
BEFORE INSERT ON hybrid_query_requests
BEGIN
    SELECT RAISE(ABORT, 'hybrid request scope or profile is inconsistent')
    WHERE NOT EXISTS (
        SELECT 1
        FROM corpus_snapshots AS snapshot
        JOIN access_policies AS policy ON policy.access_policy_id = NEW.access_policy_id
        JOIN corpus_layer_profiles AS layer ON layer.profile_hash = NEW.layer_profile_hash
        JOIN embedding_model_profiles AS model ON model.profile_hash = NEW.model_profile_hash
        JOIN model_runtime_profiles AS runtime ON runtime.profile_hash = NEW.runtime_profile_hash
        JOIN fusion_profiles AS fusion ON fusion.profile_hash = NEW.fusion_profile_hash
        WHERE snapshot.corpus_snapshot_id = NEW.corpus_snapshot_id
          AND snapshot.library_id = NEW.library_id
          AND policy.library_id = NEW.library_id
          AND layer.library_id = NEW.library_id
          AND layer.corpus_snapshot_id = NEW.corpus_snapshot_id
          AND layer.exclusion_hash = NEW.exclusion_hash
          AND (NEW.reranker_profile_hash IS NULL OR EXISTS (
              SELECT 1 FROM reranker_profiles AS reranker
              WHERE reranker.profile_hash = NEW.reranker_profile_hash
          ))
    );
END;

CREATE TRIGGER hybrid_query_collections_validate_insert
BEFORE INSERT ON hybrid_query_collections
BEGIN
    SELECT RAISE(ABORT, 'hybrid request Collection must belong to its Library')
    WHERE NOT EXISTS (
        SELECT 1
        FROM hybrid_query_requests AS request
        JOIN collections AS collection ON collection.collection_id = NEW.collection_id
        WHERE request.hybrid_query_request_id = NEW.hybrid_query_request_id
          AND collection.library_id = request.library_id
    );
    SELECT RAISE(ABORT, 'hybrid request Collection count exceeded')
    WHERE (SELECT COUNT(*) FROM hybrid_query_collections WHERE hybrid_query_request_id = NEW.hybrid_query_request_id)
          >= (SELECT collection_count FROM hybrid_query_requests WHERE hybrid_query_request_id = NEW.hybrid_query_request_id);
END;

CREATE TRIGGER hybrid_query_exclusions_validate_insert
BEFORE INSERT ON hybrid_query_exclusions
BEGIN
    SELECT RAISE(ABORT, 'hybrid request exclusion must identify an object in its Library')
    WHERE NOT EXISTS (
        SELECT 1 FROM hybrid_query_requests AS request
        JOIN sources AS source ON NEW.target_type = 'source' AND source.source_id = NEW.target_id AND source.library_id = request.library_id
        WHERE request.hybrid_query_request_id = NEW.hybrid_query_request_id
        UNION ALL
        SELECT 1 FROM hybrid_query_requests AS request
        JOIN source_families AS family ON NEW.target_type = 'source_family' AND family.source_family_id = NEW.target_id AND family.library_id = request.library_id
        WHERE request.hybrid_query_request_id = NEW.hybrid_query_request_id
        UNION ALL
        SELECT 1 FROM hybrid_query_requests AS request
        JOIN source_fragments AS fragment ON NEW.target_type = 'source_fragment' AND fragment.source_fragment_id = NEW.target_id
        JOIN source_versions AS version ON version.source_version_id = fragment.source_version_id
        JOIN sources AS source ON source.source_id = version.source_id AND source.library_id = request.library_id
        WHERE request.hybrid_query_request_id = NEW.hybrid_query_request_id
    );
    SELECT RAISE(ABORT, 'hybrid request exclusion count exceeded')
    WHERE (SELECT COUNT(*) FROM hybrid_query_exclusions WHERE hybrid_query_request_id = NEW.hybrid_query_request_id)
          >= (SELECT exclusion_count FROM hybrid_query_requests WHERE hybrid_query_request_id = NEW.hybrid_query_request_id);
END;

CREATE TRIGGER hybrid_processing_runs_validate_insert
BEFORE INSERT ON hybrid_processing_runs
BEGIN
    SELECT RAISE(ABORT, 'hybrid run request, scope, generation, or child counts are inconsistent')
    WHERE NOT EXISTS (
        SELECT 1
        FROM hybrid_query_requests AS request
        JOIN corpus_layer_profiles AS layer ON layer.profile_hash = request.layer_profile_hash
        JOIN corpus_vector_generations AS generation ON generation.corpus_vector_generation_id = NEW.corpus_vector_generation_id
        WHERE request.hybrid_query_request_id = NEW.hybrid_query_request_id
          AND generation.library_id = request.library_id
          AND generation.corpus_snapshot_id = request.corpus_snapshot_id
          AND generation.access_policy_id = request.access_policy_id
          AND generation.exclusion_hash = request.exclusion_hash
          AND generation.permitted_set_hash = layer.permitted_set_hash
          AND generation.model_profile_hash = request.model_profile_hash
          AND generation.runtime_profile_hash = request.runtime_profile_hash
          AND request.collection_count = (SELECT COUNT(*) FROM hybrid_query_collections WHERE hybrid_query_request_id = request.hybrid_query_request_id)
          AND request.exclusion_count = (SELECT COUNT(*) FROM hybrid_query_exclusions WHERE hybrid_query_request_id = request.hybrid_query_request_id)
          AND generation.item_count = (SELECT COUNT(*) FROM corpus_vector_items WHERE corpus_vector_generation_id = generation.corpus_vector_generation_id)
          AND layer.item_count = (SELECT COUNT(*) FROM corpus_layer_items WHERE corpus_layer_profile_id = layer.corpus_layer_profile_id)
    );
END;

CREATE TRIGGER hybrid_coverage_reports_validate_insert
BEFORE INSERT ON hybrid_coverage_reports
BEGIN
    SELECT RAISE(ABORT, 'hybrid coverage report must match its run request')
    WHERE NOT EXISTS (
        SELECT 1 FROM hybrid_processing_runs AS run
        WHERE run.hybrid_processing_run_id = NEW.hybrid_processing_run_id
          AND run.hybrid_query_request_id = NEW.hybrid_query_request_id
    );
END;

CREATE TRIGGER hybrid_read_receipts_validate_insert
BEFORE INSERT ON hybrid_read_receipts
BEGIN
    SELECT RAISE(ABORT, 'hybrid read receipt must match its run request')
    WHERE NOT EXISTS (
        SELECT 1 FROM hybrid_processing_runs AS run
        WHERE run.hybrid_processing_run_id = NEW.hybrid_processing_run_id
          AND run.hybrid_query_request_id = NEW.hybrid_query_request_id
    );
END;

CREATE TRIGGER hybrid_read_receipt_items_validate_insert
BEFORE INSERT ON hybrid_read_receipt_items
BEGIN
    SELECT RAISE(ABORT, 'hybrid read item must be an exact Fragment in the run generation')
    WHERE NOT EXISTS (
        SELECT 1
        FROM hybrid_read_receipts AS receipt
        JOIN hybrid_processing_runs AS run ON run.hybrid_processing_run_id = receipt.hybrid_processing_run_id
        JOIN corpus_vector_items AS vector
          ON vector.corpus_vector_generation_id = run.corpus_vector_generation_id
         AND vector.source_fragment_id = NEW.source_fragment_id
        WHERE receipt.hybrid_read_receipt_id = NEW.hybrid_read_receipt_id
          AND vector.text_sha256 = NEW.text_sha256
    );
    SELECT RAISE(ABORT, 'hybrid read receipt item count exceeded')
    WHERE (SELECT COUNT(*) FROM hybrid_read_receipt_items WHERE hybrid_read_receipt_id = NEW.hybrid_read_receipt_id)
          >= (SELECT item_count FROM hybrid_read_receipts WHERE hybrid_read_receipt_id = NEW.hybrid_read_receipt_id);
END;

CREATE TRIGGER hybrid_access_receipts_validate_insert
BEFORE INSERT ON hybrid_access_receipts
BEGIN
    SELECT RAISE(ABORT, 'hybrid access receipt does not close over the exact authorized generation')
    WHERE NOT EXISTS (
        SELECT 1
        FROM hybrid_processing_runs AS run
        JOIN hybrid_query_requests AS request ON request.hybrid_query_request_id = run.hybrid_query_request_id
        JOIN corpus_layer_profiles AS layer ON layer.profile_hash = request.layer_profile_hash
        JOIN corpus_vector_generations AS generation ON generation.corpus_vector_generation_id = run.corpus_vector_generation_id
        WHERE run.hybrid_processing_run_id = NEW.hybrid_processing_run_id
          AND request.hybrid_query_request_id = NEW.hybrid_query_request_id
          AND request.access_policy_id = NEW.access_policy_id
          AND generation.policy_hash = NEW.policy_hash
          AND generation.snapshot_hash = NEW.snapshot_hash
          AND generation.exclusion_hash = NEW.exclusion_hash
          AND generation.permitted_set_hash = NEW.permitted_set_hash
          AND layer.permitted_set_hash = NEW.permitted_set_hash
    );
END;

CREATE TRIGGER hybrid_retrieval_receipts_validate_insert
BEFORE INSERT ON hybrid_retrieval_receipts
BEGIN
    SELECT RAISE(ABORT, 'hybrid retrieval receipt does not match the exact request and generation')
    WHERE NOT EXISTS (
        SELECT 1
        FROM hybrid_processing_runs AS run
        JOIN hybrid_query_requests AS request ON request.hybrid_query_request_id = run.hybrid_query_request_id
        JOIN corpus_layer_profiles AS layer ON layer.profile_hash = request.layer_profile_hash
        JOIN corpus_vector_generations AS generation ON generation.corpus_vector_generation_id = run.corpus_vector_generation_id
        WHERE run.hybrid_processing_run_id = NEW.hybrid_processing_run_id
          AND request.hybrid_query_request_id = NEW.hybrid_query_request_id
          AND request.request_hash = NEW.request_hash
          AND request.exclusion_hash = NEW.exclusion_hash
          AND layer.permitted_set_hash = NEW.permitted_set_hash
          AND request.model_profile_hash = NEW.model_profile_hash
          AND request.runtime_profile_hash = NEW.runtime_profile_hash
          AND request.layer_profile_hash = NEW.layer_profile_hash
          AND request.fusion_profile_hash = NEW.fusion_profile_hash
          AND request.reranker_profile_hash IS NEW.reranker_profile_hash
          AND generation.generation_hash = NEW.vector_generation_hash
    );
END;

CREATE TRIGGER hybrid_retrieval_trace_items_validate_insert
BEFORE INSERT ON hybrid_retrieval_trace_items
BEGIN
    SELECT RAISE(ABORT, 'hybrid trace item must be an exact Fragment in the run generation')
    WHERE NOT EXISTS (
        SELECT 1
        FROM hybrid_retrieval_receipts AS receipt
        JOIN hybrid_processing_runs AS run ON run.hybrid_processing_run_id = receipt.hybrid_processing_run_id
        JOIN corpus_vector_items AS vector
          ON vector.corpus_vector_generation_id = run.corpus_vector_generation_id
         AND vector.source_fragment_id = NEW.source_fragment_id
        JOIN source_fragments AS fragment ON fragment.source_fragment_id = vector.source_fragment_id
        JOIN source_versions AS version ON version.source_version_id = fragment.source_version_id
        WHERE receipt.hybrid_retrieval_receipt_id = NEW.hybrid_retrieval_receipt_id
          AND version.source_id = NEW.source_id
          AND (NEW.channel <> 'reranked' OR receipt.reranker_profile_hash IS NOT NULL)
    );
    SELECT RAISE(ABORT, 'fused trace may contain only FTS or dense candidates')
    WHERE NEW.channel = 'fused' AND NOT EXISTS (
        SELECT 1 FROM hybrid_retrieval_trace_items AS input
        WHERE input.hybrid_retrieval_receipt_id = NEW.hybrid_retrieval_receipt_id
          AND input.source_fragment_id = NEW.source_fragment_id
          AND input.channel IN ('fts', 'dense')
    );
    SELECT RAISE(ABORT, 'reranked trace may contain only fused candidates')
    WHERE NEW.channel = 'reranked' AND NOT EXISTS (
        SELECT 1 FROM hybrid_retrieval_trace_items AS fused
        WHERE fused.hybrid_retrieval_receipt_id = NEW.hybrid_retrieval_receipt_id
          AND fused.source_fragment_id = NEW.source_fragment_id
          AND fused.channel = 'fused'
    );
END;

CREATE TRIGGER hybrid_evidence_packets_validate_insert
BEFORE INSERT ON hybrid_evidence_packets
BEGIN
    SELECT RAISE(ABORT, 'hybrid packet receipts do not belong to one exact request and scope')
    WHERE NOT EXISTS (
        SELECT 1
        FROM hybrid_query_requests AS request
        JOIN hybrid_read_receipts AS read ON read.receipt_hash = NEW.read_receipt_hash
        JOIN hybrid_access_receipts AS access ON access.receipt_hash = NEW.access_receipt_hash
        JOIN hybrid_coverage_reports AS coverage ON coverage.report_hash = NEW.coverage_report_hash
        JOIN hybrid_retrieval_receipts AS retrieval ON retrieval.receipt_hash = NEW.retrieval_receipt_hash
        WHERE request.hybrid_query_request_id = NEW.hybrid_query_request_id
          AND request.corpus_snapshot_id = NEW.corpus_snapshot_id
          AND request.access_policy_id = NEW.access_policy_id
          AND read.hybrid_query_request_id = request.hybrid_query_request_id
          AND access.hybrid_query_request_id = request.hybrid_query_request_id
          AND coverage.hybrid_query_request_id = request.hybrid_query_request_id
          AND retrieval.hybrid_query_request_id = request.hybrid_query_request_id
          AND read.hybrid_processing_run_id = access.hybrid_processing_run_id
          AND read.hybrid_processing_run_id = coverage.hybrid_processing_run_id
          AND read.hybrid_processing_run_id = retrieval.hybrid_processing_run_id
    );
END;

CREATE TRIGGER hybrid_packet_items_validate_insert
BEFORE INSERT ON hybrid_packet_items
BEGIN
    SELECT RAISE(ABORT, 'hybrid packet item violates read, retrieval, family, or layer closure')
    WHERE NOT EXISTS (
        SELECT 1
        FROM hybrid_evidence_packets AS packet
        JOIN hybrid_query_requests AS request ON request.hybrid_query_request_id = packet.hybrid_query_request_id
        JOIN corpus_layer_profiles AS layer_profile ON layer_profile.profile_hash = request.layer_profile_hash
        JOIN corpus_layer_items AS layer_item
          ON layer_item.corpus_layer_profile_id = layer_profile.corpus_layer_profile_id
         AND layer_item.source_id = NEW.source_id
         AND layer_item.layer = NEW.layer
        JOIN hybrid_read_receipts AS read ON read.receipt_hash = packet.read_receipt_hash
        JOIN hybrid_read_receipt_items AS read_item
          ON read_item.hybrid_read_receipt_id = read.hybrid_read_receipt_id
         AND read_item.source_fragment_id = NEW.source_fragment_id
        JOIN hybrid_retrieval_receipts AS retrieval ON retrieval.receipt_hash = packet.retrieval_receipt_hash
        JOIN hybrid_retrieval_trace_items AS trace
          ON trace.hybrid_retrieval_receipt_id = retrieval.hybrid_retrieval_receipt_id
         AND trace.source_fragment_id = NEW.source_fragment_id
         AND trace.source_id = NEW.source_id
         AND trace.channel = CASE WHEN retrieval.reranker_profile_hash IS NULL THEN 'fused' ELSE 'reranked' END
        JOIN source_family_members AS family
          ON family.source_family_id = NEW.source_family_id
         AND family.source_id = NEW.source_id
        WHERE packet.hybrid_evidence_packet_id = NEW.hybrid_evidence_packet_id
    );
    SELECT RAISE(ABORT, 'hybrid packet item count exceeded')
    WHERE (SELECT COUNT(*) FROM hybrid_packet_items WHERE hybrid_evidence_packet_id = NEW.hybrid_evidence_packet_id)
          >= (SELECT item_count FROM hybrid_evidence_packets WHERE hybrid_evidence_packet_id = NEW.hybrid_evidence_packet_id);
END;

CREATE TRIGGER hybrid_run_observations_validate_insert
BEFORE INSERT ON hybrid_run_observations
BEGIN
    SELECT RAISE(ABORT, 'hybrid observation must match its run request')
    WHERE NOT EXISTS (
        SELECT 1 FROM hybrid_processing_runs AS run
        WHERE run.hybrid_processing_run_id = NEW.hybrid_processing_run_id
          AND run.hybrid_query_request_id = NEW.hybrid_query_request_id
    );
END;

CREATE TRIGGER hybrid_run_artifacts_validate_insert
BEFORE INSERT ON hybrid_run_artifacts
BEGIN
    SELECT RAISE(ABORT, 'hybrid run artifacts do not form one complete immutable run')
    WHERE NOT EXISTS (
        SELECT 1
        FROM hybrid_processing_runs AS run
        JOIN corpus_vector_generations AS generation
          ON generation.corpus_vector_generation_id = run.corpus_vector_generation_id
        JOIN hybrid_coverage_reports AS coverage ON coverage.hybrid_coverage_report_id = NEW.hybrid_coverage_report_id
        JOIN hybrid_processing_runs AS coverage_run
          ON coverage_run.hybrid_processing_run_id = coverage.hybrid_processing_run_id
        JOIN hybrid_read_receipts AS read ON read.hybrid_read_receipt_id = NEW.hybrid_read_receipt_id
        JOIN hybrid_processing_runs AS read_run
          ON read_run.hybrid_processing_run_id = read.hybrid_processing_run_id
        JOIN hybrid_access_receipts AS access ON access.hybrid_access_receipt_id = NEW.hybrid_access_receipt_id
        JOIN hybrid_processing_runs AS access_run
          ON access_run.hybrid_processing_run_id = access.hybrid_processing_run_id
        JOIN hybrid_retrieval_receipts AS retrieval ON retrieval.hybrid_retrieval_receipt_id = NEW.hybrid_retrieval_receipt_id
        JOIN hybrid_processing_runs AS retrieval_run
          ON retrieval_run.hybrid_processing_run_id = retrieval.hybrid_processing_run_id
        JOIN hybrid_evidence_packets AS packet ON packet.hybrid_evidence_packet_id = NEW.hybrid_evidence_packet_id
        WHERE run.hybrid_processing_run_id = NEW.hybrid_processing_run_id
          AND run.status = 'succeeded'
          AND run.kind IN ('recall', 'replay')
          AND run.output_hash = packet.packet_hash
          AND coverage_run.status = 'succeeded'
          AND read_run.status = 'succeeded'
          AND access_run.status = 'succeeded'
          AND retrieval_run.status = 'succeeded'
          AND coverage_run.kind IN ('recall', 'replay')
          AND read_run.kind IN ('recall', 'replay')
          AND access_run.kind IN ('recall', 'replay')
          AND retrieval_run.kind IN ('recall', 'replay')
          AND coverage.hybrid_query_request_id = run.hybrid_query_request_id
          AND read.hybrid_query_request_id = run.hybrid_query_request_id
          AND access.hybrid_query_request_id = run.hybrid_query_request_id
          AND retrieval.hybrid_query_request_id = run.hybrid_query_request_id
          AND coverage_run.hybrid_query_request_id = run.hybrid_query_request_id
          AND read_run.hybrid_query_request_id = run.hybrid_query_request_id
          AND access_run.hybrid_query_request_id = run.hybrid_query_request_id
          AND retrieval_run.hybrid_query_request_id = run.hybrid_query_request_id
          AND coverage_run.corpus_vector_generation_id = run.corpus_vector_generation_id
          AND read_run.corpus_vector_generation_id = run.corpus_vector_generation_id
          AND access_run.corpus_vector_generation_id = run.corpus_vector_generation_id
          AND retrieval_run.corpus_vector_generation_id = run.corpus_vector_generation_id
          AND retrieval.vector_generation_hash = generation.generation_hash
          AND packet.hybrid_query_request_id = run.hybrid_query_request_id
          AND packet.coverage_report_hash = coverage.report_hash
          AND packet.read_receipt_hash = read.receipt_hash
          AND packet.access_receipt_hash = access.receipt_hash
          AND packet.retrieval_receipt_hash = retrieval.receipt_hash
          AND read.item_count = (SELECT COUNT(*) FROM hybrid_read_receipt_items WHERE hybrid_read_receipt_id = read.hybrid_read_receipt_id)
          AND packet.item_count = (SELECT COUNT(*) FROM hybrid_packet_items WHERE hybrid_evidence_packet_id = packet.hybrid_evidence_packet_id)
          AND ((retrieval.reranker_profile_hash IS NULL AND NOT EXISTS (
                  SELECT 1 FROM hybrid_retrieval_trace_items
                  WHERE hybrid_retrieval_receipt_id = retrieval.hybrid_retrieval_receipt_id AND channel = 'reranked'
              )) OR (retrieval.reranker_profile_hash IS NOT NULL AND
                  (SELECT COUNT(*) FROM hybrid_retrieval_trace_items WHERE hybrid_retrieval_receipt_id = retrieval.hybrid_retrieval_receipt_id AND channel = 'reranked') =
                  (SELECT COUNT(*) FROM hybrid_retrieval_trace_items WHERE hybrid_retrieval_receipt_id = retrieval.hybrid_retrieval_receipt_id AND channel = 'fused')
              ))
    );
END;

CREATE TRIGGER embedding_model_profiles_no_update BEFORE UPDATE ON embedding_model_profiles BEGIN SELECT RAISE(ABORT, 'embedding_model_profiles is append-only'); END;
CREATE TRIGGER embedding_model_profiles_no_delete BEFORE DELETE ON embedding_model_profiles BEGIN SELECT RAISE(ABORT, 'embedding_model_profiles is append-only'); END;
CREATE TRIGGER model_runtime_profiles_no_update BEFORE UPDATE ON model_runtime_profiles BEGIN SELECT RAISE(ABORT, 'model_runtime_profiles is append-only'); END;
CREATE TRIGGER model_runtime_profiles_no_delete BEFORE DELETE ON model_runtime_profiles BEGIN SELECT RAISE(ABORT, 'model_runtime_profiles is append-only'); END;
CREATE TRIGGER fusion_profiles_no_update BEFORE UPDATE ON fusion_profiles BEGIN SELECT RAISE(ABORT, 'fusion_profiles is append-only'); END;
CREATE TRIGGER fusion_profiles_no_delete BEFORE DELETE ON fusion_profiles BEGIN SELECT RAISE(ABORT, 'fusion_profiles is append-only'); END;
CREATE TRIGGER reranker_profiles_no_update BEFORE UPDATE ON reranker_profiles BEGIN SELECT RAISE(ABORT, 'reranker_profiles is append-only'); END;
CREATE TRIGGER reranker_profiles_no_delete BEFORE DELETE ON reranker_profiles BEGIN SELECT RAISE(ABORT, 'reranker_profiles is append-only'); END;
CREATE TRIGGER corpus_layer_profiles_no_update BEFORE UPDATE ON corpus_layer_profiles BEGIN SELECT RAISE(ABORT, 'corpus_layer_profiles is append-only'); END;
CREATE TRIGGER corpus_layer_profiles_no_delete BEFORE DELETE ON corpus_layer_profiles BEGIN SELECT RAISE(ABORT, 'corpus_layer_profiles is append-only'); END;
CREATE TRIGGER corpus_layer_items_no_update BEFORE UPDATE ON corpus_layer_items BEGIN SELECT RAISE(ABORT, 'corpus_layer_items is append-only'); END;
CREATE TRIGGER corpus_layer_items_no_delete BEFORE DELETE ON corpus_layer_items BEGIN SELECT RAISE(ABORT, 'corpus_layer_items is append-only'); END;
CREATE TRIGGER corpus_vector_generations_no_update BEFORE UPDATE ON corpus_vector_generations BEGIN SELECT RAISE(ABORT, 'corpus_vector_generations is append-only'); END;
CREATE TRIGGER corpus_vector_generations_no_delete BEFORE DELETE ON corpus_vector_generations BEGIN SELECT RAISE(ABORT, 'corpus_vector_generations is append-only'); END;
CREATE TRIGGER corpus_vector_items_no_update BEFORE UPDATE ON corpus_vector_items BEGIN SELECT RAISE(ABORT, 'corpus_vector_items is append-only'); END;
CREATE TRIGGER corpus_vector_items_no_delete BEFORE DELETE ON corpus_vector_items BEGIN SELECT RAISE(ABORT, 'corpus_vector_items is append-only'); END;
CREATE TRIGGER hybrid_query_requests_no_update BEFORE UPDATE ON hybrid_query_requests BEGIN SELECT RAISE(ABORT, 'hybrid_query_requests is append-only'); END;
CREATE TRIGGER hybrid_query_requests_no_delete BEFORE DELETE ON hybrid_query_requests BEGIN SELECT RAISE(ABORT, 'hybrid_query_requests is append-only'); END;
CREATE TRIGGER hybrid_query_collections_no_update BEFORE UPDATE ON hybrid_query_collections BEGIN SELECT RAISE(ABORT, 'hybrid_query_collections is append-only'); END;
CREATE TRIGGER hybrid_query_collections_no_delete BEFORE DELETE ON hybrid_query_collections BEGIN SELECT RAISE(ABORT, 'hybrid_query_collections is append-only'); END;
CREATE TRIGGER hybrid_query_exclusions_no_update BEFORE UPDATE ON hybrid_query_exclusions BEGIN SELECT RAISE(ABORT, 'hybrid_query_exclusions is append-only'); END;
CREATE TRIGGER hybrid_query_exclusions_no_delete BEFORE DELETE ON hybrid_query_exclusions BEGIN SELECT RAISE(ABORT, 'hybrid_query_exclusions is append-only'); END;
CREATE TRIGGER hybrid_processing_runs_no_update BEFORE UPDATE ON hybrid_processing_runs BEGIN SELECT RAISE(ABORT, 'hybrid_processing_runs is append-only'); END;
CREATE TRIGGER hybrid_processing_runs_no_delete BEFORE DELETE ON hybrid_processing_runs BEGIN SELECT RAISE(ABORT, 'hybrid_processing_runs is append-only'); END;
CREATE TRIGGER hybrid_coverage_reports_no_update BEFORE UPDATE ON hybrid_coverage_reports BEGIN SELECT RAISE(ABORT, 'hybrid_coverage_reports is append-only'); END;
CREATE TRIGGER hybrid_coverage_reports_no_delete BEFORE DELETE ON hybrid_coverage_reports BEGIN SELECT RAISE(ABORT, 'hybrid_coverage_reports is append-only'); END;
CREATE TRIGGER hybrid_read_receipts_no_update BEFORE UPDATE ON hybrid_read_receipts BEGIN SELECT RAISE(ABORT, 'hybrid_read_receipts is append-only'); END;
CREATE TRIGGER hybrid_read_receipts_no_delete BEFORE DELETE ON hybrid_read_receipts BEGIN SELECT RAISE(ABORT, 'hybrid_read_receipts is append-only'); END;
CREATE TRIGGER hybrid_read_receipt_items_no_update BEFORE UPDATE ON hybrid_read_receipt_items BEGIN SELECT RAISE(ABORT, 'hybrid_read_receipt_items is append-only'); END;
CREATE TRIGGER hybrid_read_receipt_items_no_delete BEFORE DELETE ON hybrid_read_receipt_items BEGIN SELECT RAISE(ABORT, 'hybrid_read_receipt_items is append-only'); END;
CREATE TRIGGER hybrid_access_receipts_no_update BEFORE UPDATE ON hybrid_access_receipts BEGIN SELECT RAISE(ABORT, 'hybrid_access_receipts is append-only'); END;
CREATE TRIGGER hybrid_access_receipts_no_delete BEFORE DELETE ON hybrid_access_receipts BEGIN SELECT RAISE(ABORT, 'hybrid_access_receipts is append-only'); END;
CREATE TRIGGER hybrid_retrieval_receipts_no_update BEFORE UPDATE ON hybrid_retrieval_receipts BEGIN SELECT RAISE(ABORT, 'hybrid_retrieval_receipts is append-only'); END;
CREATE TRIGGER hybrid_retrieval_receipts_no_delete BEFORE DELETE ON hybrid_retrieval_receipts BEGIN SELECT RAISE(ABORT, 'hybrid_retrieval_receipts is append-only'); END;
CREATE TRIGGER hybrid_retrieval_trace_items_no_update BEFORE UPDATE ON hybrid_retrieval_trace_items BEGIN SELECT RAISE(ABORT, 'hybrid_retrieval_trace_items is append-only'); END;
CREATE TRIGGER hybrid_retrieval_trace_items_no_delete BEFORE DELETE ON hybrid_retrieval_trace_items BEGIN SELECT RAISE(ABORT, 'hybrid_retrieval_trace_items is append-only'); END;
CREATE TRIGGER hybrid_evidence_packets_no_update BEFORE UPDATE ON hybrid_evidence_packets BEGIN SELECT RAISE(ABORT, 'hybrid_evidence_packets is append-only'); END;
CREATE TRIGGER hybrid_evidence_packets_no_delete BEFORE DELETE ON hybrid_evidence_packets BEGIN SELECT RAISE(ABORT, 'hybrid_evidence_packets is append-only'); END;
CREATE TRIGGER hybrid_packet_items_no_update BEFORE UPDATE ON hybrid_packet_items BEGIN SELECT RAISE(ABORT, 'hybrid_packet_items is append-only'); END;
CREATE TRIGGER hybrid_packet_items_no_delete BEFORE DELETE ON hybrid_packet_items BEGIN SELECT RAISE(ABORT, 'hybrid_packet_items is append-only'); END;
CREATE TRIGGER hybrid_run_observations_no_update BEFORE UPDATE ON hybrid_run_observations BEGIN SELECT RAISE(ABORT, 'hybrid_run_observations is append-only'); END;
CREATE TRIGGER hybrid_run_observations_no_delete BEFORE DELETE ON hybrid_run_observations BEGIN SELECT RAISE(ABORT, 'hybrid_run_observations is append-only'); END;
CREATE TRIGGER hybrid_run_artifacts_no_update BEFORE UPDATE ON hybrid_run_artifacts BEGIN SELECT RAISE(ABORT, 'hybrid_run_artifacts is append-only'); END;
CREATE TRIGGER hybrid_run_artifacts_no_delete BEFORE DELETE ON hybrid_run_artifacts BEGIN SELECT RAISE(ABORT, 'hybrid_run_artifacts is append-only'); END;
