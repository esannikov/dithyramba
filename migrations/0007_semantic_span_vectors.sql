-- Dithyramba P7 SemanticSpan vector-v2 persistence.
--
-- The database enforces immutable structural closure. Canonical payload hashes
-- and vector-blob SHA-256 values are re-computed transactionally by the
-- persistence layer because stock SQLite has no canonical-JSON or SHA-256
-- function and cannot defer those application-level checks until commit.

CREATE TABLE source_fragment_text_metrics (
    source_fragment_id TEXT PRIMARY KEY REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    text_char_count INTEGER NOT NULL CHECK (text_char_count BETWEEN 1 AND 10000000),
    UNIQUE (source_fragment_id, text_sha256, text_char_count)
);

INSERT INTO source_fragment_text_metrics(
    source_fragment_id, text_sha256, text_char_count
)
SELECT source_fragment_id, text_sha256, length(text)
FROM source_fragments;

CREATE TABLE semantic_span_profiles (
    semantic_span_profile_id TEXT PRIMARY KEY,
    profile_version TEXT NOT NULL CHECK (profile_version = '1.0'),
    model_profile_hash TEXT NOT NULL REFERENCES embedding_model_profiles(profile_hash) ON DELETE RESTRICT,
    runtime_profile_hash TEXT NOT NULL REFERENCES model_runtime_profiles(profile_hash) ON DELETE RESTRICT,
    provisioning_receipt_hash TEXT NOT NULL CHECK (length(provisioning_receipt_hash) = 64 AND provisioning_receipt_hash NOT GLOB '*[^0-9a-f]*'),
    max_prepared_tokens INTEGER NOT NULL CHECK (max_prepared_tokens BETWEEN 1 AND 32768),
    overlap_content_tokens INTEGER NOT NULL CHECK (overlap_content_tokens >= 0 AND overlap_content_tokens < max_prepared_tokens),
    passage_prefix TEXT NOT NULL CHECK (length(passage_prefix) <= 128 AND instr(passage_prefix, char(0)) = 0),
    profile_hash TEXT NOT NULL UNIQUE CHECK (length(profile_hash) = 64 AND profile_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    CHECK (semantic_span_profile_id = 'semantic_span_profile_' || substr(profile_hash, 1, 32)),
    UNIQUE (semantic_span_profile_id, profile_hash),
    UNIQUE (
        profile_version, model_profile_hash, runtime_profile_hash,
        provisioning_receipt_hash, max_prepared_tokens,
        overlap_content_tokens, passage_prefix
    )
);

CREATE TABLE semantic_span_plans (
    semantic_span_plan_id TEXT PRIMARY KEY,
    plan_hash TEXT NOT NULL UNIQUE CHECK (length(plan_hash) = 64 AND plan_hash NOT GLOB '*[^0-9a-f]*'),
    semantic_span_profile_hash TEXT NOT NULL REFERENCES semantic_span_profiles(profile_hash) ON DELETE RESTRICT,
    parent_manifest_hash TEXT NOT NULL CHECK (length(parent_manifest_hash) = 64 AND parent_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    span_closure_hash TEXT NOT NULL CHECK (length(span_closure_hash) = 64 AND span_closure_hash NOT GLOB '*[^0-9a-f]*'),
    parent_count INTEGER NOT NULL CHECK (parent_count BETWEEN 0 AND 10000000),
    span_count INTEGER NOT NULL CHECK (span_count BETWEEN 0 AND 100000000),
    created_at TEXT NOT NULL,
    CHECK (semantic_span_plan_id = 'semantic_span_plan_' || substr(plan_hash, 1, 32)),
    CHECK (
        (parent_count = 0 AND span_count = 0) OR
        (parent_count > 0 AND span_count >= parent_count)
    ),
    UNIQUE (semantic_span_plan_id, plan_hash),
    UNIQUE (
        semantic_span_profile_hash, parent_manifest_hash,
        span_closure_hash, parent_count, span_count
    )
);

CREATE TABLE semantic_span_parents (
    semantic_span_plan_id TEXT NOT NULL REFERENCES semantic_span_plans(semantic_span_plan_id) ON DELETE RESTRICT,
    source_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    source_version_id TEXT NOT NULL REFERENCES source_versions(source_version_id) ON DELETE RESTRICT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'),
    text_char_count INTEGER NOT NULL CHECK (text_char_count BETWEEN 1 AND 10000000),
    content_token_count INTEGER NOT NULL CHECK (content_token_count BETWEEN 0 AND 10000000),
    full_prepared_token_count INTEGER NOT NULL CHECK (full_prepared_token_count BETWEEN 1 AND 10000000),
    content_offsets_hash TEXT NOT NULL CHECK (length(content_offsets_hash) = 64 AND content_offsets_hash NOT GLOB '*[^0-9a-f]*'),
    span_count INTEGER NOT NULL CHECK (span_count BETWEEN 1 AND 10000000),
    PRIMARY KEY (semantic_span_plan_id, source_fragment_id),
    UNIQUE (semantic_span_plan_id, source_version_id, ordinal),
    UNIQUE (
        semantic_span_plan_id, source_fragment_id,
        text_sha256, ordinal
    )
);

CREATE TABLE semantic_spans (
    semantic_span_plan_id TEXT NOT NULL REFERENCES semantic_span_plans(semantic_span_plan_id) ON DELETE RESTRICT,
    semantic_span_id TEXT NOT NULL,
    semantic_span_hash TEXT NOT NULL CHECK (length(semantic_span_hash) = 64 AND semantic_span_hash NOT GLOB '*[^0-9a-f]*'),
    parent_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    parent_text_sha256 TEXT NOT NULL CHECK (length(parent_text_sha256) = 64 AND parent_text_sha256 NOT GLOB '*[^0-9a-f]*'),
    parent_ordinal INTEGER NOT NULL CHECK (parent_ordinal >= 0),
    span_ordinal INTEGER NOT NULL CHECK (span_ordinal >= 0),
    char_start INTEGER NOT NULL CHECK (char_start >= 0),
    char_end INTEGER NOT NULL CHECK (char_end > char_start),
    content_token_start INTEGER NOT NULL CHECK (content_token_start >= 0),
    content_token_end INTEGER NOT NULL CHECK (content_token_end >= content_token_start),
    span_text_sha256 TEXT NOT NULL CHECK (length(span_text_sha256) = 64 AND span_text_sha256 NOT GLOB '*[^0-9a-f]*'),
    prepared_token_count INTEGER NOT NULL CHECK (prepared_token_count >= 1),
    profile_hash TEXT NOT NULL REFERENCES semantic_span_profiles(profile_hash) ON DELETE RESTRICT,
    model_profile_hash TEXT NOT NULL REFERENCES embedding_model_profiles(profile_hash) ON DELETE RESTRICT,
    PRIMARY KEY (semantic_span_plan_id, semantic_span_id),
    UNIQUE (semantic_span_plan_id, semantic_span_hash),
    UNIQUE (semantic_span_plan_id, parent_fragment_id, span_ordinal),
    UNIQUE (
        semantic_span_plan_id, semantic_span_id, semantic_span_hash,
        parent_fragment_id, span_ordinal, span_text_sha256
    ),
    CHECK (semantic_span_id = 'semantic_span_' || substr(semantic_span_hash, 1, 32)),
    FOREIGN KEY (
        semantic_span_plan_id, parent_fragment_id,
        parent_text_sha256, parent_ordinal
    ) REFERENCES semantic_span_parents(
        semantic_span_plan_id, source_fragment_id,
        text_sha256, ordinal
    ) ON DELETE RESTRICT
);

CREATE TABLE semantic_span_vector_generations (
    semantic_span_vector_generation_id TEXT PRIMARY KEY,
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    corpus_snapshot_id TEXT NOT NULL REFERENCES corpus_snapshots(corpus_snapshot_id) ON DELETE RESTRICT,
    snapshot_hash TEXT NOT NULL CHECK (length(snapshot_hash) = 64 AND snapshot_hash NOT GLOB '*[^0-9a-f]*'),
    access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE RESTRICT,
    policy_hash TEXT NOT NULL CHECK (length(policy_hash) = 64 AND policy_hash NOT GLOB '*[^0-9a-f]*'),
    exclusion_hash TEXT NOT NULL CHECK (length(exclusion_hash) = 64 AND exclusion_hash NOT GLOB '*[^0-9a-f]*'),
    permitted_set_hash TEXT NOT NULL CHECK (length(permitted_set_hash) = 64 AND permitted_set_hash NOT GLOB '*[^0-9a-f]*'),
    semantic_audit_id TEXT NOT NULL,
    semantic_audit_hash TEXT NOT NULL CHECK (length(semantic_audit_hash) = 64 AND semantic_audit_hash NOT GLOB '*[^0-9a-f]*'),
    semantic_audit_binding_hash TEXT NOT NULL CHECK (length(semantic_audit_binding_hash) = 64 AND semantic_audit_binding_hash NOT GLOB '*[^0-9a-f]*'),
    semantic_audit_fragment_manifest_hash TEXT NOT NULL CHECK (length(semantic_audit_fragment_manifest_hash) = 64 AND semantic_audit_fragment_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    semantic_span_plan_id TEXT NOT NULL,
    semantic_span_plan_hash TEXT NOT NULL CHECK (length(semantic_span_plan_hash) = 64 AND semantic_span_plan_hash NOT GLOB '*[^0-9a-f]*'),
    semantic_span_profile_hash TEXT NOT NULL REFERENCES semantic_span_profiles(profile_hash) ON DELETE RESTRICT,
    semantic_span_parent_manifest_hash TEXT NOT NULL CHECK (length(semantic_span_parent_manifest_hash) = 64 AND semantic_span_parent_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    semantic_span_closure_hash TEXT NOT NULL CHECK (length(semantic_span_closure_hash) = 64 AND semantic_span_closure_hash NOT GLOB '*[^0-9a-f]*'),
    model_profile_hash TEXT NOT NULL REFERENCES embedding_model_profiles(profile_hash) ON DELETE RESTRICT,
    runtime_profile_hash TEXT NOT NULL REFERENCES model_runtime_profiles(profile_hash) ON DELETE RESTRICT,
    provisioning_receipt_hash TEXT NOT NULL CHECK (length(provisioning_receipt_hash) = 64 AND provisioning_receipt_hash NOT GLOB '*[^0-9a-f]*'),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 8192),
    normalization TEXT NOT NULL CHECK (normalization = 'l2_unit_float32_v1'),
    parent_count INTEGER NOT NULL CHECK (parent_count BETWEEN 0 AND 10000000),
    span_count INTEGER NOT NULL CHECK (span_count BETWEEN 0 AND 100000000),
    span_manifest_hash TEXT NOT NULL CHECK (length(span_manifest_hash) = 64 AND span_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    generation_hash TEXT NOT NULL UNIQUE CHECK (length(generation_hash) = 64 AND generation_hash NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    CHECK (semantic_span_vector_generation_id = 'semantic_span_vector_generation_' || substr(generation_hash, 1, 32)),
    CHECK (corpus_snapshot_id = 'snapshot_' || substr(snapshot_hash, 1, 32)),
    CHECK (semantic_audit_id = 'semantic_audit_' || substr(semantic_audit_hash, 1, 32)),
    CHECK (semantic_span_plan_id = 'semantic_span_plan_' || substr(semantic_span_plan_hash, 1, 32)),
    CHECK (
        (parent_count = 0 AND span_count = 0) OR
        (parent_count > 0 AND span_count >= parent_count)
    ),
    UNIQUE (semantic_span_vector_generation_id, semantic_span_plan_id),
    FOREIGN KEY (
        semantic_span_plan_id, semantic_span_plan_hash
    ) REFERENCES semantic_span_plans(
        semantic_span_plan_id, plan_hash
    ) ON DELETE RESTRICT,
    UNIQUE (
        library_id, corpus_snapshot_id, snapshot_hash,
        access_policy_id, policy_hash, exclusion_hash, permitted_set_hash,
        semantic_audit_id, semantic_audit_hash,
        semantic_audit_binding_hash, semantic_audit_fragment_manifest_hash,
        semantic_span_plan_id, semantic_span_plan_hash,
        semantic_span_profile_hash, semantic_span_parent_manifest_hash,
        semantic_span_closure_hash, model_profile_hash,
        runtime_profile_hash, provisioning_receipt_hash,
        dimensions, normalization, parent_count, span_count
    )
);

CREATE TABLE semantic_span_vector_items (
    semantic_span_vector_generation_id TEXT NOT NULL REFERENCES semantic_span_vector_generations(semantic_span_vector_generation_id) ON DELETE RESTRICT,
    semantic_span_plan_id TEXT NOT NULL,
    semantic_span_id TEXT NOT NULL,
    parent_fragment_id TEXT NOT NULL REFERENCES source_fragments(source_fragment_id) ON DELETE RESTRICT,
    span_ordinal INTEGER NOT NULL CHECK (span_ordinal >= 0),
    semantic_span_hash TEXT NOT NULL CHECK (length(semantic_span_hash) = 64 AND semantic_span_hash NOT GLOB '*[^0-9a-f]*'),
    span_text_sha256 TEXT NOT NULL CHECK (length(span_text_sha256) = 64 AND span_text_sha256 NOT GLOB '*[^0-9a-f]*'),
    vector_sha256 TEXT NOT NULL CHECK (length(vector_sha256) = 64 AND vector_sha256 NOT GLOB '*[^0-9a-f]*'),
    vector_blob BLOB NOT NULL CHECK (typeof(vector_blob) = 'blob' AND length(vector_blob) > 0),
    PRIMARY KEY (semantic_span_vector_generation_id, semantic_span_id),
    UNIQUE (semantic_span_vector_generation_id, parent_fragment_id, span_ordinal),
    FOREIGN KEY (
        semantic_span_vector_generation_id, semantic_span_plan_id
    ) REFERENCES semantic_span_vector_generations(
        semantic_span_vector_generation_id, semantic_span_plan_id
    ) ON DELETE RESTRICT,
    FOREIGN KEY (
        semantic_span_plan_id, semantic_span_id
    ) REFERENCES semantic_spans(
        semantic_span_plan_id, semantic_span_id
    ) ON DELETE RESTRICT
);

CREATE TRIGGER semantic_span_profiles_validate_insert
BEFORE INSERT ON semantic_span_profiles
BEGIN
    SELECT RAISE(ABORT, 'SemanticSpanProfile model, runtime, prefix, or token budget is inconsistent')
    WHERE NOT EXISTS (
        SELECT 1
        FROM embedding_model_profiles AS model
        JOIN model_runtime_profiles AS runtime
          ON runtime.profile_hash = NEW.runtime_profile_hash
        WHERE model.profile_hash = NEW.model_profile_hash
          AND model.passage_prefix = NEW.passage_prefix
          AND NEW.max_prepared_tokens <= model.max_tokens
    );
END;

CREATE TRIGGER semantic_span_parents_validate_insert
BEFORE INSERT ON semantic_span_parents
BEGIN
    SELECT RAISE(ABORT, 'SemanticSpan parent must match exact persisted source lineage, text, and ordinal')
    WHERE NOT EXISTS (
        SELECT 1
        FROM semantic_span_plans AS plan
        JOIN source_fragments AS fragment
          ON fragment.source_fragment_id = NEW.source_fragment_id
        JOIN source_versions AS version
          ON version.source_version_id = fragment.source_version_id
        JOIN sources AS source
          ON source.source_id = version.source_id
        JOIN source_fragment_text_metrics AS metrics
          ON metrics.source_fragment_id = fragment.source_fragment_id
        WHERE plan.semantic_span_plan_id = NEW.semantic_span_plan_id
          AND fragment.source_version_id = NEW.source_version_id
          AND fragment.ordinal = NEW.ordinal
          AND fragment.text_sha256 = NEW.text_sha256
          AND metrics.text_sha256 = NEW.text_sha256
          AND metrics.text_char_count = NEW.text_char_count
          AND version.source_id = NEW.source_id
          AND source.source_id = NEW.source_id
    );
    SELECT RAISE(ABORT, 'SemanticSpan plan parent count exceeded')
    WHERE (SELECT COUNT(*) FROM semantic_span_parents WHERE semantic_span_plan_id = NEW.semantic_span_plan_id)
          >= (SELECT parent_count FROM semantic_span_plans WHERE semantic_span_plan_id = NEW.semantic_span_plan_id);
END;

CREATE TRIGGER source_fragments_capture_text_metrics
AFTER INSERT ON source_fragments
BEGIN
    INSERT INTO source_fragment_text_metrics(
        source_fragment_id, text_sha256, text_char_count
    ) VALUES (
        NEW.source_fragment_id, NEW.text_sha256, length(NEW.text)
    );
END;

CREATE TRIGGER semantic_spans_validate_insert
BEFORE INSERT ON semantic_spans
BEGIN
    SELECT RAISE(ABORT, 'SemanticSpan parent, plan, profile, model, token budget, or bounds are inconsistent')
    WHERE NOT EXISTS (
        SELECT 1
        FROM semantic_span_plans AS plan
        JOIN semantic_span_profiles AS profile
          ON profile.profile_hash = plan.semantic_span_profile_hash
        JOIN embedding_model_profiles AS model
          ON model.profile_hash = profile.model_profile_hash
        JOIN semantic_span_parents AS parent
          ON parent.semantic_span_plan_id = plan.semantic_span_plan_id
         AND parent.source_fragment_id = NEW.parent_fragment_id
        WHERE plan.semantic_span_plan_id = NEW.semantic_span_plan_id
          AND profile.profile_hash = NEW.profile_hash
          AND model.profile_hash = NEW.model_profile_hash
          AND parent.text_sha256 = NEW.parent_text_sha256
          AND parent.ordinal = NEW.parent_ordinal
          AND NEW.span_ordinal < parent.span_count
          AND NEW.char_end <= parent.text_char_count
          AND NEW.content_token_end <= parent.content_token_count
          AND NEW.prepared_token_count <= profile.max_prepared_tokens
    );
    SELECT RAISE(ABORT, 'SemanticSpan parent span count exceeded')
    WHERE (SELECT COUNT(*) FROM semantic_spans WHERE semantic_span_plan_id = NEW.semantic_span_plan_id AND parent_fragment_id = NEW.parent_fragment_id)
          >= (SELECT span_count FROM semantic_span_parents WHERE semantic_span_plan_id = NEW.semantic_span_plan_id AND source_fragment_id = NEW.parent_fragment_id);
    SELECT RAISE(ABORT, 'SemanticSpan plan span count exceeded')
    WHERE (SELECT COUNT(*) FROM semantic_spans WHERE semantic_span_plan_id = NEW.semantic_span_plan_id)
          >= (SELECT span_count FROM semantic_span_plans WHERE semantic_span_plan_id = NEW.semantic_span_plan_id);
END;

CREATE TRIGGER semantic_span_vector_generations_validate_insert
BEFORE INSERT ON semantic_span_vector_generations
BEGIN
    SELECT RAISE(ABORT, 'SemanticSpan vector generation authority, audit, plan, or model closure is inconsistent')
    WHERE NOT EXISTS (
        SELECT 1
        FROM corpus_snapshots AS snapshot
        JOIN access_policies AS policy
          ON policy.access_policy_id = NEW.access_policy_id
        JOIN semantic_span_plans AS plan
          ON plan.semantic_span_plan_id = NEW.semantic_span_plan_id
         AND plan.plan_hash = NEW.semantic_span_plan_hash
        JOIN semantic_span_profiles AS profile
          ON profile.profile_hash = plan.semantic_span_profile_hash
        JOIN embedding_model_profiles AS model
          ON model.profile_hash = profile.model_profile_hash
        JOIN model_runtime_profiles AS runtime
          ON runtime.profile_hash = profile.runtime_profile_hash
        WHERE snapshot.corpus_snapshot_id = NEW.corpus_snapshot_id
          AND snapshot.library_id = NEW.library_id
          AND snapshot.manifest_hash = NEW.snapshot_hash
          AND policy.library_id = NEW.library_id
          AND policy.policy_hash = NEW.policy_hash
          AND plan.semantic_span_profile_hash = NEW.semantic_span_profile_hash
          AND plan.parent_manifest_hash = NEW.semantic_span_parent_manifest_hash
          AND plan.span_closure_hash = NEW.semantic_span_closure_hash
          AND plan.parent_count = NEW.parent_count
          AND plan.span_count = NEW.span_count
          AND profile.model_profile_hash = NEW.model_profile_hash
          AND profile.runtime_profile_hash = NEW.runtime_profile_hash
          AND profile.provisioning_receipt_hash = NEW.provisioning_receipt_hash
          AND model.dimensions = NEW.dimensions
    );
    SELECT RAISE(ABORT, 'SemanticSpan vector generation requires a complete persisted plan')
    WHERE NEW.parent_count <> (
              SELECT COUNT(*) FROM semantic_span_parents
              WHERE semantic_span_plan_id = NEW.semantic_span_plan_id
          )
       OR NEW.span_count <> (
              SELECT COUNT(*) FROM semantic_spans
              WHERE semantic_span_plan_id = NEW.semantic_span_plan_id
          )
       OR EXISTS (
              SELECT 1
              FROM semantic_span_parents AS parent
              WHERE parent.semantic_span_plan_id = NEW.semantic_span_plan_id
                AND parent.span_count <> (
                    SELECT COUNT(*)
                    FROM semantic_spans AS span
                    WHERE span.semantic_span_plan_id = parent.semantic_span_plan_id
                      AND span.parent_fragment_id = parent.source_fragment_id
                )
          );
    SELECT RAISE(ABORT, 'SemanticSpan vector generation plan is outside the authorized snapshot Library')
    WHERE EXISTS (
        SELECT 1
        FROM semantic_span_parents AS parent
        JOIN source_versions AS version
          ON version.source_version_id = parent.source_version_id
        JOIN sources AS source
          ON source.source_id = version.source_id
        WHERE parent.semantic_span_plan_id = NEW.semantic_span_plan_id
          AND (
              source.library_id <> NEW.library_id OR
              NOT EXISTS (
                  SELECT 1
                  FROM snapshot_members AS member
                  WHERE member.corpus_snapshot_id = NEW.corpus_snapshot_id
                    AND member.source_version_id = parent.source_version_id
                    AND member.membership_state = 'active'
              )
          )
    );
END;

CREATE TRIGGER semantic_span_vector_items_validate_insert
BEFORE INSERT ON semantic_span_vector_items
BEGIN
    SELECT RAISE(ABORT, 'SemanticSpan vector item generation, plan, span hashes, parent, or dimensions are inconsistent')
    WHERE NOT EXISTS (
        SELECT 1
        FROM semantic_span_vector_generations AS generation
        JOIN semantic_spans AS span
          ON span.semantic_span_plan_id = generation.semantic_span_plan_id
         AND span.semantic_span_id = NEW.semantic_span_id
        WHERE generation.semantic_span_vector_generation_id = NEW.semantic_span_vector_generation_id
          AND generation.semantic_span_plan_id = NEW.semantic_span_plan_id
          AND span.semantic_span_hash = NEW.semantic_span_hash
          AND span.parent_fragment_id = NEW.parent_fragment_id
          AND span.span_ordinal = NEW.span_ordinal
          AND span.span_text_sha256 = NEW.span_text_sha256
          AND length(NEW.vector_blob) = generation.dimensions * 4
    );
    SELECT RAISE(ABORT, 'SemanticSpan vector generation item count exceeded')
    WHERE (SELECT COUNT(*) FROM semantic_span_vector_items WHERE semantic_span_vector_generation_id = NEW.semantic_span_vector_generation_id)
          >= (SELECT span_count FROM semantic_span_vector_generations WHERE semantic_span_vector_generation_id = NEW.semantic_span_vector_generation_id);
END;

CREATE INDEX semantic_span_profiles_model_runtime_idx
ON semantic_span_profiles(model_profile_hash, runtime_profile_hash);
CREATE INDEX source_fragment_text_metrics_hash_idx
ON source_fragment_text_metrics(text_sha256, source_fragment_id);
CREATE INDEX semantic_span_plans_profile_idx
ON semantic_span_plans(semantic_span_profile_hash, semantic_span_plan_id);
CREATE INDEX semantic_span_parents_fragment_idx
ON semantic_span_parents(source_fragment_id, semantic_span_plan_id);
CREATE INDEX semantic_span_parents_source_ordinal_idx
ON semantic_span_parents(source_version_id, ordinal, semantic_span_plan_id);
CREATE INDEX semantic_spans_parent_group_idx
ON semantic_spans(semantic_span_plan_id, parent_fragment_id, span_ordinal);
CREATE INDEX semantic_spans_profile_model_idx
ON semantic_spans(profile_hash, model_profile_hash);
CREATE INDEX semantic_span_vector_generations_scope_idx
ON semantic_span_vector_generations(
    library_id, corpus_snapshot_id, access_policy_id,
    exclusion_hash, permitted_set_hash
);
CREATE INDEX semantic_span_vector_generations_plan_idx
ON semantic_span_vector_generations(semantic_span_plan_id, semantic_span_plan_hash);
CREATE INDEX semantic_span_vector_generations_model_runtime_idx
ON semantic_span_vector_generations(model_profile_hash, runtime_profile_hash);
CREATE INDEX semantic_span_vector_items_parent_group_idx
ON semantic_span_vector_items(
    semantic_span_vector_generation_id, parent_fragment_id, span_ordinal
);
CREATE INDEX semantic_span_vector_items_span_reverse_idx
ON semantic_span_vector_items(semantic_span_plan_id, semantic_span_id);
CREATE INDEX semantic_span_vector_items_vector_hash_idx
ON semantic_span_vector_items(vector_sha256);

CREATE TRIGGER semantic_span_profiles_no_update BEFORE UPDATE ON semantic_span_profiles BEGIN SELECT RAISE(ABORT, 'semantic_span_profiles is append-only'); END;
CREATE TRIGGER semantic_span_profiles_no_delete BEFORE DELETE ON semantic_span_profiles BEGIN SELECT RAISE(ABORT, 'semantic_span_profiles is append-only'); END;
CREATE TRIGGER source_fragment_text_metrics_no_update BEFORE UPDATE ON source_fragment_text_metrics BEGIN SELECT RAISE(ABORT, 'source_fragment_text_metrics is append-only'); END;
CREATE TRIGGER source_fragment_text_metrics_no_delete BEFORE DELETE ON source_fragment_text_metrics BEGIN SELECT RAISE(ABORT, 'source_fragment_text_metrics is append-only'); END;
CREATE TRIGGER semantic_span_plans_no_update BEFORE UPDATE ON semantic_span_plans BEGIN SELECT RAISE(ABORT, 'semantic_span_plans is append-only'); END;
CREATE TRIGGER semantic_span_plans_no_delete BEFORE DELETE ON semantic_span_plans BEGIN SELECT RAISE(ABORT, 'semantic_span_plans is append-only'); END;
CREATE TRIGGER semantic_span_parents_no_update BEFORE UPDATE ON semantic_span_parents BEGIN SELECT RAISE(ABORT, 'semantic_span_parents is append-only'); END;
CREATE TRIGGER semantic_span_parents_no_delete BEFORE DELETE ON semantic_span_parents BEGIN SELECT RAISE(ABORT, 'semantic_span_parents is append-only'); END;
CREATE TRIGGER semantic_spans_no_update BEFORE UPDATE ON semantic_spans BEGIN SELECT RAISE(ABORT, 'semantic_spans is append-only'); END;
CREATE TRIGGER semantic_spans_no_delete BEFORE DELETE ON semantic_spans BEGIN SELECT RAISE(ABORT, 'semantic_spans is append-only'); END;
CREATE TRIGGER semantic_span_vector_generations_no_update BEFORE UPDATE ON semantic_span_vector_generations BEGIN SELECT RAISE(ABORT, 'semantic_span_vector_generations is append-only'); END;
CREATE TRIGGER semantic_span_vector_generations_no_delete BEFORE DELETE ON semantic_span_vector_generations BEGIN SELECT RAISE(ABORT, 'semantic_span_vector_generations is append-only'); END;
CREATE TRIGGER semantic_span_vector_items_no_update BEFORE UPDATE ON semantic_span_vector_items BEGIN SELECT RAISE(ABORT, 'semantic_span_vector_items is append-only'); END;
CREATE TRIGGER semantic_span_vector_items_no_delete BEFORE DELETE ON semantic_span_vector_items BEGIN SELECT RAISE(ABORT, 'semantic_span_vector_items is append-only'); END;
