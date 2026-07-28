-- Dithyramba P7 explicit StructureUnits and typed, source-grounded Relations.

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

CREATE TABLE relation_evidence_links (
    relation_id TEXT NOT NULL REFERENCES relations(relation_id) ON DELETE RESTRICT,
    evidence_link_id TEXT NOT NULL REFERENCES evidence_links(evidence_link_id) ON DELETE RESTRICT,
    evidence_order INTEGER NOT NULL CHECK (evidence_order >= 0 AND evidence_order < 16),
    PRIMARY KEY (relation_id, evidence_link_id),
    UNIQUE (relation_id, evidence_order)
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

CREATE TRIGGER structure_unit_generations_no_update BEFORE UPDATE ON structure_unit_generations BEGIN SELECT RAISE(ABORT, 'structure_unit_generations is append-only'); END;
CREATE TRIGGER structure_unit_generations_no_delete BEFORE DELETE ON structure_unit_generations BEGIN SELECT RAISE(ABORT, 'structure_unit_generations is append-only'); END;
CREATE TRIGGER structure_units_no_update BEFORE UPDATE ON structure_units BEGIN SELECT RAISE(ABORT, 'structure_units is append-only'); END;
CREATE TRIGGER structure_units_no_delete BEFORE DELETE ON structure_units BEGIN SELECT RAISE(ABORT, 'structure_units is append-only'); END;
CREATE TRIGGER structure_unit_members_no_update BEFORE UPDATE ON structure_unit_members BEGIN SELECT RAISE(ABORT, 'structure_unit_members is append-only'); END;
CREATE TRIGGER structure_unit_members_no_delete BEFORE DELETE ON structure_unit_members BEGIN SELECT RAISE(ABORT, 'structure_unit_members is append-only'); END;
CREATE TRIGGER structure_unit_read_receipts_no_update BEFORE UPDATE ON structure_unit_read_receipts BEGIN SELECT RAISE(ABORT, 'structure_unit_read_receipts is append-only'); END;
CREATE TRIGGER structure_unit_read_receipts_no_delete BEFORE DELETE ON structure_unit_read_receipts BEGIN SELECT RAISE(ABORT, 'structure_unit_read_receipts is append-only'); END;
CREATE TRIGGER structure_unit_read_receipt_items_no_update BEFORE UPDATE ON structure_unit_read_receipt_items BEGIN SELECT RAISE(ABORT, 'structure_unit_read_receipt_items is append-only'); END;
CREATE TRIGGER structure_unit_read_receipt_items_no_delete BEFORE DELETE ON structure_unit_read_receipt_items BEGIN SELECT RAISE(ABORT, 'structure_unit_read_receipt_items is append-only'); END;
CREATE TRIGGER relation_import_runs_no_update BEFORE UPDATE ON relation_import_runs BEGIN SELECT RAISE(ABORT, 'relation_import_runs is append-only'); END;
CREATE TRIGGER relation_import_runs_no_delete BEFORE DELETE ON relation_import_runs BEGIN SELECT RAISE(ABORT, 'relation_import_runs is append-only'); END;
CREATE TRIGGER relations_no_update BEFORE UPDATE ON relations BEGIN SELECT RAISE(ABORT, 'relations is append-only'); END;
CREATE TRIGGER relations_no_delete BEFORE DELETE ON relations BEGIN SELECT RAISE(ABORT, 'relations is append-only'); END;
CREATE TRIGGER relation_evidence_links_no_update BEFORE UPDATE ON relation_evidence_links BEGIN SELECT RAISE(ABORT, 'relation_evidence_links is append-only'); END;
CREATE TRIGGER relation_evidence_links_no_delete BEFORE DELETE ON relation_evidence_links BEGIN SELECT RAISE(ABORT, 'relation_evidence_links is append-only'); END;
CREATE TRIGGER relation_path_receipts_no_update BEFORE UPDATE ON relation_path_receipts BEGIN SELECT RAISE(ABORT, 'relation_path_receipts is append-only'); END;
CREATE TRIGGER relation_path_receipts_no_delete BEFORE DELETE ON relation_path_receipts BEGIN SELECT RAISE(ABORT, 'relation_path_receipts is append-only'); END;
CREATE TRIGGER relation_path_collections_no_update BEFORE UPDATE ON relation_path_collections BEGIN SELECT RAISE(ABORT, 'relation_path_collections is append-only'); END;
CREATE TRIGGER relation_path_collections_no_delete BEFORE DELETE ON relation_path_collections BEGIN SELECT RAISE(ABORT, 'relation_path_collections is append-only'); END;
CREATE TRIGGER relation_path_exclusions_no_update BEFORE UPDATE ON relation_path_exclusions BEGIN SELECT RAISE(ABORT, 'relation_path_exclusions is append-only'); END;
CREATE TRIGGER relation_path_exclusions_no_delete BEFORE DELETE ON relation_path_exclusions BEGIN SELECT RAISE(ABORT, 'relation_path_exclusions is append-only'); END;
CREATE TRIGGER relation_path_relation_types_no_update BEFORE UPDATE ON relation_path_relation_types BEGIN SELECT RAISE(ABORT, 'relation_path_relation_types is append-only'); END;
CREATE TRIGGER relation_path_relation_types_no_delete BEFORE DELETE ON relation_path_relation_types BEGIN SELECT RAISE(ABORT, 'relation_path_relation_types is append-only'); END;
CREATE TRIGGER relation_path_seeds_no_update BEFORE UPDATE ON relation_path_seeds BEGIN SELECT RAISE(ABORT, 'relation_path_seeds is append-only'); END;
CREATE TRIGGER relation_path_seeds_no_delete BEFORE DELETE ON relation_path_seeds BEGIN SELECT RAISE(ABORT, 'relation_path_seeds is append-only'); END;
CREATE TRIGGER relation_paths_no_update BEFORE UPDATE ON relation_paths BEGIN SELECT RAISE(ABORT, 'relation_paths is append-only'); END;
CREATE TRIGGER relation_paths_no_delete BEFORE DELETE ON relation_paths BEGIN SELECT RAISE(ABORT, 'relation_paths is append-only'); END;
CREATE TRIGGER relation_path_steps_no_update BEFORE UPDATE ON relation_path_steps BEGIN SELECT RAISE(ABORT, 'relation_path_steps is append-only'); END;
CREATE TRIGGER relation_path_steps_no_delete BEFORE DELETE ON relation_path_steps BEGIN SELECT RAISE(ABORT, 'relation_path_steps is append-only'); END;
