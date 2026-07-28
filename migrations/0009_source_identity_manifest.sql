-- Dithyramba v9 connector-declared logical Source identity.
CREATE TABLE collection_root_identity_manifests (
    collection_root_id TEXT PRIMARY KEY REFERENCES collection_roots(collection_root_id) ON DELETE RESTRICT,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL CHECK (length(manifest_sha256) = 64 AND manifest_sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL
);

CREATE TABLE source_family_identities (
    source_family_id TEXT PRIMARY KEY REFERENCES source_families(source_family_id) ON DELETE RESTRICT,
    library_id TEXT NOT NULL REFERENCES libraries(library_id) ON DELETE RESTRICT,
    logical_source_uri TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (library_id, logical_source_uri)
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

CREATE INDEX source_identity_bindings_family_content_idx
ON source_identity_bindings(source_family_id, content_sha256, source_id);

CREATE TRIGGER collection_root_identity_manifests_validate_insert BEFORE INSERT ON collection_root_identity_manifests
BEGIN
    SELECT RAISE(ABORT, 'identity manifest path must be absolute')
    WHERE substr(NEW.manifest_path, 1, 1) <> '/';
END;

CREATE TRIGGER source_family_identities_validate_insert BEFORE INSERT ON source_family_identities
BEGIN
    SELECT RAISE(ABORT, 'SourceFamily identity Library must match SourceFamily')
    WHERE NOT EXISTS (
        SELECT 1 FROM source_families AS family
        WHERE family.source_family_id = NEW.source_family_id
          AND family.library_id = NEW.library_id
    );
END;

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

CREATE TRIGGER collection_root_identity_manifests_no_update BEFORE UPDATE ON collection_root_identity_manifests BEGIN SELECT RAISE(ABORT, 'collection_root_identity_manifests is append-only'); END;
CREATE TRIGGER collection_root_identity_manifests_no_delete BEFORE DELETE ON collection_root_identity_manifests BEGIN SELECT RAISE(ABORT, 'collection_root_identity_manifests is append-only'); END;
CREATE TRIGGER source_family_identities_no_update BEFORE UPDATE ON source_family_identities BEGIN SELECT RAISE(ABORT, 'source_family_identities is append-only'); END;
CREATE TRIGGER source_family_identities_no_delete BEFORE DELETE ON source_family_identities BEGIN SELECT RAISE(ABORT, 'source_family_identities is append-only'); END;
CREATE TRIGGER source_identity_bindings_no_update BEFORE UPDATE ON source_identity_bindings BEGIN SELECT RAISE(ABORT, 'source_identity_bindings is append-only'); END;
CREATE TRIGGER source_identity_bindings_no_delete BEFORE DELETE ON source_identity_bindings BEGIN SELECT RAISE(ABORT, 'source_identity_bindings is append-only'); END;
