-- Durable, append-only interactive research sessions and their hash-chained events.
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

CREATE INDEX research_sessions_library_created_idx
ON research_sessions(library_id, created_at, session_id);

CREATE INDEX research_session_events_session_sequence_idx
ON research_session_events(session_id, sequence);

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

CREATE TRIGGER research_sessions_no_update
BEFORE UPDATE ON research_sessions
BEGIN
    SELECT RAISE(ABORT, 'research_sessions is append-only');
END;

CREATE TRIGGER research_sessions_no_delete
BEFORE DELETE ON research_sessions
BEGIN
    SELECT RAISE(ABORT, 'research_sessions is append-only');
END;

CREATE TRIGGER research_session_events_no_update
BEFORE UPDATE ON research_session_events
BEGIN
    SELECT RAISE(ABORT, 'research_session_events is append-only');
END;

CREATE TRIGGER research_session_events_no_delete
BEFORE DELETE ON research_session_events
BEGIN
    SELECT RAISE(ABORT, 'research_session_events is append-only');
END;

