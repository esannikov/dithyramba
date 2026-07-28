-- Link random recall/replay runs to reusable content-addressed artifacts.
CREATE TABLE recall_run_artifacts (
    processing_run_id TEXT PRIMARY KEY REFERENCES processing_runs(processing_run_id) ON DELETE RESTRICT,
    coverage_report_id TEXT NOT NULL REFERENCES coverage_reports(coverage_report_id) ON DELETE RESTRICT,
    read_receipt_id TEXT NOT NULL REFERENCES read_receipts(read_receipt_id) ON DELETE RESTRICT,
    access_receipt_id TEXT NOT NULL REFERENCES access_receipts(access_receipt_id) ON DELETE RESTRICT,
    retrieval_receipt_id TEXT NOT NULL REFERENCES retrieval_receipts(retrieval_receipt_id) ON DELETE RESTRICT,
    evidence_packet_id TEXT NOT NULL REFERENCES evidence_packets(evidence_packet_id) ON DELETE RESTRICT
);

CREATE INDEX recall_run_artifacts_coverage_idx
ON recall_run_artifacts(coverage_report_id);
CREATE INDEX recall_run_artifacts_read_idx
ON recall_run_artifacts(read_receipt_id);
CREATE INDEX recall_run_artifacts_access_idx
ON recall_run_artifacts(access_receipt_id);
CREATE INDEX recall_run_artifacts_retrieval_idx
ON recall_run_artifacts(retrieval_receipt_id);
CREATE INDEX recall_run_artifacts_packet_idx
ON recall_run_artifacts(evidence_packet_id);

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

CREATE TRIGGER recall_run_artifacts_no_update
BEFORE UPDATE ON recall_run_artifacts
BEGIN
    SELECT RAISE(ABORT, 'recall_run_artifacts is append-only');
END;
CREATE TRIGGER recall_run_artifacts_no_delete
BEFORE DELETE ON recall_run_artifacts
BEGIN
    SELECT RAISE(ABORT, 'recall_run_artifacts is append-only');
END;
