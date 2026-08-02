"""Protocol and real-Library tests for the thin MCP stdio boundary."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.evidence import (
    EvidenceAnswerability,
    EvidenceGateSpec,
    EvidenceRequirement,
)
from dithyramba.ingest.service import IngestService
from dithyramba.interactive import AgentResearchFacade
from dithyramba.library import LibraryConfig
from dithyramba.mcp_stdio import (
    MCP_PROTOCOL_VERSION,
    McpStdioServer,
    _JournalInput,
    _MethodNotFound,
    _OpenSessionInput,
    _require_model,
    _rpc_error,
    _SessionInput,
    run_stdio_mcp,
)
from dithyramba.persistence import LibraryRepository, initialize_library
from dithyramba.recall import current_fts_runtime_profile


def _prepare(repository: LibraryRepository, root: Path) -> tuple[str, str, str]:
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="MCP corpus",
            kind=CollectionKind.CORPUS,
            roots=(
                build_collection_root(
                    root,
                    data_root=repository.paths.application_data_root,
                ),
            ),
        )
    )
    IngestService(repository).ingest_path(collection.config.collection_id, "evidence.md")
    snapshot = repository.freeze_snapshot((collection.config.collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id="policy_mcp_research",
        library_id=repository.library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),),
    )
    repository.persist_access_policy(name="MCP research", snapshot=policy)
    return collection.config.collection_id, snapshot.corpus_snapshot_id, policy.access_policy_id


def _rpc(
    server: McpStdioServer,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    request_id: int | None = 1,
) -> dict[str, Any] | None:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if request_id is not None:
        message["id"] = request_id
    return cast(
        dict[str, Any] | None,
        server.process_line(json.dumps(message, ensure_ascii=False)),
    )


def _initialize(server: McpStdioServer) -> dict[str, Any]:
    initialized = _rpc(
        server,
        "initialize",
        {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    )
    assert initialized is not None
    assert _rpc(server, "notifications/initialized", request_id=None) is None
    return initialized


def test_mcp_protocol_lifecycle_and_errors(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "evidence.md").write_text("# Evidence\n\nBlocking reveals power.\n")
    with initialize_library(LibraryConfig(name="MCP"), data_root=tmp_path / "data") as repo:
        _prepare(repo, root)
        facade = AgentResearchFacade(
            repo,
            agent_id="agent:mcp_test",
            profile_version=current_fts_runtime_profile().profile_version,
        )
        with pytest.raises(TypeError, match="AgentResearchFacade"):
            McpStdioServer(cast(AgentResearchFacade, object()))
        server = McpStdioServer(facade)

        assert server.process_line("{") == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        }
        for invalid in ("[]", '{"jsonrpc":"1.0","id":1,"method":"ping"}'):
            response = cast(dict[str, Any] | None, server.process_line(invalid))
            assert response is not None and response["error"]["code"] == -32600
        bad_params = cast(
            dict[str, Any] | None,
            server.process_line('{"jsonrpc":"2.0","id":1,"method":"ping","params":[]}'),
        )
        assert bad_params is not None and bad_params["error"]["code"] == -32602
        not_ready = _rpc(server, "tools/list")
        assert not_ready is not None and not_ready["error"]["code"] == -32602
        assert _rpc(server, "ping") == {"jsonrpc": "2.0", "id": 1, "result": {}}
        assert _rpc(server, "initialize", {}, request_id=None) is None

        initialized = _initialize(server)
        assert initialized["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert initialized["result"]["capabilities"] == {"tools": {"listChanged": False}}
        repeated = _rpc(server, "initialize", {"protocolVersion": MCP_PROTOCOL_VERSION})
        assert repeated is not None and repeated["error"]["code"] == -32602
        unknown = _rpc(server, "unknown/method")
        assert unknown is not None and unknown["error"]["code"] == -32601
        assert _rpc(server, "unknown/notification", request_id=None) is None
        invalid_call = _rpc(server, "tools/call", {"name": 1})
        assert invalid_call is not None and invalid_call["error"]["code"] == -32602

        tools = _rpc(server, "tools/list")
        assert tools is not None
        listed = tools["result"]["tools"]
        assert [item["name"] for item in listed] == [
            "open_session",
            "recall",
            "session_context",
            "prepare_answer",
            "record_draft",
            "record_gap",
            "reject_path",
        ]
        assert all(item["inputSchema"]["additionalProperties"] is False for item in listed)
        unsupported = _rpc(server, "tools/call", {"name": "absent", "arguments": {}})
        assert unsupported is not None and unsupported["result"]["isError"] is True
        bad_tool_input = _rpc(
            server,
            "tools/call",
            {"name": "session_context", "arguments": {"session_id": 4}},
        )
        assert bad_tool_input is not None and bad_tool_input["result"]["isError"] is True

        with pytest.raises(TypeError, match="_JournalInput"):
            _require_model(
                _OpenSessionInput(
                    question="q",
                    intended_use="use",
                    success_criteria=["criterion"],
                    corpus_snapshot_id="snapshot_test",
                    access_policy_id="policy_test",
                    purpose="research",
                    collection_ids=["collection_test"],
                ),
                _JournalInput,
            )
        assert str(_MethodNotFound("absent")) == "absent"
        assert _rpc_error(7, -32602, "bad", data={"field": "value"})["error"] == {
            "code": -32602,
            "message": "bad",
            "data": {"field": "value"},
        }


def test_mcp_protocol_boundary_suppresses_notification_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "evidence.md").write_text("# Evidence\n\nExact text.\n")
    with initialize_library(
        LibraryConfig(name="MCP boundary"), data_root=tmp_path / "data"
    ) as repo:
        _prepare(repo, root)
        facade = AgentResearchFacade(
            repo,
            agent_id="agent:mcp_boundary",
            profile_version=current_fts_runtime_profile().profile_version,
        )
        server = McpStdioServer(facade)

        before_initialize = _rpc(server, "notifications/initialized")
        assert before_initialize is not None and before_initialize["error"]["code"] == -32602
        missing_version = _rpc(server, "initialize", {})
        assert missing_version is not None and missing_version["error"]["code"] == -32602

        def validation_failure(*_args: object, **_kwargs: object) -> dict[str, object]:
            _SessionInput.model_validate({})
            raise AssertionError("unreachable")

        monkeypatch.setattr(server, "_dispatch", validation_failure)
        invalid = _rpc(server, "anything")
        assert invalid is not None and invalid["error"]["code"] == -32602
        assert _rpc(server, "anything", request_id=None) is None

        def value_failure(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise ValueError("bounded failure")

        monkeypatch.setattr(server, "_dispatch", value_failure)
        assert _rpc(server, "anything", request_id=None) is None

        def internal_failure(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(server, "_dispatch", internal_failure)
        internal = _rpc(server, "anything")
        assert internal is not None and internal["error"] == {
            "code": -32603,
            "message": "Internal error",
            "data": {"type": "RuntimeError", "message": "unexpected failure"},
        }
        assert _rpc(server, "anything", request_id=None) is None


def test_mcp_tools_drive_real_compact_research_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "evidence.md").write_text(
        "# Evidence\n\nBlocking reveals power. Eyelines reveal attention.\n",
        encoding="utf-8",
    )
    with initialize_library(LibraryConfig(name="MCP tools"), data_root=tmp_path / "data") as repo:
        collection_id, snapshot_id, policy_id = _prepare(repo, root)
        facade = AgentResearchFacade(
            repo,
            agent_id="agent:mcp_test",
            profile_version=current_fts_runtime_profile().profile_version,
        )
        server = McpStdioServer(facade)
        _initialize(server)

        opened = _rpc(
            server,
            "tools/call",
            {
                "name": "open_session",
                "arguments": {
                    "question": "How should staging carry meaning?",
                    "intended_use": "directorial research",
                    "success_criteria": ["find exact support"],
                    "boundaries": ["retrieval is not acceptance"],
                    "corpus_snapshot_id": snapshot_id,
                    "access_policy_id": policy_id,
                    "purpose": "research",
                    "collection_ids": [collection_id],
                },
            },
        )
        assert opened is not None and opened["result"]["isError"] is False
        session_id = opened["result"]["structuredContent"]["session_id"]

        recalled = _rpc(
            server,
            "tools/call",
            {
                "name": "recall",
                "arguments": {
                    "session_id": session_id,
                    "question": "blocking power eyelines attention",
                    "command_id": "command_mcp_recall_001",
                    "max_candidates": 10,
                    "max_source_fragments": 3,
                },
            },
        )
        assert recalled is not None and recalled["result"]["isError"] is False
        turn = recalled["result"]["structuredContent"]
        assert turn["evidence_packet"]["source_fragments"]
        assert "read_receipt" not in turn["evidence_packet"]
        assert json.loads(recalled["result"]["content"][0]["text"]) == turn

        repeated = _rpc(
            server,
            "tools/call",
            {
                "name": "recall",
                "arguments": {
                    "session_id": session_id,
                    "question": "blocking power eyelines attention",
                    "command_id": "command_mcp_recall_001",
                    "max_candidates": 10,
                    "max_source_fragments": 3,
                },
            },
        )
        assert repeated is not None
        assert repeated["result"]["structuredContent"]["turn_hash"] == turn["turn_hash"]

        gate_spec = EvidenceGateSpec(
            query_key="mcp_dialogue_power",
            question="blocking power eyelines attention",
            expected_answerability=EvidenceAnswerability.ANSWERABLE,
            requirements=(
                EvidenceRequirement(
                    key="exact_support",
                    label="Exact blocking support",
                    anchor_groups=(("blocking",), ("power",)),
                ),
            ),
        )
        prepared = _rpc(
            server,
            "tools/call",
            {
                "name": "prepare_answer",
                "arguments": {
                    "session_id": session_id,
                    "evidence_event_id": turn["evidence_event_id"],
                    "gate_spec": gate_spec.model_dump(mode="json"),
                },
            },
        )
        assert prepared is not None and prepared["result"]["isError"] is False
        preparation = prepared["result"]["structuredContent"]
        assert preparation["response_mode"] == "answer"

        drafted = _rpc(
            server,
            "tools/call",
            {
                "name": "record_draft",
                "arguments": {
                    "session_id": session_id,
                    "text": "Use blocking to externalize the status shift.",
                    "preparation": preparation,
                },
            },
        )
        assert drafted is not None and drafted["result"]["isError"] is False

        for name, text in (
            ("record_gap", "Need a counterexample with static staging."),
            ("reject_path", "Do not use generic advice without a source."),
        ):
            response = _rpc(
                server,
                "tools/call",
                {"name": name, "arguments": {"session_id": session_id, "text": text}},
            )
            assert response is not None and response["result"]["isError"] is False

        context = _rpc(
            server,
            "tools/call",
            {"name": "session_context", "arguments": {"session_id": session_id}},
        )
        assert context is not None
        projected = context["result"]["structuredContent"]
        assert projected["drafts"][0]["text"].startswith("Use blocking")
        assert projected["gaps"][0]["text"].startswith("Need a counterexample")
        assert projected["rejected_paths"][0]["text"].startswith("Do not use")

        def broken_context(_session_id: str) -> object:
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(facade, "context", broken_context)
        failed = _rpc(
            server,
            "tools/call",
            {"name": "session_context", "arguments": {"session_id": session_id}},
        )
        assert failed is not None and failed["result"]["isError"] is True
        with pytest.raises(ValueError, match="unsupported tool"):
            server._invoke(
                "not_registered",
                _JournalInput(session_id=session_id, text="x"),
            )


def test_stdio_serve_and_public_runner_emit_protocol_only(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "evidence.md").write_text("# Evidence\n\nExact source text.\n")
    with initialize_library(LibraryConfig(name="MCP streams"), data_root=tmp_path / "data") as repo:
        _prepare(repo, root)
        transcript = "\n".join(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "unsupported-version",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1"},
                        },
                    }
                ),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            )
        )
        output = io.StringIO()
        run_stdio_mcp(
            repo,
            agent_id="agent:mcp_stream_test",
            profile_version=current_fts_runtime_profile().profile_version,
            input_stream=io.StringIO(transcript),
            output_stream=output,
        )
        lines = output.getvalue().splitlines()
        assert len(lines) == 2
        first, second = (json.loads(line) for line in lines)
        assert first["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert second["result"]["tools"]


def test_mcp_cli_opens_verified_library_and_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dithyramba.cli import app

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "evidence.md").write_text("# Evidence\n\nExact source text.\n")
    data_home = tmp_path / "data"
    with initialize_library(LibraryConfig(name="MCP CLI"), data_root=data_home) as repo:
        _prepare(repo, root)
        library_id = repo.library_id

    captured: dict[str, object] = {}

    def fake_run(repository: LibraryRepository, **kwargs: object) -> None:
        captured["library_id"] = repository.library_id
        captured.update(kwargs)

    monkeypatch.setattr("dithyramba.cli.run_stdio_mcp", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "mcp",
            "--library",
            library_id,
            "--data-home",
            str(data_home),
            "--agent-id",
            "agent:cli_test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["library_id"] == library_id
    assert captured["agent_id"] == "agent:cli_test"
