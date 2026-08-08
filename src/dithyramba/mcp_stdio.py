"""Small dependency-free MCP stdio adapter over the agent research facade.

The transport follows MCP 2025-11-25: one UTF-8 JSON-RPC message per line,
protocol output only on stdout, and diagnostics only on stderr.  The exposed
tools remain agent-safe; human review, promotion and session closure are not
available through this boundary.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, ClassVar, TextIO, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dithyramba._version import __version__
from dithyramba.access import QueryExclusions
from dithyramba.evidence import EvidenceGateSpec
from dithyramba.interactive import AgentAnswerPreparation, AgentResearchFacade
from dithyramba.persistence import LibraryRepository
from dithyramba.recall import RetrievalBudget
from dithyramba.sessions import ResearchSessionBrief

MCP_PROTOCOL_VERSION = "2025-11-25"
_SUPPORTED_PROTOCOL_VERSIONS = frozenset({MCP_PROTOCOL_VERSION, "2025-06-18", "2025-03-26"})


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _OpenSessionInput(_ToolInput):
    question: str = Field(min_length=1, max_length=4_000)
    intended_use: str = Field(min_length=1, max_length=1_000)
    success_criteria: list[str] = Field(min_length=1, max_length=16)
    boundaries: list[str] = Field(default_factory=list, max_length=16)
    corpus_snapshot_id: str
    access_policy_id: str
    purpose: str
    collection_ids: list[str] = Field(min_length=1, max_length=16)
    excluded_source_ids: list[str] = Field(default_factory=list)
    excluded_source_family_ids: list[str] = Field(default_factory=list)
    excluded_source_fragment_ids: list[str] = Field(default_factory=list)


class _RecallInput(_ToolInput):
    session_id: str
    question: str = Field(min_length=1, max_length=2_000)
    command_id: str
    max_candidates: int = Field(default=100, ge=1, le=500)
    max_source_fragments: int = Field(default=30, ge=1, le=100)


class _SessionInput(_ToolInput):
    session_id: str


class _JournalInput(_SessionInput):
    text: str = Field(min_length=1, max_length=8_000)


class _PrepareAnswerInput(_SessionInput):
    evidence_event_id: str
    gate_spec: dict[str, object]


class _RecordDraftInput(_JournalInput):
    preparation: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ToolSpec:
    name: str
    title: str
    description: str
    input_model: type[_ToolInput]

    def payload(self) -> dict[str, object]:
        schema = self.input_model.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": schema,
            "annotations": {
                "readOnlyHint": self.name in {"prepare_answer", "session_context"},
                "destructiveHint": False,
                "idempotentHint": self.name in {"prepare_answer", "recall", "session_context"},
                "openWorldHint": False,
            },
        }


_TOOLS = (
    _ToolSpec(
        "open_session",
        "Open research session",
        "Bind one research brief to an exact local snapshot and access scope.",
        _OpenSessionInput,
    ),
    _ToolSpec(
        "recall",
        "Recall exact evidence",
        "Retrieve exact source fragments and return a compact evidence turn.",
        _RecallInput,
    ),
    _ToolSpec(
        "session_context",
        "Read compact session context",
        "Read the bounded questions, evidence references, drafts and named gaps.",
        _SessionInput,
    ),
    _ToolSpec(
        "prepare_answer",
        "Prepare evidence-backed answer",
        "Filter candidates, drill down inside found works, and run EvidenceCoverageGate.",
        _PrepareAnswerInput,
    ),
    _ToolSpec(
        "record_draft",
        "Record answer draft",
        "Append a model-authored draft only with an exactly replayed ready preparation.",
        _RecordDraftInput,
    ),
    _ToolSpec(
        "record_gap",
        "Record evidence gap",
        "Append a named unresolved gap to guide the next retrieval turn.",
        _JournalInput,
    ),
    _ToolSpec(
        "reject_path",
        "Reject research path",
        "Append a path that should not be repeated in this session.",
        _JournalInput,
    ),
)
_TOOL_BY_NAME = {tool.name: tool for tool in _TOOLS}


class McpStdioServer:
    """Stateful MCP connection over caller-owned text streams."""

    SERVER_NAME: ClassVar[str] = "dithyramba"

    def __init__(self, facade: AgentResearchFacade) -> None:
        if not isinstance(facade, AgentResearchFacade):
            raise TypeError("McpStdioServer requires an AgentResearchFacade")
        self._facade = facade
        self._initialized = False
        self._ready = False

    def serve(self, input_stream: TextIO, output_stream: TextIO) -> None:
        """Process newline-delimited MCP messages until stdin reaches EOF."""

        for raw_line in input_stream:
            response = self.process_line(raw_line)
            if response is not None:
                output_stream.write(
                    json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                output_stream.flush()

    def process_line(self, raw_line: str) -> dict[str, object] | None:
        request_id: object = None
        try:
            message = json.loads(raw_line)
        except (json.JSONDecodeError, TypeError):
            return _rpc_error(None, -32700, "Parse error")
        if type(message) is not dict:
            return _rpc_error(None, -32600, "Invalid Request")
        request_id = message.get("id")
        if message.get("jsonrpc") != "2.0" or type(message.get("method")) is not str:
            return _rpc_error(request_id, -32600, "Invalid Request")
        method = str(message["method"])
        is_notification = "id" not in message
        params = message.get("params", {})
        if type(params) is not dict:
            return None if is_notification else _rpc_error(request_id, -32602, "Invalid params")
        try:
            result = self._dispatch(method, params, is_notification=is_notification)
        except _MethodNotFound as error:
            if is_notification:
                return None
            return _rpc_error(request_id, -32601, str(error))
        except ValidationError as error:
            if is_notification:
                return None
            return _rpc_error(request_id, -32602, "Invalid params", data=error.errors())
        except (TypeError, ValueError) as error:
            if is_notification:
                return None
            return _rpc_error(request_id, -32602, str(error))
        except Exception as error:  # fail closed at the protocol boundary
            if is_notification:
                return None
            return _rpc_error(
                request_id,
                -32603,
                "Internal error",
                data={"type": type(error).__name__, "message": str(error)},
            )
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _dispatch(
        self,
        method: str,
        params: dict[str, Any],
        *,
        is_notification: bool,
    ) -> dict[str, object]:
        if method == "initialize":
            if is_notification or self._initialized:
                raise ValueError("initialize must be the first request")
            requested = params.get("protocolVersion")
            if type(requested) is not str:
                raise ValueError("initialize requires protocolVersion")
            selected = (
                requested if requested in _SUPPORTED_PROTOCOL_VERSIONS else MCP_PROTOCOL_VERSION
            )
            self._initialized = True
            return {
                "protocolVersion": selected,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.SERVER_NAME, "version": __version__},
                "instructions": (
                    "Dithyramba returns source-grounded candidate evidence. "
                    "Retrieved fragments are not automatically accepted claims."
                ),
            }
        if method == "ping":
            return {}
        if method == "notifications/initialized":
            if not self._initialized:
                raise ValueError("initialize must precede notifications/initialized")
            self._ready = True
            return {}
        if not self._ready:
            raise ValueError("MCP connection is not initialized")
        if method == "tools/list":
            return {"tools": [tool.payload() for tool in _TOOLS]}
        if method == "tools/call":
            return self._call_tool(params)
        raise _MethodNotFound(method)

    def _call_tool(self, params: dict[str, Any]) -> dict[str, object]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if type(name) is not str or type(arguments) is not dict:
            raise ValueError("tools/call requires a tool name and object arguments")
        spec = _TOOL_BY_NAME.get(name)
        if spec is None:
            return _tool_error(f"Unknown tool: {name}")
        try:
            parsed = spec.input_model.model_validate(arguments)
            output = self._invoke(name, parsed)
        except (ValidationError, TypeError, ValueError) as error:
            return _tool_error(str(error))
        except Exception as error:
            return _tool_error(f"{type(error).__name__}: {error}")
        payload = output.model_dump(mode="json")
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                }
            ],
            "structuredContent": payload,
            "isError": False,
        }

    def _invoke(self, name: str, parsed: _ToolInput) -> BaseModel:
        if name == "open_session":
            open_input = _require_model(parsed, _OpenSessionInput)
            return self._facade.open(
                brief=ResearchSessionBrief.create(
                    question=open_input.question,
                    intended_use=open_input.intended_use,
                    success_criteria=tuple(open_input.success_criteria),
                    boundaries=tuple(open_input.boundaries),
                ),
                corpus_snapshot_id=open_input.corpus_snapshot_id,
                access_policy_id=open_input.access_policy_id,
                purpose=open_input.purpose,
                collection_ids=tuple(open_input.collection_ids),
                exclusions=QueryExclusions(
                    source_ids=tuple(open_input.excluded_source_ids),
                    source_family_ids=tuple(open_input.excluded_source_family_ids),
                    source_fragment_ids=tuple(open_input.excluded_source_fragment_ids),
                ),
            )
        if name == "recall":
            recall_input = _require_model(parsed, _RecallInput)
            return self._facade.recall(
                recall_input.session_id,
                recall_input.question,
                command_id=recall_input.command_id,
                retrieval=RetrievalBudget(
                    max_candidates=recall_input.max_candidates,
                    max_source_fragments=recall_input.max_source_fragments,
                ),
            )
        if name == "session_context":
            session_input = _require_model(parsed, _SessionInput)
            return self._facade.context(session_input.session_id)
        if name == "prepare_answer":
            prepare_input = _require_model(parsed, _PrepareAnswerInput)
            gate_spec = EvidenceGateSpec.model_validate_json(
                json.dumps(prepare_input.gate_spec, ensure_ascii=False, separators=(",", ":"))
            )
            return self._facade.prepare_answer(
                prepare_input.session_id,
                evidence_event_id=prepare_input.evidence_event_id,
                gate_spec=gate_spec,
            )
        if name == "record_draft":
            draft_input = _require_model(parsed, _RecordDraftInput)
            preparation = AgentAnswerPreparation.model_validate_json(
                json.dumps(draft_input.preparation, ensure_ascii=False, separators=(",", ":"))
            )
            return self._facade.record_draft(
                draft_input.session_id,
                draft_input.text,
                preparation=preparation,
            )
        journal_input = _require_model(parsed, _JournalInput)
        if name == "record_gap":
            return self._facade.record_gap(journal_input.session_id, journal_input.text)
        if name == "reject_path":
            return self._facade.reject_path(journal_input.session_id, journal_input.text)
        raise ValueError(f"unsupported tool: {name}")


class _MethodNotFound(ValueError):
    pass


_ToolInputT = TypeVar("_ToolInputT", bound=_ToolInput)


def _require_model(value: _ToolInput, expected: type[_ToolInputT]) -> _ToolInputT:
    if not isinstance(value, expected):
        raise TypeError(f"tool input must be {expected.__name__}")
    return value


def _rpc_error(
    request_id: object,
    code: int,
    message: str,
    *,
    data: object | None = None,
) -> dict[str, object]:
    error: dict[str, object] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _tool_error(message: str) -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def run_stdio_mcp(
    repository: LibraryRepository,
    *,
    agent_id: str,
    profile_version: str,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Serve one local Library until the client closes stdin."""

    with AgentResearchFacade(
        repository,
        agent_id=agent_id,
        profile_version=profile_version,
    ) as facade:
        McpStdioServer(facade).serve(
            input_stream or sys.stdin,
            output_stream or sys.stdout,
        )
