"""Optional MCP adapter for explicitly configured local graphs.

The adapter owns transport and argument schemas only. Graph rules, access to
configured files, and revision checks belong to ``local_service``. Import this
module only when MCP is requested: the ordinary CLI has no MCP dependency.
"""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError

from .local_service import GRAPH_ID_PATTERN, LocalGraphService, LocalServiceError
from .storage import resolve_state_path


CONTRACT_VERSION = 1
Identifier = Annotated[StrictStr, Field(min_length=1)]
Revision = Annotated[StrictStr, Field(min_length=1)]
Index = Annotated[int, Field(strict=True, ge=0)]
PageSize = Annotated[int, Field(strict=True, ge=1, le=500)]


class ErrorDetails(BaseModel):
    code: StrictStr
    message: StrictStr
    cycle: list[StrictStr] | None = None


class ToolEnvelope(BaseModel):
    """Common versioned envelope; operation-specific fields remain intact."""

    model_config = ConfigDict(extra="allow")
    schema_version: Literal[1]
    error: ErrorDetails | None = None


ToolResult = Annotated[CallToolResult, ToolEnvelope]


class ConditionInput(BaseModel):
    """A new condition has a label and a real JSON boolean."""

    model_config = ConfigDict(extra="forbid")
    label: Identifier
    truth: StrictBool = False


def configured_graphs(state_path=None, *, graph_id=None, graph_specs=()):
    """Resolve trusted startup configuration, never a tool-supplied path.

    No graph is created as a side effect of starting the server. Named aliases
    are stable within a configured connection; the default single-file identity
    deliberately matches the existing browser's canonical-path hash.
    """

    if graph_specs and (state_path is not None or graph_id is not None):
        raise ValueError("--graph cannot be combined with --state or --graph-id")

    graphs = {}
    if graph_specs:
        for spec in graph_specs:
            name, separator, path = spec.partition("=")
            if not separator or not path.strip():
                raise ValueError("--graph must be ID=PATH")
            if not GRAPH_ID_PATTERN.fullmatch(name):
                raise ValueError("graph IDs must start with a letter or digit and use up to 128 letters, digits, _, . or -")
            if name in graphs:
                raise ValueError(f"duplicate graph ID: {name}")
            graphs[name] = Path(path).expanduser().resolve()
    else:
        path = resolve_state_path(state_path).resolve()
        name = graph_id if graph_id is not None else hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        if not GRAPH_ID_PATTERN.fullmatch(name):
            raise ValueError("graph IDs must start with a letter or digit and use up to 128 letters, digits, _, . or -")
        graphs[name] = path

    for path in graphs.values():
        if not path.is_file():
            raise ValueError(f"configured graph must be an existing file: {path}")
    return graphs


def _result(payload, *, is_error=False):
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=is_error,
    )


def _error(code, message, *, cycle=()):
    details = {"code": code, "message": message}
    if cycle:
        details["cycle"] = list(cycle)
    return _result({"schema_version": CONTRACT_VERSION, "error": details}, is_error=True)


class MattersMCP(FastMCP):
    """Keep SDK argument failures in the same structured error contract.

    Validate the raw arguments before FastMCP's convenience JSON-string parsing
    so a stringified array/object cannot bypass the declared JSON types. The
    SDK still validates and dispatches the actual call.
    """

    async def call_tool(self, name, arguments):
        tool = self._tool_manager.get_tool(name)
        if tool is None:
            return _error("not_found", "Unknown Matters tool.")
        if not isinstance(arguments, dict) or set(arguments) - set(tool.parameters.get("properties", {})):
            return _error("invalid", "Tool arguments do not match the declared schema.")
        try:
            tool.fn_metadata.arg_model.model_validate(arguments)
        except ValidationError:
            return _error("invalid", "Tool arguments do not match the declared schema.")
        try:
            return await super().call_tool(name, arguments)
        except ToolError:
            # Unexpected implementation failures must not expose file paths,
            # credentials, raw arguments or tracebacks to the calling client.
            return _error("internal", "The graph operation could not be completed.")


def create_mcp_server(graphs, *, read_only=False):
    """Build the stdio server without opening a listener or starting a shell."""

    service = LocalGraphService(graphs, read_only=read_only)
    server = MattersMCP(
        "Matters",
        instructions=(
            "Matters tool contract schema_version=1. First call list_graphs and "
            "select the intended graph. Every graph operation requires graph_id; "
            "a project reference selects a graph but never grants access. Do not "
            "assume ChatGPT sends its project identity. Reads return revision; "
            "every mutation requires expected_revision from the reviewed graph. "
            "On conflict, read and reassess before retrying. Condition indices "
            "are 0-based. Set explicit boolean truth only after verification. "
            "Deletion needs the user's authorization. This server grants access "
            "only to configured graphs and has no shell or filesystem tool."
        ),
        log_level="WARNING",
    )
    reads = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    writes = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
    deletions = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)

    def invoke(function, *args, **kwargs):
        try:
            return _result(function(*args, **kwargs))
        except LocalServiceError as error:
            return _error(error.code, error.message, cycle=error.cycle)

    @server.tool(annotations=reads)
    def list_graphs() -> ToolResult:
        """List only graphs configured for this local connection."""
        return invoke(service.catalog)

    @server.tool(annotations=reads)
    def list_matters(graph_id: Identifier, offset: Index = 0, limit: PageSize = 100) -> ToolResult:
        """List a page of matter IDs and the graph revision."""
        return invoke(service.read, graph_id, view="matters", offset=offset, limit=limit)

    @server.tool(annotations=reads)
    def get_matter(graph_id: Identifier, matter_id: Identifier, include_sources: StrictBool = False) -> ToolResult:
        """Inspect conditions, prerequisites and dependents; optionally citations."""
        return invoke(service.read, graph_id, view="matter", matter_id=matter_id, include_sources=include_sources)

    @server.tool(annotations=reads)
    def universe(graph_id: Identifier) -> ToolResult:
        """Return unresolved matters whose prerequisites are resolved."""
        return invoke(service.read, graph_id, view="universe")

    @server.tool(annotations=reads)
    def frontier(graph_id: Identifier, matter_id: Identifier) -> ToolResult:
        """Find actionable immediate dependents of a resolved matter."""
        return invoke(service.read, graph_id, view="frontier", matter_id=matter_id)

    @server.tool(annotations=reads)
    def horizon(graph_id: Identifier, matter_id: Identifier) -> ToolResult:
        """Find the farthest unresolved descendants of a matter."""
        return invoke(service.read, graph_id, view="horizon", matter_id=matter_id)

    @server.tool(annotations=reads)
    def unlock(graph_id: Identifier) -> ToolResult:
        """Report actionable matters, false conditions and candidate next actions."""
        return invoke(service.read, graph_id, view="unlock")

    @server.tool(annotations=writes)
    def create_matter(graph_id: Identifier, matter_id: Identifier, conditions: Annotated[list[ConditionInput], Field(min_length=1)], expected_revision: Revision) -> ToolResult:
        """Create one matter with explicitly supplied resolution conditions."""
        payload = {"id": matter_id, "conditions": [condition.model_dump() for condition in conditions]}
        return invoke(service.mutate, graph_id, "create_matter", payload, expected_revision=expected_revision)

    @server.tool(annotations=writes)
    def set_condition(graph_id: Identifier, matter_id: Identifier, index: Index, truth: StrictBool, expected_revision: Revision) -> ToolResult:
        """Set a reviewed condition to a JSON boolean using its 0-based index."""
        return invoke(service.mutate, graph_id, "set_condition", {"matter_id": matter_id, "index": index, "truth": truth}, expected_revision=expected_revision)

    @server.tool(annotations=writes)
    def add_condition(graph_id: Identifier, matter_id: Identifier, label: Identifier, expected_revision: Revision, truth: StrictBool = False) -> ToolResult:
        """Append a named condition, false by default."""
        return invoke(service.mutate, graph_id, "add_condition", {"matter_id": matter_id, "label": label, "truth": truth}, expected_revision=expected_revision)

    @server.tool(annotations=writes)
    def edit_condition(graph_id: Identifier, matter_id: Identifier, index: Index, label: Identifier, expected_revision: Revision) -> ToolResult:
        """Change a condition's label without changing its truth."""
        return invoke(service.mutate, graph_id, "edit_condition", {"matter_id": matter_id, "index": index, "label": label}, expected_revision=expected_revision)

    @server.tool(annotations=deletions)
    def delete_condition(graph_id: Identifier, matter_id: Identifier, index: Index, expected_revision: Revision, confirmed: StrictBool = False) -> ToolResult:
        """Delete a condition; removing the last one requires explicit confirmation."""
        return invoke(service.mutate, graph_id, "delete_condition", {"matter_id": matter_id, "index": index, "confirmed": confirmed}, expected_revision=expected_revision)

    @server.tool(annotations=deletions)
    def delete_matter(graph_id: Identifier, matter_id: Identifier, expected_revision: Revision, confirmed: StrictBool = False, cascade: StrictBool = False) -> ToolResult:
        """Delete a matter after confirmation; cascade removes its incident edges."""
        return invoke(service.mutate, graph_id, "delete_matter", {"matter_id": matter_id, "confirmed": confirmed, "cascade": cascade}, expected_revision=expected_revision)

    @server.tool(annotations=writes)
    def add_dependency(graph_id: Identifier, source: Identifier, target: Identifier, expected_revision: Revision) -> ToolResult:
        """Add prerequisite source -> dependent target; the engine rejects cycles."""
        return invoke(service.mutate, graph_id, "add_dependency", {"source": source, "target": target}, expected_revision=expected_revision)

    @server.tool(annotations=deletions)
    def remove_dependency(graph_id: Identifier, source: Identifier, target: Identifier, expected_revision: Revision) -> ToolResult:
        """Remove prerequisite source -> dependent target, including cycle repair."""
        return invoke(service.mutate, graph_id, "remove_dependency", {"source": source, "target": target}, expected_revision=expected_revision)

    @server.tool(annotations=reads)
    def export_graph(graph_id: Identifier) -> ToolResult:
        """Export a portable graph package with citations and no account permissions."""
        return invoke(service.read, graph_id, view="export")

    # FastMCP ignores unknown function kwargs by default. Advertise and enforce
    # a closed schema so accidental paths or unsupported options are refused.
    for tool in server._tool_manager.list_tools():
        tool.parameters["additionalProperties"] = False
    return server


def run_mcp(graphs, *, read_only=False):
    """Run MCP over stdio only; all diagnostics belong on stderr."""

    create_mcp_server(graphs, read_only=read_only).run(transport="stdio")
