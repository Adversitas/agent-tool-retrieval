"""MCP server exposing the tool registry to any MCP-capable agent (Claude, LangGraph, ...).

Instead of loading 1,000+ tool schemas into the model's context, the agent gets two
meta-tools: ``search_tools`` (semantic + lexical retrieval over the registry) and
``get_tool_schema`` (the full JSON schema of the tool it picked). Run with::

    toolret serve            # stdio transport, for desktop / IDE clients
"""

from __future__ import annotations

from dataclasses import asdict

from .data import Tool


class Registry:
    """Thin facade over a retriever, shared by the MCP server and the CLI."""

    def __init__(self, tools: list[Tool], retriever):
        self.tools = {t.name: t for t in tools}
        self.retriever = retriever

    def search(self, query: str, k: int = 5) -> list[dict]:
        k = max(1, min(int(k), 20))
        return [
            {"name": t.name, "description": t.description, "score": round(s, 4)}
            for t, s in self.retriever.search(query, k)
        ]

    def schema(self, name: str) -> dict:
        if name not in self.tools:
            raise KeyError(f"unknown tool {name!r}")
        return asdict(self.tools[name])


def build_server(registry: Registry):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        "tool-registry",
        instructions=(
            "Call search_tools with the user's request to find candidate tools, then "
            "get_tool_schema on the best one before calling it."
        ),
    )

    @server.tool()
    def search_tools(query: str, k: int = 5) -> list[dict]:
        """Return the k registered tools most relevant to a natural-language request."""
        return registry.search(query, k)

    @server.tool()
    def get_tool_schema(name: str) -> dict:
        """Return the full JSON schema (description and parameters) of a registered tool."""
        return registry.schema(name)

    return server
