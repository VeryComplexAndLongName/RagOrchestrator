from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class MCPToolProvider(Protocol):
    def list_tools(self) -> list[str]:
        ...

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class ToolContext:
    request_id: str


ToolCallable = Callable[[ToolContext, dict[str, Any]], dict[str, Any]]


class EnterpriseToolRegistry:
    def __init__(
        self,
        *,
        mcp_provider: MCPToolProvider | None = None,
        enable_mcp: bool = True,
        strict_failures: bool = False,
    ) -> None:
        self._tools: dict[str, ToolCallable] = {}
        self._mcp_provider = mcp_provider
        self._enable_mcp = enable_mcp
        self._strict_failures = strict_failures

    def register(self, name: str, tool: ToolCallable) -> None:
        self._tools[name] = tool

    def names(self) -> list[str]:
        names = list(self._tools)
        if self._mcp_provider is not None and self._enable_mcp:
            try:
                names.extend(self._mcp_provider.list_tools())
            except Exception:
                if self._strict_failures:
                    raise
        return sorted(set(names))

    def call(self, name: str, context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        local = self._tools.get(name)
        if local is not None:
            try:
                return local(context, args)
            except Exception as exc:
                if self._strict_failures:
                    raise
                return {"ok": False, "error": str(exc), "tool": name}

        if self._mcp_provider is not None and self._enable_mcp:
            try:
                return self._mcp_provider.call_tool(name, args)
            except Exception as exc:
                if self._strict_failures:
                    raise
                return {"ok": False, "error": str(exc), "tool": name}

        return {"ok": False, "error": f"Tool '{name}' is not registered", "tool": name}
