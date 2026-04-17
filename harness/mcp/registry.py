from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class MCPServerStatus:
    name: str
    enabled: bool
    transport: str
    tool_count: int = 0
    error: str = ""


class MCPToolRegistry:
    """Load MCP server tools and expose them as LangChain tools."""

    def __init__(
        self,
        *,
        config_path: Path | None,
        enabled: bool = False,
        prefix_tools: bool = True,
    ) -> None:
        self.config_path = config_path
        self.enabled = enabled
        self.prefix_tools = prefix_tools
        self._statuses: list[MCPServerStatus] = []
        self._clients: list[Any] = []

    def load_tools(self) -> list[BaseTool]:
        self._statuses = []
        self._clients = []
        if not self.enabled:
            self._statuses.append(
                MCPServerStatus(name="mcp", enabled=False, transport="", error="MCP is disabled.")
            )
            return []
        if self.config_path is None:
            self._statuses.append(
                MCPServerStatus(name="mcp", enabled=False, transport="", error="MCP config path is not set.")
            )
            return []
        if not self.config_path.exists():
            self._statuses.append(
                MCPServerStatus(
                    name="mcp",
                    enabled=False,
                    transport="",
                    error=f"MCP config file does not exist: {self.config_path}",
                )
            )
            return []

        try:
            servers = self._load_server_config(self.config_path)
        except Exception as exc:  # noqa: BLE001
            self._statuses.append(
                MCPServerStatus(name="mcp", enabled=False, transport="", error=f"Invalid MCP config: {exc}")
            )
            return []

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except Exception as exc:  # noqa: BLE001
            self._statuses.append(
                MCPServerStatus(
                    name="mcp",
                    enabled=False,
                    transport="",
                    error=f"Missing langchain-mcp-adapters dependency: {exc}",
                )
            )
            return []

        all_tools: list[BaseTool] = []
        for server_name, server_config in servers.items():
            transport = str(server_config.get("transport", "stdio"))
            try:
                client = MultiServerMCPClient({server_name: server_config})
                self._clients.append(client)
                tools = self._await_if_needed(client.get_tools())
                normalized = [
                    self._normalize_tool(tool, server_name=server_name)
                    for tool in tools
                    if isinstance(tool, BaseTool)
                ]
                all_tools.extend(normalized)
                self._statuses.append(
                    MCPServerStatus(
                        name=server_name,
                        enabled=True,
                        transport=transport,
                        tool_count=len(normalized),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._statuses.append(
                    MCPServerStatus(
                        name=server_name,
                        enabled=False,
                        transport=transport,
                        error=str(exc),
                    )
                )
        return all_tools

    def catalog(self) -> str:
        if not self._statuses:
            self.load_tools()
        rows = []
        for status in self._statuses:
            if status.enabled:
                rows.append(
                    f"- {status.name} | transport={status.transport} | tools={status.tool_count}"
                )
            else:
                detail = f" | error={status.error}" if status.error else ""
                rows.append(f"- {status.name} | disabled{detail}")
        return "\n".join(rows) if rows else "No MCP servers configured."

    @staticmethod
    def _load_server_config(path: Path) -> dict[str, dict[str, Any]]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        servers = raw.get("servers", raw)
        if not isinstance(servers, dict):
            raise ValueError("expected a JSON object or an object with a 'servers' field")

        normalized: dict[str, dict[str, Any]] = {}
        for name, config in servers.items():
            if not isinstance(config, dict):
                raise ValueError(f"server {name!r} must be an object")
            if config.get("enabled", True) is False:
                continue
            copied = dict(config)
            copied.pop("enabled", None)
            transport = copied.get("transport", "stdio")
            if transport == "stdio" and not copied.get("command"):
                raise ValueError(f"stdio server {name!r} requires command")
            if transport in {"sse", "streamable_http"} and not copied.get("url"):
                raise ValueError(f"{transport} server {name!r} requires url")
            normalized[str(name)] = copied
        return normalized

    def _normalize_tool(self, tool: BaseTool, *, server_name: str) -> BaseTool:
        original_name = tool.name
        if self.prefix_tools:
            safe_server = self._safe_name(server_name)
            safe_tool = self._safe_name(original_name)
            tool.name = f"mcp_{safe_server}_{safe_tool}"
        else:
            tool.name = self._safe_name(tool.name)

        description = tool.description or ""
        tool.description = (
            f"MCP server '{server_name}' tool '{original_name}'. {description}".strip()
        )
        return tool

    @staticmethod
    def _safe_name(raw: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip())
        safe = safe.strip("_")
        return safe or "tool"

    @staticmethod
    def _await_if_needed(value):
        if not inspect.isawaitable(value):
            return value
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(value)

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def run() -> None:
            try:
                result["value"] = asyncio.run(value)
            except BaseException as exc:  # noqa: BLE001
                error["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join()
        if error:
            raise error["error"]
        return result.get("value")
