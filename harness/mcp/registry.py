from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

SUPPORTED_TRANSPORTS = {"stdio", "sse", "streamable_http"}
TRANSPORT_ALIASES = {
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
    "stdio": "stdio",
    "sse": "sse",
}


@dataclass(frozen=True)
class MCPServerStatus:
    name: str
    enabled: bool
    transport: str
    tool_count: int = 0
    error: str = ""
    label: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    risk_level: str = "medium"


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str
    client_config: dict[str, Any]
    label: str
    description: str
    tags: tuple[str, ...]
    risk_level: str
    expose_to_agent: bool = True


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
        self._tool_names_in_use: set[str] = set()
        self._loaded_tools: list[BaseTool] = []
        self._config_fingerprint: str = ""

    def load_tools(self) -> list[BaseTool]:
        self._statuses = []
        self._clients = []
        self._tool_names_in_use = set()
        self._loaded_tools = []
        self._config_fingerprint = self._compute_config_fingerprint()
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

        server_map = {server.name: server for server in servers}
        for server in servers:
            self._statuses.append(
                MCPServerStatus(
                    name=server.name,
                    enabled=server.expose_to_agent,
                    transport=server.transport,
                    tool_count=0,
                    label=server.label,
                    description=server.description,
                    tags=server.tags,
                    risk_level=server.risk_level,
                    error="" if server.expose_to_agent else "Server is hidden from the agent by configuration.",
                )
            )

        visible_servers = {server.name: server.client_config for server in servers if server.expose_to_agent}
        if not visible_servers:
            self._loaded_tools = []
            return []

        try:
            client = MultiServerMCPClient(visible_servers)
            self._clients.append(client)
            tools = self._await_if_needed(client.get_tools())
        except Exception as exc:  # noqa: BLE001
            self._statuses = [
                MCPServerStatus(
                    name=server.name,
                    enabled=False,
                    transport=server.transport,
                    error=str(exc),
                    label=server.label,
                    description=server.description,
                    tags=server.tags,
                    risk_level=server.risk_level,
                )
                for server in servers
            ]
            return []

        counts: dict[str, int] = {server.name: 0 for server in servers}
        all_tools: list[BaseTool] = []
        for tool in tools:
            if not isinstance(tool, BaseTool):
                continue
            server_name = self._infer_server_name(tool, server_map) or "mcp"
            normalized = self._normalize_tool(tool, server_name=server_name, server_map=server_map)
            all_tools.append(normalized)
            if server_name in counts:
                counts[server_name] += 1

        self._statuses = [
            MCPServerStatus(
                name=server.name,
                enabled=server.expose_to_agent,
                transport=server.transport,
                tool_count=counts.get(server.name, 0),
                label=server.label,
                description=server.description,
                tags=server.tags,
                risk_level=server.risk_level,
                error="" if server.expose_to_agent else "Server is hidden from the agent by configuration.",
            )
            for server in servers
        ]
        self._loaded_tools = all_tools
        return list(all_tools)

    def reload_if_needed(self, *, force: bool = False) -> bool:
        fingerprint = self._compute_config_fingerprint()
        if force or fingerprint != self._config_fingerprint:
            self.load_tools()
            return True
        if not self._loaded_tools and self.enabled:
            self.load_tools()
            return True
        return False

    def tools(self) -> list[BaseTool]:
        if not self._loaded_tools:
            self.load_tools()
        return list(self._loaded_tools)

    def catalog(self) -> str:
        if not self._statuses:
            self.load_tools()
        rows = []
        for status in self._statuses:
            label = f" ({status.label})" if status.label and status.label != status.name else ""
            tags = f" | tags={', '.join(status.tags)}" if status.tags else ""
            risk = f" | risk={status.risk_level}"
            if status.enabled:
                rows.append(
                    f"- {status.name}{label} | transport={status.transport} | tools={status.tool_count}{risk}{tags}"
                )
            else:
                detail = f" | error={status.error}" if status.error else ""
                rows.append(f"- {status.name}{label} | disabled{risk}{tags}{detail}")
        return "\n".join(rows) if rows else "No MCP servers configured."

    def prompt_block(self) -> str:
        if not self._statuses:
            self.load_tools()
        lines = ["<mcp>"]
        enabled = [status for status in self._statuses if status.enabled and status.tool_count > 0]
        if not enabled:
            lines.append("- No MCP servers are currently available to the agent.")
        else:
            lines.append("- MCP tools are external capabilities. Prefer built-in tools when they already fit the task.")
            lines.append("- Use mcp_catalog if you need to inspect available external servers before choosing a tool.")
            lines.append("- Select an MCP tool only when its server purpose and tool description clearly match the user's request.")
            for status in enabled:
                label = status.label or status.name
                tags = f" tags={', '.join(status.tags)}" if status.tags else ""
                description = f" {status.description}" if status.description else ""
                lines.append(
                    f"- server={status.name} label={label} transport={status.transport} tools={status.tool_count} risk={status.risk_level}{tags}.{description}".strip()
                )
        lines.append("</mcp>")
        return "\n".join(lines)

    @staticmethod
    def _load_server_config(path: Path) -> list[MCPServerConfig]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        servers = raw.get("servers", raw)
        if not isinstance(servers, dict):
            raise ValueError("expected a JSON object or an object with a 'servers' field")

        normalized: list[MCPServerConfig] = []
        for name, config in servers.items():
            if not isinstance(config, dict):
                raise ValueError(f"server {name!r} must be an object")
            if config.get("enabled", True) is False:
                continue
            copied = dict(config)
            copied.pop("enabled", None)
            expose_to_agent = bool(copied.pop("expose_to_agent", True))
            label = str(copied.pop("label", str(name))).strip() or str(name)
            description = str(copied.pop("description", "")).strip()
            tags_raw = copied.pop("tags", [])
            tags = tuple(str(item).strip() for item in tags_raw if str(item).strip()) if isinstance(tags_raw, list) else ()
            risk_level = str(copied.pop("risk_level", "medium")).strip().lower() or "medium"
            if risk_level not in {"low", "medium", "high"}:
                raise ValueError(f"server {name!r} risk_level must be one of: low, medium, high")
            transport = MCPToolRegistry._normalize_transport(copied.get("transport", "stdio"))
            copied["transport"] = transport
            if transport == "stdio" and not copied.get("command"):
                raise ValueError(f"stdio server {name!r} requires command")
            if transport in {"sse", "streamable_http"} and not copied.get("url"):
                raise ValueError(f"{transport} server {name!r} requires url")
            MCPToolRegistry._validate_server_fields(str(name), copied)
            normalized.append(
                MCPServerConfig(
                    name=str(name),
                    transport=transport,
                    client_config=copied,
                    label=label,
                    description=description,
                    tags=tags,
                    risk_level=risk_level,
                    expose_to_agent=expose_to_agent,
                )
            )
        return normalized

    def _normalize_tool(self, tool: BaseTool, *, server_name: str, server_map: dict[str, MCPServerConfig]) -> BaseTool:
        tool = self._copy_tool(tool)
        original_name = tool.name
        if self.prefix_tools:
            safe_server = self._safe_name(server_name)
            safe_tool = self._safe_name(original_name)
            base_name = f"mcp_{safe_server}_{safe_tool}"
        else:
            base_name = self._safe_name(tool.name)
        tool.name = self._ensure_unique_tool_name(base_name)

        server = server_map.get(server_name)
        description = tool.description or ""
        server_label = server.label if server is not None else server_name
        server_tags = f" tags={', '.join(server.tags)}." if server and server.tags else ""
        risk = server.risk_level if server is not None else "medium"
        tool.description = (
            f"MCP server '{server_label}' tool '{original_name}'. risk={risk}.{server_tags} {description}".strip()
        )
        metadata = dict(getattr(tool, "metadata", {}) or {})
        metadata.update(
            {
                "mcp_server_name": server_name,
                "mcp_server_label": server_label,
                "mcp_risk_level": risk,
            }
        )
        if server and server.tags:
            metadata["mcp_tags"] = list(server.tags)
        tool.metadata = metadata
        return tool

    def _ensure_unique_tool_name(self, base_name: str) -> str:
        candidate = base_name
        suffix = 2
        while candidate in self._tool_names_in_use:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        self._tool_names_in_use.add(candidate)
        return candidate

    @staticmethod
    def _copy_tool(tool: BaseTool) -> BaseTool:
        if hasattr(tool, "model_copy"):
            return tool.model_copy(deep=False)
        if hasattr(tool, "copy"):
            return tool.copy(deep=False)
        return tool

    @staticmethod
    def _normalize_transport(raw: Any) -> str:
        value = str(raw or "stdio").strip().lower()
        value = TRANSPORT_ALIASES.get(value, value)
        if value not in SUPPORTED_TRANSPORTS:
            supported = ", ".join(sorted(SUPPORTED_TRANSPORTS))
            raise ValueError(f"unsupported MCP transport {raw!r}; expected one of: {supported}")
        return value

    @staticmethod
    def _validate_server_fields(name: str, config: dict[str, Any]) -> None:
        transport = str(config.get("transport", "stdio"))
        if transport == "stdio":
            command = config.get("command")
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"stdio server {name!r} requires a non-empty string command")
            args = config.get("args", [])
            if args is None:
                args = []
            if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
                raise ValueError(f"stdio server {name!r} args must be a list of strings")
            env = config.get("env", {})
            if env is None:
                env = {}
            if not isinstance(env, dict) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
            ):
                raise ValueError(f"stdio server {name!r} env must be an object of string pairs")
        else:
            url = config.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ValueError(f"{transport} server {name!r} requires a non-empty url")
            if not re.match(r"^https?://", url.strip(), flags=re.IGNORECASE):
                raise ValueError(f"{transport} server {name!r} url must start with http:// or https://")
            headers = config.get("headers", {})
            if headers is None:
                headers = {}
            if not isinstance(headers, dict) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
            ):
                raise ValueError(f"{transport} server {name!r} headers must be an object of string pairs")

    @staticmethod
    def _infer_server_name(tool: BaseTool, server_map: dict[str, MCPServerConfig]) -> str | None:
        metadata = getattr(tool, "metadata", {}) or {}
        for key in ("server_name", "mcp_server_name", "server", "source_server"):
            value = metadata.get(key)
            if isinstance(value, str) and value in server_map:
                return value

        tags = getattr(tool, "tags", None) or metadata.get("tags") or []
        for tag in tags:
            text = str(tag)
            prefix = "mcp:server:"
            if text.startswith(prefix):
                candidate = text[len(prefix) :].strip()
                if candidate in server_map:
                    return candidate

        raw_name = str(getattr(tool, "name", "") or "")
        lowered = raw_name.lower()
        for server_name in server_map:
            safe = MCPToolRegistry._safe_name(server_name).lower()
            if lowered.startswith(f"{safe}_") or lowered.startswith(f"{safe}__"):
                return server_name
        if len(server_map) == 1:
            return next(iter(server_map))
        return None

    def _compute_config_fingerprint(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.config_path is None:
            return "missing-path"
        if not self.config_path.exists():
            return f"missing:{self.config_path}"
        payload = self.config_path.read_bytes()
        return hashlib.sha1(payload).hexdigest()

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
        thread.join(timeout=30)
        if thread.is_alive():
            raise TimeoutError("Timed out while waiting for MCP client tools to load.")
        if error:
            raise error["error"]
        return result.get("value")
