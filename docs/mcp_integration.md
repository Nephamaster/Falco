# MCP Integration

Falco loads MCP servers as external tool providers. The lead agent and subagent receive those MCP tools together with built-in tools, skills, memory, HITL, and delegation.

## Enable

Set in `config.yaml`:

```yaml
mcp:
  enabled: true
  config_path: ./.falco/mcp.json
  tool_prefix: true
```

The dependency is declared in `requirements.txt`:

```text
langchain-mcp-adapters
```

## Config Shape

Falco accepts either a top-level server map or a `servers` object:

```json
{
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "remote_docs": {
      "transport": "sse",
      "url": "https://example.com/mcp/sse"
    }
  }
}
```

Set `"enabled": false` on a server to skip it.

## Tool Naming

By default, MCP tool names are prefixed:

```text
mcp_<server>_<tool>
```

This prevents collisions with built-in tools and with other MCP servers. Disable it in `config.yaml` with `mcp.tool_prefix: false`.

## Agent Flow

1. `FalcoOrchestrator` creates `MCPToolRegistry`.
2. `MCPToolRegistry` reads `.falco/mcp.json`.
3. Each configured server is loaded through `langchain-mcp-adapters`.
4. Loaded MCP tools are appended to the LangChain tool list.
5. The built-in `mcp_catalog` tool lets the agent inspect MCP server status before choosing a tool.

If MCP is disabled, the config file is missing, or an adapter dependency is absent, startup continues and `mcp_catalog` reports the issue.

## Service Check

After deployment, inspect MCP status with:

```text
GET /api/v1/mcp/catalog
```
