from __future__ import annotations

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    "Falco STDIO MCP",
    instructions=(
        "A minimal MCP server for validating stdio transport in Falco. "
        "It exposes a few simple tools for connectivity and integration tests."
    ),
    json_response=True,
)


@mcp.tool()
def health_check() -> dict[str, bool]:
    """Return a simple health payload to verify the stdio MCP server is alive."""
    return {"ok": True}


@mcp.tool()
def echo_text(text: str) -> dict[str, str]:
    """Echo a short string back to the caller."""
    return {"text": text}


@mcp.tool()
def summarize_patent_topic(title: str, context: str = "") -> dict[str, str]:
    """Return a placeholder patent summary for stdio MCP integration tests."""
    cleaned_title = str(title or "").strip()
    cleaned_context = str(context or "").strip()
    summary = (
        f"这是一个用于 stdio MCP 联调的示例摘要工具，主题是《{cleaned_title}》。"
        "当前返回的是占位内容，用于验证 Falco 能否通过 stdio 方式调用 MCP 工具。"
    )
    if cleaned_context:
        summary += f" 额外上下文：{cleaned_context}"
    return {
        "title": cleaned_title,
        "summary": summary,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
