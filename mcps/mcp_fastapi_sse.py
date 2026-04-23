from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from fastapi_mcp import FastApiMCP


class PatentSummaryRequest(BaseModel):
    title: str = Field(..., description="Patent title or topic to summarize.")
    context: str = Field(
        default="",
        description="Optional extra context or notes that should be included in the summary prompt.",
    )


class PatentSummaryResponse(BaseModel):
    title: str
    summary: str


app = FastAPI(
    title="Falco MCP Demo",
    description="A minimal FastAPI application exposed as MCP tools for Falco integration tests.",
    version="1.0.0",
)


@app.get("/health", operation_id="health_check", tags=["system"])
async def health() -> dict[str, bool]:
    """Simple health endpoint for validating the service is up."""
    return {"ok": True}


@app.get("/echo", operation_id="echo_text", tags=["utility"])
async def echo_text(text: str) -> dict[str, str]:
    """Echo a short string back to the caller."""
    return {"text": text}


@app.post(
    "/patent/summary",
    operation_id="summarize_patent_topic",
    tags=["patent"],
    response_model=PatentSummaryResponse,
)
async def summarize_patent_topic(payload: PatentSummaryRequest) -> PatentSummaryResponse:
    """Return a placeholder patent summary for MCP connectivity tests."""
    title = payload.title.strip()
    context = payload.context.strip()
    summary = (
        f"这是一个用于 MCP 联调的示例摘要接口，主题是《{title}》。"
        "当前返回的是占位内容，用于验证 Falco 能否通过 MCP 调用 FastAPI 暴露出来的工具。"
    )
    if context:
        summary += f" 额外上下文：{context}"
    return PatentSummaryResponse(title=title, summary=summary)


mcp = FastApiMCP(
    app,
    name="Falco FastAPI MCP",
    description="Expose selected FastAPI endpoints as MCP tools for local integration testing."
)
mcp.mount_sse(mount_path='/mcp')
