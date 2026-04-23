from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from service.app.runtime import runtime
from service.app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MCPCatalogResponse,
    RAGIndexRequest,
    RAGIndexResponse,
    RAGSearchRequest,
    RAGSearchResponse,
)


app = FastAPI(title="Falco Service", version="0.1.0")


def _cors_origins() -> list[str]:
    orchestrator = runtime.get_orchestrator()
    origins = [item.strip() for item in orchestrator.settings.cors_origins if item.strip()]
    return origins


def _cors_origin_regex() -> str | None:
    orchestrator = runtime.get_orchestrator()
    regexes = [item.strip() for item in orchestrator.settings.cors_origin_regexes if item.strip()]
    if not regexes:
        return None
    if len(regexes) == 1:
        return regexes[0]
    return "|".join(f"(?:{pattern})" for pattern in regexes)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=_cors_origin_regex(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_workspace_path(workspace_root, raw_path: str):
    target = (workspace_root / raw_path).resolve()
    try:
        rel = target.relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise ValueError("Path escapes workspace root.") from exc
    blocked_parts = {".git", ".falco", ".next", ".venv", "__pycache__", "node_modules", "venv"}
    if set(rel.parts) & blocked_parts or target.name.startswith(".env") or target.suffix.lower() in {".pem", ".key"}:
        raise ValueError("Refusing to index sensitive or generated path.")
    return target


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="falco-api")


@app.get("/api/v1/mcp/catalog", response_model=MCPCatalogResponse)
def mcp_catalog() -> MCPCatalogResponse:
    orchestrator = runtime.get_orchestrator()
    mcp = getattr(orchestrator, "mcp", None)
    if mcp is None:
        return MCPCatalogResponse(result="MCP registry is not configured.")
    return MCPCatalogResponse(result=mcp.catalog())


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    orchestrator = runtime.get_orchestrator()
    if payload.resume:
        answer = orchestrator.resume(
            user_input=payload.message,
            thread_id=payload.thread_id,
            user_response_preference=payload.user_response_preference,
        )
    else:
        answer = orchestrator.invoke(
            user_input=payload.message,
            thread_id=payload.thread_id,
            user_response_preference=payload.user_response_preference,
        )
    return ChatResponse(thread_id=payload.thread_id, answer=answer)


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 24) -> list[str]:
    if not text:
        return [""]
    return [text[index:index + size] for index in range(0, len(text), size)]


async def _stream_answer(thread_id: str, message: str, user_response_preference: str, resume: bool = False) -> AsyncIterator[str]:
    orchestrator = runtime.get_orchestrator()
    if resume:
        answer = orchestrator.resume(
            user_input=message,
            thread_id=thread_id,
            user_response_preference=user_response_preference,
        )
    else:
        answer = orchestrator.invoke(
            user_input=message,
            thread_id=thread_id,
            user_response_preference=user_response_preference,
        )
    yield _sse_event("start", {"thread_id": thread_id})

    for chunk in _chunk_text(answer):
        yield _sse_event("delta", {"content": chunk})
        await asyncio.sleep(0.01)

    if not answer.strip():
        yield _sse_event("delta", {"content": ""})
    yield _sse_event("done", {"answer": answer})


@app.post("/api/v1/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_answer(
            thread_id=payload.thread_id,
            message=payload.message,
            user_response_preference=payload.user_response_preference,
            resume=payload.resume,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/api/v1/rag/search", response_model=RAGSearchResponse)
def rag_search(payload: RAGSearchRequest) -> RAGSearchResponse:
    orchestrator = runtime.get_orchestrator()
    rag = getattr(orchestrator, "rag", None)
    if rag is None:
        return RAGSearchResponse(result="RAG is disabled.")
    result = rag.search(payload.query, top_k=payload.top_k)
    return RAGSearchResponse(result=result.render())


@app.post("/api/v1/rag/index", response_model=RAGIndexResponse)
def rag_index(payload: RAGIndexRequest) -> RAGIndexResponse:
    orchestrator = runtime.get_orchestrator()
    rag = getattr(orchestrator, "rag", None)
    if rag is None:
        return RAGIndexResponse(message="RAG is disabled.")
    try:
        target = _resolve_workspace_path(orchestrator.settings.workspace_root, payload.path)
    except ValueError as exc:
        return RAGIndexResponse(message=str(exc))
    message = rag.index_paths([target], drop_old=payload.drop_old)
    return RAGIndexResponse(message=message)
