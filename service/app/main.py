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
    RAGIndexRequest,
    RAGIndexResponse,
    RAGSearchRequest,
    RAGSearchResponse,
)


app = FastAPI(title="Falco Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="falco-api")


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    orchestrator = runtime.get_orchestrator()
    answer = orchestrator.invoke(user_input=payload.message, thread_id=payload.thread_id)
    return ChatResponse(thread_id=payload.thread_id, answer=answer)


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_answer(thread_id: str, message: str) -> AsyncIterator[str]:
    orchestrator = runtime.get_orchestrator()
    answer = orchestrator.invoke(user_input=message, thread_id=thread_id)
    yield _sse_event("start", {"thread_id": thread_id})

    for token in answer.split():
        yield _sse_event("delta", {"content": f"{token} "})
        await asyncio.sleep(0.01)

    if not answer.strip():
        yield _sse_event("delta", {"content": ""})
    yield _sse_event("done", {"answer": answer})


@app.post("/api/v1/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_answer(thread_id=payload.thread_id, message=payload.message),
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
    target = (orchestrator.settings.workspace_root / payload.path).resolve()
    message = rag.index_paths([target], drop_old=payload.drop_old)
    return RAGIndexResponse(message=message)
