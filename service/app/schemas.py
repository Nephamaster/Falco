from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str = Field(default="default", min_length=1, max_length=128)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    thread_id: str
    answer: str


class HealthResponse(BaseModel):
    status: str
    service: str


class RAGSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=12)


class RAGSearchResponse(BaseModel):
    result: str


class RAGIndexRequest(BaseModel):
    path: str = Field(default="knowledge")
    drop_old: bool = False


class RAGIndexResponse(BaseModel):
    message: str
