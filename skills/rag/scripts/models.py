from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    rewritten_query: str = Field(description="The optimized main retrieval query.")
    sub_queries: list[str] = Field(default_factory=list, description="Up to N query variants.")
    keywords: list[str] = Field(default_factory=list, description="Important retrieval keywords.")


@dataclass(frozen=True)
class SearchArtifacts:
    query_plan: QueryPlan
    candidates: list[Document]
    reranked_docs: list[Document]
    retrieval_mode: str
    warnings: list[str] = field(default_factory=list)

