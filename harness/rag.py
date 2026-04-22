from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from harness.config.config import FalcoSettings
from harness.prompts.templates import RAG_QUERY_OPTIMIZATION_PROMPT_TEMPLATE
from harness.tokenization import truncate_tokens


ALLOWED_KNOWLEDGE_EXTENSIONS = {
    ".txt",
    ".md",
    ".mdx",
    ".rst",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".log",
}

SKIPPED_KNOWLEDGE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".falco",
    ".next",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}

SKIPPED_KNOWLEDGE_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
}

SKIPPED_KNOWLEDGE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}

MAX_KNOWLEDGE_FILE_BYTES = 2_000_000


class QueryPlan(BaseModel):
    rewritten_query: str = Field(description="The optimized main retrieval query.")
    sub_queries: list[str] = Field(default_factory=list, description="Up to 3 sub-queries.")
    keywords: list[str] = Field(default_factory=list, description="Important retrieval keywords.")


@dataclass
class RAGSearchResult:
    query_plan: QueryPlan
    docs: list[Document]

    def render(self, max_tokens_per_doc: int = 700) -> str:
        if not self.docs:
            return "No knowledge found in local vector database."
        lines = [
            f"Optimized query: {self.query_plan.rewritten_query}",
            f"Query variants: {', '.join(self.query_plan.sub_queries) if self.query_plan.sub_queries else '(none)'}",
            "",
            "Retrieved context:",
        ]
        for idx, doc in enumerate(self.docs, start=1):
            source = doc.metadata.get("source", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "n/a")
            text = doc.page_content.strip().replace("\n", " ")
            lines.append(f"[{idx}] source={source} chunk={chunk_id}")
            lines.append(truncate_tokens(text, max_tokens_per_doc))
            lines.append("")
        return "\n".join(lines).strip()


class MilvusRAG:
    def __init__(self, settings: FalcoSettings, llm=None) -> None:
        self.settings = settings
        self.llm = llm
        self._vectorstore = None
        self._reranker = None
        self._embeddings = OpenAIEmbeddings(
            model=self.settings.rag_embedding_model,
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
        )
        self._hybrid_enabled = not str(self.settings.rag_milvus_uri).endswith(".db")
        if self._hybrid_enabled:
            try:
                from langchain_milvus import BM25BuiltInFunction

                self._bm25_builtin = BM25BuiltInFunction(output_field_names="sparse")
            except Exception:
                self._hybrid_enabled = False
                self._bm25_builtin = None
        else:
            self._bm25_builtin = None

    @property
    def connection_args(self) -> dict:
        args: dict = {"uri": self.settings.rag_milvus_uri}
        if self.settings.rag_milvus_token:
            args["token"] = self.settings.rag_milvus_token
        return args

    def _get_vectorstore(self):
        if self._vectorstore is not None:
            return self._vectorstore

        from langchain_milvus import Milvus

        kwargs = {
            "embedding_function": self._embeddings,
            "connection_args": self.connection_args,
            "collection_name": self.settings.rag_collection,
            "drop_old": False,
        }
        if self._hybrid_enabled and self._bm25_builtin is not None:
            kwargs["builtin_function"] = self._bm25_builtin
            kwargs["vector_field"] = ["dense", "sparse"]

        self._vectorstore = Milvus(**kwargs)
        return self._vectorstore

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        from sentence_transformers import CrossEncoder

        self._reranker = CrossEncoder(self.settings.rag_reranker_model)
        return self._reranker

    def _optimize_query(self, query: str) -> QueryPlan:
        cleaned = " ".join(query.split())
        fallback = QueryPlan(rewritten_query=cleaned, sub_queries=[], keywords=[])
        if self.llm is None:
            return fallback

        try:
            planner = self.llm.with_structured_output(QueryPlan)
            plan = planner.invoke(
                [
                    {"role": "system", "content": RAG_QUERY_OPTIMIZATION_PROMPT_TEMPLATE},
                    {"role": "user", "content": query},
                ]
            )
            if not plan.rewritten_query.strip():
                return fallback
            plan.sub_queries = [item.strip() for item in plan.sub_queries if item.strip()][:3]
            plan.keywords = [item.strip() for item in plan.keywords if item.strip()][:8]
            return plan
        except Exception:
            return fallback

    def _retrieve_candidates(self, query_plan: QueryPlan) -> list[Document]:
        queries: list[str] = [query_plan.rewritten_query]
        queries.extend(query_plan.sub_queries)
        if query_plan.keywords:
            queries.append(" ".join(query_plan.keywords))

        vectorstore = self._get_vectorstore()
        unique: dict[tuple[str, str], Document] = {}

        for q in queries:
            q = q.strip()
            if not q:
                continue
            try:
                if self._hybrid_enabled:
                    docs = vectorstore.similarity_search(
                        q,
                        k=self.settings.rag_fetch_k,
                        ranker_type="weighted",
                        ranker_params={"weights": [0.7, 0.3]},
                    )
                else:
                    docs = vectorstore.similarity_search(q, k=self.settings.rag_fetch_k)
            except Exception:
                docs = []

            for doc in docs:
                key = (
                    str(doc.metadata.get("source", "")),
                    truncate_tokens(doc.page_content, 120),
                )
                unique[key] = doc

        return list(unique.values())

    def _rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs:
            return []
        try:
            reranker = self._get_reranker()
            pairs = [(query, truncate_tokens(doc.page_content, 2200)) for doc in docs]
            scores = reranker.predict(pairs)
            ranked = sorted(
                zip(docs, scores, strict=False),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            return [doc for doc, _ in ranked[: top_k]]
        except Exception:
            return docs[: top_k]

    def search(self, query: str, top_k: int | None = None) -> RAGSearchResult:
        final_k = top_k if isinstance(top_k, int) and top_k > 0 else self.settings.rag_top_k
        plan = self._optimize_query(query)
        candidates = self._retrieve_candidates(plan)
        reranked = self._rerank(plan.rewritten_query, candidates, top_k=min(final_k, 12))
        return RAGSearchResult(query_plan=plan, docs=reranked)

    def _collect_documents(self, roots: Iterable[Path]) -> list[Document]:
        docs: list[Document] = []
        for root in roots:
            if root.is_file():
                files = [root]
            else:
                files = []
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [name for name in dirnames if name not in SKIPPED_KNOWLEDGE_DIRS]
                    current = Path(dirpath)
                    for name in filenames:
                        files.append(current / name)
            for path in files:
                if any(part in SKIPPED_KNOWLEDGE_DIRS for part in path.parts):
                    continue
                if path.name in SKIPPED_KNOWLEDGE_FILES or path.suffix.lower() in SKIPPED_KNOWLEDGE_SUFFIXES:
                    continue
                if path.suffix.lower() not in ALLOWED_KNOWLEDGE_EXTENSIONS:
                    continue
                try:
                    if path.stat().st_size > MAX_KNOWLEDGE_FILE_BYTES:
                        continue
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                if not text.strip():
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "file_name": path.name},
                    )
                )
        return docs

    def index_paths(
        self,
        paths: list[Path],
        *,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        drop_old: bool = False,
    ) -> str:
        source_docs = self._collect_documents(paths)
        if not source_docs:
            return "No eligible local documents found for indexing."

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(source_docs)
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = index

        from langchain_milvus import Milvus

        kwargs = {
            "documents": chunks,
            "embedding": self._embeddings,
            "connection_args": self.connection_args,
            "collection_name": self.settings.rag_collection,
            "drop_old": drop_old,
        }
        if self._hybrid_enabled and self._bm25_builtin is not None:
            kwargs["builtin_function"] = self._bm25_builtin
            kwargs["vector_field"] = ["dense", "sparse"]
        Milvus.from_documents(**kwargs)
        self._vectorstore = None
        return (
            f"Indexed {len(source_docs)} source files into {len(chunks)} chunks. "
            f"collection={self.settings.rag_collection} hybrid={'on' if self._hybrid_enabled else 'off'}"
        )
