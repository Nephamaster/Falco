from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from skills.rag.scripts.milvus_connection import build_connected_milvus_from_documents


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


class KnowledgeIndexer:
    def __init__(self, *, embeddings, connection_args: dict, collection_name: str, retrieval_mode: str) -> None:
        self._embeddings = embeddings
        self._connection_args = connection_args
        self._collection_name = collection_name
        self._retrieval_mode = retrieval_mode
        self._bm25_builtin = None
        self._hybrid_available = False
        if retrieval_mode == "hybrid":
            try:
                from langchain_milvus import BM25BuiltInFunction

                self._bm25_builtin = BM25BuiltInFunction(output_field_names="sparse")
                self._hybrid_available = True
            except Exception:
                self._bm25_builtin = None

    def collect_documents(self, roots: Iterable[Path]) -> list[Document]:
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
                docs.append(Document(page_content=text, metadata={"source": str(path), "file_name": path.name}))
        return docs

    def index_paths(self, paths: list[Path], *, chunk_size: int, chunk_overlap: int, drop_old: bool = False) -> str:
        source_docs = self.collect_documents(paths)
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

        kwargs = {
            "documents": chunks,
            "embedding": self._embeddings,
            "connection_args": self._connection_args,
            "collection_name": self._collection_name,
            "drop_old": drop_old,
        }
        if self._retrieval_mode == "hybrid" and self._hybrid_available and self._bm25_builtin is not None:
            kwargs["builtin_function"] = self._bm25_builtin
            kwargs["vector_field"] = ["dense", "sparse"]
        build_connected_milvus_from_documents(**kwargs)
        mode = "hybrid" if self._retrieval_mode == "hybrid" and self._hybrid_available else "dense"
        return f"Indexed {len(source_docs)} source files into {len(chunks)} chunks. collection={self._collection_name} mode={mode}"
