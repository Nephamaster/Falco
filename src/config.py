from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()


@dataclass(frozen=True)
class FalcoSettings:
    model: str
    api_key: str
    base_url: str | None
    workspace_root: Path
    memory_root: Path
    skills_root: Path
    max_context_messages: int = 12
    max_tool_steps: int = 6
    memory_recent_rounds: int = 6
    memory_key_rounds: int = 4
    memory_importance_threshold: int = 7
    memory_max_rounds: int = 160
    rag_enabled: bool = True
    rag_milvus_uri: str = "./.falco/milvus/falco_rag.db"
    rag_milvus_token: str | None = None
    rag_collection: str = "falco_knowledge"
    rag_embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 5
    rag_fetch_k: int = 18
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @classmethod
    def from_env(cls, workspace_root: str | Path | None = None) -> "FalcoSettings":
        root = Path(workspace_root or ".").resolve()
        return cls(
            model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL"),
            workspace_root=root,
            memory_root=root / ".falco" / "memory",
            skills_root=root / ".falco" / "skills",
            max_context_messages=int(os.getenv("FALCO_MAX_CONTEXT_MESSAGES", "12")),
            max_tool_steps=int(os.getenv("FALCO_MAX_TOOL_STEPS", "6")),
            memory_recent_rounds=int(os.getenv("FALCO_MEMORY_RECENT_ROUNDS", "6")),
            memory_key_rounds=int(os.getenv("FALCO_MEMORY_KEY_ROUNDS", "4")),
            memory_importance_threshold=int(os.getenv("FALCO_MEMORY_IMPORTANCE_THRESHOLD", "7")),
            memory_max_rounds=int(os.getenv("FALCO_MEMORY_MAX_ROUNDS", "160")),
            rag_enabled=os.getenv("FALCO_RAG_ENABLED", "true").lower() == "true",
            rag_milvus_uri=os.getenv("FALCO_RAG_MILVUS_URI", str(root / ".falco" / "milvus" / "falco_rag.db")),
            rag_milvus_token=os.getenv("FALCO_RAG_MILVUS_TOKEN"),
            rag_collection=os.getenv("FALCO_RAG_COLLECTION", "falco_knowledge"),
            rag_embedding_model=os.getenv("FALCO_RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            rag_top_k=int(os.getenv("FALCO_RAG_TOP_K", "5")),
            rag_fetch_k=int(os.getenv("FALCO_RAG_FETCH_K", "18")),
            rag_reranker_model=os.getenv(
                "FALCO_RAG_RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
        )
