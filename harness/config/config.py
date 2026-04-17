from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


load_dotenv()


def _resolve_rooted_path(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    return path


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
    memory_context_soft_limit_chars: int = 7000
    memory_context_max_chars: int = 9000
    memory_silent_turn_cooldown_rounds: int = 4
    memory_daily_half_life_days: int = 30
    memory_daily_lookback_days: int = 180
    memory_daily_retrieval_items: int = 8
    memory_evergreen_retrieval_items: int = 5
    rag_enabled: bool = True
    rag_milvus_uri: str = "./.falco/milvus/falco_rag.db"
    rag_milvus_token: str | None = None
    rag_collection: str = "falco_knowledge"
    rag_embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 5
    rag_fetch_k: int = 18
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    mcp_enabled: bool = False
    mcp_config_path: Path | None = None
    mcp_tool_prefix: bool = True

    @classmethod
    def from_env(cls, workspace_root: str | Path | None = None) -> "FalcoSettings":
        root = Path(workspace_root or ".").resolve()
        memory_root_raw = os.getenv("FALCO_MEMORY_ROOT", str(root / ".falco" / "memory"))
        skills_root_raw = os.getenv("FALCO_SKILLS_ROOT", str(root / ".falco" / "skills"))
        return cls(
            model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL"),
            workspace_root=root,
            memory_root=_resolve_rooted_path(root, memory_root_raw),
            skills_root=_resolve_rooted_path(root, skills_root_raw),
            max_context_messages=int(os.getenv("FALCO_MAX_CONTEXT_MESSAGES", "12")),
            max_tool_steps=int(os.getenv("FALCO_MAX_TOOL_STEPS", "6")),
            memory_recent_rounds=int(os.getenv("FALCO_MEMORY_RECENT_ROUNDS", "6")),
            memory_key_rounds=int(os.getenv("FALCO_MEMORY_KEY_ROUNDS", "4")),
            memory_importance_threshold=int(os.getenv("FALCO_MEMORY_IMPORTANCE_THRESHOLD", "7")),
            memory_max_rounds=int(os.getenv("FALCO_MEMORY_MAX_ROUNDS", "160")),
            memory_context_soft_limit_chars=int(os.getenv("FALCO_MEMORY_CONTEXT_SOFT_LIMIT_CHARS", "7000")),
            memory_context_max_chars=int(os.getenv("FALCO_MEMORY_CONTEXT_MAX_CHARS", "9000")),
            memory_silent_turn_cooldown_rounds=int(
                os.getenv("FALCO_MEMORY_SILENT_TURN_COOLDOWN_ROUNDS", "4")
            ),
            memory_daily_half_life_days=int(os.getenv("FALCO_MEMORY_DAILY_HALF_LIFE_DAYS", "30")),
            memory_daily_lookback_days=int(os.getenv("FALCO_MEMORY_DAILY_LOOKBACK_DAYS", "180")),
            memory_daily_retrieval_items=int(os.getenv("FALCO_MEMORY_DAILY_RETRIEVAL_ITEMS", "8")),
            memory_evergreen_retrieval_items=int(
                os.getenv("FALCO_MEMORY_EVERGREEN_RETRIEVAL_ITEMS", "5")
            ),
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
            mcp_enabled=os.getenv("FALCO_MCP_ENABLED", "false").lower() == "true",
            mcp_config_path=_resolve_rooted_path(
                root,
                os.getenv("FALCO_MCP_CONFIG", str(root / ".falco" / "mcp.json")),
            ),
            mcp_tool_prefix=os.getenv("FALCO_MCP_TOOL_PREFIX", "true").lower() == "true",
        )
